use core::{
    borrow::Borrow,
    fmt::Debug,
    panic::{RefUnwindSafe, UnwindSafe},
};

use alloc::sync::Arc;

#[cfg(feature = "syntax")]
use regex_syntax::hir::{literal, Hir};

use crate::util::search::{MatchKind, Span};

pub(crate) use crate::util::prefilter::{
    aho_corasick::AhoCorasick,
    byteset::ByteSet,
    memchr::{Memchr, Memchr2, Memchr3},
    memmem::Memmem,
    teddy::Teddy,
};

// Creates a new prefilter as a trait object. This is wrapped in a macro
// because we want to create both a Arc<dyn PrefilterI> and also a Arc<dyn
// meta::Strategy>. I'm not sure there's a way to get code reuse without a
// macro for this particular case, but if so and it only uses safe code, I'd be
// happy for a patch.
macro_rules! new {
    ($kind:ident, $needles:ident) => {{
        let kind = $kind;
        let needles = $needles;
        // An empty set means the regex matches nothing, so no sense in
        // building a prefilter.
        if needles.len() == 0 {
            debug!("prefilter building failed: found empty set of literals");
            return None;
        }
        // If the regex can match the empty string, then the prefilter will
        // by definition match at every position. This is obviously completely
        // ineffective.
        if needles.iter().any(|n| n.as_ref().is_empty()) {
            debug!("prefilter building failed: literals match empty string");
            return None;
        }
        if let Some(pre) = Memchr::new(kind, needles) {
            debug!("prefilter built: memchr");
            return Some(Arc::new(pre));
        }
        if let Some(pre) = Memchr2::new(kind, needles) {
            debug!("prefilter built: memchr2");
            return Some(Arc::new(pre));
        }
        if let Some(pre) = Memchr3::new(kind, needles) {
            debug!("prefilter built: memchr3");
            return Some(Arc::new(pre));
        }
        if let Some(pre) = Memmem::new(kind, needles) {
            debug!("prefilter built: memmem");
            return Some(Arc::new(pre));
        }
        if let Some(pre) = Teddy::new(kind, needles) {
            debug!("prefilter built: teddy");
            return Some(Arc::new(pre));
        }
        if let Some(pre) = ByteSet::new(kind, needles) {
            debug!("prefilter built: byteset");
            return Some(Arc::new(pre));
        }
        if let Some(pre) = AhoCorasick::new(kind, needles) {
            debug!("prefilter built: aho-corasick");
            return Some(Arc::new(pre));
        }
        debug!("prefilter building failed: no strategy could be found");
        None
    }};
}

/// Creates a new prefilter, if possible, from the given set of needles and
/// returns it as a `PrefilterI` trait object.
fn new<B: AsRef<[u8]>>(
    kind: MatchKind,
    needles: &[B],
) -> Option<Arc<dyn PrefilterI>> {
    new!(kind, needles)
}

/// Creates a new prefilter, if possible, from the given set of needles and
/// returns it as a `meta::Strategy` trait object. This is used internally
/// in the meta regex engine so that a prefilter can be used *directly* as a
/// regex engine. We could use a `PrefilterI` trait object I believe, and turn
/// that into a `meta::Strategy` trait object, but then I think this would go
/// through two virtual calls. This should only need one virtual call.
///
/// See the docs on the `impl<T: PrefilterI> Strategy for T` for more info.
/// In particular, it is not always valid to call this. There are a number of
/// subtle preconditions required that cannot be checked by this function.
#[cfg(feature = "meta")]
pub(crate) fn new_as_strategy<B: AsRef<[u8]>>(
    kind: MatchKind,
    needles: &[B],
) -> Option<Arc<dyn crate::meta::Strategy>> {
    new!(kind, needles)
}

#[derive(Clone, Debug)]
pub struct Prefilter {
    pre: Arc<dyn PrefilterI>,
    is_fast: bool,
}

impl Prefilter {
    pub fn new<B: AsRef<[u8]>>(
        kind: MatchKind,
        needles: &[B],
    ) -> Option<Prefilter> {
        let pre = new(kind, needles)?;
        let is_fast = pre.is_fast();
        Some(Prefilter { pre, is_fast })
    }

    #[cfg(feature = "syntax")]
    pub fn from_hir(kind: MatchKind, hir: &Hir) -> Option<Prefilter> {
        Prefilter::from_hirs(kind, &[hir])
    }

    #[cfg(feature = "syntax")]
    pub fn from_hirs<H: Borrow<Hir>>(
        kind: MatchKind,
        hirs: &[H],
    ) -> Option<Prefilter> {
        let mut extractor = literal::Extractor::new();
        extractor.kind(literal::ExtractKind::Prefix);

        let mut prefixes = literal::Seq::empty();
        for hir in hirs.iter() {
            prefixes.union(&mut extractor.extract(hir.borrow()));
        }
        debug!(
            "prefixes (len={:?}) extracted before optimization: {:?}",
            prefixes.len(),
            prefixes
        );
        match kind {
            MatchKind::All => {
                prefixes.sort();
                prefixes.dedup();
            }
            MatchKind::LeftmostFirst => {
                prefixes.optimize_for_prefix_by_preference();
            }
        }
        debug!(
            "prefixes (len={:?}) extracted after optimization: {:?}",
            prefixes.len(),
            prefixes
        );
        prefixes.literals().and_then(|lits| Prefilter::new(kind, lits))
    }

    pub fn find(&self, haystack: &[u8], span: Span) -> Option<Span> {
        self.pre.find(haystack, span)
    }

    pub fn prefix(&self, haystack: &[u8], span: Span) -> Option<Span> {
        self.pre.prefix(haystack, span)
    }

    pub fn memory_usage(&self) -> usize {
        self.pre.memory_usage()
    }

    pub(crate) fn is_fast(&self) -> bool {
        self.is_fast
    }
}

pub(crate) trait PrefilterI:
    Debug + Send + Sync + RefUnwindSafe + UnwindSafe + 'static
{
    fn find(&self, haystack: &[u8], span: Span) -> Option<Span>;
    fn prefix(&self, haystack: &[u8], span: Span) -> Option<Span>;
    fn memory_usage(&self) -> usize;

    /// Implementations might return true here if they believe themselves to
    /// be "fast." The concept of "fast" is deliberately left vague, but in
    /// practice this usually corresponds to whether it's believed that SIMD
    /// will be used.
    ///
    /// Why do we care about this? Well, some prefilter tricks tend to come
    /// with their own bits of overhead, and so might only make sense if we
    /// know that a scan will be *much* faster than the regex engine itself.
    /// Otherwise, the trick may not be worth doing. Whether something is
    /// "much" faster than the regex engine generally boils down to whether
    /// SIMD is used.
    ///
    /// Even if this returns true, it is still possible for the prefilter to
    /// be "slow." Remember, prefilters are just heuristics. We can't really
    /// *know* a prefilter will be fast without actually trying the prefilter.
    /// (Which of course we cannot afford to do.)
    fn is_fast(&self) -> bool;
}

impl<P: PrefilterI + ?Sized> PrefilterI for Arc<P> {
    #[inline(always)]
    fn find(&self, haystack: &[u8], span: Span) -> Option<Span> {
        (&**self).find(haystack, span)
    }

    #[inline(always)]
    fn prefix(&self, haystack: &[u8], span: Span) -> Option<Span> {
        (&**self).prefix(haystack, span)
    }

    #[inline(always)]
    fn memory_usage(&self) -> usize {
        (&**self).memory_usage()
    }

    #[inline(always)]
    fn is_fast(&self) -> bool {
        (&**self).is_fast()
    }
}
