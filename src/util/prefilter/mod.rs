mod aho_corasick;
mod byteset;
mod memchr;
mod memmem;
mod teddy;

use core::{
    borrow::Borrow,
    fmt::Debug,
    panic::{RefUnwindSafe, UnwindSafe},
};

#[cfg(feature = "alloc")]
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

#[derive(Clone, Debug)]
pub struct Prefilter {
    #[cfg(not(feature = "alloc"))]
    _unused: (),
    #[cfg(feature = "alloc")]
    pre: Arc<dyn PrefilterI>,
    #[cfg(feature = "alloc")]
    is_fast: bool,
}

impl Prefilter {
    pub fn new<B: AsRef<[u8]>>(
        kind: MatchKind,
        needles: &[B],
    ) -> Option<Prefilter> {
        #[cfg(not(feature = "alloc"))]
        {
            None
        }
        #[cfg(feature = "alloc")]
        {
            let choice = Choice::new(kind, needles)?;
            Some(Prefilter::from_choice(choice))
        }
    }

    fn from_choice(choice: Choice) -> Prefilter {
        // This is a little subtle, but it's impossible to get to this point
        // because 'Choice::new' always returns 'None' when 'alloc' is
        // unavailable.
        #[cfg(not(feature = "alloc"))]
        {
            unreachable!()
        }
        #[cfg(feature = "alloc")]
        {
            let pre: Arc<dyn PrefilterI> = match choice {
                Choice::Memchr(p) => Arc::new(p),
                Choice::Memchr2(p) => Arc::new(p),
                Choice::Memchr3(p) => Arc::new(p),
                Choice::Memmem(p) => Arc::new(p),
                Choice::Teddy(p) => Arc::new(p),
                Choice::ByteSet(p) => Arc::new(p),
                Choice::AhoCorasick(p) => Arc::new(p),
            };
            let is_fast = pre.is_fast();
            Prefilter { pre, is_fast }
        }
    }

    #[cfg(feature = "syntax")]
    pub fn from_hir_prefix(kind: MatchKind, hir: &Hir) -> Option<Prefilter> {
        Prefilter::from_hirs_prefix(kind, &[hir])
    }

    #[cfg(feature = "syntax")]
    pub fn from_hirs_prefix<H: Borrow<Hir>>(
        kind: MatchKind,
        hirs: &[H],
    ) -> Option<Prefilter> {
        prefixes(kind, hirs)
            .literals()
            .and_then(|lits| Prefilter::new(kind, lits))
    }

    #[inline]
    pub fn find(&self, haystack: &[u8], span: Span) -> Option<Span> {
        #[cfg(not(feature = "alloc"))]
        {
            unreachable!()
        }
        #[cfg(feature = "alloc")]
        {
            self.pre.find(haystack, span)
        }
    }

    #[inline]
    pub fn prefix(&self, haystack: &[u8], span: Span) -> Option<Span> {
        #[cfg(not(feature = "alloc"))]
        {
            unreachable!()
        }
        #[cfg(feature = "alloc")]
        {
            self.pre.prefix(haystack, span)
        }
    }

    #[inline]
    pub fn memory_usage(&self) -> usize {
        #[cfg(not(feature = "alloc"))]
        {
            unreachable!()
        }
        #[cfg(feature = "alloc")]
        {
            self.pre.memory_usage()
        }
    }

    #[inline]
    pub(crate) fn is_fast(&self) -> bool {
        #[cfg(not(feature = "alloc"))]
        {
            unreachable!()
        }
        #[cfg(feature = "alloc")]
        {
            self.is_fast
        }
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

#[cfg(feature = "alloc")]
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

#[derive(Clone, Debug)]
pub(crate) enum Choice {
    Memchr(Memchr),
    Memchr2(Memchr2),
    Memchr3(Memchr3),
    Memmem(Memmem),
    Teddy(Teddy),
    ByteSet(ByteSet),
    AhoCorasick(AhoCorasick),
}

impl Choice {
    pub(crate) fn new<B: AsRef<[u8]>>(
        kind: MatchKind,
        needles: &[B],
    ) -> Option<Choice> {
        #[cfg(not(feature = "alloc"))]
        {
            None
        }
        #[cfg(feature = "alloc")]
        {
            // An empty set means the regex matches nothing, so no sense in
            // building a prefilter.
            if needles.len() == 0 {
                debug!(
                    "prefilter building failed: found empty set of literals"
                );
                return None;
            }
            // If the regex can match the empty string, then the prefilter
            // will by definition match at every position. This is obviously
            // completely ineffective.
            if needles.iter().any(|n| n.as_ref().is_empty()) {
                debug!(
                    "prefilter building failed: literals match empty string"
                );
                return None;
            }
            if let Some(pre) = Memchr::new(kind, needles) {
                debug!("prefilter built: memchr");
                return Some(Choice::Memchr(pre));
            }
            if let Some(pre) = Memchr2::new(kind, needles) {
                debug!("prefilter built: memchr2");
                return Some(Choice::Memchr2(pre));
            }
            if let Some(pre) = Memchr3::new(kind, needles) {
                debug!("prefilter built: memchr3");
                return Some(Choice::Memchr3(pre));
            }
            if let Some(pre) = Memmem::new(kind, needles) {
                debug!("prefilter built: memmem");
                return Some(Choice::Memmem(pre));
            }
            if let Some(pre) = Teddy::new(kind, needles) {
                debug!("prefilter built: teddy");
                return Some(Choice::Teddy(pre));
            }
            if let Some(pre) = ByteSet::new(kind, needles) {
                debug!("prefilter built: byteset");
                return Some(Choice::ByteSet(pre));
            }
            if let Some(pre) = AhoCorasick::new(kind, needles) {
                debug!("prefilter built: aho-corasick");
                return Some(Choice::AhoCorasick(pre));
            }
            debug!("prefilter building failed: no strategy could be found");
            None
        }
    }
}

/// Extracts all of the prefix literals from the given HIR expressions into a
/// single `Seq`. The literals in the sequence are ordered with respect to the
/// order of the given HIR expressions and consistent with the match semantics
/// given.
///
/// The sequence returned is "optimized." That is, they may be shrunk or even
/// truncated according to heuristics with the intent of making them more
/// useful as a prefilter. (Which translates to both using faster algorithms
/// and minimizing the false positive rate.)
///
/// Note that this erases any connection between the literals and which pattern
/// (or patterns) they came from.
///
/// The match kind given must correspond to the match semantics of the regex
/// that is represented by the HIRs given. The match semantics may change the
/// literal sequence returned.
#[cfg(feature = "syntax")]
pub(crate) fn prefixes<H>(kind: MatchKind, hirs: &[H]) -> literal::Seq
where
    H: core::borrow::Borrow<Hir>,
{
    let mut extractor = literal::Extractor::new();
    extractor.kind(literal::ExtractKind::Prefix);

    let mut prefixes = literal::Seq::empty();
    for hir in hirs {
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
    prefixes
}

/// Like `prefixes`, but for all suffixes of all matches for the given HIRs.
#[cfg(feature = "syntax")]
pub(crate) fn suffixes<H>(kind: MatchKind, hirs: &[H]) -> literal::Seq
where
    H: core::borrow::Borrow<Hir>,
{
    let mut extractor = literal::Extractor::new();
    extractor.kind(literal::ExtractKind::Suffix);

    let mut suffixes = literal::Seq::empty();
    for hir in hirs {
        suffixes.union(&mut extractor.extract(hir.borrow()));
    }
    debug!(
        "suffixes (len={:?}) extracted before optimization: {:?}",
        suffixes.len(),
        suffixes
    );
    match kind {
        MatchKind::All => {
            suffixes.sort();
            suffixes.dedup();
        }
        MatchKind::LeftmostFirst => {
            suffixes.optimize_for_suffix_by_preference();
        }
    }
    debug!(
        "suffixes (len={:?}) extracted after optimization: {:?}",
        suffixes.len(),
        suffixes
    );
    suffixes
}
