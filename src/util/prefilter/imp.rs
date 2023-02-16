use core::{
    borrow::Borrow,
    fmt::Debug,
    panic::{RefUnwindSafe, UnwindSafe},
};

use alloc::sync::Arc;

#[cfg(feature = "perf-literal-multisubstring")]
use aho_corasick::{self, packed};
#[cfg(feature = "perf-literal-substring")]
use memchr::memmem;
#[cfg(feature = "syntax")]
use regex_syntax::hir::{literal, Hir};

use crate::util::{
    memchr::{memchr, memchr2, memchr3},
    search::{MatchKind, Span},
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
        #[cfg(feature = "perf-literal-substring")]
        if needles.len() == 1 {
            return Some(match needles[0].as_ref().len() {
                0 => return None,
                1 => {
                    debug!("prefilter built: memchr");
                    Arc::new(Memchr(needles[0].as_ref()[0]))
                }
                _ => {
                    debug!("prefilter built: memmem");
                    Arc::new(Memmem::new(needles[0].as_ref()))
                }
            });
        }
        #[cfg(feature = "perf-literal-substring")]
        if needles.len() == 2 && needles.iter().all(|n| n.as_ref().len() == 1)
        {
            debug!("prefilter built: memchr2");
            let b1 = needles[0].as_ref()[0];
            let b2 = needles[1].as_ref()[0];
            return Some(Arc::new(Memchr2(b1, b2)));
        }
        #[cfg(feature = "perf-literal-substring")]
        if needles.len() == 3 && needles.iter().all(|n| n.as_ref().len() == 1)
        {
            debug!("prefilter built: memchr3");
            let b1 = needles[0].as_ref()[0];
            let b2 = needles[1].as_ref()[0];
            let b3 = needles[2].as_ref()[0];
            return Some(Arc::new(Memchr3(b1, b2, b3)));
        }
        // Packed substring search only supports leftmost-first matching.
        #[cfg(feature = "perf-literal-multisubstring")]
        if kind == MatchKind::LeftmostFirst {
            if let Some(packed) = Packed::new(needles) {
                debug!("prefilter built: packed (Teddy)");
                return Some(Arc::new(packed));
            }
        }
        #[cfg(feature = "perf-literal-multisubstring")]
        if let Some(byteset) = ByteSet::new(needles) {
            debug!("prefilter built: byteset");
            return Some(Arc::new(byteset));
        }
        #[cfg(feature = "perf-literal-multisubstring")]
        if let Some(ac) = AhoCorasick::new(kind, needles) {
            debug!("prefilter built: aho-corasick");
            return Some(Arc::new(ac));
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

#[cfg(feature = "perf-literal-substring")]
#[derive(Clone, Debug)]
struct Memchr(u8);

#[cfg(feature = "perf-literal-substring")]
impl PrefilterI for Memchr {
    fn find(&self, haystack: &[u8], span: Span) -> Option<Span> {
        memchr(self.0, &haystack[span]).map(|i| {
            let start = span.start + i;
            let end = start + 1;
            Span { start, end }
        })
    }

    fn prefix(&self, haystack: &[u8], span: Span) -> Option<Span> {
        let b = *haystack.get(span.start)?;
        if self.0 == b {
            Some(Span { start: span.start, end: span.start + 1 })
        } else {
            None
        }
    }

    fn memory_usage(&self) -> usize {
        0
    }

    fn is_fast(&self) -> bool {
        true
    }
}

#[cfg(feature = "perf-literal-substring")]
#[derive(Clone, Debug)]
struct Memchr2(u8, u8);

#[cfg(feature = "perf-literal-substring")]
impl PrefilterI for Memchr2 {
    fn find(&self, haystack: &[u8], span: Span) -> Option<Span> {
        memchr2(self.0, self.1, &haystack[span]).map(|i| {
            let start = span.start + i;
            let end = start + 1;
            Span { start, end }
        })
    }

    fn prefix(&self, haystack: &[u8], span: Span) -> Option<Span> {
        let b = *haystack.get(span.start)?;
        if self.0 == b || self.1 == b {
            Some(Span { start: span.start, end: span.start + 1 })
        } else {
            None
        }
    }

    fn memory_usage(&self) -> usize {
        0
    }

    fn is_fast(&self) -> bool {
        true
    }
}

#[cfg(feature = "perf-literal-substring")]
#[derive(Clone, Debug)]
struct Memchr3(u8, u8, u8);

#[cfg(feature = "perf-literal-substring")]
impl PrefilterI for Memchr3 {
    fn find(&self, haystack: &[u8], span: Span) -> Option<Span> {
        memchr3(self.0, self.1, self.2, &haystack[span]).map(|i| {
            let start = span.start + i;
            let end = start + 1;
            Span { start, end }
        })
    }

    fn prefix(&self, haystack: &[u8], span: Span) -> Option<Span> {
        let b = *haystack.get(span.start)?;
        if self.0 == b || self.1 == b || self.2 == b {
            Some(Span { start: span.start, end: span.start + 1 })
        } else {
            None
        }
    }

    fn memory_usage(&self) -> usize {
        0
    }

    fn is_fast(&self) -> bool {
        true
    }
}

#[derive(Clone, Debug)]
struct ByteSet([bool; 256]);

impl ByteSet {
    fn new<B: AsRef<[u8]>>(needles: &[B]) -> Option<ByteSet> {
        let mut set = [false; 256];
        for needle in needles.iter() {
            let needle = needle.as_ref();
            if needle.len() != 1 {
                return None;
            }
            set[usize::from(needle[0])] = true;
        }
        Some(ByteSet(set))
    }
}

impl PrefilterI for ByteSet {
    fn find(&self, haystack: &[u8], span: Span) -> Option<Span> {
        haystack[span].iter().position(|&b| self.0[usize::from(b)]).map(|i| {
            let start = span.start + i;
            let end = start + 1;
            Span { start, end }
        })
    }

    fn prefix(&self, haystack: &[u8], span: Span) -> Option<Span> {
        let b = *haystack.get(span.start)?;
        if self.0[usize::from(b)] {
            Some(Span { start: span.start, end: span.start + 1 })
        } else {
            None
        }
    }

    fn memory_usage(&self) -> usize {
        0
    }

    fn is_fast(&self) -> bool {
        false
    }
}

#[cfg(feature = "perf-literal-substring")]
#[derive(Clone, Debug)]
struct Memmem(memmem::Finder<'static>);

#[cfg(feature = "perf-literal-substring")]
impl Memmem {
    fn new(needle: &[u8]) -> Memmem {
        Memmem(memmem::Finder::new(needle).into_owned())
    }
}

#[cfg(feature = "perf-literal-substring")]
impl PrefilterI for Memmem {
    fn find(&self, haystack: &[u8], span: Span) -> Option<Span> {
        self.0.find(&haystack[span]).map(|i| {
            let start = span.start + i;
            let end = start + self.0.needle().len();
            Span { start, end }
        })
    }

    fn prefix(&self, haystack: &[u8], span: Span) -> Option<Span> {
        let needle = self.0.needle();
        if haystack[span].starts_with(needle) {
            Some(Span { end: span.start + needle.len(), ..span })
        } else {
            None
        }
    }

    fn memory_usage(&self) -> usize {
        self.0.needle().len()
    }

    fn is_fast(&self) -> bool {
        true
    }
}

#[cfg(feature = "perf-literal-multisubstring")]
#[derive(Clone, Debug)]
struct Packed {
    packed: packed::Searcher,
    /// When running an anchored search, the packed searcher can't handle it so
    /// we defer to Aho-Corasick itself. Kind of sad, but changing the packed
    /// searchers to support anchored search would be difficult at worst and
    /// annoying at best. Since packed searchers only apply to small numbers of
    /// literals, we content ourselves that this is not much of an added cost.
    /// (That packed searchers only work with a small number of literals is
    /// also why we use a DFA here. Otherwise, the memory usage of a DFA would
    /// likely be unacceptable.)
    anchored_ac: aho_corasick::dfa::DFA,
    /// The length of the smallest literal we look for.
    ///
    /// We use this as a hueristic to figure out whether this will be "fast" or
    /// not. Generally, the longer the better, because longer needles are more
    /// discriminating and thus reduce false positive rate.
    minimum_len: usize,
}

#[cfg(feature = "perf-literal-multisubstring")]
impl Packed {
    fn new<B: AsRef<[u8]>>(needles: &[B]) -> Option<Packed> {
        let minimum_len =
            needles.iter().map(|n| n.as_ref().len()).min().unwrap_or(0);
        let packed = packed::Config::new()
            .match_kind(packed::MatchKind::LeftmostFirst)
            .builder()
            .extend(needles)
            .build()?;
        let anchored_ac = aho_corasick::dfa::DFA::builder()
            .match_kind(aho_corasick::MatchKind::LeftmostFirst)
            .start_kind(aho_corasick::StartKind::Anchored)
            .prefilter(false)
            .build(needles)
            .ok()?;
        Some(Packed { packed, anchored_ac, minimum_len })
    }
}

#[cfg(feature = "perf-literal-multisubstring")]
impl PrefilterI for Packed {
    fn find(&self, haystack: &[u8], span: Span) -> Option<Span> {
        let ac_span = aho_corasick::Span { start: span.start, end: span.end };
        self.packed
            .find_in(haystack, ac_span)
            .map(|m| Span { start: m.start(), end: m.end() })
    }

    fn prefix(&self, haystack: &[u8], span: Span) -> Option<Span> {
        use aho_corasick::automaton::Automaton;
        let input = aho_corasick::Input::new(haystack)
            .anchored(aho_corasick::Anchored::Yes)
            .span(span.start..span.end);
        self.anchored_ac
            .try_find(&input)
            // OK because we build the DFA with anchored support.
            .expect("aho-corasick DFA should never fail")
            .map(|m| Span { start: m.start(), end: m.end() })
    }

    fn memory_usage(&self) -> usize {
        use aho_corasick::automaton::Automaton;
        self.packed.memory_usage() + self.anchored_ac.memory_usage()
    }

    fn is_fast(&self) -> bool {
        // Teddy is usually quite fast, but I have seen some cases where a
        // large number of literals can overwhelm it and make it not so fast.
        // We make an educated but conservative guess at a limit, at which
        // point, we're not so comfortable thinking Teddy is "fast."
        //
        // Well... this used to incorporate a "limit" on the *number* of
        // literals, but I have since changed it to a minimum on the *smallest*
        // literal. Namely, when there is a very small literal (1 or 2 bytes),
        // it is far more likely that it leads to a higher false positive rate.
        // (Although, of course, not always. For example, 'zq' is likely to
        // have a very low false positive rate.) But when we have 3 bytes, we
        // have a really good chance of being quite discriminatory and thus
        // fast.
        //
        // We may still want to add some kind of limit on the number of
        // literals here, but keep in mind that Teddy already has its own
        // somewhat small limit (64 at time of writing). The main issue here
        // is that if 'is_fast' is false, it opens the door for the reverse
        // inner optimization to kick in. We really only want to resort to the
        // reverse inner optimization if we absolutely must.
        self.minimum_len >= 3
    }
}

#[cfg(feature = "perf-literal-multisubstring")]
#[derive(Clone, Debug)]
pub(crate) struct AhoCorasick {
    ac: aho_corasick::AhoCorasick,
}

#[cfg(feature = "perf-literal-multisubstring")]
impl AhoCorasick {
    pub(crate) fn new<B: AsRef<[u8]>>(
        kind: MatchKind,
        needles: &[B],
    ) -> Option<AhoCorasick> {
        let ac_match_kind = match kind {
            MatchKind::LeftmostFirst => aho_corasick::MatchKind::LeftmostFirst,
            MatchKind::All => aho_corasick::MatchKind::Standard,
        };
        let ac_kind = if needles.len() <= 500 {
            aho_corasick::AhoCorasickKind::DFA
        } else {
            aho_corasick::AhoCorasickKind::ContiguousNFA
        };
        let result = aho_corasick::AhoCorasick::builder()
            .kind(ac_kind)
            .match_kind(ac_match_kind)
            .start_kind(aho_corasick::StartKind::Both)
            // We try to handle all of the prefilter cases here, and only
            // use Aho-Corasick for the actual automaton. The aho-corasick
            // crate does have some extra prefilters, namely, looking for rare
            // bytes to feed to memchr{,2,3} instead of just the first byte.
            // If we end up wanting those---and they are somewhat tricky to
            // implement---then we could port them to this crate. Although,
            // IIRC, they do require some more prefilter infrastructure.
            //
            // The main reason for doing things this way is so we have a
            // complete and easy to understand picture of which prefilters are
            // available and how they work. Otherwise it seems too easy to get
            // into a situation where we have prefilters layered on top of
            // prefilter, and that might have unintended consequences.
            .prefilter(false)
            .build(needles);
        let ac = match result {
            Ok(ac) => ac,
            Err(_err) => {
                debug!("aho-corasick prefilter failed to build: {}", _err);
                return None;
            }
        };
        Some(AhoCorasick { ac })
    }
}

#[cfg(feature = "perf-literal-multisubstring")]
impl PrefilterI for AhoCorasick {
    fn find(&self, haystack: &[u8], span: Span) -> Option<Span> {
        let input =
            aho_corasick::Input::new(haystack).span(span.start..span.end);
        self.ac.find(input).map(|m| Span { start: m.start(), end: m.end() })
    }

    fn prefix(&self, haystack: &[u8], span: Span) -> Option<Span> {
        let input = aho_corasick::Input::new(haystack)
            .anchored(aho_corasick::Anchored::Yes)
            .span(span.start..span.end);
        self.ac.find(input).map(|m| Span { start: m.start(), end: m.end() })
    }

    fn memory_usage(&self) -> usize {
        self.ac.memory_usage()
    }

    fn is_fast(&self) -> bool {
        // Aho-Corasick is never considered "fast" because it's never going to
        // be even close to an order of magnitude faster than the regex engine
        // itself (assuming a DFA is used). In fact, it is usually slower. The
        // magic of Aho-Corasick is that it can search a *large* number of
        // literals with a relatively small amount of memory. The regex engines
        // are far more wasteful.
        //
        // Aho-Corasick may be "fast" when the regex engine corresponds to,
        // say, the PikeVM. That happens when the lazy DFA couldn't be built or
        // used for some reason. But in these cases, the regex itself is likely
        // quite big and we're probably hosed no matter what we do. (In this
        // case, the best bet is for the caller to increase some of the memory
        // limits on the hybrid cache capacity and hope that's enough.)
        false
    }
}
