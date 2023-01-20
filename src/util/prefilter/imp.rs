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
            debug!("prefilter building failed: prefixes match empty string");
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
pub struct Prefilter(Arc<dyn PrefilterI>);

impl Prefilter {
    pub fn new<B: AsRef<[u8]>>(
        kind: MatchKind,
        needles: &[B],
    ) -> Option<Prefilter> {
        new(kind, needles).map(Prefilter)
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
        match kind {
            MatchKind::All => {
                prefixes.sort();
                prefixes.dedup();
            }
            MatchKind::LeftmostFirst => {
                prefixes.optimize_for_prefix_by_preference();
            }
        }
        prefixes.literals().and_then(|lits| Prefilter::new(kind, lits))
    }

    pub fn find(&self, haystack: &[u8], span: Span) -> Option<Span> {
        self.0.find(haystack, span)
    }

    pub fn prefix(&self, haystack: &[u8], span: Span) -> Option<Span> {
        self.0.prefix(haystack, span)
    }

    pub fn memory_usage(&self) -> usize {
        self.0.memory_usage()
    }
}

pub(crate) trait PrefilterI:
    Debug + Send + Sync + RefUnwindSafe + UnwindSafe + 'static
{
    fn find(&self, haystack: &[u8], span: Span) -> Option<Span>;
    fn prefix(&self, haystack: &[u8], span: Span) -> Option<Span>;
    fn memory_usage(&self) -> usize;
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
}

#[cfg(feature = "perf-literal-multisubstring")]
impl Packed {
    fn new<B: AsRef<[u8]>>(needles: &[B]) -> Option<Packed> {
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
        Some(Packed { packed, anchored_ac })
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
}

#[cfg(feature = "perf-literal-multisubstring")]
#[derive(Clone, Debug)]
struct AhoCorasick {
    ac: aho_corasick::AhoCorasick,
}

#[cfg(feature = "perf-literal-multisubstring")]
impl AhoCorasick {
    fn new<B: AsRef<[u8]>>(
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
            Err(err) => {
                debug!("aho-corasick prefilter failed to build: {}", err);
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
}
