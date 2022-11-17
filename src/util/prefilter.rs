use core::{
    fmt::Debug,
    panic::{RefUnwindSafe, UnwindSafe},
};

use alloc::sync::Arc;

#[cfg(feature = "perf-literal-multisubstring")]
use aho_corasick::{self, packed, AhoCorasickBuilder};
#[cfg(feature = "perf-literal-substring")]
use memchr::{memchr, memchr2, memchr3, memmem};

use crate::util::search::{MatchKind, Span};

pub trait Prefilter:
    Debug + Send + Sync + RefUnwindSafe + UnwindSafe + 'static
{
    fn find(&self, haystack: &[u8], span: Span) -> Option<Span>;
    fn prefix(&self, haystack: &[u8], span: Span) -> Option<Span>;

    fn memory_usage(&self) -> usize;
}

macro_rules! new {
    ($needles:ident) => {{
        let needles = $needles;
        if needles.len() == 0 {
            return None;
        }
        #[cfg(feature = "perf-literal-substring")]
        if needles.len() == 1 {
            return Some(match needles[0].as_ref().len() {
                0 => return None,
                1 => Arc::new(Memchr(needles[0].as_ref()[0])),
                _ => Arc::new(Memmem::new(needles[0].as_ref())),
            });
        }
        #[cfg(feature = "perf-literal-multisubstring")]
        if let Some(byteset) = ByteSet::new(needles) {
            return Some(Arc::new(byteset));
        }
        #[cfg(feature = "perf-literal-multisubstring")]
        if let Some(packed) = Packed::new(needles) {
            return Some(Arc::new(packed));
        }
        #[cfg(feature = "perf-literal-multisubstring")]
        if let Some(ac) = AhoCorasick::new(needles) {
            return Some(Arc::new(ac));
        }
        None
    }};
}

pub fn new<B: AsRef<[u8]>>(
    needles: &[B],
) -> Option<Arc<dyn Prefilter + 'static>> {
    new!(needles)
}

pub(crate) fn new_as_strategy<B: AsRef<[u8]>>(
    needles: &[B],
) -> Option<Arc<dyn crate::meta::Strategy + 'static>> {
    new!(needles)
}

#[cfg(feature = "perf-literal-substring")]
#[derive(Clone, Debug)]
struct Memchr(u8);

#[cfg(feature = "perf-literal-substring")]
impl Prefilter for Memchr {
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
impl Prefilter for Memchr2 {
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
impl Prefilter for Memchr3 {
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

impl Prefilter for ByteSet {
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
impl Prefilter for Memmem {
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
    anchored_ac: aho_corasick::AhoCorasick<u32>,
}

#[cfg(feature = "perf-literal-multisubstring")]
impl Packed {
    fn new<B: AsRef<[u8]>>(needles: &[B]) -> Option<Packed> {
        let packed = packed::Config::new()
            .match_kind(packed::MatchKind::LeftmostFirst)
            .builder()
            .extend(needles)
            .build()?;
        let anchored_ac = AhoCorasickBuilder::new()
            .match_kind(aho_corasick::MatchKind::LeftmostFirst)
            .anchored(true)
            // OK because packed searchers only get build when the number of
            // needles is very small.
            .dfa(true)
            .prefilter(false)
            .build_with_size::<u32, _, _>(needles)
            .ok()?;
        Some(Packed { packed, anchored_ac })
    }
}

#[cfg(feature = "perf-literal-multisubstring")]
impl Prefilter for Packed {
    fn find(&self, haystack: &[u8], span: Span) -> Option<Span> {
        self.packed
            .find_at(haystack, span.start)
            .map(|m| Span { start: m.start(), end: m.end() })
    }

    fn prefix(&self, haystack: &[u8], span: Span) -> Option<Span> {
        self.anchored_ac.find(&haystack[span]).map(|m| {
            let start = span.start + m.start();
            let end = start + m.len();
            Span { start, end }
        })
    }

    fn memory_usage(&self) -> usize {
        self.packed.heap_bytes() + self.anchored_ac.heap_bytes()
    }
}

#[cfg(feature = "perf-literal-multisubstring")]
#[derive(Clone, Debug)]
struct AhoCorasick {
    ac: aho_corasick::AhoCorasick<u32>,
    /// An unanchored Aho-Corasick searcher cannot also do an anchored search.
    /// So we need to build an entirely separate one just for that.
    ///
    /// In the traditional Aho-Corasick formulation, adding support for
    /// anchored search is as easy as treating as a trie and not following
    /// failure transitions. That would work fine, except the aho-corasick
    /// crate might actually use a DFA internally, which doesn't have any
    /// explicit failure transitions. (The point of the DFA is to erase the
    /// failure transitions completely by pre-computing them into the DFA's
    /// transition table.) Or even a packed searcher. And so it can't just
    /// provide anchored search on demand. Thus, that's why 'anchored' is a
    /// build-time setting and we need a second one to support anchored search.
    anchored_ac: aho_corasick::AhoCorasick<u32>,
}

#[cfg(feature = "perf-literal-multisubstring")]
impl AhoCorasick {
    fn new<B: AsRef<[u8]>>(needles: &[B]) -> Option<AhoCorasick> {
        let ac = AhoCorasickBuilder::new()
            .match_kind(aho_corasick::MatchKind::LeftmostFirst)
            .dfa(needles.len() <= 5_000)
            // We try to handle all of the prefilter cases here, and only
            // use Aho-Corasick for the actual automaton. The aho-corasick
            // crate does have some extra prefilters, namely, looking for
            // rare bytes to feed to memchr{,2,3} instead of just the first
            // byte. If we end up wanting those---and they are somewhat tricky
            // to implement---then we could port them to this crate. Although,
            // IIRC, they do require some more prefilter infrastructure.
            .prefilter(false)
            .build_with_size::<u32, _, _>(needles)
            .ok()?;
        let anchored_ac = AhoCorasickBuilder::new()
            .match_kind(aho_corasick::MatchKind::LeftmostFirst)
            .anchored(true)
            .dfa(needles.len() <= 5_000)
            .prefilter(false)
            .build_with_size::<u32, _, _>(needles)
            .ok()?;
        Some(AhoCorasick { ac, anchored_ac })
    }
}

#[cfg(feature = "perf-literal-multisubstring")]
impl Prefilter for AhoCorasick {
    fn find(&self, haystack: &[u8], span: Span) -> Option<Span> {
        self.ac.find(&haystack[span]).map(|m| {
            let start = span.start + m.start();
            let end = start + m.len();
            Span { start, end }
        })
    }

    fn prefix(&self, haystack: &[u8], span: Span) -> Option<Span> {
        self.anchored_ac.find(&haystack[span]).map(|m| {
            let start = span.start + m.start();
            let end = start + m.len();
            Span { start, end }
        })
    }

    fn memory_usage(&self) -> usize {
        self.ac.heap_bytes() + self.anchored_ac.heap_bytes()
    }
}

/// A `Prefilter` implementation that reports a possible match at every
/// position.
///
/// This should generally not be used as an actual prefilter. It
/// is only useful when one needs to represent the absence of a
/// prefilter in a generic context at the type level. For example, a
/// [`dfa::regex::Regex`](crate::dfa::regex::Regex) uses this prefilter by
/// default to indicate that no prefilter should be used.
///
/// A `None` prefilter value cannot be constructed.
#[derive(Clone, Debug)]
pub struct None {
    _priv: (),
}

impl Prefilter for None {
    fn find(&self, _: &[u8], span: Span) -> Option<Span> {
        Some(Span { start: span.start, end: span.start })
    }

    fn prefix(&self, _: &[u8], span: Span) -> Option<Span> {
        Some(Span { start: span.start, end: span.start })
    }

    fn memory_usage(&self) -> usize {
        0
    }
}
