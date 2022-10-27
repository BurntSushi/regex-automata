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

pub trait Prefilter: Debug + Send + Sync + RefUnwindSafe + UnwindSafe {
    fn find(&self, haystack: &[u8], span: Span) -> Option<Span>;

    fn memory_usage(&self) -> usize;
}

impl<'a, P: Prefilter + ?Sized> Prefilter for &'a P {
    #[inline]
    fn find(&self, haystack: &[u8], span: Span) -> Option<Span> {
        (**self).find(haystack, span)
    }

    #[inline]
    fn memory_usage(&self) -> usize {
        (**self).memory_usage()
    }
}

pub fn new<B: AsRef<[u8]>>(_needles: &[B]) -> Arc<dyn Prefilter + 'static> {
    todo!()
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
            set[usize::from(*needle.get(0)?)] = true;
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

    fn memory_usage(&self) -> usize {
        self.0.needle().len()
    }
}

#[cfg(feature = "perf-literal-multisubstring")]
#[derive(Clone, Debug)]
struct Packed(packed::Searcher);

#[cfg(feature = "perf-literal-multisubstring")]
impl Packed {
    fn new<B: AsRef<[u8]>>(needles: &[B]) -> Option<Packed> {
        packed::Config::new()
            .match_kind(packed::MatchKind::LeftmostFirst)
            .builder()
            .extend(needles)
            .build()
            .map(Packed)
    }
}

#[cfg(feature = "perf-literal-multisubstring")]
impl Prefilter for Packed {
    fn find(&self, haystack: &[u8], span: Span) -> Option<Span> {
        self.0
            .find_at(haystack, span.start)
            .map(|m| Span { start: m.start(), end: m.end() })
    }

    fn memory_usage(&self) -> usize {
        self.0.heap_bytes()
    }
}

#[cfg(feature = "perf-literal-multisubstring")]
#[derive(Clone, Debug)]
struct AhoCorasick(aho_corasick::AhoCorasick<u32>);

#[cfg(feature = "perf-literal-multisubstring")]
impl AhoCorasick {
    fn new<B: AsRef<[u8]>>(needles: &[B]) -> Option<AhoCorasick> {
        AhoCorasickBuilder::new()
            .match_kind(aho_corasick::MatchKind::LeftmostFirst)
            .dfa(needles.len() <= 5_000)
            // We try to handle all of the prefilter cases here, and only
            // use Aho-Corasick for the actual automaton. The aho-corasick
            // crate does have some extra prefilters, namely, looking for
            // rare bytes to feed to memchr{,2,3} instead of just the first
            // byte. If we end up wanting those---and they are somewhat tricky
            // to implement---then we could port them to this crate.
            .prefilter(false)
            .build_with_size::<u32, _, _>(needles)
            .ok()
            .map(AhoCorasick)
    }
}

#[cfg(feature = "perf-literal-multisubstring")]
impl Prefilter for AhoCorasick {
    fn find(&self, haystack: &[u8], span: Span) -> Option<Span> {
        self.0.find(&haystack[span]).map(|m| {
            let start = span.start + m.start();
            let end = start + m.len();
            Span { start, end }
        })
    }

    fn memory_usage(&self) -> usize {
        self.0.heap_bytes()
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

    fn memory_usage(&self) -> usize {
        0
    }
}
