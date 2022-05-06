use regex_automata::{
    util::prefilter::{self, Candidate, Prefilter},
    Span,
};

#[derive(Clone, Debug)]
pub struct SubstringPrefilter(bstr::Finder<'static>);

impl SubstringPrefilter {
    pub fn new<B: AsRef<[u8]>>(needle: B) -> SubstringPrefilter {
        SubstringPrefilter(bstr::Finder::new(needle.as_ref()).into_owned())
    }
}

impl Prefilter for SubstringPrefilter {
    #[inline]
    fn next_candidate(
        &self,
        _state: &mut prefilter::State,
        haystack: &[u8],
        at: usize,
    ) -> Candidate {
        self.0
            .find(&haystack[at..])
            .map(|i| {
                let start = at + i;
                let end = start + self.0.needle().len();
                Candidate::PossibleMatch(Span::new(start, end))
            })
            .unwrap_or(Candidate::None)
    }

    fn heap_bytes(&self) -> usize {
        self.0.needle().len()
    }
}

/// A prefilter that always returns `Candidate::None`, even if it's a false
/// negative. This is useful for confirming that a prefilter is actually
/// active by asserting an incorrect result.
#[derive(Clone, Debug)]
pub struct BunkPrefilter(());

impl BunkPrefilter {
    pub fn new() -> BunkPrefilter {
        BunkPrefilter(())
    }
}

impl Prefilter for BunkPrefilter {
    #[inline]
    fn next_candidate(
        &self,
        _state: &mut prefilter::State,
        _haystack: &[u8],
        _at: usize,
    ) -> Candidate {
        Candidate::None
    }

    fn heap_bytes(&self) -> usize {
        0
    }
}
