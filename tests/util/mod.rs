use bstr::ByteSlice;
use regex_automata::{util::prefilter::Prefilter, Span};

#[derive(Clone, Debug)]
pub struct SubstringPrefilter(bstr::Finder<'static>);

impl SubstringPrefilter {
    pub fn new<B: AsRef<[u8]>>(needle: B) -> SubstringPrefilter {
        SubstringPrefilter(bstr::Finder::new(needle.as_ref()).into_owned())
    }
}

impl Prefilter for SubstringPrefilter {
    fn find(&self, haystack: &[u8], span: Span) -> Option<Span> {
        self.0.find(&haystack[span]).map(|i| {
            let start = span.start + i;
            let end = start + self.0.needle().len();
            Span { start, end }
        })
    }

    fn prefix(&self, haystack: &[u8], span: Span) -> Option<Span> {
        if haystack[span].starts_with_str(self.0.needle()) {
            Some(Span { end: span.start + self.0.needle().len(), ..span })
        } else {
            None
        }
    }

    fn memory_usage(&self) -> usize {
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
    fn find(&self, _haystack: &[u8], _span: Span) -> Option<Span> {
        None
    }

    fn prefix(&self, _: &[u8], _: Span) -> Option<Span> {
        None
    }

    fn memory_usage(&self) -> usize {
        0
    }
}
