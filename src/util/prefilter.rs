use core::{
    fmt::Debug,
    panic::{RefUnwindSafe, UnwindSafe},
};

use crate::Span;

// DREAM: When writing the prefilter APIs below, I mostly looked at what the
// regex crate was already doing in order to get regex-automata merged into
// the regex crate expeditiously. However, I would very much like to improve
// how prefilters are done, especially for the multi-regex case. I suspect the
// interface for that will look quite a bit different, since you really want
// the prefilter to report for which patterns there was a match. For now, we
// just require that the prefilter report match spans.

#[derive(Clone, Debug)]
pub enum Candidate {
    None,
    Match(Span),
    PossibleMatch(Span),
}

impl Candidate {
    /// Convert this candidate into an option.
    ///
    /// This is useful when callers do not distinguish between true positives
    /// and false positives (i.e., the caller must always confirm the match in
    /// order to update some other state).
    ///
    /// The byte offset in the option returned corresponds to the starting
    /// position of the possible match.
    pub fn into_option(self) -> Option<usize> {
        match self {
            Candidate::None => None,
            Candidate::Match(ref sp) => Some(sp.start),
            Candidate::PossibleMatch(ref sp) => Some(sp.start),
        }
    }
}

pub trait Prefilter: Debug + Send + Sync + RefUnwindSafe + UnwindSafe {
    fn find(&self, haystack: &[u8], span: Span) -> Candidate;

    fn memory_usage(&self) -> usize;

    fn reports_false_positives(&self) -> bool {
        true
    }
}

impl<'a, P: Prefilter + ?Sized> Prefilter for &'a P {
    #[inline]
    fn find(&self, haystack: &[u8], span: Span) -> Candidate {
        (**self).find(haystack, span)
    }

    #[inline]
    fn memory_usage(&self) -> usize {
        (**self).memory_usage()
    }

    #[inline]
    fn reports_false_positives(&self) -> bool {
        (**self).reports_false_positives()
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
    fn find(&self, _: &[u8], span: Span) -> Candidate {
        let span = Span { start: span.start, end: span.start };
        Candidate::PossibleMatch(span)
    }

    fn memory_usage(&self) -> usize {
        0
    }
}
