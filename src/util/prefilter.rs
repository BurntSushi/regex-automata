use core::{
    fmt::Debug,
    panic::{RefUnwindSafe, UnwindSafe},
};

use crate::Span;

// So maybe the PikeVM is kind of unique here in that it's trying to mix low
// level with high level. The DFA-based regex engines aren't so bad because
// callers can always build their own forward/reverse DFAs and piece things
// together that way. Not ideal, but not the end of the world. Certainly
// nowhere as difficult as building your own PikeVM. So maybe the answer
// really is that the PikeVM is low-level APIs only, and then we add a
// nfa::thompson::pikevm::regex module that gives the higher level niceties.
//
// I guess the same will be true for the backtracker as well. And onepass.

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
            Candidate::Match(ref m) => Some(m.start()),
            Candidate::PossibleMatch(ref m) => Some(m.start()),
        }
    }
}

pub trait Prefilter: Debug + Send + Sync + RefUnwindSafe + UnwindSafe {
    fn find(
        &self,
        state: &mut State,
        haystack: &[u8],
        span: Span,
    ) -> Candidate;

    fn memory_usage(&self) -> usize;

    fn reports_false_positives(&self) -> bool {
        true
    }
}

impl<'a, P: Prefilter + ?Sized> Prefilter for &'a P {
    #[inline]
    fn find(
        &self,
        state: &mut State,
        haystack: &[u8],
        span: Span,
    ) -> Candidate {
        (**self).find(state, haystack, span)
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

#[derive(Clone, Debug)]
pub struct Scanner<'p> {
    prefilter: &'p dyn Prefilter,
    state: State,
}

impl<'p> Scanner<'p> {
    #[inline]
    pub fn new(prefilter: &'p dyn Prefilter) -> Scanner<'p> {
        Scanner { prefilter, state: State::new() }
    }

    pub(crate) fn is_effective(&mut self, at: usize) -> bool {
        self.state.is_effective(at)
    }

    pub(crate) fn reports_false_positives(&self) -> bool {
        self.prefilter.reports_false_positives()
    }

    pub(crate) fn find(&mut self, bytes: &[u8], span: Span) -> Candidate {
        self.prefilter.find(&mut self.state, bytes, span)
    }
}

/// State tracks state associated with the effectiveness of a prefilter.
///
/// While the specifics on how it works are an implementation detail, the
/// idea here is that this will make heuristic decisions about whether it's
/// advantageuous to continue executing a prefilter or not, typically based on
/// how many bytes the prefilter tends to skip.
///
/// A prefilter state should be created for each search. (Where creating an
/// iterator is typically treated as a single search.)
#[derive(Clone, Debug)]
pub struct State {
    /// We currently don't keep track of anything and always execute
    /// prefilters. This may change in the future.
    _priv: (),
}

impl State {
    /// Create a fresh prefilter state.
    fn new() -> State {
        State { _priv: () }
    }

    /// Return true if and only if this state indicates that a prefilter is
    /// still effective. If the prefilter is not effective, then this state
    /// is rendered "inert." At which point, all subsequent calls to
    /// `is_effective` on this state will return `false`.
    ///
    /// `at` should correspond to the current starting position of the search.
    ///
    /// Callers typically do not need to use this, as it represents the
    /// default implementation of
    /// [`Prefilter::is_effective`](trait.Prefilter.html#tymethod.is_effective).
    fn is_effective(&mut self, _at: usize) -> bool {
        true
    }
}

/// A `Prefilter` implementation that reports a possible match at every
/// position.
///
/// This should generally not be used as an actual prefilter. It is only
/// useful when one needs to represent the absence of a prefilter in a generic
/// context. For example, a [`dfa::regex::Regex`](crate::dfa::regex::Regex)
/// uses this prefilter by default to indicate that no prefilter should be
/// used.
///
/// A `None` prefilter value cannot be constructed.
#[derive(Clone, Debug)]
pub struct None {
    _priv: (),
}

impl Prefilter for None {
    fn find(&self, _: &mut State, _: &[u8], span: Span) -> Candidate {
        Candidate::PossibleMatch(Span::new(span.start(), span.start()))
    }

    fn memory_usage(&self) -> usize {
        0
    }
}
