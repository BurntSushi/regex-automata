use std::fmt;

use builder::{DFABuilder, MatcherBuilder};
use dfa::DFA;
use dfa_ref::DFARef;
use error::Result;
use state_id::StateID;

#[derive(Clone)]
pub struct Matcher<'a, S = usize> {
    forward: OwnOrBorrow<'a, S>,
    reverse: OwnOrBorrow<'a, S>,
}

impl<'a, S: StateID> Matcher<'a, S> {
    pub fn new(pattern: &str) -> Result<Matcher<'static>> {
        MatcherBuilder::new().build(pattern)
    }

    pub fn from_dfa(forward: DFA<S>, reverse: DFA<S>) -> Matcher<'static, S> {
        Matcher {
            forward: OwnOrBorrow::Owned(forward),
            reverse: OwnOrBorrow::Owned(reverse),
        }
    }

    pub fn forward<'b>(&'b self) -> DFARef<'b, S> {
        match self.forward {
            OwnOrBorrow::Owned(ref dfa) => dfa.as_dfa_ref(),
            OwnOrBorrow::Borrowed(dfa) => dfa,
        }
    }

    pub fn reverse<'b>(&'b self) -> DFARef<'b, S> {
        match self.reverse {
            OwnOrBorrow::Owned(ref dfa) => dfa.as_dfa_ref(),
            OwnOrBorrow::Borrowed(dfa) => dfa,
        }
    }

    pub fn is_match(&self, input: &[u8]) -> bool {
        self.forward().is_match(input)
    }

    pub fn find(&self, input: &[u8]) -> Option<(usize, usize)> {
        let end = match self.forward().find(input) {
            None => return None,
            Some(end) => end,
        };
        let start = self
            .reverse()
            .rfind(&input[..end])
            .expect("reverse search must match if forward search does");
        Some((start, end))
    }
}

#[derive(Clone)]
enum OwnOrBorrow<'a, S = usize> {
    Owned(DFA<S>),
    Borrowed(DFARef<'a, S>),
}

impl<'a, S: StateID> fmt::Debug for Matcher<'a, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Matcher")
            .field("forward", &self.forward)
            .field("reverse", &self.reverse)
            .finish()
    }
}

impl<'a, S: StateID> fmt::Debug for OwnOrBorrow<'a, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            OwnOrBorrow::Owned(ref dfa) => {
                f.debug_tuple("Owned").field(dfa).finish()
            }
            OwnOrBorrow::Borrowed(ref dfa) => {
                f.debug_tuple("Borrowed").field(dfa).finish()
            }
        }
    }
}
