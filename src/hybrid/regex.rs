use core::borrow::Borrow;

use crate::{
    hybrid::lazy,
    nfa::thompson,
    util::matchtypes::{MatchError, MultiMatch},
};

#[derive(Debug, Clone)]
pub struct Regex<N> {
    forward: lazy::InertDFA<N>,
    reverse: lazy::InertDFA<N>,
}

#[derive(Debug, Clone)]
pub struct RegexCache {
    forward: lazy::Cache,
    reverse: lazy::Cache,
}

impl<N: Borrow<thompson::NFA>> Regex<N> {
    fn try_find_leftmost_at_imp(
        &self,
        cache: &mut RegexCache,
        haystack: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Option<MultiMatch>, MatchError> {
        let (ifwd, irev) = (self.forward().as_ref(), self.reverse().as_ref());
        let mut fwd = lazy::DFA::new(&ifwd, &mut cache.forward);
        let mut rev = lazy::DFA::new(&irev, &mut cache.reverse);
        let end = match fwd.find_leftmost_fwd_at(None, haystack, start, end)? {
            None => return Ok(None),
            Some(end) => end,
        };
        // N.B. The only time we need to tell the reverse searcher the pattern
        // to match is in the overlapping case, since it's ambiguous. In the
        // leftmost case, I have tentatively convinced myself that it isn't
        // necessary and the reverse search will always find the same pattern
        // to match as the forward search. But I lack a rigorous proof. Why not
        // just provide the pattern anyway? Well, if it is needed, then leaving
        // it out gives us a chance to find a witness.
        let start = rev
            .find_leftmost_rev_at(None, haystack, start, end.offset())?
            .expect("reverse search must match if forward search does");
        assert_eq!(
            start.pattern(),
            end.pattern(),
            "forward and reverse search must match same pattern",
        );
        assert!(start.offset() <= end.offset());
        Ok(Some(MultiMatch::new(end.pattern(), start.offset(), end.offset())))
    }

    fn forward(&self) -> &lazy::InertDFA<N> {
        &self.forward
    }

    fn reverse(&self) -> &lazy::InertDFA<N> {
        &self.reverse
    }
}
