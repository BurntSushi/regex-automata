pub(crate) use self::state::{
    State, StateBuilderEmpty, StateBuilderMatches, StateBuilderNFA,
};

use crate::{
    nfa::thompson,
    util::{matchtypes::MatchKind, sparse_set::SparseSet},
};

mod state;

/// Start represents the four possible starting configurations of a DFA based
/// on the text being searched. Ultimately, this along with a pattern ID (if
/// specified) is what selects the start state to use in a DFA.
///
/// In a DFA that doesn't have starting states for each pattern, then it will
/// have a maximum of four starting states. If the DFA was compiled with start
/// states for each pattern, then it will have a maximum of four starting
/// states for searching for any pattern, and then another maximum of four
/// starting states for executing an anchored search for each pattern.
///
/// This ends up being represented as a table in the DFA where the stride of
/// that table is 4, and each entry is an index into the state transition
/// table. Note though that multiple entries in the table might point to the
/// same state if the states would otherwise be equivalent. (This is guaranteed
/// by minimization and may even be accomplished by normal determinization,
/// since it attempts to reuse equivalent states too.)
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Start {
    /// This occurs when the starting position is not any of the ones below.
    NonWordByte = 0,
    /// This occurs when the byte immediately preceding the start of the search
    /// is an ASCII word byte.
    WordByte = 1,
    /// This occurs when the starting position of the search corresponds to the
    /// beginning of the haystack.
    Text = 2,
    /// This occurs when the byte immediately preceding the start of the search
    /// is a line terminator. Specifically, `\n`.
    Line = 3,
}

impl Start {
    /// Return the starting state corresponding to the given integer. If no
    /// starting state exists for the given integer, then None is returned.
    pub(crate) fn from_usize(n: usize) -> Option<Start> {
        match n {
            0 => Some(Start::NonWordByte),
            1 => Some(Start::WordByte),
            2 => Some(Start::Text),
            3 => Some(Start::Line),
            _ => None,
        }
    }

    /// Returns the total number of starting state configurations.
    pub(crate) fn count() -> usize {
        4
    }

    /// Returns the starting state configuration for the given search
    /// parameters. If the given offset range is not valid, then this panics.
    #[inline(always)]
    pub(crate) fn from_position_fwd(
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Start {
        assert!(
            bytes.get(start..end).is_some(),
            "{}..{} is invalid",
            start,
            end
        );
        if start == 0 {
            Start::Text
        } else if bytes[start - 1] == b'\n' {
            Start::Line
        } else if crate::util::is_word_byte(bytes[start - 1]) {
            Start::WordByte
        } else {
            Start::NonWordByte
        }
    }

    /// Returns the starting state configuration for a reverse search with the
    /// given search parameters. If the given offset range is not valid, then
    /// this panics.
    #[inline(always)]
    pub(crate) fn from_position_rev(
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Start {
        assert!(
            bytes.get(start..end).is_some(),
            "{}..{} is invalid",
            start,
            end
        );
        if end == bytes.len() {
            Start::Text
        } else if bytes[end] == b'\n' {
            Start::Line
        } else if crate::util::is_word_byte(bytes[end]) {
            Start::WordByte
        } else {
            Start::NonWordByte
        }
    }

    /// Return this starting configuration as an integer. It is guaranteed to
    /// be less than `Start::count()`.
    #[inline(always)]
    pub(crate) fn as_usize(&self) -> usize {
        *self as usize
    }
}

pub(crate) fn add_nfa_states(
    nfa: &thompson::NFA,
    match_kind: MatchKind,
    set: &SparseSet,
    builder: &mut StateBuilderNFA,
) {
    // We use this determinizer's scratch space to store the NFA state IDs
    // because this state may not wind up being used if it's identical to
    // some other existing state. When that happers, the caller should put
    // the scratch allocation back into the determinizer.
    for nfa_id in set {
        match *nfa.state(nfa_id) {
            thompson::State::Range { .. } => {
                builder.add_nfa_state_id(nfa_id);
            }
            thompson::State::Sparse { .. } => {
                builder.add_nfa_state_id(nfa_id);
            }
            thompson::State::Look { look, .. } => {
                builder.add_nfa_state_id(nfa_id);
                builder.look_need().insert(look);
            }
            thompson::State::Union { .. } => {
                // Pure epsilon transitions don't need to be tracked
                // as part of the DFA state. Tracking them is actually
                // superfluous; they won't cause any harm other than making
                // determinization slower.
                //
                // Why aren't these needed? Well, in an NFA, epsilon
                // transitions are really just jumping points to other
                // states. So once you hit an epsilon transition, the same
                // set of resulting states always appears. Therefore,
                // putting them in a DFA's set of ordered NFA states is
                // strictly redundant.
                //
                // Look-around states are also epsilon transitions, but
                // they are *conditinal*. So their presence could be
                // discriminatory, and thus, they are tracked above.
                //
                // But wait... why are epsilon states in our `set` in the
                // first place? Why not just leave them out? They're in
                // our `set` because it was generated by computing an
                // epsilon closure, and we want to keep track of all states
                // we visited to avoid re-visiting them. In exchange, we
                // have to do this second iteration over our collected
                // states to finalize our DFA state.
                //
                // Note that this optimization requires that we re-compute
                // the epsilon closure to account for look-ahead in 'next'
                // *only when necessary*. Namely, only when the set of
                // look-around assertions changes and only when those
                // changes are within the set of assertions that are
                // needed in order to step through the closure correctly.
                // Otherwise, if we re-do the epsilon closure needlessly,
                // it could change based on the fact that we are omitting
                // epsilon states here.
            }
            thompson::State::Fail => {
                break;
            }
            thompson::State::Match(_) => {
                // Normally, the NFA match state doesn't actually need to
                // be inside the DFA state. But since we delay matches by
                // one byte, the matching DFA state corresponds to states
                // that transition from the one we're building here. And
                // the way we detect those cases is by looking for an NFA
                // match state. See 'next' for how this is handled.
                builder.add_nfa_state_id(nfa_id);
                if !match_kind.continue_past_first_match() {
                    break;
                }
            }
        }
    }
    // If we know this state contains no look-around assertions, then
    // there's no reason to track which look-around assertions were
    // satisfied when this state was created.
    if builder.look_need().is_empty() {
        builder.look_have().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::Start;

    #[test]
    #[should_panic]
    fn start_fwd_bad_range() {
        Start::from_position_fwd(&[], 0, 1);
    }

    #[test]
    #[should_panic]
    fn start_rev_bad_range() {
        Start::from_position_rev(&[], 0, 1);
    }

    #[test]
    fn start_fwd() {
        let f = Start::from_position_fwd;

        assert_eq!(Start::Text, f(&[], 0, 0));
        assert_eq!(Start::Text, f(b"abc", 0, 3));
        assert_eq!(Start::Text, f(b"\nabc", 0, 3));

        assert_eq!(Start::Line, f(b"\nabc", 1, 3));

        assert_eq!(Start::WordByte, f(b"abc", 1, 3));

        assert_eq!(Start::NonWordByte, f(b" abc", 1, 3));
    }

    #[test]
    fn start_rev() {
        let f = Start::from_position_rev;

        assert_eq!(Start::Text, f(&[], 0, 0));
        assert_eq!(Start::Text, f(b"abc", 0, 3));
        assert_eq!(Start::Text, f(b"abc\n", 0, 4));

        assert_eq!(Start::Line, f(b"abc\nz", 0, 3));

        assert_eq!(Start::WordByte, f(b"abc", 0, 2));

        assert_eq!(Start::NonWordByte, f(b"abc ", 0, 3));
    }
}
