pub(crate) use self::state::{
    State, StateBuilderEmpty, StateBuilderMatches, StateBuilderNFA,
};

use crate::{
    nfa::thompson::{self, Look, LookSet},
    util::{
        alphabet,
        id::StateID,
        matchtypes::MatchKind,
        sparse_set::{SparseSet, SparseSets},
    },
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

/// Compute the set of all eachable NFA states, including the full epsilon
/// closure, from a DFA state for a single byte of input.
pub(crate) fn next(
    nfa: &thompson::NFA,
    match_kind: MatchKind,
    stack: &mut Vec<StateID>,
    state: &State,
    unit: alphabet::Unit,
    builder: &mut StateBuilderMatches,
    sparses: &mut SparseSets,
) {
    sparses.clear();

    // Put the NFA state IDs into a sparse set in case we need to
    // re-compute their epsilon closure.
    //
    // Doing this state shuffling is technically not necessary unless some
    // kind of look-around is used in the DFA. Some ad hoc experiments
    // suggested that avoiding this didn't lead to much of an improvement,
    // but perhaps more rigorous experimentation should be done. And in
    // particular, avoiding this check requires some light refactoring of
    // the code below.
    state.iter_nfa_state_ids(|nfa_id| {
        sparses.set1.insert(nfa_id);
    });

    // Compute look-ahead assertions originating from the current state.
    // Based on the input unit we're transitioning over, some additional
    // set of assertions may be true. Thus, we re-compute this state's
    // epsilon closure (but only if necessary).
    if !state.look_need().is_empty() {
        // Add look-ahead assertions that are now true based on the current
        // input unit.
        let mut look_have = state.look_have().clone();
        match unit.as_u8() {
            Some(b'\n') => {
                look_have.insert(Look::EndLine);
            }
            Some(_) => {}
            None => {
                look_have.insert(Look::EndText);
                look_have.insert(Look::EndLine);
            }
        }
        if state.is_from_word() == unit.is_word_byte() {
            look_have.insert(Look::WordBoundaryUnicodeNegate);
            look_have.insert(Look::WordBoundaryAsciiNegate);
        } else {
            look_have.insert(Look::WordBoundaryUnicode);
            look_have.insert(Look::WordBoundaryAscii);
        }
        // If we have new assertions satisfied that are among the set of
        // assertions that exist in this state (that is, just because
        // we added an EndLine assertion above doesn't mean there is an
        // EndLine conditional epsilon transition in this state), then we
        // re-compute this state's epsilon closure using the updated set of
        // assertions.
        if !look_have
            .subtract(state.look_have())
            .intersect(state.look_need())
            .is_empty()
        {
            for nfa_id in &sparses.set1 {
                epsilon_closure(
                    nfa,
                    nfa_id,
                    look_have,
                    stack,
                    &mut sparses.set2,
                );
            }
            sparses.swap();
            sparses.set2.clear();
        }
    }

    // Compute look-behind assertions that are true while entering the new
    // state we create below.

    // We only set the word byte if there's a word boundary look-around
    // anywhere in this regex. Otherwise, there's no point in bloating
    // the number of states if we don't have one.
    if nfa.has_word_boundary() && unit.is_word_byte() {
        builder.set_is_from_word();
    }
    // Similarly for the start-line look-around.
    if nfa.has_any_anchor() {
        if unit.as_u8().map_or(false, |b| b == b'\n') {
            // Why only handle StartLine here and not StartText? That's
            // because StartText can only impact the starting state, which
            // is speical cased in 'add_one_start'.
            builder.look_have().insert(Look::StartLine);
        }
    }
    for nfa_id in &sparses.set1 {
        match *nfa.state(nfa_id) {
            thompson::State::Union { .. }
            | thompson::State::Fail
            | thompson::State::Look { .. } => {}
            thompson::State::Match(pid) => {
                // Notice here that we are calling the NEW state a match
                // state if the OLD state we are transitioning from
                // contains an NFA match state. This is precisely how we
                // delay all matches by one byte and also what therefore
                // guarantees that starting states cannot be match states.
                //
                // If we didn't delay matches by one byte, then whether
                // a DFA is a matching state or not would be determined
                // by whether one of its own constituent NFA states
                // was a match state. (And that would be done in
                // 'add_nfa_states'.)
                builder.set_is_match();
                if nfa.match_len() > 1 {
                    builder.add_match_pattern_id(pid);
                }
                if !match_kind.continue_past_first_match() {
                    break;
                }
            }
            thompson::State::Range { range: ref r } => {
                if let Some(b) = unit.as_u8() {
                    if r.start <= b && b <= r.end {
                        epsilon_closure(
                            nfa,
                            r.next,
                            *builder.look_have(),
                            stack,
                            &mut sparses.set2,
                        );
                    }
                }
            }
            thompson::State::Sparse { ref ranges } => {
                let b = match unit.as_u8() {
                    None => continue,
                    Some(b) => b,
                };
                for r in ranges.iter() {
                    if r.start > b {
                        break;
                    } else if r.start <= b && b <= r.end {
                        epsilon_closure(
                            nfa,
                            r.next,
                            *builder.look_have(),
                            stack,
                            &mut sparses.set2,
                        );
                        break;
                    }
                }
            }
        }
    }
}

/// Compute the epsilon closure for the given NFA state. The epsilon
/// closure consists of all NFA state IDs, including `start`, that can be
/// reached from `start` without consuming any input. These state IDs are
/// written to `set` in the order they are visited, but only if they are
/// not already in `set`.
///
/// `look_have` consists of the satisfied assertions at the current
/// position. For conditional look-around epsilon transitions, these are
/// only followed if they are satisfied by `look_have`.
pub(crate) fn epsilon_closure(
    nfa: &thompson::NFA,
    start_nfa_id: StateID,
    look_have: LookSet,
    stack: &mut Vec<StateID>,
    set: &mut SparseSet,
) {
    debug_assert!(stack.is_empty());
    if !nfa.state(start_nfa_id).is_epsilon() {
        set.insert(start_nfa_id);
        return;
    }

    stack.push(start_nfa_id);
    while let Some(mut id) = stack.pop() {
        loop {
            if !set.insert(id) {
                break;
            }
            match *nfa.state(id) {
                thompson::State::Range { .. }
                | thompson::State::Sparse { .. }
                | thompson::State::Fail
                | thompson::State::Match(_) => break,
                thompson::State::Look { look, next } => {
                    if !look_have.contains(look) {
                        break;
                    }
                    id = next;
                }
                thompson::State::Union { ref alternates } => {
                    id = match alternates.get(0) {
                        None => break,
                        Some(&id) => id,
                    };
                    stack.extend(alternates[1..].iter().rev());
                }
            }
        }
    }
}

/// Add the NFA state IDs in the given `set` to the given DFA builder state.
/// The order in which states are added corresponds to the order in which they
/// were added to `set`.
///
/// The DFA builder state given should already have its complete set of match
/// pattern IDs added (if any) and any look-behind assertions (StartLine,
/// StartText and whether this state is being generated for a transition over a
/// word byte when applicable) that are true immediately prior to transitioning
/// into this state (via `builder.look_have()`). The match pattern IDs should
/// correspond to matches that occured on the previous transition, since all
/// matches are delayed by one byte. The things that should _not_ be set are
/// look-ahead assertions (EndLine, EndText and whether the next byte is a
/// word byte or not). The builder state should also not have anything in
/// `look_need` set, as this routine will compute that for you.
///
/// The given NFA should be able to resolve all identifiers in `set` to a
/// particular NFA state. Also, the given match kind should correspond to
/// the match semantics implemented by the DFA. Generally speaking, for
/// leftmost-first match semantics, states that appear after the first match
/// state in `set` will not be added to the given builder since they are
/// impossible to visit.
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
