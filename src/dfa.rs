#![allow(warnings)]

use state_id::StateID;

// BREADCRUMBS:
//
// So it looks like we probably want DenseDFA to be this:
//
//   enum DenseDFA {
//     Standard(dense::Standard),
//     Premultiplied(dense::Premultiplied),
//     ByteClass(dense::ByteClass),
//     PremultipliedByteClass(dense::PremultipliedByteClass),
//   }
//
// where each of the dense::* types wrap DenseDFARepr, which is basically what
// DenseDFA is now. We can then override the is_match/... routines on DenseDFA
// to do case analysis once and dispatch to its inner DFA.
//
// Determinization and minimization then operate on DenseDFARepr.
//
// Do we still need DenseDFAKind? It looks like we do, so maybe determinization
// and minimization really should just use DenseDFA.
//
// Now, what about DenseDFARef? Does it really need to copy the structure of
// DenseDFA, and therefore need dense::StandardRef and so on?
//
// OK, it looks like we need to add a second type parameter to DenseDFA and
// SparseDFA. Specifically, DenseDFA should use T for its transition table,
// where T: AsRef<[StateID]>. That way, the same types can be used for both
// Vec<S> and &[S]. This has to be threaded all the way down to DenseDFARepr.
//
// The upshot here is that Regex stays the course with one type parameter:
// an implementation of the DFA trait. The ID associated type avoids needing
// to fuss with state IDs with the Regex type directly.

pub trait DFA {
    type ID: StateID;

    fn start_state(&self) -> Self::ID;

    fn is_match_state(&self, id: Self::ID) -> bool;

    fn is_possible_match_state(&self, id: Self::ID) -> bool;

    fn is_dead_state(&self, id: Self::ID) -> bool;

    fn next_state(&self, current: Self::ID, input: u8) -> Self::ID;

    unsafe fn next_state_unchecked(
        &self,
        current: Self::ID,
        input: u8,
    ) -> Self::ID;

    fn is_match(&self, bytes: &[u8]) -> bool {
        let mut state = self.start_state();
        if self.is_possible_match_state(state) {
            return self.is_match_state(state);
        }
        for &b in bytes.iter() {
            state = unsafe { self.next_state_unchecked(state, b) };
            if self.is_possible_match_state(state) {
                return self.is_match_state(state);
            }
        }
        false
    }

    fn shortest_match(&self, bytes: &[u8]) -> Option<usize> {
        let mut state = self.start_state();
        if self.is_possible_match_state(state) {
            return if self.is_dead_state(state) { None } else { Some(0) };
        }
        for (i, &b) in bytes.iter().enumerate() {
            state = unsafe { self.next_state_unchecked(state, b) };
            if self.is_possible_match_state(state) {
                return
                    if self.is_dead_state(state) {
                        None
                    } else {
                        Some(i + 1)
                    };
            }
        }
        None
    }

    fn find(&self, bytes: &[u8]) -> Option<usize> {
        let mut state = self.start_state();
        let mut last_match =
            if self.is_dead_state(state) {
                return None;
            } else if self.is_match_state(state) {
                Some(0)
            } else {
                None
            };
        for (i, &b) in bytes.iter().enumerate() {
            state = unsafe { self.next_state_unchecked(state, b) };
            if self.is_possible_match_state(state) {
                if self.is_dead_state(state) {
                    return last_match;
                }
                last_match = Some(i + 1);
            }
        }
        last_match
    }

    fn rfind(&self, bytes: &[u8]) -> Option<usize> {
        let mut state = self.start_state();
        let mut last_match =
            if self.is_dead_state(state) {
                return None;
            } else if self.is_match_state(state) {
                Some(0)
            } else {
                None
            };
        for (i, &b) in bytes.iter().enumerate().rev() {
            state = unsafe { self.next_state_unchecked(state, b) };
            if self.is_possible_match_state(state) {
                if self.is_dead_state(state) {
                    return last_match;
                }
                last_match = Some(i);
            }
        }
        last_match
    }
}
