#![allow(warnings)]

use core::convert::TryFrom;

use alloc::vec;

use crate::{
    dfa::error::Error,
    nfa::thompson::{State, NFA},
    util::{
        alphabet::ByteClasses, primitives::StateID, sparse_set::SparseSet,
    },
};

pub struct OnePass {
    table: Vec<u64>,
    classes: ByteClasses,
    stride2: usize,
}

impl OnePass {
    fn build(&mut self, nfa: &NFA) -> Result<(), Error> {
        let mut uncompiled: Vec<(StateID, StateID)> =
            vec![(nfa.start_anchored(), self.add_empty_state()?)];
        let mut seen = SparseSet::new(nfa.states().len());
        let mut stack: Vec<(StateID, u32)> = vec![];
        // Should 'uncompiled' be NFA state IDs? I think they have to be. But
        // then how do we get their corresponding DFA state? I think we need a
        // mapping. I think the issue is that we don't necessarily build a DFA
        // state for each node we want to visit? Hmmm, no, we do. I think we
        // might just want both? So we'll need to start by allocating an empty
        // state for the start state I guess.
        while let Some((nfa_id, dfa_id)) = uncompiled.pop() {
            // TODO: Set DFA state to have "impossible" everywhere?
            let mut matched = false;
            seen.clear();
            stack.push((nfa_id, 0));
            while let Some((id, cond)) = stack.pop() {
                match nfa.state(id) {
                    State::ByteRange { ref trans } => todo!(),
                    _ => todo!(),
                }
            }
        }
        Ok(())
    }

    fn alphabet_len(&self) -> usize {
        self.classes.alphabet_len()
    }

    fn stride(&self) -> usize {
        1 << self.stride2
    }

    fn state_mut(&mut self, id: StateID) -> StateMut<'_> {
        let offset = id.as_usize();
        let len = self.alphabet_len();
        let raw = &mut self.table[offset - 1..offset + len];
        StateMut { id, raw }
    }

    fn add_empty_state(&mut self) -> Result<StateID, Error> {
        let next = match self.table.len().checked_add(1) {
            None => return Err(Error::too_many_states()),
            Some(next) => next,
        };
        let id = StateID::new(next).map_err(|_| Error::too_many_states())?;
        self.table.extend(core::iter::repeat(0).take(self.stride()));
        Ok(id)
    }
}

struct StateMut<'a> {
    id: StateID,
    raw: &'a mut [u64],
}

impl<'a> StateMut<'a> {
    fn match_info(&mut self) -> &mut u64 {
        &mut self.raw[0]
    }

    fn transitions(&mut self) -> &mut [u64] {
        &mut self.raw[1..]
    }
}

/// Computes the stride as a power of 2 for a one-pass DFA. The special sauce
/// here is that every state has 1+alphabet_len entries (each entry is a
/// u64), where the extra entry comes from match info. Like which look-around
/// assertions need to hold and which patterns have matched.
fn stride2(alphabet_len: usize) -> usize {
    let zeros = (1 + alphabet_len).next_power_of_two().trailing_zeros();
    usize::try_from(zeros).unwrap()
}
