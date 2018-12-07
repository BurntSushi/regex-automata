use std::collections::{HashMap, HashSet};
use std::mem;
use std::rc::Rc;

use dfa::{self, DFA};
use nfa::{self, NFA};
use sparse::SparseSet;

pub struct Determinizer<'a> {
    /// The NFA we're converting into a DFA.
    nfa: &'a NFA,
    /// The DFA we're building.
    dfa: DFA,
    /// Each DFA state being built is defined as an *ordered* set of NFA
    /// states.
    ///
    /// This is never empty. The first state is always a dummy state such that
    /// dfa::StateID == 0 corresponds to a dead state.
    builder_states: Vec<Rc<DeterminizerState>>,
    /// A cache of DFA states that already exist and can be easily looked up
    /// via ordered sets of NFA states.
    cache: HashMap<Rc<DeterminizerState>, dfa::StateID>,
    /// A stack of NFA states to visit, for depth first visiting.
    stack: Vec<nfa::StateID>,
    /// Scratch space for storing an ordered sequence of NFA states, for
    /// amortizing allocation.
    scratch_nfa_states: Vec<nfa::StateID>,
}

#[derive(Debug, Eq, Hash, PartialEq)]
struct DeterminizerState {
    is_match: bool,
    nfa_states: Vec<nfa::StateID>,
}

impl<'a> Determinizer<'a> {
    pub fn new(nfa: &'a NFA) -> Determinizer<'a> {
        let dead = Rc::new(DeterminizerState::dead());
        let mut cache = HashMap::new();
        cache.insert(dead.clone(), dfa::DEAD);

        Determinizer {
            nfa: nfa,
            dfa: DFA::empty(),
            builder_states: vec![dead],
            cache: cache,
            stack: vec![],
            scratch_nfa_states: vec![],
        }
    }

    pub fn build(mut self) -> DFA {
        let mut sparse = self.new_sparse_set();
        let mut uncompiled = vec![self.add_start(&mut sparse)];
        let mut queued: HashSet<dfa::StateID> = HashSet::new();
        while let Some(dfa_id) = uncompiled.pop() {
            for b in 0..=255 {
                let next_dfa_id = self.cached_state(dfa_id, b, &mut sparse);
                self.dfa.set_transition(dfa_id, b, next_dfa_id);
                if !queued.contains(&next_dfa_id) {
                    uncompiled.push(next_dfa_id);
                    queued.insert(next_dfa_id);
                }
            }
        }

        let is_match: Vec<bool> = self
            .builder_states
            .iter()
            .map(|s| s.is_match)
            .collect();
        self.dfa.shuffle_match_states(&is_match);
        self.dfa
    }

    fn cached_state(
        &mut self,
        dfa_id: dfa::StateID,
        b: u8,
        sparse: &mut SparseSet,
    ) -> dfa::StateID {
        sparse.clear();
        self.next(dfa_id, b, sparse);
        let state = self.new_state(sparse);
        if let Some(&cached_id) = self.cache.get(&state) {
            mem::replace(&mut self.scratch_nfa_states, state.nfa_states);
            return cached_id;
        }
        self.add_state(state)
    }

    fn next(
        &mut self,
        dfa_id: dfa::StateID,
        b: u8,
        next_nfa_states: &mut SparseSet,
    ) {
        next_nfa_states.clear();
        for i in 0..self.builder_states[dfa_id].nfa_states.len() {
            let nfa_id = self.builder_states[dfa_id].nfa_states[i];
            match self.nfa.states.borrow()[nfa_id] {
                nfa::State::Range { start, end, next } => {
                    if start <= b && b <= end {
                        self.epsilon_closure(next, next_nfa_states);
                    }
                }
                | nfa::State::Empty { .. }
                | nfa::State::Union { .. }
                | nfa::State::Match => {}
            }
        }
    }

    fn epsilon_closure(&mut self, start: nfa::StateID, set: &mut SparseSet) {
        if !self.nfa.states.borrow()[start].is_epsilon() {
            set.insert(start);
            return;
        }

        self.stack.push(start);
        while let Some(mut id) = self.stack.pop() {
            loop {
                if set.contains(id) {
                    break;
                }
                set.insert(id);
                match self.nfa.states.borrow()[id] {
                    nfa::State::Empty { next } => {
                        id = next;
                    }
                    nfa::State::Union { ref alternates, .. } => {
                        id = match alternates.get(0) {
                            None => break,
                            Some(&id) => id,
                        };
                        self.stack.extend(alternates[1..].iter().rev());
                    }
                    nfa::State::Range { .. } | nfa::State::Match => break,
                }
            }
        }
    }

    fn add_start(&mut self, sparse: &mut SparseSet) -> dfa::StateID {
        self.epsilon_closure(0, sparse);
        let state = self.new_state(&sparse);
        let id = self.add_state(state);
        self.dfa.set_start_state(id);
        id
    }

    fn add_state(&mut self, state: DeterminizerState) -> dfa::StateID {
        let id = self.dfa.add_empty_state();
        let rstate = Rc::new(state);
        self.builder_states.push(rstate.clone());
        self.cache.insert(rstate, id);
        id
    }

    fn new_state(&mut self, set: &SparseSet) -> DeterminizerState {
        let mut state = DeterminizerState {
            is_match: false,
            nfa_states: mem::replace(&mut self.scratch_nfa_states, vec![]),
        };
        state.nfa_states.clear();

        for &id in set {
            match self.nfa.states.borrow()[id] {
                nfa::State::Range { .. } => {
                    state.nfa_states.push(id);
                }
                nfa::State::Match => {
                    state.is_match = true;
                    break;
                }
                nfa::State::Empty { .. } | nfa::State::Union { .. } => {}
            }
        }
        state
    }

    fn new_sparse_set(&self) -> SparseSet {
        SparseSet::new(self.nfa.states.borrow().len())
    }
}

impl DeterminizerState {
    fn dead() -> DeterminizerState {
        DeterminizerState { nfa_states: vec![], is_match: false }
    }
}
