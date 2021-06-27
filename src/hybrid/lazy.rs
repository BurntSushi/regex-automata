use core::{borrow::Borrow, iter};

use crate::{
    hybrid::{error::Error, id::LazyStateID, Config},
    nfa::thompson,
    util::{
        alphabet::{self, ByteClasses, ByteSet},
        determinize::{
            self, Start, State, StateBuilderEmpty, StateBuilderNFA,
        },
        id::{PatternID, StateID as NFAStateID},
        matchtypes::{MatchError, MatchKind},
        sparse_set::SparseSets,
    },
};

#[derive(Clone, Debug)]
pub struct InertDFA<N> {
    nfa: N,
    stride2: usize,
    classes: ByteClasses,
    quit: ByteSet,
    anchored: bool,
    match_kind: MatchKind,
    starts_for_each_pattern: bool,
}

impl<N: Borrow<thompson::NFA>> InertDFA<N> {
    pub(crate) fn new(config: &Config, nfa: N) -> Result<InertDFA<N>, Error> {
        let mut quit = config.quit.unwrap_or(ByteSet::empty());
        if nfa.borrow().has_word_boundary_unicode() {
            if config.get_unicode_word_boundary() {
                for b in 0x80..=0xFF {
                    quit.add(b);
                }
            } else {
                // If heuristic support for Unicode word boundaries wasn't
                // enabled, then we can still check if our quit set is correct.
                // If the caller set their quit bytes in a way that causes the
                // DFA to quit on at least all non-ASCII bytes, then that's all
                // we need for heuristic support to work.
                if !quit.contains_range(0x80, 0xFF) {
                    return Err(Error::unsupported_dfa_word_boundary_unicode());
                }
            }
        }
        let classes = if !config.get_byte_classes() {
            // The lazy DFA will always use the equivalence class map, but
            // enabling this option is useful for debugging. Namely, this will
            // cause all transitions to be defined over their actual bytes
            // instead of an opaque equivalence class identifier. The former is
            // much easier to grok as a human.
            ByteClasses::singletons()
        } else {
            let mut set = nfa.borrow().byte_class_set().clone();
            // It is important to distinguish any "quit" bytes from all other
            // bytes. Otherwise, a non-quit byte may end up in the same class
            // as a quit byte, and thus cause the DFA stop when it shouldn't.
            if !quit.is_empty() {
                set.add_set(&quit);
            }
            set.byte_classes()
        };
        let stride2 = classes.stride2();
        Ok(InertDFA {
            nfa,
            stride2,
            classes,
            quit,
            anchored: config.get_anchored(),
            match_kind: config.get_match_kind(),
            starts_for_each_pattern: config.get_starts_for_each_pattern(),
        })
    }

    /// Returns the number of patterns in this DFA. (It is possible for this
    /// to be zero, for a DFA that never matches anything.)
    pub fn patterns(&self) -> usize {
        self.nfa.borrow().match_len()
    }
}

#[derive(Clone, Debug)]
pub struct Cache {
    sparses: SparseSets,
    fsm: CacheFSM,
}

#[derive(Clone, Debug)]
struct CacheFSM {
    trans: Vec<LazyStateID>,
    starts: Vec<LazyStateID>,
    states: Vec<State>,
    states_to_id: StateMap,
    stack: Vec<NFAStateID>,
    scratch_state_builder: StateBuilderEmpty,
}

impl Cache {
    pub fn new<N: Borrow<thompson::NFA>>(dfa: &InertDFA<N>) -> Cache {
        let mut starts_len = Start::count();
        if dfa.starts_for_each_pattern {
            starts_len += Start::count() * dfa.patterns();
        }
        Cache {
            sparses: SparseSets::new(dfa.nfa.borrow().len()),
            fsm: CacheFSM {
                trans: vec![],
                starts: vec![LazyStateID::SENTINEL_UNKNOWN; starts_len],
                states: vec![],
                states_to_id: StateMap::new(),
                stack: vec![],
                scratch_state_builder: StateBuilderEmpty::new(),
            },
        }
    }
}

/// A map from states to state identifiers. When using std, we use a standard
/// hashmap, since it's a bit faster for this use case. (Other maps, like
/// one's based on FNV, have not yet been benchmarked.)
///
/// The main purpose of this map is to reuse states where possible. This won't
/// fully minimize the DFA, but it works well in a lot of cases.
#[cfg(feature = "std")]
type StateMap = std::collections::HashMap<State, LazyStateID>;
#[cfg(not(feature = "std"))]
type StateMap = BTreeMap<State, LazyStateID>;

#[derive(Debug)]
pub struct DFA<'i, 'c, N> {
    inert: &'i InertDFA<N>,
    cache: &'c mut Cache,
}

impl<'i, 'c, N: Borrow<thompson::NFA>> DFA<'i, 'c, N> {
    pub fn start_state_forward(
        &mut self,
        pattern_id: Option<PatternID>,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> LazyStateID {
        let start_type = Start::from_position_fwd(bytes, start, end);
        let sid = self.get_cached_start(pattern_id, start_type);
        if !sid.is_unknown() {
            return sid;
        }
        // TODO: Get rid of this unwrap.
        self.cache_start_group(pattern_id, start_type).unwrap()
    }

    pub fn start_state_reverse(
        &mut self,
        pattern_id: Option<PatternID>,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> LazyStateID {
        let start_type = Start::from_position_rev(bytes, start, end);
        let sid = self.get_cached_start(pattern_id, start_type);
        if !sid.is_unknown() {
            return sid;
        }
        // TODO: Get rid of this unwrap.
        self.cache_start_group(pattern_id, start_type).unwrap()
    }

    fn get_cached_start(
        &self,
        pattern_id: Option<PatternID>,
        start: Start,
    ) -> LazyStateID {
        let start_index = start.as_usize();
        let index = match pattern_id {
            None => start_index,
            Some(pid) => {
                let pid = pid.as_usize();
                assert!(
                    pid < self.patterns(),
                    "invalid pattern ID: {:?}",
                    pid
                );
                Start::count() + (Start::count() * pid) + start_index
            }
        };
        self.cache.fsm.starts[index]
    }

    fn cache_start_group(
        &mut self,
        pattern_id: Option<PatternID>,
        start: Start,
    ) -> Result<LazyStateID, MatchError> {
        let nfa = self.inert.nfa.borrow();
        let nfa_start_id = match pattern_id {
            Some(pid) => nfa.start_pattern(pid),
            None if self.inert.anchored => nfa.start_anchored(),
            None => nfa.start_unanchored(),
        };

        let id = self.cache_start_one(nfa_start_id, start)?;
        self.set_start_state(pattern_id, start, id);
        Ok(id)
    }

    fn cache_start_one(
        &mut self,
        nfa_start_id: NFAStateID,
        start: Start,
    ) -> Result<LazyStateID, MatchError> {
        let mut builder_matches = self.get_state_builder().into_matches();
        start.set_state(&mut builder_matches);
        determinize::epsilon_closure(
            self.inert.nfa.borrow(),
            nfa_start_id,
            *builder_matches.look_have(),
            &mut self.cache.fsm.stack,
            &mut self.cache.sparses.set1,
        );
        let mut builder = builder_matches.into_nfa();
        determinize::add_nfa_states(
            self.inert.nfa.borrow(),
            &self.cache.sparses.set1,
            &mut builder,
        );
        self.maybe_add_state(builder).map(|(sid, _)| sid.as_start())
    }

    fn maybe_add_state(
        &mut self,
        builder: StateBuilderNFA,
    ) -> Result<(LazyStateID, bool), MatchError> {
        if let Some(&cached_id) =
            self.cache.fsm.states_to_id.get(builder.as_bytes())
        {
            // Since we have a cached state, put the constructed state's
            // memory back into our scratch space, so that it can be reused.
            self.put_state_builder(builder);
            return Ok((cached_id, false));
        }
        self.add_state(builder).map(|sid| (sid, true))
    }

    fn add_state(
        &mut self,
        builder: StateBuilderNFA,
    ) -> Result<LazyStateID, MatchError> {
        let id = self.add_empty_state()?;
        if !self.inert.quit.is_empty() {
            for b in self.inert.quit.iter() {
                self.set_transition(
                    id,
                    alphabet::Unit::u8(b),
                    LazyStateID::SENTINEL_QUIT,
                );
            }
        }
        let state = builder.to_state();
        self.cache.fsm.states.push(state.clone());
        self.cache.fsm.states_to_id.insert(state, id);
        self.put_state_builder(builder);
        Ok(id)
    }

    fn add_empty_state(&mut self) -> Result<LazyStateID, MatchError> {
        let next = self.cache.fsm.trans.len();
        // TODO: Get rid of this unwrap.
        let sid = LazyStateID::new(next).unwrap();
        self.cache.fsm.trans.extend(
            iter::repeat(LazyStateID::SENTINEL_UNKNOWN).take(self.stride()),
        );
        Ok(sid)
    }

    fn set_transition(
        &mut self,
        from: LazyStateID,
        unit: alphabet::Unit,
        to: LazyStateID,
    ) {
        assert!(!from.is_unknown() && !from.is_dead() && !from.is_quit());
        assert!(self.is_valid(from));
        assert!(self.is_valid(to));
        let offset =
            from.as_usize_unmasked() + self.inert.classes.get_by_unit(unit);
        self.cache.fsm.trans[offset] = to;
    }

    fn set_start_state(
        &mut self,
        pattern_id: Option<PatternID>,
        start: Start,
        id: LazyStateID,
    ) {
        assert!(self.is_valid(id));
        let start_index = start.as_usize();
        let index = match pattern_id {
            None => start_index,
            Some(pid) => {
                let pid = pid.as_usize();
                Start::count() + (Start::count() * pid) + start_index
            }
        };
        self.cache.fsm.starts[index] = id;
    }

    fn is_valid(&self, id: LazyStateID) -> bool {
        let id = id.as_usize_unmasked();
        id < self.cache.fsm.trans.len() && id % self.stride() == 0
    }

    /// Returns the number of patterns in this DFA. (It is possible for this
    /// to be zero, for a DFA that never matches anything.)
    fn patterns(&self) -> usize {
        self.inert.patterns()
    }

    /// Returns the stride, as a base-2 exponent, required for these
    /// equivalence classes.
    ///
    /// The stride is always the smallest power of 2 that is greater than or
    /// equal to the alphabet length. This is done so that converting between
    /// state IDs and indices can be done with shifts alone, which is much
    /// faster than integer division.
    fn stride2(&self) -> usize {
        self.inert.stride2
    }

    /// Returns the total stride for every state in this lazy DFA. This
    /// corresponds to the total number of transitions used by each state in
    /// this DFA's transition table.
    fn stride(&self) -> usize {
        1 << self.stride2()
    }

    /// Returns a state builder from this DFA that might have existing
    /// capacity. This helps avoid allocs in cases where a state is built that
    /// turns out to already be cached.
    ///
    /// Callers must put the state builder back with 'put_state_builder',
    /// otherwise the allocation reuse won't work.
    fn get_state_builder(&mut self) -> StateBuilderEmpty {
        core::mem::replace(
            &mut self.cache.fsm.scratch_state_builder,
            StateBuilderEmpty::new(),
        )
    }

    /// Puts the given state builder back into this DFA for reuse.
    ///
    /// Note that building a 'State' from a builder always creates a new alloc,
    /// so callers should always put the builder back.
    fn put_state_builder(&mut self, builder: StateBuilderNFA) {
        let _ = core::mem::replace(
            &mut self.cache.fsm.scratch_state_builder,
            builder.clear(),
        );
    }
}
