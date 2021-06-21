use core::borrow::Borrow;

use crate::{
    hybrid::{error::Error, Config},
    nfa::thompson,
    util::{
        alphabet::{ByteClasses, ByteSet},
        determinize::{Start, State},
        id::{PatternID, StateID},
        matchtypes::MatchKind,
        sparse_set::SparseSets,
    },
};

#[derive(Clone, Debug)]
pub struct InertDFA<N> {
    nfa: N,
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
        Ok(InertDFA {
            nfa,
            classes,
            quit,
            anchored: config.get_anchored(),
            match_kind: config.get_match_kind(),
            starts_for_each_pattern: config.get_starts_for_each_pattern(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct Cache {
    sparses: SparseSets,
    fsm: CacheFSM,
}

#[derive(Clone, Debug)]
struct CacheFSM {
    trans: Vec<StateID>,
    starts: Vec<StateID>,
    states: Vec<State>,
    states_to_id: StateMap,
    stack: Vec<StateID>,
    scratch_nfa_states: Vec<StateID>,
}

/// A map from states to state identifiers. When using std, we use a standard
/// hashmap, since it's a bit faster for this use case. (Other maps, like
/// one's based on FNV, have not yet been benchmarked.)
///
/// The main purpose of this map is to reuse states where possible. This won't
/// fully minimize the DFA, but it works well in a lot of cases.
#[cfg(feature = "std")]
type StateMap = std::collections::HashMap<State, StateID>;
#[cfg(not(feature = "std"))]
type StateMap = BTreeMap<State, StateID>;

#[derive(Debug)]
pub struct DFA<'i, 'c, N> {
    inert: &'i InertDFA<N>,
    cache: &'c mut Cache,
}

impl<'i, 'c, N: Borrow<thompson::NFA>> DFA<'i, 'c, N> {
    fn start_state_forward(
        &mut self,
        pattern_id: Option<PatternID>,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> StateID {
        let index = Start::from_position_fwd(bytes, start, end);
        todo!()
    }
}
