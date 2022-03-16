use core::{convert::TryFrom, mem};

use alloc::{sync::Arc, vec::Vec};

use crate::{
    nfa::thompson::{
        error::Error,
        nfa::{self, Look, SparseTransitions, Transition, NFA},
    },
    util::id::{IteratorIDExt, PatternID, StateID},
};

/// An intermediate NFA state used during construction.
///
/// During construction of an NFA, it is often convenient to work with states
/// that are amenable to mutation and other carry more information than we
/// otherwise need once an NFA has been built. This type represents those
/// needs.
///
/// Once construction is finished, the builder will convert these states to
/// [`nfa::thompson::State`](crate::nfa::thompson::State). This conversion not
/// only results in a simpler representation, but in some cases, entire classes
/// of states are completely removed (such as [`State::Empty`]).
#[derive(Clone, Debug, Eq, PartialEq)]
enum State {
    /// An empty state whose only purpose is to forward the automaton to
    /// another state via an unconditional epsilon transition.
    ///
    /// Unconditional epsilon transitions are quite useful during the
    /// construction of an NFA, as they permit the insertion of no-op
    /// placeholders that make it easier to compose NFA sub-graphs. When
    /// the Thompson NFA builder produces a final NFA, all unconditional
    /// epsilon transitions are removed, and state identifiers are remapped
    /// accordingly.
    Empty {
        /// The next state that this state should transition to.
        next: StateID,
    },
    /// A state that only transitions to another state if the current input
    /// byte is in a particular range of bytes.
    Range { range: Transition },
    /// A state with possibly many transitions, represented in a sparse
    /// fashion. Transitions must be ordered lexicographically by input range
    /// and be non-overlapping. As such, this may only be used when every
    /// transition has equal priority. (In practice, this is only used for
    /// encoding large UTF-8 automata.) In contrast, a `Union` state has each
    /// alternate in order of priority. Priority is used to implement greedy
    /// matching and also alternations themselves, e.g., `abc|a` where `abc`
    /// has priority over `a`.
    ///
    /// To clarify, it is possible to remove `Sparse` and represent all things
    /// that `Sparse` is used for via `Union`. But this creates a more bloated
    /// NFA with more epsilon transitions than is necessary in the special case
    /// of character classes.
    Sparse { ranges: Vec<Transition> },
    /// A conditional epsilon transition satisfied via some sort of
    /// look-around.
    Look { look: Look, next: StateID },
    /// An empty state that records the start of a capture location. This is an
    /// unconditional epsilon transition like `Empty`, except it can be used to
    /// record position information for a captue group when using the NFA for
    /// search.
    CaptureStart {
        /// The ID of the pattern that this capture was defined.
        pattern_id: PatternID,
        /// The capture group index that this capture state corresponds to.
        /// The capture group index is always relative to its corresponding
        /// pattern. Therefore, in the presence of multiple patterns, both the
        /// pattern ID and the capture group index are required to uniquely
        /// identify a capturing group.
        capture_index: usize,
        /// The next state that this state should transition to.
        next: StateID,
    },
    /// An empty state that records the end of a capture location. This is an
    /// unconditional epsilon transition like `Empty`, except it can be used to
    /// record position information for a captue group when using the NFA for
    /// search.
    CaptureEnd {
        /// The ID of the pattern that this capture was defined.
        pattern_id: PatternID,
        /// The capture group index that this capture state corresponds to.
        /// The capture group index is always relative to its corresponding
        /// pattern. Therefore, in the presence of multiple patterns, both the
        /// pattern ID and the capture group index are required to uniquely
        /// identify a capturing group.
        capture_index: usize,
        /// The next state that this state should transition to.
        next: StateID,
    },
    /// An alternation such that there exists an epsilon transition to all
    /// states in `alternates`, where matches found via earlier transitions
    /// are preferred over later transitions.
    Union { alternates: Vec<StateID> },
    /// An alternation such that there exists an epsilon transition to all
    /// states in `alternates`, where matches found via later transitions are
    /// preferred over earlier transitions.
    ///
    /// This "reverse" state exists for convenience during compilation that
    /// permits easy construction of non-greedy combinations of NFA states. At
    /// the end of compilation, Union and UnionReverse states are merged into
    /// one Union type of state, where the latter has its epsilon transitions
    /// reversed to reflect the priority inversion.
    ///
    /// The "convenience" here arises from the fact that as new states are
    /// added to the list of `alternates`, we would like that add operation
    /// to be amortized constant time. But if we used a `Union`, we'd need to
    /// prepend the state, which takes O(n) time. There are other approaches we
    /// could use to solve this, but this seems simple enough.
    UnionReverse { alternates: Vec<StateID> },
    /// A state that cannot be transitioned out of. This is useful for cases
    /// where you want to prevent matching from occurring. For example, if your
    /// regex parser permits empty character classes, then one could choose a
    /// `Fail` state to represent it.
    Fail,
    /// A match state. There is at most one such occurrence of this state in
    /// an NFA for each pattern compiled into the NFA. At time of writing, a
    /// match state is always produced for every pattern given, but in theory,
    /// if a pattern can never lead to a match, then the match state could be
    /// omitted.
    ///
    /// `pattern_id` refers to the ID of the pattern itself, which corresponds
    /// to the pattern's index (starting at 0).
    Match { pattern_id: PatternID },
}

impl State {
    fn memory_usage(&self) -> usize {
        match *self {
            State::Empty { .. }
            | State::Range { .. }
            | State::Look { .. }
            | State::CaptureStart { .. }
            | State::CaptureEnd { .. }
            | State::Fail
            | State::Match { .. } => 0,
            State::Sparse { ref ranges } => {
                ranges.len() * mem::size_of::<Transition>()
            }
            State::Union { ref alternates } => {
                alternates.len() * mem::size_of::<StateID>()
            }
            State::UnionReverse { ref alternates } => {
                alternates.len() * mem::size_of::<StateID>()
            }
        }
    }
}

/// TODO
#[derive(Clone, Debug, Default)]
pub struct Builder {
    /// The ID of the pattern that we're currently building.
    ///
    /// Callers are required to set (and unset) this by calling
    /// {start,finish}_pattern. Otherwise, most methods will panic.
    pattern_id: Option<PatternID>,
    /// A sequence of intermediate NFA states. Once a state is added to this
    /// sequence, it is assigned a state ID equivalent to its index. Once a
    /// state is added, it is still expected to be mutated, e.g., to set its
    /// transition to a state that didn't exist at the time it was added.
    states: Vec<State>,
    /// The starting states for each individual pattern. Starting at any
    /// of these states will result in only an anchored search for the
    /// corresponding pattern. The vec is indexed by pattern ID. When the NFA
    /// contains a single regex, then `start_pattern[0]` and `start_anchored`
    /// are always equivalent.
    start_pattern: Vec<StateID>,
    /// A map from pattern ID to capture group index to name. (If no name
    /// exists, then a None entry is present. Thus, all capturing groups are
    /// present in this mapping.)
    ///
    /// The outer vec is indexed by pattern ID, while the inner vec is indexed
    /// by capture index offset for the corresponding pattern.
    ///
    /// The first capture group for each pattern is always unnamed and is thus
    /// always None.
    captures: Vec<Vec<Option<Arc<str>>>>,
    /// The total number of slots required to represent capturing groups in the
    /// NFA. This builder doesn't use this for anything other than validating
    /// that we don't have any capture indices that are too big. (It's likely
    /// that such a thing is only possible on 16-bit systems.)
    slots: usize,
    /// The combined memory used by each of the 'State's in 'states'. This
    /// only includes heap usage by each state, and not the size of the state
    /// itself. In other words, this tracks heap memory used that isn't
    /// captured via `size_of::<State>() * states.len()`.
    memory_states: usize,
    /// A size limit to respect when building an NFA. If the total heap memory
    /// of the intermediate NFA states exceeds (or would exceed) this amount,
    /// then an error is returned.
    size_limit: Option<usize>,
}

// BREADCRUMBS: I think we're ready to rewrite the compiler in terms of this
// new builder. One little annoying hiccup is that the compiler really wants
// to use interior mutability, which means putting this builder in a RefCell.
// But we want to use this builder everywhere, which means lots of annoying
// `self.builder.borrow_mut()` calls. We could define wrapper methods for each
// routine on Builder, but that's *also* annoying. Meh.

impl Builder {
    pub fn new() -> Builder {
        Builder::default()
    }

    pub fn clear(&mut self) {
        self.pattern_id = None;
        self.states.clear();
        self.start_pattern.clear();
        self.captures.clear();
        self.slots = 0;
        self.memory_states = 0;
    }

    pub fn build(
        &self,
        start_anchored: StateID,
        start_unanchored: StateID,
    ) -> Result<NFA, Error> {
        assert!(self.pattern_id.is_none(), "must call 'finish_pattern' first");
        trace!(
            "intermediate NFA compilation via builder is complete, \
             intermediate NFA size: {} states, {} bytes on heap",
            self.states.len(),
            self.memory_usage(),
        );

        let mut nfa = nfa::Inner::default();
        // A set of compiler internal state IDs that correspond to states
        // that are exclusively epsilon transitions, i.e., goto instructions,
        // combined with the state that they point to. This is used to
        // record said states while transforming the compiler's internal NFA
        // representation to the external form.
        let mut empties = vec![];
        // A map used to re-map state IDs when translating this builder's
        // internal NFA state representation to the final NFA representation.
        let mut remap = vec![];
        remap.resize(self.states.len(), StateID::ZERO);

        nfa.set_starts(start_anchored, start_unanchored, &self.start_pattern);
        nfa.set_captures(&self.captures);
        // The idea here is to convert our intermediate states to their final
        // form. The only real complexity here is the process of converting
        // transitions, which are expressed in terms of state IDs. The new
        // set of states will be smaller because of partial epsilon removal,
        // so the state IDs will not be the same.
        for (sid, state) in self.states.iter().with_state_ids() {
            match *state {
                State::Empty { next } => {
                    // Since we're removing empty states, we need to handle
                    // them later since we don't yet know which new state this
                    // empty state will be mapped to.
                    empties.push((sid, next));
                }
                State::Range { range } => {
                    remap[sid] = nfa.add(nfa::State::Range { range })?;
                }
                State::Sparse { ref ranges } => {
                    let ranges = ranges.to_owned().into_boxed_slice();
                    remap[sid] =
                        nfa.add(nfa::State::Sparse(SparseTransitions {
                            ranges,
                        }))?;
                }
                State::Look { look, next } => {
                    remap[sid] = nfa.add(nfa::State::Look { look, next })?;
                }
                State::CaptureStart { pattern_id, capture_index, next } => {
                    // We can't remove this empty state because of the side
                    // effect of capturing an offset for this capture slot.
                    let slot = nfa.slot(pattern_id, capture_index);
                    remap[sid] =
                        nfa.add(nfa::State::Capture { next, slot })?;
                }
                State::CaptureEnd { pattern_id, capture_index, next } => {
                    // We can't remove this empty state because of the side
                    // effect of capturing an offset for this capture slot.
                    let slot = nfa.slot(pattern_id, capture_index);
                    remap[sid] =
                        nfa.add(nfa::State::Capture { next, slot })?;
                }
                State::Union { ref alternates } => {
                    let alternates = alternates.to_owned().into_boxed_slice();
                    remap[sid] = nfa.add(nfa::State::Union { alternates })?;
                }
                State::UnionReverse { ref alternates } => {
                    let mut alternates =
                        alternates.to_owned().into_boxed_slice();
                    alternates.reverse();
                    remap[sid] = nfa.add(nfa::State::Union { alternates })?;
                }
                State::Fail => {
                    remap[sid] = nfa.add(nfa::State::Fail)?;
                }
                State::Match { pattern_id } => {
                    remap[sid] = nfa.add(nfa::State::Match { pattern_id })?;
                }
            }
        }
        for &(empty_id, mut empty_next) in empties.iter() {
            // empty states can point to other empty states, forming a chain.
            // So we must follow the chain until the end, which must end at
            // a non-empty state, and therefore, a state that is correctly
            // remapped. We are guaranteed to terminate because our compiler
            // never builds a loop among only empty states.
            while let State::Empty { next } = self.states[empty_next] {
                empty_next = next;
            }
            remap[empty_id] = remap[empty_next];
        }
        nfa.remap(&remap);
        let final_nfa = NFA(Arc::new(nfa));
        trace!(
            "NFA compilation via builder complete, \
             final NFA size: {} states, {} bytes on heap",
            final_nfa.states().len(),
            final_nfa.memory_usage(),
        );
        Ok(final_nfa)
    }

    pub fn start_pattern(&mut self) -> Result<PatternID, Error> {
        assert!(self.pattern_id.is_none(), "must call 'finish_pattern' first");

        let proposed = self.start_pattern.len();
        let pid = PatternID::new(proposed)
            .map_err(|_| Error::too_many_patterns(proposed))?;
        self.pattern_id = Some(pid);
        // This gets filled in when 'finish_pattern' is called.
        self.start_pattern.push(StateID::ZERO);
        Ok(pid)
    }

    pub fn finish_pattern(
        &mut self,
        start_id: StateID,
    ) -> Result<PatternID, Error> {
        let pid = self.current_pattern_id();
        self.start_pattern[pid] = start_id;
        Ok(pid)
    }

    pub fn current_pattern_id(&self) -> PatternID {
        self.pattern_id.expect("must call 'start_pattern' first")
    }

    pub fn pattern_len(&self) -> usize {
        self.start_pattern.len()
    }

    pub fn add_empty(&mut self) -> Result<StateID, Error> {
        let _ = self.current_pattern_id();
        self.add(State::Empty { next: StateID::ZERO })
    }

    pub fn add_union(
        &mut self,
        alternates: Vec<StateID>,
    ) -> Result<StateID, Error> {
        self.add(State::Union { alternates })
    }

    pub fn add_union_reverse(
        &mut self,
        alternates: Vec<StateID>,
    ) -> Result<StateID, Error> {
        self.add(State::UnionReverse { alternates })
    }

    pub fn add_range(&mut self, range: Transition) -> Result<StateID, Error> {
        // self.byte_class_set.set_range(range.start, range.end);
        self.add(State::Range { range })
    }

    pub fn add_sparse(
        &mut self,
        ranges: Vec<Transition>,
    ) -> Result<StateID, Error> {
        // for range in sparse.ranges.iter() {
        // self.byte_class_set.set_range(range.start, range.end);
        // }
        self.add(State::Sparse { ranges })
    }

    pub fn add_look(
        &mut self,
        next: StateID,
        look: Look,
    ) -> Result<StateID, Error> {
        self.add(State::Look { look, next })
    }

    pub fn add_capture_start(
        &mut self,
        capture_index: u32,
        name: Option<Arc<str>>,
    ) -> Result<StateID, Error> {
        let pid = self.current_pattern_id();
        let capture_index = match usize::try_from(capture_index) {
            Err(_) => {
                return Err(Error::invalid_capture_index(core::usize::MAX))
            }
            Ok(capture_index) => capture_index,
        };
        // Make sure we have space to insert our (pid,index)|-->name mapping.
        if pid.as_usize() >= self.captures.len() {
            // Note that we require that if you're adding capturing groups,
            // then there must be at least one capturing group per pattern.
            // Moreover, whenever we expand our space here, it should always
            // first be for the first capture group (at index==0).
            if pid.as_usize() > self.captures.len() || capture_index > 0 {
                return Err(Error::invalid_capture_index(capture_index));
            }
            self.captures.push(vec![]);
        }
        if capture_index >= self.captures[pid].len() {
            // We require that capturing groups are added in correspondence
            // to their index. So no discontinuous indices. This is likely
            // overly strict, but also makes it simpler to provide guarantees
            // about our capturing group data.
            if capture_index > self.captures[pid].len() {
                return Err(Error::invalid_capture_index(capture_index));
            }
            self.captures[pid].push(None);
            // We check that 'slots' remains valid, since slots could in theory
            // overflow 'usize' without capture indices overflowing usize.
            // (Although, it seems only likely on 16-bit systems.) Either way,
            // we check that no overflows occur here. Also, note that we add
            // 2 because each capture group has two slots (start and end).
            // Otherwise, the NFA itself ultimately owns the allocation of
            // slots. We only track it here in the builder to ensure that the
            // total number ends up being valid.
            self.slots = match self.slots.checked_add(2) {
                Some(slots) => slots,
                None => {
                    return Err(Error::invalid_capture_index(capture_index))
                }
            };
        }
        self.add(State::CaptureStart {
            pattern_id: pid,
            capture_index,
            next: StateID::ZERO,
        })
    }

    pub fn add_capture_end(
        &mut self,
        capture_index: u32,
    ) -> Result<StateID, Error> {
        let pid = self.current_pattern_id();
        let capture_index = match usize::try_from(capture_index) {
            Err(_) => {
                return Err(Error::invalid_capture_index(core::usize::MAX))
            }
            Ok(capture_index) => capture_index,
        };
        // If we haven't already added this capture group via a corresponding
        // 'add_capture_start' call, then we consider the index given to be
        // invalid.
        if pid.as_usize() >= self.captures.len()
            || capture_index >= self.captures[pid].len()
        {
            return Err(Error::invalid_capture_index(capture_index));
        }
        self.add(State::CaptureEnd {
            pattern_id: pid,
            capture_index,
            next: StateID::ZERO,
        })
    }

    pub fn add_fail(&mut self) -> Result<StateID, Error> {
        self.add(State::Fail)
    }

    pub fn add_match(&mut self) -> Result<StateID, Error> {
        let pattern_id = self.current_pattern_id();
        let sid = self.add(State::Match { pattern_id })?;
        Ok(sid)
    }

    fn add(&mut self, state: State) -> Result<StateID, Error> {
        let id = StateID::new(self.states.len())
            .map_err(|_| Error::too_many_states(self.states.len()))?;
        self.memory_states += state.memory_usage();
        self.states.push(state);
        self.check_size_limit()?;
        Ok(id)
    }

    pub fn patch(&mut self, from: StateID, to: StateID) -> Result<(), Error> {
        let old_memory_states = self.memory_states;
        match self.states[from] {
            State::Empty { ref mut next } => {
                *next = to;
            }
            State::Range { ref mut range } => {
                range.next = to;
            }
            State::Sparse { .. } => {
                panic!("cannot patch from a sparse NFA state")
            }
            State::Look { ref mut next, .. } => {
                *next = to;
            }
            State::Union { ref mut alternates } => {
                alternates.push(to);
                self.memory_states += mem::size_of::<StateID>();
            }
            State::UnionReverse { ref mut alternates } => {
                alternates.push(to);
                self.memory_states += mem::size_of::<StateID>();
            }
            State::CaptureStart { ref mut next, .. } => {
                *next = to;
            }
            State::CaptureEnd { ref mut next, .. } => {
                *next = to;
            }
            State::Fail => {}
            State::Match { .. } => {}
        }
        if old_memory_states != self.memory_states {
            self.check_size_limit()?;
        }
        Ok(())
    }

    pub fn set_size_limit(
        &mut self,
        limit: Option<usize>,
    ) -> Result<(), Error> {
        self.size_limit = limit;
        self.check_size_limit()
    }

    /// Returns the heap memory usage, in bytes, used by the NFA states added
    /// so far.
    ///
    /// Note that this is an approximation of how big the final NFA will be.
    /// In practice, the final NFA will likely be a bit smaller because of
    /// its simpler state representation. (For example, using things like
    /// `Box<[StateID]>` instead of `Vec<StateID>`.)
    pub fn memory_usage(&self) -> usize {
        self.states.len() * mem::size_of::<State>() + self.memory_states
    }

    fn check_size_limit(&self) -> Result<(), Error> {
        if let Some(limit) = self.size_limit {
            if self.memory_usage() > limit {
                return Err(Error::exceeded_size_limit(limit));
            }
        }
        Ok(())
    }
}
