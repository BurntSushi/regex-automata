use core::{convert::TryFrom, mem};

use alloc::{sync::Arc, vec::Vec};

use crate::{
    nfa::thompson::{
        error::Error,
        nfa::{self, Look, SparseTransitions, Transition, NFA},
    },
    util::id::{PatternID, StateID},
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
    /// A map used to re-map state IDs when translating this builder's internal
    /// NFA state representation to the final NFA representation.
    remap: Vec<StateID>,
    /// A set of compiler internal state IDs that correspond to states that are
    /// exclusively epsilon transitions, i.e., goto instructions, combined with
    /// the state that they point to. This is used to record said states while
    /// transforming the compiler's internal NFA representation to the external
    /// form.
    empties: Vec<(StateID, StateID)>,
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

impl Builder {
    pub fn new() -> Builder {
        Builder::default()
    }

    pub fn clear(&mut self) {
        self.pattern_id = None;
        self.states.clear();
        self.start_pattern.clear();
        self.remap.clear();
        self.empties.clear();
        self.memory_states = 0;
    }

    pub fn build(&self) -> Result<NFA, Error> {
        assert!(self.pattern_id.is_none(), "must call 'finish_pattern' first");

        let inner = nfa::Inner::default();
        Ok(NFA(Arc::new(inner)))
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
        // BREADCRUMBS: Do we build facts and byteset here in the builder?
        // Or keep that in mutating methods on the NFA? If we want to avoid
        // mutating methods on the NFA, then it seems like we should do it here
        // in the builder......?

        // self.facts.set_has_any_look(true);
        // look.add_to_byteset(&mut self.byte_class_set);
        // match look {
        // Look::StartLine
        // | Look::EndLine
        // | Look::StartText
        // | Look::EndText => {
        // self.facts.set_has_any_anchor(true);
        // }
        // Look::WordBoundaryUnicode | Look::WordBoundaryUnicodeNegate => {
        // self.facts.set_has_word_boundary_unicode(true);
        // }
        // Look::WordBoundaryAscii | Look::WordBoundaryAsciiNegate => {
        // self.facts.set_has_word_boundary_ascii(true);
        // }
        // }
        self.add(State::Look { look, next })
    }

    pub fn add_capture_start(
        &mut self,
        capture_index: u32,
        name: Option<Arc<str>>,
    ) -> Result<StateID, Error> {
        // BREADCRUMBS: Here in the builder, we have some flexibility with how
        // we deal with capturing groups, since we can put them through another
        // transformation step when we produce the final NFA. So how should we
        // store them?
        //
        // At minimum, we need an efficient mapping of the form:
        //
        //   (PatternID, PatternGroupIndex) |--> Absolute Slot Offset
        //
        // We could use Vec<Range<usize>>, which is indexed by pattern and the
        // range is either absolute capture group index or slots. Slots seem
        // a little weird, but they tend to be the only thing we care about
        // in "absolute" terms, so that might actually be the most natural
        // representation. But... capture group index is seemingly easier to
        // reason about and seems more "fundamental." Either way, we can go
        // between them very easily. Using slots directly would avoid some
        // arithmetic I think on capture group lookup.
        //
        // What else? We also need to track capture group names. In the regex
        // public API, we need:
        //
        //   (PatternID, PatternGroupName) |--> PatternGroupIndex
        //
        // but we also need the sequence:
        //
        //   <for each pattern <for each group, yield Option<NameString>>
        //
        // For the builder, I think if we only store the above map, then we
        // can produce the sequence from it and the map above when we build the
        // final NFA. Hmmm... Nope. I think it's actually the other way around!
        //
        // I suppose the other question is whether the builder should even
        // concern itself with slots in the first place.
        //
        // And in fact, I think all we actually need is the sequence! We can
        // build *both* maps above from that sequence I think.
        //
        // I think we should validate as much---perhaps all---properties of
        // capturing groups as we can here. It would be better to fail fast
        // than to wait until we go to build the final NFA.
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

    fn patch(&mut self, from: StateID, to: StateID) -> Result<(), Error> {
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
