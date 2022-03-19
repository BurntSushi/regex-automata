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

/// An abstraction for building Thompson NFAs by hand.
///
/// A builder is what a [`thompson::Compiler`](nfa::Compiler) uses internally
/// to translate a regex's high-level intermediate representation into an
/// [`NFA`].
///
/// The primary function of this builder is to abstract away the internal
/// representation of an NFA and make it difficult to produce NFAs are that
/// internally invalid or inconsistent. This builder also provides a way to
/// add "empty" states (which can be thought of as unconditional epsilon
/// transitions), despite the fact that [`thompson::State`](nfa::State) does
/// not have any "empty" representation. The advantage of "empty" states is
/// that they make the code for constructing a Thompson NFA logically simpler.
///
/// Many of the routines on this builder may panic or return errors. Generally
/// speaking, panics occur when an invalid sequence of method calls were made,
/// where as an error occurs if things get too big. (Where "too big" might mean
/// exhausting identifier space or using up too much heap memory in accordance
/// with the configured [`size_limit`](Builder::set_size_limit).)
///
/// # Overview
///
/// ## Adding multiple patterns
///
/// Each pattern you add to an NFA should correspond to a pair of
/// [`Builder::start_pattern`] and [`Builder::end_pattern`] calls, with calls
/// inbetween that add NFA states for that pattern. NFA states may be added
/// without first calling `start_pattern`, with the exception of adding
/// capturing states.
///
/// ## Adding NFA states
///
/// Here is a very brief overview of each of the methods that add NFA states.
/// Every method adds a single state.
///
/// * [`add_empty`](Builder::add_empty): Add a state with a single
/// unconditional epsilon transition to another state.
/// * [`add_union`](Builder::add_union): Adds a state with unconditional
/// epsilon transitions to two or more states, with earlier transitions
/// preferred over later ones.
/// * [`add_union_reverse`](Builder::add_union_reverse): Adds a state with
/// unconditional epsilon transitions to two or more states, with later
/// transitions preferred over earlier ones.
/// * [`add_range`](Builder::add_range): Adds a state with a single transition
/// to another state that can only be followed if the current input byte is
/// within the range given.
/// * [`add_sparse`](Builder::add_sparse): Adds a state with two or more
/// range transitions to other states, where a transition is only followed
/// if the current input byte is within one of the ranges. All transitions
/// in this state have equal priority, and the corresponding ranges must be
/// non-overlapping.
/// * [`add_look`](Builder::add_look): Adds a state with a single *conditional*
/// epsilon transition to another state, where the condition depends on a
/// limited look-around property.
/// * [`add_capture_start`](Builder::add_capture_start): Adds a state with
/// a single unconditional epsilon transition that also instructs an NFA
/// simulation to record the current input position to a specific location in
/// memory. This is intended to represent the starting location of a capturing
/// group.
/// * [`add_capture_end`](Builder::add_capture_end): Adds a state with
/// a single unconditional epsilon transition that also instructs an NFA
/// simulation to record the current input position to a specific location in
/// memory. This is intended to represent the ending location of a capturing
/// group.
/// * [`add_fail`](Builder::add_fail): Adds a state that never transitions to
/// another state.
/// * [`add_match`](Builder::add_match): Add a state that indicates a match has
/// been found for a particular pattern. A match state is a final state with
/// no outgoing transitions.
///
/// ## Setting transitions between NFA states
///
/// The [`Builder::patch`] method creates a transition from one state to the
/// next. If the `from` state corresponds to a state that supports multiple
/// outgoing transitions (such as "union"), then this adds the corresponding
/// transition. Otherwise, it sets the single transition. (This routine panics
/// if `from` corresponds to a state added by `add_sparse`, since sparse states
/// need more specialized handling.)
///
/// # Example
///
/// This annotated example shows how to hand construct the regex `[a-z]+`
/// (without an unanchored prefix).
///
/// ```
/// use regex_automata::{
///     nfa::thompson::{pikevm::PikeVM, Builder, Transition},
///     util::id::StateID,
/// };
///
/// let mut builder = Builder::new();
/// // Before adding NFA states for our pattern, we need to tell the builder
/// // that we are starting the pattern.
/// builder.start_pattern()?;
/// // Since we use the Pike VM below for searching, we need to add capturing
/// // states. If you're just going to build a DFA from the NFA, then capturing
/// // states do not need to be added.
/// let start = builder.add_capture_start(StateID::ZERO, 0, None)?;
/// let range = builder.add_range(Transition {
///     // We don't know the state ID of the 'next' state yet, so we just fill
///     // in a dummy 'ZERO' value.
///     start: b'a', end: b'z', next: StateID::ZERO
/// })?;
/// // This state will point back to 'range', but also enable us to move ahead.
/// // That is, this implements the '+' repetition operator. We add 'range' and
/// // then 'end' below to this alternation.
/// let alt = builder.add_union(vec![])?;
/// // The final state before the match state, which serves to capture the
/// // end location of the match.
/// let end = builder.add_capture_end(StateID::ZERO, 0)?;
/// // The match state for our pattern.
/// let mat = builder.add_match()?;
/// // Now we fill in the transitions between states.
/// builder.patch(start, range);
/// builder.patch(range, alt);
/// // If we added 'end' before 'range', then we'd implement non-greedy
/// // matching, i.e., '+?'.
/// builder.patch(alt, range);
/// builder.patch(alt, end);
/// builder.patch(end, mat);
/// // We must explicitly finish pattern and provide the starting state ID for
/// // this particular pattern.
/// builder.finish_pattern(start)?;
/// // Finally, when we build the NFA, we provide the anchored and unanchored
/// // starting state IDs. Since we didn't bother with an unanchored prefix
/// // here, we only support anchored searching. Thus, both starting states are
/// // the same.
/// let nfa = builder.build(start, start)?;
///
/// // Now build a Pike VM from our NFA, and use it for searching. This shows
/// // how we can use a regex engine without ever worry about syntax!
/// let vm = PikeVM::new_from_nfa(nfa)?;
/// let mut cache = vm.create_cache();
/// let mut caps = vm.create_captures();
/// let m = vm.find_leftmost(&mut cache, b"foo0", &mut caps).unwrap();
/// assert_eq!(m.pattern().as_usize(), 0);
/// assert_eq!(m.start(), 0);
/// assert_eq!(m.end(), 3);
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
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

impl Builder {
    /// Create a new builder for hand-assembling NFAs.
    pub fn new() -> Builder {
        Builder::default()
    }

    /// Clear this builder.
    ///
    /// Clearing removes all state associated with building an NFA, but does
    /// not reset configuration (such as size limits). After clearing, the
    /// builder can be reused to assemble an entirely new DFA.
    pub fn clear(&mut self) {
        self.pattern_id = None;
        self.states.clear();
        self.start_pattern.clear();
        self.captures.clear();
        self.slots = 0;
        self.memory_states = 0;
    }

    /// Assemble a [`NFA`] from the states added so far.
    ///
    /// After building an NFA, more states may be added and `build` may be
    /// called again. To reuse a builder to produce an entirely new NFA from
    /// scratch, call the [`clear`](Builder::clear) method first.
    ///
    /// `start_anchored` refers to the ID of the starting state that anchored
    /// searches should use. That is, searches who matches are limited to the
    /// starting position of the search.
    ///
    /// `start_unanchored` refers to the ID of the starting state that
    /// unanchored searches should use. This permits searches to report matches
    /// that start after the beginning of the search. In cases where unanchored
    /// searches are not supported, the unanchored starting state ID must be
    /// the same as the anchored starting state ID.
    ///
    /// # Errors
    ///
    /// This currently never returns an error, but this is subject to change.
    ///
    /// # Panics
    ///
    /// If `start_pattern` was called, then `finish_pattern` must be called
    /// before `build`, otherwise this panics.
    ///
    /// This may panic for other invalid uses of a builder. For example, if
    /// a "start capture" state was added without a corresponding "end capture"
    /// state.
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
                    remap[sid] = nfa.add(nfa::State::Range { range });
                }
                State::Sparse { ref ranges } => {
                    remap[sid] = match ranges.len() {
                        0 => nfa.add(nfa::State::Fail),
                        1 => nfa.add(nfa::State::Range { range: ranges[0] }),
                        _ => {
                            let ranges = ranges.to_owned().into_boxed_slice();
                            let sparse = SparseTransitions { ranges };
                            nfa.add(nfa::State::Sparse(sparse))
                        }
                    }
                }
                State::Look { look, next } => {
                    remap[sid] = nfa.add(nfa::State::Look { look, next });
                }
                State::CaptureStart { pattern_id, capture_index, next } => {
                    // We can't remove this empty state because of the side
                    // effect of capturing an offset for this capture slot.
                    let slot = nfa.slot(pattern_id, capture_index);
                    remap[sid] = nfa.add(nfa::State::Capture { next, slot });
                }
                State::CaptureEnd { pattern_id, capture_index, next } => {
                    // We can't remove this empty state because of the side
                    // effect of capturing an offset for this capture slot.
                    // Also, this always succeeds because we check that all
                    // slot indices are valid for all capture indices when they
                    // are initially added.
                    let slot = nfa
                        .slot(pattern_id, capture_index)
                        .checked_add(1)
                        .unwrap();
                    remap[sid] = nfa.add(nfa::State::Capture { next, slot });
                }
                State::Union { ref alternates } => {
                    if alternates.is_empty() {
                        remap[sid] = nfa.add(nfa::State::Fail);
                    } else if alternates.len() == 1 {
                        empties.push((sid, alternates[0]));
                    } else {
                        let alternates =
                            alternates.to_owned().into_boxed_slice();
                        remap[sid] = nfa.add(nfa::State::Union { alternates });
                    }
                }
                State::UnionReverse { ref alternates } => {
                    if alternates.is_empty() {
                        remap[sid] = nfa.add(nfa::State::Fail);
                    } else if alternates.len() == 1 {
                        empties.push((sid, alternates[0]));
                    } else {
                        let mut alternates =
                            alternates.to_owned().into_boxed_slice();
                        alternates.reverse();
                        remap[sid] = nfa.add(nfa::State::Union { alternates });
                    }
                }
                State::Fail => {
                    remap[sid] = nfa.add(nfa::State::Fail);
                }
                State::Match { pattern_id } => {
                    remap[sid] = nfa.add(nfa::State::Match { pattern_id });
                }
            }
        }
        // Some of the new states still point to empty state IDs, so we need to
        // follow each of them and remap the empty state IDs to their non-empty
        // state IDs.
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
        // Finally remap all of the state IDs.
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

    /// Start the assembly of a pattern in this NFA.
    ///
    /// Upon success, this returns the identifier for the new pattern.
    /// Identifiers start at `0` and are incremented by 1 for each new pattern.
    ///
    /// It is necessary to call this routine before adding capturing states.
    /// Otherwise, any other NFA state may be added before starting a pattern.
    ///
    /// # Errors
    ///
    /// If the pattern identifier space is exhausted, then this returns an
    /// error.
    ///
    /// # Panics
    ///
    /// If this is called while assembling another pattern (i.e., before
    /// `finish_pattern` is called), then this panics.
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

    /// Finish the assembly of a pattern in this NFA.
    ///
    /// Upon success, this returns the identifier for the new pattern.
    /// Identifiers start at `0` and are incremented by 1 for each new
    /// pattern. This is the same identifier returned by the corresponding
    /// `start_pattern` call.
    ///
    /// Note that `start_pattern` and `finish_pattern` pairs cannot be
    /// interleaved or nested. A correct `finish_pattern` call _always_
    /// corresponds to the most recently called `start_pattern` routine.
    ///
    /// # Errors
    ///
    /// This currently never returns an error, but this is subject to change.
    ///
    /// # Panics
    ///
    /// If this is called without a corresponding `start_pattern` call, then
    /// this panics.
    pub fn finish_pattern(
        &mut self,
        start_id: StateID,
    ) -> Result<PatternID, Error> {
        let pid = self.current_pattern_id();
        self.start_pattern[pid] = start_id;
        self.pattern_id = None;
        Ok(pid)
    }

    /// Returns the pattern identifier of the current pattern.
    ///
    /// # Panics
    ///
    /// If this doesn't occur after a `start_pattern` call and before the
    /// corresponding `finish_pattern` call, then this panics.
    pub fn current_pattern_id(&self) -> PatternID {
        self.pattern_id.expect("must call 'start_pattern' first")
    }

    /// Returns the number of patterns added to this builder so far.
    ///
    /// This only includes patterns that have had `finish_pattern` called
    /// for them.
    pub fn pattern_len(&self) -> usize {
        self.start_pattern.len()
    }

    /// Add an "empty" NFA state.
    ///
    /// An "empty" NFA state is a state with a single unconditional epsilon
    /// transition to another NFA state. Such empty states are removed before
    /// building the final [`NFA`] (which has no such "empty" states), but they
    /// can be quite useful in the construction process of an NFA.
    ///
    /// # Errors
    ///
    /// This returns an error if the state identifier space is exhausted, or if
    /// the configured heap size limit has been exceeded.
    pub fn add_empty(&mut self) -> Result<StateID, Error> {
        self.add(State::Empty { next: StateID::ZERO })
    }

    /// Add a "union" NFA state.
    ///
    /// A "union" NFA state that contains zero or more unconditional epsilon
    /// transitions to other NFA states. The order of these transitions
    /// reflects a priority order where earlier transitions are preferred over
    /// later transitions.
    ///
    /// Callers may provide an empty set of alternates to this method call, and
    /// then later add transitions via `patch`. At final build time, a "reverse
    /// union" state with no alternates is converted to a "fail" state, and a
    /// "reverse union" state with exactly one alternate is treated as if it
    /// were an "empty" state.
    ///
    /// # Errors
    ///
    /// This returns an error if the state identifier space is exhausted, or if
    /// the configured heap size limit has been exceeded.
    pub fn add_union(
        &mut self,
        alternates: Vec<StateID>,
    ) -> Result<StateID, Error> {
        self.add(State::Union { alternates })
    }

    /// Add a "reverse union" NFA state.
    ///
    /// A "reverse union" NFA state contains zero or more unconditional epsilon
    /// transitions to other NFA states. The order of these transitions
    /// reflects a priority order where later transitions are preferred
    /// over earlier transitions. This is an inverted priority order when
    /// compared to `add_union`. This is useful, for example, for implementing
    /// non-greedy repetition operators.
    ///
    /// Callers may provide an empty set of alternates to this method call, and
    /// then later add transitions via `patch`. At final build time, a "reverse
    /// union" state with no alternates is converted to a "fail" state, and a
    /// "reverse union" state with exactly one alternate is treated as if it
    /// were an "empty" state.
    ///
    /// # Errors
    ///
    /// This returns an error if the state identifier space is exhausted, or if
    /// the configured heap size limit has been exceeded.
    pub fn add_union_reverse(
        &mut self,
        alternates: Vec<StateID>,
    ) -> Result<StateID, Error> {
        self.add(State::UnionReverse { alternates })
    }

    /// Add a "range" NFA state.
    ///
    /// A "range" NFA state is a state with one outgoing transition to another
    /// state, where that transition may only be followed if the current input
    /// byte falls between a range of bytes given.
    ///
    /// # Errors
    ///
    /// This returns an error if the state identifier space is exhausted, or if
    /// the configured heap size limit has been exceeded.
    pub fn add_range(&mut self, range: Transition) -> Result<StateID, Error> {
        self.add(State::Range { range })
    }

    /// Add a "sparse" NFA state.
    ///
    /// A "sparse" NFA state contains zero or more outgoing transitions, where
    /// the transition to be followed (if any) is chosen based on whether the
    /// current input byte falls in the range of one such transition. The
    /// transitions given *must* be non-overlapping and in ascending order. (A
    /// "sparse" state with no transitions is equivalent to a "fail" state.)
    ///
    /// A "sparse" state is like adding a "union" state and pointing it at a
    /// bunch of "range" states, except that the different alternates have
    /// equal priority.
    ///
    /// Note that a "sparse" state is the only state that cannot be patched.
    /// This is because a "sparse" state has many transitions, each of which
    /// may point to a different NFA state. Moreover, adding more such
    /// transitions requires more than just an NFA state ID to point to. It
    /// also requires a byte range. The `patch` routine does not support the
    /// additional information required. Therefore, callers must ensure that
    /// all outgoing transitions for this state are included when `add_sparse`
    /// is called. There is no way to add more later.
    ///
    /// # Errors
    ///
    /// This returns an error if the state identifier space is exhausted, or if
    /// the configured heap size limit has been exceeded.
    ///
    /// # Panics
    ///
    /// This routine _may_ panic if the transitions given overlap or are not
    /// in ascending order.
    pub fn add_sparse(
        &mut self,
        ranges: Vec<Transition>,
    ) -> Result<StateID, Error> {
        self.add(State::Sparse { ranges })
    }

    /// Add a "look" NFA state.
    ///
    /// A "look" NFA state corresponds to a state with exactly one
    /// *conditional* epsilon transition to another NFA state. Namely, it
    /// represents one of a small set of simplistic look-around operators.
    ///
    /// Callers may provide a "dummy" state ID (typically [`StateID::ZERO`]),
    /// and then change it later with [`patch`](Builder::patch).
    ///
    /// # Errors
    ///
    /// This returns an error if the state identifier space is exhausted, or if
    /// the configured heap size limit has been exceeded.
    pub fn add_look(
        &mut self,
        next: StateID,
        look: Look,
    ) -> Result<StateID, Error> {
        self.add(State::Look { look, next })
    }

    /// Add a "start capture" NFA state.
    ///
    /// A "start capture" NFA state corresponds to a state with exactly one
    /// outgoing unconditional epsilon transition to another state. Unlike
    /// "empty" states, a "start capture" state also carries with it an
    /// instruction for saving the current position of input to a particular
    /// location in memory. NFA simulations, like the Pike VM, may use this
    /// information to report the match locations of capturing groups in a
    /// regex pattern.
    ///
    /// If the corresponding capturing group has a name, then callers should
    /// include it here.
    ///
    /// Callers may provide a "dummy" state ID (typically [`StateID::ZERO`]),
    /// and then change it later with [`patch`](Builder::patch).
    ///
    /// Note that unlike `start_pattern`/`finish_pattern`, capturing start and
    /// end states may be interleaved. Indeed, it is typical for many "start
    /// capture" NFA states to appear before the first "end capture" state.
    ///
    /// # Errors
    ///
    /// This returns an error if the state identifier space is exhausted, or if
    /// the configured heap size limit has been exceeded.
    ///
    /// This also returns an error if an invalid capture index is given.
    /// A capture index is invalid if it cannot fit into a `usize` or is
    /// otherwise discontiguous with other capture indices that have been
    /// added. Moreover, if any capturing states are added, then at least one
    /// pair of capturing states must be present for _every_ pattern added to
    /// this builder. Otherwise, an error will be returned.
    pub fn add_capture_start(
        &mut self,
        next: StateID,
        capture_index: u32,
        name: Option<Arc<str>>,
    ) -> Result<StateID, Error> {
        // FIXME: It seems like it would be nice to track whether an 'end'
        // capture is added for each 'start' capture. It will require extra
        // memory, but it should overall be very little compared to what we're
        // otherwise using.

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
        self.add(State::CaptureStart { pattern_id: pid, capture_index, next })
    }

    /// Add a "end capture" NFA state.
    ///
    /// A "end capture" NFA state corresponds to a state with exactly one
    /// outgoing unconditional epsilon transition to another state. Unlike
    /// "empty" states, a "end capture" state also carries with it an
    /// instruction for saving the current position of input to a particular
    /// location in memory. NFA simulations, like the Pike VM, may use this
    /// information to report the match locations of capturing groups in a
    ///
    /// Callers may provide a "dummy" state ID (typically [`StateID::ZERO`]),
    /// and then change it later with [`patch`](Builder::patch).
    /// regex pattern.
    ///
    /// Note that unlike `start_pattern`/`finish_pattern`, capturing start and
    /// end states may be interleaved. Indeed, it is typical for many "start
    /// capture" NFA states to appear before the first "end capture" state.
    ///
    /// # Errors
    ///
    /// This returns an error if the state identifier space is exhausted, or if
    /// the configured heap size limit has been exceeded.
    ///
    /// This also returns an error if an invalid capture index is given.
    /// A capture index is invalid if it cannot fit into a `usize` or is
    /// otherwise discontiguous with other capture indices that have been
    /// added. Moreover, if any capturing states are added, then at least one
    /// pair of capturing states must be present for _every_ pattern added to
    /// this builder. Otherwise, an error will be returned.
    ///
    /// If there was no corresponding "start capture" NFA state added, then
    /// this returns an error.
    pub fn add_capture_end(
        &mut self,
        next: StateID,
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
        self.add(State::CaptureEnd { pattern_id: pid, capture_index, next })
    }

    /// Adds a "fail" NFA state.
    ///
    /// A "fail" state is simply a state that has no outgoing transitions. It
    /// acts as a way to cause a search to stop without reporting a match.
    /// For example, one way to represent an NFA with zero patterns is with a
    /// single "fail" state.
    ///
    /// # Errors
    ///
    /// This returns an error if the state identifier space is exhausted, or if
    /// the configured heap size limit has been exceeded.
    pub fn add_fail(&mut self) -> Result<StateID, Error> {
        self.add(State::Fail)
    }

    /// Adds a "match" NFA state.
    ///
    /// A "match" state has no outgoing transitions (just like a "fail"
    /// state), but it has special significance in that if a search enters
    /// this state, then a match has been found. The match state that is added
    /// automatically has the current pattern ID associated with it. This is
    /// used to report the matching pattern ID at search time.
    ///
    /// # Errors
    ///
    /// This returns an error if the state identifier space is exhausted, or if
    /// the configured heap size limit has been exceeded.
    ///
    /// # Panics
    ///
    /// This must be called after a `start_pattern` call but before the
    /// corresponding `finish_pattern` call. Otherwise, it panics.
    pub fn add_match(&mut self) -> Result<StateID, Error> {
        let pattern_id = self.current_pattern_id();
        let sid = self.add(State::Match { pattern_id })?;
        Ok(sid)
    }

    /// The common implementation of "add a state." It handles the common
    /// error cases of state ID exhausting (by owning state ID allocation) and
    /// whether the size limit has been exceeded.
    fn add(&mut self, state: State) -> Result<StateID, Error> {
        let id = StateID::new(self.states.len())
            .map_err(|_| Error::too_many_states(self.states.len()))?;
        self.memory_states += state.memory_usage();
        self.states.push(state);
        self.check_size_limit()?;
        Ok(id)
    }

    /// Add a transition from one state to another.
    ///
    /// This routine is called "patch" since it is very common to add the
    /// states you want, typically with "dummy" state ID transitions, and then
    /// "patch" in the real state IDs later. This is because you don't always
    /// know all of the necessary state IDs to add because they might not
    /// exist yet.
    ///
    /// # Errors
    ///
    /// This may error if patching leads to an increase in heap usage beyond
    /// the configured size limit. Heap usage only grows when patching adds a
    /// new transition (as in the case of a "union" state).
    ///
    /// # Panics
    ///
    /// This panics if `from` corresponds to a "sparse" state. When "sparse"
    /// states are added, there is no way to patch them after-the-fact. (If you
    /// have a use case where this would be helpful, please file an issue. It
    /// will likely require a new API.)
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

    /// Set the size limit on this builder.
    ///
    /// Setting the size limit will also check whether the NFA built so far
    /// fits within the given size limit. If it doesn't, then an error is
    /// returned.
    ///
    /// By default, there is no configured size limit.
    pub fn set_size_limit(
        &mut self,
        limit: Option<usize>,
    ) -> Result<(), Error> {
        self.size_limit = limit;
        self.check_size_limit()
    }

    /// Return the currently configured size limit.
    ///
    /// By default, this returns `None`, which corresponds to no configured
    /// size limit.
    pub fn get_size_limit(&self) -> Option<usize> {
        self.size_limit
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