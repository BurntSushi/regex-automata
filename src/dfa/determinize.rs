use alloc::{collections::BTreeMap, rc::Rc, vec, vec::Vec};

use crate::{
    classes::{Byte, ByteSet},
    dfa::{automaton::Start, dense, Error},
    nfa::thompson::{self, Look, LookSet},
    sparse_set::{SparseSet, SparseSets},
    state_id::{dead_id, StateID},
    MatchKind, PatternID,
};

/// A map from states to state identifiers. When using std, we use a standard
/// hashmap, since it's a bit faster for this use case. (Other maps, like
/// one's based on FNV, have not yet been benchmarked.)
///
/// The main purpose of this map is to reuse states where possible. This won't
/// fully minimize the DFA, but it works well in a lot of cases.
#[cfg(feature = "std")]
type StateMap<S> = std::collections::HashMap<Rc<State>, S>;
#[cfg(not(feature = "std"))]
type StateMap<S> = BTreeMap<Rc<State>, S>;

/// A builder for configuring and running a DFA determinizer.
#[derive(Clone, Debug)]
pub(crate) struct Determinizer {
    anchored: bool,
    match_kind: MatchKind,
    quit: ByteSet,
}

impl Determinizer {
    /// Create a new determinizer. The determinizer may be configured before
    /// calling `run`.
    pub fn new() -> Determinizer {
        Determinizer {
            anchored: false,
            match_kind: MatchKind::LeftmostFirst,
            quit: ByteSet::empty(),
        }
    }

    /// Run determinization on the given NFA and write the resulting DFA into
    /// the one given. The DFA given should be initialized but otherwise empty.
    /// "Initialized" means that it is setup to handle the NFA's byte classes,
    /// number of patterns and whether to build start states for each pattern.
    pub fn run<S: StateID>(
        &self,
        nfa: &thompson::NFA,
        dfa: &mut dense::OwnedDFA<S>,
    ) -> Result<(), Error> {
        let dead = Rc::new(State::dead());
        let quit = Rc::new(State::dead());
        let mut cache = StateMap::default();
        // We only insert the dead state here since its representation is
        // identical to the quit state. And we never want anything pointing
        // to the quit state other than specific transitions derived from the
        // determinizer's configured "quit" bytes.
        cache.insert(dead.clone(), dead_id());

        Runner {
            nfa,
            dfa,
            builder_states: vec![dead, quit],
            cache,
            stack: vec![],
            scratch_nfa_states: vec![],
            anchored: self.anchored,
            match_kind: self.match_kind,
            quit: self.quit,
        }
        .run()
    }

    /// Whether to build an anchored DFA or not. When disabled (the default),
    /// the unanchored prefix from the NFA is used to start the DFA. Otherwise,
    /// the anchored start state of the NFA is used to start the DFA.
    pub fn anchored(&mut self, yes: bool) -> &mut Determinizer {
        self.anchored = yes;
        self
    }

    /// The match semantics to use for determinization.
    ///
    /// MatchKind::All corresponds to the standard textbook construction.
    /// All possible match states are represented in the DFA.
    /// MatchKind::LeftmostFirst permits greediness and otherwise tries to
    /// simulate the match semantics of backtracking regex engines. Namely,
    /// only a subset of match states are built, and dead states are used to
    /// stop searches with an unanchored prefix.
    ///
    /// The default is MatchKind::LeftmostFirst.
    pub fn match_kind(&mut self, kind: MatchKind) -> &mut Determinizer {
        self.match_kind = kind;
        self
    }

    /// The set of bytes to use that will cause the DFA to enter a quit state,
    /// stop searching and return an error. By default, this is empty.
    pub fn quit(&mut self, set: ByteSet) -> &mut Determinizer {
        self.quit = set;
        self
    }
}

/// The actual implementation of determinization that converts an NFA to a DFA
/// through powerset construction.
///
/// This determinizer roughly follows the typical powerset construction, where
/// each DFA state is comprised of one or more NFA states. In the worst case,
/// there is one DFA state for every possible combination of NFA states. In
/// practice, this only happens in certain conditions, typically when there are
/// bounded repetitions.
///
/// The main differences between this implementation and typical deteminization
/// are that this implementation delays matches by one state and hackily makes
/// look-around work. Comments below attempt to explain this.
///
/// The type variable `S` refers to the chosen state identifier representation
/// used for the DFA.
///
/// The lifetime variable `'a` refers to the lifetime of the NFA and DFA,
/// whichever is shorter.
#[derive(Debug)]
struct Runner<'a, S: StateID> {
    /// The NFA we're converting into a DFA.
    nfa: &'a thompson::NFA,
    /// The DFA we're building.
    dfa: &'a mut dense::OwnedDFA<S>,
    /// Each DFA state being built is defined as an *ordered* set of NFA
    /// states, along with some meta facts about the ordered set of NFA states.
    ///
    /// This is never empty. The first state is always a dummy state such that
    /// a state id == 0 corresponds to a dead state. The second state is always
    /// the quit state.
    ///
    /// Why do we have states in both a `Vec` and in a cache map below?
    /// Well, they serve two different roles based on access patterns.
    /// `builder_states` is the canonical home of each state, and provides
    /// constant random access by a state's ID. The cache map below, on the
    /// other hand, provides a quick way of searching for identical DFA states
    /// by using the DFA state as a key in the map. Of course, we use reference
    /// counting to avoid actually duplicating the state's data itself.
    /// (Although this has never been benchmarked.)
    builder_states: Vec<Rc<State>>,
    /// A cache of DFA states that already exist and can be easily looked up
    /// via ordered sets of NFA states.
    ///
    /// See `builder_states` docs for why we store states in two different
    /// ways.
    cache: StateMap<S>,
    /// Scratch space for a stack of NFA states to visit, for depth first
    /// visiting without recursion.
    stack: Vec<thompson::StateID>,
    /// Scratch space for storing an ordered sequence of NFA states, for
    /// amortizing allocation. This is principally useful for when we avoid
    /// adding a new DFA state since it already exists. In order to detect this
    /// case though, we still need an ordered set of NFA state IDs. So we use
    /// this space to stage that ordered set before we know whether we need to
    /// create a new DFA state or not.
    scratch_nfa_states: Vec<thompson::StateID>,
    /// Whether to build an anchored DFA or not.
    anchored: bool,
    /// Match semantics for this DFA.
    match_kind: MatchKind,
    /// Bytes on which the DFA should halt the search and report an error.
    quit: ByteSet,
}

impl<'a, S: StateID> Runner<'a, S> {
    /// Build the DFA. If there was a problem constructing the DFA (e.g., if
    /// the chosen state identifier representation is too small), then an error
    /// is returned.
    fn run(mut self) -> Result<(), Error> {
        if self.nfa.has_word_boundary_unicode()
            && !self.quit.contains_range(0x80, 0xFF)
        {
            return Err(Error::unsupported_dfa_word_boundary_unicode());
        }

        // A sequence of "representative" bytes drawn from each equivalence
        // class. These representative bytes are fed to the NFA to compute
        // state transitions. This allows us to avoid re-computing state
        // transitions for bytes that are guaranteed to produce identical
        // results.
        let representative_bytes: Vec<Byte> =
            self.dfa.byte_classes().representatives().collect();
        // A pair of sparse sets for tracking ordered sets of NFA state IDs.
        // These are reused throughout determinization.
        let mut sparses = SparseSets::new(self.nfa.len());
        // The set of all DFA state IDs that still need to have their
        // transitions set. We start by seeding this will all starting states.
        let mut uncompiled = vec![];
        self.add_all_starts(&mut sparses.set1, &mut uncompiled)?;
        while let Some(dfa_id) = uncompiled.pop() {
            for &b in &representative_bytes {
                if b.as_u8().map_or(false, |b| self.quit.contains(b)) {
                    continue;
                }
                // In many cases, the state we transition too has already been
                // computed. 'cached_state' will do the minimal amount of work
                // to check this, and if it exists, immediately return an
                // already existing state ID.
                let (next_dfa_id, is_new) =
                    self.cached_state(&mut sparses, dfa_id, b)?;
                self.dfa.add_transition(dfa_id, b, next_dfa_id);
                // If the state ID we got back is newly created, then we need
                // to compile it, so add it to our uncompiled frontier.
                if is_new {
                    uncompiled.push(next_dfa_id);
                }
            }
        }

        // A map from DFA state ID to one or more NFA match IDs. Each NFA match
        // ID corresponds to a distinct regex pattern that matches in the state
        // corresponding to the key.
        let mut matches: BTreeMap<S, Vec<PatternID>> = BTreeMap::new();
        self.cache.clear();
        for (i, state) in self.builder_states.into_iter().enumerate() {
            // This unwrap is okay, because the only other reference to a state
            // is in this builder's cache, which we cleared above. This unwrap
            // avoids copying the state's Vec<PatternID>.
            let state = Rc::try_unwrap(state).unwrap();
            if let Some(match_ids) = state.facts.into_match_pattern_ids() {
                let id = self.dfa.from_index(i);
                matches.insert(id, match_ids);
            }
        }
        // At this point, we shuffle the "special" states in the final DFA.
        // This permits a DFA's match loop to detect a match condition by
        // merely inspecting the current state's identifier, and avoids the
        // need for any additional auxiliary storage.
        self.dfa.shuffle(matches);
        Ok(())
    }

    /// Return the identifier for the next DFA state given an existing DFA
    /// state and an input byte. If the next DFA state already exists, then
    /// return its identifier from the cache. Otherwise, build the state, cache
    /// it and return its identifier.
    ///
    /// The given sparse set is used for scratch space. It must have a capacity
    /// equivalent to the total number of NFA states, but its contents are
    /// otherwise unspecified.
    ///
    /// This routine returns a boolean indicating whether a new state was
    /// built. If a new state is built, then the caller needs to add it to its
    /// frontier of uncompiled DFA states to compute transitions for.
    fn cached_state(
        &mut self,
        sparses: &mut SparseSets,
        dfa_id: S,
        b: Byte,
    ) -> Result<(S, bool), Error> {
        sparses.clear();
        // Compute the set of all reachable NFA states, including epsilons.
        let facts = self.next(sparses, dfa_id, b);
        if sparses.set2.is_empty() && !facts.is_match() {
            return Ok((dead_id(), false));
        }
        // Build a candidate state and check if it has already been built.
        let state = self.new_state(&sparses.set2, facts);
        if let Some(&cached_id) = self.cache.get(&state) {
            // Since we have a cached state, put the constructed state's
            // memory back into our scratch space, so that it can be reused.
            let _ = core::mem::replace(
                &mut self.scratch_nfa_states,
                state.nfa_states,
            );
            return Ok((cached_id, false));
        }
        // Nothing was in the cache, so add this state to the cache.
        self.add_state(state).map(|s| (s, true))
    }

    /// Compute the set of all eachable NFA states, including the full epsilon
    /// closure, from a DFA state for a single byte of input.
    fn next(&mut self, sparses: &mut SparseSets, dfa_id: S, b: Byte) -> Facts {
        sparses.clear();

        // Put the NFA state IDs into a sparse set in case we need to
        // re-compute their epsilon closure.
        for i in 0..self.state(dfa_id).nfa_states.len() {
            let nfa_id = self.state(dfa_id).nfa_states[i];
            sparses.set1.insert(nfa_id);
        }

        let facts = &self.state(dfa_id).facts;
        if !facts.look_need.is_empty() {
            let mut look_have = facts.look_have.clone();
            match b {
                Byte::U8(b'\n') => {
                    look_have.insert(Look::EndLine);
                }
                Byte::U8(_) => {}
                Byte::EOF(_) => {
                    look_have.insert(Look::EndText);
                    look_have.insert(Look::EndLine);
                }
            }
            if facts.from_word() == b.is_word_byte() {
                look_have.insert(Look::WordBoundaryUnicodeNegate);
                look_have.insert(Look::WordBoundaryAsciiNegate);
            } else {
                look_have.insert(Look::WordBoundaryUnicode);
                look_have.insert(Look::WordBoundaryAscii);
            }
            if !look_have
                .subtract(facts.look_have)
                .intersect(facts.look_need)
                .is_empty()
            {
                for nfa_id in &sparses.set1 {
                    self.epsilon_closure(nfa_id, look_have, &mut sparses.set2);
                }
                sparses.swap();
                sparses.set2.clear();
            }
        }

        let mut facts = Facts::default();
        // We only set the word byte if there's a word boundary look-around
        // anywhere in this regex. Otherwise, there's no point in bloating
        // the number of states if we don't have one.
        if self.nfa.has_word_boundary() {
            facts.set_from_word(b.is_word_byte());
        }
        // Similarly for the start-line look-around.
        if self.nfa.has_any_anchor() {
            if b.as_u8().map_or(false, |b| b == b'\n') {
                // Why only handle StartLine here and not StartText? That's
                // because StartText can only impact the starting state, which
                // is speical cased in 'add_one_start'.
                facts.look_have.insert(Look::StartLine);
            }
        }
        for nfa_id in &sparses.set1 {
            match *self.nfa.state(nfa_id) {
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
                    // by whether one of its own constituent NFA states was a
                    // match state. (And that would be done in 'new_state'.)
                    facts.set_is_match(true);
                    if self.nfa.match_len() > 1 {
                        facts.add_match_pattern_id(pid);
                    }
                    if !self.continue_past_first_match() {
                        break;
                    }
                }
                thompson::State::Range { range: ref r } => {
                    if let Some(b) = b.as_u8() {
                        if r.start <= b && b <= r.end {
                            self.epsilon_closure(
                                r.next,
                                facts.look_have,
                                &mut sparses.set2,
                            );
                        }
                    }
                }
                thompson::State::Sparse { ref ranges } => {
                    let b = match b.as_u8() {
                        None => continue,
                        Some(b) => b,
                    };
                    for r in ranges.iter() {
                        if r.start > b {
                            break;
                        } else if r.start <= b && b <= r.end {
                            self.epsilon_closure(
                                r.next,
                                facts.look_have,
                                &mut sparses.set2,
                            );
                            break;
                        }
                    }
                }
            }
        }
        facts
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
    fn epsilon_closure(
        &mut self,
        start: thompson::StateID,
        look_have: LookSet,
        set: &mut SparseSet,
    ) {
        if !self.nfa.state(start).is_epsilon() {
            set.insert(start);
            return;
        }

        self.stack.push(start);
        while let Some(mut id) = self.stack.pop() {
            loop {
                if !set.insert(id) {
                    break;
                }
                match *self.nfa.state(id) {
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
                        self.stack.extend(alternates[1..].iter().rev());
                    }
                }
            }
        }
    }

    /// Compute the set of DFA start states and return their identifiers in
    /// `dfa_state_ids`.
    ///
    /// The sparse set given is used for scratch space, and must have capacity
    /// equal to the total number of NFA states. Its contents are unspecified.
    fn add_all_starts(
        &mut self,
        sparse: &mut SparseSet,
        dfa_state_ids: &mut Vec<S>,
    ) -> Result<(), Error> {
        self.add_start_group(sparse, dfa_state_ids, None)?;
        if self.dfa.has_starts_for_each_pattern() {
            for pid in 0..self.dfa.pattern_count() {
                self.add_start_group(sparse, dfa_state_ids, Some(pid as u32))?;
            }
        }
        Ok(())
    }

    fn add_start_group(
        &mut self,
        sparse: &mut SparseSet,
        dfa_state_ids: &mut Vec<S>,
        pattern_id: Option<PatternID>,
    ) -> Result<(), Error> {
        let nfa_start = match pattern_id {
            Some(pid) => self.nfa.start_pattern(pid),
            None if self.anchored => self.nfa.start_anchored(),
            None => self.nfa.start_unanchored(),
        };

        let id = self.add_one_start(sparse, nfa_start, Start::NonWordByte)?;
        self.dfa.set_start_state(Start::NonWordByte, pattern_id, id);
        dfa_state_ids.push(id);

        if !self.nfa.has_word_boundary() {
            self.dfa.set_start_state(Start::WordByte, pattern_id, id);
        } else {
            let id = self.add_one_start(sparse, nfa_start, Start::WordByte)?;
            self.dfa.set_start_state(Start::WordByte, pattern_id, id);
            dfa_state_ids.push(id);
        }
        if !self.nfa.has_any_anchor() {
            self.dfa.set_start_state(Start::Text, pattern_id, id);
            self.dfa.set_start_state(Start::Line, pattern_id, id);
        } else {
            let id = self.add_one_start(sparse, nfa_start, Start::Text)?;
            self.dfa.set_start_state(Start::Text, pattern_id, id);
            dfa_state_ids.push(id);

            let id = self.add_one_start(sparse, nfa_start, Start::Line)?;
            self.dfa.set_start_state(Start::Line, pattern_id, id);
            dfa_state_ids.push(id);
        }

        Ok(())
    }

    fn add_one_start(
        &mut self,
        sparse: &mut SparseSet,
        nfa_start: thompson::StateID,
        start: Start,
    ) -> Result<S, Error> {
        sparse.clear();

        let facts = Facts::start(start);
        self.epsilon_closure(nfa_start, facts.look_have, sparse);
        let state = self.new_state(&sparse, facts);
        if let Some(&cached_id) = self.cache.get(&state) {
            // Since we have a cached state, put the constructed state's
            // memory back into our scratch space, so that it can be reused.
            let _ = core::mem::replace(
                &mut self.scratch_nfa_states,
                state.nfa_states,
            );
            return Ok(cached_id);
        }
        let id = self.add_state(state)?;
        Ok(id)
    }

    /// Add the given state to the DFA and make it available in the cache.
    ///
    /// The state initially has no transitions. That is, it transitions to the
    /// dead state for all possible inputs.
    fn add_state(&mut self, state: State) -> Result<S, Error> {
        let id = self.dfa.add_empty_state()?;
        if !self.quit.is_empty() {
            for b in self.quit.iter() {
                self.dfa.add_transition(id, Byte::U8(b), self.dfa.quit_id());
            }
        }
        let rstate = Rc::new(state);
        self.builder_states.push(rstate.clone());
        self.cache.insert(rstate, id);
        Ok(id)
    }

    /// Convert the given set of ordered NFA states to a DFA state.
    fn new_state(&mut self, set: &SparseSet, facts: Facts) -> State {
        let mut state = State {
            nfa_states: core::mem::replace(
                &mut self.scratch_nfa_states,
                vec![],
            ),
            facts,
        };
        state.nfa_states.clear();

        for nfa_id in set {
            match *self.nfa.state(nfa_id) {
                thompson::State::Range { .. } => {
                    state.nfa_states.push(nfa_id);
                }
                thompson::State::Sparse { .. } => {
                    state.nfa_states.push(nfa_id);
                }
                thompson::State::Look { look, .. } => {
                    state.nfa_states.push(nfa_id);
                    state.facts.look_need.insert(look);
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
                    state.nfa_states.push(nfa_id);
                    if !self.continue_past_first_match() {
                        break;
                    }
                }
            }
        }
        // If we know this state contains no look-around assertions, then
        // there's no reason to track which look-around assertions were
        // satisfied when this state was created.
        if state.facts.look_need.is_empty() {
            state.facts.look_have = LookSet::empty();
        }
        state
    }

    /// Return a reference to this builder's representation of the state with
    /// the given identifier. A builder's representation of a state contains
    /// the IDs of its constituent NFA states.
    fn state(&self, id: S) -> &State {
        &self.builder_states[self.dfa.to_index(id)]
    }

    /// Returns true if and only if the DFA should be built to include all
    /// possible match states.
    ///
    /// Generally speaking, this is false when we want to impose some kind of
    /// match priority, like for leftmost-first.
    fn continue_past_first_match(&self) -> bool {
        self.match_kind.continue_past_first_match()
    }
}

/// An intermediate representation for a DFA state during determinization.
///
/// This representation is used as a key in a map from states to their
/// identifiers.
#[derive(Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
struct State {
    /// An ordered sequence of NFA states that make up this DFA state.
    nfa_states: Vec<thompson::StateID>,
    /// A collection of "facts" about this state that, in addition to the NFA
    /// state IDs above, contributes to this state's identity.
    facts: Facts,
}

impl State {
    /// Create a new empty dead state.
    fn dead() -> State {
        State { nfa_states: vec![], facts: Facts::default() }
    }
}

/// A collection of "facts" or metadata about a DFA state. These facts include
/// whether the state is matching or not in addition to the actual pattern
/// IDs corresponding to that match. The facts also include information about
/// look-around assertions, if relevant.
///
/// In addition to a DFA state's constituent NFA state IDs, these facts
/// comprise the identity of a DFA state. That is, any two DFA states with
/// unequal NFA state IDs or unequal facts is considered distinct. (This does
/// not necessarily imply they are literally distinct from a theoretical
/// perspective, but it's what we do here. Minimization, a separate thing,
/// handles the fully general case.)
///
/// For that reason, we are careful to only set facts when they are needed.
/// For example, one fact says whether the state was generated by a transition
/// corresponding to a "word" byte, and this fact is used to determine whether
/// a word boundary assertion is satisfied or not. But of course, if the NFA
/// contains no word boundary assertions, then there is no reason to track
/// whether transitions are "word" bytes or not. If we did, we would wind up
/// generating more DFA states than we have too.
///
/// Similarly, if 'look_need' is empty, then we forcefully set 'look_have'
/// to be empty. Namely, if a DFA state contains no NFA states that are
/// look-around assertions, then there is no reason to track which assertions
/// were true when the DFA state was created.
#[derive(Debug, Default, Eq, Hash, PartialEq, PartialOrd, Ord)]
struct Facts {
    /// A bitfield of flags about a DFA state.
    ///
    /// Bit 0 corresponds to whether this is a match state. (And since match
    /// states are delayed by 1 byte, this means this state was created by
    /// transitioning from a DFA state whose set of NFA states included a match
    /// state.)
    ///
    /// Bit 1 corresponds to whether this DFA state was created from a
    /// transition over an ASCII "word" byte. Note that this is only ever set
    /// if the NFA has a word boundary assertion. Otherwise, this bit is always
    /// unset since it would otherwise bloat the size of the DFA for no reason.
    /// This information is used to determine whether a particular transition
    /// satisfies a "word boundary" assertion.
    bools: u8,
    /// An ordered list of pattern IDs corresponding to the NFA match states
    /// that produced this DFA state.
    ///
    /// Pattern IDs are only explicitly tracked if the NFA contains more than
    /// 1 pattern. Otherwise, this is always empty since the only possible
    /// matching pattern ID is `0`.
    ///
    /// This must only be non-empty when Bit 0 of 'bools' is set.
    match_pattern_ids: Vec<PatternID>,
    /// The set of look-around assertions that were true when this state was
    /// created.
    ///
    /// Like look_need, this is also used to gate the re-computation of the
    /// epsilon closure in 'next' when determining the next transition. Namely,
    /// an assertion that is satisfied is only considered "new" if it is not
    /// already in look_have. Why? Because if the assertion was satisfied
    /// when the state was created, then the NFA states inside that DFA state
    /// already account for that assertion being satisfied since that was the
    /// context in which the epsilon closure was originally computed.
    look_have: LookSet,
    /// The set of look-around assertions contained within this DFA state, where
    /// a look-around assertion is present if and only if one of this DFA
    /// state's NFA states corresponds to that look-around assertion.
    ///
    /// In essence, this is an efficient representation of the "interesting"
    /// look-around assertions for this state. This is used to gate the
    /// re-computation of the epsilon closure of the preceding state in
    /// 'next'. That is, the epsilon closure is only re-computed if there are
    /// new AND relevant look-around assertions.
    look_need: LookSet,
}

impl Facts {
    /// Compute the set of facts possible given the type of starting
    /// conditions.
    fn start(start: Start) -> Facts {
        let mut facts = Facts::default();
        match start {
            Start::NonWordByte => {}
            Start::WordByte => {
                facts.set_from_word(true);
            }
            Start::Text => {
                facts.look_have.insert(Look::StartText);
                facts.look_have.insert(Look::StartLine);
            }
            Start::Line => {
                facts.look_have.insert(Look::StartLine);
            }
        }
        facts
    }

    // Get and set whether this state is a match state or not.
    define_bool!(0, is_match, set_is_match);

    // Get and set whether this state was created from a transition over an
    // ASCII "word" byte.
    define_bool!(1, from_word, set_from_word);

    /// Add the given pattern ID to this state.
    ///
    /// Callers must ensure that this state is marked as a match state.
    fn add_match_pattern_id(&mut self, pid: PatternID) {
        self.match_pattern_ids.push(pid);
    }

    /// Return the match pattern IDs for this state, but only if this state
    /// is a matching state. Otherwise, this returns None.
    fn into_match_pattern_ids(self) -> Option<Vec<PatternID>> {
        if !self.is_match() {
            return None;
        }
        Some(if self.match_pattern_ids.is_empty() {
            vec![0]
        } else {
            self.match_pattern_ids
        })
    }
}
