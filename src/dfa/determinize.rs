use alloc::{
    collections::BTreeMap,
    vec::{self, Vec},
};

use crate::{
    dfa::{dense, Error, DEAD},
    nfa::thompson,
    util::{
        self,
        alphabet::{self, ByteSet},
        determinize::{Start, State, StateBuilderEmpty, StateBuilderNFA},
        id::{PatternID, StateID},
        matchtypes::MatchKind,
        sparse_set::{SparseSet, SparseSets},
    },
};

/// A builder for configuring and running a DFA determinizer.
#[derive(Clone, Debug)]
pub(crate) struct Config {
    anchored: bool,
    match_kind: MatchKind,
    quit: ByteSet,
}

impl Config {
    /// Create a new default config for a determinizer. The determinizer may be
    /// configured before calling `run`.
    pub fn new() -> Config {
        Config {
            anchored: false,
            match_kind: MatchKind::LeftmostFirst,
            quit: ByteSet::empty(),
        }
    }

    /// Run determinization on the given NFA and write the resulting DFA into
    /// the one given. The DFA given should be initialized but otherwise empty.
    /// "Initialized" means that it is setup to handle the NFA's byte classes,
    /// number of patterns and whether to build start states for each pattern.
    pub fn run(
        &self,
        nfa: &thompson::NFA,
        dfa: &mut dense::OwnedDFA,
    ) -> Result<(), Error> {
        let dead = State::dead();
        let quit = State::dead();
        let mut cache = StateMap::default();
        // We only insert the dead state here since its representation is
        // identical to the quit state. And we never want anything pointing
        // to the quit state other than specific transitions derived from the
        // determinizer's configured "quit" bytes.
        cache.insert(dead.clone(), DEAD);

        Runner {
            config: self.clone(),
            nfa,
            dfa,
            builder_states: vec![dead, quit],
            cache,
            memory_usage_state: 0,
            stack: vec![],
            scratch_state_builder: StateBuilderEmpty::new(),
        }
        .run()
    }

    /// Whether to build an anchored DFA or not. When disabled (the default),
    /// the unanchored prefix from the NFA is used to start the DFA. Otherwise,
    /// the anchored start state of the NFA is used to start the DFA.
    pub fn anchored(&mut self, yes: bool) -> &mut Config {
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
    pub fn match_kind(&mut self, kind: MatchKind) -> &mut Config {
        self.match_kind = kind;
        self
    }

    /// The set of bytes to use that will cause the DFA to enter a quit state,
    /// stop searching and return an error. By default, this is empty.
    pub fn quit(&mut self, set: ByteSet) -> &mut Config {
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
/// The lifetime variable `'a` refers to the lifetime of the NFA or DFA,
/// whichever is shorter.
#[derive(Debug)]
struct Runner<'a> {
    /// The configuration used to initialize determinization.
    config: Config,
    /// The NFA we're converting into a DFA.
    nfa: &'a thompson::NFA,
    /// The DFA we're building.
    dfa: &'a mut dense::OwnedDFA,
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
    /// constant random access by a DFA state's ID. The cache map below, on
    /// the other hand, provides a quick way of searching for identical DFA
    /// states by using the DFA state as a key in the map. Of course, we use
    /// reference counting to avoid actually duplicating the state's data
    /// itself. (Although this has never been benchmarked.) Note that the cache
    /// map does not give us full minimization; it just lets us avoid some very
    /// obvious redundant states.
    ///
    /// Note that the index into this Vec isn't quite the DFA's state ID.
    /// Rather, it's just an index. To get the state ID, you have to multiply
    /// it by the DFA's stride. That's done by self.dfa.from_index. And the
    /// inverse is self.dfa.to_index.
    ///
    /// Moreover, DFA states don't usually retain the IDs assigned to them
    /// by their position in this Vec. After determinization completes,
    /// states are shuffled around to support other optimizations. See the
    /// sibling 'special' module for more details on that. (The reason for
    /// mentioning this is that if you print out the DFA for debugging during
    /// determinization, and then print out the final DFA after it is fully
    /// built, then the state IDs likely won't match up.)
    builder_states: Vec<State>,
    /// A cache of DFA states that already exist and can be easily looked up
    /// via ordered sets of NFA states.
    ///
    /// See `builder_states` docs for why we store states in two different
    /// ways.
    cache: StateMap,
    /// The memory usage, in bytes, used by builder_states and cache. We track
    /// this as new states are added since states use a variable amount of
    /// heap. Tracking this as we add states makes it possible to compute the
    /// total amount of memory used by the determinizer in constant time.
    memory_usage_state: usize,
    /// Scratch space for a stack of NFA states to visit, for depth first
    /// visiting without recursion.
    stack: Vec<StateID>,
    /// Scratch space for storing an ordered sequence of NFA states, for
    /// amortizing allocation. This is principally useful for when we avoid
    /// adding a new DFA state since it already exists. In order to detect this
    /// case though, we still need an ordered set of NFA state IDs. So we use
    /// this space to stage that ordered set before we know whether we need to
    /// create a new DFA state or not.
    scratch_state_builder: StateBuilderEmpty,
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

impl<'a> Runner<'a> {
    /// Build the DFA. If there was a problem constructing the DFA (e.g., if
    /// the chosen state identifier representation is too small), then an error
    /// is returned.
    fn run(mut self) -> Result<(), Error> {
        if self.nfa.has_word_boundary_unicode()
            && !self.config.quit.contains_range(0x80, 0xFF)
        {
            return Err(Error::unsupported_dfa_word_boundary_unicode());
        }

        // A sequence of "representative" bytes drawn from each equivalence
        // class. These representative bytes are fed to the NFA to compute
        // state transitions. This allows us to avoid re-computing state
        // transitions for bytes that are guaranteed to produce identical
        // results.
        let representative_bytes: Vec<alphabet::Unit> =
            self.dfa.byte_classes().representatives().collect();
        // A pair of sparse sets for tracking ordered sets of NFA state IDs.
        // These are reused throughout determinization. A bounded sparse set
        // gives us constant time insertion, membership testing and clearing.
        let mut sparses = SparseSets::new(self.nfa.len());
        // The set of all DFA state IDs that still need to have their
        // transitions set. We start by seeding this will all starting states.
        let mut uncompiled = vec![];
        self.add_all_starts(&mut sparses.set1, &mut uncompiled)?;
        while let Some(dfa_id) = uncompiled.pop() {
            for &b in &representative_bytes {
                if b.as_u8().map_or(false, |b| self.config.quit.contains(b)) {
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

        trace!(
            "determinization complete, memory usage: {}, dense DFA size: {}",
            self.memory_usage(),
            self.dfa.memory_usage(),
        );

        // A map from DFA state ID to one or more NFA match IDs. Each NFA match
        // ID corresponds to a distinct regex pattern that matches in the state
        // corresponding to the key.
        let mut matches: BTreeMap<StateID, Vec<PatternID>> = BTreeMap::new();
        self.cache.clear();
        #[allow(unused_variables)]
        let mut total_pat_count = 0;
        for (i, state) in self.builder_states.into_iter().enumerate() {
            if let Some(pat_ids) = state.match_pattern_ids() {
                let id = self.dfa.from_index(i);
                total_pat_count += pat_ids.len();
                matches.insert(id, pat_ids);
            }
        }
        log! {
            use core::mem::size_of;
            let per_elem = size_of::<StateID>() + size_of::<Vec<PatternID>>();
            let pats = total_pat_count * size_of::<PatternID>();
            let mem = (matches.len() * per_elem) + pats;
            log::trace!("matches map built, memory usage: {}", mem);
        }
        // At this point, we shuffle the "special" states in the final DFA.
        // This permits a DFA's match loop to detect a match condition by
        // merely inspecting the current state's identifier, and avoids the
        // need for any additional auxiliary storage.
        self.dfa.shuffle(matches)?;
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
        dfa_id: StateID,
        unit: alphabet::Unit,
    ) -> Result<(StateID, bool), Error> {
        sparses.clear();
        // Compute the set of all reachable NFA states, including epsilons.
        let empty_builder = self.get_state_builder();
        let builder = util::determinize::next(
            self.nfa,
            self.config.match_kind,
            sparses,
            &mut self.stack,
            &self.builder_states[self.dfa.to_index(dfa_id)],
            unit,
            empty_builder,
        );
        self.maybe_add_state(builder)
    }

    /// Compute the set of DFA start states and add their identifiers in
    /// 'dfa_state_ids' (no duplicates are added).
    ///
    /// The sparse set given is used for scratch space, and must have capacity
    /// equal to the total number of NFA states. Its value given does not
    /// matter and its value when this function returns is unspecified.
    fn add_all_starts(
        &mut self,
        sparse: &mut SparseSet,
        dfa_state_ids: &mut Vec<StateID>,
    ) -> Result<(), Error> {
        // Always add the (possibly unanchored) start states for matching any
        // of the patterns in this DFA.
        self.add_start_group(sparse, None, dfa_state_ids)?;
        // We only need to compute anchored start states for each pattern if it
        // was requested to do so.
        if self.dfa.has_starts_for_each_pattern() {
            for pid in PatternID::iter(self.dfa.pattern_count()) {
                self.add_start_group(sparse, Some(pid), dfa_state_ids)?;
            }
        }
        Ok(())
    }

    /// Add a group of start states for the given match pattern ID. Any new
    /// DFA states added are pushed on to 'dfa_state_ids'. (No duplicates are
    /// pushed.) Also, 'sparse' is used as scratch space; its value given does
    /// not matter and its value when this function returns is unspecified.
    ///
    /// When pattern_id is None, then this will compile a group of unanchored
    /// start states (if the DFA is unanchored). When the pattern_id is
    /// present, then this will compile a group of anchored start states that
    /// only match the given pattern.
    fn add_start_group(
        &mut self,
        sparse: &mut SparseSet,
        pattern_id: Option<PatternID>,
        dfa_state_ids: &mut Vec<StateID>,
    ) -> Result<(), Error> {
        let nfa_start = match pattern_id {
            Some(pid) => self.nfa.start_pattern(pid),
            None if self.config.anchored => self.nfa.start_anchored(),
            None => self.nfa.start_unanchored(),
        };

        // When compiling start states, we're careful not to build additional
        // states that aren't necessary. For example, if the NFA has no word
        // boundary assertion, then there's no reason to have distinct start
        // states for 'NonWordByte' and 'WordByte' starting configurations.
        // Instead, the 'WordByte' starting configuration can just point
        // directly to the start state for the 'NonWordByte' config.

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

    /// Add a new DFA start state corresponding to the given starting NFA
    /// state, and the starting search configuration. (The starting search
    /// configuration essentially tells us which look-behind assertions are
    /// true for this particular state.)
    ///
    /// The 'sparse' set given can have unspecified contents. It is used as
    /// scratch space to store the epislon closure of NFA states (beginning at
    /// the given NFA start state).
    fn add_one_start(
        &mut self,
        sparse: &mut SparseSet,
        nfa_start: StateID,
        start: Start,
    ) -> Result<StateID, Error> {
        sparse.clear();

        // Compute the look-behind assertions that are true in this starting
        // configuration, and the determine the epsilon closure. While
        // computing the epsilon closure, we only follow condiional epsilon
        // transitions that satisfy the look-behind assertions in 'facts'.
        let mut builder_matches = self.get_state_builder().into_matches();
        start.set_state(&mut builder_matches);
        util::determinize::epsilon_closure(
            self.nfa,
            nfa_start,
            *builder_matches.look_have(),
            &mut self.stack,
            sparse,
        );
        let mut builder = builder_matches.into_nfa();
        util::determinize::add_nfa_states(&self.nfa, &sparse, &mut builder);
        self.maybe_add_state(builder).map(|(sid, _)| sid)
    }

    /// Adds the given state to the DFA being built depending on whether it
    /// already exists in this determinizer's cache.
    ///
    /// If it does exist, then the memory used by 'state' is put back into the
    /// determinizer and the previously created state's ID is returned. (Along
    /// with 'false', indicating that no new state was added.)
    ///
    /// If it does not exist, then the state is added to the DFA being built
    /// and a fresh ID is allocated (if ID allocation fails, then an error is
    /// returned) and returned. (Along with 'true', indicating that a new state
    /// was added.)
    fn maybe_add_state(
        &mut self,
        builder: StateBuilderNFA,
    ) -> Result<(StateID, bool), Error> {
        if let Some(&cached_id) = self.cache.get(builder.as_bytes()) {
            // Since we have a cached state, put the constructed state's
            // memory back into our scratch space, so that it can be reused.
            self.put_state_builder(builder);
            return Ok((cached_id, false));
        }
        self.add_state(builder).map(|sid| (sid, true))
    }

    /// Add the given state to the DFA and make it available in the cache.
    ///
    /// The state initially has no transitions. That is, it transitions to the
    /// dead state for all possible inputs, and transitions to the quit state
    /// for all quit bytes.
    ///
    /// If adding the state would exceed the maximum value for StateID, then an
    /// error is returned.
    fn add_state(
        &mut self,
        builder: StateBuilderNFA,
    ) -> Result<StateID, Error> {
        let id = self.dfa.add_empty_state()?;
        if !self.config.quit.is_empty() {
            for b in self.config.quit.iter() {
                self.dfa.add_transition(
                    id,
                    alphabet::Unit::u8(b),
                    self.dfa.quit_id(),
                );
            }
        }
        let state = builder.to_state();
        // States use reference counting internally, so we only need to count
        // their memroy usage once.
        self.memory_usage_state += state.memory_usage();
        self.builder_states.push(state.clone());
        self.cache.insert(state, id);
        self.put_state_builder(builder);
        Ok(id)
    }

    /// Returns a state builder from this determinizer that might have existing
    /// capacity. This helps avoid allocs in cases where a state is built that
    /// turns out to already be cached.
    ///
    /// Callers must put the state builder back with 'put_state_builder',
    /// otherwise the allocation reuse won't work.
    fn get_state_builder(&mut self) -> StateBuilderEmpty {
        core::mem::replace(
            &mut self.scratch_state_builder,
            StateBuilderEmpty::new(),
        )
    }

    /// Puts the given state builder back into this determinizer for reuse.
    ///
    /// Note that building a 'State' from a builder always creates a new
    /// alloc, so callers should always put the builder back.
    fn put_state_builder(&mut self, builder: StateBuilderNFA) {
        let _ = core::mem::replace(
            &mut self.scratch_state_builder,
            builder.clear(),
        );
    }

    /// Return the memory usage, in bytes, of this determinizer at the current
    /// point in time. This does not include memory used by the NFA or the
    /// dense DFA itself.
    #[cfg(feature = "logging")]
    fn memory_usage(&self) -> usize {
        use core::mem::size_of;

        self.builder_states.len() * size_of::<State>()
        // Maps likely use more memory than this, but it's probably close.
        + self.cache.len() * (size_of::<State>() + size_of::<StateID>())
        + self.memory_usage_state
        + self.stack.capacity() * size_of::<StateID>()
        + self.scratch_state_builder.capacity()
    }
}
