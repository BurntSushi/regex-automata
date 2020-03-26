use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::mem;
use std::rc::Rc;

use crate::classes::{Byte, ByteClasses, ByteSet};
use crate::dfa::automaton::Start;
use crate::dfa::{dense, Error};
use crate::nfa::thompson::{self, Look};
use crate::sparse_set::SparseSet;
use crate::state_id::{dead_id, StateID};
use crate::{MatchKind, PatternID};

#[derive(Clone, Debug)]
pub(crate) struct Determinizer {
    anchored: bool,
    match_kind: MatchKind,
    quit: ByteSet,
}

impl Determinizer {
    pub fn new() -> Determinizer {
        Determinizer {
            anchored: false,
            match_kind: MatchKind::LeftmostFirst,
            quit: ByteSet::empty(),
        }
    }

    pub fn run<S: StateID>(
        &self,
        nfa: &thompson::NFA,
        dfa: &mut dense::OwnedDFA<S>,
    ) -> Result<(), Error> {
        let dead = Rc::new(State::dead());
        let quit = Rc::new(State::dead());
        let mut cache = HashMap::default();
        cache.insert(dead.clone(), dead_id());
        cache.insert(quit.clone(), dfa.quit_id());

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

    pub fn anchored(&mut self, yes: bool) -> &mut Determinizer {
        self.anchored = yes;
        self
    }

    pub fn match_kind(&mut self, kind: MatchKind) -> &mut Determinizer {
        self.match_kind = kind;
        self
    }

    pub fn quit(&mut self, set: ByteSet) -> &mut Determinizer {
        self.quit = set;
        self
    }
}

/// A actual implementation of determinization that converts an NFA to a DFA
/// through powerset construction.
///
/// This determinizer follows the typical powerset construction, where each
/// DFA state is comprised of one or more NFA states. In the worst case, there
/// is one DFA state for every possible combination of NFA states. In practice,
/// this only happens in certain conditions, typically when there are bounded
/// repetitions.
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
    /// states, along with a flag indicating whether the state is a match
    /// state or not.
    ///
    /// This is never empty. The first state is always a dummy state such that
    /// a state id == 0 corresponds to a dead state.
    builder_states: Vec<Rc<State>>,
    /// A cache of DFA states that already exist and can be easily looked up
    /// via ordered sets of NFA states.
    cache: HashMap<Rc<State>, S>,
    /// Scratch space for a stack of NFA states to visit, for depth first
    /// visiting without recursion.
    stack: Vec<thompson::StateID>,
    /// Scratch space for storing an ordered sequence of NFA states, for
    /// amortizing allocation.
    scratch_nfa_states: Vec<thompson::StateID>,
    /// Whether to build an anchored DFA or not.
    anchored: bool,
    /// Match semantics for this DFA.
    match_kind: MatchKind,
    /// Bytes on which the DFA should halt the search and report an error.
    quit: ByteSet,
}

/// An intermediate representation for a DFA state during determinization.
#[derive(Debug, Eq, Hash, PartialEq)]
struct State {
    /// An ordered sequence of NFA states that make up this DFA state.
    nfa_states: Vec<thompson::StateID>,
    /// Whether this state is a match state or not. Note that when true,
    /// this does NOT mean that this contains an NFA match state. Namely,
    /// this represents the single source of truth about whether a DFA state
    /// is matching or not. In particular, since the DFA delays matches by
    /// a single byte (to handle $ and \b), the actual match state typically
    /// comes immediately after a state containing an NFA match state.
    facts: Facts,
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

        let representative_bytes: Vec<Byte> =
            self.dfa.byte_classes().representatives().collect();
        let mut sparses = self.new_sparse_sets();
        let mut uncompiled = vec![];
        self.add_starts(&mut sparses.cur, &mut uncompiled)?;
        while let Some(dfa_id) = uncompiled.pop() {
            for &b in &representative_bytes {
                if b.as_u8().map_or(false, |b| self.quit.contains(b)) {
                    continue;
                }
                let (next_dfa_id, is_new) =
                    self.cached_state(&mut sparses, dfa_id, b)?;
                self.dfa.add_transition(dfa_id, b, next_dfa_id);
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
            if let Some(match_ids) = state.facts.state.matches.into_vec() {
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
        if sparses.next.is_empty() && !facts.state.is_match() {
            return Ok((dead_id(), false));
        }
        // Build a candidate state and check if it has already been built.
        let state = self.new_state(&sparses.next, facts);
        if let Some(&cached_id) = self.cache.get(&state) {
            // Since we have a cached state, put the constructed state's
            // memory back into our scratch space, so that it can be reused.
            mem::replace(&mut self.scratch_nfa_states, state.nfa_states);
            return Ok((cached_id, false));
        }
        // Nothing was in the cache, so add this state to the cache.
        self.add_state(state).map(|s| (s, true))
    }

    /// Compute the set of all eachable NFA states, including the full epsilon
    /// closure, from a DFA state for a single byte of input.
    fn next(&mut self, sparses: &mut SparseSets, dfa_id: S, b: Byte) -> Facts {
        sparses.clear();

        for i in 0..self.state(dfa_id).nfa_states.len() {
            let nfa_id = self.state(dfa_id).nfa_states[i];
            sparses.cur.insert(nfa_id);
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
            if facts.state.from_word() == b.is_word_byte() {
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
                for &nfa_id in &sparses.cur {
                    self.epsilon_closure(nfa_id, look_have, &mut sparses.next);
                }
                sparses.swap();
                sparses.next.clear();
            }
        }

        let mut facts = Facts::default();
        if b.as_u8().map_or(false, |b| b == b'\n') {
            facts.look_have.insert(Look::StartLine);
        }
        // We only set the word byte if there's a word boundary look-around
        // anywhere in this regex. Otherwise, there's no point in bloating
        // the number of states if we don't have one.
        if self.nfa.has_word_boundary() {
            if b.is_word_byte() {
                facts.state.set_from_word(true);
            }
        }
        for &nfa_id in &sparses.cur {
            match *self.nfa.state(nfa_id) {
                thompson::State::Union { .. }
                | thompson::State::Fail
                | thompson::State::Look { .. } => {}
                thompson::State::Match(mid) => {
                    // TODO: Make this work. Currently this fails with
                    // MatchStates serialization. Think of something elegant.
                    // if self.nfa.match_len() <= 1 {
                    // facts.state.matches = Matches::One;
                    // } else {
                    facts.state.matches.add(mid);
                    // }
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
                                &mut sparses.next,
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
                                &mut sparses.next,
                            );
                            break;
                        }
                    }
                }
            }
        }
        facts
    }

    /// Compute the epsilon closure for the given NFA state.
    fn epsilon_closure(
        &mut self,
        start: thompson::StateID,
        look_have: LookSet,
        set: &mut SparseSet,
    ) {
        if !self.nfa.state(start).is_epsilon() {
            if !set.contains(start) {
                set.insert(start);
            }
            return;
        }

        self.stack.push(start);
        while let Some(mut id) = self.stack.pop() {
            loop {
                if set.contains(id) {
                    break;
                }
                set.insert(id);
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

    /// Compute the initial DFA state and return its identifier.
    ///
    /// The sparse set given is used for scratch space, and must have capacity
    /// equal to the total number of NFA states. Its contents are unspecified.
    fn add_starts(
        &mut self,
        sparse: &mut SparseSet,
        dfa_state_ids: &mut Vec<S>,
    ) -> Result<(), Error> {
        let nfa_start = if self.anchored {
            self.nfa.start_anchored()
        } else {
            self.nfa.start_unanchored()
        };

        let id = self.add_start(sparse, nfa_start, Start::NonWordByte)?;
        self.dfa.set_start_state(Start::NonWordByte, id);
        dfa_state_ids.push(id);

        if !self.nfa.has_word_boundary() {
            self.dfa.set_start_state(Start::WordByte, id);
        } else {
            let id = self.add_start(sparse, nfa_start, Start::WordByte)?;
            self.dfa.set_start_state(Start::WordByte, id);
            dfa_state_ids.push(id);
        }
        if !self.nfa.has_any_anchor() {
            self.dfa.set_start_state(Start::Text, id);
            self.dfa.set_start_state(Start::Line, id);
        } else {
            let id = self.add_start(sparse, nfa_start, Start::Text)?;
            self.dfa.set_start_state(Start::Text, id);
            dfa_state_ids.push(id);

            let id = self.add_start(sparse, nfa_start, Start::Line)?;
            self.dfa.set_start_state(Start::Line, id);
            dfa_state_ids.push(id);
        }

        Ok(())
    }

    fn add_start(
        &mut self,
        sparse: &mut SparseSet,
        nfa_start: thompson::StateID,
        start: Start,
    ) -> Result<S, Error> {
        sparse.clear();

        let facts = Facts::start(start);
        self.epsilon_closure(nfa_start, facts.look_have, sparse);
        let state = self.new_state(&sparse, facts);
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
            nfa_states: mem::replace(&mut self.scratch_nfa_states, vec![]),
            facts,
        };
        state.nfa_states.clear();

        for &id in set {
            match *self.nfa.state(id) {
                thompson::State::Range { .. } => {
                    state.nfa_states.push(id);
                }
                thompson::State::Sparse { .. } => {
                    state.nfa_states.push(id);
                }
                thompson::State::Look { look, .. } => {
                    state.nfa_states.push(id);
                    state.facts.look_need.insert(look);
                }
                thompson::State::Union { .. } => {
                    state.nfa_states.push(id);
                }
                thompson::State::Fail => {
                    break;
                }
                thompson::State::Match(_) => {
                    state.nfa_states.push(id);
                    if !self.continue_past_first_match() {
                        break;
                    }
                }
            }
        }
        if state.facts.look_need.is_empty() {
            state.facts.look_have = LookSet::default();
        }
        state
    }

    /// Return a reference to this builder's representation of the state with
    /// the given identifier. A builder's representation of a state contains
    /// the IDs of its constituent NFA states.
    fn state(&self, id: S) -> &State {
        &self.builder_states[self.dfa.to_index(id)]
    }

    fn continue_past_first_match(&self) -> bool {
        self.match_kind.continue_past_first_match()
    }

    /// Create a new pair of sparse sets with enough capacity to hold all NFA
    /// states.
    fn new_sparse_sets(&self) -> SparseSets {
        SparseSets {
            cur: SparseSet::new(self.nfa.len()),
            next: SparseSet::new(self.nfa.len()),
        }
    }

    /// Return an empty set of matches for a state that is a match state.
    fn new_matches(&self) -> Matches {
        if self.nfa.match_len() <= 1 {
            Matches::One
        } else {
            Matches::Many(vec![])
        }
    }
}

impl State {
    /// Create a new empty dead state.
    fn dead() -> State {
        State { nfa_states: vec![], facts: Facts::default() }
    }

    fn is_match(&self) -> bool {
        self.facts.state.matches.is_match()
    }
}

#[derive(Debug)]
struct SparseSets {
    cur: SparseSet,
    next: SparseSet,
}

impl SparseSets {
    fn clear(&mut self) {
        self.cur.clear();
        self.next.clear();
    }

    fn swap(&mut self) {
        std::mem::swap(&mut self.cur, &mut self.next);
    }
}

#[derive(Debug, Default, Eq, Hash, PartialEq)]
struct Facts {
    state: StateFacts,
    look_need: LookSet,
    look_have: LookSet,
}

impl Facts {
    fn start(start: Start) -> Facts {
        let mut state = StateFacts::default();
        let mut look_have = LookSet::default();
        match start {
            Start::NonWordByte => {}
            Start::WordByte => {
                state.set_from_word(true);
            }
            Start::Text => {
                look_have.insert(Look::StartText);
                look_have.insert(Look::StartLine);
            }
            Start::Line => {
                look_have.insert(Look::StartLine);
            }
        }
        Facts { state, look_need: LookSet::default(), look_have }
    }
}

/// Various facts about a builder-DFA state.
///
/// These make up the state's identity, such that two states with the same
/// set of NFA states but different facts are considered distinct states.
#[derive(Default, Eq, Hash, PartialEq)]
struct StateFacts {
    bools: u8,
    matches: Matches,
}

impl StateFacts {
    define_bool!(0, from_word, set_from_word);

    fn is_match(&self) -> bool {
        self.matches.is_match()
    }
}

impl std::fmt::Debug for StateFacts {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("StateFacts")
            .field("from_word", &self.from_word())
            .field("matches", &self.matches)
            .finish()
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum Matches {
    /// This variant is always used for non-matching states.
    None,
    /// This variant is always used for matching states when the total number
    /// of regexes in the NFA is 1.
    One,
    /// This variant is always used for matching states when the total number
    /// of regexes in the NFA is greater than 1.
    Many(Vec<PatternID>),
}

impl Default for Matches {
    fn default() -> Matches {
        Matches::None
    }
}

impl Matches {
    fn add(&mut self, pid: PatternID) {
        match *self {
            Matches::None => {
                *self = Matches::Many(vec![pid]);
            }
            Matches::One => {
                panic!("cannot add NFA match ID when compiling a single regex")
            }
            Matches::Many(ref mut many) => {
                many.push(pid);
            }
        }
    }

    fn is_match(&self) -> bool {
        match *self {
            Matches::None => false,
            Matches::One | Matches::Many(_) => true,
        }
    }

    fn into_vec(self) -> Option<Vec<PatternID>> {
        match self {
            Matches::None => None,
            Matches::One => Some(vec![]),
            Matches::Many(pids) => Some(pids),
        }
    }
}

/// TODO
///
/// Various facts about a position in the input with respect to look-around
/// conditions. These facts are used as a filter when following epsilon
/// transitions. That is, only epsilon transitions (that also have look-around
/// conditions attached to them) satisfying these facts are followed.
#[derive(Clone, Copy, Default, Eq, Hash, PartialEq)]
struct LookSet {
    fields: u8,
}

impl LookSet {
    fn is_empty(&self) -> bool {
        self.fields == 0
    }

    fn subtract(&self, other: LookSet) -> LookSet {
        LookSet { fields: self.fields & !other.fields }
    }

    fn intersect(&self, other: LookSet) -> LookSet {
        LookSet { fields: self.fields & other.fields }
    }

    fn insert(&mut self, look: Look) {
        self.fields |= look.as_bit_field();
    }

    fn remove(&mut self, look: Look) {
        self.fields &= !look.as_bit_field();
    }

    fn contains(&self, look: Look) -> bool {
        Look::bit_set_contains(self.fields, look)
    }
}

impl std::fmt::Debug for LookSet {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut members = vec![];
        for i in 0..8 {
            let look = match Look::from_u8(1 << i) {
                None => continue,
                Some(look) => look,
            };
            if self.contains(look) {
                members.push(look);
            }
        }
        f.debug_tuple("LookSet").field(&members).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_facts() {
        let mut f = StateFacts::default();
        assert!(!f.is_match());
        assert!(!f.from_word());

        f.set_from_word(true);
        assert!(f.from_word());
        f.set_from_word(false);
        assert!(!f.from_word());
    }

    #[test]
    fn look_set() {
        let mut f = LookSet::default();
        assert!(!f.contains(Look::StartText));
        assert!(!f.contains(Look::EndText));
        assert!(!f.contains(Look::StartLine));
        assert!(!f.contains(Look::EndLine));
        assert!(!f.contains(Look::WordBoundaryUnicode));
        assert!(!f.contains(Look::WordBoundaryUnicodeNegate));
        assert!(!f.contains(Look::WordBoundaryAscii));
        assert!(!f.contains(Look::WordBoundaryAsciiNegate));

        f.insert(Look::StartText);
        assert!(f.contains(Look::StartText));
        f.remove(Look::StartText);
        assert!(!f.contains(Look::StartText));

        f.insert(Look::EndText);
        assert!(f.contains(Look::EndText));
        f.remove(Look::EndText);
        assert!(!f.contains(Look::EndText));

        f.insert(Look::StartLine);
        assert!(f.contains(Look::StartLine));
        f.remove(Look::StartLine);
        assert!(!f.contains(Look::StartLine));

        f.insert(Look::EndLine);
        assert!(f.contains(Look::EndLine));
        f.remove(Look::EndLine);
        assert!(!f.contains(Look::EndLine));

        f.insert(Look::WordBoundaryUnicode);
        assert!(f.contains(Look::WordBoundaryUnicode));
        f.remove(Look::WordBoundaryUnicode);
        assert!(!f.contains(Look::WordBoundaryUnicode));

        f.insert(Look::WordBoundaryUnicodeNegate);
        assert!(f.contains(Look::WordBoundaryUnicodeNegate));
        f.remove(Look::WordBoundaryUnicodeNegate);
        assert!(!f.contains(Look::WordBoundaryUnicodeNegate));

        f.insert(Look::WordBoundaryAscii);
        assert!(f.contains(Look::WordBoundaryAscii));
        f.remove(Look::WordBoundaryAscii);
        assert!(!f.contains(Look::WordBoundaryAscii));

        f.insert(Look::WordBoundaryAsciiNegate);
        assert!(f.contains(Look::WordBoundaryAsciiNegate));
        f.remove(Look::WordBoundaryAsciiNegate);
        assert!(!f.contains(Look::WordBoundaryAsciiNegate));
    }
}
