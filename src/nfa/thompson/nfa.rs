use core::{cmp, convert::TryFrom, fmt, mem, ops::Range};

use alloc::{boxed::Box, format, string::String, sync::Arc, vec, vec::Vec};

use crate::{
    nfa::thompson::{builder::Builder, compiler::Config, error::Error},
    util::{
        alphabet::{self, ByteClassSet},
        decode_last_utf8, decode_utf8,
        id::{IteratorIDExt, PatternID, PatternIDIter, StateID},
        is_word_byte, is_word_char_fwd, is_word_char_rev,
    },
};

#[derive(Clone)]
pub struct NFA(
    // We make NFAs reference counted primarily for two reasons. First is that
    // the NFA type itself is quite large (at least 0.5KB), and so it makes
    // sense to put it on the heap by default anyway. Second is that, for Arc
    // specifically, this enables cheap clones. This tends to be useful because
    // several structures (the backtracker, the Pike VM, the hybrid NFA/DFA)
    // all want to hang on to an NFA for use during search time. We could
    // provide the NFA at search time, but this makes for an unnecessarily
    // annoying API. Instead, we just let each structure share ownership of the
    // NFA. Using a deep clone would not be smart, since the NFA can use quite
    // a bit of heap space.
    pub(super) Arc<Inner>,
);

impl NFA {
    pub fn config() -> Config {
        Config::new()
    }

    pub fn builder() -> Builder {
        Builder::new()
    }

    // pub fn compiler() -> Compiler {
    // Compiler::new()
    // }

    /// Returns an NFA with a single regex pattern that always matches at every
    /// position.
    #[inline]
    pub fn always_match() -> NFA {
        let mut builder = NFA::builder();
        let pid = builder.start_pattern().unwrap();
        assert_eq!(pid.as_usize(), 0);
        let start_id = builder.add_match().unwrap();
        let pid = builder.finish_pattern(start_id).unwrap();
        assert_eq!(pid.as_usize(), 0);
        builder.build(start_id, start_id).unwrap()
    }

    /// Returns an NFA that never matches at any position. It contains no
    /// regexes.
    #[inline]
    pub fn never_match() -> NFA {
        let mut builder = NFA::builder();
        let start_id = builder.add_fail().unwrap();
        builder.build(start_id, start_id).unwrap()
    }

    /// Returns an iterator over all pattern IDs in this NFA.
    #[inline]
    pub fn patterns(&self) -> PatternIter {
        PatternIter {
            it: PatternID::iter(self.pattern_len()),
            _marker: core::marker::PhantomData,
        }
    }

    /// Returns the total number of regex patterns in this NFA.
    ///
    /// This may return zero if the NFA was constructed with no patterns. In
    /// this case, the NFA can never produce a match for any input.
    ///
    /// This is guaranteed to be no bigger than [`PatternID::LIMIT`].
    #[inline]
    pub fn pattern_len(&self) -> usize {
        self.0.start_pattern.len()
    }

    /// Return the ID of the initial anchored state of this NFA.
    #[inline]
    pub fn start_anchored(&self) -> StateID {
        self.0.start_anchored
    }

    /// Return the ID of the initial unanchored state of this NFA.
    #[inline]
    pub fn start_unanchored(&self) -> StateID {
        self.0.start_unanchored
    }

    /// Return the ID of the initial anchored state for the given pattern.
    ///
    /// # Panics
    ///
    /// If the pattern doesn't exist in this NFA, then this panics.
    #[inline]
    pub fn start_pattern(&self, pid: PatternID) -> StateID {
        assert!(pid.as_usize() < self.pattern_len(), "invalid pattern ID");
        self.0.start_pattern[pid]
    }

    /// Get the byte class set for this NFA.
    #[inline]
    pub fn byte_class_set(&self) -> &ByteClassSet {
        &self.0.byte_class_set
    }

    /// Return a reference to the NFA state corresponding to the given ID.
    ///
    /// This is a convenience routine for `nfa.states()[id]`.
    ///
    /// # Panics
    ///
    /// This panics when the given identifier does not reference a valid state.
    /// That is, when `id.as_usize() >= nfa.states().len()`.
    #[inline]
    pub fn state(&self, id: StateID) -> &State {
        &self.states()[id]
    }

    /// Returns a slice of all states in this NFA.
    ///
    /// The slice returned is indexed by `StateID`. This provides a convenient
    /// way to access states while following transitions among those states.
    #[inline]
    pub fn states(&self) -> &[State] {
        &self.0.states
    }

    /// Returns the total number of capturing slots in this NFA.
    ///
    /// This value is guaranteed to be a multiple of 2. (Where each capturing
    /// group across all patterns has precisely two capturing slots in the
    /// NFA.)
    #[inline]
    pub fn capture_slot_len(&self) -> usize {
        self.0.capture_slot_len
    }

    /// Returns the slot corresponding to the given capturing group for the
    /// given pattern.
    ///
    /// # Panics
    ///
    /// If either the pattern ID or the capture index is invalid, then this
    /// panics.
    #[inline]
    pub fn slot(&self, pid: PatternID, capture_index: usize) -> usize {
        assert!(pid.as_usize() < self.pattern_len(), "invalid pattern ID");
        self.0.slot(pid, capture_index)
    }

    /// Return the capture group index corresponding to the given name in the
    /// given pattern. If no such capture group name exists in the given
    /// pattern, then this returns `None`.
    ///
    /// # Panics
    ///
    /// If the given pattern ID is invalid, then this panics.
    #[inline]
    pub fn capture_name_to_index(
        &self,
        pid: PatternID,
        name: &str,
    ) -> Option<usize> {
        assert!(pid.as_usize() < self.pattern_len(), "invalid pattern ID");
        self.0.capture_name_to_index[pid].get(name).cloned()
    }

    #[inline]
    pub fn has_any_captures(&self) -> bool {
        self.0.facts.has_captures
    }

    #[inline]
    pub fn is_always_start_anchored(&self) -> bool {
        self.start_anchored() == self.start_unanchored()
    }

    #[inline]
    pub fn has_any_look(&self) -> bool {
        self.0.facts.has_any_look
    }

    #[inline]
    pub fn has_any_anchor(&self) -> bool {
        self.0.facts.has_any_anchor
    }

    #[inline]
    pub fn has_word_boundary(&self) -> bool {
        self.has_word_boundary_unicode() || self.has_word_boundary_ascii()
    }

    #[inline]
    pub fn has_word_boundary_unicode(&self) -> bool {
        self.0.facts.has_word_boundary_unicode
    }

    #[inline]
    pub fn has_word_boundary_ascii(&self) -> bool {
        self.0.facts.has_word_boundary_ascii
    }

    /// Returns the memory usage, in bytes, of this NFA.
    ///
    /// This does **not** include the stack size used up by this NFA. To
    /// compute that, use `std::mem::size_of::<NFA>()`.
    #[inline]
    pub fn memory_usage(&self) -> usize {
        use core::mem::size_of as s;

        self.0.states.len() * s::<State>()
            + self.0.start_pattern.len() * s::<StateID>()
            + self.0.capture_to_slots.len() * s::<Vec<usize>>()
            + self.0.capture_name_to_index.len() * s::<CaptureNameMap>()
            + self.0.capture_index_to_name.len() * s::<Vec<Option<Arc<str>>>>()
            + self.0.memory_extra
    }
}

impl fmt::Debug for NFA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// The "inner" part of the NFA. We split this part out so that we can easily
/// wrap it in an `Arc` above in the definition of `NFA`.
///
/// See builder.rs for the code that actually builds this type. This module
/// does provide (internal) mutable methods for adding things to this
/// NFA before finalizing it, but the high level construction process is
/// controlled by the builder abstraction. (Which is complicated enough to
/// get its own module.)
#[derive(Default)]
pub(super) struct Inner {
    /// The state sequence. This sequence is guaranteed to be indexable by all
    /// starting state IDs, and it is also guaranteed to contain at most one
    /// `Match` state for each pattern compiled into this NFA. (A pattern may
    /// not have a corresponding `Match` state if a `Match` state is impossible
    /// to reach.)
    states: Vec<State>,
    /// The anchored starting state of this NFA.
    start_anchored: StateID,
    /// The unanchored starting state of this NFA.
    start_unanchored: StateID,
    /// The starting states for each individual pattern. Starting at any
    /// of these states will result in only an anchored search for the
    /// corresponding pattern. The vec is indexed by pattern ID. When the NFA
    /// contains a single regex, then `start_pattern[0]` and `start_anchored`
    /// are always equivalent.
    start_pattern: Vec<StateID>,
    /// A map from PatternID to capture group index to its corresponding slot
    /// for the capture's starting location. The end location is always at
    /// `slot+1`. Since every capture group has two slots, it follows that all
    /// slots in this map correspond to starting locations and are thus always
    /// even.
    ///
    /// The way slots are distributed is unfortunately a bit complicated.
    /// Namely, the first (at index 0, always unnamed and always present)
    /// capturing group in every pattern is allocated a pair of slots before
    /// any other capturing group. So for example, the 0th capture group in
    /// pattern 2 will have a slot index less than the 1st capture group in
    /// pattern 1. The motivation for this representation is so that all slots
    /// corresponding to the overall match start/end offsets for each pattern
    /// appear contiguously. This permits, e.g., the Pike VM to execute in a
    /// mode that only tracks overall match offsets without also tracking all
    /// capture group offsets.
    ///
    /// While the number of slots required can be computing by adding 2
    /// to the maximum value found in this mapping, in practice, it takes
    /// linear time with respect to the number of patterns because of our
    /// odd representation. To avoid that inefficiency, the number of slots
    /// is recorded independently via the 'slots' field. This way, one can
    /// allocate the space needed for, say, running a Pike VM without iterating
    /// over all of the patterns.
    capture_to_slots: Vec<Vec<usize>>,
    /// As described above, this is the number of slots required handle all
    /// capturing groups during an NFA search.
    ///
    /// Another important number is the number of slots required to handle just
    /// the start/end offsets of an entire match for each pattern. This number
    /// is always twice the number of patterns.
    ///
    /// This number is always zero if there are no capturing groups in this
    /// NFA.
    capture_slot_len: usize,
    /// A map from capture name to its corresponding index. So e.g., given
    /// a single regex like '(\w+) (\w+) (?P<word>\w+)', the capture name
    /// 'word' for pattern ID=0 would corresponding to the index '3'. Its
    /// corresponding slots would then be '3 * 2 = 6' and '3 * 2 + 1 = 7'.
    capture_name_to_index: Vec<CaptureNameMap>,
    /// A map from pattern ID to capture group index to name, if one exists.
    /// This is effectively the inverse of 'capture_name_to_index'. The outer
    /// vec is indexed by pattern ID, while the inner vec is index by capture
    /// index offset for the corresponding pattern.
    ///
    /// The first capture group for each pattern is always unnamed and is thus
    /// always None.
    capture_index_to_name: Vec<Vec<Option<Arc<str>>>>,
    /// A representation of equivalence classes over the transitions in this
    /// NFA. Two bytes in the same equivalence class must not discriminate
    /// between a match or a non-match. This map can be used to shrink the
    /// total size of a DFA's transition table with a small match-time cost.
    ///
    /// Note that the NFA's transitions are *not* defined in terms of these
    /// equivalence classes. The NFA's transitions are defined on the original
    /// byte values. For the most part, this is because they wouldn't really
    /// help the NFA much since the NFA already uses a sparse representation
    /// to represent transitions. Byte classes are most effective in a dense
    /// representation.
    byte_class_set: ByteClassSet,
    /// Various facts about this NFA, which can be used to improve failure
    /// modes (e.g., rejecting DFA construction if an NFA has Unicode word
    /// boundaries) or for performing optimizations (avoiding an increase in
    /// states if there are no look-around states).
    facts: Facts,
    /// Heap memory used indirectly by NFA states and other things (like the
    /// various capturing group representations above). Since each state
    /// might use a different amount of heap, we need to keep track of this
    /// incrementally.
    memory_extra: usize,
}

impl Inner {
    pub(super) fn slot(&self, pid: PatternID, capture_index: usize) -> usize {
        self.capture_to_slots[pid][capture_index]
    }

    pub(super) fn add(&mut self, state: State) -> Result<StateID, Error> {
        match state {
            State::Range { ref range } => {
                self.byte_class_set.set_range(range.start, range.end);
            }
            State::Sparse(ref sparse) => {
                for range in sparse.ranges.iter() {
                    self.byte_class_set.set_range(range.start, range.end);
                }
            }
            State::Look { ref look, .. } => {
                self.facts.has_any_look = true;
                look.add_to_byteset(&mut self.byte_class_set);
                match look {
                    Look::StartLine
                    | Look::EndLine
                    | Look::StartText
                    | Look::EndText => {
                        self.facts.has_any_anchor = true;
                    }
                    Look::WordBoundaryUnicode
                    | Look::WordBoundaryUnicodeNegate => {
                        self.facts.has_word_boundary_unicode = true;
                    }
                    Look::WordBoundaryAscii
                    | Look::WordBoundaryAsciiNegate => {
                        self.facts.has_word_boundary_ascii = true;
                    }
                }
            }
            State::Capture { .. } => {
                self.facts.has_captures = true;
            }
            State::Union { .. } | State::Fail | State::Match { .. } => {}
        }

        let id = StateID::new(self.states.len())
            .map_err(|_| Error::too_many_states(self.states.len()))?;
        self.memory_extra += state.memory_usage();
        self.states.push(state);
        Ok(id)
    }

    pub(super) fn set_starts(
        &mut self,
        start_anchored: StateID,
        start_unanchored: StateID,
        start_pattern: &[StateID],
    ) {
        self.start_anchored = start_anchored;
        self.start_unanchored = start_unanchored;
        self.start_pattern = start_pattern.to_owned();
    }

    pub(super) fn set_captures(&mut self, captures: &[Vec<Option<Arc<str>>>]) {
        // IDEA: I wonder if it makes sense to split this routine up by
        // defining smaller mutator methods. We are manipulating a lot of state
        // here, and the code below looks fairly hairy.

        assert!(
            self.states.is_empty(),
            "set_captures must be called before adding states",
        );
        let numpats = captures.len();
        let mut next_slot_pattern = 0usize;
        // The builder verifies that all capture group indices are valid with
        // respect to the number of slots required, so this unwrap is OK.
        let mut next_slot_group = numpats.checked_mul(2).unwrap();
        for (pid, groups) in captures.iter().with_pattern_ids() {
            assert_eq!(
                pid.as_usize(),
                self.capture_name_to_index.len(),
                "pattern IDs should be in correspondence",
            );

            let mut slots = vec![next_slot_pattern];
            self.memory_extra += mem::size_of::<usize>();
            // Since next_slot_group will always be greater and we know it's
            // valid, we know that adding 2 `numpats` times will always
            // succeed.
            next_slot_pattern = next_slot_pattern.checked_add(2).unwrap();
            self.capture_slot_len =
                cmp::max(self.capture_slot_len, next_slot_pattern);
            self.capture_name_to_index.push(CaptureNameMap::new());
            self.capture_index_to_name.push(vec![None]);
            self.memory_extra += mem::size_of::<Option<Arc<str>>>();
            // Since we added group[0] above (corresponding to the capture for
            // the entire pattern), we skip that here.
            for (cap_idx, group_name) in groups.iter().enumerate().skip(1) {
                slots.push(next_slot_group);
                self.memory_extra += mem::size_of::<usize>();
                // As above, we know all capture indices are valid.
                next_slot_group = next_slot_group.checked_add(2).unwrap();
                self.capture_slot_len =
                    cmp::max(self.capture_slot_len, next_slot_group);
                if let Some(ref name) = group_name {
                    self.capture_name_to_index[pid]
                        .insert(Arc::clone(name), cap_idx);
                    // Since we're using a hash/btree map, these are more
                    // like minimum amounts of memory used rather than known
                    // actual memory used. (Where actual memory used is an
                    // implementation detail of the hash/btree map itself.)
                    self.memory_extra += mem::size_of::<Arc<str>>();
                    self.memory_extra += mem::size_of::<usize>();
                }
                assert_eq!(
                    cap_idx,
                    self.capture_index_to_name[pid].len(),
                    "capture indices should be in correspondence",
                );
                self.capture_index_to_name[pid].push(group_name.clone());
                self.memory_extra += mem::size_of::<Option<Arc<str>>>();
            }
            self.capture_to_slots.push(slots);
        }
    }

    /// Remap the transitions in every state of this NFA using the given map.
    /// The given map should be indexed according to state ID namespace used by
    /// the transitions of the states currently in this NFA.
    ///
    /// This is particularly useful to the NFA builder, since it is convenient
    /// to add NFA states in order to produce their final IDs. Then, after all
    /// of the intermediate "empty" states (unconditional epsilon transitions)
    /// have been removed from the builder's representation, we can re-map all
    /// of the transitions in the states already added to their final IDs.
    pub(super) fn remap(&mut self, old_to_new: &[StateID]) {
        for state in &mut self.states {
            state.remap(old_to_new);
        }
        self.start_anchored = old_to_new[self.start_anchored];
        self.start_unanchored = old_to_new[self.start_unanchored];
        for (pid, id) in self.start_pattern.iter_mut().with_pattern_ids() {
            *id = old_to_new[*id];
        }
    }
}

impl fmt::Debug for Inner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "thompson::NFA(")?;
        for (sid, state) in self.states.iter().with_state_ids() {
            let status = if sid == self.start_anchored {
                '^'
            } else if sid == self.start_unanchored {
                '>'
            } else {
                ' '
            };
            writeln!(f, "{}{:06?}: {:?}", status, sid.as_usize(), state)?;
        }
        let pattern_len = self.start_pattern.len();
        if pattern_len > 1 {
            writeln!(f, "")?;
            for pid in 0..pattern_len {
                let sid = self.start_pattern[pid];
                writeln!(f, "START({:06?}): {:?}", pid, sid.as_usize())?;
            }
        }
        writeln!(f, "")?;
        writeln!(
            f,
            "transition equivalence classes: {:?}",
            self.byte_class_set.byte_classes(),
        )?;
        writeln!(f, ")")?;
        Ok(())
    }
}

/// A map from capture group name to its corresponding capture index.
///
/// Since there are always two slots for each capture index, the pair of slots
/// corresponding to the capture index for a pattern ID of 0 are indexed at
/// `map["<name>"] * 2` and `map["<name>"] * 2 + 1`.
///
/// This type is actually wrapped inside a Vec indexed by pattern ID on the
/// NFA, since multiple patterns may have the same capture group name.
///
/// Note that this is somewhat of a sub-optimal representation, since it
/// requires a hashmap for each pattern. A better representation would be
/// HashMap<(PatternID, Arc<str>), usize>, but this makes it difficult to look
/// up a capture index by name without producing a `Arc<str>`, which requires
/// an allocation. To fix this, I think we'd need to define our own unsized
/// type or something?
#[cfg(feature = "std")]
type CaptureNameMap = std::collections::HashMap<Arc<str>, usize>;
#[cfg(not(feature = "std"))]
type CaptureNameMap = alloc::collections::BTreeMap<Arc<str>, usize>;

/// A state in a final compiled NFA.
#[derive(Clone, Eq, PartialEq)]
pub enum State {
    /// A state that transitions to `next` if and only if the current input
    /// byte is in the range `[start, end]` (inclusive).
    ///
    /// This is a special case of Sparse in that it encodes only one transition
    /// (and therefore avoids the allocation).
    Range { range: Transition },
    /// A state with possibly many transitions, represented in a sparse
    /// fashion. Transitions are ordered lexicographically by input range. As
    /// such, this may only be used when every transition has equal priority.
    /// (In practice, this is only used for encoding UTF-8 automata.)
    Sparse(SparseTransitions),
    /// A conditional epsilon transition satisfied via some sort of
    /// look-around.
    Look { look: Look, next: StateID },
    /// An alternation such that there exists an epsilon transition to all
    /// states in `alternates`, where matches found via earlier transitions
    /// are preferred over later transitions.
    Union { alternates: Box<[StateID]> },
    /// An empty state that records a capture location.
    ///
    /// From the perspective of finite automata, this is precisely equivalent
    /// to an epsilon transition, but serves the purpose of instructing NFA
    /// simulations to record additional state when the finite state machine
    /// passes through this epsilon transition.
    ///
    /// These transitions are treated as epsilon transitions with no additional
    /// effects in DFAs.
    ///
    /// 'slot' in this context refers to the specific capture group offset that
    /// is being recorded. Each capturing group has two slots corresponding to
    /// the start and end of the matching portion of that group.
    /// A fail state. When encountered, the automaton is guaranteed to never
    /// reach a match state.
    Capture { next: StateID, slot: usize },
    /// A state that cannot be transitioned out of. This is useful for cases
    /// where you want to prevent matching from occurring. For example, if your
    /// regex parser permits empty character classes, then one could choose a
    /// `Fail` state to represent it.
    Fail,
    /// A match state. There is exactly one such occurrence of this state for
    /// each regex compiled into the NFA.
    Match { pattern_id: PatternID },
}

impl State {
    /// Returns true if and only if this state contains one or more epsilon
    /// transitions.
    #[inline]
    pub fn is_epsilon(&self) -> bool {
        match *self {
            State::Range { .. }
            | State::Sparse { .. }
            | State::Fail
            | State::Match { .. } => false,
            State::Look { .. }
            | State::Union { .. }
            | State::Capture { .. } => true,
        }
    }

    /// Returns the heap memory usage of this NFA state in bytes.
    fn memory_usage(&self) -> usize {
        match *self {
            State::Range { .. }
            | State::Look { .. }
            | State::Capture { .. }
            | State::Match { .. }
            | State::Fail => 0,
            State::Sparse(SparseTransitions { ref ranges }) => {
                ranges.len() * mem::size_of::<Transition>()
            }
            State::Union { ref alternates } => {
                alternates.len() * mem::size_of::<StateID>()
            }
        }
    }

    /// Remap the transitions in this state using the given map. Namely, the
    /// given map should be indexed according to the transitions currently
    /// in this state.
    ///
    /// This is used during the final phase of the NFA compiler, which turns
    /// its intermediate NFA into the final NFA.
    fn remap(&mut self, remap: &[StateID]) {
        match *self {
            State::Range { ref mut range } => range.next = remap[range.next],
            State::Sparse(SparseTransitions { ref mut ranges }) => {
                for r in ranges.iter_mut() {
                    r.next = remap[r.next];
                }
            }
            State::Look { ref mut next, .. } => *next = remap[*next],
            State::Union { ref mut alternates } => {
                for alt in alternates.iter_mut() {
                    *alt = remap[*alt];
                }
            }
            State::Capture { ref mut next, .. } => *next = remap[*next],
            State::Fail => {}
            State::Match { .. } => {}
        }
    }
}

impl fmt::Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            State::Range { ref range } => range.fmt(f),
            State::Sparse(SparseTransitions { ref ranges }) => {
                let rs = ranges
                    .iter()
                    .map(|t| format!("{:?}", t))
                    .collect::<Vec<String>>()
                    .join(", ");
                write!(f, "sparse({})", rs)
            }
            State::Look { ref look, next } => {
                write!(f, "{:?} => {:?}", look, next.as_usize())
            }
            State::Union { ref alternates } => {
                let alts = alternates
                    .iter()
                    .map(|id| format!("{:?}", id.as_usize()))
                    .collect::<Vec<String>>()
                    .join(", ");
                write!(f, "alt({})", alts)
            }
            State::Capture { next, slot } => {
                write!(f, "capture({:?}) => {:?}", slot, next.as_usize())
            }
            State::Fail => write!(f, "FAIL"),
            State::Match { pattern_id } => {
                write!(f, "MATCH({:?})", pattern_id.as_usize())
            }
        }
    }
}

/// A collection of facts about an NFA.
///
/// There are no real cohesive principles behind what gets put in here. For
/// the most part, it is implementation driven.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Facts {
    pub(super) has_captures: bool,
    pub(super) has_any_look: bool,
    pub(super) has_any_anchor: bool,
    pub(super) has_word_boundary_unicode: bool,
    pub(super) has_word_boundary_ascii: bool,
}

/// A sequence of transitions used to represent a sparse state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SparseTransitions {
    pub(super) ranges: Box<[Transition]>,
}

impl SparseTransitions {
    pub fn matches(&self, haystack: &[u8], at: usize) -> Option<StateID> {
        haystack.get(at).and_then(|&b| self.matches_byte(b))
    }

    pub fn matches_unit(&self, unit: alphabet::Unit) -> Option<StateID> {
        unit.as_u8().map_or(None, |byte| self.matches_byte(byte))
    }

    pub fn matches_byte(&self, byte: u8) -> Option<StateID> {
        for t in self.ranges.iter() {
            if t.start > byte {
                break;
            } else if t.matches_byte(byte) {
                return Some(t.next);
            }
        }
        None

        /*
        // This is an alternative implementation that uses binary search. In
        // some ad hoc experiments, like
        //
        //   smallishru=OpenSubtitles2018.raw.sample.smallish.ru
        //   regex-cli find nfa thompson pikevm -b "@$smallishru" '\b\w+\b'
        //
        // I could not observe any improvement, and in fact, things seemed to
        // be a bit slower.
        self.ranges
            .binary_search_by(|t| {
                if t.end < byte {
                    core::cmp::Ordering::Less
                } else if t.start > byte {
                    core::cmp::Ordering::Greater
                } else {
                    core::cmp::Ordering::Equal
                }
            })
            .ok()
            .map(|i| self.ranges[i].next)
        */
    }
}

/// A transition to another state, only if the given byte falls in the
/// inclusive range specified.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct Transition {
    pub(super) start: u8,
    pub(super) end: u8,
    pub(super) next: StateID,
}

impl Transition {
    pub fn start(&self) -> u8 {
        self.start
    }

    pub fn end(&self) -> u8 {
        self.end
    }

    pub fn next(&self) -> StateID {
        self.next
    }

    pub fn matches(&self, haystack: &[u8], at: usize) -> bool {
        haystack.get(at).map_or(false, |&b| self.matches_byte(b))
    }

    pub fn matches_unit(&self, unit: alphabet::Unit) -> bool {
        unit.as_u8().map_or(false, |byte| self.matches_byte(byte))
    }

    pub fn matches_byte(&self, byte: u8) -> bool {
        self.start <= byte && byte <= self.end
    }
}

impl fmt::Debug for Transition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::util::DebugByte;

        let Transition { start, end, next } = *self;
        if self.start == self.end {
            write!(f, "{:?} => {:?}", DebugByte(start), next.as_usize())
        } else {
            write!(
                f,
                "{:?}-{:?} => {:?}",
                DebugByte(start),
                DebugByte(end),
                next.as_usize(),
            )
        }
    }
}

/// A conditional NFA epsilon transition.
///
/// A simulation of the NFA can only move through this epsilon transition if
/// the current position satisfies some look-around property. Some assertions
/// are look-behind (StartLine, StartText), some assertions are look-ahead
/// (EndLine, EndText) while other assertions are both look-behind and
/// look-ahead (WordBoundary*).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Look {
    /// The previous position is either `\n` or the current position is the
    /// beginning of the haystack (i.e., at position `0`).
    StartLine = 1 << 0,
    /// The next position is either `\n` or the current position is the end of
    /// the haystack (i.e., at position `haystack.len()`).
    EndLine = 1 << 1,
    /// The current position is the beginning of the haystack (i.e., at
    /// position `0`).
    StartText = 1 << 2,
    /// The current position is the end of the haystack (i.e., at position
    /// `haystack.len()`).
    EndText = 1 << 3,
    /// When tested at position `i`, where `p=decode_utf8_rev(&haystack[..i])`
    /// and `n=decode_utf8(&haystack[i..])`, this assertion passes if and only
    /// if `is_word(p) != is_word(n)`. If `i=0`, then `is_word(p)=false` and if
    /// `i=haystack.len()`, then `is_word(n)=false`.
    WordBoundaryUnicode = 1 << 4,
    /// Same as for `WordBoundaryUnicode`, but requires that
    /// `is_word(p) == is_word(n)`.
    WordBoundaryUnicodeNegate = 1 << 5,
    /// When tested at position `i`, where `p=haystack[i-1]` and
    /// `n=haystack[i]`, this assertion passes if and only if `is_word(p)
    /// != is_word(n)`. If `i=0`, then `is_word(p)=false` and if
    /// `i=haystack.len()`, then `is_word(n)=false`.
    WordBoundaryAscii = 1 << 6,
    /// Same as for `WordBoundaryAscii`, but requires that
    /// `is_word(p) == is_word(n)`.
    ///
    /// Note that it is possible for this assertion to match at positions that
    /// split the UTF-8 encoding of a codepoint. For this reason, this may only
    /// be used when UTF-8 mode is disable in the regex syntax.
    WordBoundaryAsciiNegate = 1 << 7,
}

impl Look {
    pub fn matches(&self, bytes: &[u8], at: usize) -> bool {
        match *self {
            Look::StartLine => at == 0 || bytes[at - 1] == b'\n',
            Look::EndLine => at == bytes.len() || bytes[at] == b'\n',
            Look::StartText => at == 0,
            Look::EndText => at == bytes.len(),
            Look::WordBoundaryUnicode => {
                let word_before = is_word_char_rev(bytes, at);
                let word_after = is_word_char_fwd(bytes, at);
                word_before != word_after
            }
            Look::WordBoundaryUnicodeNegate => {
                // This is pretty subtle. Why do we need to do UTF-8 decoding
                // here? Well... at time of writing, the is_word_char_{fwd,rev}
                // routines will only return true if there is a valid UTF-8
                // encoding of a "word" codepoint, and false in every other
                // case (including invalid UTF-8). This means that in regions
                // of invalid UTF-8 (which might be a subset of valid UTF-8!),
                // it would result in \B matching. While this would be
                // questionable in the context of truly invalid UTF-8, it is
                // *certainly* wrong to report match boundaries that split the
                // encoding of a codepoint. So to work around this, we ensure
                // that we can decode a codepoint on either side of `at`. If
                // either direction fails, then we don't permit \B to match at
                // all.
                //
                // Now, this isn't exactly optimal from a perf perspective. We
                // could try and detect this in is_word_char_{fwd,rev}, but
                // it's not clear if it's worth it. \B is, after all, rarely
                // used.
                //
                // And in particular, we do *not* have to do this with \b,
                // because \b *requires* that at least one side of `at` be a
                // "word" codepoint, which in turn implies one side of `at`
                // must be valid UTF-8. This in turn implies that \b can never
                // split a valid UTF-8 encoding of a codepoint. In the case
                // where one side of `at` is truly invalid UTF-8 and the other
                // side IS a word codepoint, then we want \b to match since it
                // represents a valid UTF-8 boundary. It also makes sense. For
                // example, you'd want \b\w+\b to match 'abc' in '\xFFabc\xFF'.
                let word_before = at > 0
                    && match decode_last_utf8(&bytes[..at]) {
                        None | Some(Err(_)) => return false,
                        Some(Ok(_)) => is_word_char_rev(bytes, at),
                    };
                let word_after = at < bytes.len()
                    && match decode_utf8(&bytes[at..]) {
                        None | Some(Err(_)) => return false,
                        Some(Ok(_)) => is_word_char_fwd(bytes, at),
                    };
                word_before == word_after
            }
            Look::WordBoundaryAscii => {
                let word_before = at > 0 && is_word_byte(bytes[at - 1]);
                let word_after = at < bytes.len() && is_word_byte(bytes[at]);
                word_before != word_after
            }
            Look::WordBoundaryAsciiNegate => {
                let word_before = at > 0 && is_word_byte(bytes[at - 1]);
                let word_after = at < bytes.len() && is_word_byte(bytes[at]);
                word_before == word_after
            }
        }
    }

    /// Create a look-around assertion from its corresponding integer (as
    /// defined in `Look`). If the given integer does not correspond to any
    /// assertion, then None is returned.
    pub fn from_int(n: u8) -> Option<Look> {
        match n {
            0b0000_0001 => Some(Look::StartLine),
            0b0000_0010 => Some(Look::EndLine),
            0b0000_0100 => Some(Look::StartText),
            0b0000_1000 => Some(Look::EndText),
            0b0001_0000 => Some(Look::WordBoundaryUnicode),
            0b0010_0000 => Some(Look::WordBoundaryUnicodeNegate),
            0b0100_0000 => Some(Look::WordBoundaryAscii),
            0b1000_0000 => Some(Look::WordBoundaryAsciiNegate),
            _ => None,
        }
    }

    /// Flip the look-around assertion to its equivalent for reverse searches.
    pub fn reversed(&self) -> Look {
        match *self {
            Look::StartLine => Look::EndLine,
            Look::EndLine => Look::StartLine,
            Look::StartText => Look::EndText,
            Look::EndText => Look::StartText,
            Look::WordBoundaryUnicode => Look::WordBoundaryUnicode,
            Look::WordBoundaryUnicodeNegate => Look::WordBoundaryUnicodeNegate,
            Look::WordBoundaryAscii => Look::WordBoundaryAscii,
            Look::WordBoundaryAsciiNegate => Look::WordBoundaryAsciiNegate,
        }
    }

    /// Split up the given byte classes into equivalence classes in a way that
    /// is consistent with this look-around assertion.
    fn add_to_byteset(&self, set: &mut ByteClassSet) {
        match *self {
            Look::StartText | Look::EndText => {}
            Look::StartLine | Look::EndLine => {
                set.set_range(b'\n', b'\n');
            }
            Look::WordBoundaryUnicode
            | Look::WordBoundaryUnicodeNegate
            | Look::WordBoundaryAscii
            | Look::WordBoundaryAsciiNegate => {
                // We need to mark all ranges of bytes whose pairs result in
                // evaluating \b differently. This isn't technically correct
                // for Unicode word boundaries, but DFAs can't handle those
                // anyway, and thus, the byte classes don't need to either
                // since they are themselves only used in DFAs.
                let iswb = regex_syntax::is_word_byte;
                let mut b1: u16 = 0;
                let mut b2: u16;
                while b1 <= 255 {
                    b2 = b1 + 1;
                    while b2 <= 255 && iswb(b1 as u8) == iswb(b2 as u8) {
                        b2 += 1;
                    }
                    set.set_range(b1 as u8, (b2 - 1) as u8);
                    b1 = b2;
                }
            }
        }
    }
}

/// LookSet is a memory-efficient set of look-around assertions. Callers may
/// idempotently insert or remove any look-around assertion from a set.
#[repr(transparent)]
#[derive(Clone, Copy, Default, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub(crate) struct LookSet {
    set: u8,
}

impl LookSet {
    /// Return a LookSet from its representation.
    pub(crate) fn from_repr(repr: u8) -> LookSet {
        LookSet { set: repr }
    }

    /// Return a mutable LookSet from a mutable pointer to its representation.
    pub(crate) fn from_repr_mut(repr: &mut u8) -> &mut LookSet {
        // SAFETY: This is safe since a LookSet is repr(transparent) where its
        // repr is a u8.
        unsafe { core::mem::transmute::<&mut u8, &mut LookSet>(repr) }
    }

    /// Return true if and only if this set is empty.
    pub(crate) fn is_empty(&self) -> bool {
        self.set == 0
    }

    /// Clears this set such that it has no assertions in it.
    pub(crate) fn clear(&mut self) {
        self.set = 0;
    }

    /// Insert the given look-around assertion into this set. If the assertion
    /// already exists, then this is a no-op.
    pub(crate) fn insert(&mut self, look: Look) {
        self.set |= look as u8;
    }

    /// Remove the given look-around assertion from this set. If the assertion
    /// is not in this set, then this is a no-op.
    #[cfg(test)]
    pub(crate) fn remove(&mut self, look: Look) {
        self.set &= !(look as u8);
    }

    /// Return true if and only if the given assertion is in this set.
    pub(crate) fn contains(&self, look: Look) -> bool {
        (look as u8) & self.set != 0
    }

    /// Subtract the given `other` set from the `self` set and return a new
    /// set.
    pub(crate) fn subtract(&self, other: LookSet) -> LookSet {
        LookSet { set: self.set & !other.set }
    }

    /// Return the intersection of the given `other` set with the `self` set
    /// and return the resulting set.
    pub(crate) fn intersect(&self, other: LookSet) -> LookSet {
        LookSet { set: self.set & other.set }
    }
}

impl core::fmt::Debug for LookSet {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        let mut members = vec![];
        for i in 0..8 {
            let look = match Look::from_int(1 << i) {
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

/// An iterator over all pattern IDs in an NFA.
pub struct PatternIter<'a> {
    it: PatternIDIter,
    /// We explicitly associate a lifetime with this iterator even though we
    /// don't actually borrow anything from the NFA. We do this for backward
    /// compatibility purposes. If we ever do need to borrow something from
    /// the NFA, then we can and just get rid of this marker without breaking
    /// the public API.
    _marker: core::marker::PhantomData<&'a ()>,
}

impl<'a> Iterator for PatternIter<'a> {
    type Item = PatternID;

    fn next(&mut self) -> Option<PatternID> {
        self.it.next()
    }
}
