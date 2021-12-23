use core::{fmt, mem, ops::Range};

use alloc::{boxed::Box, format, string::String, sync::Arc, vec, vec::Vec};

use crate::util::{
    alphabet::{self, ByteClassSet},
    decode_last_utf8, decode_utf8,
    id::{IteratorIDExt, PatternID, PatternIDIter, StateID},
    is_word_byte, is_word_char, is_word_char_fwd, is_word_char_rev,
};

pub use self::{
    compiler::{Builder, Config},
    error::Error,
};

mod compiler;
mod error;
mod map;
pub mod pikevm;
mod range_trie;

/// A map from capture group name to its corresponding capture index.
///
/// Since there are always two slots for each capture index, the pair of slots
/// corresponding to the capture index for a pattern ID of 0 are indexed at
/// `map["<name>"] * 2` and `map["<name>"] * 2 + 1`.
///
/// The key type is defined to include the pattern ID since multiple patterns
/// in the same NFA may have the same capture group name.
#[cfg(feature = "std")]
type CaptureNameMap = std::collections::HashMap<(PatternID, Arc<str>), usize>;
#[cfg(not(feature = "std"))]
type CaptureNameMap =
    alloc::collections::BTreeMap<(PatternID, Arc<str>), usize>;

/// A fully compiled Thompson NFA.
///
/// The states of the NFA are indexed by state IDs, which are how transitions
/// are expressed.
#[derive(Clone)]
pub struct NFA {
    /// The state list. This list is guaranteed to be indexable by the starting
    /// state ID, and it is also guaranteed to contain exactly one `Match`
    /// state.
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
    /// A map from PatternID to its corresponding range of capture groups.
    patterns_to_captures: Vec<Range<usize>>,
    /// The total number of capturing slots in this NFA.
    ///
    /// Generally speaking, it is expected that this value be a multiple of
    /// 2. (Where each capturing group has precisely two capturing slots in the
    /// NFA.) However, this invariant is not enforced and an ill-formed NFA
    /// created by the caller is possible.
    slots: usize,
    /// A map from capture name to its corresponding index. So e.g., given
    /// a regex like '(\w+) (\w+) (?P<word>\w+)', the capture name 'word'
    /// would corresponding to the index '3'. Its corresponding slots would
    /// then be '3 * 2 = 6' and '3 * 2 + 1 = 7'.
    capture_name_to_index: CaptureNameMap,
    /// A map from capture group index to name, if one exists. This is
    /// effectively the inverse of 'capture_name_to_index'.
    ///
    /// The first capture group for each pattern is always unnamed and is thus
    /// always None.
    capture_index_to_name: Vec<Option<Arc<str>>>,
    /// A representation of equivalence classes over the transitions in this
    /// NFA. Two bytes in the same equivalence class cannot discriminate
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
    /// Heap memory used indirectly by NFA states. Since each state might use a
    /// different amount of heap, we need to keep track of this incrementally.
    memory_states: usize,
}

impl NFA {
    pub fn config() -> Config {
        Config::new()
    }

    pub fn builder() -> Builder {
        Builder::new()
    }

    /// Returns an NFA with no states. Its match semantics are unspecified.
    ///
    /// An empty NFA is useful as a starting point for building one. It is
    /// itself not intended to be used for matching. For example, its starting
    /// state identifiers are configured to be `0`, but since it has no states,
    /// the identifiers are invalid.
    ///
    /// If you need an NFA that never matches is anything and can be correctly
    /// used for matching, use [`NFA::never_match`].
    #[inline]
    pub fn empty() -> NFA {
        NFA {
            states: vec![],
            start_anchored: StateID::ZERO,
            start_unanchored: StateID::ZERO,
            start_pattern: vec![],
            patterns_to_captures: vec![],
            slots: 0,
            capture_name_to_index: CaptureNameMap::new(),
            capture_index_to_name: vec![],
            byte_class_set: ByteClassSet::empty(),
            facts: Facts::default(),
            memory_states: 0,
        }
    }

    /// Returns an NFA with a single regex that always matches at every
    /// position.
    #[inline]
    pub fn always_match() -> NFA {
        let mut nfa = NFA::empty();
        nfa.add(State::Match { id: PatternID::ZERO });
        nfa
    }

    /// Returns an NFA that never matches at any position. It contains no
    /// regexes.
    #[inline]
    pub fn never_match() -> NFA {
        let mut nfa = NFA::empty();
        nfa.add(State::Fail);
        nfa
    }

    /// Return the number of states in this NFA.
    ///
    /// This is guaranteed to be no bigger than [`StateID::LIMIT`].
    #[inline]
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Returns the total number of distinct match states in this NFA.
    /// Stated differently, this returns the total number of regex patterns
    /// used to build this NFA.
    ///
    /// This may return zero if the NFA was constructed with no patterns. In
    /// this case, and only this case, the NFA can never produce a match for
    /// any input.
    ///
    /// This is guaranteed to be no bigger than [`PatternID::LIMIT`].
    #[inline]
    pub fn match_len(&self) -> usize {
        self.start_pattern.len()
    }

    /// Configures the NFA to match the specified number of patterns.
    #[inline]
    pub(crate) fn set_match_len(&mut self, patterns: usize) {
        self.start_pattern.resize(patterns, StateID::ZERO);
    }

    /// Returns the total number of capturing groups in this NFA.
    ///
    /// This includes the special 0th capture group that is always present and
    /// captures the start and end offset of the entire match.
    ///
    /// This is a convenience routine for `nfa.capture_slot_len() / 2`.
    ///
    /// # Panics
    ///
    /// This panics if the number of slots is not divisible by `2`.
    #[inline]
    pub fn capture_len(&self) -> usize {
        let slots = self.capture_slot_len();
        assert_eq!(slots % 2, 0, "capture slots must be divisible by 2");
        slots / 2
    }

    /// Returns the total number of capturing slots in this NFA.
    ///
    /// Generally speaking, it is expected that this value be a multiple of
    /// 2. (Where each capturing group has precisely two capturing slots in the
    /// NFA.) However, this invariant is not enforced when an ill-formed NFA
    /// created by the caller is possible.
    #[inline]
    pub fn capture_slot_len(&self) -> usize {
        self.slots
    }

    /// Configures the NFA to have the specified number of slots.
    #[inline]
    pub(crate) fn set_capture_slot_len(&mut self, slots: usize) {
        self.slots = slots;
    }

    /// Returns an iterator over all pattern IDs in this NFA.
    #[inline]
    pub fn patterns(&self) -> PatternIter {
        PatternIter {
            it: PatternID::iter(self.match_len()),
            _marker: core::marker::PhantomData,
        }
    }

    /// Return the ID of the initial anchored state of this NFA.
    #[inline]
    pub fn start_anchored(&self) -> StateID {
        self.start_anchored
    }

    /// Set the anchored starting state ID for this NFA.
    #[inline]
    pub fn set_start_anchored(&mut self, id: StateID) {
        self.start_anchored = id;
    }

    /// Return the ID of the initial unanchored state of this NFA.
    #[inline]
    pub fn start_unanchored(&self) -> StateID {
        self.start_unanchored
    }

    /// Set the unanchored starting state ID for this NFA.
    #[inline]
    pub fn set_start_unanchored(&mut self, id: StateID) {
        self.start_unanchored = id;
    }

    /// Return the ID of the initial anchored state for the given pattern.
    ///
    /// If the pattern doesn't exist in this NFA, then this panics.
    #[inline]
    pub fn start_pattern(&self, pid: PatternID) -> StateID {
        self.start_pattern[pid]
    }

    /// Set the anchored starting state ID for the given pattern in this NFA.
    ///
    /// If the pattern doesn't exist in this NFA, then this panics.
    #[inline]
    pub fn set_start_pattern(&mut self, pid: PatternID, id: StateID) {
        self.start_pattern[pid] = id;
    }

    /// Get the byte class set for this NFA.
    #[inline]
    pub fn byte_class_set(&self) -> &ByteClassSet {
        &self.byte_class_set
    }

    /// Set the byte class set for this NFA.
    #[inline]
    pub fn set_byte_class_set(&mut self, set: ByteClassSet) {
        self.byte_class_set = set;
    }

    /// Return a reference to the NFA state corresponding to the given ID.
    #[inline]
    pub fn state(&self, id: StateID) -> &State {
        &self.states[id]
    }

    /// Returns a slice of all states in this NFA.
    ///
    /// The slice returned may be indexed by a `StateID` generated by `add`.
    #[inline]
    pub fn states(&self) -> &[State] {
        &self.states
    }

    /// Remap the transitions in every state of this NFA using the given map.
    /// The given map should be indexed according to state ID namespace used by
    /// the transitions of the states currently in this NFA.
    ///
    /// This may be used during the final phases of an NFA compiler, which
    /// turns its intermediate NFA into the final NFA. Remapping may be
    /// required to bring the state pointers from the intermediate NFA to the
    /// final NFA.
    pub fn remap(&mut self, old_to_new: &[StateID]) {
        for state in &mut self.states {
            state.remap(old_to_new);
        }
    }

    /// Add a new state to this NFA and return its ID.
    #[inline]
    pub fn add(&mut self, state: State) -> Result<StateID, Error> {
        match state {
            State::Range { .. }
            | State::Sparse { .. }
            | State::Union { .. }
            | State::Fail => {}
            State::Capture { slot, .. } => {
                let len = slot.checked_add(1).unwrap();
                if len > self.capture_slot_len() {
                    self.set_capture_slot_len(len);
                }
            }
            State::Match { id } => {
                let len = id.one_more();
                if len > self.match_len() {
                    self.set_match_len(len);
                }
            }
            State::Look { look, .. } => {
                self.facts.set_has_any_look(true);
                match look {
                    Look::StartLine
                    | Look::EndLine
                    | Look::StartText
                    | Look::EndText => {
                        self.facts.set_has_any_anchor(true);
                    }
                    Look::WordBoundaryUnicode
                    | Look::WordBoundaryUnicodeNegate => {
                        self.facts.set_has_word_boundary_unicode(true);
                    }
                    Look::WordBoundaryAscii
                    | Look::WordBoundaryAsciiNegate => {
                        self.facts.set_has_word_boundary_ascii(true);
                    }
                }
            }
        }
        let id = StateID::new(self.states.len())
            .map_err(|_| Error::too_many_states(self.states.len()))?;
        self.memory_states += state.memory_usage();
        self.states.push(state);
        Ok(id)
    }

    /// Clear this NFA such that it has zero states and is otherwise "empty."
    ///
    /// An empty NFA is useful as a starting point for building one. It is
    /// itself not intended to be used for matching. For example, its starting
    /// state identifiers are configured to be `0`, but since it has no states,
    /// the identifiers are invalid.
    #[inline]
    pub fn clear(&mut self) {
        self.start_anchored = StateID::ZERO;
        self.start_unanchored = StateID::ZERO;
        self.states.clear();
        self.start_pattern.clear();
        self.slots = 0;
        self.byte_class_set = ByteClassSet::empty();
        // These are directly derived from the states added, so they must also
        // be cleared. They will be regenerated as new states are added.
        self.facts = Facts::default();
        self.memory_states = 0;
    }

    #[inline]
    pub fn is_always_start_anchored(&self) -> bool {
        self.start_anchored() == self.start_unanchored()
    }

    #[inline]
    pub fn has_any_look(&self) -> bool {
        self.facts.has_any_look()
    }

    #[inline]
    pub fn has_any_anchor(&self) -> bool {
        self.facts.has_any_anchor()
    }

    #[inline]
    pub fn has_word_boundary(&self) -> bool {
        self.has_word_boundary_unicode() || self.has_word_boundary_ascii()
    }

    #[inline]
    pub fn has_word_boundary_unicode(&self) -> bool {
        self.facts.has_word_boundary_unicode()
    }

    #[inline]
    pub fn has_word_boundary_ascii(&self) -> bool {
        self.facts.has_word_boundary_ascii()
    }

    /// Returns the memory usage, in bytes, of this NFA.
    ///
    /// This does **not** include the stack size used up by this NFA. To
    /// compute that, use `std::mem::size_of::<NFA>()`.
    #[inline]
    pub fn memory_usage(&self) -> usize {
        self.states.len() * mem::size_of::<State>()
            + self.memory_states
            + self.start_pattern.len() * mem::size_of::<StateID>()
    }
}

impl fmt::Debug for NFA {
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
        if self.match_len() > 1 {
            writeln!(f, "")?;
            for pid in self.patterns() {
                let sid = self.start_pattern(pid);
                writeln!(
                    f,
                    "START({:06?}): {:?}",
                    pid.as_usize(),
                    sid.as_usize()
                )?;
            }
        }
        writeln!(f, "")?;
        writeln!(
            f,
            "transition equivalence classes: {:?}",
            self.byte_class_set().byte_classes()
        )?;
        writeln!(f, ")")?;
        Ok(())
    }
}

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
    /// A state that cannot be transitioned out of. If a search reaches this
    /// state, then no match is possible and the search should terminate.
    Fail,
    /// A match state. There is exactly one such occurrence of this state for
    /// each regex compiled into the NFA.
    Match { id: PatternID },
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
            State::Match { id } => write!(f, "MATCH({:?})", id.as_usize()),
        }
    }
}

/// A collection of facts about an NFA.
///
/// There are no real cohesive principles behind what gets put in here. For
/// the most part, it is implementation driven.
#[derive(Clone, Copy, Debug, Default)]
struct Facts {
    /// Various yes/no facts about this NFA.
    bools: u16,
}

impl Facts {
    define_bool!(0, has_any_look, set_has_any_look);
    define_bool!(1, has_any_anchor, set_has_any_anchor);
    define_bool!(2, has_word_boundary_unicode, set_has_word_boundary_unicode);
    define_bool!(3, has_word_boundary_ascii, set_has_word_boundary_ascii);
}

/// A sequence of transitions used to represent a sparse state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SparseTransitions {
    pub ranges: Box<[Transition]>,
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
    pub start: u8,
    pub end: u8,
    pub next: StateID,
}

impl Transition {
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
    #[inline(always)]
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
    fn from_int(n: u8) -> Option<Look> {
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
    fn reversed(&self) -> Look {
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

#[cfg(test)]
mod tests {
    use super::*;
    // TODO: Replace tests using DFA with NFA matching engine once implemented.
    use crate::dfa::{dense, Automaton};

    #[test]
    fn always_match() {
        let nfa = NFA::always_match();
        let dfa = dense::Builder::new().build_from_nfa(&nfa).unwrap();
        let find = |input, start, end| {
            dfa.find_leftmost_fwd_at(None, None, input, start, end)
                .unwrap()
                .map(|m| m.offset())
        };

        assert_eq!(Some(0), find(b"", 0, 0));
        assert_eq!(Some(0), find(b"a", 0, 1));
        assert_eq!(Some(1), find(b"a", 1, 1));
        assert_eq!(Some(0), find(b"ab", 0, 2));
        assert_eq!(Some(1), find(b"ab", 1, 2));
        assert_eq!(Some(2), find(b"ab", 2, 2));
    }

    #[test]
    fn never_match() {
        let nfa = NFA::never_match();
        let dfa = dense::Builder::new().build_from_nfa(&nfa).unwrap();
        let find = |input, start, end| {
            dfa.find_leftmost_fwd_at(None, None, input, start, end)
                .unwrap()
                .map(|m| m.offset())
        };

        assert_eq!(None, find(b"", 0, 0));
        assert_eq!(None, find(b"a", 0, 1));
        assert_eq!(None, find(b"a", 1, 1));
        assert_eq!(None, find(b"ab", 0, 2));
        assert_eq!(None, find(b"ab", 1, 2));
        assert_eq!(None, find(b"ab", 2, 2));
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

    #[test]
    fn look_matches_start_line() {
        let look = Look::StartLine;

        assert!(look.matches(B(""), 0));
        assert!(look.matches(B("\n"), 0));
        assert!(look.matches(B("\n"), 1));
        assert!(look.matches(B("a"), 0));
        assert!(look.matches(B("\na"), 1));

        assert!(!look.matches(B("a"), 1));
        assert!(!look.matches(B("a\na"), 1));
    }

    #[test]
    fn look_matches_end_line() {
        let look = Look::EndLine;

        assert!(look.matches(B(""), 0));
        assert!(look.matches(B("\n"), 1));
        assert!(look.matches(B("\na"), 0));
        assert!(look.matches(B("\na"), 2));
        assert!(look.matches(B("a\na"), 1));

        assert!(!look.matches(B("a"), 0));
        assert!(!look.matches(B("\na"), 1));
        assert!(!look.matches(B("a\na"), 0));
        assert!(!look.matches(B("a\na"), 2));
    }

    #[test]
    fn look_matches_start_text() {
        let look = Look::StartText;

        assert!(look.matches(B(""), 0));
        assert!(look.matches(B("\n"), 0));
        assert!(look.matches(B("a"), 0));

        assert!(!look.matches(B("\n"), 1));
        assert!(!look.matches(B("\na"), 1));
        assert!(!look.matches(B("a"), 1));
        assert!(!look.matches(B("a\na"), 1));
    }

    #[test]
    fn look_matches_end_text() {
        let look = Look::EndText;

        assert!(look.matches(B(""), 0));
        assert!(look.matches(B("\n"), 1));
        assert!(look.matches(B("\na"), 2));

        assert!(!look.matches(B("\na"), 0));
        assert!(!look.matches(B("a\na"), 1));
        assert!(!look.matches(B("a"), 0));
        assert!(!look.matches(B("\na"), 1));
        assert!(!look.matches(B("a\na"), 0));
        assert!(!look.matches(B("a\na"), 2));
    }

    #[test]
    fn look_matches_word_unicode() {
        let look = Look::WordBoundaryUnicode;

        // \xF0\x9D\x9B\x83 = 𝛃 (in \w)
        // \xF0\x90\x86\x80 = 𐆀 (not in \w)

        // Simple ASCII word boundaries.
        assert!(look.matches(B("a"), 0));
        assert!(look.matches(B("a"), 1));
        assert!(look.matches(B("a "), 1));
        assert!(look.matches(B(" a "), 1));
        assert!(look.matches(B(" a "), 2));

        // Unicode word boundaries with a non-ASCII codepoint.
        assert!(look.matches(B("𝛃"), 0));
        assert!(look.matches(B("𝛃"), 4));
        assert!(look.matches(B("𝛃 "), 4));
        assert!(look.matches(B(" 𝛃 "), 1));
        assert!(look.matches(B(" 𝛃 "), 5));

        // Unicode word boundaries between non-ASCII codepoints.
        assert!(look.matches(B("𝛃𐆀"), 0));
        assert!(look.matches(B("𝛃𐆀"), 4));

        // Non word boundaries for ASCII.
        assert!(!look.matches(B(""), 0));
        assert!(!look.matches(B("ab"), 1));
        assert!(!look.matches(B("a "), 2));
        assert!(!look.matches(B(" a "), 0));
        assert!(!look.matches(B(" a "), 3));

        // Non word boundaries with a non-ASCII codepoint.
        assert!(!look.matches(B("𝛃b"), 4));
        assert!(!look.matches(B("𝛃 "), 5));
        assert!(!look.matches(B(" 𝛃 "), 0));
        assert!(!look.matches(B(" 𝛃 "), 6));
        assert!(!look.matches(B("𝛃"), 1));
        assert!(!look.matches(B("𝛃"), 2));
        assert!(!look.matches(B("𝛃"), 3));

        // Non word boundaries with non-ASCII codepoints.
        assert!(!look.matches(B("𝛃𐆀"), 1));
        assert!(!look.matches(B("𝛃𐆀"), 2));
        assert!(!look.matches(B("𝛃𐆀"), 3));
        assert!(!look.matches(B("𝛃𐆀"), 5));
        assert!(!look.matches(B("𝛃𐆀"), 6));
        assert!(!look.matches(B("𝛃𐆀"), 7));
        assert!(!look.matches(B("𝛃𐆀"), 8));
    }

    #[test]
    fn look_matches_word_ascii() {
        let look = Look::WordBoundaryAscii;

        // \xF0\x9D\x9B\x83 = 𝛃 (in \w)
        // \xF0\x90\x86\x80 = 𐆀 (not in \w)

        // Simple ASCII word boundaries.
        assert!(look.matches(B("a"), 0));
        assert!(look.matches(B("a"), 1));
        assert!(look.matches(B("a "), 1));
        assert!(look.matches(B(" a "), 1));
        assert!(look.matches(B(" a "), 2));

        // Unicode word boundaries with a non-ASCII codepoint. Since this is
        // an ASCII word boundary, none of these match.
        assert!(!look.matches(B("𝛃"), 0));
        assert!(!look.matches(B("𝛃"), 4));
        assert!(!look.matches(B("𝛃 "), 4));
        assert!(!look.matches(B(" 𝛃 "), 1));
        assert!(!look.matches(B(" 𝛃 "), 5));

        // Unicode word boundaries between non-ASCII codepoints. Again, since
        // this is an ASCII word boundary, none of these match.
        assert!(!look.matches(B("𝛃𐆀"), 0));
        assert!(!look.matches(B("𝛃𐆀"), 4));

        // Non word boundaries for ASCII.
        assert!(!look.matches(B(""), 0));
        assert!(!look.matches(B("ab"), 1));
        assert!(!look.matches(B("a "), 2));
        assert!(!look.matches(B(" a "), 0));
        assert!(!look.matches(B(" a "), 3));

        // Non word boundaries with a non-ASCII codepoint.
        assert!(look.matches(B("𝛃b"), 4));
        assert!(!look.matches(B("𝛃 "), 5));
        assert!(!look.matches(B(" 𝛃 "), 0));
        assert!(!look.matches(B(" 𝛃 "), 6));
        assert!(!look.matches(B("𝛃"), 1));
        assert!(!look.matches(B("𝛃"), 2));
        assert!(!look.matches(B("𝛃"), 3));

        // Non word boundaries with non-ASCII codepoints.
        assert!(!look.matches(B("𝛃𐆀"), 1));
        assert!(!look.matches(B("𝛃𐆀"), 2));
        assert!(!look.matches(B("𝛃𐆀"), 3));
        assert!(!look.matches(B("𝛃𐆀"), 5));
        assert!(!look.matches(B("𝛃𐆀"), 6));
        assert!(!look.matches(B("𝛃𐆀"), 7));
        assert!(!look.matches(B("𝛃𐆀"), 8));
    }

    #[test]
    fn look_matches_word_unicode_negate() {
        let look = Look::WordBoundaryUnicodeNegate;

        // \xF0\x9D\x9B\x83 = 𝛃 (in \w)
        // \xF0\x90\x86\x80 = 𐆀 (not in \w)

        // Simple ASCII word boundaries.
        assert!(!look.matches(B("a"), 0));
        assert!(!look.matches(B("a"), 1));
        assert!(!look.matches(B("a "), 1));
        assert!(!look.matches(B(" a "), 1));
        assert!(!look.matches(B(" a "), 2));

        // Unicode word boundaries with a non-ASCII codepoint.
        assert!(!look.matches(B("𝛃"), 0));
        assert!(!look.matches(B("𝛃"), 4));
        assert!(!look.matches(B("𝛃 "), 4));
        assert!(!look.matches(B(" 𝛃 "), 1));
        assert!(!look.matches(B(" 𝛃 "), 5));

        // Unicode word boundaries between non-ASCII codepoints.
        assert!(!look.matches(B("𝛃𐆀"), 0));
        assert!(!look.matches(B("𝛃𐆀"), 4));

        // Non word boundaries for ASCII.
        assert!(look.matches(B(""), 0));
        assert!(look.matches(B("ab"), 1));
        assert!(look.matches(B("a "), 2));
        assert!(look.matches(B(" a "), 0));
        assert!(look.matches(B(" a "), 3));

        // Non word boundaries with a non-ASCII codepoint.
        assert!(look.matches(B("𝛃b"), 4));
        assert!(look.matches(B("𝛃 "), 5));
        assert!(look.matches(B(" 𝛃 "), 0));
        assert!(look.matches(B(" 𝛃 "), 6));
        // These don't match because they could otherwise return an offset that
        // splits the UTF-8 encoding of a codepoint.
        assert!(!look.matches(B("𝛃"), 1));
        assert!(!look.matches(B("𝛃"), 2));
        assert!(!look.matches(B("𝛃"), 3));

        // Non word boundaries with non-ASCII codepoints. These also don't
        // match because they could otherwise return an offset that splits the
        // UTF-8 encoding of a codepoint.
        assert!(!look.matches(B("𝛃𐆀"), 1));
        assert!(!look.matches(B("𝛃𐆀"), 2));
        assert!(!look.matches(B("𝛃𐆀"), 3));
        assert!(!look.matches(B("𝛃𐆀"), 5));
        assert!(!look.matches(B("𝛃𐆀"), 6));
        assert!(!look.matches(B("𝛃𐆀"), 7));
        // But this one does, since 𐆀 isn't a word codepoint, and 8 is the end
        // of the haystack. So the "end" of the haystack isn't a word and 𐆀
        // isn't a word, thus, \B matches.
        assert!(look.matches(B("𝛃𐆀"), 8));
    }

    #[test]
    fn look_matches_word_ascii_negate() {
        let look = Look::WordBoundaryAsciiNegate;

        // \xF0\x9D\x9B\x83 = 𝛃 (in \w)
        // \xF0\x90\x86\x80 = 𐆀 (not in \w)

        // Simple ASCII word boundaries.
        assert!(!look.matches(B("a"), 0));
        assert!(!look.matches(B("a"), 1));
        assert!(!look.matches(B("a "), 1));
        assert!(!look.matches(B(" a "), 1));
        assert!(!look.matches(B(" a "), 2));

        // Unicode word boundaries with a non-ASCII codepoint. Since this is
        // an ASCII word boundary, none of these match.
        assert!(look.matches(B("𝛃"), 0));
        assert!(look.matches(B("𝛃"), 4));
        assert!(look.matches(B("𝛃 "), 4));
        assert!(look.matches(B(" 𝛃 "), 1));
        assert!(look.matches(B(" 𝛃 "), 5));

        // Unicode word boundaries between non-ASCII codepoints. Again, since
        // this is an ASCII word boundary, none of these match.
        assert!(look.matches(B("𝛃𐆀"), 0));
        assert!(look.matches(B("𝛃𐆀"), 4));

        // Non word boundaries for ASCII.
        assert!(look.matches(B(""), 0));
        assert!(look.matches(B("ab"), 1));
        assert!(look.matches(B("a "), 2));
        assert!(look.matches(B(" a "), 0));
        assert!(look.matches(B(" a "), 3));

        // Non word boundaries with a non-ASCII codepoint.
        assert!(!look.matches(B("𝛃b"), 4));
        assert!(look.matches(B("𝛃 "), 5));
        assert!(look.matches(B(" 𝛃 "), 0));
        assert!(look.matches(B(" 𝛃 "), 6));
        assert!(look.matches(B("𝛃"), 1));
        assert!(look.matches(B("𝛃"), 2));
        assert!(look.matches(B("𝛃"), 3));

        // Non word boundaries with non-ASCII codepoints.
        assert!(look.matches(B("𝛃𐆀"), 1));
        assert!(look.matches(B("𝛃𐆀"), 2));
        assert!(look.matches(B("𝛃𐆀"), 3));
        assert!(look.matches(B("𝛃𐆀"), 5));
        assert!(look.matches(B("𝛃𐆀"), 6));
        assert!(look.matches(B("𝛃𐆀"), 7));
        assert!(look.matches(B("𝛃𐆀"), 8));
    }

    fn B<'a, T: 'a + ?Sized + AsRef<[u8]>>(string: &'a T) -> &'a [u8] {
        string.as_ref()
    }
}
