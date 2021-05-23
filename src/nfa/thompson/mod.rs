use core::fmt;

use alloc::{boxed::Box, format, string::String, vec, vec::Vec};

use crate::{
    classes::ByteClassSet,
    id::{PatternID, StateID},
    nfa::error::Error,
};

pub use self::compiler::{Builder, Config};

mod compiler;
mod map;
mod range_trie;

/// A final compiled NFA.
///
/// The states of the NFA are indexed by state IDs, which are how transitions
/// are expressed.
#[derive(Clone)]
pub struct NFA {
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
    /// The state list. This list is guaranteed to be indexable by the starting
    /// state ID, and it is also guaranteed to contain exactly one `Match`
    /// state.
    states: Vec<State>,
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
}

impl NFA {
    /// Returns an NFA with a single regex that always matches at every
    /// position.
    #[inline]
    pub fn always_match() -> NFA {
        let mut nfa = NFA {
            start_anchored: StateID::ZERO,
            start_unanchored: StateID::ZERO,
            start_pattern: vec![StateID::ZERO],
            states: vec![State::Match(PatternID::ZERO)],
            byte_class_set: ByteClassSet::empty(),
            facts: Facts::default(),
        };
        nfa.set_match_len(1);
        nfa
    }

    /// Returns an NFA that never matches at any position. It contains no
    /// regexes.
    #[inline]
    pub fn never_match() -> NFA {
        NFA {
            start_anchored: StateID::ZERO,
            start_unanchored: StateID::ZERO,
            start_pattern: vec![],
            states: vec![State::Fail],
            byte_class_set: ByteClassSet::empty(),
            facts: Facts::default(),
        }
    }

    /// Return the number of states in this NFA.
    ///
    /// This is guaranteed to be no bigger than one more than
    /// [`StateID::MAX`].
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
    /// This is guaranteed to be no bigger than one more than
    /// [`PatternID::MAX`].
    #[inline]
    pub fn match_len(&self) -> usize {
        self.start_pattern.len()
    }

    /// Returns an iterator over all pattern IDs in this NFA.
    #[inline]
    pub fn patterns(&self) -> PatternIter {
        PatternIter {
            it: 0..self.match_len(),
            _marker: core::marker::PhantomData,
        }
    }

    /// Configures the NFA to match the specified number of patterns.
    #[inline]
    pub(crate) fn set_match_len(&mut self, patterns: usize) {
        self.start_pattern.resize(patterns, StateID::ZERO);
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

    /// Return the NFA state corresponding to the given ID.
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

    /// Returns a mutable slice of all states in this NFA.
    ///
    /// The slice returned may be indexed by a `StateID` generated by `add`.
    #[inline]
    pub fn states_mut(&mut self) -> &mut [State] {
        &mut self.states
    }

    /// Add a new state to this NFA and return its ID.
    #[inline]
    pub fn add(&mut self, state: State) -> Result<StateID, Error> {
        match state {
            State::Range { .. }
            | State::Sparse { .. }
            | State::Union { .. }
            | State::Fail => {}
            State::Match(pid) => {
                let len = pid.one_more();
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
        let id = StateID::new(self.states.len()).map_err(|_| {
            Error::too_many_states(self.states.len(), StateID::LIMIT)
        })?;
        self.states.push(state);
        Ok(id)
    }

    /// Clear this NFA such that it has zero states.
    ///
    /// Note that this does not reset other state in this NFA. Callers
    /// modifying an NFA using methods such as `clear` are responsible for
    /// updating other state as well.
    #[inline]
    pub fn clear(&mut self) {
        self.states.clear();
        // These are directly derived from the states added, so they must also
        // be cleared. They will be regenerated as new states are added.
        self.facts = Facts::default();
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
}

impl fmt::Debug for NFA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "thompson::NFA(")?;
        for (i, state) in self.states.iter().enumerate() {
            let sid = StateID::new(i).unwrap();
            let status = if sid == self.start_anchored {
                '^'
            } else if sid == self.start_unanchored {
                '>'
            } else {
                ' '
            };
            writeln!(f, "{}{:06?}: {:?}", status, sid, state)?;
        }
        if self.match_len() > 1 {
            writeln!(f, "")?;
            for pid in self.patterns() {
                let id = self.start_pattern(pid);
                writeln!(f, "START({:06?}): {:?}", pid, id)?;
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
    Sparse { ranges: Box<[Transition]> },
    /// A conditional epsilon transition satisfied via some sort of
    /// look-around.
    Look { look: Look, next: StateID },
    /// An alternation such that there exists an epsilon transition to all
    /// states in `alternates`, where matches found via earlier transitions
    /// are preferred over later transitions.
    Union { alternates: Box<[StateID]> },
    /// A fail state. When encountered, the automaton is guaranteed to never
    /// reach a match state.
    Fail,
    /// A match state. There is exactly one such occurrence of this state for
    /// each regex compiled into the NFA.
    Match(PatternID),
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
            | State::Match(_) => false,
            State::Look { .. } | State::Union { .. } => true,
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
            State::Sparse { ref mut ranges } => {
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
            State::Fail => {}
            State::Match(_) => {}
        }
    }
}

impl fmt::Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            State::Range { ref range } => range.fmt(f),
            State::Sparse { ref ranges } => {
                let rs = ranges
                    .iter()
                    .map(|t| format!("{:?}", t))
                    .collect::<Vec<String>>()
                    .join(", ");
                write!(f, "sparse({})", rs)
            }
            State::Look { ref look, next } => {
                write!(f, "{:?} => {:?}", look, next)
            }
            State::Union { ref alternates } => {
                let alts = alternates
                    .iter()
                    .map(|id| format!("{:?}", id))
                    .collect::<Vec<String>>()
                    .join(", ");
                write!(f, "alt({})", alts)
            }
            State::Fail => write!(f, "FAIL"),
            State::Match(id) => write!(f, "MATCH({:?})", id),
        }
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

impl fmt::Debug for Transition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::util::DebugByte;

        let Transition { start, end, next } = *self;
        if self.start == self.end {
            write!(f, "{:?} => {:?}", DebugByte(start), next)
        } else {
            write!(
                f,
                "{:?}-{:?} => {:?}",
                DebugByte(start),
                DebugByte(end),
                next
            )
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

/// A conditional NFA epsilon transition that can only be passed through if
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
#[derive(Clone, Copy, Default, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub(crate) struct LookSet {
    set: u8,
}

impl LookSet {
    /// Create an empty set of look-around assertions.
    pub(crate) fn empty() -> LookSet {
        LookSet::default()
    }

    /// Return true if and only if this set is empty.
    pub(crate) fn is_empty(&self) -> bool {
        self.set == 0
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
    it: core::ops::Range<usize>,
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
        // CORRECTNESS: the unwrap is okay here since NFA construction
        // guarantees that its pattern IDs never exceed PatternID::MAX.
        self.it.next().map(|id| PatternID::new(id).unwrap())
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
}
