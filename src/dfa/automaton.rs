use crate::dfa::search;
use crate::prefilter::{self, Prefilter};
use crate::state_id::StateID;
use crate::word::is_word_byte;
use crate::{NoMatch, PatternID};

/// The size of the alphabet in a standard DFA.
///
/// Specifically, this length controls the number of transitions present in
/// each DFA state. However, when the byte class optimization is enabled,
/// then each DFA maps the space of all possible 256 byte values to at most
/// 256 distinct equivalence classes. In this case, the number of distinct
/// equivalence classes corresponds to the internal alphabet of the DFA, in the
/// sense that each DFA state has a number of transitions equal to the number
/// of equivalence classes despite supporting matching on all possible byte
/// values.
pub const ALPHABET_LEN: usize = 256 + 1;

/// The offset, in bytes, that a match is delayed by in the DFAs generated
/// by this crate.
pub const MATCH_OFFSET: usize = 1;

/// The special EOF sentinel value.
pub const EOF: usize = ALPHABET_LEN - 1;

/// A representation of a match reported by a DFA.
///
/// This is called a "half" match because it only includes the end location
/// (or start location for a reverse match) of a match. This corresponds to the
/// information that a single DFA scan can report. Getting the other half of
/// the match requires a second scan with a reversed DFA.
///
/// A half match also includes the pattern that matched. The pattern is
/// identified by an ID, which corresponds to its position (starting from `0`)
/// relative to other patterns used to construct the corresponding DFA. If only
/// a single pattern is provided, then all matches are guaranteed to have a
/// pattern ID of `0`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct HalfMatch {
    /// The pattern ID.
    pub pattern: PatternID,
    /// The offset of the match.
    ///
    /// For forward searches, the offset is exclusive. For reverse searches,
    /// the offset is inclusive.
    pub offset: usize,
}

impl HalfMatch {
    /// Create a new half match from a pattern ID and a byte offset.
    #[inline]
    pub fn new(pattern: PatternID, offset: usize) -> HalfMatch {
        HalfMatch { pattern, offset }
    }

    /// Returns the ID of the pattern that matched.
    ///
    /// The ID of a pattern is derived from the position in which it was
    /// originally inserted into the corresponding regex engine. The first
    /// pattern has identifier `0`, and each subsequent pattern is `1`, `2` and
    /// so on.
    #[inline]
    pub fn pattern(&self) -> PatternID {
        self.pattern
    }

    /// The position of the match.
    ///
    /// If this match was produced by a forward search, then the offset is
    /// exclusive. If this match was produced by a reverse search, then the
    /// offset is inclusive.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }
}

/// A trait describing the interface of a deterministic finite automaton (DFA).
///
/// Every DFA has exactly one start state and at least one dead state (which
/// may be the same, as in the case of an empty DFA). In all cases, a state
/// identifier of `0` must be a dead state such that `DFA::is_dead_state(0)`
/// always returns `true`.
///
/// Every DFA also has zero or more match states, such that
/// `DFA::is_match_state(id)` returns `true` if and only if `id` corresponds to
/// a match state.
///
/// In general, users of this trait likely will only need to use the search
/// routines such as `is_match`, `shortest_match`, `find` or `rfind`. The other
/// methods are lower level and are used for walking the transitions of a DFA
/// manually. In particular, the aforementioned search routines are implemented
/// generically in terms of the lower level transition walking routines.
///
/// # Safety
///
/// This trait is unsafe to implement because DFA searching may rely on the
/// correctness of the implementation for memory safety. For example, DFA
/// searching may use explicit bounds check elision, which will in turn rely
/// on the correctness of every function that returns a state ID.
///
/// When implementing this trait, one must uphold the documented correctness
/// guarantees. Otherwise, undefined behavior may occur.
pub unsafe trait Automaton {
    /// The representation used for state identifiers in this DFA.
    ///
    /// Typically, this is one of `u8`, `u16`, `u32`, `u64` or `usize`.
    type ID: StateID;

    /// Returns true if and only if the given identifier corresponds to either
    /// a dead state or a match state, such that one of `is_match_state(id)`
    /// or `is_dead_state(id)` must return true.
    ///
    /// Depending on the implementation of the DFA, this routine can be used
    /// to save a branch in the core matching loop. Nevertheless,
    /// `is_match_state(id) || is_dead_state(id)` is always a valid
    /// implementation.
    fn is_special_state(&self, id: Self::ID) -> bool;

    /// Returns true if and only if the given identifier corresponds to a dead
    /// state. When a DFA enters a dead state, it is impossible to leave and
    /// thus can never lead to a match.
    fn is_dead_state(&self, id: Self::ID) -> bool;

    /// Returns true if and only if the given identifier corresponds to a
    /// quit state. A quit state is like a dead state (it has no outgoing
    /// transitions), except it indicates that the DFA failed to complete the
    /// search. When this occurs, callers can neither accept or reject that a
    /// match occurred.
    fn is_quit_state(&self, id: Self::ID) -> bool;

    /// Returns true if and only if the given identifier corresponds to a match
    /// state.
    fn is_match_state(&self, id: Self::ID) -> bool;

    /// Returns true if and only if the given identifier corresponds to a start
    /// state.
    fn is_start_state(&self, id: Self::ID) -> bool;

    /// Returns true if and only if the given identifier corresponds to an
    /// accelerated state.
    fn is_accel_state(&self, id: Self::ID) -> bool;

    /// Given the current state that this DFA is in and the next input byte,
    /// this method returns the identifier of the next state. The identifier
    /// returned is always valid, but it may correspond to a dead state.
    fn next_state(&self, current: Self::ID, input: u8) -> Self::ID;

    /// Like `next_state`, but its implementation may look up the next state
    /// without memory safety checks such as bounds checks. As such, callers
    /// must ensure that the given identifier corresponds to a valid DFA
    /// state. Implementors must, in turn, ensure that this routine is safe
    /// for all valid state identifiers and for all possible `u8` values.
    unsafe fn next_state_unchecked(
        &self,
        current: Self::ID,
        input: u8,
    ) -> Self::ID;

    /// Given the current state and an input that has reached EOF, attempt the
    /// final state transition.
    ///
    /// For DFAs that do not delay matches, this should always return the given
    /// state ID.
    fn next_eof_state(&self, current: Self::ID) -> Self::ID;

    /// Returns the total number of patterns compiled into this DFA.
    ///
    /// In the case of a DFA that never matches any pattern, this should
    /// return `0`.
    fn patterns(&self) -> usize;

    /// Return the match offset of this DFA. This corresponds to the number
    /// of bytes that a match is delayed by. This is typically set to `1`,
    /// which means that a match is always reported exactly one byte after it
    /// occurred.
    fn match_offset(&self) -> usize;

    /// Returns the total number of patterns that match in this state.
    ///
    /// If the given state is not a match state, then this always returns zero.
    fn match_count(&self, id: Self::ID) -> usize;

    /// Returns the pattern ID corresponding to the given match index in the
    /// given state. This must panic if the given match index is out of bounds
    /// or if the given state ID does not correspond to a match state.
    fn match_pattern(&self, id: Self::ID, index: usize) -> PatternID;

    /// Return the identifier of this DFA's start state for the given haystack
    /// when matching in the forward direction.
    fn start_state_forward(
        &self,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Self::ID;

    /// Return the identifier of this DFA's start state for the given haystack
    /// when matching in the reverse direction.
    fn start_state_reverse(
        &self,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Self::ID;

    /// Return a slice of bytes to accelerate for the given style, if possible.
    ///
    /// If the given state has no accelerator, then an empty slice should be
    /// returned.
    fn accelerator(&self, id: Self::ID) -> &[u8] {
        &[]
    }

    /// Returns the first position at which a match is found.
    ///
    /// This routine stops scanning input in precisely the same circumstances
    /// as `is_match`. The key difference is that this routine returns the
    /// position at which it stopped scanning input if and only if a match
    /// was found. If no match is found, then `None` is returned.
    ///
    /// # Example
    ///
    /// This example shows how to use this method with a
    /// [`dense::DFA`](struct.DFA.html).
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, dense};
    ///
    /// let dfa = dense::DFA::new("foo[0-9]+")?;
    /// let expected = HalfMatch { pattern: 0, offset: 4 };
    /// assert_eq!(Some(expected), dfa.find_earliest_fwd(b"foo12345")?);
    ///
    /// // Normally, the end of the leftmost first match here would be 3,
    /// // but the shortest match semantics detect a match earlier.
    /// let dfa = dense::DFA::new("abc|a")?;
    /// let expected = HalfMatch { pattern: 0, offset: 1 };
    /// assert_eq!(Some(expected), dfa.find_earliest_fwd(b"abc")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    fn find_earliest_fwd(
        &self,
        bytes: &[u8],
    ) -> Result<Option<HalfMatch>, NoMatch> {
        self.find_earliest_fwd_at(None, bytes, 0, bytes.len())
    }

    #[inline]
    fn find_earliest_rev(
        &self,
        bytes: &[u8],
    ) -> Result<Option<HalfMatch>, NoMatch> {
        self.find_earliest_rev_at(bytes, 0, bytes.len())
    }

    /// Returns the end offset of the longest match. If no match exists,
    /// then `None` is returned.
    ///
    /// Implementors of this trait are not required to implement any particular
    /// match semantics (such as leftmost-first), which are instead manifest in
    /// the DFA's topology itself.
    ///
    /// In particular, this method must continue searching even after it
    /// enters a match state. The search should only terminate once it has
    /// reached the end of the input or when it has entered a dead state. Upon
    /// termination, the position of the last byte seen while still in a match
    /// state is returned.
    ///
    /// # Example
    ///
    /// This example shows how to use this method with a
    /// [`dense::DFA`](struct.DFA.html). By default, a dense DFA uses
    /// "leftmost first" match semantics.
    ///
    /// Leftmost first match semantics corresponds to the match with the
    /// smallest starting offset, but where the end offset is determined by
    /// preferring earlier branches in the original regular expression. For
    /// example, `Sam|Samwise` will match `Sam` in `Samwise`, but `Samwise|Sam`
    /// will match `Samwise` in `Samwise`.
    ///
    /// Generally speaking, the "leftmost first" match is how most backtracking
    /// regular expressions tend to work. This is in contrast to POSIX-style
    /// regular expressions that yield "leftmost longest" matches. Namely,
    /// both `Sam|Samwise` and `Samwise|Sam` match `Samwise` when using
    /// leftmost longest semantics.
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, dense};
    ///
    /// let dfa = dense::DFA::new("foo[0-9]+")?;
    /// let expected = HalfMatch { pattern: 0, offset: 8 };
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo12345")?);
    ///
    /// // Even though a match is found after reading the first byte (`a`),
    /// // the leftmost first match semantics demand that we find the earliest
    /// // match that prefers earlier parts of the pattern over latter parts.
    /// let dfa = dense::DFA::new("abc|a")?;
    /// let expected = HalfMatch { pattern: 0, offset: 3 };
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"abc")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    fn find_leftmost_fwd(
        &self,
        bytes: &[u8],
    ) -> Result<Option<HalfMatch>, NoMatch> {
        self.find_leftmost_fwd_at(None, bytes, 0, bytes.len())
    }

    /// Returns the start offset of the longest match in reverse, by searching
    /// from the end of the input towards the start of the input. If no match
    /// exists, then `None` is returned. In other words, this has the same
    /// match semantics as `find`, but in reverse.
    ///
    /// # Example
    ///
    /// This example shows how to use this method with a
    /// [`dense::DFA`](struct.DFA.html). In particular, this routine
    /// is principally useful when used in conjunction with the
    /// [`dense::Builder::reverse`](dense/struct.Builder.html#method.reverse)
    /// configuration knob. In general, it's unlikely to be correct to use both
    /// `find` and `rfind` with the same DFA since any particular DFA will only
    /// support searching in one direction.
    ///
    /// ```
    /// use regex_automata::nfa::thompson;
    /// use regex_automata::dfa::{Automaton, HalfMatch, dense};
    ///
    /// let dfa = dense::Builder::new()
    ///     .thompson(thompson::Config::new().reverse(true))
    ///     .build("foo[0-9]+")?;
    /// let expected = HalfMatch { pattern: 0, offset: 0 };
    /// assert_eq!(Some(expected), dfa.find_leftmost_rev(b"foo12345")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    fn find_leftmost_rev(
        &self,
        bytes: &[u8],
    ) -> Result<Option<HalfMatch>, NoMatch> {
        self.find_leftmost_rev_at(bytes, 0, bytes.len())
    }

    #[inline]
    fn find_overlapping_fwd(
        &self,
        bytes: &[u8],
        state: &mut State<Self::ID>,
    ) -> Result<Option<HalfMatch>, NoMatch> {
        self.find_overlapping_fwd_at(None, bytes, 0, bytes.len(), state)
    }

    /// Returns the same as `shortest_match`, but starts the search at the
    /// given offset.
    ///
    /// The significance of the starting point is that it takes the surrounding
    /// context into consideration. For example, if the DFA is anchored, then
    /// a match can only occur when `start == 0`.
    #[inline]
    fn find_earliest_fwd_at(
        &self,
        pre: Option<&mut prefilter::Scanner>,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Option<HalfMatch>, NoMatch> {
        search::find_earliest_fwd(pre, self, bytes, start, end)
    }

    #[inline]
    fn find_earliest_rev_at(
        &self,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Option<HalfMatch>, NoMatch> {
        search::find_earliest_rev(self, bytes, start, end)
    }

    /// Returns the same as `find`, but starts the search at the given
    /// offset.
    ///
    /// The significance of the starting point is that it takes the surrounding
    /// context into consideration. For example, if the DFA is anchored, then
    /// a match can only occur when `start == 0`.
    #[inline]
    fn find_leftmost_fwd_at(
        &self,
        pre: Option<&mut prefilter::Scanner>,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Option<HalfMatch>, NoMatch> {
        search::find_leftmost_fwd(pre, self, bytes, start, end)
    }

    /// Returns the same as `rfind`, but starts the search at the given
    /// offset.
    ///
    /// The significance of the starting point is that it takes the surrounding
    /// context into consideration. For example, if the DFA is anchored, then
    /// a match can only occur when `start == bytes.len()`.
    #[inline]
    fn find_leftmost_rev_at(
        &self,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Option<HalfMatch>, NoMatch> {
        search::find_leftmost_rev(self, bytes, start, end)
    }

    #[inline]
    fn find_overlapping_fwd_at(
        &self,
        pre: Option<&mut prefilter::Scanner>,
        bytes: &[u8],
        start: usize,
        end: usize,
        state: &mut State<Self::ID>,
    ) -> Result<Option<HalfMatch>, NoMatch> {
        search::find_overlapping_fwd(pre, self, bytes, start, end, state)
    }
}

unsafe impl<'a, T: Automaton> Automaton for &'a T {
    type ID = T::ID;

    #[inline]
    fn is_match_state(&self, id: Self::ID) -> bool {
        (**self).is_match_state(id)
    }

    #[inline]
    fn is_start_state(&self, id: Self::ID) -> bool {
        (**self).is_start_state(id)
    }

    #[inline]
    fn is_accel_state(&self, id: Self::ID) -> bool {
        (**self).is_accel_state(id)
    }

    #[inline]
    fn is_special_state(&self, id: Self::ID) -> bool {
        (**self).is_special_state(id)
    }

    #[inline]
    fn is_dead_state(&self, id: Self::ID) -> bool {
        (**self).is_dead_state(id)
    }

    #[inline]
    fn is_quit_state(&self, id: Self::ID) -> bool {
        (**self).is_quit_state(id)
    }

    #[inline(always)]
    fn next_state(&self, current: Self::ID, input: u8) -> Self::ID {
        (**self).next_state(current, input)
    }

    #[inline]
    unsafe fn next_state_unchecked(
        &self,
        current: Self::ID,
        input: u8,
    ) -> Self::ID {
        (**self).next_state_unchecked(current, input)
    }

    fn next_eof_state(&self, current: Self::ID) -> Self::ID {
        (**self).next_eof_state(current)
    }

    #[inline]
    fn patterns(&self) -> usize {
        (**self).patterns()
    }

    #[inline]
    fn match_offset(&self) -> usize {
        (**self).match_offset()
    }

    #[inline]
    fn match_count(&self, id: Self::ID) -> usize {
        (**self).match_count(id)
    }

    #[inline]
    fn match_pattern(&self, id: Self::ID, index: usize) -> PatternID {
        (**self).match_pattern(id, index)
    }

    #[inline]
    fn start_state_forward(
        &self,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Self::ID {
        (**self).start_state_forward(bytes, start, end)
    }

    #[inline]
    fn start_state_reverse(
        &self,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Self::ID {
        (**self).start_state_reverse(bytes, start, end)
    }

    fn accelerator(&self, id: Self::ID) -> &[u8] {
        (**self).accelerator(id)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct State<S> {
    id: Option<S>,
    last_match: Option<StateMatch>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StateMatch {
    pub match_index: usize,
    pub offset: usize,
}

impl<S: StateID> State<S> {
    pub fn start() -> State<S> {
        State { id: None, last_match: None }
    }

    pub(crate) fn id(&self) -> Option<S> {
        self.id
    }

    pub(crate) fn set_id(&mut self, id: S) {
        self.id = Some(id);
    }

    pub(crate) fn last_match(&mut self) -> Option<&mut StateMatch> {
        self.last_match.as_mut()
    }

    pub(crate) fn set_last_match(&mut self, last_match: StateMatch) {
        self.last_match = Some(last_match);
    }

    pub(crate) fn clear_last_match(&mut self) {
        self.last_match = None;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Start {
    NonWordByte = 0,
    WordByte = 1,
    Text = 2,
    Line = 3,
}

impl Start {
    pub fn all() -> [Start; 4] {
        [Start::NonWordByte, Start::WordByte, Start::Text, Start::Line]
    }

    pub fn from_usize(n: usize) -> Option<Start> {
        match n {
            0 => Some(Start::NonWordByte),
            1 => Some(Start::WordByte),
            2 => Some(Start::Text),
            3 => Some(Start::Line),
            _ => None,
        }
    }

    pub fn count() -> usize {
        4
    }

    #[inline(always)]
    pub fn from_position_fwd(bytes: &[u8], start: usize, end: usize) -> Start {
        if start == 0 {
            Start::Text
        } else if bytes[start - 1] == b'\n' {
            Start::Line
        } else if is_word_byte(bytes[start - 1]) {
            Start::WordByte
        } else {
            Start::NonWordByte
        }
    }

    #[inline(always)]
    pub fn from_position_rev(bytes: &[u8], start: usize, end: usize) -> Start {
        if end == bytes.len() {
            Start::Text
        } else if bytes[end] == b'\n' {
            Start::Line
        } else if is_word_byte(bytes[end]) {
            Start::WordByte
        } else {
            Start::NonWordByte
        }
    }

    #[inline(always)]
    pub fn as_usize(&self) -> usize {
        *self as usize
    }
}

pub fn fmt_state_indicator<A: Automaton>(
    f: &mut core::fmt::Formatter<'_>,
    dfa: A,
    id: A::ID,
) -> core::fmt::Result {
    if dfa.is_dead_state(id) {
        write!(f, "D")?;
        if dfa.is_start_state(id) {
            write!(f, ">")?;
        } else {
            write!(f, " ")?;
        }
    } else if dfa.is_quit_state(id) {
        write!(f, "Q ")?;
    } else if dfa.is_start_state(id) {
        if dfa.is_accel_state(id) {
            write!(f, "A>")?;
        } else {
            write!(f, " >")?;
        }
    } else if dfa.is_match_state(id) {
        if dfa.is_accel_state(id) {
            write!(f, "A*")?;
        } else {
            write!(f, " *")?;
        }
    } else if dfa.is_accel_state(id) {
        write!(f, "A ")?;
    } else {
        write!(f, "  ")?;
    }
    Ok(())
}
