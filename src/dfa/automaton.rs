use crate::{
    dfa::search,
    prefilter::{self, Prefilter},
    state_id::StateID,
    word::is_word_byte,
    MatchError, PatternID,
};

/// The offset, in bytes, that a match is delayed by in the DFAs generated
/// by this crate.
///
/// The purpose of this delay is to support look-around such as \b (ASCII-only)
/// and $. In particular, both of these operators may require the
/// identification of the end of input in order to confirm a match. Not only
/// does this mean that all matches must therefore be delayed by a single byte,
/// but that a special EOF value is added to the alphabet of all DFAs. (Which
/// means that even though the alphabet of a DFA is all byte values, the actual
/// maximum alphabet size is 257 due to the extra EOF value.)
///
/// Since we delay matches by only 1 byte, this can't fully support a
/// Unicode-aware \b operator. Indeed, DFAs in this crate do not support
/// it. (It's not as simple as just increasing the match offset to do
/// it---otherwise we would---but building the full Unicode-aware word boundary
/// detection into an automaton is quite tricky.)
pub const MATCH_OFFSET: usize = 1;

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
    pub(crate) pattern: PatternID,
    /// The offset of the match.
    ///
    /// For forward searches, the offset is exclusive. For reverse searches,
    /// the offset is inclusive.
    pub(crate) offset: usize,
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
    /// originally inserted into the corresponding DFA. The first pattern has
    /// identifier `0`, and each subsequent pattern is `1`, `2` and so on.
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
/// The complexity of this trait probably means that it's unlikely for others
/// to implement it. The primary purpose of the trait is to provide for a way
/// of abstracting over different types of DFAs. In this crate, that means
/// dense DFAs and sparse DFAs. (Dense DFAs are fast but memory hungry, where
/// as sparse DFAs are slower but come with a smaller memory footprint. But
/// they otherwise provide exactly equivalent expressive power.) For example, a
/// [`dfa::Regex`](struct.Regex.html) is generic over this trait.
///
/// Normally, a DFA's execution model is very simple. You might have a single
/// start state, zero or more final or "match" states and a function that
/// transitions from one state to the next given the next byte of input.
/// Unfortunately, the interface described by this trait is significantly
/// more complicated than this. The complexity has a number of different
/// reasons, mostly motivated by performance, functionality or space savings:
///
/// * A DFA can search for multiple patterns simultaneously. This
/// means extra information is returned when a match occurs. Namely,
/// a match is not just an offset, but an offset plus a pattern ID.
/// [`Automaton::pattern_count`](trait.Automaton.html#tymethod.pattern_count)
/// returns the number of patterns compiled into the DFA,
/// [`Automaton::match_count`](trait.Automaton.html#tymethod.match_count)
/// returns the total number of patterns that match in a particular state and
/// [`Automaton::match_pattern`](trait.Automaton.html#tymethod.match_pattern)
/// permits iterating over the patterns that match in a particular state.
/// * A DFA can have multiple start states, and the choice of which
/// start state to use depends on the content of the string being searched and
/// position of the search, as well as whether the search is an anchored search
/// for a specific pattern in the DFA. Moreover, computing the start state also
/// depends on whether you're doing a forward or a reverse search.
/// [`Automaton::start_state_forward`](trait.Automaton.html#tymethod.start_state_forward)
/// and
/// [`Automaton::start_state_reverse`](trait.Automaton.html#tymethod.start_state_reverse)
/// are used to compute the start state for forward and reverse searches,
/// respectively.
/// * All matches are delayed by one byte to support things like `$` and `\b`
/// at the end of a pattern. Therefore, every use of a DFA is required to use
/// [`Automaton::next_eof_state`](trait.Automaton.html#tymethod.next_eof_state)
/// at the end of the search to compute the final transition.
/// * For optimization reasons, some states are treated specially. Every state
/// is either special or not, which can be determined via the
/// [`Automaton::is_special_state`](trait.Automaton.html#tymethod.is_special_state)
/// method. If it's special, then the state must be at least one of a few
/// possible types of states. (Note that some types can overlap, for example,
/// a match state can also be an accel state. But some types can't. If a state
/// is a dead state, then it can never be any other type of state.) Those
/// types are:
///     * A dead state. A dead state means the DFA will never enter a match
///     state.
///     This can be queried via the
///     [`Automaton::is_dead_state`](trait.Automaton.html#tymethod.is_dead_state)
///     method.
///     * A quit state. A quit state occurs if the DFA had to stop the search
///     prematurely for some reason.
///     This can be queried via the
///     [`Automaton::is_quit_state`](trait.Automaton.html#tymethod.is_quit_state)
///     method.
///     * A match state. A match state occurs when a match is found. When a DFA
///     enters a match state, the search may stop immediately (when looking for
///     the earliest match), or it may continue to find the leftmost-first
///     match.
///     This can be queried via the
///     [`Automaton::is_match_state`](trait.Automaton.html#tymethod.is_match_state)
///     method.
///     * A start state. A start state is where a search begins. For every
///     search, there is exactly one start state that is used, however, a DFA
///     may contain many start states. When the search is in a start state, it
///     may use a prefilter to quickly skip to candidate matches without
///     executing the DFA on every byte.
///     This can be queried via the
///     [`Automaton::is_start_state`](trait.Automaton.html#tymethod.is_start_state)
///     method.
///     * An accel state. An accel state is a state that is accelerated. That
///     is, it is a state where _most_ of its transitions loop back to itself
///     and only a small number of transitions lead to other states. This kind
///     of state is said to be accelerated because a search routine can quickly
///     look for the bytes leading out of the state instead of continuing to
///     execute the DFA on each byte.
///     This can be queried via the
///     [`Automaton::is_accel_state`](trait.Automaton.html#tymethod.is_accel_state)
///     method. And the bytes that lead out of the state can be queried via the
///     [`Automaton::accelerator`](trait.Automaton.html#tymethod.accelerator)
///     method.
///
/// There are a number of provided methods on this trait that implement
/// efficient searching (for forwards and backwards) with a DFA using all of
/// the above features of this trait. In particular, given the complexity of
/// all these features, implementing a search routine in this trait is not
/// straight forward. If you need to do this for specialized reasons, then
/// it's recommended to look at the source of this crate. It is intentionally
/// well commented to help with this. With that said, it is possible to
/// somewhat simplify the search routine. For example, handling accelerated
/// states is strictly optional, since it is always correct to assume that
/// `Automaton::is_accel_state` returns false. However, one complex part of
/// writing a search routine using this trait is handling the 1-byte delay of a
/// match. That is not optional.
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
    /// This is required to be one of `u8`, `u16`, `u32`, `u64` or `usize`.
    /// The restriction on this representation permits aspects of the
    /// implementation to make assumptions about the state ID representation,
    /// which are necessary for providing cheap deserialization of dense and
    /// sparse DFAs.
    ///
    /// This restriction does unfortunately limit the ability to develop more
    /// expressive APIs, such as providing intersection, union and complement
    /// adapters.
    ///
    /// If you have a use case for relaxing this restriction, please propose a
    /// design on the issue tracker.
    type ID: StateID;

    /// Transitions from the current state to the next state, given the next
    /// byte of input.
    ///
    /// Implementations must guarantee that the returned ID is always a valid
    /// ID when `current` refers to a valid ID. Moreover, the transition
    /// function must be defined for all possible values of `input`.
    ///
    /// # Panics
    ///
    /// If the given ID does not refer to a valid state, then this routine may
    /// panic but it also may not panic and return an incorrect ID. However, an
    /// incorrect ID may never sacrifice memory safety.
    ///
    /// # Example
    ///
    /// This shows a simplistic example for walking a DFA for a given haystack
    /// by using the `next_state` method.
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, dense};
    ///
    /// let dfa = dense::DFA::new(r"[a-z]+r")?;
    /// let haystack = "bar".as_bytes();
    ///
    /// // The start state is determined by inspecting the position and the
    /// // initial bytes of the haystack.
    /// let mut state = dfa.start_state_forward(
    ///     None, haystack, 0, haystack.len(),
    /// );
    /// // Walk all the bytes in the haystack.
    /// for &b in haystack {
    ///     state = dfa.next_state(state, b);
    /// }
    /// // Matches are always delayed by 1 byte, so we must explicitly walk the
    /// // special "EOF" transition at the end of the search.
    /// state = dfa.next_eof_state(state);
    /// assert!(dfa.is_match_state(state));
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn next_state(&self, current: Self::ID, input: u8) -> Self::ID;

    /// Like `next_state`, but its implementation may look up the next state
    /// without memory safety checks such as bounds checks. As such, callers
    /// must ensure that the given identifier corresponds to a valid DFA
    /// state. Implementors must, in turn, ensure that this routine is safe
    /// for all valid state identifiers and for all possible `u8` values.

    /// Transitions from the current state to the next state, given the next
    /// byte of input.
    ///
    /// Unlike
    /// [`Automaton::next_state`](trait.Automaton.html#tymethod.next_state),
    /// implementations may implement this more efficiently by assuming that
    /// the `current` state ID is valid. Typically, this manifests by eliding
    /// bounds checks.
    ///
    /// # Safety
    ///
    /// Callers of this method must guarantee that `current` refers to a valid
    /// state ID. If `current` is not a valid state ID, then calling this
    /// routine may result in undefined behavior.
    ///
    /// If `current` is valid, then implementations must guarantee that the ID
    /// returned is valid for all possible values of `input`.
    unsafe fn next_state_unchecked(
        &self,
        current: Self::ID,
        input: u8,
    ) -> Self::ID;

    /// Transitions from the current state to the next state for the special
    /// EOF symbol.
    ///
    /// Implementations must guarantee that the returned ID is always a valid
    /// ID when `current` refers to a valid ID.
    ///
    /// This routine must be called at the end of every search in a correct
    /// implementation. Namely, DFAs in this crate delay matches by one byte
    /// in order to support look-around operators. Thus, after reaching the end
    /// of a haystack, a search implementation must follow one last EOF
    /// transition.
    ///
    /// It is best to think of EOF as an additional symbol in the alphabet of
    /// a DFA that is distinct from every other symbol. That is, the alphabet
    /// of DFAs in this crate has a logical size 257 instead of 256, where 256
    /// corresponds to every possible inhabitant of `u8`. (In practice, the
    /// physical alphabet size may be smaller because of alphabet compression
    /// via equivalence classes, but EOF is always represented somehow in the
    /// alphabet.)
    ///
    /// # Panics
    ///
    /// If the given ID does not refer to a valid state, then this routine may
    /// panic but it also may not panic and return an incorrect ID. However, an
    /// incorrect ID may never sacrifice memory safety.
    ///
    /// # Example
    ///
    /// This shows a simplistic example for walking a DFA for a given haystack,
    /// and then finishing the search with the final EOF transition.
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, dense};
    ///
    /// let dfa = dense::DFA::new(r"[a-z]+r")?;
    /// let haystack = "bar".as_bytes();
    ///
    /// // The start state is determined by inspecting the position and the
    /// // initial bytes of the haystack.
    /// let mut state = dfa.start_state_forward(
    ///     None, haystack, 0, haystack.len(),
    /// );
    /// // Walk all the bytes in the haystack.
    /// for &b in haystack {
    ///     state = dfa.next_state(state, b);
    /// }
    /// // Matches are always delayed by 1 byte, so we must explicitly walk
    /// // the special "EOF" transition at the end of the search. Without this
    /// // final transition, the assert below will fail since the DFA will not
    /// // have entered a match state yet!
    /// state = dfa.next_eof_state(state);
    /// assert!(dfa.is_match_state(state));
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn next_eof_state(&self, current: Self::ID) -> Self::ID;

    /// Return the identifier of this DFA's start state for the given haystack
    /// when matching in the forward direction.
    fn start_state_forward(
        &self,
        pattern_id: Option<PatternID>,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Self::ID;

    /// Return the identifier of this DFA's start state for the given haystack
    /// when matching in the reverse direction.
    fn start_state_reverse(
        &self,
        pattern_id: Option<PatternID>,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Self::ID;

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

    /// Returns the total number of patterns compiled into this DFA.
    ///
    /// In the case of a DFA that never matches any pattern, this should
    /// return `0`.
    fn pattern_count(&self) -> usize;

    /// Returns the total number of patterns that match in this state.
    ///
    /// If the given state is not a match state, then this always returns zero.
    fn match_count(&self, id: Self::ID) -> usize;

    /// Returns the pattern ID corresponding to the given match index in the
    /// given state. This must panic if the given match index is out of bounds
    /// or if the given state ID does not correspond to a match state.
    fn match_pattern(&self, id: Self::ID, index: usize) -> PatternID;

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
    /// let expected = HalfMatch::new(0, 4);
    /// assert_eq!(Some(expected), dfa.find_earliest_fwd(b"foo12345")?);
    ///
    /// // Normally, the end of the leftmost first match here would be 3,
    /// // but the shortest match semantics detect a match earlier.
    /// let dfa = dense::DFA::new("abc|a")?;
    /// let expected = HalfMatch::new(0, 1);
    /// assert_eq!(Some(expected), dfa.find_earliest_fwd(b"abc")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    fn find_earliest_fwd(
        &self,
        bytes: &[u8],
    ) -> Result<Option<HalfMatch>, MatchError> {
        self.find_earliest_fwd_at(None, None, bytes, 0, bytes.len())
    }

    #[inline]
    fn find_earliest_rev(
        &self,
        bytes: &[u8],
    ) -> Result<Option<HalfMatch>, MatchError> {
        self.find_earliest_rev_at(None, bytes, 0, bytes.len())
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
    /// let expected = HalfMatch::new(0, 8);
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo12345")?);
    ///
    /// // Even though a match is found after reading the first byte (`a`),
    /// // the leftmost first match semantics demand that we find the earliest
    /// // match that prefers earlier parts of the pattern over latter parts.
    /// let dfa = dense::DFA::new("abc|a")?;
    /// let expected = HalfMatch::new(0, 3);
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"abc")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    fn find_leftmost_fwd(
        &self,
        bytes: &[u8],
    ) -> Result<Option<HalfMatch>, MatchError> {
        self.find_leftmost_fwd_at(None, None, bytes, 0, bytes.len())
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
    /// let expected = HalfMatch::new(0, 0);
    /// assert_eq!(Some(expected), dfa.find_leftmost_rev(b"foo12345")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    fn find_leftmost_rev(
        &self,
        bytes: &[u8],
    ) -> Result<Option<HalfMatch>, MatchError> {
        self.find_leftmost_rev_at(None, bytes, 0, bytes.len())
    }

    #[inline]
    fn find_overlapping_fwd(
        &self,
        bytes: &[u8],
        state: &mut State<Self::ID>,
    ) -> Result<Option<HalfMatch>, MatchError> {
        self.find_overlapping_fwd_at(None, None, bytes, 0, bytes.len(), state)
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
        pattern_id: Option<PatternID>,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Option<HalfMatch>, MatchError> {
        search::find_earliest_fwd(pre, self, pattern_id, bytes, start, end)
    }

    #[inline]
    fn find_earliest_rev_at(
        &self,
        pattern_id: Option<PatternID>,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Option<HalfMatch>, MatchError> {
        search::find_earliest_rev(self, pattern_id, bytes, start, end)
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
        pattern_id: Option<PatternID>,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Option<HalfMatch>, MatchError> {
        search::find_leftmost_fwd(pre, self, pattern_id, bytes, start, end)
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
        pattern_id: Option<PatternID>,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Option<HalfMatch>, MatchError> {
        search::find_leftmost_rev(self, pattern_id, bytes, start, end)
    }

    #[inline]
    fn find_overlapping_fwd_at(
        &self,
        pre: Option<&mut prefilter::Scanner>,
        pattern_id: Option<PatternID>,
        bytes: &[u8],
        start: usize,
        end: usize,
        state: &mut State<Self::ID>,
    ) -> Result<Option<HalfMatch>, MatchError> {
        search::find_overlapping_fwd(
            pre, self, pattern_id, bytes, start, end, state,
        )
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
    fn pattern_count(&self) -> usize {
        (**self).pattern_count()
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
        pattern_id: Option<PatternID>,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Self::ID {
        (**self).start_state_forward(pattern_id, bytes, start, end)
    }

    #[inline]
    fn start_state_reverse(
        &self,
        pattern_id: Option<PatternID>,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Self::ID {
        (**self).start_state_reverse(pattern_id, bytes, start, end)
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
    pub(crate) match_index: usize,
    pub(crate) offset: usize,
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
