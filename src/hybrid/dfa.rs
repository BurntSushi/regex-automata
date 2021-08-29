use core::{borrow::Borrow, iter, mem::size_of};

use alloc::{sync::Arc, vec::Vec};

use crate::{
    hybrid::{
        error::{BuildError, CacheError},
        id::{LazyStateID, LazyStateIDError, OverlappingState},
        search,
    },
    nfa::thompson,
    util::{
        alphabet::{self, ByteClasses, ByteSet},
        determinize::{self, State, StateBuilderEmpty, StateBuilderNFA},
        id::{PatternID, StateID as NFAStateID},
        matchtypes::{HalfMatch, MatchError, MatchKind},
        prefilter,
        sparse_set::SparseSets,
        start::Start,
    },
};

/// The mininum number of states that a lazy DFA's cache size must support.
///
/// This is checked at time of construction to ensure that at least some small
/// number of states can fit in the given capacity allotment. If we can't fit
/// at least this number of states, then the thinking is that it's pretty
/// senseless to use the lazy DFA. More to the point, parts of the code do
/// assume that the cache can fit at least some small number of states.
const MIN_STATES: usize = 5;

#[derive(Clone, Debug)]
pub struct DFA {
    nfa: Arc<thompson::NFA>,
    stride2: usize,
    classes: ByteClasses,
    quitset: ByteSet,
    anchored: bool,
    match_kind: MatchKind,
    starts_for_each_pattern: bool,
    cache_capacity: usize,
    minimum_cache_clear_count: Option<usize>,
}

impl DFA {
    /// Parse the given regular expression using a default configuration and
    /// return the corresponding lazy DFA.
    ///
    /// If you want a non-default configuration, then use the [`Builder`] to
    /// set your own configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{hybrid::dfa::DFA, HalfMatch};
    ///
    /// let dfa = DFA::new("foo[0-9]+bar")?;
    /// let mut cache = dfa.create_cache();
    ///
    /// let expected = HalfMatch::must(0, 11);
    /// assert_eq!(
    ///     Some(expected),
    ///     dfa.find_leftmost_fwd(&mut cache, b"foo12345bar")?,
    /// );
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(pattern: &str) -> Result<DFA, BuildError> {
        DFA::builder().build(pattern)
    }

    /// Parse the given regular expressions using a default configuration and
    /// return the corresponding lazy multi-DFA.
    ///
    /// If you want a non-default configuration, then use the [`Builder`] to
    /// set your own configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{hybrid::dfa::DFA, HalfMatch};
    ///
    /// let dfa = DFA::new_many(&["[0-9]+", "[a-z]+"])?;
    /// let mut cache = dfa.create_cache();
    ///
    /// let expected = HalfMatch::must(1, 3);
    /// assert_eq!(
    ///     Some(expected),
    ///     dfa.find_leftmost_fwd(&mut cache, b"foo12345bar")?,
    /// );
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new_many<P: AsRef<str>>(patterns: &[P]) -> Result<DFA, BuildError> {
        DFA::builder().build_many(patterns)
    }

    /// Create a new lazy DFA that matches every input.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{hybrid::dfa::DFA, HalfMatch};
    ///
    /// let dfa = DFA::always_match()?;
    /// let mut cache = dfa.create_cache();
    ///
    /// let expected = HalfMatch::must(0, 0);
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(&mut cache, b"")?);
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(&mut cache, b"foo")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn always_match() -> Result<DFA, BuildError> {
        let nfa = thompson::NFA::always_match();
        Builder::new().build_from_nfa(Arc::new(nfa))
    }

    /// Create a new lazy DFA that never matches any input.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::hybrid::dfa::DFA;
    ///
    /// let dfa = DFA::never_match()?;
    /// let mut cache = dfa.create_cache();
    ///
    /// assert_eq!(None, dfa.find_leftmost_fwd(&mut cache, b"")?);
    /// assert_eq!(None, dfa.find_leftmost_fwd(&mut cache, b"foo")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn never_match() -> Result<DFA, BuildError> {
        let nfa = thompson::NFA::never_match();
        Builder::new().build_from_nfa(Arc::new(nfa))
    }

    /// Return a default configuration for a `DFA`.
    ///
    /// This is a convenience routine to avoid needing to import the `Config`
    /// type when customizing the construction of a lazy DFA.
    ///
    /// # Example
    ///
    /// This example shows how to build a lazy DFA that only executes searches
    /// in anchored mode.
    ///
    /// ```
    /// use regex_automata::{hybrid::dfa::DFA, HalfMatch};
    ///
    /// let re = DFA::builder()
    ///     .configure(DFA::config().anchored(true))
    ///     .build(r"[0-9]+")?;
    /// let mut cache = re.create_cache();
    ///
    /// let haystack = "abc123xyz".as_bytes();
    /// assert_eq!(None, re.find_leftmost_fwd(&mut cache, haystack)?);
    /// assert_eq!(
    ///     Some(HalfMatch::must(0, 3)),
    ///     re.find_leftmost_fwd(&mut cache, &haystack[3..6])?,
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn config() -> Config {
        Config::new()
    }

    /// Return a builder for configuring the construction of a `Regex`.
    ///
    /// This is a convenience routine to avoid needing to import the
    /// [`Builder`] type in common cases.
    ///
    /// # Example
    ///
    /// This example shows how to use the builder to disable UTF-8 mode
    /// everywhere for lazy DFAs. This includes disabling it for both the
    /// concrete syntax (e.g., `.` matches any byte and Unicode character
    /// classes like `\p{Letter}` are not allowed) and for the unanchored
    /// search prefix. The latter enables the regex to match anywhere in a
    /// sequence of arbitrary bytes. (Typically, the unanchored search prefix
    /// will only permit matching valid UTF-8.)
    ///
    /// ```
    /// use regex_automata::{
    ///     hybrid::dfa::DFA,
    ///     nfa::thompson,
    ///     HalfMatch, SyntaxConfig,
    /// };
    ///
    /// let re = DFA::builder()
    ///     .syntax(SyntaxConfig::new().utf8(false))
    ///     .thompson(thompson::Config::new().utf8(false))
    ///     .build(r"foo(?-u:[^b])ar.*")?;
    /// let mut cache = re.create_cache();
    ///
    /// let haystack = b"\xFEfoo\xFFarzz\xE2\x98\xFF\n";
    /// let expected = Some(HalfMatch::must(0, 9));
    /// let got = re.find_leftmost_fwd(&mut cache, haystack)?;
    /// assert_eq!(expected, got);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn builder() -> Builder {
        Builder::new()
    }

    pub fn create_cache(&self) -> Cache {
        Cache::new(self)
    }

    pub fn reset_cache(&self, cache: &mut Cache) {
        Lazy::new(self, cache).reset_cache()
    }

    /// Returns the total number of patterns compiled into this lazy DFA.
    ///
    /// In the case of a DFA that contains no patterns, this returns `0`.
    ///
    /// # Example
    ///
    /// This example shows the pattern count for a DFA that never matches:
    ///
    /// ```
    /// use regex_automata::hybrid::dfa::DFA;
    ///
    /// let dfa = DFA::never_match()?;
    /// assert_eq!(dfa.pattern_count(), 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// And another example for a DFA that matches at every position:
    ///
    /// ```
    /// use regex_automata::hybrid::dfa::DFA;
    ///
    /// let dfa = DFA::always_match()?;
    /// assert_eq!(dfa.pattern_count(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// And finally, a DFA that was constructed from multiple patterns:
    ///
    /// ```
    /// use regex_automata::hybrid::dfa::DFA;
    ///
    /// let dfa = DFA::new_many(&["[0-9]+", "[a-z]+", "[A-Z]+"])?;
    /// assert_eq!(dfa.pattern_count(), 3);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn pattern_count(&self) -> usize {
        self.nfa.match_len()
    }

    /// Returns a reference to the underlying NFA.
    pub fn nfa(&self) -> &Arc<thompson::NFA> {
        &self.nfa
    }

    /// Returns the stride, as a base-2 exponent, required for these
    /// equivalence classes.
    ///
    /// The stride is always the smallest power of 2 that is greater than or
    /// equal to the alphabet length. This is done so that converting between
    /// state IDs and indices can be done with shifts alone, which is much
    /// faster than integer division.
    fn stride2(&self) -> usize {
        self.stride2
    }

    /// Returns the total stride for every state in this lazy DFA. This
    /// corresponds to the total number of transitions used by each state in
    /// this DFA's transition table.
    fn stride(&self) -> usize {
        1 << self.stride2()
    }

    /// Returns the total number of elements in the alphabet for this
    /// transition table. This is always less than or equal to `self.stride()`.
    /// It is only equal when the alphabet length is a power of 2. Otherwise,
    /// it is always strictly less.
    fn alphabet_len(&self) -> usize {
        self.classes.alphabet_len()
    }

    /// Returns the memory usage, in bytes, of this lazy DFA.
    ///
    /// This does **not** include the stack size used up by this lazy DFA. To
    /// compute that, use `std::mem::size_of::<DFA>()`. This also does
    /// not include the size of the `Cache` used.
    pub fn memory_usage(&self) -> usize {
        // Everything else is on the stack.
        self.nfa.memory_usage()
    }
}

impl DFA {
    /// Executes a forward search and returns the end position of the first
    /// match that is found as early as possible. If no match exists, then
    /// `None` is returned.
    ///
    /// This routine stops scanning input as soon as the search observes a
    /// match state. This is useful for implementing boolean `is_match`-like
    /// routines, where as little work is done as possible.
    ///
    /// See [`DFA::find_earliest_fwd_at`] for additional functionality, such as
    /// providing a prefilter, a specific pattern to match and the bounds of
    /// the search within the haystack. This routine is meant as a convenience
    /// for common cases where the additional functionality is not needed.
    ///
    /// # Errors
    ///
    /// This routine only errors if the search could not complete. For
    /// lazy DFAs generated by this crate, this only occurs in non-default
    /// configurations where quit bytes are used, Unicode word boundaries are
    /// heuristically enabled or limits are set on the number of times the lazy
    /// DFA's cache may be cleared.
    ///
    /// When a search cannot complete, callers cannot know whether a match
    /// exists or not.
    ///
    /// # Example
    ///
    /// This example demonstrates how the position returned might differ from
    /// what one might expect when executing a traditional leftmost search.
    ///
    /// ```
    /// use regex_automata::{hybrid::dfa::DFA, HalfMatch};
    ///
    /// let dfa = DFA::new("foo[0-9]+")?;
    /// let mut cache = dfa.create_cache();
    /// // Normally, the end of the leftmost first match here would be 8,
    /// // corresponding to the end of the input. But the "earliest" semantics
    /// // this routine cause it to stop as soon as a match is known, which
    /// // occurs once 'foo[0-9]' has matched.
    /// let expected = HalfMatch::must(0, 4);
    /// assert_eq!(
    ///     Some(expected),
    ///     dfa.find_earliest_fwd(&mut cache, b"foo12345")?,
    /// );
    ///
    /// let dfa = DFA::new("abc|a")?;
    /// let mut cache = dfa.create_cache();
    /// // Normally, the end of the leftmost first match here would be 3,
    /// // but the shortest match semantics detect a match earlier.
    /// let expected = HalfMatch::must(0, 1);
    /// assert_eq!(Some(expected), dfa.find_earliest_fwd(&mut cache, b"abc")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn find_earliest_fwd(
        &self,
        cache: &mut Cache,
        bytes: &[u8],
    ) -> Result<Option<HalfMatch>, MatchError> {
        self.find_earliest_fwd_at(cache, None, None, bytes, 0, bytes.len())
    }

    /// Executes a reverse search and returns the start position of the first
    /// match that is found as early as possible. If no match exists, then
    /// `None` is returned.
    ///
    /// This routine stops scanning input as soon as the search observes a
    /// match state.
    ///
    /// Note that while it is not technically necessary to build a reverse
    /// automaton to use a reverse search, it is likely that you'll want to do
    /// so. Namely, the typical use of a reverse search is to find the starting
    /// location of a match once its end is discovered from a forward search. A
    /// reverse DFA automaton can be built by configuring the intermediate NFA
    /// to be reversed via
    /// [`nfa::thompson::Config::reverse`](crate::nfa::thompson::Config::reverse).
    ///
    /// # Errors
    ///
    /// This routine only errors if the search could not complete. For
    /// lazy DFAs generated by this crate, this only occurs in non-default
    /// configurations where quit bytes are used, Unicode word boundaries are
    /// heuristically enabled or limits are set on the number of times the lazy
    /// DFA's cache may be cleared.
    ///
    /// When a search cannot complete, callers cannot know whether a match
    /// exists or not.
    ///
    /// # Example
    ///
    /// This example demonstrates how the position returned might differ from
    /// what one might expect when executing a traditional leftmost reverse
    /// search.
    ///
    /// ```
    /// use regex_automata::{hybrid::dfa::DFA, nfa::thompson, HalfMatch};
    ///
    /// let dfa = DFA::builder()
    ///     .thompson(thompson::Config::new().reverse(true))
    ///     .build("[a-z]+[0-9]+")?;
    /// let mut cache = dfa.create_cache();
    /// // Normally, the end of the leftmost first match here would be 0,
    /// // corresponding to the beginning of the input. But the "earliest"
    /// // semantics of this routine cause it to stop as soon as a match is
    /// // known, which occurs once '[a-z][0-9]+' has matched.
    /// let expected = HalfMatch::must(0, 2);
    /// assert_eq!(
    ///     Some(expected),
    ///     dfa.find_earliest_rev(&mut cache, b"foo12345")?,
    /// );
    ///
    /// let dfa = DFA::builder()
    ///     .thompson(thompson::Config::new().reverse(true))
    ///     .build("abc|c")?;
    /// let mut cache = dfa.create_cache();
    /// // Normally, the end of the leftmost first match here would be 0,
    /// // but the shortest match semantics detect a match earlier.
    /// let expected = HalfMatch::must(0, 2);
    /// assert_eq!(Some(expected), dfa.find_earliest_rev(&mut cache, b"abc")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn find_earliest_rev(
        &self,
        cache: &mut Cache,
        bytes: &[u8],
    ) -> Result<Option<HalfMatch>, MatchError> {
        self.find_earliest_rev_at(cache, None, bytes, 0, bytes.len())
    }

    /// Executes a forward search and returns the end position of the leftmost
    /// match that is found. If no match exists, then `None` is returned.
    ///
    /// In particular, this method continues searching even after it enters
    /// a match state. The search only terminates once it has reached the
    /// end of the input or when it has entered a dead or quit state. Upon
    /// termination, the position of the last byte seen while still in a match
    /// state is returned.
    ///
    /// # Errors
    ///
    /// This routine only errors if the search could not complete. For
    /// lazy DFAs generated by this crate, this only occurs in non-default
    /// configurations where quit bytes are used, Unicode word boundaries are
    /// heuristically enabled or limits are set on the number of times the lazy
    /// DFA's cache may be cleared.
    ///
    /// When a search cannot complete, callers cannot know whether a match
    /// exists or not.
    ///
    /// # Example
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
    /// leftmost longest semantics. (This crate does not currently support
    /// leftmost longest semantics.)
    ///
    /// ```
    /// use regex_automata::{hybrid::dfa::DFA, HalfMatch};
    ///
    /// let dfa = DFA::new("foo[0-9]+")?;
    /// let mut cache = dfa.create_cache();
    /// let expected = HalfMatch::must(0, 8);
    /// assert_eq!(
    ///     Some(expected),
    ///     dfa.find_leftmost_fwd(&mut cache, b"foo12345")?,
    /// );
    ///
    /// // Even though a match is found after reading the first byte (`a`),
    /// // the leftmost first match semantics demand that we find the earliest
    /// // match that prefers earlier parts of the pattern over latter parts.
    /// let dfa = DFA::new("abc|a")?;
    /// let mut cache = dfa.create_cache();
    /// let expected = HalfMatch::must(0, 3);
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(&mut cache, b"abc")?);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn find_leftmost_fwd(
        &self,
        cache: &mut Cache,
        bytes: &[u8],
    ) -> Result<Option<HalfMatch>, MatchError> {
        self.find_leftmost_fwd_at(cache, None, None, bytes, 0, bytes.len())
    }

    /// Executes a reverse search and returns the start of the position of the
    /// leftmost match that is found. If no match exists, then `None` is
    /// returned.
    ///
    /// In particular, this method continues searching even after it enters
    /// a match state. The search only terminates once it has reached the
    /// end of the input or when it has entered a dead or quit state. Upon
    /// termination, the position of the last byte seen while still in a match
    /// state is returned.
    ///
    /// # Errors
    ///
    /// This routine only errors if the search could not complete. For
    /// lazy DFAs generated by this crate, this only occurs in non-default
    /// configurations where quit bytes are used, Unicode word boundaries are
    /// heuristically enabled or limits are set on the number of times the lazy
    /// DFA's cache may be cleared.
    ///
    /// When a search cannot complete, callers cannot know whether a match
    /// exists or not.
    ///
    /// # Example
    ///
    /// In particular, this routine is principally
    /// useful when used in conjunction with the
    /// [`nfa::thompson::Config::reverse`](crate::nfa::thompson::Config::revers
    /// e) configuration. In general, it's unlikely to be correct to use both
    /// `find_leftmost_fwd` and `find_leftmost_rev` with the same DFA since
    /// any particular DFA will only support searching in one direction with
    /// respect to the pattern.
    ///
    /// ```
    /// use regex_automata::{nfa::thompson, hybrid::dfa::DFA, HalfMatch};
    ///
    /// let dfa = DFA::builder()
    ///     .thompson(thompson::Config::new().reverse(true))
    ///     .build("foo[0-9]+")?;
    /// let mut cache = dfa.create_cache();
    /// let expected = HalfMatch::must(0, 0);
    /// assert_eq!(
    ///     Some(expected),
    ///     dfa.find_leftmost_rev(&mut cache, b"foo12345")?,
    /// );
    ///
    /// // Even though a match is found after reading the last byte (`c`),
    /// // the leftmost first match semantics demand that we find the earliest
    /// // match that prefers earlier parts of the pattern over latter parts.
    /// let dfa = DFA::builder()
    ///     .thompson(thompson::Config::new().reverse(true))
    ///     .build("abc|c")?;
    /// let mut cache = dfa.create_cache();
    /// let expected = HalfMatch::must(0, 0);
    /// assert_eq!(Some(expected), dfa.find_leftmost_rev(&mut cache, b"abc")?);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn find_leftmost_rev(
        &self,
        cache: &mut Cache,
        bytes: &[u8],
    ) -> Result<Option<HalfMatch>, MatchError> {
        self.find_leftmost_rev_at(cache, None, bytes, 0, bytes.len())
    }

    /// Executes an overlapping forward search and returns the end position of
    /// matches as they are found. If no match exists, then `None` is returned.
    ///
    /// This routine is principally only useful when searching for multiple
    /// patterns on inputs where multiple patterns may match the same regions
    /// of text. In particular, callers must preserve the automaton's search
    /// state from prior calls so that the implementation knows where the last
    /// match occurred.
    ///
    /// # Errors
    ///
    /// This routine only errors if the search could not complete. For
    /// lazy DFAs generated by this crate, this only occurs in non-default
    /// configurations where quit bytes are used, Unicode word boundaries are
    /// heuristically enabled or limits are set on the number of times the lazy
    /// DFA's cache may be cleared.
    ///
    /// When a search cannot complete, callers cannot know whether a match
    /// exists or not.
    ///
    /// # Example
    ///
    /// This example shows how to run a basic overlapping search. Notice
    /// that we build the automaton with a `MatchKind::All` configuration.
    /// Overlapping searches are unlikely to work as one would expect when
    /// using the default `MatchKind::LeftmostFirst` match semantics, since
    /// leftmost-first matching is fundamentally incompatible with overlapping
    /// searches. Namely, overlapping searches need to report matches as they
    /// are seen, where as leftmost-first searches will continue searching even
    /// after a match has been observed in order to find the conventional end
    /// position of the match. More concretely, leftmost-first searches use
    /// dead states to terminate a search after a specific match can no longer
    /// be extended. Overlapping searches instead do the opposite by continuing
    /// the search to find totally new matches (potentially of other patterns).
    ///
    /// ```
    /// use regex_automata::{
    ///     hybrid::{dfa::DFA, OverlappingState},
    ///     HalfMatch,
    ///     MatchKind,
    /// };
    ///
    /// let dfa = DFA::builder()
    ///     .configure(DFA::config().match_kind(MatchKind::All))
    ///     .build_many(&[r"\w+$", r"\S+$"])?;
    /// let mut cache = dfa.create_cache();
    ///
    /// let haystack = "@foo".as_bytes();
    /// let mut state = OverlappingState::start();
    ///
    /// let expected = Some(HalfMatch::must(1, 4));
    /// let got = dfa.find_overlapping_fwd(&mut cache, haystack, &mut state)?;
    /// assert_eq!(expected, got);
    ///
    /// // The first pattern also matches at the same position, so re-running
    /// // the search will yield another match. Notice also that the first
    /// // pattern is returned after the second. This is because the second
    /// // pattern begins its match before the first, is therefore an earlier
    /// // match and is thus reported first.
    /// let expected = Some(HalfMatch::must(0, 4));
    /// let got = dfa.find_overlapping_fwd(&mut cache, haystack, &mut state)?;
    /// assert_eq!(expected, got);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn find_overlapping_fwd(
        &self,
        cache: &mut Cache,
        bytes: &[u8],
        state: &mut OverlappingState,
    ) -> Result<Option<HalfMatch>, MatchError> {
        self.find_overlapping_fwd_at(
            cache,
            None,
            None,
            bytes,
            0,
            bytes.len(),
            state,
        )
    }

    /// Executes a forward search and returns the end position of the first
    /// match that is found as early as possible. If no match exists, then
    /// `None` is returned.
    ///
    /// This routine stops scanning input as soon as the search observes a
    /// match state. This is useful for implementing boolean `is_match`-like
    /// routines, where as little work is done as possible.
    ///
    /// This is like [`DFA::find_earliest_fwd`], except it provides some
    /// additional control over how the search is executed:
    ///
    /// * `pre` is a prefilter scanner that, when given, is used whenever the
    /// DFA enters its starting state. This is meant to speed up searches where
    /// one or a small number of literal prefixes are known.
    /// * `pattern_id` specifies a specific pattern in the DFA to run an
    /// anchored search for. If not given, then a search for any pattern is
    /// performed. For lazy DFAs, [`Config::starts_for_each_pattern`] must be
    /// enabled to use this functionality.
    /// * `start` and `end` permit searching a specific region of the haystack
    /// `bytes`. This is useful when implementing an iterator over matches
    /// within the same haystack, which cannot be done correctly by simply
    /// providing a subslice of `bytes`. (Because the existence of look-around
    /// operations such as `\b`, `^` and `$` need to take the surrounding
    /// context into account. This cannot be done if the haystack doesn't
    /// contain it.)
    ///
    /// The examples below demonstrate each of these additional parameters.
    ///
    /// # Errors
    ///
    /// This routine only errors if the search could not complete. For
    /// lazy DFAs generated by this crate, this only occurs in non-default
    /// configurations where quit bytes are used, Unicode word boundaries are
    /// heuristically enabled or limits are set on the number of times the lazy
    /// DFA's cache may be cleared.
    ///
    /// When a search cannot complete, callers cannot know whether a match
    /// exists or not.
    ///
    /// # Panics
    ///
    /// This routine panics if a `pattern_id` is given and this lazy DFA does
    /// not support specific pattern searches.
    ///
    /// It also panics if the given haystack range is not valid.
    ///
    /// # Example: prefilter
    ///
    /// This example shows how to provide a prefilter for a pattern where all
    /// matches start with a `z` byte.
    ///
    /// ```
    /// use regex_automata::{
    ///     hybrid::dfa::DFA,
    ///     util::prefilter::{Candidate, Prefilter, Scanner, State},
    ///     HalfMatch,
    /// };
    ///
    /// #[derive(Debug)]
    /// pub struct ZPrefilter;
    ///
    /// impl Prefilter for ZPrefilter {
    ///     fn next_candidate(
    ///         &self,
    ///         _: &mut State,
    ///         haystack: &[u8],
    ///         at: usize,
    ///     ) -> Candidate {
    ///         // Try changing b'z' to b'q' and observe this test fail since
    ///         // the prefilter will skip right over the match.
    ///         match haystack.iter().position(|&b| b == b'z') {
    ///             None => Candidate::None,
    ///             Some(i) => Candidate::PossibleStartOfMatch(at + i),
    ///         }
    ///     }
    ///
    ///     fn heap_bytes(&self) -> usize {
    ///         0
    ///     }
    /// }
    ///
    /// let dfa = DFA::new("z[0-9]{3}")?;
    /// let mut cache = dfa.create_cache();
    ///
    /// let haystack = "foobar z123 q123".as_bytes();
    /// // A scanner executes a prefilter while tracking some state that helps
    /// // determine whether a prefilter is still "effective" or not.
    /// let mut scanner = Scanner::new(&ZPrefilter);
    ///
    /// let expected = Some(HalfMatch::must(0, 11));
    /// let got = dfa.find_earliest_fwd_at(
    ///     &mut cache,
    ///     Some(&mut scanner),
    ///     None,
    ///     haystack,
    ///     0,
    ///     haystack.len(),
    /// )?;
    /// assert_eq!(expected, got);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: specific pattern search
    ///
    /// This example shows how to build a lazy multi-DFA that permits searching
    /// for specific patterns.
    ///
    /// ```
    /// use regex_automata::{
    ///     hybrid::dfa::DFA,
    ///     HalfMatch,
    ///     PatternID,
    /// };
    ///
    /// let dfa = DFA::builder()
    ///     .configure(DFA::config().starts_for_each_pattern(true))
    ///     .build_many(&["[a-z0-9]{6}", "[a-z][a-z0-9]{5}"])?;
    /// let mut cache = dfa.create_cache();
    /// let haystack = "foo123".as_bytes();
    ///
    /// // Since we are using the default leftmost-first match and both
    /// // patterns match at the same starting position, only the first pattern
    /// // will be returned in this case when doing a search for any of the
    /// // patterns.
    /// let expected = Some(HalfMatch::must(0, 6));
    /// let got = dfa.find_earliest_fwd_at(
    ///     &mut cache,
    ///     None,
    ///     None,
    ///     haystack,
    ///     0,
    ///     haystack.len(),
    /// )?;
    /// assert_eq!(expected, got);
    ///
    /// // But if we want to check whether some other pattern matches, then we
    /// // can provide its pattern ID.
    /// let expected = Some(HalfMatch::must(1, 6));
    /// let got = dfa.find_earliest_fwd_at(
    ///     &mut cache,
    ///     None,
    ///     Some(PatternID::must(1)),
    ///     haystack,
    ///     0,
    ///     haystack.len(),
    /// )?;
    /// assert_eq!(expected, got);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: specifying the bounds of a search
    ///
    /// This example shows how providing the bounds of a search can produce
    /// different results than simply sub-slicing the haystack.
    ///
    /// ```
    /// use regex_automata::{hybrid::dfa::DFA, HalfMatch};
    ///
    /// // N.B. We disable Unicode here so that we use a simple ASCII word
    /// // boundary. Alternatively, we could enable heuristic support for
    /// // Unicode word boundaries since our haystack is pure ASCII.
    /// let dfa = DFA::new(r"(?-u)\b[0-9]{3}\b")?;
    /// let mut cache = dfa.create_cache();
    /// let haystack = "foo123bar".as_bytes();
    ///
    /// // Since we sub-slice the haystack, the search doesn't know about the
    /// // larger context and assumes that `123` is surrounded by word
    /// // boundaries. And of course, the match position is reported relative
    /// // to the sub-slice as well, which means we get `3` instead of `6`.
    /// let expected = Some(HalfMatch::must(0, 3));
    /// let got = dfa.find_earliest_fwd_at(
    ///     &mut cache,
    ///     None,
    ///     None,
    ///     &haystack[3..6],
    ///     0,
    ///     haystack[3..6].len(),
    /// )?;
    /// assert_eq!(expected, got);
    ///
    /// // But if we provide the bounds of the search within the context of the
    /// // entire haystack, then the search can take the surrounding context
    /// // into account. (And if we did find a match, it would be reported
    /// // as a valid offset into `haystack` instead of its sub-slice.)
    /// let expected = None;
    /// let got = dfa.find_earliest_fwd_at(
    ///     &mut cache,
    ///     None,
    ///     None,
    ///     haystack,
    ///     3,
    ///     6,
    /// )?;
    /// assert_eq!(expected, got);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn find_earliest_fwd_at(
        &self,
        cache: &mut Cache,
        pre: Option<&mut prefilter::Scanner>,
        pattern_id: Option<PatternID>,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Option<HalfMatch>, MatchError> {
        search::find_earliest_fwd(
            pre, self, cache, pattern_id, bytes, start, end,
        )
    }

    /// Executes a reverse search and returns the start position of the first
    /// match that is found as early as possible. If no match exists, then
    /// `None` is returned.
    ///
    /// This routine stops scanning input as soon as the search observes a
    /// match state.
    ///
    /// This is like [`DFA::find_earliest_rev`], except it provides some
    /// additional control over how the search is executed. See the
    /// documentation of [`DFA::find_earliest_fwd_at`] for more details
    /// on the additional parameters along with examples of their usage.
    ///
    /// # Errors
    ///
    /// This routine only errors if the search could not complete. For
    /// lazy DFAs generated by this crate, this only occurs in non-default
    /// configurations where quit bytes are used, Unicode word boundaries are
    /// heuristically enabled or limits are set on the number of times the lazy
    /// DFA's cache may be cleared.
    ///
    /// When a search cannot complete, callers cannot know whether a match
    /// exists or not.
    ///
    /// # Panics
    ///
    /// This routine panics if a `pattern_id` is given and the underlying
    /// DFA does not support specific pattern searches.
    ///
    /// It also panics if the given haystack range is not valid.
    #[inline]
    pub fn find_earliest_rev_at(
        &self,
        cache: &mut Cache,
        pattern_id: Option<PatternID>,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Option<HalfMatch>, MatchError> {
        search::find_earliest_rev(self, cache, pattern_id, bytes, start, end)
    }

    /// Executes a forward search and returns the end position of the leftmost
    /// match that is found. If no match exists, then `None` is returned.
    ///
    /// This is like [`DFA::find_leftmost_fwd`], except it provides some
    /// additional control over how the search is executed. See the
    /// documentation of [`DFA::find_earliest_fwd_at`] for more details on the
    /// additional parameters along with examples of their usage.
    ///
    /// # Errors
    ///
    /// This routine only errors if the search could not complete. For
    /// lazy DFAs generated by this crate, this only occurs in non-default
    /// configurations where quit bytes are used, Unicode word boundaries are
    /// heuristically enabled or limits are set on the number of times the lazy
    /// DFA's cache may be cleared.
    ///
    /// When a search cannot complete, callers cannot know whether a match
    /// exists or not.
    ///
    /// # Panics
    ///
    /// This routine panics if a `pattern_id` is given and the underlying
    /// DFA does not support specific pattern searches.
    ///
    /// It also panics if the given haystack range is not valid.
    #[inline]
    pub fn find_leftmost_fwd_at(
        &self,
        cache: &mut Cache,
        pre: Option<&mut prefilter::Scanner>,
        pattern_id: Option<PatternID>,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Option<HalfMatch>, MatchError> {
        search::find_leftmost_fwd(
            pre, self, cache, pattern_id, bytes, start, end,
        )
    }

    /// Executes a reverse search and returns the start of the position of the
    /// leftmost match that is found. If no match exists, then `None` is
    /// returned.
    ///
    /// This is like [`DFA::find_leftmost_rev`], except it provides some
    /// additional control over how the search is executed. See the
    /// documentation of [`DFA::find_earliest_fwd_at`] for more details on the
    /// additional parameters along with examples of their usage.
    ///
    /// # Errors
    ///
    /// This routine only errors if the search could not complete. For
    /// lazy DFAs generated by this crate, this only occurs in non-default
    /// configurations where quit bytes are used, Unicode word boundaries are
    /// heuristically enabled or limits are set on the number of times the lazy
    /// DFA's cache may be cleared.
    ///
    /// When a search cannot complete, callers cannot know whether a match
    /// exists or not.
    ///
    /// # Panics
    ///
    /// This routine panics if a `pattern_id` is given and the underlying
    /// DFA does not support specific pattern searches.
    ///
    /// It also panics if the given haystack range is not valid.
    #[inline]
    pub fn find_leftmost_rev_at(
        &self,
        cache: &mut Cache,
        pattern_id: Option<PatternID>,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Option<HalfMatch>, MatchError> {
        search::find_leftmost_rev(self, cache, pattern_id, bytes, start, end)
    }

    /// Executes an overlapping forward search and returns the end position of
    /// matches as they are found. If no match exists, then `None` is returned.
    ///
    /// This routine is principally only useful when searching for multiple
    /// patterns on inputs where multiple patterns may match the same regions
    /// of text. In particular, callers must preserve the automaton's search
    /// state from prior calls so that the implementation knows where the last
    /// match occurred.
    ///
    /// This is like [`DFA::find_overlapping_fwd`], except it provides
    /// some additional control over how the search is executed. See the
    /// documentation of [`DFA::find_earliest_fwd_at`] for more details
    /// on the additional parameters along with examples of their usage.
    ///
    /// When using this routine to implement an iterator of overlapping
    /// matches, the `start` of the search should always be set to the end
    /// of the last match. If more patterns match at the previous location,
    /// then they will be immediately returned. (This is tracked by the given
    /// overlapping state.) Otherwise, the search continues at the starting
    /// position given.
    ///
    /// If for some reason you want the search to forget about its previous
    /// state and restart the search at a particular position, then setting the
    /// state to [`OverlappingState::start`] will accomplish that.
    ///
    /// # Errors
    ///
    /// This routine only errors if the search could not complete. For
    /// lazy DFAs generated by this crate, this only occurs in non-default
    /// configurations where quit bytes are used, Unicode word boundaries are
    /// heuristically enabled or limits are set on the number of times the lazy
    /// DFA's cache may be cleared.
    ///
    /// When a search cannot complete, callers cannot know whether a match
    /// exists or not.
    ///
    /// # Panics
    ///
    /// This routine panics if a `pattern_id` is given and the underlying
    /// DFA does not support specific pattern searches.
    ///
    /// It also panics if the given haystack range is not valid.
    #[inline]
    pub fn find_overlapping_fwd_at(
        &self,
        cache: &mut Cache,
        pre: Option<&mut prefilter::Scanner>,
        pattern_id: Option<PatternID>,
        bytes: &[u8],
        start: usize,
        end: usize,
        state: &mut OverlappingState,
    ) -> Result<Option<HalfMatch>, MatchError> {
        search::find_overlapping_fwd(
            pre, self, cache, pattern_id, bytes, start, end, state,
        )
    }
}

impl DFA {
    /// Transitions from the current state to the next state, given the next
    /// byte of input.
    ///
    /// The given cache is used to either reuse pre-computed state
    /// transitions, or to store this newly computed transition for future
    /// reuse. Thus, this routine guarantees that it will never return a state
    /// ID that has an "unknown" tag.
    ///
    /// # State identifier validity
    ///
    /// The only valid value for `current` is the lazy state ID returned
    /// by the most recent call to `next_state`, `next_state_untagged`,
    /// `next_state_untagged_unchecked`, `start_state_forward` or
    /// `state_state_reverse` for the given `cache`. Any state ID returned from
    /// prior calls to these routines (with the same `cache`) is considered
    /// invalid (even if it gives an appearance of working). State IDs returned
    /// from _any_ prior call for different `cache` values are also always
    /// invalid.
    ///
    /// The returned ID is always a valid ID when `current` refers to a valid
    /// ID. Moreover, this routine is defined for all possible values of
    /// `input`.
    ///
    /// These validity rules are not checked, even in debug mode. Callers are
    /// required to uphold these rules themselves.
    ///
    /// Violating these state ID validity rules will not sacrifice memory
    /// safety, but _may_ produce an incorrect result or a panic.
    ///
    /// # Panics
    ///
    /// If the given ID does not refer to a valid state, then this routine
    /// may panic but it also may not panic and instead return an invalid or
    /// incorrect ID.
    ///
    /// # Example
    ///
    /// This shows a simplistic example for walking a lazy DFA for a given
    /// haystack by using the `next_state` method.
    ///
    /// ```
    /// use regex_automata::hybrid::dfa::DFA;
    ///
    /// let dfa = DFA::new(r"[a-z]+r")?;
    /// let mut cache = dfa.create_cache();
    /// let haystack = "bar".as_bytes();
    ///
    /// // The start state is determined by inspecting the position and the
    /// // initial bytes of the haystack.
    /// let mut sid = dfa.start_state_forward(
    ///     &mut cache, None, haystack, 0, haystack.len(),
    /// )?;
    /// // Walk all the bytes in the haystack.
    /// for &b in haystack {
    ///     sid = dfa.next_state(&mut cache, sid, b)?;
    /// }
    /// // Matches are always delayed by 1 byte, so we must explicitly walk the
    /// // special "EOI" transition at the end of the search.
    /// sid = dfa.next_eoi_state(&mut cache, sid)?;
    /// assert!(sid.is_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn next_state(
        &self,
        cache: &mut Cache,
        current: LazyStateID,
        input: u8,
    ) -> Result<LazyStateID, CacheError> {
        let class = usize::from(self.classes.get(input));
        let offset = current.as_usize_untagged() + class;
        let sid = cache.trans[offset];
        if !sid.is_unknown() {
            return Ok(sid);
        }
        Lazy::new(self, cache)
            .cache_next_state(current, alphabet::Unit::u8(input))
    }

    /// Transitions from the current state to the next state, given the next
    /// byte of input and a state ID that is not tagged.
    ///
    /// The only reason to use this routine is performance. In particular, the
    /// `next_state` method needs to do some additional checks, among them is
    /// to account for identifiers to states that are not yet computed. In
    /// such a case, the transition is computed on the fly. However, if it is
    /// known that the `current` state ID is untagged, then these checks can be
    /// omitted.
    ///
    /// Since this routine does not compute states on the fly, it does not
    /// modify the cache and thus cannot return an error. Consequently, `cache`
    /// does not need to be mutable and it is possible for this routine to
    /// return a state ID corresponding to the special "unknown" state. In
    /// this case, it is the caller's responsibility to use the prior state
    /// ID and `input` with `next_state` in order to force the computation of
    /// the unknown transition. Otherwise, trying to use the "unknown" state
    /// ID will just result in transitioning back to itself, and thus never
    /// terminating. (This is technically a special exemption to the state ID
    /// validity rules, but is permissible since this routine is guarateed to
    /// never mutate the given `cache`, and thus the identifier is guaranteed
    /// to remain valid.)
    ///
    /// See [`LazyStateID`] for more details on what it means for a state ID
    /// to be tagged. Also, see
    /// [`next_state_untagged_unchecked`](DFA::next_state_untagged_unchecked)
    /// for this same idea, but with bounds checks forcefully elided.
    ///
    /// # State identifier validity
    ///
    /// The only valid value for `current` is an **untagged** lazy
    /// state ID returned by the most recent call to `next_state`,
    /// `next_state_untagged`, `next_state_untagged_unchecked`,
    /// `start_state_forward` or `state_state_reverse` for the given `cache`.
    /// Any state ID returned from prior calls to these routines (with the
    /// same `cache`) is considered invalid (even if it gives an appearance
    /// of working). State IDs returned from _any_ prior call for different
    /// `cache` values are also always invalid.
    ///
    /// The returned ID is always a valid ID when `current` refers to a valid
    /// ID, although it may be tagged. Moreover, this routine is defined for
    /// all possible values of `input`.
    ///
    /// Not all validity rules are checked, even in debug mode. Callers are
    /// required to uphold these rules themselves.
    ///
    /// Violating these state ID validity rules will not sacrifice memory
    /// safety, but _may_ produce an incorrect result or a panic.
    ///
    /// # Panics
    ///
    /// If the given ID does not refer to a valid state, then this routine
    /// may panic but it also may not panic and instead return an invalid or
    /// incorrect ID.
    ///
    /// # Example
    ///
    /// This shows a simplistic example for walking a lazy DFA for a given
    /// haystack by using the `next_state_untagged` method where possible.
    ///
    /// ```
    /// use regex_automata::hybrid::dfa::DFA;
    ///
    /// let dfa = DFA::new(r"[a-z]+r")?;
    /// let mut cache = dfa.create_cache();
    /// let haystack = "bar".as_bytes();
    ///
    /// // The start state is determined by inspecting the position and the
    /// // initial bytes of the haystack.
    /// let mut sid = dfa.start_state_forward(
    ///     &mut cache, None, haystack, 0, haystack.len(),
    /// )?;
    /// // Walk all the bytes in the haystack.
    /// let mut at = 0;
    /// while at < haystack.len() {
    ///     if sid.is_tagged() {
    ///         sid = dfa.next_state(&mut cache, sid, haystack[at])?;
    ///     } else {
    ///         let mut prev_sid = sid;
    ///         // We attempt to chew through as much as we can while moving
    ///         // through untagged state IDs. Thus, the transition function
    ///         // does less work on average per byte. (Unrolling this loop
    ///         // may help even more.)
    ///         while at < haystack.len() {
    ///             prev_sid = sid;
    ///             sid = dfa.next_state_untagged(
    ///                 &mut cache, sid, haystack[at],
    ///             );
    ///             at += 1;
    ///             if sid.is_tagged() {
    ///                 break;
    ///             }
    ///         }
    ///         // We must ensure that we never proceed to the next iteration
    ///         // with an unknown state ID. If we don't account for this
    ///         // case, then search isn't guaranteed to terminate since all
    ///         // transitions on unknown states loop back to itself.
    ///         if sid.is_unknown() {
    ///             sid = dfa.next_state(
    ///                 &mut cache, prev_sid, haystack[at - 1],
    ///             )?;
    ///         }
    ///     }
    /// }
    /// // Matches are always delayed by 1 byte, so we must explicitly walk the
    /// // special "EOI" transition at the end of the search.
    /// sid = dfa.next_eoi_state(&mut cache, sid)?;
    /// assert!(sid.is_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn next_state_untagged(
        &self,
        cache: &Cache,
        current: LazyStateID,
        input: u8,
    ) -> LazyStateID {
        debug_assert!(!current.is_tagged());
        let class = usize::from(self.classes.get(input));
        let offset = current.as_usize_unchecked() + class;
        cache.trans[offset]
    }

    /// Transitions from the current state to the next state, eliding bounds
    /// checks, given the next byte of input and a state ID that is not tagged.
    ///
    /// The only reason to use this routine is performance. In particular, the
    /// `next_state` method needs to do some additional checks, among them is
    /// to account for identifiers to states that are not yet computed. In
    /// such a case, the transition is computed on the fly. However, if it is
    /// known that the `current` state ID is untagged, then these checks can be
    /// omitted.
    ///
    /// Since this routine does not compute states on the fly, it does not
    /// modify the cache and thus cannot return an error. Consequently, `cache`
    /// does not need to be mutable and it is possible for this routine to
    /// return a state ID corresponding to the special "unknown" state. In
    /// this case, it is the caller's responsibility to use the prior state
    /// ID and `input` with `next_state` in order to force the computation of
    /// the unknown transition. Otherwise, trying to use the "unknown" state
    /// ID will just result in transitioning back to itself, and thus never
    /// terminating. (This is technically a special exemption to the state ID
    /// validity rules, but is permissible since this routine is guarateed to
    /// never mutate the given `cache`, and thus the identifier is guaranteed
    /// to remain valid.)
    ///
    /// See [`LazyStateID`] for more details on what it means for a state ID
    /// to be tagged. Also, see
    /// [`next_state_untagged`](DFA::next_state_untagged)
    /// for this same idea, but with memory safety guaranteed by retaining
    /// bounds checks.
    ///
    /// # State identifier validity
    ///
    /// The only valid value for `current` is an **untagged** lazy
    /// state ID returned by the most recent call to `next_state`,
    /// `next_state_untagged`, `next_state_untagged_unchecked`,
    /// `start_state_forward` or `state_state_reverse` for the given `cache`.
    /// Any state ID returned from prior calls to these routines (with the
    /// same `cache`) is considered invalid (even if it gives an appearance
    /// of working). State IDs returned from _any_ prior call for different
    /// `cache` values are also always invalid.
    ///
    /// The returned ID is always a valid ID when `current` refers to a valid
    /// ID, although it may be tagged. Moreover, this routine is defined for
    /// all possible values of `input`.
    ///
    /// Not all validity rules are checked, even in debug mode. Callers are
    /// required to uphold these rules themselves.
    ///
    /// Violating these state ID validity rules will not sacrifice memory
    /// safety, but _may_ produce an incorrect result or a panic.
    ///
    /// # Safety
    ///
    /// Callers of this method must guarantee that `current` refers to a valid
    /// state ID according to the rules described above. If `current` is not a
    /// valid state ID for this automaton, then calling this routine may result
    /// in undefined behavior.
    ///
    /// If `current` is valid, then the ID returned is valid for all possible
    /// values of `input`.
    #[inline]
    pub unsafe fn next_state_untagged_unchecked(
        &self,
        cache: &Cache,
        current: LazyStateID,
        input: u8,
    ) -> LazyStateID {
        debug_assert!(!current.is_tagged());
        let class = usize::from(self.classes.get(input));
        let offset = current.as_usize_unchecked() + class;
        *cache.trans.get_unchecked(offset)
    }

    /// Transitions from the current state to the next state for the special
    /// EOI symbol.
    ///
    /// The given cache is used to either reuse pre-computed state
    /// transitions, or to store this newly computed transition for future
    /// reuse. Thus, this routine guarantees that it will never return a state
    /// ID that has an "unknown" tag.
    ///
    /// This routine must be called at the end of every search in a correct
    /// implementation of search. Namely, lazy DFAs in this crate delay matches
    /// by one byte in order to support look-around operators. Thus, after
    /// reaching the end of a haystack, a search implementation must follow one
    /// last EOI transition.
    ///
    /// It is best to think of EOI as an additional symbol in the alphabet of a
    /// DFA that is distinct from every other symbol. That is, the alphabet of
    /// lazy DFAs in this crate has a logical size of 257 instead of 256, where
    /// 256 corresponds to every possible inhabitant of `u8`. (In practice, the
    /// physical alphabet size may be smaller because of alphabet compression
    /// via equivalence classes, but EOI is always represented somehow in the
    /// alphabet.)
    ///
    /// # State identifier validity
    ///
    /// The only valid value for `current` is the lazy state ID returned
    /// by the most recent call to `next_state`, `next_state_untagged`,
    /// `next_state_untagged_unchecked`, `start_state_forward` or
    /// `state_state_reverse` for the given `cache`. Any state ID returned from
    /// prior calls to these routines (with the same `cache`) is considered
    /// invalid (even if it gives an appearance of working). State IDs returned
    /// from _any_ prior call for different `cache` values are also always
    /// invalid.
    ///
    /// The returned ID is always a valid ID when `current` refers to a valid
    /// ID.
    ///
    /// These validity rules are not checked, even in debug mode. Callers are
    /// required to uphold these rules themselves.
    ///
    /// Violating these state ID validity rules will not sacrifice memory
    /// safety, but _may_ produce an incorrect result or a panic.
    ///
    /// # Panics
    ///
    /// If the given ID does not refer to a valid state, then this routine
    /// may panic but it also may not panic and instead return an invalid or
    /// incorrect ID.
    ///
    /// # Example
    ///
    /// This shows a simplistic example for walking a DFA for a given haystack,
    /// and then finishing the search with the final EOI transition.
    ///
    /// ```
    /// use regex_automata::hybrid::dfa::DFA;
    ///
    /// let dfa = DFA::new(r"[a-z]+r")?;
    /// let mut cache = dfa.create_cache();
    /// let haystack = "bar".as_bytes();
    ///
    /// // The start state is determined by inspecting the position and the
    /// // initial bytes of the haystack.
    /// let mut sid = dfa.start_state_forward(
    ///     &mut cache, None, haystack, 0, haystack.len(),
    /// )?;
    /// // Walk all the bytes in the haystack.
    /// for &b in haystack {
    ///     sid = dfa.next_state(&mut cache, sid, b)?;
    /// }
    /// // Matches are always delayed by 1 byte, so we must explicitly walk
    /// // the special "EOI" transition at the end of the search. Without this
    /// // final transition, the assert below will fail since the DFA will not
    /// // have entered a match state yet!
    /// sid = dfa.next_eoi_state(&mut cache, sid)?;
    /// assert!(sid.is_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn next_eoi_state(
        &self,
        cache: &mut Cache,
        current: LazyStateID,
    ) -> Result<LazyStateID, CacheError> {
        let eoi = self.classes.eoi().as_usize();
        let offset = current.as_usize_untagged() + eoi;
        let sid = cache.trans[offset];
        if !sid.is_unknown() {
            return Ok(sid);
        }
        Lazy::new(self, cache).cache_next_state(current, self.classes.eoi())
    }

    /// Return the ID of the start state for this lazy DFA when executing a
    /// forward search.
    ///
    /// Unlike typical DFA implementations, the start state for DFAs in this
    /// crate is dependent on a few different factors:
    ///
    /// * The pattern ID, if present. When the underlying DFA has been
    /// configured with multiple patterns _and_ the DFA has been configured to
    /// build an anchored start state for each pattern, then a pattern ID may
    /// be specified to execute an anchored search for that specific pattern.
    /// If `pattern_id` is invalid or if the DFA isn't configured to build
    /// start states for each pattern, then implementations must panic. DFAs in
    /// this crate can be configured to build start states for each pattern via
    /// [`Config::starts_for_each_pattern`].
    /// * When `start > 0`, the byte at index `start - 1` may influence the
    /// start state if the regex uses `^` or `\b`.
    /// * Similarly, when `start == 0`, it may influence the start state when
    /// the regex uses `^` or `\A`.
    /// * Currently, `end` is unused.
    /// * Whether the search is a forward or reverse search. This routine can
    /// only be used for forward searches.
    ///
    /// # Panics
    ///
    /// This panics if `start..end` is not a valid sub-slice of `bytes`. This
    /// also panics if `pattern_id` is non-None and does not refer to a valid
    /// pattern, or if the DFA was not configured to build anchored start
    /// states for each pattern.
    #[inline]
    pub fn start_state_forward(
        &self,
        cache: &mut Cache,
        pattern_id: Option<PatternID>,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Result<LazyStateID, CacheError> {
        let mut lazy = Lazy::new(self, cache);
        let start_type = Start::from_position_fwd(bytes, start, end);
        let sid = lazy.get_cached_start(pattern_id, start_type);
        if !sid.is_unknown() {
            return Ok(sid);
        }
        lazy.cache_start_group(pattern_id, start_type)
    }

    /// Return the ID of the start state for this lazy DFA when executing a
    /// reverse search.
    ///
    /// Unlike typical DFA implementations, the start state for DFAs in this
    /// crate is dependent on a few different factors:
    ///
    /// * The pattern ID, if present. When the underlying DFA has been
    /// configured with multiple patterns _and_ the DFA has been configured to
    /// build an anchored start state for each pattern, then a pattern ID may
    /// be specified to execute an anchored search for that specific pattern.
    /// If `pattern_id` is invalid or if the DFA isn't configured to build
    /// start states for each pattern, then implementations must panic. DFAs in
    /// this crate can be configured to build start states for each pattern via
    /// [`Config::starts_for_each_pattern`].
    /// * When `end < bytes.len()`, the byte at index `end` may influence the
    /// start state if the regex uses `$` or `\b`.
    /// * Similarly, when `end == bytes.len()`, it may influence the start
    /// state when the regex uses `$` or `\z`.
    /// * Currently, `start` is unused.
    /// * Whether the search is a forward or reverse search. This routine can
    /// only be used for reverse searches.
    ///
    /// # Panics
    ///
    /// This panics if `start..end` is not a valid sub-slice of `bytes`. This
    /// also panics if `pattern_id` is non-None and does not refer to a valid
    /// pattern, or if the DFA was not configured to build anchored start
    /// states for each pattern.
    #[inline]
    pub fn start_state_reverse(
        &self,
        cache: &mut Cache,
        pattern_id: Option<PatternID>,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Result<LazyStateID, CacheError> {
        let mut lazy = Lazy::new(self, cache);
        let start_type = Start::from_position_rev(bytes, start, end);
        let sid = lazy.get_cached_start(pattern_id, start_type);
        if !sid.is_unknown() {
            return Ok(sid);
        }
        lazy.cache_start_group(pattern_id, start_type)
    }

    /// Returns the total number of patterns that match in this state.
    ///
    /// If the given state is not a match state, then this may panic.
    ///
    /// If the lazy DFA was compiled with one pattern, then this must
    /// necessarily always return `1` for all match states.
    ///
    /// A lazy DFA guarantees that [`DFA::match_pattern`] can be called with
    /// indices up to (but not including) the count returned by this routine
    /// without panicking.
    ///
    /// # Example
    ///
    /// This example shows a simple instance of implementing overlapping
    /// matches. In particular, it shows not only how to determine how many
    /// patterns have matched in a particular state, but also how to access
    /// which specific patterns have matched.
    ///
    /// Notice that we must use [`MatchKind::All`](crate::MatchKind::All)
    /// when building the DFA. If we used
    /// [`MatchKind::LeftmostFirst`](crate::MatchKind::LeftmostFirst)
    /// instead, then the DFA would not be constructed in a way that supports
    /// overlapping matches. (It would only report a single pattern that
    /// matches at any particular point in time.)
    ///
    /// Another thing to take note of is the patterns used and the order in
    /// which the pattern IDs are reported. In the example below, pattern `3`
    /// is yielded first. Why? Because it corresponds to the match that
    /// appears first. Namely, the `@` symbol is part of `\S+` but not part
    /// of any of the other patterns. Since the `\S+` pattern has a match that
    /// starts to the left of any other pattern, its ID is returned before any
    /// other.
    ///
    /// ```
    /// use regex_automata::{hybrid::dfa::DFA, MatchKind};
    ///
    /// let dfa = DFA::builder()
    ///     .configure(DFA::config().match_kind(MatchKind::All))
    ///     .build_many(&[
    ///         r"\w+", r"[a-z]+", r"[A-Z]+", r"\S+",
    ///     ])?;
    /// let mut cache = dfa.create_cache();
    /// let haystack = "@bar".as_bytes();
    ///
    /// // The start state is determined by inspecting the position and the
    /// // initial bytes of the haystack.
    /// let mut sid = dfa.start_state_forward(
    ///     &mut cache, None, haystack, 0, haystack.len(),
    /// )?;
    /// // Walk all the bytes in the haystack.
    /// for &b in haystack {
    ///     sid = dfa.next_state(&mut cache, sid, b)?;
    /// }
    /// sid = dfa.next_eoi_state(&mut cache, sid)?;
    ///
    /// assert!(sid.is_match());
    /// assert_eq!(dfa.match_count(&mut cache, sid), 3);
    /// // The following calls are guaranteed to not panic since `match_count`
    /// // returned `3` above.
    /// assert_eq!(dfa.match_pattern(&mut cache, sid, 0).as_usize(), 3);
    /// assert_eq!(dfa.match_pattern(&mut cache, sid, 1).as_usize(), 0);
    /// assert_eq!(dfa.match_pattern(&mut cache, sid, 2).as_usize(), 1);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn match_count(&self, cache: &mut Cache, id: LazyStateID) -> usize {
        assert!(id.is_match());
        Lazy::new(self, cache).get_cached_state(id).match_count()
    }

    /// Returns the pattern ID corresponding to the given match index in the
    /// given state.
    ///
    /// See [`DFA::match_count`] for an example of how to use this method
    /// correctly. Note that if you know your lazy DFA is configured with a
    /// single pattern, then this routine is never necessary since it will
    /// always return a pattern ID of `0` for an index of `0` when `id`
    /// corresponds to a match state.
    ///
    /// Typically, this routine is used when implementing an overlapping
    /// search, as the example for `DFA::match_count` does.
    ///
    /// # Panics
    ///
    /// If the state ID is not a match state or if the match index is out
    /// of bounds for the given state, then this routine may either panic
    /// or produce an incorrect result. If the state ID is correct and the
    /// match index is correct, then this routine must always produce a valid
    /// `PatternID`.
    #[inline]
    pub fn match_pattern(
        &self,
        cache: &mut Cache,
        id: LazyStateID,
        match_index: usize,
    ) -> PatternID {
        // This is an optimization for the very common case of a DFA with a
        // single pattern. This conditional avoids a somewhat more costly path
        // that finds the pattern ID from the corresponding `State`, which
        // requires a bit of slicing/pointer-chasing. This optimization tends
        // to only matter when matches are frequent.
        if self.pattern_count() == 1 {
            return PatternID::ZERO;
        }
        Lazy::new(self, cache).get_cached_state(id).match_pattern(match_index)
    }
}

#[derive(Clone, Debug)]
pub struct Cache {
    trans: Vec<LazyStateID>,
    starts: Vec<LazyStateID>,
    states: Vec<State>,
    states_to_id: StateMap,
    sparses: SparseSets,
    stack: Vec<NFAStateID>,
    scratch_state_builder: StateBuilderEmpty,
    state_saver: StateSaver,
    /// The memory usage, in bytes, used by 'states' and 'states_to_id'. We
    /// track this as new states are added since states use a variable amount
    /// of heap. Tracking this as we add states makes it possible to compute
    /// the total amount of memory used by the determinizer in constant time.
    memory_usage_state: usize,
    /// The number of times the cache has been cleared. When a minimum cache
    /// clear count is set, then the cache will return an error instead of
    /// clearing the cache if the count has been exceeded.
    clear_count: usize,
}

impl Cache {
    pub fn new(dfa: &DFA) -> Cache {
        let mut cache = Cache {
            trans: alloc::vec![],
            starts: alloc::vec![],
            states: alloc::vec![],
            states_to_id: StateMap::new(),
            sparses: SparseSets::new(dfa.nfa.len()),
            stack: alloc::vec![],
            scratch_state_builder: StateBuilderEmpty::new(),
            state_saver: StateSaver::none(),
            memory_usage_state: 0,
            clear_count: 0,
        };
        Lazy { dfa, cache: &mut cache }.init_cache();
        cache
    }

    pub fn reset(&mut self, dfa: &DFA) {
        Lazy::new(dfa, self).reset_cache()
    }

    pub fn memory_usage(&self) -> usize {
        const ID_SIZE: usize = size_of::<LazyStateID>();
        const STATE_SIZE: usize = size_of::<State>();

        self.trans.len() * ID_SIZE
        + self.starts.len() * ID_SIZE
        + self.states.len() * STATE_SIZE
        // Maps likely use more memory than this, but it's probably close.
        + self.states_to_id.len() * (STATE_SIZE + ID_SIZE)
        + self.sparses.memory_usage()
        + self.stack.capacity() * ID_SIZE
        + self.scratch_state_builder.capacity()
        // Heap memory used by 'State' in both 'states' and 'states_to_id'.
        + self.memory_usage_state
    }
}

/// A map from states to state identifiers. When using std, we use a standard
/// hashmap, since it's a bit faster for this use case. (Other maps, like
/// one's based on FNV, have not yet been benchmarked.)
///
/// The main purpose of this map is to reuse states where possible. This won't
/// fully minimize the DFA, but it works well in a lot of cases.
#[cfg(feature = "std")]
type StateMap = std::collections::HashMap<State, LazyStateID>;
#[cfg(not(feature = "std"))]
type StateMap = alloc::collections::BTreeMap<State, LazyStateID>;

#[derive(Debug)]
struct Lazy<'i, 'c> {
    dfa: &'i DFA,
    cache: &'c mut Cache,
}

impl<'i, 'c> Lazy<'i, 'c> {
    /// Creates a new 'Lazy' wrapper for a DFA and its corresponding cache.
    fn new(dfa: &'i DFA, cache: &'c mut Cache) -> Lazy<'i, 'c> {
        Lazy { dfa, cache }
    }
}

impl<'i, 'c> Lazy<'i, 'c> {
    #[inline(never)]
    fn cache_next_state(
        &mut self,
        mut current: LazyStateID,
        unit: alphabet::Unit,
    ) -> Result<LazyStateID, CacheError> {
        let stride2 = self.dfa.stride2();
        let empty_builder = self.get_state_builder();
        let builder = determinize::next(
            &self.dfa.nfa,
            self.dfa.match_kind,
            &mut self.cache.sparses,
            &mut self.cache.stack,
            &self.cache.states[current.as_usize_untagged() >> stride2],
            unit,
            empty_builder,
        );
        let save_state = !self.state_fits_in_cache(&builder);
        if save_state {
            self.save_state(current);
        }
        let next = self.add_builder_state(builder, |sid| sid)?;
        if save_state {
            current = self.saved_state_id();
        }
        // This is the payoff. The next time 'next_state' is called with this
        // state and alphabet unit, it will find this transition and avoid
        // having to re-determinize this transition.
        self.set_transition(current, unit, next);
        Ok(next)
    }

    fn get_cached_start(
        &self,
        pattern_id: Option<PatternID>,
        start: Start,
    ) -> LazyStateID {
        let start_index = start.as_usize();
        let index = match pattern_id {
            None => start_index,
            Some(pid) => {
                let pid = pid.as_usize();
                assert!(
                    pid < self.dfa.pattern_count(),
                    "invalid pattern ID: {:?}",
                    pid
                );
                Start::count() + (Start::count() * pid) + start_index
            }
        };
        self.cache.starts[index]
    }

    fn cache_start_group(
        &mut self,
        pattern_id: Option<PatternID>,
        start: Start,
    ) -> Result<LazyStateID, CacheError> {
        let nfa_start_id = match pattern_id {
            Some(pid) => self.dfa.nfa.start_pattern(pid),
            None if self.dfa.anchored => self.dfa.nfa.start_anchored(),
            None => self.dfa.nfa.start_unanchored(),
        };

        let id = self.cache_start_one(nfa_start_id, start)?;
        self.set_start_state(pattern_id, start, id);
        Ok(id)
    }

    fn cache_start_one(
        &mut self,
        nfa_start_id: NFAStateID,
        start: Start,
    ) -> Result<LazyStateID, CacheError> {
        let mut builder_matches = self.get_state_builder().into_matches();
        determinize::set_lookbehind_from_start(&start, &mut builder_matches);
        self.cache.sparses.set1.clear();
        determinize::epsilon_closure(
            self.dfa.nfa.borrow(),
            nfa_start_id,
            *builder_matches.look_have(),
            &mut self.cache.stack,
            &mut self.cache.sparses.set1,
        );
        let mut builder = builder_matches.into_nfa();
        determinize::add_nfa_states(
            self.dfa.nfa.borrow(),
            &self.cache.sparses.set1,
            &mut builder,
        );
        self.add_builder_state(builder, |id| id.to_start())
    }

    fn get_cached_state(&self, sid: LazyStateID) -> &State {
        let index = sid.as_usize_untagged() >> self.dfa.stride2();
        &self.cache.states[index]
    }

    fn add_builder_state(
        &mut self,
        builder: StateBuilderNFA,
        idmap: impl Fn(LazyStateID) -> LazyStateID,
    ) -> Result<LazyStateID, CacheError> {
        if let Some(&cached_id) =
            self.cache.states_to_id.get(builder.as_bytes())
        {
            // Since we have a cached state, put the constructed state's
            // memory back into our scratch space, so that it can be reused.
            self.put_state_builder(builder);
            return Ok(cached_id);
        }
        let result = self.add_state(builder.to_state(), idmap);
        self.put_state_builder(builder);
        result
    }

    fn add_state(
        &mut self,
        state: State,
        idmap: impl Fn(LazyStateID) -> LazyStateID,
    ) -> Result<LazyStateID, CacheError> {
        let required_cap = self.cache.memory_usage()
            + self.memory_usage_for_one_more_state(state.memory_usage());
        if required_cap > self.dfa.cache_capacity {
            self.try_clear_cache()?;
        }
        // It's important for this to come second, since the above may clear
        // the cache. If we clear the cache after ID generation, then the ID
        // is likely bunk since it would have been generated based on a larger
        // transition table.
        let mut id = idmap(self.next_state_id()?);
        if state.is_match() {
            id = id.to_match();
        }
        // Add room in the transition table. Since this is a fresh state, all
        // of its transitions are unknown.
        self.cache
            .trans
            .extend(iter::repeat(self.unknown_id()).take(self.dfa.stride()));
        // When we add a sentinel state, we never want to set any quit
        // transitions. Technically, this is harmless, since sentinel states
        // have all of their transitions set to loop back to themselves. But
        // when creating sentinel states before the quit sentinel state,
        // this will try to call 'set_transition' on a state ID that doesn't
        // actually exist yet, which isn't allowed. So we just skip doing so
        // entirely.
        if !self.dfa.quitset.is_empty() && !self.is_sentinel(id) {
            let quit_id = self.quit_id();
            for b in self.dfa.quitset.iter() {
                self.set_transition(id, alphabet::Unit::u8(b), quit_id);
            }
        }
        self.cache.memory_usage_state += state.memory_usage();
        self.cache.states.push(state.clone());
        self.cache.states_to_id.insert(state, id);
        Ok(id)
    }

    fn next_state_id(&mut self) -> Result<LazyStateID, CacheError> {
        let sid = match LazyStateID::new(self.cache.trans.len()) {
            Ok(sid) => sid,
            Err(_) => {
                self.try_clear_cache()?;
                // This has to pass since we check that ID capacity at
                // construction time can fit at least MIN_STATES states.
                LazyStateID::new(self.cache.trans.len()).unwrap()
            }
        };
        Ok(sid)
    }

    fn try_clear_cache(&mut self) -> Result<(), CacheError> {
        // Currently, the only heuristic we use is the minimum cache clear
        // count. If we pass that minimum, then we give up.
        //
        // It would be good to also add a heuristic based on "bytes searched
        // per generated state," but this requires API design work. Namely,
        // we really do not want to add a counter increment to the transition
        // function, which implies we need to expose APIs to update the number
        // of bytes searched by implementers of the search routines. And that
        // doesn't seem great... But we should do it if this heuristic isn't
        // enough. (The original lazy DFA implementation in the 'regex' crate
        // had this heuristic, since the lazy DFA was coupled with the search
        // routines.)
        if let Some(min_count) = self.dfa.minimum_cache_clear_count {
            if self.cache.clear_count >= min_count {
                return Err(CacheError::too_many_cache_resets());
            }
        }
        self.clear_cache();
        Ok(())
    }

    fn reset_cache(&mut self) {
        self.cache.state_saver = StateSaver::none();
        self.clear_cache();
        self.cache.clear_count = 0;
    }

    fn clear_cache(&mut self) {
        self.cache.trans.clear();
        self.cache.starts.clear();
        self.cache.states.clear();
        self.cache.states_to_id.clear();
        self.cache.memory_usage_state = 0;
        self.cache.clear_count += 1;
        trace!(
            "lazy DFA cache has been cleared (count: {})",
            self.cache.clear_count
        );
        self.init_cache();
        // If the state we want to save is one of the sentinel
        // (unknown/dead/quit) states, then 'init_cache' adds those back, and
        // their identifier values remains invariant. So there's no need to add
        // it again. (And indeed, doing so would be incorrect!)
        if let Some((old_id, state)) = self.cache.state_saver.take_to_save() {
            // If the state is one of the special sentinel states, then it is
            // automatically added by cache initialization and its ID always
            // remains the same. With that said, this should never occur since
            // the sentinel states are all loop states back to themselves. So
            // we should never be in a position where we're attempting to save
            // a sentinel state since we never compute transitions out of a
            // sentinel state.
            assert!(!self.is_sentinel(old_id), "cannot save sentinel state");
            let new_id = self
                .add_state(state, |id| {
                    if old_id.is_start() {
                        id.to_start()
                    } else {
                        id
                    }
                })
                // The unwrap here is OK because lazy DFA creation ensures that
                // we have room in the cache to add MIN_STATES states. Since
                // 'init_cache' above adds 3, this adds a 4th.
                .expect("adding one state after cache reset must work");
            self.cache.state_saver = StateSaver::Saved(new_id);
        }
    }

    fn init_cache(&mut self) {
        let mut starts_len = Start::count();
        if self.dfa.starts_for_each_pattern {
            starts_len += Start::count() * self.dfa.pattern_count();
        }
        self.cache
            .starts
            .extend(iter::repeat(self.unknown_id()).take(starts_len));
        // This is the set of NFA states that corresponds to each of our three
        // sentinel states: the empty set.
        let dead = State::dead();
        // This sets up some states that we use as sentinels that are present
        // in every DFA. While it would be technically possible to implement
        // this DFA without explicitly putting these states in the transition
        // table, this is convenient to do to make `next_state` correct for all
        // valid state IDs without needing explicit conditionals to special
        // case these sentinel states.
        //
        // All three of these states are "dead" states. That is, all of
        // them transition only to themselves. So once you enter one of
        // these states, it's impossible to leave them. Thus, any correct
        // search routine must explicitly check for these state types. (Sans
        // `unknown`, since that is only used internally to represent missing
        // states.)
        let unk_id =
            self.add_state(dead.clone(), |id| id.to_unknown()).unwrap();
        let dead_id = self.add_state(dead.clone(), |id| id.to_dead()).unwrap();
        let quit_id = self.add_state(dead.clone(), |id| id.to_quit()).unwrap();
        assert_eq!(unk_id, self.unknown_id());
        assert_eq!(dead_id, self.dead_id());
        assert_eq!(quit_id, self.quit_id());
        // The idea here is that if you start in an unknown/dead/quit state and
        // try to transition on them, then you should end up where you started.
        self.set_all_transitions(unk_id, unk_id);
        self.set_all_transitions(dead_id, dead_id);
        self.set_all_transitions(quit_id, quit_id);
        // All of these states are technically equivalent from the FSM
        // perspective, so putting all three of them in the cache isn't
        // possible. (They are distinct merely because we use their
        // identifiers as sentinels to mean something, as indicated by the
        // names.) Moreover, we wouldn't want to do that. Unknown and quit
        // states are special in that they are artificial constructions
        // this implementation. But dead states are a natural part of
        // determinization. When you reach a point in the NFA where you cannot
        // go anywhere else, a dead state will naturally arise and we MUST
        // reuse the canonical dead state that we've created here. Why? Because
        // it is the state ID that tells the search routine whether a state is
        // dead or not, and thus, whether to stop the search. Having a bunch of
        // distinct dead states would be quite wasteful!
        self.cache.states_to_id.insert(dead, dead_id);
    }

    fn save_state(&mut self, id: LazyStateID) {
        let state = self.get_cached_state(id).clone();
        self.cache.state_saver = StateSaver::ToSave { id, state };
    }

    fn saved_state_id(&mut self) -> LazyStateID {
        self.cache
            .state_saver
            .take_saved()
            .expect("state saver does not have saved state ID")
    }

    fn set_all_transitions(&mut self, from: LazyStateID, to: LazyStateID) {
        for unit in self.dfa.classes.representatives() {
            self.set_transition(from, unit, to);
        }
    }

    fn set_transition(
        &mut self,
        from: LazyStateID,
        unit: alphabet::Unit,
        to: LazyStateID,
    ) {
        assert!(self.is_valid(from), "invalid 'from' id: {:?}", from);
        assert!(self.is_valid(to), "invalid 'to' id: {:?}", to);
        let offset =
            from.as_usize_untagged() + self.dfa.classes.get_by_unit(unit);
        self.cache.trans[offset] = to;
    }

    fn set_start_state(
        &mut self,
        pattern_id: Option<PatternID>,
        start: Start,
        id: LazyStateID,
    ) {
        assert!(self.is_valid(id));
        let start_index = start.as_usize();
        let index = match pattern_id {
            None => start_index,
            Some(pid) => {
                let pid = pid.as_usize();
                Start::count() + (Start::count() * pid) + start_index
            }
        };
        self.cache.starts[index] = id;
    }

    fn unknown_id(&self) -> LazyStateID {
        // This unwrap is OK since 0 is always a valid state ID.
        LazyStateID::new(0).unwrap().to_unknown()
    }

    fn dead_id(&self) -> LazyStateID {
        // This unwrap is OK since the maximum value here is 1 * 512 = 512,
        // which is <= 2047 (the maximum state ID on 16-bit systems). Where
        // 512 is the worst case for our equivalence classes (every byte is a
        // distinct class).
        LazyStateID::new(1 << self.dfa.stride2()).unwrap().to_dead()
    }

    fn quit_id(&self) -> LazyStateID {
        // This unwrap is OK since the maximum value here is 2 * 512 = 1024,
        // which is <= 2047 (the maximum state ID on 16-bit systems). Where
        // 512 is the worst case for our equivalence classes (every byte is a
        // distinct class).
        LazyStateID::new(2 << self.dfa.stride2()).unwrap().to_quit()
    }

    fn is_sentinel(&self, id: LazyStateID) -> bool {
        id == self.unknown_id() || id == self.dead_id() || id == self.quit_id()
    }

    fn is_valid(&self, id: LazyStateID) -> bool {
        let id = id.as_usize_untagged();
        id < self.cache.trans.len() && id % self.dfa.stride() == 0
    }

    /// Returns a state builder from this DFA that might have existing
    /// capacity. This helps avoid allocs in cases where a state is built that
    /// turns out to already be cached.
    ///
    /// Callers must put the state builder back with 'put_state_builder',
    /// otherwise the allocation reuse won't work.
    fn get_state_builder(&mut self) -> StateBuilderEmpty {
        core::mem::replace(
            &mut self.cache.scratch_state_builder,
            StateBuilderEmpty::new(),
        )
    }

    /// Puts the given state builder back into this DFA for reuse.
    ///
    /// Note that building a 'State' from a builder always creates a new alloc,
    /// so callers should always put the builder back.
    fn put_state_builder(&mut self, builder: StateBuilderNFA) {
        let _ = core::mem::replace(
            &mut self.cache.scratch_state_builder,
            builder.clear(),
        );
    }

    fn state_fits_in_cache(&self, state: &StateBuilderNFA) -> bool {
        let needed = self.cache.memory_usage()
            + self.memory_usage_for_one_more_state(state.as_bytes().len());
        needed <= self.dfa.cache_capacity
    }

    fn memory_usage_for_one_more_state(
        &self,
        state_heap_size: usize,
    ) -> usize {
        const ID_SIZE: usize = size_of::<LazyStateID>();
        const STATE_SIZE: usize = size_of::<State>();

        self.dfa.stride() * ID_SIZE // additional space needed in trans table
        + STATE_SIZE // space in cache.states
        + (STATE_SIZE + ID_SIZE) // space in cache.states_to_id
        + state_heap_size // heap memory used by state itself
    }
}

/// A simple type that encapsulates the saving of a state ID through a cache
/// clearing.
///
/// A state ID can be marked for saving with ToSave, while a state ID can be
/// saved itself with Saved.
#[derive(Clone, Debug)]
enum StateSaver {
    /// An empty state saver. In this case, no states (other than the special
    /// sentinel states) are preserved after clearing the cache.
    None,
    /// An ID of a state (and the state itself) that should be preserved after
    /// the lazy DFA's cache has been cleared. After clearing, the updated ID
    /// is stored in 'Saved' since it may have changed.
    ToSave { id: LazyStateID, state: State },
    /// An ID that of a state that has been persisted through a lazy DFA
    /// cache clearing. The ID recorded here corresonds to an ID that was
    /// once marked as ToSave. The IDs are likely not equivalent even though
    /// the states they point to are.
    Saved(LazyStateID),
}

impl StateSaver {
    /// Create an empty state saver.
    fn none() -> StateSaver {
        StateSaver::None
    }

    /// Replace this state saver with an empty saver, and if this saver is a
    /// request to save a state, return that request.
    fn take_to_save(&mut self) -> Option<(LazyStateID, State)> {
        match core::mem::replace(self, StateSaver::None) {
            StateSaver::None | StateSaver::Saved(_) => None,
            StateSaver::ToSave { id, state } => Some((id, state)),
        }
    }

    /// Replace this state saver with an empty saver, and if this saver is a
    /// saved state (or a request to save a state), return that state's ID.
    ///
    /// The idea here is that a request to save a state isn't necessarily
    /// honored because it might not be needed. e.g., Some higher level code
    /// might request a state to be saved on the off chance that the cache gets
    /// cleared when a new state is added at a lower level. But if that new
    /// state is never added, then the cache is never cleared and the state and
    /// its ID remain unchanged.
    fn take_saved(&mut self) -> Option<LazyStateID> {
        match core::mem::replace(self, StateSaver::None) {
            StateSaver::None => None,
            StateSaver::Saved(id) | StateSaver::ToSave { id, .. } => Some(id),
        }
    }
}

/// The configuration used for building a lazy DFA.
///
/// As a convenience, [`DFA::config`] is an alias for [`Config::new`]. The
/// advantage of the former is that it often lets you avoid importing the
/// `Config` type directly.
///
/// A lazy DFA configuration is a simple data object that is typically used
/// with [`Builder::configure`].
///
/// The default configuration guarantees that a search will _never_ return
/// a [`MatchError`] for any haystack or pattern. Setting a quit byte with
/// [`Config::quit`], enabling heuristic support for Unicode word boundaries
/// with [`Config::unicode_word_boundary`], or setting a minimum cache clear
/// count with [`Config::minimum_cache_clear_count`] can in turn cause a search
/// to return an error. See the corresponding configuration options for more
/// details on when those error conditions arise.
#[derive(Clone, Copy, Debug, Default)]
pub struct Config {
    // As with other configuration types in this crate, we put all our knobs
    // in options so that we can distinguish between "default" and "not set."
    // This makes it possible to easily combine multiple configurations
    // without default values overwriting explicitly specified values. See the
    // 'overwrite' method.
    //
    // For docs on the fields below, see the corresponding method setters.
    anchored: Option<bool>,
    match_kind: Option<MatchKind>,
    starts_for_each_pattern: Option<bool>,
    byte_classes: Option<bool>,
    unicode_word_boundary: Option<bool>,
    quitset: Option<ByteSet>,
    cache_capacity: Option<usize>,
    skip_cache_capacity_check: Option<bool>,
    minimum_cache_clear_count: Option<Option<usize>>,
}

impl Config {
    /// Return a new default lazy DFA builder configuration.
    pub fn new() -> Config {
        Config::default()
    }

    /// Set whether matching must be anchored at the beginning of the input.
    ///
    /// When enabled, a match must begin at the start of a search. When
    /// disabled (the default), the lazy DFA will act as if the pattern started
    /// with a `(?s:.)*?`, which enables a match to appear anywhere.
    ///
    /// Note that if you want to run both anchored and unanchored
    /// searches without building multiple automatons, you can enable the
    /// [`Config::starts_for_each_pattern`] configuration instead. This will
    /// permit unanchored any-pattern searches and pattern-specific anchored
    /// searches. See the documentation for that configuration for an example.
    ///
    /// By default this is disabled.
    ///
    /// **WARNING:** this is subtly different than using a `^` at the start of
    /// your regex. A `^` forces a regex to match exclusively at the start of
    /// input, regardless of where you begin your search. In contrast, enabling
    /// this option will allow your regex to match anywhere in your input,
    /// but the match must start at the beginning of a search. (Most of the
    /// higher level convenience search routines make "start of input" and
    /// "start of search" equivalent, but some routines allow treating these as
    /// orthogonal.)
    ///
    /// For example, consider the haystack `aba` and the following searches:
    ///
    /// 1. The regex `^a` is compiled with `anchored=false` and searches
    ///    `aba` starting at position `2`. Since `^` requires the match to
    ///    start at the beginning of the input and `2 > 0`, no match is found.
    /// 2. The regex `a` is compiled with `anchored=true` and searches `aba`
    ///    starting at position `2`. This reports a match at `[2, 3]` since
    ///    the match starts where the search started. Since there is no `^`,
    ///    there is no requirement for the match to start at the beginning of
    ///    the input.
    /// 3. The regex `a` is compiled with `anchored=true` and searches `aba`
    ///    starting at position `1`. Since `b` corresponds to position `1` and
    ///    since the regex is anchored, it finds no match.
    /// 4. The regex `a` is compiled with `anchored=false` and searches `aba`
    ///    startting at position `1`. Since the regex is neither anchored nor
    ///    starts with `^`, the regex is compiled with an implicit `(?s:.)*?`
    ///    prefix that permits it to match anywhere. Thus, it reports a match
    ///    at `[2, 3]`.
    ///
    /// # Example
    ///
    /// This demonstrates the differences between an anchored search and
    /// a pattern that begins with `^` (as described in the above warning
    /// message).
    ///
    /// ```
    /// use regex_automata::{hybrid::dfa::DFA, HalfMatch};
    ///
    /// let haystack = "aba".as_bytes();
    ///
    /// let dfa = DFA::builder()
    ///     .configure(DFA::config().anchored(false)) // default
    ///     .build(r"^a")?;
    /// let mut cache = dfa.create_cache();
    /// let got = dfa.find_leftmost_fwd_at(
    ///     &mut cache, None, None, haystack, 2, 3,
    /// )?;
    /// // No match is found because 2 is not the beginning of the haystack,
    /// // which is what ^ requires.
    /// let expected = None;
    /// assert_eq!(expected, got);
    ///
    /// let dfa = DFA::builder()
    ///     .configure(DFA::config().anchored(true))
    ///     .build(r"a")?;
    /// let mut cache = dfa.create_cache();
    /// let got = dfa.find_leftmost_fwd_at(
    ///     &mut cache, None, None, haystack, 2, 3,
    /// )?;
    /// // An anchored search can still match anywhere in the haystack, it just
    /// // must begin at the start of the search which is '2' in this case.
    /// let expected = Some(HalfMatch::must(0, 3));
    /// assert_eq!(expected, got);
    ///
    /// let dfa = DFA::builder()
    ///     .configure(DFA::config().anchored(true))
    ///     .build(r"a")?;
    /// let mut cache = dfa.create_cache();
    /// let got = dfa.find_leftmost_fwd_at(
    ///     &mut cache, None, None, haystack, 1, 3,
    /// )?;
    /// // No match is found since we start searching at offset 1 which
    /// // corresponds to 'b'. Since there is no '(?s:.)*?' prefix, no match
    /// // is found.
    /// let expected = None;
    /// assert_eq!(expected, got);
    ///
    /// let dfa = DFA::builder()
    ///     .configure(DFA::config().anchored(false))
    ///     .build(r"a")?;
    /// let mut cache = dfa.create_cache();
    /// let got = dfa.find_leftmost_fwd_at(
    ///     &mut cache, None, None, haystack, 1, 3,
    /// )?;
    /// // Since anchored=false, an implicit '(?s:.)*?' prefix was added to the
    /// // pattern. Even though the search starts at 'b', the 'match anything'
    /// // prefix allows the search to match 'a'.
    /// let expected = Some(HalfMatch::must(0, 3));
    /// assert_eq!(expected, got);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn anchored(mut self, yes: bool) -> Config {
        self.anchored = Some(yes);
        self
    }

    /// Set the desired match semantics.
    ///
    /// The default is [`MatchKind::LeftmostFirst`], which corresponds to the
    /// match semantics of Perl-like regex engines. That is, when multiple
    /// patterns would match at the same leftmost position, the pattern that
    /// appears first in the concrete syntax is chosen.
    ///
    /// Currently, the only other kind of match semantics supported is
    /// [`MatchKind::All`]. This corresponds to classical DFA construction
    /// where all possible matches are added to the lazy DFA.
    ///
    /// Typically, `All` is used when one wants to execute an overlapping
    /// search and `LeftmostFirst` otherwise. In particular, it rarely makes
    /// sense to use `All` with the various "leftmost" find routines, since the
    /// leftmost routines depend on the `LeftmostFirst` automata construction
    /// strategy. Specifically, `LeftmostFirst` adds dead states to the
    /// lazy DFA as a way to terminate the search and report a match.
    /// `LeftmostFirst` also supports non-greedy matches using this strategy
    /// where as `All` does not.
    ///
    /// # Example: overlapping search
    ///
    /// This example shows the typical use of `MatchKind::All`, which is to
    /// report overlapping matches.
    ///
    /// ```
    /// use regex_automata::{
    ///     hybrid::{dfa::DFA, OverlappingState},
    ///     HalfMatch, MatchKind,
    /// };
    ///
    /// let dfa = DFA::builder()
    ///     .configure(DFA::config().match_kind(MatchKind::All))
    ///     .build_many(&[r"\w+$", r"\S+$"])?;
    /// let mut cache = dfa.create_cache();
    /// let haystack = "@foo".as_bytes();
    /// let mut state = OverlappingState::start();
    ///
    /// let expected = Some(HalfMatch::must(1, 4));
    /// let got = dfa.find_overlapping_fwd(&mut cache, haystack, &mut state)?;
    /// assert_eq!(expected, got);
    ///
    /// // The first pattern also matches at the same position, so re-running
    /// // the search will yield another match. Notice also that the first
    /// // pattern is returned after the second. This is because the second
    /// // pattern begins its match before the first, is therefore an earlier
    /// // match and is thus reported first.
    /// let expected = Some(HalfMatch::must(0, 4));
    /// let got = dfa.find_overlapping_fwd(&mut cache, haystack, &mut state)?;
    /// assert_eq!(expected, got);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: reverse automaton to find start of match
    ///
    /// Another example for using `MatchKind::All` is for constructing a
    /// reverse automaton to find the start of a match. `All` semantics are
    /// used for this in order to find the longest possible match, which
    /// corresponds to the leftmost starting position.
    ///
    /// Note that if you need the starting position then
    /// [`hybrid::regex::Regex`](crate::hybrid::regex::Regex) will handle this
    /// for you, so it's usually not necessary to do this yourself.
    ///
    /// ```
    /// use regex_automata::{hybrid::dfa::DFA, HalfMatch, MatchKind};
    ///
    /// let haystack = "123foobar456".as_bytes();
    /// let pattern = r"[a-z]+";
    ///
    /// let dfa_fwd = DFA::new(pattern)?;
    /// let dfa_rev = DFA::builder()
    ///     .configure(DFA::config()
    ///         .anchored(true)
    ///         .match_kind(MatchKind::All)
    ///     )
    ///     .build(pattern)?;
    /// let mut cache_fwd = dfa_fwd.create_cache();
    /// let mut cache_rev = dfa_rev.create_cache();
    ///
    /// let expected_fwd = HalfMatch::must(0, 9);
    /// let expected_rev = HalfMatch::must(0, 3);
    /// let got_fwd = dfa_fwd.find_leftmost_fwd(
    ///     &mut cache_fwd, haystack,
    /// )?.unwrap();
    /// // Here we don't specify the pattern to search for since there's only
    /// // one pattern and we're doing a leftmost search. But if this were an
    /// // overlapping search, you'd need to specify the pattern that matched
    /// // in the forward direction. (Otherwise, you might wind up finding the
    /// // starting position of a match of some other pattern.) That in turn
    /// // requires building the reverse automaton with starts_for_each_pattern
    /// // enabled. Indeed, this is what Regex does internally.
    /// let got_rev = dfa_rev.find_leftmost_rev_at(
    ///     &mut cache_rev, None, haystack, 0, got_fwd.offset(),
    /// )?.unwrap();
    /// assert_eq!(expected_fwd, got_fwd);
    /// assert_eq!(expected_rev, got_rev);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn match_kind(mut self, kind: MatchKind) -> Config {
        self.match_kind = Some(kind);
        self
    }

    pub fn byte_classes(mut self, yes: bool) -> Config {
        self.byte_classes = Some(yes);
        self
    }

    pub fn starts_for_each_pattern(mut self, yes: bool) -> Config {
        self.starts_for_each_pattern = Some(yes);
        self
    }

    pub fn unicode_word_boundary(mut self, yes: bool) -> Config {
        // We have a separate option for this instead of just setting the
        // appropriate quit bytes here because we don't want to set quit bytes
        // for every regex. We only want to set them when the regex contains a
        // Unicode word boundary.
        self.unicode_word_boundary = Some(yes);
        self
    }

    pub fn quit(mut self, byte: u8, yes: bool) -> Config {
        if self.get_unicode_word_boundary() && !byte.is_ascii() && !yes {
            panic!(
                "cannot set non-ASCII byte to be non-quit when \
                 Unicode word boundaries are enabled"
            );
        }
        if self.quitset.is_none() {
            self.quitset = Some(ByteSet::empty());
        }
        if yes {
            self.quitset.as_mut().unwrap().add(byte);
        } else {
            self.quitset.as_mut().unwrap().remove(byte);
        }
        self
    }

    pub fn cache_capacity(mut self, bytes: usize) -> Config {
        self.cache_capacity = Some(bytes);
        self
    }

    pub fn skip_cache_capacity_check(mut self, yes: bool) -> Config {
        self.skip_cache_capacity_check = Some(yes);
        self
    }

    pub fn minimum_cache_clear_count(mut self, min: Option<usize>) -> Config {
        self.minimum_cache_clear_count = Some(min);
        self
    }

    /// Returns whether this configuration has enabled anchored searches.
    pub fn get_anchored(&self) -> bool {
        self.anchored.unwrap_or(false)
    }

    /// Returns the match semantics set in this configuration.
    pub fn get_match_kind(&self) -> MatchKind {
        self.match_kind.unwrap_or(MatchKind::LeftmostFirst)
    }

    /// Returns whether this configuration has enabled anchored starting states
    /// for every pattern in the DFA.
    pub fn get_starts_for_each_pattern(&self) -> bool {
        self.starts_for_each_pattern.unwrap_or(false)
    }

    /// Returns whether this configuration has enabled byte classes or not.
    /// This is typically a debugging oriented option, as disabling it confers
    /// no speed benefit.
    pub fn get_byte_classes(&self) -> bool {
        self.byte_classes.unwrap_or(true)
    }

    /// Returns whether this configuration has enabled heuristic Unicode word
    /// boundary support. When enabled, it is possible for a search to return
    /// an error.
    pub fn get_unicode_word_boundary(&self) -> bool {
        self.unicode_word_boundary.unwrap_or(false)
    }

    /// Returns whether this configuration will instruct the DFA to enter a
    /// quit state whenever the given byte is seen during a search. When at
    /// least one byte has this enabled, it is possible for a search to return
    /// an error.
    pub fn get_quit(&self, byte: u8) -> bool {
        self.quitset.map_or(false, |q| q.contains(byte))
    }

    pub fn get_cache_capacity(&self) -> usize {
        self.cache_capacity.unwrap_or(2 * (1 << 20))
    }

    pub fn get_skip_cache_capacity_check(&self) -> bool {
        self.skip_cache_capacity_check.unwrap_or(false)
    }

    pub fn get_minimum_cache_clear_count(&self) -> Option<usize> {
        self.minimum_cache_clear_count.unwrap_or(None)
    }

    pub fn minimum_cache_capacity(
        &self,
        nfa: &thompson::NFA,
    ) -> Result<usize, BuildError> {
        let quitset = self.quit_set_from_nfa(nfa)?;
        let classes = self.byte_classes_from_nfa(nfa, &quitset);
        let starts = self.get_starts_for_each_pattern();
        Ok(minimum_cache_capacity(nfa, &classes, starts))
    }

    fn byte_classes_from_nfa(
        &self,
        nfa: &thompson::NFA,
        quit: &ByteSet,
    ) -> ByteClasses {
        if !self.get_byte_classes() {
            // The lazy DFA will always use the equivalence class map, but
            // enabling this option is useful for debugging. Namely, this will
            // cause all transitions to be defined over their actual bytes
            // instead of an opaque equivalence class identifier. The former is
            // much easier to grok as a human.
            ByteClasses::singletons()
        } else {
            let mut set = nfa.byte_class_set().clone();
            // It is important to distinguish any "quit" bytes from all other
            // bytes. Otherwise, a non-quit byte may end up in the same class
            // as a quit byte, and thus cause the DFA stop when it shouldn't.
            if !quit.is_empty() {
                set.add_set(&quit);
            }
            set.byte_classes()
        }
    }

    fn quit_set_from_nfa(
        &self,
        nfa: &thompson::NFA,
    ) -> Result<ByteSet, BuildError> {
        let mut quit = self.quitset.unwrap_or(ByteSet::empty());
        if nfa.has_word_boundary_unicode() {
            if self.get_unicode_word_boundary() {
                for b in 0x80..=0xFF {
                    quit.add(b);
                }
            } else {
                // If heuristic support for Unicode word boundaries wasn't
                // enabled, then we can still check if our quit set is correct.
                // If the caller set their quit bytes in a way that causes the
                // DFA to quit on at least all non-ASCII bytes, then that's all
                // we need for heuristic support to work.
                if !quit.contains_range(0x80, 0xFF) {
                    return Err(
                        BuildError::unsupported_dfa_word_boundary_unicode(),
                    );
                }
            }
        }
        Ok(quit)
    }

    /// Overwrite the default configuration such that the options in `o` are
    /// always used. If an option in `o` is not set, then the corresponding
    /// option in `self` is used. If it's not set in `self` either, then it
    /// remains not set.
    fn overwrite(self, o: Config) -> Config {
        Config {
            anchored: o.anchored.or(self.anchored),
            match_kind: o.match_kind.or(self.match_kind),
            starts_for_each_pattern: o
                .starts_for_each_pattern
                .or(self.starts_for_each_pattern),
            byte_classes: o.byte_classes.or(self.byte_classes),
            unicode_word_boundary: o
                .unicode_word_boundary
                .or(self.unicode_word_boundary),
            quitset: o.quitset.or(self.quitset),
            cache_capacity: o.cache_capacity.or(self.cache_capacity),
            skip_cache_capacity_check: o
                .skip_cache_capacity_check
                .or(self.skip_cache_capacity_check),
            minimum_cache_clear_count: o
                .minimum_cache_clear_count
                .or(self.minimum_cache_clear_count),
        }
    }
}

/// A builder for constructing a lazy deterministic finite automaton from
/// regular expressions.
///
/// As a convenience, [`DFA::builder`] is an alias for [`Builder::new`]. The
/// advantage of the former is that it often lets you avoid importing the
/// `Builder` type directly.
///
/// This builder provides two main things:
///
/// 1. It provides a few different `build` routines for actually constructing
/// a DFA from different kinds of inputs. The most convenient is
/// [`Builder::build`], which builds a DFA directly from a pattern string. The
/// most flexible is [`Builder::build_from_nfa`], which builds a DFA straight
/// from an NFA.
/// 2. The builder permits configuring a number of things.
/// [`Builder::configure`] is used with [`Config`] to configure aspects of
/// the DFA and the construction process itself. [`Builder::syntax`] and
/// [`Builder::thompson`] permit configuring the regex parser and Thompson NFA
/// construction, respectively. The syntax and thompson configurations only
/// apply when building from a pattern string.
///
/// This builder always constructs a *single* lazy DFA. As such, this builder
/// can only be used to construct regexes that either detect the presence
/// of a match or find the end location of a match. A single DFA cannot
/// produce both the start and end of a match. For that information, use a
/// [`Regex`](crate::hybrid::regex::Regex), which can be similarly configured
/// using [`regex::Builder`](crate::hybrid::regex::Builder). The main reason
/// to use a DFA directly is if the end location of a match is enough for your
/// use case. Namely, a `Regex` will construct two lazy DFAs instead of one,
/// since a second reverse DFA is needed to find the start of a match.
///
/// # Example
///
/// This example shows how to build a lazy DFA that uses a tiny cache capacity
/// and completely disables Unicode. That is:
///
/// * Things such as `\w`, `.` and `\b` are no longer Unicode-aware. `\w`
///   and `\b` are ASCII-only while `.` matches any byte except for `\n`
///   (instead of any UTF-8 encoding of a Unicode scalar value except for
///   `\n`). Things that are Unicode only, such as `\pL`, are not allowed.
/// * The pattern itself is permitted to match invalid UTF-8. For example,
///   things like `[^a]` that match any byte except for `a` are permitted.
/// * Unanchored patterns can search through invalid UTF-8. That is, for
///   unanchored patterns, the implicit prefix is `(?s-u:.)*?` instead of
///   `(?s:.)*?`.
///
/// ```
/// use regex_automata::{
///     hybrid::dfa::DFA,
///     nfa::thompson,
///     HalfMatch, SyntaxConfig,
/// };
///
/// let dfa = DFA::builder()
///     .configure(DFA::config().cache_capacity(5_000))
///     .syntax(SyntaxConfig::new().unicode(false).utf8(false))
///     .thompson(thompson::Config::new().utf8(false))
///     .build(r"foo[^b]ar.*")?;
/// let mut cache = dfa.create_cache();
///
/// let haystack = b"\xFEfoo\xFFar\xE2\x98\xFF\n";
/// let expected = Some(HalfMatch::must(0, 10));
/// let got = dfa.find_leftmost_fwd(&mut cache, haystack)?;
/// assert_eq!(expected, got);
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug)]
pub struct Builder {
    config: Config,
    thompson: thompson::Builder,
}

impl Builder {
    /// Create a new lazy DFA builder with the default configuration.
    pub fn new() -> Builder {
        Builder {
            config: Config::default(),
            thompson: thompson::Builder::new(),
        }
    }

    /// Build a lazy DFA from the given pattern.
    ///
    /// If there was a problem parsing or compiling the pattern, then an error
    /// is returned.
    pub fn build(&self, pattern: &str) -> Result<DFA, BuildError> {
        self.build_many(&[pattern])
    }

    /// Build a lazy DFA from the given patterns.
    ///
    /// When matches are returned, the pattern ID corresponds to the index of
    /// the pattern in the slice given.
    pub fn build_many<P: AsRef<str>>(
        &self,
        patterns: &[P],
    ) -> Result<DFA, BuildError> {
        let nfa =
            self.thompson.build_many(patterns).map_err(BuildError::nfa)?;
        self.build_from_nfa(Arc::new(nfa))
    }

    /// Build a DFA from the given NFA.
    ///
    /// Note that this requires an `Arc<thompson::NFA>` instead of a
    /// `&thompson::NFA` because the lazy DFA builds itself from the NFA at
    /// search time. This means that the lazy DFA must hold on to its source
    /// NFA for the entirety of its lifetime. An `Arc` is used so that callers
    /// aren't forced to clone the NFA if it is needed elsewhere.
    ///
    /// # Example
    ///
    /// This example shows how to build a lazy DFA if you already have an NFA
    /// in hand.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use regex_automata::{hybrid::dfa::DFA, nfa::thompson, HalfMatch};
    ///
    /// let haystack = "foo123bar".as_bytes();
    ///
    /// // This shows how to set non-default options for building an NFA.
    /// let nfa = thompson::Builder::new()
    ///     .configure(thompson::Config::new().shrink(false))
    ///     .build(r"[0-9]+")?;
    /// let dfa = DFA::builder().build_from_nfa(Arc::new(nfa))?;
    /// let mut cache = dfa.create_cache();
    /// let expected = Some(HalfMatch::must(0, 6));
    /// let got = dfa.find_leftmost_fwd(&mut cache, haystack)?;
    /// assert_eq!(expected, got);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn build_from_nfa(
        &self,
        nfa: Arc<thompson::NFA>,
    ) -> Result<DFA, BuildError> {
        let quitset = self.config.quit_set_from_nfa(&nfa)?;
        let classes = self.config.byte_classes_from_nfa(&nfa, &quitset);
        // Check that we can fit at least a few states into our cache,
        // otherwise it's pretty senseless to use the lazy DFA. This does have
        // a possible failure mode though. This assumes the maximum size of a
        // state in powerset space (so, the total number of NFA states), which
        // may never actually materialize, and could be quite a bit larger
        // than the actual biggest state. If this turns out to be a problem,
        // we could expose a knob that disables this check. But if so, we have
        // to be careful not to panic in other areas of the code (the cache
        // clearing and init code) that tend to assume some minimum useful
        // cache capacity.
        let min_cache = minimum_cache_capacity(
            &nfa,
            &classes,
            self.config.get_starts_for_each_pattern(),
        );
        let mut cache_capacity = self.config.get_cache_capacity();
        if cache_capacity < min_cache {
            // When the caller has asked us to skip the cache capacity check,
            // then we simply force the cache capacity to its minimum amount
            // and mush on.
            if self.config.get_skip_cache_capacity_check() {
                trace!(
                    "given capacity ({}) is too small, \
                     since skip_cache_capacity_check is enabled, \
                     setting cache capacity to minimum ({})",
                    cache_capacity,
                    min_cache,
                );
                cache_capacity = min_cache;
            } else {
                return Err(BuildError::insufficient_cache_capacity(
                    min_cache,
                    cache_capacity,
                ));
            }
        }
        // We also need to check that we can fit at least some small number
        // of states in our state ID space. This is unlikely to trigger in
        // >=32-bit systems, but 16-bit systems have a pretty small state ID
        // space since a number of bits are used up as sentinels.
        if let Err(err) = minimum_lazy_state_id(&nfa, &classes) {
            return Err(BuildError::insufficient_state_id_capacity(err));
        }
        let stride2 = classes.stride2();
        Ok(DFA {
            nfa,
            stride2,
            classes,
            quitset,
            anchored: self.config.get_anchored(),
            match_kind: self.config.get_match_kind(),
            starts_for_each_pattern: self.config.get_starts_for_each_pattern(),
            cache_capacity,
            minimum_cache_clear_count: self
                .config
                .get_minimum_cache_clear_count(),
        })
    }

    /// Apply the given lazy DFA configuration options to this builder.
    pub fn configure(&mut self, config: Config) -> &mut Builder {
        self.config = self.config.overwrite(config);
        self
    }

    /// Set the syntax configuration for this builder using
    /// [`SyntaxConfig`](crate::SyntaxConfig).
    ///
    /// This permits setting things like case insensitivity, Unicode and multi
    /// line mode.
    ///
    /// These settings only apply when constructing a lazy DFA directly from a
    /// pattern.
    pub fn syntax(
        &mut self,
        config: crate::util::syntax::SyntaxConfig,
    ) -> &mut Builder {
        self.thompson.syntax(config);
        self
    }

    /// Set the Thompson NFA configuration for this builder using
    /// [`nfa::thompson::Config`](crate::nfa::thompson::Config).
    ///
    /// This permits setting things like whether the DFA should match the regex
    /// in reverse or if additional time should be spent shrinking the size of
    /// the NFA.
    ///
    /// These settings only apply when constructing a DFA directly from a
    /// pattern.
    pub fn thompson(&mut self, config: thompson::Config) -> &mut Builder {
        self.thompson.configure(config);
        self
    }
}

/// Based on the minimum number of states required for a useful lazy DFA cache,
/// this returns the minimum lazy state ID that must be representable.
///
/// It's likely not plausible for this to impose constraints on 32-bit systems
/// (or higher), but on 16-bit systems, the lazy state ID space is quite
/// constrained and thus may be insufficient for bigger regexes.
fn minimum_lazy_state_id(
    nfa: &thompson::NFA,
    classes: &ByteClasses,
) -> Result<LazyStateID, LazyStateIDError> {
    let stride = 1 << classes.stride2();
    let min_state_index = MIN_STATES.checked_sub(1).unwrap();
    LazyStateID::new(min_state_index * stride)
}

/// Based on the minimum number of states required for a useful lazy DFA cache,
/// this returns a heuristic minimum number of bytes of heap space required.
///
/// This is a "heuristic" because the minimum it returns is likely bigger than
/// the true minimum. Namely, it assumes that each powerset NFA/DFA state uses
/// the maximum number of NFA states (all of them). This is likely bigger
/// than what is required in practice. Computing the true minimum effectively
/// requires determinization, which is probably too much work to do for a
/// simple check like this.
fn minimum_cache_capacity(
    nfa: &thompson::NFA,
    classes: &ByteClasses,
    starts_for_each_pattern: bool,
) -> usize {
    const ID_SIZE: usize = size_of::<LazyStateID>();
    let stride = 1 << classes.stride2();

    let sparses = 2 * nfa.len() * NFAStateID::SIZE;
    let trans = MIN_STATES * stride * ID_SIZE;

    let mut starts = Start::count() * ID_SIZE;
    if starts_for_each_pattern {
        starts += (Start::count() * nfa.match_len()) * ID_SIZE;
    }

    // Every `State` has three bytes for flags, 4 bytes (max) for the number
    // of patterns, followed by 32-bit encodings of patterns and then delta
    // varint encodings of NFA state IDs. We use the worst case (which isn't
    // technically possible) of 5 bytes for each NFA state ID.
    let max_state_size = 3 + 4 + (nfa.match_len() * 4) + (nfa.len() * 5);
    let states = MIN_STATES * (size_of::<State>() + max_state_size);
    let states_to_sid = states + (MIN_STATES * ID_SIZE);
    let stack = nfa.len() * NFAStateID::SIZE;
    let scratch_state_builder = max_state_size;

    trans
        + starts
        + states
        + states_to_sid
        + sparses
        + stack
        + scratch_state_builder
}
