use crate::dfa::automaton::{Automaton, State};
#[cfg(feature = "std")]
use crate::dfa::dense;
#[cfg(feature = "std")]
use crate::dfa::error::Error;
#[cfg(feature = "std")]
use crate::dfa::sparse;
use crate::nfa::thompson;
use crate::prefilter::{self, Prefilter};
#[cfg(feature = "std")]
use crate::state_id::StateID;
use crate::{Match, MatchKind, MultiMatch, NoMatch};

/// A regular expression that uses deterministic finite automata for fast
/// searching.
///
/// A regular expression is comprised of two DFAs, a "forward" DFA and a
/// "reverse" DFA. The forward DFA is responsible for detecting the end of a
/// match while the reverse DFA is responsible for detecting the start of a
/// match. Thus, in order to find the bounds of any given match, a forward
/// search must first be run followed by a reverse search. A match found by
/// the forward DFA guarantees that the reverse DFA will also find a match.
///
/// The type of the DFA used by a `Regex` corresponds to the `A` type
/// parameter, which must satisfy the [`Automaton`](trait.Automaton.html)
/// trait. Typically, `A` is either a
/// [`dense::DFA`](dense/struct.DFA.html)
/// or a
/// [`sparse::DFA`](sparse/struct.DFA.html),
/// where dense DFAs use more memory but search faster, while sparse DFAs use
/// less memory but search more slowly.
///
/// By default, a regex's automaton type parameter is set to
/// `dense::DFA<Vec<usize>, Vec<u8>, usize>`. For most in-memory work loads,
/// this is the most convenient type that gives the best search performance.
///
/// # Sparse DFAs
///
/// Since a `Regex` is generic over the `Automaton` trait, it can be used with
/// any kind of DFA. While this crate constructs dense DFAs by default, it is
/// easy enough to build corresponding sparse DFAs, and then build a regex from
/// them:
///
/// ```
/// use regex_automata::dfa::Regex;
///
/// # fn example() -> Result<(), regex_automata::dfa::Error> {
/// // First, build a regex that uses dense DFAs.
/// let dense_re = Regex::new("foo[0-9]+")?;
///
/// // Second, build sparse DFAs from the forward and reverse dense DFAs.
/// let fwd = dense_re.forward().to_sparse()?;
/// let rev = dense_re.reverse().to_sparse()?;
///
/// // Third, build a new regex from the constituent sparse DFAs.
/// let sparse_re = Regex::from_dfas(fwd, rev);
///
/// // A regex that uses sparse DFAs can be used just like with dense DFAs.
/// assert_eq!(true, sparse_re.is_match(b"foo123"));
/// # Ok(()) }; example().unwrap()
/// ```
#[cfg(feature = "std")]
#[derive(Clone, Debug)]
pub struct Regex<A = dense::OwnedDFA<usize>, P = prefilter::None> {
    prefilter: Option<P>,
    forward: A,
    reverse: A,
}

/// A regular expression that uses deterministic finite automata for fast
/// searching.
///
/// A regular expression is comprised of two DFAs, a "forward" DFA and a
/// "reverse" DFA. The forward DFA is responsible for detecting the end of a
/// match while the reverse DFA is responsible for detecting the start of a
/// match. Thus, in order to find the bounds of any given match, a forward
/// search must first be run followed by a reverse search. A match found by
/// the forward DFA guarantees that the reverse DFA will also find a match.
///
/// The type of the DFA used by a `Regex` corresponds to the `A` type
/// parameter, which must satisfy the [`Automaton`](trait.Automaton.html)
/// trait. Typically, `A` is either a
/// [`dense::DFA`](dense/struct.DFA.html)
/// or a
/// [`sparse::DFA`](sparse/struct.DFA.html),
/// where dense DFAs use more memory but search faster, while sparse DFAs use
/// less memory but search more slowly.
///
/// When using this crate without the standard library, the `Regex` type has
/// no default type parameter.
///
/// # Sparse DFAs
///
/// Since a `Regex` is generic over the `Automaton` trait, it can be used with
/// any kind of DFA. While this crate constructs dense DFAs by default, it is
/// easy enough to build corresponding sparse DFAs, and then build a regex from
/// them:
///
/// ```
/// use regex_automata::dfa::Regex;
///
/// # fn example() -> Result<(), regex_automata::dfa::Error> {
/// // First, build a regex that uses dense DFAs.
/// let dense_re = Regex::new("foo[0-9]+")?;
///
/// // Second, build sparse DFAs from the forward and reverse dense DFAs.
/// let fwd = dense_re.forward().to_sparse()?;
/// let rev = dense_re.reverse().to_sparse()?;
///
/// // Third, build a new regex from the constituent sparse DFAs.
/// let sparse_re = Regex::from_dfas(fwd, rev);
///
/// // A regex that uses sparse DFAs can be used just like with dense DFAs.
/// assert_eq!(true, sparse_re.is_match(b"foo123"));
/// # Ok(()) }; example().unwrap()
/// ```
#[cfg(not(feature = "std"))]
#[derive(Clone, Debug)]
pub struct Regex<A, P = prefilter::None> {
    prefilter: Option<P>,
    forward: A,
    reverse: A,
}

#[cfg(feature = "std")]
impl Regex {
    /// Parse the given regular expression using a default configuration and
    /// return the corresponding regex.
    ///
    /// The default configuration uses `usize` for state IDs. The underlying
    /// DFAs are *not* minimized.
    ///
    /// If you want a non-default configuration, then use the
    /// [`RegexBuilder`](struct.RegexBuilder.html)
    /// to set your own configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{MultiMatch, dfa::Regex};
    ///
    /// let re = Regex::new("foo[0-9]+bar")?;
    /// assert_eq!(
    ///     Some(MultiMatch::new(0, 3, 14)),
    ///     re.find_leftmost(b"zzzfoo12345barzzz"),
    /// );
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(pattern: &str) -> Result<Regex, Error> {
        RegexBuilder::new().build(pattern)
    }
}

#[cfg(feature = "std")]
impl Regex<sparse::DFA<Vec<u8>, usize>> {
    /// Parse the given regular expression using a default configuration and
    /// return the corresponding regex using sparse DFAs.
    ///
    /// The default configuration uses `usize` for state IDs, reduces the
    /// alphabet size by splitting bytes into equivalence classes. The
    /// underlying DFAs are *not* minimized.
    ///
    /// If you want a non-default configuration, then use the
    /// [`RegexBuilder`](struct.RegexBuilder.html)
    /// to set your own configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{MultiMatch, dfa::Regex};
    ///
    /// let re = Regex::new_sparse("foo[0-9]+bar")?;
    /// assert_eq!(
    ///     Some(MultiMatch::new(0, 3, 14)),
    ///     re.find_leftmost(b"zzzfoo12345barzzz"),
    /// );
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new_sparse(
        pattern: &str,
    ) -> Result<Regex<sparse::DFA<Vec<u8>, usize>>, Error> {
        RegexBuilder::new().build_sparse(pattern)
    }
}

impl<A: Automaton, P: Prefilter> Regex<A, P> {
    /// Returns true if and only if the given bytes match.
    ///
    /// This routine may short circuit if it knows that scanning future input
    /// will never lead to a different result. In particular, if the underlying
    /// DFA enters a match state or a dead state, then this routine will return
    /// `true` or `false`, respectively, without inspecting any future input.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::dfa::Regex;
    ///
    /// let re = Regex::new("foo[0-9]+bar")?;
    /// assert_eq!(true, re.is_match(b"foo12345bar"));
    /// assert_eq!(false, re.is_match(b"foobar"));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn is_match(&self, input: &[u8]) -> bool {
        self.is_match_at(input, 0, input.len())
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
    /// ```
    /// use regex_automata::{MultiMatch, dfa::Regex};
    ///
    /// let re = Regex::new("foo[0-9]+")?;
    /// assert_eq!(
    ///     Some(MultiMatch::new(0, 0, 4)),
    ///     re.find_earliest(b"foo12345"),
    /// );
    ///
    /// // Normally, the end of the leftmost first match here would be 3,
    /// // but the shortest match semantics detect a match earlier.
    /// let re = Regex::new("abc|a")?;
    /// assert_eq!(Some(MultiMatch::new(0, 0, 1)), re.find_earliest(b"abc"));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn find_earliest(&self, input: &[u8]) -> Option<MultiMatch> {
        self.find_earliest_at(input, 0, input.len())
    }

    /// Returns the start and end offset of the leftmost first match. If no
    /// match exists, then `None` is returned.
    ///
    /// The "leftmost first" match corresponds to the match with the smallest
    /// starting offset, but where the end offset is determined by preferring
    /// earlier branches in the original regular expression. For example,
    /// `Sam|Samwise` will match `Sam` in `Samwise`, but `Samwise|Sam` will
    /// match `Samwise` in `Samwise`.
    ///
    /// Generally speaking, the "leftmost first" match is how most backtracking
    /// regular expressions tend to work. This is in contrast to POSIX-style
    /// regular expressions that yield "leftmost longest" matches. Namely,
    /// both `Sam|Samwise` and `Samwise|Sam` match `Samwise` when using
    /// leftmost longest semantics.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{MultiMatch, dfa::Regex};
    ///
    /// let re = Regex::new("foo[0-9]+")?;
    /// assert_eq!(
    ///     Some(MultiMatch::new(0, 3, 11)),
    ///     re.find_leftmost(b"zzzfoo12345zzz"),
    /// );
    ///
    /// // Even though a match is found after reading the first byte (`a`),
    /// // the leftmost first match semantics demand that we find the earliest
    /// // match that prefers earlier parts of the pattern over latter parts.
    /// let re = Regex::new("abc|a")?;
    /// assert_eq!(Some(MultiMatch::new(0, 0, 3)), re.find_leftmost(b"abc"));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn find_leftmost(&self, input: &[u8]) -> Option<MultiMatch> {
        self.find_leftmost_at(input, 0, input.len())
    }

    pub fn find_overlapping(
        &self,
        input: &[u8],
        state: &mut State<A::ID>,
    ) -> Option<MultiMatch> {
        self.find_overlapping_at(input, 0, input.len(), state)
    }

    pub fn find_earliest_iter<'r, 't>(
        &'r self,
        input: &'t [u8],
    ) -> FindEarliestMatches<'r, 't, A, P> {
        FindEarliestMatches::new(self, input)
    }

    /// Returns an iterator over all non-overlapping leftmost first matches
    /// in the given bytes. If no match exists, then the iterator yields no
    /// elements.
    ///
    /// Note that if the regex can match the empty string, then it is
    /// possible for the iterator to yield a zero-width match at a location
    /// that is not a valid UTF-8 boundary (for example, between the code units
    /// of a UTF-8 encoded codepoint). This can happen regardless of whether
    /// [`allow_invalid_utf8`](struct.RegexBuilder.html#method.allow_invalid_utf8)
    /// was enabled or not.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{MultiMatch, dfa::Regex};
    ///
    /// let re = Regex::new("foo[0-9]+")?;
    /// let text = b"foo1 foo12 foo123";
    /// let matches: Vec<MultiMatch> = re.find_leftmost_iter(text).collect();
    /// assert_eq!(matches, vec![
    ///     MultiMatch::new(0, 0, 4),
    ///     MultiMatch::new(0, 5, 10),
    ///     MultiMatch::new(0, 11, 17),
    /// ]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn find_leftmost_iter<'r, 't>(
        &'r self,
        input: &'t [u8],
    ) -> FindLeftmostMatches<'r, 't, A, P> {
        FindLeftmostMatches::new(self, input)
    }

    pub fn find_overlapping_iter<'r, 't>(
        &'r self,
        input: &'t [u8],
    ) -> FindOverlappingMatches<'r, 't, A, P> {
        FindOverlappingMatches::new(self, input)
    }

    /// Attach the given prefilter to this regex.
    pub fn with_prefilter<Q: Prefilter>(self, prefilter: Q) -> Regex<A, Q> {
        Regex {
            prefilter: Some(prefilter),
            forward: self.forward,
            reverse: self.reverse,
        }
    }

    /// Remove any prefilter from this regex.
    pub fn without_prefilter(self) -> Regex<A> {
        Regex { prefilter: None, forward: self.forward, reverse: self.reverse }
    }

    /// Return the underlying DFA responsible for forward matching.
    pub fn forward(&self) -> &A {
        &self.forward
    }

    /// Return the underlying DFA responsible for reverse matching.
    pub fn reverse(&self) -> &A {
        &self.reverse
    }

    /// Returns the total number of patterns matched by this regex.
    pub fn patterns(&self) -> usize {
        assert_eq!(self.forward().patterns(), self.reverse().patterns());
        self.forward().patterns()
    }

    /// Convenience function for returning a prefilter scanner.
    fn scanner(&self) -> Option<prefilter::Scanner> {
        self.prefilter().map(prefilter::Scanner::new)
    }

    /// Convenience function for returning this regex's prefilter as a trait
    /// object.
    fn prefilter(&self) -> Option<&dyn Prefilter> {
        match self.prefilter {
            None => None,
            Some(ref x) => Some(&*x),
        }
    }
}

impl<A: Automaton> Regex<A> {
    /// Build a new regex from its constituent forward and reverse DFAs.
    ///
    /// This is useful when deserializing a regex from some arbitrary
    /// memory region. This is also useful for building regexes from other
    /// types of DFAs.
    ///
    /// # Example
    ///
    /// This example is a bit a contrived. The usual use of these methods
    /// would involve serializing `initial_re` somewhere and then deserializing
    /// it later to build a regex.
    ///
    /// ```
    /// use regex_automata::dfa::Regex;
    ///
    /// let initial_re = Regex::new("foo[0-9]+")?;
    /// assert_eq!(true, initial_re.is_match(b"foo123"));
    ///
    /// let (fwd, rev) = (initial_re.forward(), initial_re.reverse());
    /// let re = Regex::from_dfas(fwd, rev);
    /// assert_eq!(true, re.is_match(b"foo123"));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// This example shows how you might build smaller DFAs, and then use those
    /// smaller DFAs to build a new regex.
    ///
    /// ```
    /// use regex_automata::dfa::Regex;
    ///
    /// let initial_re = Regex::new("foo[0-9]+")?;
    /// assert_eq!(true, initial_re.is_match(b"foo123"));
    ///
    /// let fwd = initial_re.forward().to_sized::<u16>()?;
    /// let rev = initial_re.reverse().to_sized::<u16>()?;
    /// let re = Regex::from_dfas(fwd, rev);
    /// assert_eq!(true, re.is_match(b"foo123"));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// This example shows how to build a `Regex` that uses sparse DFAs instead
    /// of dense DFAs:
    ///
    /// ```
    /// use regex_automata::dfa::Regex;
    ///
    /// let initial_re = Regex::new("foo[0-9]+")?;
    /// assert_eq!(true, initial_re.is_match(b"foo123"));
    ///
    /// let fwd = initial_re.forward().to_sparse()?;
    /// let rev = initial_re.reverse().to_sparse()?;
    /// let re = Regex::from_dfas(fwd, rev);
    /// assert_eq!(true, re.is_match(b"foo123"));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_dfas(forward: A, reverse: A) -> Regex<A> {
        Regex { prefilter: None, forward, reverse }
    }
}

/// Lower level infallible search routines that permit controlling where the
/// search starts and ends in a particular sequence.
impl<A: Automaton, P: Prefilter> Regex<A, P> {
    /// Returns the same as `is_match`, but starts the search at the given
    /// offset.
    ///
    /// The significance of the starting point is that it takes the surrounding
    /// context into consideration. For example, if the DFA is anchored, then
    /// a match can only occur when `start == 0`.
    pub fn is_match_at(&self, input: &[u8], start: usize, end: usize) -> bool {
        self.try_is_match_at(input, start, end).unwrap()
    }

    /// Returns the same as `earliest_match`, but starts the search at the
    /// given offsets.
    ///
    /// The significance of the starting point is that it takes the surrounding
    /// context into consideration. For example, if the DFA is anchored, then
    /// a match can only occur when `start == 0`.
    pub fn find_earliest_at(
        &self,
        input: &[u8],
        start: usize,
        end: usize,
    ) -> Option<MultiMatch> {
        self.try_find_earliest_at(input, start, end).unwrap()
    }

    /// Returns the same as `find`, but starts the search at the given
    /// offset.
    ///
    /// The significance of the starting point is that it takes the surrounding
    /// context into consideration. For example, if the DFA is anchored, then
    /// a match can only occur when `start == 0`.
    pub fn find_leftmost_at(
        &self,
        input: &[u8],
        start: usize,
        end: usize,
    ) -> Option<MultiMatch> {
        self.try_find_leftmost_at(input, start, end).unwrap()
    }

    pub fn find_overlapping_at(
        &self,
        input: &[u8],
        start: usize,
        end: usize,
        state: &mut State<A::ID>,
    ) -> Option<MultiMatch> {
        self.try_find_overlapping_at(input, start, end, state).unwrap()
    }
}

/// Fallible search routines. These may return an error when the underlying
/// DFAs have been configured in a way that permits them to fail during a
/// search.
///
/// Errors during search only occur when the DFA has been explicitly
/// configured to do so, usually by specifying one or more "quit" bytes or by
/// heuristically enabling Unicode word boundaries.
///
/// Errors will never be returned using the default configuration. So these
/// fallible routines are only needed for particular configurations.
impl<A: Automaton, P: Prefilter> Regex<A, P> {
    pub fn try_is_match(&self, input: &[u8]) -> Result<bool, NoMatch> {
        self.try_is_match_at(input, 0, input.len())
    }

    pub fn try_find_earliest(
        &self,
        input: &[u8],
    ) -> Result<Option<MultiMatch>, NoMatch> {
        self.try_find_earliest_at(input, 0, input.len())
    }

    pub fn try_find_leftmost(
        &self,
        input: &[u8],
    ) -> Result<Option<MultiMatch>, NoMatch> {
        self.try_find_leftmost_at(input, 0, input.len())
    }

    pub fn try_find_overlapping(
        &self,
        input: &[u8],
        state: &mut State<A::ID>,
    ) -> Result<Option<MultiMatch>, NoMatch> {
        self.try_find_overlapping_at(input, 0, input.len(), state)
    }

    pub fn try_find_earliest_iter<'r, 't>(
        &'r self,
        input: &'t [u8],
    ) -> TryFindEarliestMatches<'r, 't, A, P> {
        TryFindEarliestMatches::new(self, input)
    }

    pub fn try_find_leftmost_iter<'r, 't>(
        &'r self,
        input: &'t [u8],
    ) -> TryFindLeftmostMatches<'r, 't, A, P> {
        TryFindLeftmostMatches::new(self, input)
    }

    pub fn try_find_overlapping_iter<'r, 't>(
        &'r self,
        input: &'t [u8],
    ) -> TryFindOverlappingMatches<'r, 't, A, P> {
        TryFindOverlappingMatches::new(self, input)
    }
}

/// Lower level fallible search routines that permit controlling where the
/// search starts and ends in a particular sequence.
impl<A: Automaton, P: Prefilter> Regex<A, P> {
    /// Returns the same as `is_match`, but starts the search at the given
    /// offset.
    ///
    /// The significance of the starting point is that it takes the surrounding
    /// context into consideration. For example, if the DFA is anchored, then
    /// a match can only occur when `start == 0`.
    pub fn try_is_match_at(
        &self,
        input: &[u8],
        start: usize,
        end: usize,
    ) -> Result<bool, NoMatch> {
        self.forward()
            .find_earliest_fwd_at(self.scanner().as_mut(), input, start, end)
            .map(|x| x.is_some())
    }

    /// Returns the same as `earliest_match`, but starts the search at the
    /// given offsets.
    ///
    /// The significance of the starting point is that it takes the surrounding
    /// context into consideration. For example, if the DFA is anchored, then
    /// a match can only occur when `start == 0`.
    pub fn try_find_earliest_at(
        &self,
        input: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Option<MultiMatch>, NoMatch> {
        self.try_find_earliest_at_imp(
            self.scanner().as_mut(),
            input,
            start,
            end,
        )
    }

    fn try_find_earliest_at_imp(
        &self,
        pre: Option<&mut prefilter::Scanner>,
        input: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Option<MultiMatch>, NoMatch> {
        // N.B. We use `&&A` here to call `Automaton` methods, which ensures
        // that we always use the `impl Automaton for &A` for calling methods.
        // Since this is the usual way that automata are used, this helps
        // reduce the number of monomorphized copies of the search code.
        let (fwd, rev) = (self.forward(), self.reverse());
        let end = match (&fwd).find_earliest_fwd_at(pre, input, start, end)? {
            None => return Ok(None),
            Some(end) => end,
        };
        let start = (&rev)
            .find_earliest_rev_at(input, start, end.offset())?
            .expect("reverse search must match if forward search does");
        assert_eq!(
            start.pattern(),
            end.pattern(),
            "forward and reverse search must match same pattern"
        );
        assert!(start.offset() <= end.offset());
        Ok(Some(MultiMatch::new(end.pattern(), start.offset(), end.offset())))
    }

    /// Returns the same as `find`, but starts the search at the given
    /// offset.
    ///
    /// The significance of the starting point is that it takes the surrounding
    /// context into consideration. For example, if the DFA is anchored, then
    /// a match can only occur when `start == 0`.
    pub fn try_find_leftmost_at(
        &self,
        input: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Option<MultiMatch>, NoMatch> {
        self.try_find_leftmost_at_imp(
            self.scanner().as_mut(),
            input,
            start,
            end,
        )
    }

    fn try_find_leftmost_at_imp(
        &self,
        scanner: Option<&mut prefilter::Scanner>,
        input: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Option<MultiMatch>, NoMatch> {
        // N.B. We use `&&A` here to call `Automaton` methods, which ensures
        // that we always use the `impl Automaton for &A` for calling methods.
        // Since this is the usual way that automata are used, this helps
        // reduce the number of monomorphized copies of the search code.
        let (fwd, rev) = (self.forward(), self.reverse());
        let end =
            match (&fwd).find_leftmost_fwd_at(scanner, input, start, end)? {
                None => return Ok(None),
                Some(end) => end,
            };
        let start = (&rev)
            .find_leftmost_rev_at(input, start, end.offset())?
            .expect("reverse search must match if forward search does");
        assert_eq!(
            start.pattern(),
            end.pattern(),
            "forward and reverse search must match same pattern"
        );
        assert!(start.offset() <= end.offset());
        Ok(Some(MultiMatch::new(end.pattern(), start.offset(), end.offset())))
    }

    pub fn try_find_overlapping_at(
        &self,
        input: &[u8],
        start: usize,
        end: usize,
        state: &mut State<A::ID>,
    ) -> Result<Option<MultiMatch>, NoMatch> {
        self.try_find_overlapping_at_imp(
            self.scanner().as_mut(),
            input,
            start,
            end,
            state,
        )
    }

    fn try_find_overlapping_at_imp(
        &self,
        scanner: Option<&mut prefilter::Scanner>,
        input: &[u8],
        start: usize,
        end: usize,
        state: &mut State<A::ID>,
    ) -> Result<Option<MultiMatch>, NoMatch> {
        // N.B. We use `&&A` here to call `Automaton` methods, which ensures
        // that we always use the `impl Automaton for &A` for calling methods.
        // Since this is the usual way that automata are used, this helps
        // reduce the number of monomorphized copies of the search code.
        let (fwd, rev) = (self.forward(), self.reverse());
        let end = match (&fwd)
            .find_overlapping_fwd_at(scanner, input, start, end, state)?
        {
            None => return Ok(None),
            Some(end) => end,
        };
        let start = (&rev)
            .find_leftmost_rev_at(input, 0, end.offset())?
            .expect("reverse search must match if forward search does");
        // Unlike in the leftmost cases, in the overlapping case, the reverse
        // search may not match the same pattern as the forward search.
        // Consider a trivial case such as searching the patterns [a, a]
        // against 'a'. A second forward search will find the second pattern,
        // but the same reverse search lacks the prior search context and will
        // instead yield the first pattern.
        assert!(start.offset() <= end.offset());
        Ok(Some(MultiMatch::new(end.pattern(), start.offset(), end.offset())))
    }
}

/// An iterator over all non-overlapping earliest matches for a particular
/// search.
///
/// The iterator yields a `(usize, usize)` value until no more matches could be
/// found. The first `usize` is the start of the match (inclusive) while the
/// second `usize` is the end of the match (exclusive).
///
/// `A` is the type used to represent the underlying DFAs used by the regex.
/// The lifetime variables are as follows:
///
/// * `'r` is the lifetime of the regular expression value itself.
/// * `'t` is the lifetime of the text being searched.
#[derive(Clone, Debug)]
pub struct FindEarliestMatches<'r, 't, A, P>(
    TryFindEarliestMatches<'r, 't, A, P>,
);

impl<'r, 't, A: Automaton, P: Prefilter> FindEarliestMatches<'r, 't, A, P> {
    fn new(
        re: &'r Regex<A, P>,
        text: &'t [u8],
    ) -> FindEarliestMatches<'r, 't, A, P> {
        FindEarliestMatches(TryFindEarliestMatches::new(re, text))
    }
}

impl<'r, 't, A: Automaton, P: Prefilter> Iterator
    for FindEarliestMatches<'r, 't, A, P>
{
    type Item = MultiMatch;

    fn next(&mut self) -> Option<MultiMatch> {
        next_unwrap(self.0.next())
    }
}

/// An iterator over all non-overlapping matches for a particular search.
///
/// The iterator yields a `(usize, usize)` value until no more matches could be
/// found. The first `usize` is the start of the match (inclusive) while the
/// second `usize` is the end of the match (exclusive).
///
/// `A` is the type used to represent the underlying DFAs used by the regex.
/// The lifetime variables are as follows:
///
/// * `'r` is the lifetime of the regular expression value itself.
/// * `'t` is the lifetime of the text being searched.
#[derive(Clone, Debug)]
pub struct FindLeftmostMatches<'r, 't, A, P>(
    TryFindLeftmostMatches<'r, 't, A, P>,
);

impl<'r, 't, A: Automaton, P: Prefilter> FindLeftmostMatches<'r, 't, A, P> {
    fn new(
        re: &'r Regex<A, P>,
        text: &'t [u8],
    ) -> FindLeftmostMatches<'r, 't, A, P> {
        FindLeftmostMatches(TryFindLeftmostMatches::new(re, text))
    }
}

impl<'r, 't, A: Automaton, P: Prefilter> Iterator
    for FindLeftmostMatches<'r, 't, A, P>
{
    type Item = MultiMatch;

    fn next(&mut self) -> Option<MultiMatch> {
        next_unwrap(self.0.next())
    }
}

/// An iterator over all overlapping matches for a particular search.
///
/// The iterator yields a `(usize, usize)` value until no more matches could be
/// found. The first `usize` is the start of the match (inclusive) while the
/// second `usize` is the end of the match (exclusive).
///
/// `A` is the type used to represent the underlying DFAs used by the regex.
/// The lifetime variables are as follows:
///
/// * `'r` is the lifetime of the regular expression value itself.
/// * `'t` is the lifetime of the text being searched.
#[derive(Clone, Debug)]
pub struct FindOverlappingMatches<'r, 't, A: Automaton, P>(
    TryFindOverlappingMatches<'r, 't, A, P>,
);

impl<'r, 't, A: Automaton, P: Prefilter> FindOverlappingMatches<'r, 't, A, P> {
    fn new(
        re: &'r Regex<A, P>,
        text: &'t [u8],
    ) -> FindOverlappingMatches<'r, 't, A, P> {
        FindOverlappingMatches(TryFindOverlappingMatches::new(re, text))
    }
}

impl<'r, 't, A: Automaton, P: Prefilter> Iterator
    for FindOverlappingMatches<'r, 't, A, P>
{
    type Item = MultiMatch;

    fn next(&mut self) -> Option<MultiMatch> {
        next_unwrap(self.0.next())
    }
}

/// An iterator over all non-overlapping earliest matches for a particular
/// search.
///
/// The iterator yields a `(usize, usize)` value until no more matches could be
/// found. The first `usize` is the start of the match (inclusive) while the
/// second `usize` is the end of the match (exclusive).
///
/// `A` is the type used to represent the underlying DFAs used by the regex.
/// The lifetime variables are as follows:
///
/// * `'r` is the lifetime of the regular expression value itself.
/// * `'t` is the lifetime of the text being searched.
#[derive(Clone, Debug)]
pub struct TryFindEarliestMatches<'r, 't, A, P> {
    re: &'r Regex<A, P>,
    scanner: Option<prefilter::Scanner<'r>>,
    text: &'t [u8],
    last_end: usize,
    last_match: Option<usize>,
}

impl<'r, 't, A: Automaton, P: Prefilter> TryFindEarliestMatches<'r, 't, A, P> {
    fn new(
        re: &'r Regex<A, P>,
        text: &'t [u8],
    ) -> TryFindEarliestMatches<'r, 't, A, P> {
        let scanner = re.scanner();
        TryFindEarliestMatches {
            re,
            scanner,
            text,
            last_end: 0,
            last_match: None,
        }
    }
}

impl<'r, 't, A: Automaton, P: Prefilter> Iterator
    for TryFindEarliestMatches<'r, 't, A, P>
{
    type Item = Result<MultiMatch, NoMatch>;

    fn next(&mut self) -> Option<Result<MultiMatch, NoMatch>> {
        if self.last_end > self.text.len() {
            return None;
        }
        let result = self.re.try_find_earliest_at_imp(
            self.scanner.as_mut(),
            self.text,
            self.last_end,
            self.text.len(),
        );
        let m = match result {
            Err(err) => return Some(Err(err)),
            Ok(None) => return None,
            Ok(Some(m)) => m,
        };
        if m.is_empty() {
            // This is an empty match. To ensure we make progress, start
            // the next search at the smallest possible starting position
            // of the next match following this one.
            self.last_end = m.end() + 1;
            // Don't accept empty matches immediately following a match.
            // Just move on to the next match.
            if Some(m.end()) == self.last_match {
                return self.next();
            }
        } else {
            self.last_end = m.end();
        }
        self.last_match = Some(m.end());
        Some(Ok(m))
    }
}

/// An iterator over all non-overlapping matches for a particular search.
///
/// The iterator yields a `(usize, usize)` value until no more matches could be
/// found. The first `usize` is the start of the match (inclusive) while the
/// second `usize` is the end of the match (exclusive).
///
/// `A` is the type used to represent the underlying DFAs used by the regex.
/// The lifetime variables are as follows:
///
/// * `'r` is the lifetime of the regular expression value itself.
/// * `'t` is the lifetime of the text being searched.
#[derive(Clone, Debug)]
pub struct TryFindLeftmostMatches<'r, 't, A, P> {
    re: &'r Regex<A, P>,
    scanner: Option<prefilter::Scanner<'r>>,
    text: &'t [u8],
    last_end: usize,
    last_match: Option<usize>,
}

impl<'r, 't, A: Automaton, P: Prefilter> TryFindLeftmostMatches<'r, 't, A, P> {
    fn new(
        re: &'r Regex<A, P>,
        text: &'t [u8],
    ) -> TryFindLeftmostMatches<'r, 't, A, P> {
        let scanner = re.scanner();
        TryFindLeftmostMatches {
            re,
            scanner,
            text,
            last_end: 0,
            last_match: None,
        }
    }
}

impl<'r, 't, A: Automaton, P: Prefilter> Iterator
    for TryFindLeftmostMatches<'r, 't, A, P>
{
    type Item = Result<MultiMatch, NoMatch>;

    fn next(&mut self) -> Option<Result<MultiMatch, NoMatch>> {
        if self.last_end > self.text.len() {
            return None;
        }
        let result = self.re.try_find_leftmost_at_imp(
            self.scanner.as_mut(),
            self.text,
            self.last_end,
            self.text.len(),
        );
        let m = match result {
            Err(err) => return Some(Err(err)),
            Ok(None) => return None,
            Ok(Some(m)) => m,
        };
        if m.is_empty() {
            // This is an empty match. To ensure we make progress, start
            // the next search at the smallest possible starting position
            // of the next match following this one.
            self.last_end = m.end() + 1;
            // Don't accept empty matches immediately following a match.
            // Just move on to the next match.
            if Some(m.end()) == self.last_match {
                return self.next();
            }
        } else {
            self.last_end = m.end();
        }
        self.last_match = Some(m.end());
        Some(Ok(m))
    }
}

/// An iterator over all overlapping matches for a particular search.
///
/// The iterator yields a `(usize, usize)` value until no more matches could be
/// found. The first `usize` is the start of the match (inclusive) while the
/// second `usize` is the end of the match (exclusive).
///
/// `A` is the type used to represent the underlying DFAs used by the regex.
/// The lifetime variables are as follows:
///
/// * `'r` is the lifetime of the regular expression value itself.
/// * `'t` is the lifetime of the text being searched.
#[derive(Clone, Debug)]
pub struct TryFindOverlappingMatches<'r, 't, A: Automaton, P> {
    re: &'r Regex<A, P>,
    scanner: Option<prefilter::Scanner<'r>>,
    text: &'t [u8],
    last_end: usize,
    state: State<A::ID>,
}

impl<'r, 't, A: Automaton, P: Prefilter>
    TryFindOverlappingMatches<'r, 't, A, P>
{
    fn new(
        re: &'r Regex<A, P>,
        text: &'t [u8],
    ) -> TryFindOverlappingMatches<'r, 't, A, P> {
        let scanner = re.scanner();
        TryFindOverlappingMatches {
            re,
            scanner,
            text,
            last_end: 0,
            state: State::start(),
        }
    }
}

impl<'r, 't, A: Automaton, P: Prefilter> Iterator
    for TryFindOverlappingMatches<'r, 't, A, P>
{
    type Item = Result<MultiMatch, NoMatch>;

    fn next(&mut self) -> Option<Result<MultiMatch, NoMatch>> {
        if self.last_end > self.text.len() {
            return None;
        }
        let result = self.re.try_find_overlapping_at_imp(
            self.scanner.as_mut(),
            self.text,
            self.last_end,
            self.text.len(),
            &mut self.state,
        );
        let m = match result {
            Err(err) => return Some(Err(err)),
            Ok(None) => return None,
            Ok(Some(m)) => m,
        };
        // Unlike the non-overlapping case, we're OK with empty matches at this
        // level. In particular, the overlapping search algorithm is itself
        // responsible for ensuring that progress is always made. (The starting
        // position of the search is incremented by 1 whenever a non-None state
        // ID is given.)
        self.last_end = m.end();
        Some(Ok(m))
    }
}

/// A builder for a regex based on deterministic finite automatons.
///
/// This builder permits configuring several aspects of the construction
/// process such as case insensitivity, Unicode support and various options
/// that impact the size of the underlying DFAs. In some cases, options (like
/// performing DFA minimization) can come with a substantial additional cost.
///
/// This builder generally constructs two DFAs, where one is responsible for
/// finding the end of a match and the other is responsible for finding the
/// start of a match. If you only need to detect whether something matched,
/// or only the end of a match, then you should use a
/// [`dense::Builder`](dense/struct.Builder.html)
/// to construct a single DFA, which is cheaper than building two DFAs.
#[cfg(feature = "std")]
#[derive(Clone, Debug)]
pub struct RegexBuilder {
    dfa: dense::Builder,
}

#[cfg(feature = "std")]
impl RegexBuilder {
    /// Create a new regex builder with the default configuration.
    pub fn new() -> RegexBuilder {
        RegexBuilder { dfa: dense::Builder::new() }
    }

    /// Build a regex from the given pattern.
    ///
    /// If there was a problem parsing or compiling the pattern, then an error
    /// is returned.
    pub fn build(&self, pattern: &str) -> Result<Regex, Error> {
        self.build_many(&[pattern])
    }

    /// Build a regex from the given pattern using sparse DFAs.
    ///
    /// If there was a problem parsing or compiling the pattern, then an error
    /// is returned.
    pub fn build_sparse(
        &self,
        pattern: &str,
    ) -> Result<Regex<sparse::DFA<Vec<u8>, usize>>, Error> {
        self.build_with_size_sparse::<usize>(pattern)
    }

    /// Build a regex from the given patterns.
    pub fn build_many<P: AsRef<str>>(
        &self,
        patterns: &[P],
    ) -> Result<Regex, Error> {
        self.build_many_with_size::<usize, _>(patterns)
    }

    /// Build a sparse regex from the given patterns.
    pub fn build_many_sparse<P: AsRef<str>>(
        &self,
        patterns: &[P],
    ) -> Result<Regex<sparse::DFA<Vec<u8>, usize>>, Error> {
        self.build_many_with_size_sparse::<usize, _>(patterns)
    }

    /// Build a regex from the given pattern using a specific representation
    /// for the underlying DFA state IDs.
    ///
    /// If there was a problem parsing or compiling the pattern, then an error
    /// is returned.
    ///
    /// The representation of state IDs is determined by the `S` type
    /// parameter. In general, `S` is usually one of `u8`, `u16`, `u32`, `u64`
    /// or `usize`, where `usize` is the default used for `build`. The purpose
    /// of specifying a representation for state IDs is to reduce the memory
    /// footprint of the underlying DFAs.
    ///
    /// When using this routine, the chosen state ID representation will be
    /// used throughout determinization and minimization, if minimization was
    /// requested. Even if the minimized DFAs can fit into the chosen state ID
    /// representation but the initial determinized DFA cannot, then this will
    /// still return an error. To get a minimized DFA with a smaller state ID
    /// representation, first build it with a bigger state ID representation,
    /// and then shrink the sizes of the DFAs using one of its conversion
    /// routines, such as
    /// [`dense::DFA::to_sized`](struct.DFA.html#method.to_sized).
    /// Finally, reconstitute the regex via
    /// [`Regex::from_dfa`](struct.Regex.html#method.from_dfa).
    pub fn build_with_size<S: StateID>(
        &self,
        pattern: &str,
    ) -> Result<Regex<dense::OwnedDFA<S>>, Error> {
        self.build_many_with_size(&[pattern])
    }

    /// Build a regex from the given pattern using a specific representation
    /// for the underlying DFA state IDs using sparse DFAs.
    pub fn build_with_size_sparse<S: StateID>(
        &self,
        pattern: &str,
    ) -> Result<Regex<sparse::DFA<Vec<u8>, S>>, Error> {
        self.build_many_with_size_sparse(&[pattern])
    }

    /// Build a regex from the given patterns using `S` as the state identifier
    /// representation.
    pub fn build_many_with_size<S: StateID, P: AsRef<str>>(
        &self,
        patterns: &[P],
    ) -> Result<Regex<dense::OwnedDFA<S>>, Error> {
        let forward = self.dfa.build_many_with_size(patterns)?;
        let reverse = self
            .dfa
            .clone()
            .configure(
                dense::Config::new().anchored(true).match_kind(MatchKind::All),
            )
            .thompson(thompson::Config::new().reverse(true))
            .build_many_with_size(patterns)?;
        Ok(Regex::from_dfas(forward, reverse))
    }

    /// Build a sparse regex from the given patterns using `S` as the state
    /// identifier representation.
    pub fn build_many_with_size_sparse<S: StateID, P: AsRef<str>>(
        &self,
        patterns: &[P],
    ) -> Result<Regex<sparse::DFA<Vec<u8>, S>>, Error> {
        let re = self.build_many_with_size(patterns)?;
        let fwd = re.forward().to_sparse()?;
        let rev = re.reverse().to_sparse()?;
        Ok(Regex::from_dfas(fwd, rev))
    }

    /// Set the syntax configuration for this builder using
    /// [`SyntaxConfig`](../struct.SyntaxConfig.html).
    ///
    /// This permits setting things like case insensitivity, Unicode and multi
    /// line mode.
    pub fn syntax(
        &mut self,
        config: crate::SyntaxConfig,
    ) -> &mut RegexBuilder {
        self.dfa.syntax(config);
        self
    }

    /// Set the Thompson NFA configuration for this builder using
    /// [`nfa::thompson::Config`](../nfa/thompson/struct.Config.html).
    ///
    /// This permits setting things like whether additional time should be
    /// spent shrinking the size of the NFA.
    pub fn thompson(&mut self, config: thompson::Config) -> &mut RegexBuilder {
        self.dfa.thompson(config);
        self
    }

    /// Set the dense DFA compilation configuration for this builder using
    /// [`dfa::dense::Config`](dense/struct.Config.html).
    ///
    /// This permits setting things like whether the underlying DFAs should
    /// be minimized.
    pub fn dense(&mut self, config: dense::Config) -> &mut RegexBuilder {
        self.dfa.configure(config);
        self
    }
}

#[cfg(feature = "std")]
impl Default for RegexBuilder {
    fn default() -> RegexBuilder {
        RegexBuilder::new()
    }
}

#[inline(always)]
fn next_unwrap(
    item: Option<Result<MultiMatch, NoMatch>>,
) -> Option<MultiMatch> {
    match item {
        None => None,
        Some(Ok(m)) => Some(m),
        Some(Err(err)) => panic!(
            "unexpected regex search error: {}\n\
             to handle search errors, use try_ methods",
            err,
        ),
    }
}
