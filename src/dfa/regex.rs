/*!
A DFA-backed `Regex`.

This module provides [`Regex`], which is defined generically over the
[`Automaton`] trait. A `Regex` implements convenience routines you might have
come to expect, such as finding the start/end of a match and iterating over
all non-overlapping matches. This `Regex` type is limited in its capabilities
to what a DFA can provide. Therefore, APIs involving capturing groups, for
example, are not provided.

Internally, a `Regex` is composed of two DFAs. One is a "forward" DFA that
finds the end offset of a match, where as the other is a "reverse" DFA that
find the start offset of a match.

See the [parent module](crate::dfa) for examples.
*/

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "dfa-build")]
use crate::dfa::dense::BuildError;
use crate::{
    dfa::{automaton::Automaton, dense},
    util::{iter, search::Input},
    Anchored, Match, MatchError,
};
#[cfg(feature = "alloc")]
use crate::{
    dfa::{sparse, StartKind},
    nfa::thompson,
    util::search::MatchKind,
};

// When the alloc feature is enabled, the regex type sets its A type parameter
// to default to an owned dense DFA. But without alloc, we set no default. This
// makes things a lot more convenient in the common case, since writing out the
// DFA types is pretty annoying.
//
// Since we have two different definitions but only want to write one doc
// string, we use a macro to capture the doc and other attributes once and then
// repeat them for each definition.
macro_rules! define_regex_type {
    ($(#[$doc:meta])*) => {
        #[cfg(feature = "alloc")]
        $(#[$doc])*
        pub struct Regex<A = dense::OwnedDFA> {
            forward: A,
            reverse: A,
        }

        #[cfg(not(feature = "alloc"))]
        $(#[$doc])*
        pub struct Regex<A> {
            forward: A,
            reverse: A,
        }
    };
}

define_regex_type!(
    /// A regular expression that uses deterministic finite automata for fast
    /// searching.
    ///
    /// A regular expression is comprised of two DFAs, a "forward" DFA and a
    /// "reverse" DFA. The forward DFA is responsible for detecting the end of
    /// a match while the reverse DFA is responsible for detecting the start
    /// of a match. Thus, in order to find the bounds of any given match, a
    /// forward search must first be run followed by a reverse search. A match
    /// found by the forward DFA guarantees that the reverse DFA will also find
    /// a match.
    ///
    /// The type of the DFA used by a `Regex` corresponds to the `A` type
    /// parameter, which must satisfy the [`Automaton`] trait. Typically,
    /// `A` is either a [`dense::DFA`](crate::dfa::dense::DFA) or a
    /// [`sparse::DFA`](crate::dfa::sparse::DFA), where dense DFAs use more
    /// memory but search faster, while sparse DFAs use less memory but search
    /// more slowly.
    ///
    /// By default, a regex's automaton type parameter is set to
    /// `dense::DFA<Vec<u32>>` when the `alloc` feature is enabled. For most
    /// in-memory work loads, this is the most convenient type that gives the
    /// best search performance. When the `alloc` feature is disabled, no
    /// default type is used.
    ///
    /// # When should I use this?
    ///
    /// Generally speaking, if you can afford the overhead of building a full
    /// DFA for your regex, and you don't need things like capturing groups,
    /// then this is a good choice if you're looking to optimize for matching
    /// speed. Note however that its speed may be worse than a general purpose
    /// regex engine if you don't provide a [`dense::Config::prefilter`] to the
    /// underlying DFA.
    ///
    /// # Sparse DFAs
    ///
    /// Since a `Regex` is generic over the [`Automaton`] trait, it can be
    /// used with any kind of DFA. While this crate constructs dense DFAs by
    /// default, it is easy enough to build corresponding sparse DFAs, and then
    /// build a regex from them:
    ///
    /// ```
    /// use regex_automata::dfa::regex::Regex;
    ///
    /// // First, build a regex that uses dense DFAs.
    /// let dense_re = Regex::new("foo[0-9]+")?;
    ///
    /// // Second, build sparse DFAs from the forward and reverse dense DFAs.
    /// let fwd = dense_re.forward().to_sparse()?;
    /// let rev = dense_re.reverse().to_sparse()?;
    ///
    /// // Third, build a new regex from the constituent sparse DFAs.
    /// let sparse_re = Regex::builder().build_from_dfas(fwd, rev);
    ///
    /// // A regex that uses sparse DFAs can be used just like with dense DFAs.
    /// assert_eq!(true, sparse_re.is_match(b"foo123"));
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// Alternatively, one can use a [`Builder`] to construct a sparse DFA
    /// more succinctly. (Note though that dense DFAs are still constructed
    /// first internally, and then converted to sparse DFAs, as in the example
    /// above.)
    ///
    /// ```
    /// use regex_automata::dfa::regex::Regex;
    ///
    /// let sparse_re = Regex::builder().build_sparse(r"foo[0-9]+")?;
    /// // A regex that uses sparse DFAs can be used just like with dense DFAs.
    /// assert!(sparse_re.is_match(b"foo123"));
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Fallibility
    ///
    /// In non-default configurations, the DFAs generated in this module may
    /// return an error during a search. (Currently, the only way this happens
    /// is if quit bytes are added or Unicode word boundaries are heuristically
    /// enabled, both of which are turned off by default.) For convenience,
    /// the main search routines, like [`find`](Regex::find), will panic if
    /// an error occurs. However, if you need to use DFAs which may produce
    /// an error at search time, then there are fallible equivalents of all
    /// search routines. For example, for `find`, its fallible analog is
    /// [`try_find`](Regex::try_find). The routines prefixed with `try_` return
    /// `Result<Option<Match>, MatchError>`, where as the infallible routines
    /// simply return `Option<Match>`.
    ///
    /// # Example
    ///
    /// This example shows how to cause a search to terminate if it sees a
    /// `\n` byte, and handle the error returned. This could be useful if, for
    /// example, you wanted to prevent a user supplied pattern from matching
    /// across a line boundary.
    ///
    /// ```
    /// # if cfg!(miri) { return Ok(()); } // miri takes too long
    /// use regex_automata::{dfa::{self, regex::Regex}, MatchError};
    ///
    /// let re = Regex::builder()
    ///     .dense(dfa::dense::Config::new().quit(b'\n', true))
    ///     .build(r"foo\p{any}+bar")?;
    ///
    /// let haystack = "foo\nbar".as_bytes();
    /// // Normally this would produce a match, since \p{any} contains '\n'.
    /// // But since we instructed the automaton to enter a quit state if a
    /// // '\n' is observed, this produces a match error instead.
    /// let expected = MatchError::quit(b'\n', 3);
    /// let got = re.try_find(haystack).unwrap_err();
    /// assert_eq!(expected, got);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[derive(Clone, Debug)]
);

#[cfg(all(feature = "syntax", feature = "dfa-build"))]
impl Regex {
    /// Parse the given regular expression using the default configuration and
    /// return the corresponding regex.
    ///
    /// If you want a non-default configuration, then use the [`Builder`] to
    /// set your own configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{Match, dfa::regex::Regex};
    ///
    /// let re = Regex::new("foo[0-9]+bar")?;
    /// assert_eq!(
    ///     Some(Match::must(0, 3..14)),
    ///     re.find(b"zzzfoo12345barzzz"),
    /// );
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(pattern: &str) -> Result<Regex, BuildError> {
        Builder::new().build(pattern)
    }

    /// Like `new`, but parses multiple patterns into a single "regex set."
    /// This similarly uses the default regex configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{Match, dfa::regex::Regex};
    ///
    /// let re = Regex::new_many(&["[a-z]+", "[0-9]+"])?;
    ///
    /// let mut it = re.find_iter(b"abc 1 foo 4567 0 quux");
    /// assert_eq!(Some(Match::must(0, 0..3)), it.next());
    /// assert_eq!(Some(Match::must(1, 4..5)), it.next());
    /// assert_eq!(Some(Match::must(0, 6..9)), it.next());
    /// assert_eq!(Some(Match::must(1, 10..14)), it.next());
    /// assert_eq!(Some(Match::must(1, 15..16)), it.next());
    /// assert_eq!(Some(Match::must(0, 17..21)), it.next());
    /// assert_eq!(None, it.next());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new_many<P: AsRef<str>>(
        patterns: &[P],
    ) -> Result<Regex, BuildError> {
        Builder::new().build_many(patterns)
    }
}

#[cfg(all(feature = "syntax", feature = "dfa-build"))]
impl Regex<sparse::DFA<Vec<u8>>> {
    /// Parse the given regular expression using the default configuration,
    /// except using sparse DFAs, and return the corresponding regex.
    ///
    /// If you want a non-default configuration, then use the [`Builder`] to
    /// set your own configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{Match, dfa::regex::Regex};
    ///
    /// let re = Regex::new_sparse("foo[0-9]+bar")?;
    /// assert_eq!(
    ///     Some(Match::must(0, 3..14)),
    ///     re.find(b"zzzfoo12345barzzz"),
    /// );
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new_sparse(
        pattern: &str,
    ) -> Result<Regex<sparse::DFA<Vec<u8>>>, BuildError> {
        Builder::new().build_sparse(pattern)
    }

    /// Like `new`, but parses multiple patterns into a single "regex set"
    /// using sparse DFAs. This otherwise similarly uses the default regex
    /// configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{Match, dfa::regex::Regex};
    ///
    /// let re = Regex::new_many_sparse(&["[a-z]+", "[0-9]+"])?;
    ///
    /// let mut it = re.find_iter(b"abc 1 foo 4567 0 quux");
    /// assert_eq!(Some(Match::must(0, 0..3)), it.next());
    /// assert_eq!(Some(Match::must(1, 4..5)), it.next());
    /// assert_eq!(Some(Match::must(0, 6..9)), it.next());
    /// assert_eq!(Some(Match::must(1, 10..14)), it.next());
    /// assert_eq!(Some(Match::must(1, 15..16)), it.next());
    /// assert_eq!(Some(Match::must(0, 17..21)), it.next());
    /// assert_eq!(None, it.next());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new_many_sparse<P: AsRef<str>>(
        patterns: &[P],
    ) -> Result<Regex<sparse::DFA<Vec<u8>>>, BuildError> {
        Builder::new().build_many_sparse(patterns)
    }
}

/// Convenience routines for regex construction.
impl Regex<dense::DFA<&'static [u32]>> {
    /// Return a builder for configuring the construction of a `Regex`.
    ///
    /// This is a convenience routine to avoid needing to import the
    /// [`Builder`] type in common cases.
    ///
    /// # Example
    ///
    /// This example shows how to use the builder to disable UTF-8 mode
    /// everywhere.
    ///
    /// ```
    /// # if cfg!(miri) { return Ok(()); } // miri takes too long
    /// use regex_automata::{
    ///     dfa::regex::Regex, nfa::thompson, util::syntax, Match,
    /// };
    ///
    /// let re = Regex::builder()
    ///     .syntax(syntax::Config::new().utf8(false))
    ///     .thompson(thompson::Config::new().utf8(false))
    ///     .build(r"foo(?-u:[^b])ar.*")?;
    /// let haystack = b"\xFEfoo\xFFarzz\xE2\x98\xFF\n";
    /// let expected = Some(Match::must(0, 1..9));
    /// let got = re.find(haystack);
    /// assert_eq!(expected, got);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn builder() -> Builder {
        Builder::new()
    }
}

/// Standard search routines for finding and iterating over matches.
impl<A: Automaton> Regex<A> {
    /// Returns true if and only if this regex matches the given haystack.
    ///
    /// This routine may short circuit if it knows that scanning future input
    /// will never lead to a different result. In particular, if the underlying
    /// DFA enters a match state or a dead state, then this routine will return
    /// `true` or `false`, respectively, without inspecting any future input.
    ///
    /// # Panics
    ///
    /// If the underlying DFAs return an error, then this routine panics. This
    /// only occurs in non-default configurations where quit bytes are used or
    /// Unicode word boundaries are heuristically enabled.
    ///
    /// The fallible version of this routine is
    /// [`try_is_match`](Regex::try_is_match).
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::dfa::regex::Regex;
    ///
    /// let re = Regex::new("foo[0-9]+bar")?;
    /// assert_eq!(true, re.is_match(b"foo12345bar"));
    /// assert_eq!(false, re.is_match(b"foobar"));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn is_match<H: AsRef<[u8]>>(&self, haystack: H) -> bool {
        self.try_is_match(haystack.as_ref()).unwrap()
    }

    /// Returns the start and end offset of the leftmost match. If no match
    /// exists, then `None` is returned.
    ///
    /// # Panics
    ///
    /// If the underlying DFAs return an error, then this routine panics. This
    /// only occurs in non-default configurations where quit bytes are used or
    /// Unicode word boundaries are heuristically enabled.
    ///
    /// The fallible version of this routine is [`try_find`](Regex::try_find).
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{Match, dfa::regex::Regex};
    ///
    /// // Greediness is applied appropriately.
    /// let re = Regex::new("foo[0-9]+")?;
    /// assert_eq!(
    ///     Some(Match::must(0, 3..11)),
    ///     re.find(b"zzzfoo12345zzz"),
    /// );
    ///
    /// // Even though a match is found after reading the first byte (`a`),
    /// // the default leftmost-first match semantics demand that we find the
    /// // earliest match that prefers earlier parts of the pattern over latter
    /// // parts.
    /// let re = Regex::new("abc|a")?;
    /// assert_eq!(Some(Match::must(0, 0..3)), re.find(b"abc"));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn find<H: AsRef<[u8]>>(&self, haystack: H) -> Option<Match> {
        self.try_find(haystack.as_ref()).unwrap()
    }

    /// Returns an iterator over all non-overlapping leftmost matches in the
    /// given bytes. If no match exists, then the iterator yields no elements.
    ///
    /// This corresponds to the "standard" regex search iterator.
    ///
    /// # Panics
    ///
    /// If the underlying DFAs return an error during iteration, then iteration
    /// panics. This only occurs in non-default configurations where quit bytes
    /// are used or Unicode word boundaries are heuristically enabled.
    ///
    /// The fallible version of this routine is
    /// [`try_find_iter`](Regex::try_find_iter).
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{Match, dfa::regex::Regex};
    ///
    /// let re = Regex::new("foo[0-9]+")?;
    /// let text = b"foo1 foo12 foo123";
    /// let matches: Vec<Match> = re.find_iter(text).collect();
    /// assert_eq!(matches, vec![
    ///     Match::must(0, 0..4),
    ///     Match::must(0, 5..10),
    ///     Match::must(0, 11..17),
    /// ]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn find_iter<'r, 'h, H: AsRef<[u8]> + ?Sized>(
        &'r self,
        haystack: &'h H,
    ) -> FindMatches<'r, 'h, A> {
        let input = self.create_input(haystack.as_ref());
        let it = iter::Searcher::new(input);
        FindMatches { re: self, it }
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
impl<A: Automaton> Regex<A> {
    /// Returns true if and only if this regex matches the given haystack.
    ///
    /// This routine may short circuit if it knows that scanning future input
    /// will never lead to a different result. In particular, if the underlying
    /// DFA enters a match state or a dead state, then this routine will return
    /// `true` or `false`, respectively, without inspecting any future input.
    ///
    /// # Errors
    ///
    /// This routine only errors if the search could not complete. For
    /// DFA-based regexes, this only occurs in a non-default configuration
    /// where quit bytes are used or Unicode word boundaries are heuristically
    /// enabled.
    ///
    /// When a search cannot complete, callers cannot know whether a match
    /// exists or not.
    ///
    /// The infallible (panics on error) version of this routine is
    /// [`is_match`](Regex::is_match).
    #[inline]
    pub fn try_is_match<H: AsRef<[u8]>>(
        &self,
        haystack: H,
    ) -> Result<bool, MatchError> {
        // Not only can we do an "earliest" search, but we can avoid doing a
        // reverse scan too.
        let input = self.create_input(haystack.as_ref()).earliest(true);
        self.forward().try_search_fwd(&input).map(|x| x.is_some())
    }

    /// Returns the start and end offset of the leftmost match. If no match
    /// exists, then `None` is returned.
    ///
    /// # Errors
    ///
    /// This routine only errors if the search could not complete. For
    /// DFA-based regexes, this only occurs in a non-default configuration
    /// where quit bytes are used or Unicode word boundaries are heuristically
    /// enabled.
    ///
    /// When a search cannot complete, callers cannot know whether a match
    /// exists or not.
    ///
    /// The infallible (panics on error) version of this routine is
    /// [`find`](Regex::find).
    #[inline]
    pub fn try_find<H: AsRef<[u8]>>(
        &self,
        haystack: H,
    ) -> Result<Option<Match>, MatchError> {
        let input = self.create_input(haystack.as_ref());
        self.try_search(&input)
    }

    /// Returns an iterator over all non-overlapping leftmost matches in the
    /// given bytes. If no match exists, then the iterator yields no elements.
    ///
    /// This corresponds to the "standard" regex search iterator.
    ///
    /// # Errors
    ///
    /// This iterator only yields errors if the search could not complete. For
    /// DFA-based regexes, this only occurs in a non-default configuration
    /// where quit bytes are used or Unicode word boundaries are heuristically
    /// enabled.
    ///
    /// When a search cannot complete, callers cannot know whether a match
    /// exists or not.
    ///
    /// The infallible (panics on error) version of this routine is
    /// [`find_iter`](Regex::find_iter).
    #[inline]
    pub fn try_find_iter<'r, 'h, H: AsRef<[u8]> + ?Sized>(
        &'r self,
        haystack: &'h H,
    ) -> TryFindMatches<'r, 'h, A> {
        let input = self.create_input(haystack.as_ref());
        let it = iter::Searcher::new(input);
        TryFindMatches { re: self, it }
    }
}

/// Lower level fallible search routines that permit controlling where the
/// search starts and ends in a particular sequence.
impl<A: Automaton> Regex<A> {
    /// Returns the start and end offset of the leftmost match. If no match
    /// exists, then `None` is returned.
    ///
    /// # Searching a substring of the haystack
    ///
    /// Being an "at" search routine, this permits callers to search a
    /// substring of `haystack` by specifying a range in `haystack`.
    /// Why expose this as an API instead of just asking callers to use
    /// `&input[start..end]`? The reason is that regex matching often wants
    /// to take the surrounding context into account in order to handle
    /// look-around (`^`, `$` and `\b`).
    ///
    /// This is useful when implementing an iterator over matches
    /// within the same haystack, which cannot be done correctly by simply
    /// providing a subslice of `haystack`.
    ///
    /// # Errors
    ///
    /// This routine only errors if the search could not complete. For
    /// DFA-based regexes, this only occurs in a non-default configuration
    /// where quit bytes are used or Unicode word boundaries are heuristically
    /// enabled.
    ///
    /// When a search cannot complete, callers cannot know whether a match
    /// exists or not.
    #[inline]
    pub fn try_search(
        &self,
        input: &Input<'_, '_>,
    ) -> Result<Option<Match>, MatchError> {
        // N.B. We use `&&A` here to call `Automaton` methods, which ensures
        // that we always use the `impl Automaton for &A` for calling methods.
        // Since this is the usual way that automata are used, this helps
        // reduce the number of monomorphized copies of the search code.
        let (fwd, rev) = (self.forward(), self.reverse());
        let end = match (&fwd).try_search_fwd(input)? {
            None => return Ok(None),
            Some(end) => end,
        };
        // This special cases an empty match at the beginning of the search. If
        // our end matches our start, then since a reverse DFA can't match past
        // the start, it must follow that our starting position is also our end
        // position. So short circuit and skip the reverse search.
        if input.start() == end.offset() {
            return Ok(Some(Match::new(
                end.pattern(),
                end.offset()..end.offset(),
            )));
        }
        // We can also skip the reverse search if we know our search was
        // anchored. This occurs either when the input config is anchored or
        // when we know the regex itself is anchored. In this case, we know the
        // start of the match, if one is found, must be the start of the
        // search.
        if self.is_anchored(input) {
            return Ok(Some(Match::new(
                end.pattern(),
                input.start()..end.offset(),
            )));
        }
        // N.B. I have tentatively convinced myself that it isn't necessary
        // to specify the specific pattern for the reverse search since the
        // reverse search will always find the same pattern to match as the
        // forward search. But I lack a rigorous proof. Why not just provide
        // the pattern anyway? Well, if it is needed, then leaving it out
        // gives us a chance to find a witness. (Also, if we don't need to
        // specify the pattern, then we don't need to build the reverse DFA
        // with 'starts_for_each_pattern' enabled. It doesn't matter too much
        // for the lazy DFA, but does make the overall DFA bigger.)
        //
        // We also need to be careful to disable 'earliest' for the reverse
        // search, since it could be enabled for the forward search. In the
        // reverse case, to satisfy "leftmost" criteria, we need to match
        // as much as we can. We also need to be careful to make the search
        // anchored. We don't want the reverse search to report any matches
        // other than the one beginning at the end of our forward search.
        let revsearch = input
            .clone()
            .span(input.start()..end.offset())
            .anchored(Anchored::Yes)
            .earliest(false);
        let start = (&rev)
            .try_search_rev(&revsearch)?
            .expect("reverse search must match if forward search does");
        assert_eq!(
            start.pattern(),
            end.pattern(),
            "forward and reverse search must match same pattern",
        );
        assert!(start.offset() <= end.offset());
        Ok(Some(Match::new(end.pattern(), start.offset()..end.offset())))
    }

    /// Returns true if either the given input specifies an anchored search
    /// or if the underlying DFA is always anchored.
    fn is_anchored(&self, input: &Input<'_, '_>) -> bool {
        match input.get_anchored() {
            Anchored::No => self.forward().is_always_start_anchored(),
            Anchored::Yes | Anchored::Pattern(_) => true,
        }
    }
}

/// Non-search APIs for querying information about the regex and setting a
/// prefilter.
impl<A: Automaton> Regex<A> {
    /// Create a new `Input` for the given haystack.
    #[inline]
    pub fn create_input<'p, 'h, H: ?Sized + AsRef<[u8]>>(
        &'p self,
        haystack: &'h H,
    ) -> Input<'h, 'p> {
        Input::new(haystack)
    }

    /// Return the underlying DFA responsible for forward matching.
    ///
    /// This is useful for accessing the underlying DFA and converting it to
    /// some other format or size. See the [`Builder::build_from_dfas`] docs
    /// for an example of where this might be useful.
    pub fn forward(&self) -> &A {
        &self.forward
    }

    /// Return the underlying DFA responsible for reverse matching.
    ///
    /// This is useful for accessing the underlying DFA and converting it to
    /// some other format or size. See the [`Builder::build_from_dfas`] docs
    /// for an example of where this might be useful.
    pub fn reverse(&self) -> &A {
        &self.reverse
    }

    /// Returns the total number of patterns matched by this regex.
    ///
    /// # Example
    ///
    /// ```
    /// # if cfg!(miri) { return Ok(()); } // miri takes too long
    /// use regex_automata::dfa::regex::Regex;
    ///
    /// let re = Regex::new_many(&[r"[a-z]+", r"[0-9]+", r"\w+"])?;
    /// assert_eq!(3, re.pattern_len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn pattern_len(&self) -> usize {
        assert_eq!(self.forward().pattern_len(), self.reverse().pattern_len());
        self.forward().pattern_len()
    }
}

/// An iterator over all non-overlapping matches for an infallible search.
///
/// The iterator yields a [`Match`] value until no more matches could be found.
/// If the underlying regex engine returns an error, then a panic occurs.
///
/// The lifetime parameters are as follows:
///
/// * `'h` represents the lifetime of the haystack being searched.
/// * `'r` represents the lifetime of the regex object itself.
///
/// This iterator can be created with the [`Regex::find_iter`] method.
#[derive(Debug)]
pub struct FindMatches<'r, 'h, A> {
    re: &'r Regex<A>,
    it: iter::Searcher<'h, 'r>,
}

impl<'r, 'h, A: Automaton> Iterator for FindMatches<'r, 'h, A> {
    type Item = Match;

    #[inline]
    fn next(&mut self) -> Option<Match> {
        let FindMatches { re, ref mut it } = *self;
        it.advance(|input| re.try_search(input))
    }
}

/// An iterator over all non-overlapping matches for a fallible search.
///
/// The iterator yields a `Result<Match, MatchError>` value until no more
/// matches could be found.
///
/// The lifetime parameters are as follows:
///
/// * `'h` represents the lifetime of the haystack being searched.
/// * `'r` represents the lifetime of the regex object itself.
///
/// This iterator can be created with the [`Regex::try_find_iter`] method.
#[derive(Debug)]
pub struct TryFindMatches<'r, 'h, A> {
    re: &'r Regex<A>,
    it: iter::Searcher<'h, 'r>,
}

impl<'r, 'h, A: Automaton> Iterator for TryFindMatches<'r, 'h, A> {
    type Item = Result<Match, MatchError>;

    #[inline]
    fn next(&mut self) -> Option<Result<Match, MatchError>> {
        let TryFindMatches { re, ref mut it } = *self;
        it.try_advance(|input| re.try_search(input)).transpose()
    }
}

/// A builder for a regex based on deterministic finite automatons.
///
/// This builder permits configuring options for the syntax of a pattern, the
/// NFA construction, the DFA construction and finally the regex searching
/// itself. This builder is different from a general purpose regex builder in
/// that it permits fine grain configuration of the construction process. The
/// trade off for this is complexity, and the possibility of setting a
/// configuration that might not make sense. For example, there are two
/// different UTF-8 modes:
///
/// * [`syntax::Config::utf8`](crate::util::syntax::Config::utf8) controls
/// whether the pattern itself can contain sub-expressions that match invalid
/// UTF-8.
/// * [`thompson::Config::utf8`] controls how the regex iterators themselves
/// advance the starting position of the next search when a match with zero
/// length is found.
///
/// Generally speaking, callers will want to either enable all of these or
/// disable all of these.
///
/// Internally, building a regex requires building two DFAs, where one is
/// responsible for finding the end of a match and the other is responsible
/// for finding the start of a match. If you only need to detect whether
/// something matched, or only the end of a match, then you should use a
/// [`dense::Builder`] to construct a single DFA, which is cheaper than
/// building two DFAs.
///
/// # Build methods
///
/// This builder has a few "build" methods. In general, it's the result of
/// combining the following parameters:
///
/// * Building one or many regexes.
/// * Building a regex with dense or sparse DFAs.
///
/// The simplest "build" method is [`Builder::build`]. It accepts a single
/// pattern and builds a dense DFA using `usize` for the state identifier
/// representation.
///
/// The most general "build" method is [`Builder::build_many`], which permits
/// building a regex that searches for multiple patterns simultaneously while
/// using a specific state identifier representation.
///
/// The most flexible "build" method, but hardest to use, is
/// [`Builder::build_from_dfas`]. This exposes the fact that a [`Regex`] is
/// just a pair of DFAs, and this method allows you to specify those DFAs
/// exactly.
///
/// # Example
///
/// This example shows how to disable UTF-8 mode in the syntax and the regex
/// itself. This is generally what you want for matching on arbitrary bytes.
///
/// ```
/// # if cfg!(miri) { return Ok(()); } // miri takes too long
/// use regex_automata::{
///     dfa::regex::Regex, nfa::thompson, util::syntax, Match,
/// };
///
/// let re = Regex::builder()
///     .syntax(syntax::Config::new().utf8(false))
///     .thompson(thompson::Config::new().utf8(false))
///     .build(r"foo(?-u:[^b])ar.*")?;
/// let haystack = b"\xFEfoo\xFFarzz\xE2\x98\xFF\n";
/// let expected = Some(Match::must(0, 1..9));
/// let got = re.find(haystack);
/// assert_eq!(expected, got);
/// // Notice that `(?-u:[^b])` matches invalid UTF-8,
/// // but the subsequent `.*` does not! Disabling UTF-8
/// // on the syntax permits this. Notice also that the
/// // search was unanchored and skipped over invalid UTF-8.
/// // Disabling UTF-8 on the Thompson NFA permits this.
/// //
/// // N.B. This example does not show the impact of
/// // disabling UTF-8 mode on Config, since that
/// // only impacts regexes that can produce matches of
/// // length 0.
/// assert_eq!(b"foo\xFFarzz", &haystack[got.unwrap().range()]);
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug)]
pub struct Builder {
    #[cfg(feature = "dfa-build")]
    dfa: dense::Builder,
}

impl Builder {
    /// Create a new regex builder with the default configuration.
    pub fn new() -> Builder {
        Builder {
            #[cfg(feature = "dfa-build")]
            dfa: dense::Builder::new(),
        }
    }

    /// Build a regex from the given pattern.
    ///
    /// If there was a problem parsing or compiling the pattern, then an error
    /// is returned.
    #[cfg(all(feature = "syntax", feature = "dfa-build"))]
    pub fn build(&self, pattern: &str) -> Result<Regex, BuildError> {
        self.build_many(&[pattern])
    }

    /// Build a regex from the given pattern using sparse DFAs.
    ///
    /// If there was a problem parsing or compiling the pattern, then an error
    /// is returned.
    #[cfg(all(feature = "syntax", feature = "dfa-build"))]
    pub fn build_sparse(
        &self,
        pattern: &str,
    ) -> Result<Regex<sparse::DFA<Vec<u8>>>, BuildError> {
        self.build_many_sparse(&[pattern])
    }

    /// Build a regex from the given patterns.
    #[cfg(all(feature = "syntax", feature = "dfa-build"))]
    pub fn build_many<P: AsRef<str>>(
        &self,
        patterns: &[P],
    ) -> Result<Regex, BuildError> {
        let forward = self.dfa.build_many(patterns)?;
        let reverse = self
            .dfa
            .clone()
            .configure(
                dense::Config::new()
                    .prefilter(None)
                    .start_kind(StartKind::Anchored)
                    .match_kind(MatchKind::All),
            )
            .thompson(thompson::Config::new().reverse(true))
            .build_many(patterns)?;
        Ok(self.build_from_dfas(forward, reverse))
    }

    /// Build a sparse regex from the given patterns.
    #[cfg(all(feature = "syntax", feature = "dfa-build"))]
    pub fn build_many_sparse<P: AsRef<str>>(
        &self,
        patterns: &[P],
    ) -> Result<Regex<sparse::DFA<Vec<u8>>>, BuildError> {
        let re = self.build_many(patterns)?;
        let forward = re.forward().to_sparse()?;
        let reverse = re.reverse().to_sparse()?;
        Ok(self.build_from_dfas(forward, reverse))
    }

    /// Build a regex from its component forward and reverse DFAs.
    ///
    /// This is useful when deserializing a regex from some arbitrary
    /// memory region. This is also useful for building regexes from other
    /// types of DFAs.
    ///
    /// If you're building the DFAs from scratch instead of building new DFAs
    /// from other DFAs, then you'll need to make sure that the reverse DFA is
    /// configured correctly to match the intended semantics. Namely:
    ///
    /// * It should be anchored.
    /// * It should use [`MatchKind::All`] semantics.
    /// * It should match in reverse.
    /// * It should have anchored start states compiled for each pattern.
    /// * Otherwise, its configuration should match the forward DFA.
    ///
    /// If these conditions aren't satisfied, then the behavior of searches is
    /// unspecified.
    ///
    /// Note that when using this constructor, no configuration is applied.
    /// Since this routine provides the DFAs to the builder, there is no
    /// opportunity to apply other configuration options.
    ///
    /// # Example
    ///
    /// This example is a bit a contrived. The usual use of these methods
    /// would involve serializing `initial_re` somewhere and then deserializing
    /// it later to build a regex. But in this case, we do everything in
    /// memory.
    ///
    /// ```
    /// use regex_automata::dfa::regex::Regex;
    ///
    /// let initial_re = Regex::new("foo[0-9]+")?;
    /// assert_eq!(true, initial_re.is_match(b"foo123"));
    ///
    /// let (fwd, rev) = (initial_re.forward(), initial_re.reverse());
    /// let re = Regex::builder().build_from_dfas(fwd, rev);
    /// assert_eq!(true, re.is_match(b"foo123"));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// This example shows how to build a `Regex` that uses sparse DFAs instead
    /// of dense DFAs without using one of the convenience `build_sparse`
    /// routines:
    ///
    /// ```
    /// use regex_automata::dfa::regex::Regex;
    ///
    /// let initial_re = Regex::new("foo[0-9]+")?;
    /// assert_eq!(true, initial_re.is_match(b"foo123"));
    ///
    /// let fwd = initial_re.forward().to_sparse()?;
    /// let rev = initial_re.reverse().to_sparse()?;
    /// let re = Regex::builder().build_from_dfas(fwd, rev);
    /// assert_eq!(true, re.is_match(b"foo123"));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn build_from_dfas<A: Automaton>(
        &self,
        forward: A,
        reverse: A,
    ) -> Regex<A> {
        Regex { forward, reverse }
    }

    /// Set the syntax configuration for this builder using
    /// [`syntax::Config`](crate::util::syntax::Config).
    ///
    /// This permits setting things like case insensitivity, Unicode and multi
    /// line mode.
    #[cfg(all(feature = "syntax", feature = "dfa-build"))]
    pub fn syntax(
        &mut self,
        config: crate::util::syntax::Config,
    ) -> &mut Builder {
        self.dfa.syntax(config);
        self
    }

    /// Set the Thompson NFA configuration for this builder using
    /// [`nfa::thompson::Config`](thompson::Config).
    ///
    /// This permits setting things like whether additional time should be
    /// spent shrinking the size of the NFA.
    #[cfg(all(feature = "syntax", feature = "dfa-build"))]
    pub fn thompson(&mut self, config: thompson::Config) -> &mut Builder {
        self.dfa.thompson(config);
        self
    }

    /// Set the dense DFA compilation configuration for this builder using
    /// [`dense::Config`](dense::Config).
    ///
    /// This permits setting things like whether the underlying DFAs should
    /// be minimized.
    #[cfg(all(feature = "syntax", feature = "dfa-build"))]
    pub fn dense(&mut self, config: dense::Config) -> &mut Builder {
        self.dfa.configure(config);
        self
    }
}

impl Default for Builder {
    fn default() -> Builder {
        Builder::new()
    }
}
