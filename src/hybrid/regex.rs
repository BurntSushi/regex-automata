/*!
A lazy DFA backed `Regex`.

This module provides [`Regex`] using lazy DFA. A `Regex` implements convenience
routines you might have come to expect, such as finding a match and iterating
over all non-overlapping matches. This `Regex` type is limited in its
capabilities to what a lazy DFA can provide. Therefore, APIs involving
capturing groups, for example, are not provided.

Internally, a `Regex` is composed of two DFAs. One is a "forward" DFA that
finds the end offset of a match, where as the other is a "reverse" DFA that
find the start offset of a match.

See the [parent module](crate::hybrid) for examples.
*/

use core::borrow::Borrow;

use alloc::{boxed::Box, sync::Arc};

use crate::{
    hybrid::{
        dfa::{self, DFA},
        error::BuildError,
        search,
    },
    nfa::thompson,
    util::{
        iter,
        prefilter::{self, Prefilter},
        search::{Match, MatchError, MatchKind, Search, Span},
    },
};

/// A regular expression that uses hybrid NFA/DFAs (also called "lazy DFAs")
/// for searching.
///
/// A regular expression is comprised of two lazy DFAs, a "forward" DFA and a
/// "reverse" DFA. The forward DFA is responsible for detecting the end of
/// a match while the reverse DFA is responsible for detecting the start
/// of a match. Thus, in order to find the bounds of any given match, a
/// forward search must first be run followed by a reverse search. A match
/// found by the forward DFA guarantees that the reverse DFA will also find
/// a match.
///
/// A `Regex` can also have a prefilter set via the
/// [`set_prefilter`](Regex::set_prefilter) method. By default, no prefilter is
/// enabled.
///
/// # Fallibility
///
/// In non-default configurations, the lazy DFAs generated in this module may
/// return an error during a search. (Currently, the only way this happens
/// is if quit bytes are added, Unicode word boundaries are heuristically
/// enabled, or if the cache is configured to "give up" on a search if it
/// has been cleared too many times. All of these are turned off by default,
/// which means a search can never fail in the default configuration.) For
/// convenience, the main search routines, like [`find`](Regex::find), will
/// panic if an error occurs. However, if you need to use DFAs which may
/// produce an error at search time, then there are fallible equivalents of
/// all search routines. For example, for `find`, its fallible analog is
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
/// use regex_automata::{hybrid::{dfa, regex::Regex}, MatchError};
///
/// let re = Regex::builder()
///     .dfa(dfa::Config::new().quit(b'\n', true))
///     .build(r"foo\p{any}+bar")?;
/// let mut cache = re.create_cache();
///
/// let haystack = "foo\nbar";
/// // Normally this would produce a match, since \p{any} contains '\n'.
/// // But since we instructed the automaton to enter a quit state if a
/// // '\n' is observed, this produces a match error instead.
/// let expected = MatchError::Quit { byte: 0x0A, offset: 3 };
/// let got = re.try_find(&mut cache, haystack).unwrap_err();
/// assert_eq!(expected, got);
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct Regex {
    /// The config for this regex.
    config: Config,
    /// The forward lazy DFA. This can only find the end of a match.
    forward: DFA,
    /// The reverse lazy DFA. This can only find the start of a match.
    ///
    /// This is built with 'all' match semantics (instead of leftmost-first)
    /// so that it always finds the longest possible match (which corresponds
    /// to the leftmost starting position). It is also compiled as an anchored
    /// matcher and has 'starts_for_each_pattern' enabled. Including starting
    /// states for each pattern is necessary to ensure that we only look for
    /// matches of a pattern that matched in the forward direction. Otherwise,
    /// we might wind up finding the "leftmost" starting position of a totally
    /// different pattern!
    reverse: DFA,
}

/// Convenience routines for regex and cache construction.
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
    /// use regex_automata::{Match, hybrid::regex::Regex};
    ///
    /// let re = Regex::new("foo[0-9]+bar")?;
    /// let mut cache = re.create_cache();
    /// assert_eq!(
    ///     Some(Match::must(0, 3..14)),
    ///     re.find(&mut cache, "zzzfoo12345barzzz"),
    /// );
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(pattern: &str) -> Result<Regex, BuildError> {
        Regex::builder().build(pattern)
    }

    /// Like `new`, but parses multiple patterns into a single "regex set."
    /// This similarly uses the default regex configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{Match, hybrid::regex::Regex};
    ///
    /// let re = Regex::new_many(&["[a-z]+", "[0-9]+"])?;
    /// let mut cache = re.create_cache();
    ///
    /// let mut it = re.find_iter(&mut cache, "abc 1 foo 4567 0 quux");
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
        Regex::builder().build_many(patterns)
    }

    /// Return a default configuration for a `Regex`.
    ///
    /// This is a convenience routine to avoid needing to import the `Config`
    /// type when customizing the construction of a regex.
    ///
    /// # Example
    ///
    /// This example shows how to disable UTF-8 mode for `Regex` iteration.
    /// When UTF-8 mode is disabled, the position immediately following an
    /// empty match is where the next search begins, instead of the next
    /// position of a UTF-8 encoded codepoint.
    ///
    /// ```
    /// use regex_automata::{hybrid::regex::Regex, Match};
    ///
    /// let re = Regex::builder()
    ///     .configure(Regex::config().utf8(false))
    ///     .build(r"")?;
    /// let mut cache = re.create_cache();
    ///
    /// let haystack = "a☃z";
    /// let mut it = re.find_iter(&mut cache, haystack);
    /// assert_eq!(Some(Match::must(0, 0..0)), it.next());
    /// assert_eq!(Some(Match::must(0, 1..1)), it.next());
    /// assert_eq!(Some(Match::must(0, 2..2)), it.next());
    /// assert_eq!(Some(Match::must(0, 3..3)), it.next());
    /// assert_eq!(Some(Match::must(0, 4..4)), it.next());
    /// assert_eq!(Some(Match::must(0, 5..5)), it.next());
    /// assert_eq!(None, it.next());
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
    /// everywhere.
    ///
    /// ```
    /// use regex_automata::{
    ///     hybrid::regex::Regex,
    ///     nfa::thompson,
    ///     Match, SyntaxConfig,
    /// };
    ///
    /// let re = Regex::builder()
    ///     .configure(Regex::config().utf8(false))
    ///     .syntax(SyntaxConfig::new().utf8(false))
    ///     .build(r"foo(?-u:[^b])ar.*")?;
    /// let mut cache = re.create_cache();
    ///
    /// let haystack = b"\xFEfoo\xFFarzz\xE2\x98\xFF\n";
    /// let expected = Some(Match::must(0, 1..9));
    /// let got = re.find(&mut cache, haystack);
    /// assert_eq!(expected, got);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn builder() -> Builder {
        Builder::new()
    }

    /// Create a new `Search` for the given haystack.
    ///
    /// The `Search` returned is configured to match the configuration of this
    /// `Regex`. For example, if this `Regex` was built with [`Config::utf8`]
    /// enabled, then the `Search` returned will also have its [`Search::utf8`]
    /// knob enabled.
    ///
    /// This routine is useful when using the lower-level [`Regex::try_search`]
    /// API.
    #[inline]
    pub fn create_search<'h, H: ?Sized + AsRef<[u8]>>(
        &self,
        haystack: &'h H,
    ) -> Search<'h> {
        Search::new(haystack).utf8(self.get_config().get_utf8())
    }

    /// Create a new cache for this `Regex`.
    ///
    /// The cache returned should only be used for searches for this
    /// `Regex`. If you want to reuse the cache for another `Regex`, then
    /// you must call [`Cache::reset`] with that `Regex` (or, equivalently,
    /// [`Regex::reset_cache`]).
    pub fn create_cache(&self) -> Cache {
        Cache::new(self)
    }

    /// Reset the given cache such that it can be used for searching with the
    /// this `Regex` (and only this `Regex`).
    ///
    /// A cache reset permits reusing memory already allocated in this cache
    /// with a different `Regex`.
    ///
    /// Resetting a cache sets its "clear count" to 0. This is relevant if the
    /// `Regex` has been configured to "give up" after it has cleared the cache
    /// a certain number of times.
    ///
    /// # Example
    ///
    /// This shows how to re-purpose a cache for use with a different `Regex`.
    ///
    /// ```
    /// use regex_automata::{hybrid::regex::Regex, Match};
    ///
    /// let re1 = Regex::new(r"\w")?;
    /// let re2 = Regex::new(r"\W")?;
    ///
    /// let mut cache = re1.create_cache();
    /// assert_eq!(
    ///     Some(Match::must(0, 0..2)),
    ///     re1.find(&mut cache, "Δ"),
    /// );
    ///
    /// // Using 'cache' with re2 is not allowed. It may result in panics or
    /// // incorrect results. In order to re-purpose the cache, we must reset
    /// // it with the Regex we'd like to use it with.
    /// //
    /// // Similarly, after this reset, using the cache with 're1' is also not
    /// // allowed.
    /// re2.reset_cache(&mut cache);
    /// assert_eq!(
    ///     Some(Match::must(0, 0..3)),
    ///     re2.find(&mut cache, "☃"),
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reset_cache(&self, cache: &mut Cache) {
        self.forward().reset_cache(&mut cache.forward);
        self.reverse().reset_cache(&mut cache.reverse);
    }
}

/// Standard infallible search routines for finding and iterating over matches.
impl Regex {
    /// Returns true if and only if this regex matches the given haystack.
    ///
    /// This routine may short circuit if it knows that scanning future input
    /// will never lead to a different result. In particular, if the underlying
    /// DFA enters a match state or a dead state, then this routine will return
    /// `true` or `false`, respectively, without inspecting any future input.
    ///
    /// # Panics
    ///
    /// If the underlying lazy DFAs return an error, then this routine panics.
    /// This only occurs in non-default configurations where quit bytes are
    /// used, Unicode word boundaries are heuristically enabled or limits are
    /// set on the number of times the lazy DFA's cache may be cleared.
    ///
    /// The fallible version of this routine is
    /// [`try_is_match`](Regex::try_is_match).
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::hybrid::regex::Regex;
    ///
    /// let re = Regex::new("foo[0-9]+bar")?;
    /// let mut cache = re.create_cache();
    ///
    /// assert_eq!(true, re.is_match(&mut cache, "foo12345bar"));
    /// assert_eq!(false, re.is_match(&mut cache, "foobar"));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn is_match<H: AsRef<[u8]>>(
        &self,
        cache: &mut Cache,
        haystack: H,
    ) -> bool {
        self.try_is_match(cache, haystack.as_ref()).unwrap()
    }

    /// Returns the start and end offset of the leftmost match. If no match
    /// exists, then `None` is returned.
    ///
    /// # Panics
    ///
    /// If the underlying lazy DFAs return an error, then this routine panics.
    /// This only occurs in non-default configurations where quit bytes are
    /// used, Unicode word boundaries are heuristically enabled or limits are
    /// set on the number of times the lazy DFA's cache may be cleared.
    ///
    /// The fallible version of this routine is [`try_find`](Regex::try_find).
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{Match, hybrid::regex::Regex};
    ///
    /// let re = Regex::new("foo[0-9]+")?;
    /// let mut cache = re.create_cache();
    /// assert_eq!(
    ///     Some(Match::must(0, 3..11)),
    ///     re.find(&mut cache, "zzzfoo12345zzz"),
    /// );
    ///
    /// // Even though a match is found after reading the first byte (`a`),
    /// // the default leftmost-first match semantics demand that we find the
    /// // earliest match that prefers earlier parts of the pattern over latter
    /// // parts.
    /// let re = Regex::new("abc|a")?;
    /// let mut cache = re.create_cache();
    /// assert_eq!(Some(Match::must(0, 0..3)), re.find(&mut cache, "abc"));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn find<H: AsRef<[u8]>>(
        &self,
        cache: &mut Cache,
        haystack: H,
    ) -> Option<Match> {
        self.try_find(cache, haystack.as_ref()).unwrap()
    }

    /// Returns an iterator over all non-overlapping leftmost matches in the
    /// given bytes. If no match exists, then the iterator yields no elements.
    ///
    /// This corresponds to the "standard" regex search iterator.
    ///
    /// # Panics
    ///
    /// If the underlying lazy DFAs return an error, then this routine panics.
    /// This only occurs in non-default configurations where quit bytes are
    /// used, Unicode word boundaries are heuristically enabled or limits are
    /// set on the number of times the lazy DFA's cache may be cleared.
    ///
    /// The fallible version of this routine is
    /// [`try_find_iter`](Regex::try_find_iter).
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{Match, hybrid::regex::Regex};
    ///
    /// let re = Regex::new("foo[0-9]+")?;
    /// let mut cache = re.create_cache();
    ///
    /// let text = "foo1 foo12 foo123";
    /// let matches: Vec<Match> = re.find_iter(&mut cache, text).collect();
    /// assert_eq!(matches, vec![
    ///     Match::must(0, 0..4),
    ///     Match::must(0, 5..10),
    ///     Match::must(0, 11..17),
    /// ]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn find_iter<'r: 'c, 'c, 'h, H: AsRef<[u8]> + ?Sized>(
        &'r self,
        cache: &'c mut Cache,
        haystack: &'h H,
    ) -> FindMatches<'h, 'c> {
        let search = self.create_search(haystack.as_ref());
        FindMatches(self.try_matches_iter(cache, search).infallible())
    }
}

/// Fallible search routines. These may return an error when the underlying
/// lazy DFAs have been configured in a way that permits them to fail during a
/// search.
///
/// Errors during search only occur when the lazy DFA has been explicitly
/// configured to do so, usually by specifying one or more "quit" bytes or by
/// heuristically enabling Unicode word boundaries.
///
/// Errors will never be returned using the default configuration. So these
/// fallible routines are only needed for particular configurations.
impl Regex {
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
    /// where quit bytes are used, Unicode word boundaries are heuristically
    /// enabled or limits are set on the number of times the lazy DFA's cache
    /// may be cleared.
    ///
    /// When a search cannot complete, callers cannot know whether a match
    /// exists or not.
    ///
    /// The infallible (panics on error) version of this routine is
    /// [`is_match`](Regex::is_match).
    #[inline]
    pub fn try_is_match<H: AsRef<[u8]>>(
        &self,
        cache: &mut Cache,
        haystack: H,
    ) -> Result<bool, MatchError> {
        // Not only can we do an "earliest" search, but we can avoid doing a
        // reverse scan too.
        let search = self.create_search(haystack.as_ref()).earliest(true);
        let dfa = self.forward();
        let cache = &mut cache.forward;
        search::find_fwd(dfa, cache, self.scanner().as_mut(), &search)
            .map(|m| m.is_some())
    }

    /// Returns the start and end offset of the leftmost match. If no match
    /// exists, then `None` is returned.
    ///
    /// # Errors
    ///
    /// This routine only errors if the search could not complete. For
    /// DFA-based regexes, this only occurs in a non-default configuration
    /// where quit bytes are used, Unicode word boundaries are heuristically
    /// enabled or limits are set on the number of times the lazy DFA's cache
    /// may be cleared.
    ///
    /// When a search cannot complete, callers cannot know whether a match
    /// exists or not.
    ///
    /// The infallible (panics on error) version of this routine is
    /// [`find`](Regex::find).
    #[inline]
    pub fn try_find<H: AsRef<[u8]>>(
        &self,
        cache: &mut Cache,
        haystack: H,
    ) -> Result<Option<Match>, MatchError> {
        let search = self.create_search(haystack.as_ref());
        self.try_search(cache, self.scanner().as_mut(), &search)
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
    /// where quit bytes are used, Unicode word boundaries are heuristically
    /// enabled or limits are set on the number of times the lazy DFA's cache
    /// may be cleared.
    ///
    /// When a search cannot complete, callers cannot know whether a match
    /// exists or not.
    ///
    /// The infallible (panics on error) version of this routine is
    /// [`find_iter`](Regex::find_iter).
    #[inline]
    pub fn try_find_iter<'r: 'c, 'c, 'h, H: AsRef<[u8]> + ?Sized>(
        &'r self,
        cache: &'c mut Cache,
        haystack: &'h H,
    ) -> TryFindMatches<'h, 'c> {
        let search = self.create_search(haystack.as_ref());
        TryFindMatches(self.try_matches_iter(cache, search))
    }
}

/// Lower level "search" primitives that permit complete control over the
/// regex search.
impl Regex {
    /// Returns the start and end offset of the leftmost match. If no match
    /// exists, then `None` is returned.
    ///
    /// # Searching a substring of the haystack
    ///
    /// Being a "search" routine, this permits callers to search a substring
    /// of `haystack` by specifying a range in `haystack`. Why expose this as
    /// an API instead of just asking callers to use `&slice[start..end]`?
    /// The reason is that regex matching often wants to take the surrounding
    /// context into account in order to handle look-around (`^`, `$` and
    /// `\b`).
    ///
    /// This is useful when implementing an iterator over matches
    /// within the same haystack, which cannot be done correctly by simply
    /// providing a subslice of `haystack`.
    ///
    /// # Prefilter
    ///
    /// Unlike the "find" routines on `Regex`, this is a lower level search
    /// primitive that permits callers to pass a prefilter explicitly. Since
    /// this routine asks for an explicit prefilter, any prefilter that is
    /// attached to this `Regex` is ignored. To use the prefilter on this
    /// regex for a "search" routine, use the [`Regex::scanner`] method.
    ///
    /// # Errors
    ///
    /// This routine only errors if the search could not complete. For
    /// DFA-based regexes, this only occurs in a non-default configuration
    /// where quit bytes are used, Unicode word boundaries are heuristically
    /// enabled or limits are set on the number of times the lazy DFA's cache
    /// may be cleared.
    ///
    /// When a search cannot complete, callers cannot know whether a match
    /// exists or not.
    #[inline]
    pub fn try_search(
        &self,
        cache: &mut Cache,
        pre: Option<&mut prefilter::Scanner<'_>>,
        search: &Search<'_>,
    ) -> Result<Option<Match>, MatchError> {
        let m = match self.try_search_fwd_back(cache, pre, search)? {
            None => return Ok(None),
            Some(m) => m,
        };
        if m.is_empty() {
            search.skip_empty_utf8_splits(m, |search| {
                self.try_search_fwd_back(cache, None, search)
            })
        } else {
            Ok(Some(m))
        }
    }

    /// This search routine runs the regex engine forwards to find the end
    /// of a match, and then backwards to find the start of the match.
    #[inline(always)]
    fn try_search_fwd_back(
        &self,
        cache: &mut Cache,
        pre: Option<&mut prefilter::Scanner<'_>>,
        search: &Search<'_>,
    ) -> Result<Option<Match>, MatchError> {
        // N.B. We don't use the DFA::try_search_{fwd,rev} methods because they
        // appear to have a bit more latency due to the 'search.as_ref()' call.
        // So we reach around them. This also avoids generics.
        let (fdfa, rdfa) = (self.forward(), self.reverse());
        let (fcache, rcache) = (&mut cache.forward, &mut cache.reverse);
        let end = match search::find_fwd(fdfa, fcache, pre, search)? {
            None => return Ok(None),
            Some(end) => end,
        };
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
        // reverse case, to satisfy "leftmost" criteria, we need to match as
        // much as we can.
        let revsearch = search
            .clone()
            .earliest(false)
            .span((search.start()..end.offset()).into());
        let start = search::find_rev(rdfa, rcache, &revsearch)?
            .expect("reverse search must match if forward search does");
        debug_assert_eq!(
            start.pattern(),
            end.pattern(),
            "forward and reverse search must match same pattern",
        );
        debug_assert!(start.offset() <= end.offset());
        Ok(Some(Match::new(end.pattern(), start.offset()..end.offset())))
    }
}

type TryMatchesClosure<'h, 'c> =
    Box<dyn FnMut(&Search<'h>) -> Result<Option<Match>, MatchError> + 'c>;

impl Regex {
    fn try_matches_iter<'r: 'c, 'c, 'h>(
        &'r self,
        cache: &'c mut Cache,
        search: Search<'h>,
    ) -> iter::TryMatches<'h, TryMatchesClosure<'h, 'c>> {
        let mut scanner = self.scanner();
        iter::TryMatches::boxed(search, move |search| {
            let pre = scanner.as_mut();
            self.try_search(cache, pre, search)
        })
    }
}

/// Non-search APIs for querying information about the regex and setting a
/// prefilter.
impl Regex {
    /// Return the config for this regex.
    pub fn get_config(&self) -> &Config {
        &self.config
    }

    /// Return the underlying lazy DFA responsible for forward matching.
    ///
    /// This is useful for accessing the underlying lazy DFA and using it
    /// directly if the situation calls for it.
    pub fn forward(&self) -> &DFA {
        &self.forward
    }

    /// Return the underlying lazy DFA responsible for reverse matching.
    ///
    /// This is useful for accessing the underlying lazy DFA and using it
    /// directly if the situation calls for it.
    pub fn reverse(&self) -> &DFA {
        &self.reverse
    }

    /// Returns the total number of patterns matched by this regex.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{Match, hybrid::regex::Regex};
    ///
    /// let re = Regex::new_many(&[r"[a-z]+", r"[0-9]+", r"\w+"])?;
    /// assert_eq!(3, re.pattern_len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn pattern_len(&self) -> usize {
        assert_eq!(self.forward().pattern_len(), self.reverse().pattern_len());
        self.forward().pattern_len()
    }

    /// Create and return a prefilter scanner if this regex's configuration has
    /// a prefilter.
    pub fn scanner(&self) -> Option<prefilter::Scanner> {
        self.get_config().get_prefilter().map(prefilter::Scanner::new)
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
/// * `'c` represents the lifetime of the regex cache. The lifetime of the
/// regex object itself must outlive `'c`.
///
/// This iterator can be created with the [`Regex::find_iter`]
/// method.
#[derive(Debug)]
pub struct FindMatches<'h, 'c>(iter::Matches<'h, TryMatchesClosure<'h, 'c>>);

impl<'h, 'c> Iterator for FindMatches<'h, 'c> {
    type Item = Match;

    #[inline]
    fn next(&mut self) -> Option<Match> {
        self.0.next()
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
/// * `'c` represents the lifetime of the regex cache. The lifetime of the
/// regex object itself must outlive `'c`.
///
/// This iterator can be created with the [`Regex::try_find_iter`]
/// method.
#[derive(Debug)]
pub struct TryFindMatches<'h, 'c>(
    iter::TryMatches<'h, TryMatchesClosure<'h, 'c>>,
);

impl<'h, 'c> Iterator for TryFindMatches<'h, 'c> {
    type Item = Result<Match, MatchError>;

    #[inline]
    fn next(&mut self) -> Option<Result<Match, MatchError>> {
        self.0.next()
    }
}

/// A cache represents a partially computed forward and reverse DFA.
///
/// A cache is the key component that differentiates a classical DFA and a
/// hybrid NFA/DFA (also called a "lazy DFA"). Where a classical DFA builds a
/// complete transition table that can handle all possible inputs, a hybrid
/// NFA/DFA starts with an empty transition table and builds only the parts
/// required during search. The parts that are built are stored in a cache. For
/// this reason, a cache is a required parameter for nearly every operation on
/// a [`Regex`].
///
/// Caches can be created from their corresponding `Regex` via
/// [`Regex::create_cache`]. A cache can only be used with either the `Regex`
/// that created it, or the `Regex` that was most recently used to reset it
/// with [`Cache::reset`]. Using a cache with any other `Regex` may result in
/// panics or incorrect results.
#[derive(Debug, Clone)]
pub struct Cache {
    forward: dfa::Cache,
    reverse: dfa::Cache,
}

impl Cache {
    /// Create a new cache for the given `Regex`.
    ///
    /// The cache returned should only be used for searches for the given
    /// `Regex`. If you want to reuse the cache for another `Regex`, then you
    /// must call [`Cache::reset`] with that `Regex`.
    pub fn new(re: &Regex) -> Cache {
        let forward = dfa::Cache::new(re.forward());
        let reverse = dfa::Cache::new(re.reverse());
        Cache { forward, reverse }
    }

    /// Reset this cache such that it can be used for searching with the given
    /// `Regex` (and only that `Regex`).
    ///
    /// A cache reset permits reusing memory already allocated in this cache
    /// with a different `Regex`.
    ///
    /// Resetting a cache sets its "clear count" to 0. This is relevant if the
    /// `Regex` has been configured to "give up" after it has cleared the cache
    /// a certain number of times.
    ///
    /// # Example
    ///
    /// This shows how to re-purpose a cache for use with a different `Regex`.
    ///
    /// ```
    /// use regex_automata::{hybrid::regex::Regex, Match};
    ///
    /// let re1 = Regex::new(r"\w")?;
    /// let re2 = Regex::new(r"\W")?;
    ///
    /// let mut cache = re1.create_cache();
    /// assert_eq!(
    ///     Some(Match::must(0, 0..2)),
    ///     re1.find(&mut cache, "Δ"),
    /// );
    ///
    /// // Using 'cache' with re2 is not allowed. It may result in panics or
    /// // incorrect results. In order to re-purpose the cache, we must reset
    /// // it with the Regex we'd like to use it with.
    /// //
    /// // Similarly, after this reset, using the cache with 're1' is also not
    /// // allowed.
    /// cache.reset(&re2);
    /// assert_eq!(
    ///     Some(Match::must(0, 0..3)),
    ///     re2.find(&mut cache, "☃"),
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reset(&mut self, re: &Regex) {
        self.forward.reset(re.forward());
        self.reverse.reset(re.reverse());
    }

    /// Returns the heap memory usage, in bytes, as a sum of the forward and
    /// reverse lazy DFA caches.
    ///
    /// This does **not** include the stack size used up by this cache. To
    /// compute that, use `std::mem::size_of::<Cache>()`.
    pub fn memory_usage(&self) -> usize {
        self.forward.memory_usage() + self.reverse.memory_usage()
    }

    /// Return references to the forward and reverse caches, respectively.
    pub fn as_parts(&self) -> (&dfa::Cache, &dfa::Cache) {
        (&self.forward, &self.reverse)
    }

    /// Return mutable references to the forward and reverse caches,
    /// respectively.
    pub fn as_parts_mut(&mut self) -> (&mut dfa::Cache, &mut dfa::Cache) {
        (&mut self.forward, &mut self.reverse)
    }
}

/// The configuration used for compiling a hybrid NFA/DFA regex.
///
/// A regex configuration is a simple data object that is typically used with
/// [`Builder::configure`].
#[derive(Clone, Debug, Default)]
pub struct Config {
    utf8: Option<bool>,
    pre: Option<Option<Arc<dyn Prefilter>>>,
}

impl Config {
    /// Return a new default regex compiler configuration.
    pub fn new() -> Config {
        Config::default()
    }

    /// Whether to enable UTF-8 mode or not.
    ///
    /// When UTF-8 mode is enabled (the default) and an empty match is seen,
    /// the iterators on [`Regex`] will always start the next search at the
    /// next UTF-8 encoded codepoint when searching valid UTF-8. When UTF-8
    /// mode is disabled, such searches are begun at the next byte offset.
    ///
    /// If this mode is enabled and invalid UTF-8 is given to search, then
    /// behavior is unspecified.
    ///
    /// Generally speaking, one should enable this when
    /// [`SyntaxConfig::utf8`](crate::SyntaxConfig::utf8)
    /// and
    /// [`thompson::Config::utf8`](crate::nfa::thompson::Config::utf8)
    /// are enabled, and disable it otherwise.
    ///
    /// # Example
    ///
    /// This example demonstrates the differences between when this option is
    /// enabled and disabled. The differences only arise when the regex can
    /// return matches of length zero.
    ///
    /// In this first snippet, we show the results when UTF-8 mode is disabled.
    ///
    /// ```
    /// use regex_automata::{hybrid::regex::Regex, Match};
    ///
    /// let re = Regex::builder()
    ///     .configure(Regex::config().utf8(false))
    ///     .build(r"")?;
    /// let mut cache = re.create_cache();
    ///
    /// let haystack = "a☃z";
    /// let mut it = re.find_iter(&mut cache, haystack);
    /// assert_eq!(Some(Match::must(0, 0..0)), it.next());
    /// assert_eq!(Some(Match::must(0, 1..1)), it.next());
    /// assert_eq!(Some(Match::must(0, 2..2)), it.next());
    /// assert_eq!(Some(Match::must(0, 3..3)), it.next());
    /// assert_eq!(Some(Match::must(0, 4..4)), it.next());
    /// assert_eq!(Some(Match::must(0, 5..5)), it.next());
    /// assert_eq!(None, it.next());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// And in this snippet, we execute the same search on the same haystack,
    /// but with UTF-8 mode enabled. Notice that byte offsets that would
    /// otherwise split the encoding of `☃` are not returned.
    ///
    /// ```
    /// use regex_automata::{hybrid::regex::Regex, Match};
    ///
    /// let re = Regex::builder()
    ///     .configure(Regex::config().utf8(true))
    ///     .build(r"")?;
    /// let mut cache = re.create_cache();
    ///
    /// let haystack = "a☃z";
    /// let mut it = re.find_iter(&mut cache, haystack);
    /// assert_eq!(Some(Match::must(0, 0..0)), it.next());
    /// assert_eq!(Some(Match::must(0, 1..1)), it.next());
    /// assert_eq!(Some(Match::must(0, 4..4)), it.next());
    /// assert_eq!(Some(Match::must(0, 5..5)), it.next());
    /// assert_eq!(None, it.next());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn utf8(mut self, yes: bool) -> Config {
        self.utf8 = Some(yes);
        self
    }

    /// Attach the given prefilter to this configuration.
    ///
    /// The given prefilter is automatically applied to every search done by
    /// a `Regex`, except for the lower level routines that accept a prefilter
    /// parameter from the caller.
    pub fn prefilter(mut self, pre: Option<Arc<dyn Prefilter>>) -> Config {
        self.pre = Some(pre);
        self
    }

    /// Returns true if and only if this configuration has UTF-8 mode enabled.
    ///
    /// When UTF-8 mode is enabled and an empty match is seen, the iterators on
    /// [`Regex`] will always start the next search at the next UTF-8 encoded
    /// codepoint. When UTF-8 mode is disabled, such searches are begun at the
    /// next byte offset.
    pub fn get_utf8(&self) -> bool {
        self.utf8.unwrap_or(true)
    }

    pub fn get_prefilter(&self) -> Option<&dyn Prefilter> {
        self.pre.as_ref().unwrap_or(&None).as_deref()
    }

    /// Overwrite the default configuration such that the options in `o` are
    /// always used. If an option in `o` is not set, then the corresponding
    /// option in `self` is used. If it's not set in `self` either, then it
    /// remains not set.
    pub(crate) fn overwrite(&self, o: Config) -> Config {
        Config {
            utf8: o.utf8.or(self.utf8),
            pre: o.pre.or_else(|| self.pre.clone()),
        }
    }
}

/// A builder for a regex based on a hybrid NFA/DFA.
///
/// This builder permits configuring options for the syntax of a pattern, the
/// NFA construction, the lazy DFA construction and finally the regex searching
/// itself. This builder is different from a general purpose regex builder
/// in that it permits fine grain configuration of the construction process.
/// The trade off for this is complexity, and the possibility of setting a
/// configuration that might not make sense. For example, there are three
/// different UTF-8 modes:
///
/// * [`SyntaxConfig::utf8`](crate::SyntaxConfig::utf8) controls whether the
/// pattern itself can contain sub-expressions that match invalid UTF-8.
/// * [`nfa::thompson::Config::utf8`](crate::nfa::thompson::Config::utf8)
/// controls whether the implicit unanchored prefix added to the NFA can
/// match through invalid UTF-8 or not.
/// * [`Config::utf8`] controls how the regex iterators themselves advance
/// the starting position of the next search when a match with zero length is
/// found.
///
/// Generally speaking, callers will want to either enable all of these or
/// disable all of these.
///
/// Internally, building a regex requires building two hybrid NFA/DFAs,
/// where one is responsible for finding the end of a match and the other is
/// responsible for finding the start of a match. If you only need to detect
/// whether something matched, or only the end of a match, then you should use
/// a [`dfa::Builder`] to construct a single hybrid NFA/DFA, which is cheaper
/// than building two of them.
///
/// # Example
///
/// This example shows how to disable UTF-8 mode in the syntax and the regex
/// itself. This is generally what you want for matching on arbitrary bytes.
///
/// ```
/// use regex_automata::{
///     hybrid::regex::Regex, nfa::thompson, Match, SyntaxConfig
/// };
///
/// let re = Regex::builder()
///     .configure(Regex::config().utf8(false))
///     .syntax(SyntaxConfig::new().utf8(false))
///     .build(r"foo(?-u:[^b])ar.*")?;
/// let mut cache = re.create_cache();
///
/// let haystack = b"\xFEfoo\xFFarzz\xE2\x98\xFF\n";
/// let expected = Some(Match::must(0, 1..9));
/// let got = re.find(&mut cache, haystack);
/// assert_eq!(expected, got);
/// // Notice that `(?-u:[^b])` matches invalid UTF-8,
/// // but the subsequent `.*` does not! Disabling UTF-8
/// // on the syntax permits this.
/// //
/// // N.B. This example does not show the impact of
/// // disabling UTF-8 mode on a regex Config, since that
/// // only impacts regexes that can produce matches of
/// // length 0.
/// assert_eq!(b"foo\xFFarzz", &haystack[got.unwrap().range()]);
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug)]
pub struct Builder {
    config: Config,
    dfa: dfa::Builder,
    pre: Option<Arc<dyn Prefilter>>,
}

impl Builder {
    /// Create a new regex builder with the default configuration.
    pub fn new() -> Builder {
        Builder { config: Config::default(), dfa: DFA::builder(), pre: None }
    }

    /// Build a regex from the given pattern.
    ///
    /// If there was a problem parsing or compiling the pattern, then an error
    /// is returned.
    pub fn build(&self, pattern: &str) -> Result<Regex, BuildError> {
        self.build_many(&[pattern])
    }

    /// Build a regex from the given patterns.
    pub fn build_many<P: AsRef<str>>(
        &self,
        patterns: &[P],
    ) -> Result<Regex, BuildError> {
        let forward = self.dfa.build_many(patterns)?;
        let reverse = self
            .dfa
            .clone()
            .configure(DFA::config().anchored(true).match_kind(MatchKind::All))
            .thompson(thompson::Config::new().reverse(true))
            .build_many(patterns)?;
        Ok(self.build_from_dfas(forward, reverse))
    }

    /// Build a regex from its component forward and reverse hybrid NFA/DFAs.
    fn build_from_dfas(&self, forward: DFA, reverse: DFA) -> Regex {
        // The congruous method on DFA-backed regexes is exposed, but it's
        // not clear this builder is useful here since lazy DFAs can't be
        // serialized and there is only one type of them.
        let config = self.config.clone();
        Regex { config, forward, reverse }
    }

    /// Apply the given regex configuration options to this builder.
    pub fn configure(&mut self, config: Config) -> &mut Builder {
        self.config = self.config.overwrite(config);
        self
    }

    /// Set the syntax configuration for this builder using
    /// [`SyntaxConfig`](crate::SyntaxConfig).
    ///
    /// This permits setting things like case insensitivity, Unicode and multi
    /// line mode.
    pub fn syntax(
        &mut self,
        config: crate::util::syntax::SyntaxConfig,
    ) -> &mut Builder {
        self.dfa.syntax(config);
        self
    }

    /// Set the Thompson NFA configuration for this builder using
    /// [`nfa::thompson::Config`](thompson::Config).
    ///
    /// This permits setting things like whether additional time should be
    /// spent shrinking the size of the NFA.
    pub fn thompson(&mut self, config: thompson::Config) -> &mut Builder {
        self.dfa.thompson(config);
        self
    }

    /// Set the lazy DFA compilation configuration for this builder using
    /// [`dfa::Config`](dfa::Config).
    ///
    /// This permits setting things like whether Unicode word boundaries should
    /// be heuristically supported or settings how the behavior of the cache.
    pub fn dfa(&mut self, config: dfa::Config) -> &mut Builder {
        self.dfa.configure(config);
        self
    }
}

impl Default for Builder {
    fn default() -> Builder {
        Builder::new()
    }
}

#[inline(always)]
fn next_unwrap(item: Option<Result<Match, MatchError>>) -> Option<Match> {
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
