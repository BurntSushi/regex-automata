use core::borrow::Borrow;

use crate::{
    hybrid::{
        dfa::{self, DFA},
        error::BuildError,
        OverlappingState,
    },
    nfa::thompson,
    util::{
        matchtypes::{MatchError, MatchKind, MultiMatch},
        prefilter::{self, Prefilter},
    },
};

/// A regular expression that uses hybrid NFA/DFAs (also called "lazy DFAs")
/// for fast searching.
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
/// # When should I use this?
///
/// Generally speaking, if you can abide the use of mutable state during
/// search, and you don't need things like capturing groups or Unicode word
/// boundaries support in non-ASCII text, then lazy DFA is likely a robust
/// choice with respect to both search speed and memory usage. Note however
/// that its speed may be worse than a general purpose regex engine if you
/// don't select a good [prefilter].
///
/// If you know ahead of time that your pattern would result in a very large
/// DFA if it was fully compiled, it may be better to use an NFA simulation
/// instead of a lazy DFA. Either that, or increase the cache capacity of your
/// lazy DFA to something that is big enough to hold the state machine (likely
/// through experimentation). The issue here is that if the cache is too small,
/// then it could wind up being reset too frequently and this might decrease
/// searching speed significantly.
///
/// # Differences with fully compiled DFAs
///
/// A [`hybrid::regex::Regex`](crate::hybrid::regex::Regex) and a
/// [`dfa::regex::Regex`](crate::dfa::regex::Regex) both have the same
/// capabilities, but they achieve them through different means. The main
/// difference is that a hybrid or "lazy" regex builds its DFA lazily during
/// search, where as a fully compiled regex will build its DFA at construction
/// time. While building a DFA at search time might sound like it's slow, it
/// tends to work out where most bytes seen reuse pre-built parts of the DFA
/// and thus can be almost as fast as a fully compiled DFA. The main downside
/// is that searching requires mutable space to store the DFA, and, in the
/// worst case, a search can result in a new state being created for each byte
/// seen, which would make searching quite a bit slower.
///
/// A fully compiled DFA never has to worry about searches being slower once
/// it's built. (Aside from, say, the transition table being so large that it
/// is subject to harsh CPU cache effects.) However, of course, building a full
/// DFA can be quite time consuming and memory hungry. Particularly when it's
/// so easy to build large DFAs when Unicode mode is enabled.
///
/// A lazy DFA strikes a nice balance _in practice_, particularly in the
/// presence of Unicode mode, by only building what is needed. It avoids the
/// worst exponential time complexity of DFA compilation by guaranteeing that
/// it will only build at most one state per byte searched. While the worst
/// case here can lead to a very high constant, it will never be exponential.
///
/// # Earliest vs Leftmost vs Overlapping
///
/// The search routines exposed on a `Regex` reflect three different ways
/// of searching:
///
/// * "earliest" means to stop as soon as a match has been detected.
/// * "leftmost" means to continue matching until the underlying
///   automaton cannot advance. This reflects "standard" searching you
///   might be used to in other regex engines. e.g., This permits
///   non-greedy and greedy searching to work as you would expect.
/// * "overlapping" means to find all possible matches, even if they
///   overlap.
///
/// Generally speaking, when doing an overlapping search, you'll want to
/// build your regex lazy DFAs with [`MatchKind::All`] semantics. Using
/// [`MatchKind::LeftmostFirst`] semantics with overlapping searches is
/// likely to lead to odd behavior since `LeftmostFirst` specifically omits
/// some matches that can never be reported due to its semantics.
///
/// The following example shows the differences between how these different
/// types of searches impact looking for matches of `[a-z]+` in the
/// haystack `abc`.
///
/// ```
/// use regex_automata::{hybrid::{dfa, regex}, MatchKind, MultiMatch};
///
/// let pattern = r"[a-z]+";
/// let haystack = "abc".as_bytes();
///
/// // With leftmost-first semantics, we test "earliest" and "leftmost".
/// let re = regex::Builder::new()
///     .dfa(dfa::Config::new().match_kind(MatchKind::LeftmostFirst))
///     .build(pattern)?;
/// let mut cache = re.create_cache();
///
/// // "earliest" searching isn't impacted by greediness
/// let mut it = re.find_earliest_iter(&mut cache, haystack);
/// assert_eq!(Some(MultiMatch::must(0, 0, 1)), it.next());
/// assert_eq!(Some(MultiMatch::must(0, 1, 2)), it.next());
/// assert_eq!(Some(MultiMatch::must(0, 2, 3)), it.next());
/// assert_eq!(None, it.next());
///
/// // "leftmost" searching supports greediness (and non-greediness)
/// let mut it = re.find_leftmost_iter(&mut cache, haystack);
/// assert_eq!(Some(MultiMatch::must(0, 0, 3)), it.next());
/// assert_eq!(None, it.next());
///
/// // For overlapping, we want "all" match kind semantics.
/// let re = regex::Builder::new()
///     .dfa(dfa::Config::new().match_kind(MatchKind::All))
///     .build(pattern)?;
/// let mut cache = re.create_cache();
///
/// // In the overlapping search, we find all three possible matches
/// // starting at the beginning of the haystack.
/// let mut it = re.find_overlapping_iter(&mut cache, haystack);
/// assert_eq!(Some(MultiMatch::must(0, 0, 1)), it.next());
/// assert_eq!(Some(MultiMatch::must(0, 0, 2)), it.next());
/// assert_eq!(Some(MultiMatch::must(0, 0, 3)), it.next());
/// assert_eq!(None, it.next());
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Fallibility
///
/// In non-default configurations, the lazy DFAs generated in this module may
/// return an error during a search. (Currently, the only way this happens is
/// if quit bytes are added, Unicode word boundaries are heuristically enabled,
/// or if the cache is configured to "give up" on a search if it has been
/// cleared too many times. All of these are turned off by default, which means
/// a search can never fail in the default configuration.) For convenience,
/// the main search routines, like [`find_leftmost`](Regex::find_leftmost),
/// will panic if an error occurs. However, if you need to use DFAs which may
/// produce an error at search time, then there are fallible equivalents of
/// all search routines. For example, for `find_leftmost`, its fallible analog
/// is [`try_find_leftmost`](Regex::try_find_leftmost). The routines prefixed
/// with `try_` return `Result<Option<MultiMatch>, MatchError>`, where as the
/// infallible routines simply return `Option<MultiMatch>`.
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
/// let haystack = "foo\nbar".as_bytes();
/// // Normally this would produce a match, since \p{any} contains '\n'.
/// // But since we instructed the automaton to enter a quit state if a
/// // '\n' is observed, this produces a match error instead.
/// let expected = MatchError::Quit { byte: 0x0A, offset: 3 };
/// let got = re.try_find_leftmost(&mut cache, haystack).unwrap_err();
/// assert_eq!(expected, got);
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct Regex {
    pre: Option<Box<dyn Prefilter>>,
    forward: DFA,
    reverse: DFA,
    utf8: bool,
}

#[derive(Debug, Clone)]
pub struct Cache {
    forward: dfa::Cache,
    reverse: dfa::Cache,
}

/// Convenience routines for regex and cache construction.
impl Regex {
    pub fn new(pattern: &str) -> Result<Regex, BuildError> {
        Regex::builder().build(pattern)
    }

    pub fn new_many<P: AsRef<str>>(
        patterns: &[P],
    ) -> Result<Regex, BuildError> {
        Regex::builder().build_many(patterns)
    }

    pub fn config() -> Config {
        Config::new()
    }

    pub fn builder() -> Builder {
        Builder::new()
    }

    pub fn create_cache(&self) -> Cache {
        let forward = dfa::Cache::new(self.forward());
        let reverse = dfa::Cache::new(self.reverse());
        Cache { forward, reverse }
    }
}

/// Standard infallible search routines for finding and iterating over matches.
impl Regex {
    pub fn is_match(&self, cache: &mut Cache, haystack: &[u8]) -> bool {
        self.try_is_match(cache, haystack).unwrap()
    }

    pub fn find_earliest(
        &self,
        cache: &mut Cache,
        haystack: &[u8],
    ) -> Option<MultiMatch> {
        self.try_find_earliest(cache, haystack).unwrap()
    }

    pub fn find_leftmost(
        &self,
        cache: &mut Cache,
        haystack: &[u8],
    ) -> Option<MultiMatch> {
        self.try_find_leftmost(cache, haystack).unwrap()
    }

    pub fn find_overlapping(
        &self,
        cache: &mut Cache,
        haystack: &[u8],
        state: &mut OverlappingState,
    ) -> Option<MultiMatch> {
        self.try_find_overlapping(cache, haystack, state).unwrap()
    }

    pub fn find_earliest_iter<'r, 'c, 't>(
        &'r self,
        cache: &'c mut Cache,
        haystack: &'t [u8],
    ) -> FindEarliestMatches<'r, 'c, 't> {
        FindEarliestMatches::new(self, cache, haystack)
    }

    pub fn find_leftmost_iter<'r, 'c, 't>(
        &'r self,
        cache: &'c mut Cache,
        haystack: &'t [u8],
    ) -> FindLeftmostMatches<'r, 'c, 't> {
        FindLeftmostMatches::new(self, cache, haystack)
    }

    pub fn find_overlapping_iter<'r, 'c, 't>(
        &'r self,
        cache: &'c mut Cache,
        haystack: &'t [u8],
    ) -> FindOverlappingMatches<'r, 'c, 't> {
        FindOverlappingMatches::new(self, cache, haystack)
    }
}

/// Lower level infallible search routines that permit controlling where
/// the search starts and ends in a particular sequence. This is useful for
/// executing searches that need to take surrounding context into account. This
/// is required for correctly implementing iteration because of look-around
/// operators (`^`, `$`, `\b`).
impl Regex {
    pub fn is_match_at(
        &self,
        cache: &mut Cache,
        haystack: &[u8],
        start: usize,
        end: usize,
    ) -> bool {
        self.try_is_match_at(cache, haystack, start, end).unwrap()
    }

    pub fn find_earliest_at(
        &self,
        cache: &mut Cache,
        haystack: &[u8],
        start: usize,
        end: usize,
    ) -> Option<MultiMatch> {
        self.try_find_earliest_at(cache, haystack, start, end).unwrap()
    }

    pub fn find_leftmost_at(
        &self,
        cache: &mut Cache,
        haystack: &[u8],
        start: usize,
        end: usize,
    ) -> Option<MultiMatch> {
        self.try_find_leftmost_at(cache, haystack, start, end).unwrap()
    }

    pub fn find_overlapping_at(
        &self,
        cache: &mut Cache,
        haystack: &[u8],
        start: usize,
        end: usize,
        state: &mut OverlappingState,
    ) -> Option<MultiMatch> {
        self.try_find_overlapping_at(cache, haystack, start, end, state)
            .unwrap()
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
    pub fn try_is_match(
        &self,
        cache: &mut Cache,
        haystack: &[u8],
    ) -> Result<bool, MatchError> {
        self.try_is_match_at(cache, haystack, 0, haystack.len())
    }

    pub fn try_find_earliest(
        &self,
        cache: &mut Cache,
        haystack: &[u8],
    ) -> Result<Option<MultiMatch>, MatchError> {
        self.try_find_earliest_at(cache, haystack, 0, haystack.len())
    }

    pub fn try_find_leftmost(
        &self,
        cache: &mut Cache,
        haystack: &[u8],
    ) -> Result<Option<MultiMatch>, MatchError> {
        self.try_find_leftmost_at(cache, haystack, 0, haystack.len())
    }

    pub fn try_find_overlapping(
        &self,
        cache: &mut Cache,
        haystack: &[u8],
        state: &mut OverlappingState,
    ) -> Result<Option<MultiMatch>, MatchError> {
        self.try_find_overlapping_at(cache, haystack, 0, haystack.len(), state)
    }

    pub fn try_find_earliest_iter<'r, 'c, 't>(
        &'r self,
        cache: &'c mut Cache,
        haystack: &'t [u8],
    ) -> TryFindEarliestMatches<'r, 'c, 't> {
        TryFindEarliestMatches::new(self, cache, haystack)
    }

    pub fn try_find_leftmost_iter<'r, 'c, 't>(
        &'r self,
        cache: &'c mut Cache,
        haystack: &'t [u8],
    ) -> TryFindLeftmostMatches<'r, 'c, 't> {
        TryFindLeftmostMatches::new(self, cache, haystack)
    }

    pub fn try_find_overlapping_iter<'r, 'c, 't>(
        &'r self,
        cache: &'c mut Cache,
        haystack: &'t [u8],
    ) -> TryFindOverlappingMatches<'r, 'c, 't> {
        TryFindOverlappingMatches::new(self, cache, haystack)
    }
}

/// Lower level fallible search routines that permit controlling where the
/// search starts and ends in a particular sequence.
impl Regex {
    pub fn try_is_match_at(
        &self,
        cache: &mut Cache,
        haystack: &[u8],
        start: usize,
        end: usize,
    ) -> Result<bool, MatchError> {
        self.forward()
            .find_leftmost_fwd_at(
                &mut cache.forward,
                self.scanner().as_mut(),
                None,
                haystack,
                start,
                end,
            )
            .map(|x| x.is_some())
    }

    pub fn try_find_earliest_at(
        &self,
        cache: &mut Cache,
        haystack: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Option<MultiMatch>, MatchError> {
        self.try_find_earliest_at_imp(
            self.scanner().as_mut(),
            cache,
            haystack,
            start,
            end,
        )
    }

    pub fn try_find_leftmost_at(
        &self,
        cache: &mut Cache,
        haystack: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Option<MultiMatch>, MatchError> {
        self.try_find_leftmost_at_imp(
            self.scanner().as_mut(),
            cache,
            haystack,
            start,
            end,
        )
    }

    pub fn try_find_overlapping_at(
        &self,
        cache: &mut Cache,
        haystack: &[u8],
        start: usize,
        end: usize,
        state: &mut OverlappingState,
    ) -> Result<Option<MultiMatch>, MatchError> {
        self.try_find_overlapping_at_imp(
            self.scanner().as_mut(),
            cache,
            haystack,
            start,
            end,
            state,
        )
    }
}

impl Regex {
    fn try_find_earliest_at_imp(
        &self,
        pre: Option<&mut prefilter::Scanner>,
        cache: &mut Cache,
        haystack: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Option<MultiMatch>, MatchError> {
        let (fdfa, rdfa) = (self.forward(), self.reverse());
        let (fcache, rcache) = (&mut cache.forward, &mut cache.reverse);
        let end = match fdfa
            .find_earliest_fwd_at(fcache, pre, None, haystack, start, end)?
        {
            None => return Ok(None),
            Some(end) => end,
        };
        // N.B. The only time we need to tell the reverse searcher the pattern
        // to match is in the overlapping case, since it's ambiguous. In the
        // earliest case, I have tentatively convinced myself that it isn't
        // necessary and the reverse search will always find the same pattern
        // to match as the forward search. But I lack a rigorous proof. Why not
        // just provide the pattern anyway? Well, if it is needed, then leaving
        // it out gives us a chance to find a witness.
        let start = rdfa
            .find_earliest_rev_at(rcache, None, haystack, start, end.offset())?
            .expect("reverse search must match if forward search does");
        assert_eq!(
            start.pattern(),
            end.pattern(),
            "forward and reverse search must match same pattern",
        );
        assert!(start.offset() <= end.offset());
        Ok(Some(MultiMatch::new(end.pattern(), start.offset(), end.offset())))
    }

    #[inline(always)]
    fn try_find_leftmost_at_imp(
        &self,
        pre: Option<&mut prefilter::Scanner>,
        cache: &mut Cache,
        haystack: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Option<MultiMatch>, MatchError> {
        let (fdfa, rdfa) = (self.forward(), self.reverse());
        let (fcache, rcache) = (&mut cache.forward, &mut cache.reverse);
        let end = match fdfa
            .find_leftmost_fwd_at(fcache, pre, None, haystack, start, end)?
        {
            None => return Ok(None),
            Some(end) => end,
        };
        // N.B. The only time we need to tell the reverse searcher the pattern
        // to match is in the overlapping case, since it's ambiguous. In the
        // leftmost case, I have tentatively convinced myself that it isn't
        // necessary and the reverse search will always find the same pattern
        // to match as the forward search. But I lack a rigorous proof. Why not
        // just provide the pattern anyway? Well, if it is needed, then leaving
        // it out gives us a chance to find a witness.
        let start = rdfa
            .find_leftmost_rev_at(rcache, None, haystack, start, end.offset())?
            .expect("reverse search must match if forward search does");
        assert_eq!(
            start.pattern(),
            end.pattern(),
            "forward and reverse search must match same pattern",
        );
        assert!(start.offset() <= end.offset());
        Ok(Some(MultiMatch::new(end.pattern(), start.offset(), end.offset())))
    }

    fn try_find_overlapping_at_imp(
        &self,
        pre: Option<&mut prefilter::Scanner>,
        cache: &mut Cache,
        haystack: &[u8],
        start: usize,
        end: usize,
        state: &mut OverlappingState,
    ) -> Result<Option<MultiMatch>, MatchError> {
        let (fdfa, rdfa) = (self.forward(), self.reverse());
        let (fcache, rcache) = (&mut cache.forward, &mut cache.reverse);
        let end = match fdfa.find_overlapping_fwd_at(
            fcache, pre, None, haystack, start, end, state,
        )? {
            None => return Ok(None),
            Some(end) => end,
        };
        // Unlike the leftmost cases, the reverse overlapping search may match
        // a different pattern than the forward search. See test failures when
        // using `None` instead of `Some(end.pattern())` below. Thus, we must
        // run our reverse search using the pattern that matched in the forward
        // direction.
        let start = rdfa
            .find_leftmost_rev_at(
                rcache,
                Some(end.pattern()),
                haystack,
                0,
                end.offset(),
            )?
            .expect("reverse search must match if forward search does");
        assert_eq!(
            start.pattern(),
            end.pattern(),
            "forward and reverse search must match same pattern",
        );
        assert!(start.offset() <= end.offset());
        Ok(Some(MultiMatch::new(end.pattern(), start.offset(), end.offset())))
    }
}

/// Non-search APIs for queryig information about the regex and setting a
/// prefilter.
impl Regex {
    pub fn forward(&self) -> &DFA {
        &self.forward
    }

    pub fn reverse(&self) -> &DFA {
        &self.reverse
    }

    pub fn pattern_count(&self) -> usize {
        assert_eq!(
            self.forward().pattern_count(),
            self.reverse().pattern_count()
        );
        self.forward().pattern_count()
    }

    pub fn prefilter(&self) -> Option<&dyn Prefilter> {
        self.pre.as_ref().map(|x| &**x)
    }

    pub fn set_prefilter(&mut self, pre: Option<Box<dyn Prefilter>>) {
        self.pre = pre;
    }

    fn scanner(&self) -> Option<prefilter::Scanner> {
        self.prefilter().map(prefilter::Scanner::new)
    }
}

/// An iterator over all non-overlapping earliest matches for a particular
/// infallible search.
///
/// The iterator yields a [`MultiMatch`] value until no more matches could be
/// found. If the underlying search returns an error, then this panics.
///
/// The lifetime variables are as follows:
///
/// * `'r` is the lifetime of the regular expression itself.
/// * `'c` is the lifetime of the mutable cache used during search.
/// * `'t` is the lifetime of the text being searched.
#[derive(Debug)]
pub struct FindEarliestMatches<'r, 'c, 't>(TryFindEarliestMatches<'r, 'c, 't>);

impl<'r, 'c, 't> FindEarliestMatches<'r, 'c, 't> {
    fn new(
        re: &'r Regex,
        cache: &'c mut Cache,
        text: &'t [u8],
    ) -> FindEarliestMatches<'r, 'c, 't> {
        FindEarliestMatches(TryFindEarliestMatches::new(re, cache, text))
    }
}

impl<'r, 'c, 't> Iterator for FindEarliestMatches<'r, 'c, 't> {
    type Item = MultiMatch;

    fn next(&mut self) -> Option<MultiMatch> {
        next_unwrap(self.0.next())
    }
}

/// An iterator over all non-overlapping leftmost matches for a particular
/// infallible search.
///
/// The iterator yields a [`MultiMatch`] value until no more matches could be
/// found. If the underlying search returns an error, then this panics.
///
/// The lifetime variables are as follows:
///
/// * `'r` is the lifetime of the regular expression itself.
/// * `'c` is the lifetime of the mutable cache used during search.
/// * `'t` is the lifetime of the text being searched.
#[derive(Debug)]
pub struct FindLeftmostMatches<'r, 'c, 't>(TryFindLeftmostMatches<'r, 'c, 't>);

impl<'r, 'c, 't> FindLeftmostMatches<'r, 'c, 't> {
    fn new(
        re: &'r Regex,
        cache: &'c mut Cache,
        text: &'t [u8],
    ) -> FindLeftmostMatches<'r, 'c, 't> {
        FindLeftmostMatches(TryFindLeftmostMatches::new(re, cache, text))
    }
}

impl<'r, 'c, 't> Iterator for FindLeftmostMatches<'r, 'c, 't> {
    type Item = MultiMatch;

    fn next(&mut self) -> Option<MultiMatch> {
        next_unwrap(self.0.next())
    }
}

/// An iterator over all overlapping matches for a particular infallible
/// search.
///
/// The iterator yields a [`MultiMatch`] value until no more matches could be
/// found. If the underlying search returns an error, then this panics.
///
/// The lifetime variables are as follows:
///
/// * `'r` is the lifetime of the regular expression itself.
/// * `'c` is the lifetime of the mutable cache used during search.
/// * `'t` is the lifetime of the text being searched.
#[derive(Debug)]
pub struct FindOverlappingMatches<'r, 'c, 't>(
    TryFindOverlappingMatches<'r, 'c, 't>,
);

impl<'r, 'c, 't> FindOverlappingMatches<'r, 'c, 't> {
    fn new(
        re: &'r Regex,
        cache: &'c mut Cache,
        text: &'t [u8],
    ) -> FindOverlappingMatches<'r, 'c, 't> {
        FindOverlappingMatches(TryFindOverlappingMatches::new(re, cache, text))
    }
}

impl<'r, 'c, 't> Iterator for FindOverlappingMatches<'r, 'c, 't> {
    type Item = MultiMatch;

    fn next(&mut self) -> Option<MultiMatch> {
        next_unwrap(self.0.next())
    }
}

/// An iterator over all non-overlapping earliest matches for a particular
/// fallible search.
///
/// The iterator yields a [`MultiMatch`] value until no more matches could be
/// found.
///
/// The lifetime variables are as follows:
///
/// * `'r` is the lifetime of the regular expression itself.
/// * `'c` is the lifetime of the mutable cache used during search.
/// * `'t` is the lifetime of the text being searched.
#[derive(Debug)]
pub struct TryFindEarliestMatches<'r, 'c, 't> {
    re: &'r Regex,
    cache: &'c mut Cache,
    scanner: Option<prefilter::Scanner<'r>>,
    text: &'t [u8],
    last_end: usize,
    last_match: Option<usize>,
}

impl<'r, 'c, 't> TryFindEarliestMatches<'r, 'c, 't> {
    fn new(
        re: &'r Regex,
        cache: &'c mut Cache,
        text: &'t [u8],
    ) -> TryFindEarliestMatches<'r, 'c, 't> {
        let scanner = re.scanner();
        TryFindEarliestMatches {
            re,
            cache,
            scanner,
            text,
            last_end: 0,
            last_match: None,
        }
    }
}

impl<'r, 'c, 't> Iterator for TryFindEarliestMatches<'r, 'c, 't> {
    type Item = Result<MultiMatch, MatchError>;

    fn next(&mut self) -> Option<Result<MultiMatch, MatchError>> {
        if self.last_end > self.text.len() {
            return None;
        }
        let result = self.re.try_find_earliest_at_imp(
            self.scanner.as_mut(),
            self.cache,
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
            self.last_end = if self.re.utf8 {
                crate::util::next_utf8(self.text, m.end())
            } else {
                m.end() + 1
            };
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

/// An iterator over all non-overlapping leftmost matches for a particular
/// fallible search.
///
/// The iterator yields a [`MultiMatch`] value until no more matches could be
/// found.
///
/// The lifetime variables are as follows:
///
/// * `'r` is the lifetime of the regular expression itself.
/// * `'c` is the lifetime of the mutable cache used during search.
/// * `'t` is the lifetime of the text being searched.
#[derive(Debug)]
pub struct TryFindLeftmostMatches<'r, 'c, 't> {
    re: &'r Regex,
    cache: &'c mut Cache,
    scanner: Option<prefilter::Scanner<'r>>,
    text: &'t [u8],
    last_end: usize,
    last_match: Option<usize>,
}

impl<'r, 'c, 't> TryFindLeftmostMatches<'r, 'c, 't> {
    fn new(
        re: &'r Regex,
        cache: &'c mut Cache,
        text: &'t [u8],
    ) -> TryFindLeftmostMatches<'r, 'c, 't> {
        let scanner = re.scanner();
        TryFindLeftmostMatches {
            re,
            cache,
            scanner,
            text,
            last_end: 0,
            last_match: None,
        }
    }
}

impl<'r, 'c, 't> Iterator for TryFindLeftmostMatches<'r, 'c, 't> {
    type Item = Result<MultiMatch, MatchError>;

    fn next(&mut self) -> Option<Result<MultiMatch, MatchError>> {
        if self.last_end > self.text.len() {
            return None;
        }
        let result = self.re.try_find_leftmost_at_imp(
            self.scanner.as_mut(),
            self.cache,
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
            self.last_end = if self.re.utf8 {
                crate::util::next_utf8(self.text, m.end())
            } else {
                m.end() + 1
            };
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

/// An iterator over all overlapping matches for a particular fallible search.
///
/// The iterator yields a [`MultiMatch`] value until no more matches could be
/// found.
///
/// The lifetime variables are as follows:
///
/// * `'r` is the lifetime of the regular expression itself.
/// * `'c` is the lifetime of the mutable cache used during search.
/// * `'t` is the lifetime of the text being searched.
#[derive(Debug)]
pub struct TryFindOverlappingMatches<'r, 'c, 't> {
    re: &'r Regex,
    cache: &'c mut Cache,
    scanner: Option<prefilter::Scanner<'r>>,
    text: &'t [u8],
    last_end: usize,
    state: OverlappingState,
}

impl<'r, 'c, 't> TryFindOverlappingMatches<'r, 'c, 't> {
    fn new(
        re: &'r Regex,
        cache: &'c mut Cache,
        text: &'t [u8],
    ) -> TryFindOverlappingMatches<'r, 'c, 't> {
        let scanner = re.scanner();
        TryFindOverlappingMatches {
            re,
            cache,
            scanner,
            text,
            last_end: 0,
            state: OverlappingState::start(),
        }
    }
}

impl<'r, 'c, 't> Iterator for TryFindOverlappingMatches<'r, 'c, 't> {
    type Item = Result<MultiMatch, MatchError>;

    fn next(&mut self) -> Option<Result<MultiMatch, MatchError>> {
        if self.last_end > self.text.len() {
            return None;
        }
        let result = self.re.try_find_overlapping_at_imp(
            self.scanner.as_mut(),
            self.cache,
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
        // responsible for ensuring that progress is always made.
        self.last_end = m.end();
        Some(Ok(m))
    }
}

/// The configuration used for compiling a hybrid NFA/DFA regex.
///
/// A regex configuration is a simple data object that is typically used with
/// [`Builder::configure`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Config {
    utf8: Option<bool>,
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
    /// use regex_automata::{hybrid::regex::Regex, MultiMatch};
    ///
    /// let re = Regex::builder()
    ///     .configure(Regex::config().utf8(false))
    ///     .build(r"")?;
    /// let mut cache = re.create_cache();
    ///
    /// let haystack = "a☃z".as_bytes();
    /// let mut it = re.find_leftmost_iter(&mut cache, haystack);
    /// assert_eq!(Some(MultiMatch::must(0, 0, 0)), it.next());
    /// assert_eq!(Some(MultiMatch::must(0, 1, 1)), it.next());
    /// assert_eq!(Some(MultiMatch::must(0, 2, 2)), it.next());
    /// assert_eq!(Some(MultiMatch::must(0, 3, 3)), it.next());
    /// assert_eq!(Some(MultiMatch::must(0, 4, 4)), it.next());
    /// assert_eq!(Some(MultiMatch::must(0, 5, 5)), it.next());
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
    /// use regex_automata::{hybrid::regex::Regex, MultiMatch};
    ///
    /// let re = Regex::builder()
    ///     .configure(Regex::config().utf8(true))
    ///     .build(r"")?;
    /// let mut cache = re.create_cache();
    ///
    /// let haystack = "a☃z".as_bytes();
    /// let mut it = re.find_leftmost_iter(&mut cache, haystack);
    /// assert_eq!(Some(MultiMatch::must(0, 0, 0)), it.next());
    /// assert_eq!(Some(MultiMatch::must(0, 1, 1)), it.next());
    /// assert_eq!(Some(MultiMatch::must(0, 4, 4)), it.next());
    /// assert_eq!(Some(MultiMatch::must(0, 5, 5)), it.next());
    /// assert_eq!(None, it.next());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn utf8(mut self, yes: bool) -> Config {
        self.utf8 = Some(yes);
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

    /// Overwrite the default configuration such that the options in `o` are
    /// always used. If an option in `o` is not set, then the corresponding
    /// option in `self` is used. If it's not set in `self` either, then it
    /// remains not set.
    pub(crate) fn overwrite(self, o: Config) -> Config {
        Config { utf8: o.utf8.or(self.utf8) }
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
/// This example shows how to disable UTF-8 mode in the syntax, the NFA and
/// the regex itself. This is generally what you want for matching on
/// arbitrary bytes.
///
/// ```
/// use regex_automata::{
///     hybrid::regex::Regex, nfa::thompson, MultiMatch, SyntaxConfig
/// };
///
/// let re = Regex::builder()
///     .configure(Regex::config().utf8(false))
///     .syntax(SyntaxConfig::new().utf8(false))
///     .thompson(thompson::Config::new().utf8(false))
///     .build(r"foo(?-u:[^b])ar.*")?;
/// let mut cache = re.create_cache();
///
/// let haystack = b"\xFEfoo\xFFarzz\xE2\x98\xFF\n";
/// let expected = Some(MultiMatch::must(0, 1, 9));
/// let got = re.find_leftmost(&mut cache, haystack);
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
    config: Config,
    dfa: dfa::Builder,
}

impl Builder {
    /// Create a new regex builder with the default configuration.
    pub fn new() -> Builder {
        Builder { config: Config::default(), dfa: DFA::builder() }
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
            .configure(
                DFA::config()
                    .anchored(true)
                    .match_kind(MatchKind::All)
                    .starts_for_each_pattern(true),
            )
            .thompson(thompson::Config::new().reverse(true))
            .build_many(patterns)?;
        Ok(self.build_from_dfas(forward, reverse))
    }

    /// Build a regex from its component forward and reverse hybrid NFA/DFAs.
    fn build_from_dfas(&self, forward: DFA, reverse: DFA) -> Regex {
        // The congruous method on DFA-backed regexes is exposed, but it's
        // not clear this builder is useful here since lazy DFAs can't be
        // serialized and there is only one type of them.
        let utf8 = self.config.get_utf8();
        Regex { pre: None, forward, reverse, utf8 }
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
fn next_unwrap(
    item: Option<Result<MultiMatch, MatchError>>,
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
