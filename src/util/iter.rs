use crate::util::{
    is_char_boundary,
    matchtypes::{Match, MatchError, Search},
    prefilter,
};

/// An iterator over all non-overlapping matches for a fallible search.
///
/// The iterator yields a `Result<Match, MatchError>` value until no more
/// matches could be found.
///
/// The type parameters are as follows:
///
/// * `F` represents the type of a closure that executes the search.
/// * `H` is the type of the underlying haystack. This is usually one of
/// `&[u8]`, `Vec<u8>`, `&str` or `String`. But it can be anything that
/// satisfies `AsRef<[u8]>`.
///
/// When possible, prefer the iterators defined on the regex engine you're
/// using. This type serves as the common implementation for the class of
/// "non-overlapping matches" iterator, and is thus a bit more unwieldy to use.
pub struct TryMatches<F, H> {
    /// The regex engine execution function.
    finder: F,
    /// The search configuration.
    search: Search<H>,
    /// Records when end offset of the most recent match. This is necessary to
    /// handle a corner case for preventing empty matches from overlapping with
    /// the ending bounds of a prior match.
    last_match_end: Option<usize>,
}

impl<'c, F, H: AsRef<[u8]>> TryMatches<F, H>
where
    F: FnMut(&Search<H>) -> Result<Option<Match>, MatchError> + 'c,
{
    /// Create a new fallible non-overlapping matches iterator.
    ///
    /// The given `search` provides the parameters (including the haystack),
    /// while the `finder` represents a closure that calls the underlying regex
    /// engine. The closure may borrow any additional state that is needed,
    /// such as a prefilter scanner.
    pub fn new(search: Search<H>, finder: F) -> TryMatches<F, H> {
        TryMatches { finder, search, last_match_end: None }
    }

    /// Like `new`, but boxes the given closure into a `dyn` object.
    ///
    /// This is useful when you can give up function inlining in favor of being
    /// able to write the type of the closure. This is often necessary for
    /// composition to work cleanly.
    pub fn boxed(
        search: Search<H>,
        finder: F,
    ) -> TryMatches<
        Box<dyn FnMut(&Search<H>) -> Result<Option<Match>, MatchError> + 'c>,
        H,
    > {
        TryMatches::new(search, Box::new(finder))
    }

    /// Return an infallible version of this iterator.
    ///
    /// Any item yielded that corresponds to an error results in a panic.
    pub fn infallible(self) -> Matches<F, H> {
        Matches(self)
    }

    /// Handles the special case of an empty match by ensuring that 1) the
    /// iterator always advances and 2) empty matches never overlap with other
    /// matches.
    ///
    /// (1) is necessary because we principally make progress by setting the
    /// starting location of the next search to the ending location of the last
    /// match. But if a match is empty, then this results in a search that does
    /// not advance and thus does not terminate.
    ///
    /// (2) is not strictly necessary, but makes intuitive sense and matches
    /// the presiding behavior of most general purpose regex engines. The
    /// "intuitive sense" here is that we want to report NON-overlapping
    /// matches. So for example, given the regex 'a|(?:)' against the haystack
    /// 'a', without the special handling, you'd get the matches [0, 1) and [1,
    /// 1), where the latter overlaps with the end bounds of the former.
    ///
    /// Note that we mark this cold and forcefully prevent inlining because
    /// handling empty matches like this is extremely rare and does require
    /// quite a bit of code. Keeping this code out of the main iterator
    /// function keeps it smaller and more amenable to inlining itself.
    #[cold]
    #[inline(never)]
    fn handle_empty(
        &mut self,
        mut m: Match,
    ) -> Option<Result<Match, MatchError>> {
        assert!(m.is_empty());
        // Since an empty match doesn't advance the search position on its own,
        // we have to do it ourselves.
        self.search.step();
        // But! We never permit an empty match to match at the ending position
        // of the previous match. This makes intuitive sense and matches the
        // presiding behavior of most general purpose regex engines. So if
        // the match we have overlaps with the previous one, then we just run
        // another search and report that.
        if Some(m.end()) == self.last_match_end {
            if self.search.is_done() {
                return None;
            }
            m = match (self.finder)(&self.search).transpose()? {
                Err(err) => return Some(Err(err)),
                Ok(m) => m,
            };
            self.search.set_start(m.end());
            // This is not striclty necessary, but if we got an empty match
            // here, then the next call to 'self.finder' should always return
            // the same result as it previously did, which will cause us to
            // enter this branch again. But if we advance the search by a step
            // here---which is what we'll always ultimately wind up doing
            // anyway---then we can avoid an extra 'self.finder' call on the
            // next iteration.
            if m.is_empty() {
                self.search.step();
            }
        }
        Some(Ok(m))
    }
}

impl<'c, F, H: AsRef<[u8]>> Iterator for TryMatches<F, H>
where
    F: FnMut(&Search<H>) -> Result<Option<Match>, MatchError> + 'c,
{
    type Item = Result<Match, MatchError>;

    #[inline]
    fn next(&mut self) -> Option<Result<Match, MatchError>> {
        if self.search.is_done() {
            return None;
        }
        let mut m = match (self.finder)(&self.search).transpose()? {
            Err(err) => return Some(Err(err)),
            Ok(m) => m,
        };
        self.search.set_start(m.end());
        if m.is_empty() {
            m = match self.handle_empty(m)? {
                Err(err) => return Some(Err(err)),
                Ok(m) => m,
            };
        }
        self.last_match_end = Some(m.end());
        Some(Ok(m))
    }
}

impl<F, H: AsRef<[u8]>> core::fmt::Debug for TryMatches<F, H> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("TryMatches")
            .field("finder", &"<closure>")
            .field("search", &self.search)
            .field("last_match_end", &self.last_match_end)
            .finish()
    }
}

/// An iterator over all non-overlapping matches for an infallible search.
///
/// The iterator yields a [`Match`] value until no more matches could be found.
/// If the underlying regex engine returns an error, then a panic occurs.
///
/// The type parameters are as follows:
///
/// * `F` represents the type of a closure that executes the search.
/// * `H` is the type of the underlying haystack. This is usually one of
/// `&[u8]`, `Vec<u8>`, `&str` or `String`. But it can be anything that
/// satisfies `AsRef<[u8]>`.
///
/// When possible, prefer the iterators defined on the regex engine you're
/// using. This type serves as the common implementation for the class of
/// "non-overlapping matches" iterator, and is thus a bit more unwieldy to use.
pub struct Matches<F, H>(TryMatches<F, H>);

impl<'c, F, H: AsRef<[u8]>> Iterator for Matches<F, H>
where
    F: FnMut(&Search<H>) -> Result<Option<Match>, MatchError> + 'c,
{
    type Item = Match;

    #[inline]
    fn next(&mut self) -> Option<Match> {
        match self.0.next()? {
            Ok(m) => Some(m),
            Err(err) => panic!(
                "unexpected regex find error: {}\n\
                 to handle find errors, use try_ methods",
                err,
            ),
        }
    }
}

impl<F, H: AsRef<[u8]>> core::fmt::Debug for Matches<F, H> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_tuple("Matches").field(&self.0).finish()
    }
}

/// An iterator over all overlapping matches for a fallible search.
///
/// The iterator yields a `Result<Match, MatchError>` value until no more
/// matches could be found.
///
/// The type parameters are as follows:
///
/// * `F` represents the type of a closure that executes the search.
/// * `H` is the type of the underlying haystack. This is usually one of
/// `&[u8]`, `Vec<u8>`, `&str` or `String`. But it can be anything that
/// satisfies `AsRef<[u8]>`.
///
/// When possible, prefer the iterators defined on the regex engine you're
/// using. This type serves as the common implementation for the class of
/// "overlapping matches" iterator, and is thus a bit more unwieldy to use.
pub struct TryOverlappingMatches<F, H> {
    finder: F,
    search: Search<H>,
}

impl<'c, F, H: AsRef<[u8]>> TryOverlappingMatches<F, H>
where
    F: FnMut(&Search<H>) -> Result<Option<Match>, MatchError> + 'c,
{
    /// Create a new fallible overlapping matches iterator.
    ///
    /// The given `search` provides the parameters (including the haystack),
    /// while the `finder` represents a closure that calls the underlying regex
    /// engine. The closure may borrow any additional state that is needed,
    /// such as a prefilter or an overlapping's search's caller provided state.
    pub fn new(search: Search<H>, finder: F) -> TryOverlappingMatches<F, H> {
        TryOverlappingMatches { finder, search }
    }

    /// Like `new`, but boxes the given closure into a `dyn` object.
    ///
    /// This is useful when you can give up function inlining in favor of being
    /// able to write the type of the closure. This is often necessary for
    /// composition to work cleanly.
    pub fn boxed(
        search: Search<H>,
        finder: F,
    ) -> TryOverlappingMatches<
        Box<dyn FnMut(&Search<H>) -> Result<Option<Match>, MatchError> + 'c>,
        H,
    > {
        TryOverlappingMatches::new(search, Box::new(finder))
    }

    /// Return an infallible version of this iterator.
    ///
    /// Any item yielded that corresponds to an error results in a panic.
    pub fn infallible(self) -> OverlappingMatches<F, H> {
        OverlappingMatches(self)
    }

    /// If the given empty match is invalid, then throw it away and keep
    /// executing the underlying finder until a valid match is returned.
    ///
    /// The only way an empty match is invalid is if it splits a UTF-8 encoding
    /// of a Unicode scalar value when the search has [`Search::utf8`] enabled.
    /// Otherwise, all empty matches are valid.
    ///
    /// The handling of empty matches is otherwise much simpler than it is for
    /// non-overlapping searches, since overlapping empty matches are perfectly
    /// fine. We just need to throw away matches that split a codepoint.
    ///
    /// Why not do this in the regex engine? An easy way of doing it in the
    /// regex engine itself eludes me. In particular, some regex engines can
    /// only report one half of a match, and thus can't actually know whether
    /// they're reporting an empty match or not and thus cannot special case
    /// it.
    #[cold]
    #[inline(never)]
    fn skip_invalid_empty_matches(
        &mut self,
        mut m: Match,
    ) -> Option<Result<Match, MatchError>> {
        assert!(m.is_empty());
        if !self.search.get_utf8() {
            return Some(Ok(m));
        }
        while m.is_empty() && !self.search.is_char_boundary(m.end()) {
            m = match (self.finder)(&self.search).transpose()? {
                Err(err) => return Some(Err(err)),
                Ok(m) => m,
            };
            self.search.set_start(m.end());
        }
        Some(Ok(m))
    }
}

impl<'c, F, H: AsRef<[u8]>> Iterator for TryOverlappingMatches<F, H>
where
    F: FnMut(&Search<H>) -> Result<Option<Match>, MatchError> + 'c,
{
    type Item = Result<Match, MatchError>;

    #[inline]
    fn next(&mut self) -> Option<Result<Match, MatchError>> {
        if self.search.is_done() {
            return None;
        }
        let mut m = match (self.finder)(&self.search).transpose()? {
            Err(err) => return Some(Err(err)),
            Ok(m) => m,
        };
        self.search.set_start(m.end());
        if m.is_empty() {
            m = match self.skip_invalid_empty_matches(m)? {
                Err(err) => return Some(Err(err)),
                Ok(m) => m,
            };
        }
        Some(Ok(m))
    }
}

impl<F, H: AsRef<[u8]>> core::fmt::Debug for TryOverlappingMatches<F, H> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("TryOverlappingMatches")
            .field("finder", &"<closure>")
            .field("search", &self.search)
            .finish()
    }
}

/// An iterator over all overlapping matches for an infallible search.
///
/// The iterator yields a [`Match`] value until no more matches could be found.
/// If the underlying regex engine returns an error, then a panic occurs.
///
/// The type parameters are as follows:
///
/// * `F` represents the type of a closure that executes the search.
/// * `H` is the type of the underlying haystack. This is usually one of
/// `&[u8]`, `Vec<u8>`, `&str` or `String`. But it can be anything that
/// satisfies `AsRef<[u8]>`.
///
/// When possible, prefer the iterators defined on the regex engine you're
/// using. This type serves as the common implementation for the class of
/// "overlapping matches" iterator, and is thus a bit more unwieldy to use.
pub struct OverlappingMatches<F, H>(TryOverlappingMatches<F, H>);

impl<'c, F, H: AsRef<[u8]>> Iterator for OverlappingMatches<F, H>
where
    F: FnMut(&Search<H>) -> Result<Option<Match>, MatchError> + 'c,
{
    type Item = Match;

    #[inline]
    fn next(&mut self) -> Option<Match> {
        match self.0.next() {
            None => None,
            Some(Ok(m)) => Some(m),
            Some(Err(err)) => panic!(
                "unexpected regex overlapping find error: {}\n\
                 to handle find errors, use try_ methods",
                err,
            ),
        }
    }
}

impl<F, H: AsRef<[u8]>> core::fmt::Debug for OverlappingMatches<F, H> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_tuple("OverlappingMatches").field(&self.0).finish()
    }
}
