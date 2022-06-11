use crate::util::{
    prefilter,
    search::{HalfMatch, Match, MatchError, Search},
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
pub struct TryMatches<'h, 'p, F> {
    /// The regex engine execution function.
    finder: F,
    /// The search configuration.
    search: Search<'h, 'p>,
    /// Records the end offset of the most recent match. This is necessary to
    /// handle a corner case for preventing empty matches from overlapping with
    /// the ending bounds of a prior match.
    last_match_end: Option<usize>,
}

impl<'c, 'h, 'p, F> TryMatches<'h, 'p, F>
where
    F: FnMut(&Search<'h, 'p>) -> Result<Option<Match>, MatchError> + 'c,
{
    /// Create a new fallible non-overlapping matches iterator.
    ///
    /// The given `search` provides the parameters (including the haystack),
    /// while the `finder` represents a closure that calls the underlying regex
    /// engine. The closure may borrow any additional state that is needed,
    /// such as a prefilter scanner.
    pub fn new(search: Search<'h, 'p>, finder: F) -> TryMatches<'h, 'p, F> {
        TryMatches { finder, search, last_match_end: None }
    }

    /// Like `new`, but boxes the given closure into a `dyn` object.
    ///
    /// This is useful when you can give up function inlining in favor of being
    /// able to write the type of the closure. This is often necessary for
    /// composition to work cleanly.
    pub fn boxed(
        search: Search<'h, 'p>,
        finder: F,
    ) -> TryMatches<
        'h,
        'p,
        Box<
            dyn FnMut(&Search<'h, 'p>) -> Result<Option<Match>, MatchError>
                + 'c,
        >,
    > {
        TryMatches::new(search, Box::new(finder))
    }

    /// Return an infallible version of this iterator.
    ///
    /// Any item yielded that corresponds to an error results in a panic.
    pub fn infallible(self) -> Matches<'h, 'p, F> {
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
    /// quite a bit of code, comparatively. Keeping this code out of the main
    /// iterator function keeps it smaller and more amenable to inlining
    /// itself.
    #[cold]
    #[inline(never)]
    fn handle_overlapping_empty_match(
        &mut self,
        mut m: Match,
    ) -> Option<Result<Match, MatchError>> {
        assert!(m.is_empty());
        // We never permit an empty match to match at the ending position of
        // the previous match. This makes intuitive sense and matches the
        // presiding behavior of most general purpose regex engines. So if
        // the match we have overlaps with the previous one, then we just run
        // another search and report that.
        //
        // We only need to handle this case for empty matches because it's
        // the only way to get an overlapping match in a leftmost search. Any
        // non-empty match will have an end position at least one past the
        // start position, which means it can't overlap with a previous match
        // (but may be adjacent).
        //
        // Intuitively, this is also required to make forward progress. If we
        // didn't handle this case specifically, then an empty match would
        // result in setting the next search's start position to the same
        // as the previous search's start position. Thus, it would never
        // terminate.
        if Some(m.end()) == self.last_match_end {
            self.search.set_start(self.search.start().checked_add(1).unwrap());
            m = match (self.finder)(&self.search).transpose()? {
                Err(err) => return Some(Err(err)),
                Ok(m) => m,
            };
            self.search.set_start(m.end());
        }
        Some(Ok(m))
    }
}

impl<'c, 'h, 'p, F> Iterator for TryMatches<'h, 'p, F>
where
    F: FnMut(&Search<'h, 'p>) -> Result<Option<Match>, MatchError> + 'c,
{
    type Item = Result<Match, MatchError>;

    #[inline]
    fn next(&mut self) -> Option<Result<Match, MatchError>> {
        let mut m = match (self.finder)(&self.search).transpose()? {
            Err(err) => return Some(Err(err)),
            Ok(m) => m,
        };
        self.search.set_start(m.end());
        if m.is_empty() {
            m = match self.handle_overlapping_empty_match(m)? {
                Err(err) => return Some(Err(err)),
                Ok(m) => m,
            };
        }
        self.last_match_end = Some(m.end());
        Some(Ok(m))
    }
}

impl<'h, 'p, F> core::fmt::Debug for TryMatches<'h, 'p, F> {
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
pub struct Matches<'h, 'p, F>(TryMatches<'h, 'p, F>);

impl<'c, 'h, 'p, F> Iterator for Matches<'h, 'p, F>
where
    F: FnMut(&Search<'h, 'p>) -> Result<Option<Match>, MatchError> + 'c,
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

impl<'h, 'p, F> core::fmt::Debug for Matches<'h, 'p, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_tuple("Matches").field(&self.0).finish()
    }
}

/// An iterator over all non-overlapping half matches for a fallible search.
///
/// The iterator yields a `Result<HalfMatch, MatchError>` value until no more
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
///
/// **WARNING:** Unlike other iterators that require both the start and end
/// bounds of a match, this iterator does not respect the [`Search::utf8`]
/// setting. Namely, if the underlying regex engine reports an empty match
/// that falls on an invalid UTF-8 boundary, then this iterator will yield it.
pub struct TryHalfMatches<'h, 'p, F> {
    /// The regex engine execution function.
    finder: F,
    /// The search configuration.
    search: Search<'h, 'p>,
    /// Records the end offset of the most recent match. This is necessary to
    /// handle a corner case for preventing empty matches from overlapping with
    /// the ending bounds of a prior match.
    last_match_end: Option<usize>,
}

impl<'c, 'h, 'p, F> TryHalfMatches<'h, 'p, F>
where
    F: FnMut(&Search<'h, 'p>) -> Result<Option<HalfMatch>, MatchError> + 'c,
{
    /// Create a new fallible non-overlapping matches iterator.
    ///
    /// The given `search` provides the parameters (including the haystack),
    /// while the `finder` represents a closure that calls the underlying regex
    /// engine. The closure may borrow any additional state that is needed,
    /// such as a prefilter scanner.
    pub fn new(
        search: Search<'h, 'p>,
        finder: F,
    ) -> TryHalfMatches<'h, 'p, F> {
        TryHalfMatches { finder, search, last_match_end: None }
    }

    /// Like `new`, but boxes the given closure into a `dyn` object.
    ///
    /// This is useful when you can give up function inlining in favor of being
    /// able to write the type of the closure. This is often necessary for
    /// composition to work cleanly.
    pub fn boxed(
        search: Search<'h, 'p>,
        finder: F,
    ) -> TryHalfMatches<
        'h,
        'p,
        Box<
            dyn FnMut(&Search<'h, 'p>) -> Result<Option<HalfMatch>, MatchError>
                + 'c,
        >,
    > {
        TryHalfMatches::new(search, Box::new(finder))
    }

    /// Return an infallible version of this iterator.
    ///
    /// Any item yielded that corresponds to an error results in a panic.
    pub fn infallible(self) -> HalfMatches<'h, 'p, F> {
        HalfMatches(self)
    }

    /// Handles the special case of a match that begins where the previous
    /// match ended. Without this special handling, it'd be possible to get
    /// stuck where an empty match never results in forward progress. This
    /// also makes it more consistent with how presiding general purpose regex
    /// engines work.
    #[cold]
    #[inline(never)]
    fn handle_overlapping_empty_match(
        &mut self,
        m: HalfMatch,
    ) -> Option<Result<HalfMatch, MatchError>> {
        // Since we are only here when 'm.offset()' matches the offset of the
        // last match, it follows that this must have been an empty match.
        // Since we both need to make progress *and* prevent overlapping
        // matches, we discard this match and advance the search by 1.
        //
        // Note that we do not prevent this iterator from returning an offset
        // that splits a codepoint. Our other iterators that detect this work
        // with the full match offsets and so know only to check this case when
        // an empty match is found. But here, all we have is one half of the
        // match, which means we don't know if it's empty or not.
        //
        // We could prevent *any* match from being returned if it splits a
        // codepoint, but that seems like it's going too far.
        self.search.set_start(self.search.start().checked_add(1).unwrap());
        if self.search.is_done() {
            return None;
        }
        (self.finder)(&self.search).transpose()
    }
}

impl<'c, 'h, 'p, F> Iterator for TryHalfMatches<'h, 'p, F>
where
    F: FnMut(&Search<'h, 'p>) -> Result<Option<HalfMatch>, MatchError> + 'c,
{
    type Item = Result<HalfMatch, MatchError>;

    #[inline]
    fn next(&mut self) -> Option<Result<HalfMatch, MatchError>> {
        if self.search.is_done() {
            return None;
        }
        let mut m = match (self.finder)(&self.search).transpose()? {
            Err(err) => return Some(Err(err)),
            Ok(m) => m,
        };
        if Some(m.offset()) == self.last_match_end {
            m = match self.handle_overlapping_empty_match(m)? {
                Err(err) => return Some(Err(err)),
                Ok(m) => m,
            };
        }
        self.search.set_start(m.offset());
        self.last_match_end = Some(m.offset());
        Some(Ok(m))
    }
}

impl<'h, 'p, F> core::fmt::Debug for TryHalfMatches<'h, 'p, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("TryHalfMatches")
            .field("finder", &"<closure>")
            .field("search", &self.search)
            .field("last_match_end", &self.last_match_end)
            .finish()
    }
}

/// An iterator over all non-overlapping half matches for an infallible search.
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
///
/// **WARNING:** Unlike other iterators that require both the start and end
/// bounds of a match, this iterator does not respect the [`Search::utf8`]
/// setting. Namely, if the underlying regex engine reports an empty match
/// that falls on an invalid UTF-8 boundary, then this iterator will yield it.
pub struct HalfMatches<'h, 'p, F>(TryHalfMatches<'h, 'p, F>);

impl<'c, 'h, 'p, F> Iterator for HalfMatches<'h, 'p, F>
where
    F: FnMut(&Search<'h, 'p>) -> Result<Option<HalfMatch>, MatchError> + 'c,
{
    type Item = HalfMatch;

    #[inline]
    fn next(&mut self) -> Option<HalfMatch> {
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

impl<'h, 'p, F> core::fmt::Debug for HalfMatches<'h, 'p, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_tuple("Matches").field(&self.0).finish()
    }
}
