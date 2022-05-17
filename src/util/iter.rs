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
pub struct TryMatches<'h, F> {
    /// The regex engine execution function.
    finder: F,
    /// The search configuration.
    search: Search<'h>,
    /// Records the end offset of the most recent match. This is necessary to
    /// handle a corner case for preventing empty matches from overlapping with
    /// the ending bounds of a prior match.
    last_match_end: Option<usize>,
}

impl<'c, 'h, F> TryMatches<'h, F>
where
    F: FnMut(&Search<'h>) -> Result<Option<Match>, MatchError> + 'c,
{
    /// Create a new fallible non-overlapping matches iterator.
    ///
    /// The given `search` provides the parameters (including the haystack),
    /// while the `finder` represents a closure that calls the underlying regex
    /// engine. The closure may borrow any additional state that is needed,
    /// such as a prefilter scanner.
    pub fn new(search: Search<'h>, finder: F) -> TryMatches<'h, F> {
        TryMatches { finder, search, last_match_end: None }
    }

    /// Like `new`, but boxes the given closure into a `dyn` object.
    ///
    /// This is useful when you can give up function inlining in favor of being
    /// able to write the type of the closure. This is often necessary for
    /// composition to work cleanly.
    pub fn boxed(
        search: Search<'h>,
        finder: F,
    ) -> TryMatches<
        'h,
        Box<dyn FnMut(&Search<'h>) -> Result<Option<Match>, MatchError> + 'c>,
    > {
        TryMatches::new(search, Box::new(finder))
    }

    /// Return an infallible version of this iterator.
    ///
    /// Any item yielded that corresponds to an error results in a panic.
    pub fn infallible(self) -> Matches<'h, F> {
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

    #[cold]
    #[inline(never)]
    fn handle_overlapping_empty_match(
        &mut self,
        mut m: Match,
    ) -> Option<Result<Match, MatchError>> {
        assert!(m.is_empty());
        // This is an optimization that makes matching the empty string a bit
        // faster by avoiding extra calls to the regex engine. Namely, instead
        // of forcing the regex engine to detect and skip empty matches that
        // split codepoints, we detect when an empty match occurs and move
        // forward in the haystack by one codepoint.
        //
        // This helps even in the pure ASCII case, since otherwise we'll
        // wind up finding the same empty match as the previous search,
        // and only then do we discard it and try the search again after
        // incrementing the starting position. So even in the ASCII case, this
        // cuts the number of calls to the regex engine in half, which can be
        // substantial for pathological cases like the empty regex that match
        // at every position.
        self.search.step();
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
            // self.search.step_one();
            m = match (self.finder)(&self.search).transpose()? {
                Err(err) => return Some(Err(err)),
                Ok(m) => m,
            };
            self.search.set_start(m.end());
        }
        Some(Ok(m))
    }
}

impl<'c, 'h, F> Iterator for TryMatches<'h, F>
where
    F: FnMut(&Search<'h>) -> Result<Option<Match>, MatchError> + 'c,
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

impl<'h, F> core::fmt::Debug for TryMatches<'h, F> {
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
pub struct Matches<'h, F>(TryMatches<'h, F>);

impl<'c, 'h, F> Iterator for Matches<'h, F>
where
    F: FnMut(&Search<'h>) -> Result<Option<Match>, MatchError> + 'c,
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

impl<'h, F> core::fmt::Debug for Matches<'h, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_tuple("Matches").field(&self.0).finish()
    }
}

// BREADCRUMBS: It really just seems like the Captures iterator doesn't belong
// here. First of all, it's tied to a specific breed of regex engine (currently
// a Thompson NFA). Second of all, iterating over 'Captures' values is not
// terribly efficient. It's fine for a super high level API in something like
// the regex crate, but do we really need to provide it in this crate..?
//
// OK, so let's say we don't. And someone wants to loop over all matches via
// the PikeVM but with capturing groups. Great. So they call PikeVM and pass
// in a '&mut Captures'. And... how do they deal with empty matches?
//
// That's the perennial problem. That's a big reason why these iterator helpers
// exist in the first place. Is there any way we can solve the sub-problem
// of dealing with empty matches via a smaller logical component than an
// iterator? Then folks using the lower level '&mut Captures' API could just
// use that...
//
// The main issue is that dealing with empty matches *really* wants to both
// be able to control how the cursor is advanced and whether a match result
// should be accepted.
//
// But there is perhaps an easier way to think about this. We can remove the
// need to care about how we advance (just always adding 1) at the cost of
// perf in some degenerate cases, which is maybe OK. In that case, the logical
// implementation looks something like this:
//
// prelude:
//   let (start, end) = (0, haystack.len());
//   let last_match = None;
// loop:
//   let m = next_match(start..end);
//   while m.is_empty()
//     && (Some(m.end()) == last_match
//       || (utf8mode && !is_char_boundary(end))) {
//     start += 1;
//     m = next_match(start..end);
//   }
//   start = m.end();
//   last_match = m.end();
//   yield m
//
// Maybe we can just write a simple 'next' higher-order function?
//
// Or... what if we really do just handle these cases in the "regex" APIs?
// That is, the code that is responsible for returning a Match is also
// responsible for respecting UTF-8..? The major problem there though is that
// it doesn't prevent the overlapping matches since it doesn't know about the
// previous match. So I think that piece always has to be a property of
// iteration unfortunately. But, if we make the regex engine always avoid
// splitting codepoints, even for empty matches, then iteration itself can
// be simpler and not worry about UTF-8 at all. So in that case, iteration
// would look like this I think?
//
// prelude:
//   let (start, end) = (0, haystack.len());
//   let last_match = None;
// loop:
//   let m = next_match(start..end);
//   if m.is_empty() && Some(m.end()) == last_match {
//     start += 1;
//     m = next_match(start..end);
//   }
//   start = m.end();
//   last_match = m.end();
//   yield m
//
// Which isn't a ton simpler, although it does avoid needing to thread down a
// "UTF-8" parameter and avoids a loop, since when we get an overlapping match,
// all we need to do is start the next search at +1, which guarantees it won't
// overlap with the previous. And it does kind of seem like UTF-8-ness should
// be part of the regex engine anyway. The only reason it isn't currently is
// because I had always seen it as hard.
//
// But the way we would do it in the regex engine is basically exactly how
// we would do it above: if we get an empty match, we check if it splits a
// codepoint and if so, increment start position by 1 and try again.
//
// The other bummer is that this would mean BOTH the iterators and the regex
// find routines would have a 'm.is_empty()' check. Probably not a huge deal
// and is likely very branch-predictor friendly, given that empty matches are
// themselves rare. (And in general, we don't care too much about the perf of
// empty matches, because they are typically useless.)
//
// It also occurs to me that this might actually alter the leftmost-first match
// semantics in some way. That is, avoiding the splitting of the codepoint
// for an empty match can't be done in the DFA (without implementing some
// kind of crazy look-around), which means that we are skipping over that
// match at a higher level, which in turn means we might not be skipping over
// that match in the same way that the regex engine *would* if it could. If
// that's true, then we should be able to come with an example where the higher
// level skipping logic is inconsistent with leftmost-first matching... After
// thinking on it and trying some examples, I don't believe it's possible
// to get inconsistent matches. Namely, if an empty match is reported, it's
// because no earlier match is possible and also no longer match is possible.

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
pub struct TryOverlappingMatches<'h, F> {
    finder: F,
    search: Search<'h>,
}

impl<'c, 'h, F> TryOverlappingMatches<'h, F>
where
    F: FnMut(&Search<'h>) -> Result<Option<Match>, MatchError> + 'c,
{
    /// Create a new fallible overlapping matches iterator.
    ///
    /// The given `search` provides the parameters (including the haystack),
    /// while the `finder` represents a closure that calls the underlying regex
    /// engine. The closure may borrow any additional state that is needed,
    /// such as a prefilter or an overlapping's search's caller provided state.
    pub fn new(search: Search<'h>, finder: F) -> TryOverlappingMatches<'h, F> {
        TryOverlappingMatches { finder, search }
    }

    /// Like `new`, but boxes the given closure into a `dyn` object.
    ///
    /// This is useful when you can give up function inlining in favor of being
    /// able to write the type of the closure. This is often necessary for
    /// composition to work cleanly.
    pub fn boxed(
        search: Search<'h>,
        finder: F,
    ) -> TryOverlappingMatches<
        'h,
        Box<dyn FnMut(&Search<'h>) -> Result<Option<Match>, MatchError> + 'c>,
    > {
        TryOverlappingMatches::new(search, Box::new(finder))
    }

    /// Return an infallible version of this iterator.
    ///
    /// Any item yielded that corresponds to an error results in a panic.
    pub fn infallible(self) -> OverlappingMatches<'h, F> {
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

impl<'c, 'h, F> Iterator for TryOverlappingMatches<'h, F>
where
    F: FnMut(&Search<'h>) -> Result<Option<Match>, MatchError> + 'c,
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

impl<'h, F> core::fmt::Debug for TryOverlappingMatches<'h, F> {
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
pub struct OverlappingMatches<'h, F>(TryOverlappingMatches<'h, F>);

impl<'c, 'h, F> Iterator for OverlappingMatches<'h, F>
where
    F: FnMut(&Search<'h>) -> Result<Option<Match>, MatchError> + 'c,
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

impl<'h, F> core::fmt::Debug for OverlappingMatches<'h, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_tuple("OverlappingMatches").field(&self.0).finish()
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
pub struct TryHalfMatches<'h, F> {
    /// The regex engine execution function.
    finder: F,
    /// The search configuration.
    search: Search<'h>,
    /// Records the end offset of the most recent match. This is necessary to
    /// handle a corner case for preventing empty matches from overlapping with
    /// the ending bounds of a prior match.
    last_match_end: Option<usize>,
}

impl<'c, 'h, F> TryHalfMatches<'h, F>
where
    F: FnMut(&Search<'h>) -> Result<Option<HalfMatch>, MatchError> + 'c,
{
    /// Create a new fallible non-overlapping matches iterator.
    ///
    /// The given `search` provides the parameters (including the haystack),
    /// while the `finder` represents a closure that calls the underlying regex
    /// engine. The closure may borrow any additional state that is needed,
    /// such as a prefilter scanner.
    pub fn new(search: Search<'h>, finder: F) -> TryHalfMatches<'h, F> {
        TryHalfMatches { finder, search, last_match_end: None }
    }

    /// Like `new`, but boxes the given closure into a `dyn` object.
    ///
    /// This is useful when you can give up function inlining in favor of being
    /// able to write the type of the closure. This is often necessary for
    /// composition to work cleanly.
    pub fn boxed(
        search: Search<'h>,
        finder: F,
    ) -> TryHalfMatches<
        'h,
        Box<
            dyn FnMut(&Search<'h>) -> Result<Option<HalfMatch>, MatchError>
                + 'c,
        >,
    > {
        TryHalfMatches::new(search, Box::new(finder))
    }

    /// Return an infallible version of this iterator.
    ///
    /// Any item yielded that corresponds to an error results in a panic.
    pub fn infallible(self) -> HalfMatches<'h, F> {
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
        // Why not use 'self.search.step()' here? Well, that accounts for
        // UTF-8, which this iterator cannot handle in the general case because
        // we cannot detect every empty match.
        self.search.set_start(self.search.start().checked_add(1).unwrap());
        if self.search.is_done() {
            return None;
        }
        (self.finder)(&self.search).transpose()
    }
}

impl<'c, 'h, F> Iterator for TryHalfMatches<'h, F>
where
    F: FnMut(&Search<'h>) -> Result<Option<HalfMatch>, MatchError> + 'c,
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

impl<'h, F> core::fmt::Debug for TryHalfMatches<'h, F> {
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
pub struct HalfMatches<'h, F>(TryHalfMatches<'h, F>);

impl<'c, 'h, F> Iterator for HalfMatches<'h, F>
where
    F: FnMut(&Search<'h>) -> Result<Option<HalfMatch>, MatchError> + 'c,
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

impl<'h, F> core::fmt::Debug for HalfMatches<'h, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_tuple("Matches").field(&self.0).finish()
    }
}

/// An iterator over all overlapping half matches for a fallible search.
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
///
/// **WARNING:** Unlike other iterators that require both the start and end
/// bounds of a match, this iterator does not respect the [`Search::utf8`]
/// setting. Namely, if the underlying regex engine reports an empty match
/// that falls on an invalid UTF-8 boundary, then this iterator will yield it.
pub struct TryOverlappingHalfMatches<'h, F> {
    finder: F,
    search: Search<'h>,
}

impl<'c, 'h, F> TryOverlappingHalfMatches<'h, F>
where
    F: FnMut(&Search<'h>) -> Result<Option<HalfMatch>, MatchError> + 'c,
{
    /// Create a new fallible overlapping matches iterator.
    ///
    /// The given `search` provides the parameters (including the haystack),
    /// while the `finder` represents a closure that calls the underlying regex
    /// engine. The closure may borrow any additional state that is needed,
    /// such as a prefilter or an overlapping's search's caller provided state.
    pub fn new(
        search: Search<'h>,
        finder: F,
    ) -> TryOverlappingHalfMatches<'h, F> {
        TryOverlappingHalfMatches { finder, search }
    }

    /// Like `new`, but boxes the given closure into a `dyn` object.
    ///
    /// This is useful when you can give up function inlining in favor of being
    /// able to write the type of the closure. This is often necessary for
    /// composition to work cleanly.
    pub fn boxed(
        search: Search<'h>,
        finder: F,
    ) -> TryOverlappingHalfMatches<
        'h,
        Box<
            dyn FnMut(&Search<'h>) -> Result<Option<HalfMatch>, MatchError>
                + 'c,
        >,
    > {
        TryOverlappingHalfMatches::new(search, Box::new(finder))
    }

    /// Return an infallible version of this iterator.
    ///
    /// Any item yielded that corresponds to an error results in a panic.
    pub fn infallible(self) -> OverlappingHalfMatches<'h, F> {
        OverlappingHalfMatches(self)
    }
}

impl<'c, 'h, F> Iterator for TryOverlappingHalfMatches<'h, F>
where
    F: FnMut(&Search<'h>) -> Result<Option<HalfMatch>, MatchError> + 'c,
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
        self.search.set_start(m.offset());
        Some(Ok(m))
    }
}

impl<'h, F> core::fmt::Debug for TryOverlappingHalfMatches<'h, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("TryOverlappingHalfMatches")
            .field("finder", &"<closure>")
            .field("search", &self.search)
            .finish()
    }
}

/// An iterator over all overlapping half matches for an infallible search.
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
///
/// **WARNING:** Unlike other iterators that require both the start and end
/// bounds of a match, this iterator does not respect the [`Search::utf8`]
/// setting. Namely, if the underlying regex engine reports an empty match
/// that falls on an invalid UTF-8 boundary, then this iterator will yield it.
pub struct OverlappingHalfMatches<'h, F>(TryOverlappingHalfMatches<'h, F>);

impl<'c, 'h, F> Iterator for OverlappingHalfMatches<'h, F>
where
    F: FnMut(&Search<'h>) -> Result<Option<HalfMatch>, MatchError> + 'c,
{
    type Item = HalfMatch;

    #[inline]
    fn next(&mut self) -> Option<HalfMatch> {
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

impl<'h, F> core::fmt::Debug for OverlappingHalfMatches<'h, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_tuple("OverlappingHalfMatches").field(&self.0).finish()
    }
}
