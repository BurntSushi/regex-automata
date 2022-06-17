use crate::{
    nfa::thompson::Captures,
    util::{
        prefilter,
        search::{HalfMatch, Input, Match, MatchError},
    },
};

/// A searcher for advancing through all non-overlapping matches in a haystack.
///
/// This searcher encapsulates the logic required for finding all successive
/// non-overlapping matches in a haystack. In theory this would be something
/// like this:
///
/// 1. Setting the start position to `0`.
/// 2. Execute a regex search. If no match, end iteration.
/// 3. Report the match and set the start position to the end of the match.
/// 4. Go back to (2).
///
/// And if this were indeed the case, it's likely that `Searcher` wouldn't
/// exist. Unfortunately, because a regex may match the empty string, the above
/// logic won't work for all possible regexes. Namely, if an empty match is
/// found, then step (3) would set the start position of the search to the
/// position it was at. Thus, iteration would never end.
///
/// Instead, a `Searcher` knows how to detect these cases and forcefully
/// advance iteration in the case of an empty match that overlaps with a
/// previous match.
///
/// If you know that your regex cannot match any empty string, then the simple
/// algorithm described above will work correctly.
///
/// When possible, prefer the iterators defined on the regex engine you're
/// using. This type serves as the common implementation for the class of
/// "non-overlapping matches" iterator, and is thus a bit more unwieldy to use.
/// In particular, a `Searcher` is not itself an iterator. Instead, it provides
/// `advance` routines that permit moving the search along explicitly. It also
/// provides various routines, like [`Searcher::into_matches_iter`], that
/// accept a closure (representing how a regex engine executes a search) and
/// returns a conventional iterator.
///
/// The lifetime parameters, `'h` and `'p`, come from the [`Input`] type:
///
/// * `'h` is the lifetime of the underlying haystack.
/// * `'p` is the lifetime of the prefilter.
#[derive(Clone, Debug)]
pub struct Searcher<'h, 'p> {
    /// The search configuration.
    input: Input<'h, 'p>,
    /// Records the end offset of the most recent match. This is necessary to
    /// handle a corner case for preventing empty matches from overlapping with
    /// the ending bounds of a prior match.
    last_match_end: Option<usize>,
}

impl<'h, 'p> Searcher<'h, 'p> {
    /// Create a new fallible non-overlapping matches iterator.
    ///
    /// The given `input` provides the parameters (including the haystack),
    /// while the `finder` represents a closure that calls the underlying regex
    /// engine. The closure may borrow any additional state that is needed,
    /// such as a prefilter scanner.
    pub fn new(input: Input<'h, 'p>) -> Searcher<'h, 'p> {
        Searcher { input, last_match_end: None }
    }

    #[inline]
    pub fn advance_half<F>(&mut self, mut finder: F) -> Option<HalfMatch>
    where
        F: FnMut(&Input<'_, '_>) -> Result<Option<HalfMatch>, MatchError>,
    {
        match self.try_advance_half(finder) {
            Ok(m) => m,
            Err(err) => panic!(
                "unexpected regex half find error: {}\n\
                 to handle find errors, use 'try' or 'search' methods",
                err,
            ),
        }
    }

    #[inline]
    pub fn try_advance_half<F>(
        &mut self,
        mut finder: F,
    ) -> Result<Option<HalfMatch>, MatchError>
    where
        F: FnMut(&Input<'_, '_>) -> Result<Option<HalfMatch>, MatchError>,
    {
        let mut m = match finder(&self.input)? {
            None => return Ok(None),
            Some(m) => m,
        };
        if Some(m.offset()) == self.last_match_end {
            m = match self.handle_overlapping_empty_half_match(m, finder)? {
                None => return Ok(None),
                Some(m) => m,
            };
        }
        self.input.set_start(m.offset());
        self.last_match_end = Some(m.offset());
        Ok(Some(m))
    }

    #[inline]
    pub fn advance<F>(&mut self, mut finder: F) -> Option<Match>
    where
        F: FnMut(&Input<'_, '_>) -> Result<Option<Match>, MatchError>,
    {
        match self.try_advance(finder) {
            Ok(m) => m,
            Err(err) => panic!(
                "unexpected regex find error: {}\n\
                 to handle find errors, use 'try' or 'search' methods",
                err,
            ),
        }
    }

    #[inline]
    pub fn try_advance<F>(
        &mut self,
        mut finder: F,
    ) -> Result<Option<Match>, MatchError>
    where
        F: FnMut(&Input<'_, '_>) -> Result<Option<Match>, MatchError>,
    {
        let mut m = match finder(&self.input)? {
            None => return Ok(None),
            Some(m) => m,
        };
        if m.is_empty() {
            m = match self.handle_overlapping_empty_match(m, finder)? {
                None => return Ok(None),
                Some(m) => m,
            };
        }
        self.input.set_start(m.end());
        self.last_match_end = Some(m.end());
        Ok(Some(m))
    }

    #[inline]
    pub fn into_half_matches_iter<F>(
        self,
        finder: F,
    ) -> TryHalfMatchesIter<'h, 'p, F>
    where
        F: FnMut(&Input<'_, '_>) -> Result<Option<HalfMatch>, MatchError>,
    {
        TryHalfMatchesIter { it: self, finder }
    }

    #[inline]
    pub fn into_matches_iter<F>(self, finder: F) -> TryMatchesIter<'h, 'p, F>
    where
        F: FnMut(&Input<'_, '_>) -> Result<Option<Match>, MatchError>,
    {
        TryMatchesIter { it: self, finder }
    }

    #[inline]
    pub fn into_captures_iter<F>(
        self,
        caps: Captures,
        finder: F,
    ) -> TryCapturesIter<'h, 'p, F>
    where
        F: FnMut(&Input<'_, '_>, &mut Captures) -> Result<(), MatchError>,
    {
        TryCapturesIter { it: self, caps, finder }
    }

    /// Handles the special case of a match that begins where the previous
    /// match ended. Without this special handling, it'd be possible to get
    /// stuck where an empty match never results in forward progress. This
    /// also makes it more consistent with how presiding general purpose regex
    /// engines work.
    #[cold]
    #[inline(never)]
    fn handle_overlapping_empty_half_match<F>(
        &mut self,
        mut m: HalfMatch,
        mut finder: F,
    ) -> Result<Option<HalfMatch>, MatchError>
    where
        F: FnMut(&Input<'_, '_>) -> Result<Option<HalfMatch>, MatchError>,
    {
        // Since we are only here when 'm.offset()' matches the offset of the
        // last match, it follows that this must have been an empty match.
        // Since we both need to make progress *and* prevent overlapping
        // matches, we discard this match and advance the search by 1.
        //
        // Note that we do not prevent this iterator from returning an offset
        // that splits a codepoint. In order to do that, we need both the start
        // and end of a match, which is not available when we only have a half
        // match.
        //
        // We could prevent *any* match from being returned if it splits a
        // codepoint, but that seems like it's going too far.
        self.input.set_start(self.input.start().checked_add(1).unwrap());
        finder(&self.input)
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
    fn handle_overlapping_empty_match<F>(
        &mut self,
        mut m: Match,
        mut finder: F,
    ) -> Result<Option<Match>, MatchError>
    where
        F: FnMut(&Input<'_, '_>) -> Result<Option<Match>, MatchError>,
    {
        assert!(m.is_empty());
        if Some(m.end()) == self.last_match_end {
            self.input.set_start(self.input.start().checked_add(1).unwrap());
            m = match finder(&self.input)? {
                None => return Ok(None),
                Some(m) => m,
            };
        }
        Ok(Some(m))
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
///
/// The lifetime parameters, `'h` and `'p`, come from the [`Input`] type:
///
/// * `'h` is the lifetime of the underlying haystack.
/// * `'p` is the lifetime of the prefilter.
///
/// When possible, prefer the iterators defined on the regex engine you're
/// using. This type serves as the common implementation for the class of
/// "non-overlapping matches" iterator, and is thus a bit more unwieldy to use.
///
/// This iterator is created by [`Searcher::into_half_matches_iter`].
pub struct TryHalfMatchesIter<'h, 'p, F> {
    it: Searcher<'h, 'p>,
    finder: F,
}

impl<'h, 'p, F> TryHalfMatchesIter<'h, 'p, F> {
    /// Return an infallible version of this iterator.
    ///
    /// Any item yielded that corresponds to an error results in a panic. This
    /// is useful if your underlying regex engine is configured in a way that
    /// it is guaranteed to never return an error.
    pub fn infallible(self) -> HalfMatchesIter<'h, 'p, F> {
        HalfMatchesIter(self)
    }
}

impl<'h, 'p, F> Iterator for TryHalfMatchesIter<'h, 'p, F>
where
    F: FnMut(&Input<'_, '_>) -> Result<Option<HalfMatch>, MatchError>,
{
    type Item = Result<HalfMatch, MatchError>;

    #[inline]
    fn next(&mut self) -> Option<Result<HalfMatch, MatchError>> {
        self.it.try_advance_half(&mut self.finder).transpose()
    }
}

impl<'h, 'p, F> core::fmt::Debug for TryHalfMatchesIter<'h, 'p, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TryHalfMatchesIter")
            .field("it", &self.it)
            .field("finder", &"<closure>")
            .finish()
    }
}

/// An iterator over all non-overlapping half matches for an infallible search.
///
/// The iterator yields a [`HalfMatch`] value until no more matches could be
/// found.
///
/// The type parameters are as follows:
///
/// * `F` represents the type of a closure that executes the search.
///
/// The lifetime parameters, `'h` and `'p`, come from the [`Input`] type:
///
/// * `'h` is the lifetime of the underlying haystack.
/// * `'p` is the lifetime of the prefilter.
///
/// When possible, prefer the iterators defined on the regex engine you're
/// using. This type serves as the common implementation for the class of
/// "non-overlapping matches" iterator, and is thus a bit more unwieldy to use.
///
/// This iterator is created by [`Searcher::into_half_matches_iter`] and
/// then calling [`TryHalfMatchesIter::infallible`].
#[derive(Debug)]
pub struct HalfMatchesIter<'h, 'p, F>(TryHalfMatchesIter<'h, 'p, F>);

impl<'h, 'p, F> Iterator for HalfMatchesIter<'h, 'p, F>
where
    F: FnMut(&Input<'_, '_>) -> Result<Option<HalfMatch>, MatchError>,
{
    type Item = HalfMatch;

    #[inline]
    fn next(&mut self) -> Option<HalfMatch> {
        match self.0.next()? {
            Ok(m) => Some(m),
            Err(err) => panic!(
                "unexpected regex half find error: {}\n\
                 to handle find errors, use 'try' or 'search' methods",
                err,
            ),
        }
    }
}

/// An iterator over all non-overlapping matches for a fallible search.
///
/// The iterator yields a `Result<Match, MatchError>` value until no more
/// matches could be found.
///
/// The type parameters are as follows:
///
/// * `F` represents the type of a closure that executes the search.
///
/// The lifetime parameters, `'h` and `'p`, come from the [`Input`] type:
///
/// * `'h` is the lifetime of the underlying haystack.
/// * `'p` is the lifetime of the prefilter.
///
/// When possible, prefer the iterators defined on the regex engine you're
/// using. This type serves as the common implementation for the class of
/// "non-overlapping matches" iterator, and is thus a bit more unwieldy to use.
///
/// This iterator is created by [`Searcher::into_matches_iter`].
pub struct TryMatchesIter<'h, 'p, F> {
    it: Searcher<'h, 'p>,
    finder: F,
}

impl<'h, 'p, F> TryMatchesIter<'h, 'p, F> {
    /// Return an infallible version of this iterator.
    ///
    /// Any item yielded that corresponds to an error results in a panic. This
    /// is useful if your underlying regex engine is configured in a way that
    /// it is guaranteed to never return an error.
    pub fn infallible(self) -> MatchesIter<'h, 'p, F> {
        MatchesIter(self)
    }
}

impl<'h, 'p, F> Iterator for TryMatchesIter<'h, 'p, F>
where
    F: FnMut(&Input<'_, '_>) -> Result<Option<Match>, MatchError>,
{
    type Item = Result<Match, MatchError>;

    #[inline]
    fn next(&mut self) -> Option<Result<Match, MatchError>> {
        self.it.try_advance(&mut self.finder).transpose()
    }
}

impl<'h, 'p, F> core::fmt::Debug for TryMatchesIter<'h, 'p, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TryMatchesIter")
            .field("it", &self.it)
            .field("finder", &"<closure>")
            .finish()
    }
}

/// An iterator over all non-overlapping matches for an infallible search.
///
/// The iterator yields a [`Match`] value until no more matches could be found.
///
/// The type parameters are as follows:
///
/// * `F` represents the type of a closure that executes the search.
///
/// The lifetime parameters, `'h` and `'p`, come from the [`Input`] type:
///
/// * `'h` is the lifetime of the underlying haystack.
/// * `'p` is the lifetime of the prefilter.
///
/// When possible, prefer the iterators defined on the regex engine you're
/// using. This type serves as the common implementation for the class of
/// "non-overlapping matches" iterator, and is thus a bit more unwieldy to use.
///
/// This iterator is created by [`Searcher::into_matches_iter`] and
/// then calling [`TryMatchesIter::infallible`].
#[derive(Debug)]
pub struct MatchesIter<'h, 'p, F>(TryMatchesIter<'h, 'p, F>);

impl<'h, 'p, F> Iterator for MatchesIter<'h, 'p, F>
where
    F: FnMut(&Input<'_, '_>) -> Result<Option<Match>, MatchError>,
{
    type Item = Match;

    #[inline]
    fn next(&mut self) -> Option<Match> {
        match self.0.next()? {
            Ok(m) => Some(m),
            Err(err) => panic!(
                "unexpected regex find error: {}\n\
                 to handle find errors, use 'try' or 'search' methods",
                err,
            ),
        }
    }
}

/// An iterator over all non-overlapping captures for a fallible search.
///
/// The iterator yields a `Result<Captures, MatchError>` value until no more
/// matches could be found.
///
/// The type parameters are as follows:
///
/// * `F` represents the type of a closure that executes the search.
///
/// The lifetime parameters, `'h` and `'p`, come from the [`Input`] type:
///
/// * `'h` is the lifetime of the underlying haystack.
/// * `'p` is the lifetime of the prefilter.
///
/// When possible, prefer the iterators defined on the regex engine you're
/// using. This type serves as the common implementation for the class of
/// "non-overlapping matches" iterator, and is thus a bit more unwieldy to use.
///
/// This iterator is created by [`Searcher::into_captures_iter`].
pub struct TryCapturesIter<'h, 'p, F> {
    it: Searcher<'h, 'p>,
    caps: Captures,
    finder: F,
}

impl<'h, 'p, F> TryCapturesIter<'h, 'p, F> {
    /// Return an infallible version of this iterator.
    ///
    /// Any item yielded that corresponds to an error results in a panic. This
    /// is useful if your underlying regex engine is configured in a way that
    /// it is guaranteed to never return an error.
    pub fn infallible(self) -> CapturesIter<'h, 'p, F> {
        CapturesIter(self)
    }
}

impl<'h, 'p, F> Iterator for TryCapturesIter<'h, 'p, F>
where
    F: FnMut(&Input<'_, '_>, &mut Captures) -> Result<(), MatchError>,
{
    type Item = Result<Captures, MatchError>;

    #[inline]
    fn next(&mut self) -> Option<Result<Captures, MatchError>> {
        let TryCapturesIter { ref mut it, ref mut caps, ref mut finder } =
            *self;
        let result = it
            .try_advance(|input| {
                (finder)(input, caps)?;
                Ok(caps.get_match())
            })
            .transpose()?;
        match result {
            Ok(_) => Some(Ok(caps.clone())),
            Err(err) => Some(Err(err)),
        }
    }
}

impl<'h, 'p, F> core::fmt::Debug for TryCapturesIter<'h, 'p, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TryCapturesIter")
            .field("it", &self.it)
            .field("caps", &self.caps)
            .field("finder", &"<closure>")
            .finish()
    }
}

/// An iterator over all non-overlapping captures for an infallible search.
///
/// The iterator yields a [`Captures`] value until no more matches could be
/// found.
///
/// The type parameters are as follows:
///
/// * `F` represents the type of a closure that executes the search.
///
/// The lifetime parameters, `'h` and `'p`, come from the [`Input`] type:
///
/// * `'h` is the lifetime of the underlying haystack.
/// * `'p` is the lifetime of the prefilter.
///
/// When possible, prefer the iterators defined on the regex engine you're
/// using. This type serves as the common implementation for the class of
/// "non-overlapping matches" iterator, and is thus a bit more unwieldy to use.
///
/// This iterator is created by [`Searcher::into_captures_iter`] and then
/// calling [`TryCapturesIter::infallible`].
#[derive(Debug)]
pub struct CapturesIter<'h, 'p, F>(TryCapturesIter<'h, 'p, F>);

impl<'h, 'p, F> Iterator for CapturesIter<'h, 'p, F>
where
    F: FnMut(&Input<'_, '_>, &mut Captures) -> Result<(), MatchError>,
{
    type Item = Captures;

    #[inline]
    fn next(&mut self) -> Option<Captures> {
        match self.0.next()? {
            Ok(m) => Some(m),
            Err(err) => panic!(
                "unexpected regex captures error: {}\n\
                 to handle find errors, use 'try' or 'search' methods",
                err,
            ),
        }
    }
}
