use core::ops::Range;

use crate::util::{escape::DebugByte, id::PatternID, utf8};

// TODO: These were docs on a 'find' method before 'Search' supplanted it. They
// look useful as holistic docs for explaining the various parameters.
//
// * `pre` is a prefilter scanner that, when given, is used whenever the
// DFA enters its starting state. This is meant to speed up searches where
// one or a small number of literal prefixes are known.
// * `pattern_id` specifies a specific pattern in the DFA to run an
// anchored search for. If not given, then a search for any pattern is
// performed. For lazy DFAs, [`Config::starts_for_each_pattern`] must be
// enabled to use this functionality.
// * `start` and `end` permit searching a specific region of the haystack
// `bytes`. This is useful when implementing an iterator over matches
// within the same haystack, which cannot be done correctly by simply
// providing a subslice of `bytes`. (Because the existence of look-around
// operations such as `\b`, `^` and `$` need to take the surrounding
// context into account. This cannot be done if the haystack doesn't
// contain it.)
//
// TODO: More docs, with an example:
// ```
// use regex_automata::{hybrid::{dfa, regex}, MatchKind, Match};
//
// let pattern = r"[a-z]+";
// let haystack = "abc".as_bytes();
//
// // With leftmost-first semantics, we test "earliest" and "leftmost".
// let re = regex::Builder::new()
//     .dfa(dfa::Config::new().match_kind(MatchKind::LeftmostFirst))
//     .build(pattern)?;
// let mut cache = re.create_cache();
//
// // "earliest" searching isn't impacted by greediness
// let mut it = re.find_earliest_iter(&mut cache, haystack);
// assert_eq!(Some(Match::must(0, 0..1)), it.next());
// assert_eq!(Some(Match::must(0, 1..2)), it.next());
// assert_eq!(Some(Match::must(0, 2..3)), it.next());
// assert_eq!(None, it.next());
//
// // "leftmost" searching supports greediness (and non-greediness)
// let mut it = re.find_leftmost_iter(&mut cache, haystack);
// assert_eq!(Some(Match::must(0, 0..3)), it.next());
// assert_eq!(None, it.next());
//
// // For overlapping, we want "all" match kind semantics.
// let re = regex::Builder::new()
//     .dfa(dfa::Config::new().match_kind(MatchKind::All))
//     .build(pattern)?;
// let mut cache = re.create_cache();
//
// // In the overlapping search, we find all three possible matches
// // starting at the beginning of the haystack.
// let mut it = re.find_overlapping_iter(&mut cache, haystack);
// assert_eq!(Some(Match::must(0, 0..1)), it.next());
// assert_eq!(Some(Match::must(0, 0..2)), it.next());
// assert_eq!(Some(Match::must(0, 0..3)), it.next());
// assert_eq!(None, it.next());
//
// # Ok::<(), Box<dyn std::error::Error>>(())
// ```

#[derive(Clone)]
pub struct Search<'h> {
    haystack: &'h [u8],
    span: Span,
    pattern: Option<PatternID>,
    earliest: bool,
    utf8: bool,
}

impl<'h> Search<'h> {
    /// Create a new search configuration for the given haystack.
    #[inline]
    pub fn new<H: ?Sized + AsRef<[u8]>>(haystack: &'h H) -> Search<'h> {
        let haystack = haystack.as_ref();
        let span = Span { start: 0, end: haystack.len() };
        Search { haystack, span, pattern: None, earliest: false, utf8: true }
    }

    /// Set the span for this search.
    ///
    /// This routine does not panic if the span given is not a valid range for
    /// this search's haystack. If this search is run with an invalid range,
    /// then the most likely outcome is that the actual search execution will
    /// panic.
    ///
    /// This constructor is generic over how a span is provided. While
    /// a [`Span`] may be given directly, one may also provide a
    /// `std::ops::Range<usize>`. To provide anything supported by range
    /// syntax, use the [`Search::range`] method.
    #[inline]
    pub fn span<S: Into<Span>>(self, span: S) -> Search<'h> {
        Search { span: span.into(), ..self }
    }

    /// Like `Search::span`, but accepts any range instead.
    ///
    /// This routine does not panic if the span given is not a valid range for
    /// this search's haystack. If this search is run with an invalid range,
    /// then the most likely outcome is that the actual search execution will
    /// panic.
    ///
    /// # Panics
    ///
    /// This routine will panic if the given range could not be converted
    /// to a valid [`Range`]. For example, this would panic when given
    /// `0..=usize::MAX` since it cannot be represented using a half-open
    /// interval in terms of `usize`.
    #[inline]
    pub fn range<R: core::ops::RangeBounds<usize>>(
        self,
        range: R,
    ) -> Search<'h> {
        use core::ops::Bound;

        // It's a little weird to convert ranges into spans, and then spans
        // back into ranges when we actually slice the haystack. Because
        // of that process, we always represent everything as a half-open
        // internal. Therefore, handling things like m..=n is a little awkward.
        let start = match range.start_bound() {
            Bound::Included(&i) => i,
            // Can this case ever happen? Range syntax doesn't support it...
            Bound::Excluded(&i) => i.checked_add(1).unwrap(),
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&i) => i.checked_add(1).unwrap(),
            Bound::Excluded(&i) => i,
            Bound::Unbounded => self.haystack().len(),
        };
        self.span(Span { start, end })
    }

    /// Set the pattern to search for, if supported.
    ///
    /// When given, the an anchored search for only the specified pattern will
    /// be executed. If not given, then the search will look for any pattern
    /// that matches. (Whether that search is anchored or not depends on the
    /// configuration of your regex engine and, ultimately, the pattern
    /// itself.)
    ///
    /// If a pattern ID is given and a regex engine doesn't support searching
    /// by a specific pattern, then the regex engine must panic.
    #[inline]
    pub fn pattern(self, pattern: Option<PatternID>) -> Search<'h> {
        Search { pattern, ..self }
    }

    // BREADCRUMBS: Continue auditing and documenting 'Search'.
    //
    // The inconsistent naming is unfortunate. e.g., 'start()'/'end()' are
    // getters despite things like 'pattern()' being setters. And to make things
    // worse, we also have 'set_{start,end}'.
    //
    // I guess 'start()' and 'end()' really should be 'get_{start,end}', but
    // that does make them more verbose. Their terseness is convenience inside
    // the implementation of search routines...
    //
    // The setters which mutate 'Search' are probably necessary to keep around
    // to avoid forcing callers to create copies of 'Search' just to update the
    // search position. Should we thus make setters for everything else too?

    /// Whether to execute an "earliest" search or not.
    ///
    /// When running a non-overlapping search, an "earliest" search will return
    /// the match location as early as possible. For example, given a pattern
    /// of `foo[0-9]+` and a haystack of `foo12345`, a normal leftmost search
    /// will return `foo12345` as a match. But an "earliest" search for regex
    /// engines that support "earliest" semantics will return `foo1` as a
    /// match, since as soon as the first digit following `foo` is seen, it is
    /// known to have found a match.
    ///
    /// Note that "earliest" semantics generally depend on the regex engine.
    /// Different regex engines may determine there is a match at different
    /// points. So there is no guarantee that "earliest" matches will always
    /// return the same offsets for all regex engines. The "earliest" notion
    /// is really about when the particular regex engine determines there is
    /// a match. This is often useful for implementing "did a match occur or
    /// not" predicates, but sometimes the offset is useful as well.
    ///
    /// This is disabled by default.
    #[inline]
    pub fn earliest(self, yes: bool) -> Search<'h> {
        Search { earliest: yes, ..self }
    }

    #[inline]
    pub fn utf8(self, yes: bool) -> Search<'h> {
        Search { utf8: yes, ..self }
    }

    /// Return a borrow of the underlying haystack.
    #[inline]
    pub fn haystack(&self) -> &[u8] {
        self.haystack
    }

    /// Return the start position of this search.
    ///
    /// This is a convenience routine for `search.get_span().start()`.
    #[inline]
    pub fn get_start(&self) -> usize {
        self.get_span().start
    }

    /// Return the end position of this search.
    ///
    /// This is a convenience routine for `search.get_span().end()`.
    #[inline]
    pub fn get_end(&self) -> usize {
        self.get_span().end
    }

    /// Set the span for this search configuration.
    ///
    /// This is like the [`Search::span`] method, except this mutates the
    /// span in place.
    #[inline]
    pub fn set_span(&mut self, span: Span) {
        self.span = span;
    }

    /// Set the starting offset for the span for this search configuration.
    ///
    /// This is a convenience routine for only mutating the start of a span
    /// without having to set the entire span.
    #[inline]
    pub fn set_start(&mut self, start: usize) {
        self.span.start = start;
    }

    /// Set the ending offset for the span for this search configuration.
    ///
    /// This is a convenience routine for only mutating the end of a span
    /// without having to set the entire span.
    #[inline]
    pub fn set_end(&mut self, end: usize) {
        self.span.end = end;
    }

    /// Return the span for this search configuration.
    ///
    /// If one was not explicitly set, then the span corresponds to the entire
    /// range of the haystack.
    #[inline]
    pub fn get_span(&self) -> Span {
        self.span
    }

    /// Return the span as a range for this search configuration.
    ///
    /// If one was not explicitly set, then the span corresponds to the entire
    /// range of the haystack.
    #[inline]
    pub fn get_range(&self) -> Range<usize> {
        self.get_span().range()
    }

    /// Return the pattern ID for this search configuration, if one was set.
    #[inline]
    pub fn get_pattern(&self) -> Option<PatternID> {
        self.pattern
    }

    /// Return whether this search should execute in "earliest" mode.
    #[inline]
    pub fn get_earliest(&self) -> bool {
        self.earliest
    }

    /// Return whether this search should execute in "UTF-8" mode.
    #[inline]
    pub fn get_utf8(&self) -> bool {
        self.utf8
    }

    /// Return true if and only if this search can never return any other
    /// matches.
    ///
    /// For example, if the start position of this search is greater than the
    /// end position of the search.
    #[inline]
    pub fn is_done(&self) -> bool {
        self.get_span().start > self.get_span().end
    }

    /// Returns true if and only if the given offset in this search's haystack
    /// falls on a valid UTF-8 encoded codepoint boundary.
    ///
    /// If the haystack is not valid UTF-8, then the behavior of this routine
    /// is unspecified.
    #[inline]
    pub fn is_char_boundary(&self, offset: usize) -> bool {
        utf8::is_boundary(self.haystack(), offset)
    }

    /// This skips any empty matches that split a codepoint when this search's
    /// "utf8" option is enabled. The match given should be the initial match
    /// found, and 'find' should be a closure that can execute a regex search.
    ///
    /// We don't export this routine because it could be quite confusing. Folks
    /// might use this to call another regex engine's find routine that already
    /// calls this internally. Plus, its implementation can be written entirely
    /// using existing public APIs.
    ///
    /// N.B. This is written as a non-inlineable cold function that accepts
    /// a pre-existing match because it generally leads to better codegen in
    /// my experience. Namely, we could write a routine that doesn't accept
    /// a pre-existing match and just does the initial search for you. But
    /// doing it this way forcefully separates the hot path from the handling
    /// of pathological cases. That is, one can guard calls to this with
    /// 'm.is_empty()', even though it isn't necessary for correctness.
    #[cold]
    #[inline(never)]
    pub(crate) fn skip_empty_utf8_splits<F>(
        &self,
        mut m: Match,
        mut find: F,
    ) -> Result<Option<Match>, MatchError>
    where
        F: FnMut(&Search<'_>) -> Result<Option<Match>, MatchError>,
    {
        if !self.get_utf8() || !m.is_empty() {
            return Ok(Some(m));
        }
        let mut search = self.clone();
        while m.is_empty() && !search.is_char_boundary(m.end()) {
            search.set_start(search.get_start().checked_add(1).unwrap());
            m = match find(&search)? {
                None => return Ok(None),
                Some(m) => m,
            };
        }
        Ok(Some(m))
    }
}

impl<'h> core::fmt::Debug for Search<'h> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use crate::util::escape::DebugHaystack;

        f.debug_struct("Search")
            .field("span", &self.span)
            .field("pattern", &self.pattern)
            .field("earliest", &self.earliest)
            .field("utf8", &self.utf8)
            .field("haystack", &DebugHaystack(self.haystack()))
            .finish()
    }
}

/// A representation of a span reported by a regex engine.
///
/// A span corresponds to the starting and ending _byte offsets_ of a
/// contiguous region of bytes. The starting offset is inclusive while the
/// ending offset is exclusive. That is, a span is a half-open interval.
///
/// A span is used to report the offsets of a match, but it is also used to
/// convey which region of a haystack should be searched via routines like
/// [`Search::span`].
///
/// This is basically equivalent to a `std::ops::Range<usize>`, except this
/// type implements `Copy` which makes it more ergonomic to use in the context
/// of this crate. Like a range, this implements `Index` for `[u8]` and `str`,
/// and `IndexMut` for `[u8]`. For convenience, this also impls `From<Range>`,
/// which means things like `Span::from(5..10)` work.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Span {
    /// The start offset of the span, inclusive.
    pub start: usize,
    /// The end offset of the span, exclusive.
    pub end: usize,
}

impl Span {
    /// Returns this span as a range.
    #[inline]
    pub fn range(&self) -> Range<usize> {
        Range::from(*self)
    }

    /// Returns true when this span is empty. That is, when `start >= end`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    /// Returns true when the given offset is contained within this span.
    ///
    /// Note that an empty span contains no offsets and will always return
    /// false.
    #[inline]
    pub fn contains(&self, offset: usize) -> bool {
        !self.is_empty() && self.start <= offset && offset <= self.end
    }
}

impl core::ops::Index<Span> for [u8] {
    type Output = [u8];

    #[inline]
    fn index(&self, index: Span) -> &[u8] {
        &self[index.range()]
    }
}

impl core::ops::IndexMut<Span> for [u8] {
    #[inline]
    fn index_mut(&mut self, index: Span) -> &mut [u8] {
        &mut self[index.range()]
    }
}

impl core::ops::Index<Span> for str {
    type Output = str;

    #[inline]
    fn index(&self, index: Span) -> &str {
        &self[index.range()]
    }
}

impl From<Range<usize>> for Span {
    #[inline]
    fn from(range: Range<usize>) -> Span {
        Span { start: range.start, end: range.end }
    }
}

impl From<Span> for Range<usize> {
    #[inline]
    fn from(span: Span) -> Range<usize> {
        Range { start: span.start, end: span.end }
    }
}

impl PartialEq<Range<usize>> for Span {
    #[inline]
    fn eq(&self, range: &Range<usize>) -> bool {
        self.start == range.start && self.end == range.end
    }
}

impl PartialEq<Span> for Range<usize> {
    #[inline]
    fn eq(&self, span: &Span) -> bool {
        self.start == span.start && self.end == span.end
    }
}

/// A representation of "half" of a match reported by a DFA.
///
/// This is called a "half" match because it only includes the end location (or
/// start location for a reverse search) of a match. This corresponds to the
/// information that a single DFA scan can report. Getting the other half of
/// the match requires a second scan with a reversed DFA.
///
/// A half match also includes the pattern that matched. The pattern is
/// identified by an ID, which corresponds to its position (starting from `0`)
/// relative to other patterns used to construct the corresponding DFA. If only
/// a single pattern is provided to the DFA, then all matches are guaranteed to
/// have a pattern ID of `0`.
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

    /// Create a new half match from a pattern ID and a byte offset.
    ///
    /// This is like [`HalfMatch::new`], but accepts a `usize` instead of a
    /// [`PatternID`]. This panics if the given `usize` is not representable
    /// as a `PatternID`.
    #[inline]
    pub fn must(pattern: usize, offset: usize) -> HalfMatch {
        HalfMatch::new(PatternID::new(pattern).unwrap(), offset)
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

/// A representation of a match reported by a regex engine.
///
/// A match has two essential pieces of information: the [`PatternID`] that
/// matches, and the [`Span`] of the match in a haystack.
///
/// The pattern is identified by an ID, which corresponds to its position
/// (starting from `0`) relative to other patterns used to construct the
/// corresponding regex engine. If only a single pattern is provided, then all
/// matches are guaranteed to have a pattern ID of `0`.
///
/// Every match reported by a regex engine guarantees that its span has its
/// start offset as less than or equal to its end offset.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Match {
    /// The pattern ID.
    pattern: PatternID,
    /// The underlying match span.
    span: Span,
}

impl Match {
    /// Create a new match from a pattern ID and a span.
    ///
    /// This constructor is generic over how a span is provided. While
    /// a [`Span`] may be given directly, one may also provide a
    /// `std::ops::Range<usize>`.
    ///
    /// # Panics
    ///
    /// This panics if `end < start`.
    ///
    /// # Example
    ///
    /// This shows how to create a match for the first pattern in a regex
    /// object using convenient range syntax.
    ///
    /// ```
    /// use regex_automata::{Match, PatternID};
    ///
    /// let m = Match::new(PatternID::ZERO, 5..10);
    /// assert_eq!(0, m.pattern().as_usize());
    /// assert_eq!(5, m.start());
    /// assert_eq!(10, m.end());
    /// ```
    #[inline]
    pub fn new<S: Into<Span>>(pattern: PatternID, span: S) -> Match {
        let span = span.into();
        assert!(span.start <= span.end, "invalid match span");
        Match { pattern, span }
    }

    /// Create a new match from a pattern ID and a byte offset span.
    ///
    /// This constructor is generic over how a span is provided. While
    /// a [`Span`] may be given directly, one may also provide a
    /// `std::ops::Range<usize>`.
    ///
    /// This is like [`Match::new`], but accepts a `usize` instead of a
    /// [`PatternID`]. This panics if the given `usize` is not representable
    /// as a `PatternID`.
    ///
    /// # Panics
    ///
    /// This panics if `end < start` or if `pattern > PatternID::MAX`.
    ///
    /// # Example
    ///
    /// This shows how to create a match for the third pattern in a regex
    /// object using convenient range syntax.
    ///
    /// ```
    /// use regex_automata::Match;
    ///
    /// let m = Match::must(3, 5..10);
    /// assert_eq!(3, m.pattern().as_usize());
    /// assert_eq!(5, m.start());
    /// assert_eq!(10, m.end());
    /// ```
    #[inline]
    pub fn must<S: Into<Span>>(pattern: usize, span: S) -> Match {
        Match::new(PatternID::must(pattern), span)
    }

    /// Returns the ID of the pattern that matched.
    ///
    /// The ID of a pattern is derived from the position in which it was
    /// originally inserted into the corresponding regex engine. The first
    /// pattern has identifier `0`, and each subsequent pattern is `1`, `2` and
    /// so on.
    #[inline]
    pub fn pattern(&self) -> PatternID {
        self.pattern
    }

    /// The starting position of the match.
    ///
    /// This is a convenience routine for `Match::span().start`.
    #[inline]
    pub fn start(&self) -> usize {
        self.span().start
    }

    /// The ending position of the match.
    ///
    /// This is a convenience routine for `Match::span().end`.
    #[inline]
    pub fn end(&self) -> usize {
        self.span().end
    }

    /// Returns the match span as a range.
    ///
    /// This is a convenience routine for `Match::span().range()`.
    #[inline]
    pub fn range(&self) -> core::ops::Range<usize> {
        self.span().range()
    }

    /// Returns the span for this match.
    #[inline]
    pub fn span(&self) -> Span {
        self.span
    }

    /// Returns true when the span in this match is empty.
    ///
    /// An empty match can only be returned when the regex itself can match
    /// the empty string.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.span().is_empty()
    }
}

/// A set of `PatternID`s.
///
/// A set of pattern identifiers is useful for recording which patterns have
/// matched a particular haystack. A pattern set _only_ includes pattern
/// identifiers. It does not include offset information.
///
/// # Example
///
/// This shows basic usage of a set.
///
/// ```
/// use regex_automata::{PatternSet, PatternID};
///
/// let pid1 = PatternID::must(5);
/// let pid2 = PatternID::must(8);
/// // Create a new empty set.
/// let mut set = PatternSet::new(10);
/// // Insert pattern IDs.
/// set.insert(pid1);
/// set.insert(pid2);
/// // Test membership.
/// assert!(set.contains(pid1));
/// assert!(set.contains(pid2));
/// // Get all members.
/// assert_eq!(
///     vec![5, 8],
///     set.iter().map(|p| p.as_usize()).collect::<Vec<usize>>(),
/// );
/// // Clear the set.
/// set.clear();
/// // Test that it is indeed empty.
/// assert!(set.is_empty());
/// ```
#[cfg(feature = "alloc")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PatternSet {
    /// The number of patterns set to 'true' in this set.
    len: usize,
    /// A map from PatternID to boolean of whether a pattern matches or not.
    ///
    /// This should probably be a bitset, but it's probably unlikely to matter
    /// much in practice.
    ///
    /// The main downside of this representation (and similarly for a bitset)
    /// is that iteration scales with the capacity of the set instead of
    /// the length of the set. This doesn't seem likely to be a problem in
    /// practice.
    which: alloc::boxed::Box<[bool]>,
}

#[cfg(feature = "alloc")]
impl PatternSet {
    /// Create a new set of pattern identifiers with the given capacity.
    ///
    /// The given capacity typically corresponds to (at least) the number of
    /// patterns in a compiled regex object.
    ///
    /// # Panics
    ///
    /// This panics if the given capacity exceeds [`PatternID::LIMIT`].
    pub fn new(capacity: usize) -> PatternSet {
        assert!(
            capacity <= PatternID::LIMIT,
            "pattern set capacity exceeds limit of {}",
            PatternID::LIMIT,
        );
        PatternSet {
            len: 0,
            which: alloc::vec![false; capacity].into_boxed_slice(),
        }
    }

    /// Clear this set such that it contains no pattern IDs.
    pub fn clear(&mut self) {
        self.len = 0;
        for matched in self.which.iter_mut() {
            *matched = false;
        }
    }

    /// Return true if and only if the given pattern identifier is in this set.
    ///
    /// # Panics
    ///
    /// This panics if `pid` exceeds the capacity of this set.
    pub fn contains(&self, pid: PatternID) -> bool {
        self.which[pid]
    }

    /// Insert the given pattern identifier into this set.
    ///
    /// If the pattern identifier is already in this set, then this is a no-op.
    ///
    /// # Panics
    ///
    /// This panics if `pid` exceeds the capacity of this set.
    pub fn insert(&mut self, pid: PatternID) {
        if self.which[pid] {
            return;
        }
        self.len += 1;
        self.which[pid] = true;
    }

    /// Remove the given pattern identifier from this set.
    ///
    /// If the pattern identifier was not previously in this set, then this
    /// does not change the set and returns `false`.
    ///
    /// # Panics
    ///
    /// This panics if `pid` exceeds the capacity of this set.
    pub fn remove(&mut self, pid: PatternID) -> bool {
        if !self.which[pid] {
            return false;
        }
        self.len -= 1;
        self.which[pid] = false;
        true
    }

    /// Return true if and only if this set has no pattern identifiers in it.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return true if and only if this set has the maximum number of pattern
    /// identifiers in the set. This occurs precisely when `PatternSet::len()
    /// == PatternSet::capacity()`.
    ///
    /// This particular property is useful to test because it may allow one to
    /// stop a search earlier than you might otherwise. Namely, if a search is
    /// only reporting which patterns match a haystack and if you know all of
    /// the patterns match at a given point, then there's no new information
    /// that can be learned by continuing the search. (Because a pattern set
    /// does not keep track of offset information.)
    pub fn is_full(&self) -> bool {
        self.len() == self.capacity()
    }

    /// Returns the total number of pattern identifiers in this set.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the total number of pattern identifiers that may be stored
    /// in this set.
    ///
    /// This is guaranteed to be less than or equal to [`PatternID::LIMIT`].
    ///
    /// Typically, the capacity of a pattern set matches the number of patterns
    /// in a regex object with which you are searching.
    pub fn capacity(&self) -> usize {
        self.which.len()
    }

    /// Returns an iterator over all pattern identifiers in this set.
    ///
    /// The iterator yields pattern identifiers in ascending order, starting
    /// at zero.
    pub fn iter(&self) -> PatternSetIter<'_> {
        PatternSetIter { it: self.which.iter().enumerate() }
    }
}

/// An iterator over all pattern identifiers in a [`PatternSet`].
///
/// The lifetime parameter `'a` refers to the lifetime of the pattern set being
/// iterated over.
///
/// This iterator is created by the [`PatternSet::iter`] method.
#[cfg(feature = "alloc")]
#[derive(Debug)]
pub struct PatternSetIter<'a> {
    it: core::iter::Enumerate<core::slice::Iter<'a, bool>>,
}

#[cfg(feature = "alloc")]
impl<'a> Iterator for PatternSetIter<'a> {
    type Item = PatternID;

    fn next(&mut self) -> Option<PatternID> {
        while let Some((index, &yes)) = self.it.next() {
            if yes {
                // Only valid 'PatternID' values can be inserted into the set
                // and construction of the set panics if the capacity would
                // permit storing invalid pattern IDs. Thus, 'yes' is only true
                // precisely when 'index' corresponds to a valid 'PatternID'.
                return Some(PatternID::new_unchecked(index));
            }
        }
        None
    }
}

/// The kind of match semantics to use for a regex pattern.
///
/// The default match kind is `LeftmostFirst`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatchKind {
    /// Report all possible matches.
    All,
    /// Report only the leftmost matches. When multiple leftmost matches exist,
    /// report the match corresponding to the part of the regex that appears
    /// first in the syntax.
    LeftmostFirst,
    /// Hints that destructuring should not be exhaustive.
    ///
    /// This enum may grow additional variants, so this makes sure clients
    /// don't count on exhaustive matching. (Otherwise, adding a new variant
    /// could break existing code.)
    #[doc(hidden)]
    __Nonexhaustive,
    // There is prior art in RE2 that shows that we should be able to add
    // LeftmostLongest too. The tricky part of it is supporting ungreedy
    // repetitions. Instead of treating all NFA states as having equivalent
    // priority (as in 'All') or treating all NFA states as having distinct
    // priority based on order (as in 'LeftmostFirst'), we instead group NFA
    // states into sets, and treat members of each set as having equivalent
    // priority, but having greater priority than all following members
    // of different sets.
    //
    // However, it's not clear whether it's really worth adding this. After
    // all, leftmost-longest can be emulated when using literals by using
    // leftmost-first and sorting the literals by length in descending order.
    // However, this won't work for arbitrary regexes. e.g., `\w|\w\w` will
    // always match `a` in `ab` when using leftmost-first, but leftmost-longest
    // would match `ab`.
}

impl MatchKind {
    #[cfg(feature = "alloc")]
    pub(crate) fn continue_past_first_match(&self) -> bool {
        *self == MatchKind::All
    }
}

impl Default for MatchKind {
    fn default() -> MatchKind {
        MatchKind::LeftmostFirst
    }
}

/// An error indicating that a search stopped before reporting whether a
/// match exists or not.
///
/// To be very clear, this error type implies that one cannot assume that no
/// matches occur, since the search stopped before completing. That is, if
/// you're looking for information about where a search determined that no
/// match can occur, then this error type does *not* give you that. (Indeed, at
/// the time of writing, if you need such a thing, you have to write your own
/// search routine.)
///
/// Normally, when one searches for something, the response is either an
/// affirmative "it was found at this location" or a negative "not found at
/// all." However, in some cases, a regex engine can be configured to stop its
/// search before concluding whether a match exists or not. When this happens,
/// it may be important for the caller to know why the regex engine gave up and
/// where in the input it gave up at. This error type exposes the 'why' and the
/// 'where.'
///
/// For example, the DFAs provided by this library generally cannot correctly
/// implement Unicode word boundaries. Instead, they provide an option to
/// eagerly support them on ASCII text (since Unicode word boundaries are
/// equivalent to ASCII word boundaries when searching ASCII text), but will
/// "give up" if a non-ASCII byte is seen. In such cases, one is usually
/// required to either report the failure to the caller (unergonomic) or
/// otherwise fall back to some other regex engine (ergonomic, but potentially
/// costly).
///
/// More generally, some regex engines offer the ability for callers to specify
/// certain bytes that will trigger the regex engine to automatically quit if
/// they are seen.
///
/// Still yet, there may be other reasons for a failed match. For example,
/// the hybrid DFA provided by this crate can be configured to give up if it
/// believes that it is not efficient. This in turn permits callers to choose a
/// different regex engine.
///
/// # Advice
///
/// While this form of error reporting adds complexity, it is generally
/// possible for callers to configure regex engines to never give up on a
/// search, and thus never return an error. Indeed, the default configuration
/// for every regex engine in this crate is such that they will never stop
/// searching early. Therefore, the only way to get a match error is if the
/// regex engine is explicitly configured to do so. Options that enable this
/// behavior document the new error conditions they imply.
///
/// For example, regex engines in the `dfa` sub-module will only report
/// `MatchError::Quit` if instructed by either
/// [enabling Unicode word boundaries](crate::dfa::dense::Config::unicode_word_boundary)
/// or by
/// [explicitly specifying one or more quit bytes](crate::dfa::dense::Config::quit).
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum MatchError {
    // A previous iteration of this error type specifically encoded "did not
    // match" as a None variant. Instead of fallible regex searches returning
    // Result<Option<Match>, MatchError>, they would return the simpler
    // Result<Match, MatchError>. The appeal of this is the simpler return
    // type. The inherent problem, though, is that "did not match" is not
    // actually an error case. It's an expected behavior of a regex search
    // and is therefore typically handled differently than a real error that
    // prevents one from knowing whether a match occurs at all. Thus, the
    // simpler return type often requires explicit case analysis to deal with
    // the None variant. More to the point, the iteration protocol for the
    // simpler return type was quite awkward, because the iteration protocol
    // really wants an Option<Match> and cannot deal with the None variant
    // inside of the error type.
    /// The search saw a "quit" byte at which it was instructed to stop
    /// searching.
    Quit {
        /// The "quit" byte that was observed that caused the search to stop.
        byte: u8,
        /// The offset at which the quit byte was observed.
        offset: usize,
    },
    /// The search, based on heuristics, determined that it would be better
    /// to stop, typically to provide the caller an opportunity to use an
    /// alternative regex engine.
    ///
    /// Currently, the only way for this to occur is via the lazy DFA and
    /// only when it is configured to do so (it will not return this error by
    /// default).
    GaveUp {
        /// The offset at which the search stopped. This corresponds to the
        /// position immediately following the last byte scanned.
        offset: usize,
    },
}

#[cfg(feature = "std")]
impl std::error::Error for MatchError {}

impl core::fmt::Display for MatchError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match *self {
            MatchError::Quit { byte, offset } => write!(
                f,
                "quit search after observing byte {:?} at offset {}",
                DebugByte(byte),
                offset,
            ),
            MatchError::GaveUp { offset } => {
                write!(f, "gave up searching at offset {}", offset)
            }
        }
    }
}
