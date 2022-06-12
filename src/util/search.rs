use core::ops::{Range, RangeBounds};

use crate::util::{
    escape::DebugByte, id::PatternID, prefilter::Prefilter, utf8,
};

// TODO: We should litigate whether we can stuff prefilters into a 'Search'.
// It certainly feels like prefilters *belong* here, since they are really
// a parameter of a search. And it would simplify a lot of annoying method
// signatures.
//
// The main point of litigation is that, currently, the prefilter
// infrastructure uses a "scanner" that wraps the actual prefilter. The scanner
// is supposed to record state about how "effective" the prefilter is, and then
// dynamically disable it if it's doing poorly. That requires mutation and a
// 'Search' is, ideally, immutable.
//
// But, the question is, do we really need to track whether a prefilter is
// effective or not? If not, then we can just stuff a '&dyn Prefilter' into a
// 'Search' and be done with it. If so, we either need to make 'Search' mutable
// with respect to regex search implementations, or we need some other way of
// determining whether a prefilter is effective or not.
//
// Branching off of "effective or not," there's a key advantage to the
// 'scanner' approach: the caller controls the "scope" of when a prefilter is
// determined to be effective or not. e.g., One scanner for an entire iterator.
// If instead effectiveness were determined a layer down---say in the regex
// search itself---then there really wouldn't be an opportunity for increasing
// scope. It would need to be re-litigated for each search.
//
// ... unless, we went with my idea about a "match" context. That is, every
// search has a 'Search' configuration that represents the input, some mutable
// scratch space (if necessary) and an output 'Match' context. That match
// context includes the match (if one exists), but it might also include other
// information, like where the search stopped if it failed. Or... prefilter
// effectiveness! Because then the caller could control the scope of the
// prefilter in that case and it would make sense for it to be mutable.
//
// How inconvenient is this though? The "match" context would only be used on
// the lower level search APIs. And we're already doing something *very* close
// to a match context with the PikeVM's 'Captures' API.
//
// Speaking of which, how do 'Captures' fit in with "match context"? The latter
// should be generic across all regex engines. So do we have both a match
// context and a captures output for the PikeVM? So our choices are:
//
// 1. A 'Captures' contains a match context.
// 2. A match context is generic over anything that can produce a 'Match'.
//    (Which a 'Captures' can.)
// 3. Just keep 'Captures' and "match context" entirely separate.
//
// My instinct is that (2) feels like the "most" correct, but I don't like the
// generic machinery that it inspires. It *does* provide some extensibility for
// other regex engines doing different things, though.
//
// Overall I do really like the match context idea... It provides for a lot of
// flexibility, because it means the "output" of a regex search doesn't need
// to be confined to "match at location." And we can add to it in the future
// too.
//
// So maybe we stuff a prefilter into a 'Search' for now, get rid of scanner,
// and then do the "match context" thing? And if we decide we need to be able
// to track the effectiveness of a prefilter, we can add that to the match
// context later!

/// The parameters for a regex search.
///
/// While most regex engines in this crate expose a convenience `find`-like
/// routine that accepts a haystack and returns a match if one was found, it
/// turns out that regex searches have a lot of parameters. The `find`-like
/// methods represent the common use case, while this `Search` type represents
/// the full configurability of a regex search. That configurability includes:
///
/// * Search only a substring of a haystack, while taking the broader context
/// into account for resolving look-around assertions.
/// * Whether to use a prefilter for the search or not.
/// * Indicating whether to search for all patterns in a regex object, or to
/// only search for one pattern in particular.
/// * Whether to report a match as early as possible.
/// * Whether to report matches that might split a codepoint in valid UTF-8.
///
/// All of these parameters, except for the haystack, have sensible default
/// values. This means that the minimal search configuration is simply a call
/// to [`Search::new`] with your haystack. Setting any other parameter is
/// optional.
///
/// The API of `Search` is split into a few different parts:
///
/// * A builder-like API that transforms a `Search` by value. Examples:
/// [`Search::span`] and [`Search::prefilter`].
/// * A setter API that permits mutating parameters in place. Examples:
/// [`Search::set_span`] and [`Search::set_prefilter`].
/// * A getter API that permits retrieving any of the search parameters.
/// Examples: [`Search::get_span`] and [`Search::get_prefilter`].
/// * A few convenience getter routines that don't conform to the above naming
/// pattern due to how common they are. Examples: [`Search::haystack`],
/// [`Search::start`] and [`Search::end`].
/// * Miscellaneous predicates and other helper routines that are useful
/// in some contexts. Examples: [`Search::is_char_boundary`].
///
/// A `Search` exposes so much because it is meant to be used by both
/// callers of regex engines _and_ implementors of regex engines.
///
/// The lifetime parameters have the following meaning:
///
/// * `'h` refers to the lifetime of the haystack.
/// * `'p` refers to the lifetime of the prefilter. Since a prefilter is
/// optional, this defaults to the `'static` lifetime when a prefilter is not
/// present.
#[derive(Clone)]
pub struct Search<'h, 'p> {
    haystack: &'h [u8],
    span: Span,
    pattern: Option<PatternID>,
    prefilter: Option<&'p dyn Prefilter>,
    earliest: bool,
    utf8: bool,
}

impl<'h, 'p> Search<'h, 'p> {
    /// Create a new search configuration for the given haystack.
    #[inline]
    pub fn new<H: ?Sized + AsRef<[u8]>>(
        haystack: &'h H,
    ) -> Search<'h, 'static> {
        Search {
            haystack: haystack.as_ref(),
            span: Span { start: 0, end: haystack.as_ref().len() },
            pattern: None,
            prefilter: None,
            earliest: false,
            utf8: true,
        }
    }

    /// Set the span for this search.
    ///
    /// This routine does not panic if the span given is not a valid range for
    /// this search's haystack. If this search is run with an invalid range,
    /// then the most likely outcome is that the actual search execution will
    /// panic.
    ///
    /// This routine is generic over how a span is provided. While
    /// a [`Span`] may be given directly, one may also provide a
    /// `std::ops::Range<usize>`. To provide anything supported by range
    /// syntax, use the [`Search::range`] method.
    ///
    /// # Example
    ///
    /// This example shows how the span of the search can impact whether a
    /// match is reported or not. This is particularly relevant for look-around
    /// operators, which might take things outside of the span into account
    /// when determining whether they match.
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::pikevm::PikeVM,
    ///     Match, Search,
    /// };
    ///
    /// // Look for 'at', but as a distinct word.
    /// let vm = PikeVM::new(r"\bat\b")?;
    /// let mut cache = vm.create_cache();
    /// let mut caps = vm.create_captures();
    ///
    /// // Our haystack contains 'at', but not as a distinct word.
    /// let haystack = "batter";
    ///
    /// // A standard search finds nothing, as expected.
    /// let search = Search::new(haystack);
    /// vm.search(&mut cache, &search, &mut caps);
    /// assert_eq!(None, caps.get_match());
    ///
    /// // But if we wanted to search starting at position '1', we might
    /// // slice the haystack. If we do this, it's impossible for the \b
    /// // anchors to take the surrounding context into account! And thus,
    /// // a match is produced.
    /// let search = Search::new(&haystack[1..3]);
    /// vm.search(&mut cache, &search, &mut caps);
    /// assert_eq!(Some(Match::must(0, 0..2)), caps.get_match());
    ///
    /// // But if we specify the span of the search instead of slicing the
    /// // haystack, then the regex engine can "see" outside of the span
    /// // and resolve the anchors correctly.
    /// let search = Search::new(haystack).span(1..3);
    /// vm.search(&mut cache, &search, &mut caps);
    /// assert_eq!(None, caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// This may seem a little ham-fisted, but this scenario tends to come up
    /// if some other regex engine found the match span and now you need to
    /// re-process that span to look for capturing groups. (e.g., Run a faster
    /// DFA first, find a match, then run the PikeVM on just the match span to
    /// resolve capturing groups.) In order to implement that sort of logic
    /// correctly, you need to set the span on the search instead of slicing
    /// the haystack directly.
    ///
    /// The other advantage of using this routine to specify the bounds of the
    /// search is that the match offsets are still reported in terms of the
    /// original haystack. For example, the second search in the example above
    /// reported a match at position `0`, even though `at` starts at offset
    /// `1` because we sliced the haystack.
    #[inline]
    pub fn span<S: Into<Span>>(mut self, span: S) -> Search<'h, 'p> {
        self.set_span(span);
        self
    }

    /// Like `Search::span`, but accepts any range instead.
    ///
    /// This routine does not panic if the range given is not a valid range for
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
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::Search;
    ///
    /// let search = Search::new("foobar");
    /// assert_eq!(0..6, search.get_range());
    ///
    /// let search = Search::new("foobar").range(2..=4);
    /// assert_eq!(2..5, search.get_range());
    /// ```
    #[inline]
    pub fn range<R: RangeBounds<usize>>(mut self, range: R) -> Search<'h, 'p> {
        self.set_range(range);
        self
    }

    /// Set the pattern to search for, if supported.
    ///
    /// When given, an anchored search for only the specified pattern will
    /// be executed. If not given, then the search will look for any pattern
    /// that matches. (Whether that search is anchored or not depends on
    /// the configuration of your regex engine and, ultimately, the pattern
    /// itself.)
    ///
    /// If a pattern ID is given and a regex engine doesn't support searching
    /// by a specific pattern, then the regex engine must panic.
    ///
    /// # Example
    ///
    /// This example shows how to search for a specific pattern.
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::pikevm::PikeVM,
    ///     Match, PatternID, Search,
    /// };
    ///
    /// let vm = PikeVM::new_many(&[r"[a-z0-9]{6}", r"[a-z][a-z0-9]{5}"])?;
    /// let mut cache = vm.create_cache();
    /// let mut caps = vm.create_captures();
    ///
    /// // A standard search looks for any pattern.
    /// let search = Search::new("bar foo123");
    /// vm.search(&mut cache, &search, &mut caps);
    /// assert_eq!(Some(Match::must(0, 4..10)), caps.get_match());
    ///
    /// // But we can also check whether a specific pattern
    /// // matches at a particular position.
    /// let search = Search::new("bar foo123")
    ///     .range(4..)
    ///     .pattern(Some(PatternID::must(1)));
    /// vm.search(&mut cache, &search, &mut caps);
    /// assert_eq!(Some(Match::must(1, 4..10)), caps.get_match());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn pattern(mut self, pattern: Option<PatternID>) -> Search<'h, 'p> {
        self.set_pattern(pattern);
        self
    }

    #[inline]
    pub fn prefilter(
        mut self,
        prefilter: Option<&'p dyn Prefilter>,
    ) -> Search<'h, 'p> {
        self.set_prefilter(prefilter);
        self
    }

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
    /// a match rather than a consistent semantic unto itself. This is often
    /// useful for implementing "did a match occur or not" predicates, but
    /// sometimes the offset is useful as well.
    ///
    /// This is disabled by default.
    ///
    /// # Example
    ///
    /// This example shows the difference between "earliest" searching and
    /// normal searching.
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::pikevm::PikeVM,
    ///     Match, PatternID, Search,
    /// };
    ///
    /// let vm = PikeVM::new(r"foo[0-9]+")?;
    /// let mut cache = vm.create_cache();
    /// let mut caps = vm.create_captures();
    ///
    /// // A normal search implements greediness like you expect.
    /// let search = Search::new("foo12345");
    /// vm.search(&mut cache, &search, &mut caps);
    /// assert_eq!(Some(Match::must(0, 0..8)), caps.get_match());
    ///
    /// // When 'earliest' is enabled and the regex engine supports
    /// // it, the search will bail once it knows a match has been
    /// // found.
    /// let search = Search::new("foo12345").earliest(true);
    /// vm.search(&mut cache, &search, &mut caps);
    /// assert_eq!(Some(Match::must(0, 0..4)), caps.get_match());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn earliest(mut self, yes: bool) -> Search<'h, 'p> {
        self.set_earliest(yes);
        self
    }

    /// Whether to enable UTF-8 mode during search or not.
    ///
    /// UTF-8 mode on a `Search` refers to whether a regex engine should
    /// treat the haystack as valid UTF-8 in cases where that could make a
    /// difference.
    ///
    /// An example of this occurs when a regex pattern semantically matches the
    /// empty string. In such cases, the underlying finite state machine will
    /// likely not distiguish between empty strings that do and do not split
    /// codepoints in UTF-8 haystacks. When this option is enabled, the regex
    /// engine will can insert higher level code that checks for whether the
    /// match splits a codepoint, and if so, skip that match entirely and look
    /// for the next one.
    ///
    /// In effect, this option is useful to enable when both of the following
    /// are true:
    ///
    /// 1. Your haystack is valid UTF-8.
    /// 2. You never want to report spans that fall on invalid UTF-8
    /// boundaries.
    ///
    /// Typically, this is enabled in concert with
    /// [`SyntaxConfig::utf8`](crate::SyntaxConfig::utf8).
    ///
    /// # Example
    ///
    /// This example shows how UTF-8 mode can impact the match spans that may
    /// be reported in certain cases.
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::pikevm::PikeVM,
    ///     Match, Search,
    /// };
    ///
    /// let vm = PikeVM::new("")?;
    /// let mut cache = vm.create_cache();
    /// let mut caps = vm.create_captures();
    ///
    /// // UTF-8 mode is enabled by default.
    /// let mut search = Search::new("☃");
    /// vm.search(&mut cache, &search, &mut caps);
    /// assert_eq!(Some(Match::must(0, 0..0)), caps.get_match());
    ///
    /// // Even though an empty regex matches at 1..1, our next match is
    /// // 3..3 because 1..1 and 2..2 split the snowman codepoint (which is
    /// // three bytes long).
    /// search.set_start(1);
    /// vm.search(&mut cache, &search, &mut caps);
    /// assert_eq!(Some(Match::must(0, 3..3)), caps.get_match());
    ///
    /// // But if we disable UTF-8, then we'll get matches at 1..1 and 2..2:
    /// let mut noutf8 = search.clone().utf8(false);
    /// vm.search(&mut cache, &noutf8, &mut caps);
    /// assert_eq!(Some(Match::must(0, 1..1)), caps.get_match());
    ///
    /// noutf8.set_start(2);
    /// vm.search(&mut cache, &noutf8, &mut caps);
    /// assert_eq!(Some(Match::must(0, 2..2)), caps.get_match());
    ///
    /// noutf8.set_start(3);
    /// vm.search(&mut cache, &noutf8, &mut caps);
    /// assert_eq!(Some(Match::must(0, 3..3)), caps.get_match());
    ///
    /// noutf8.set_start(4);
    /// vm.search(&mut cache, &noutf8, &mut caps);
    /// assert_eq!(None, caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn utf8(mut self, yes: bool) -> Search<'h, 'p> {
        self.set_utf8(yes);
        self
    }

    /// Set the span for this search configuration.
    ///
    /// This is like the [`Search::span`] method, except this mutates the
    /// span in place.
    ///
    /// This routine is generic over how a span is provided. While
    /// a [`Span`] may be given directly, one may also provide a
    /// `std::ops::Range<usize>`.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::Search;
    ///
    /// let mut search = Search::new("foobar");
    /// assert_eq!(0..6, search.get_range());
    /// search.set_span(2..4);
    /// assert_eq!(2..4, search.get_range());
    /// ```
    #[inline]
    pub fn set_span<S: Into<Span>>(&mut self, span: S) {
        self.span = span.into();
    }

    /// Set the span for this search configuration given any range.
    ///
    /// This is like the [`Search::range`] method, except this mutates the
    /// span in place.
    ///
    /// This routine does not panic if the range given is not a valid range for
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
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::Search;
    ///
    /// let mut search = Search::new("foobar");
    /// assert_eq!(0..6, search.get_range());
    /// search.set_range(2..=4);
    /// assert_eq!(2..5, search.get_range());
    /// ```
    #[inline]
    pub fn set_range<R: RangeBounds<usize>>(&mut self, range: R) {
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
        self.set_span(Span { start, end });
    }

    /// Set the starting offset for the span for this search configuration.
    ///
    /// This is a convenience routine for only mutating the start of a span
    /// without having to set the entire span.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::Search;
    ///
    /// let mut search = Search::new("foobar");
    /// assert_eq!(0..6, search.get_range());
    /// search.set_start(5);
    /// assert_eq!(5..6, search.get_range());
    /// ```
    #[inline]
    pub fn set_start(&mut self, start: usize) {
        self.span.start = start;
    }

    /// Set the ending offset for the span for this search configuration.
    ///
    /// This is a convenience routine for only mutating the end of a span
    /// without having to set the entire span.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::Search;
    ///
    /// let mut search = Search::new("foobar");
    /// assert_eq!(0..6, search.get_range());
    /// search.set_end(5);
    /// assert_eq!(0..5, search.get_range());
    /// ```
    #[inline]
    pub fn set_end(&mut self, end: usize) {
        self.span.end = end;
    }

    /// Set the pattern to search for.
    ///
    /// This is like [`Search::pattern`], except it mutates the search
    /// configuration in place.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{PatternID, Search};
    ///
    /// let mut search = Search::new("foobar");
    /// assert_eq!(None, search.get_pattern());
    /// search.set_pattern(Some(PatternID::must(5)));
    /// assert_eq!(Some(PatternID::must(5)), search.get_pattern());
    /// ```
    #[inline]
    pub fn set_pattern(&mut self, pattern: Option<PatternID>) {
        self.pattern = pattern;
    }

    #[inline]
    pub fn set_prefilter(&mut self, prefilter: Option<&'p dyn Prefilter>) {
        self.prefilter = prefilter;
    }

    /// Set whether the search should execute in "earliest" mode or not.
    ///
    /// This is like [`Search::earliest`], except it mutates the search
    /// configuration in place.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::Search;
    ///
    /// let mut search = Search::new("foobar");
    /// assert!(!search.get_earliest());
    /// search.set_earliest(true);
    /// assert!(search.get_earliest());
    /// ```
    #[inline]
    pub fn set_earliest(&mut self, yes: bool) {
        self.earliest = yes;
    }

    /// Set whether the search should execute in UTF-8 mode or not.
    ///
    /// This is like [`Search::utf8`], except it mutates the search
    /// configuration in place.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::Search;
    ///
    /// let mut search = Search::new("foobar");
    /// assert!(search.get_utf8());
    /// search.set_utf8(false);
    /// assert!(!search.get_utf8());
    /// ```
    #[inline]
    pub fn set_utf8(&mut self, yes: bool) {
        self.utf8 = yes;
    }

    /// Return a borrow of the underlying haystack as a slice of bytes.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::Search;
    ///
    /// let search = Search::new("foobar");
    /// assert_eq!(b"foobar", search.haystack());
    /// ```
    #[inline]
    pub fn haystack(&self) -> &[u8] {
        self.haystack
    }

    /// Return the start position of this search.
    ///
    /// This is a convenience routine for `search.get_span().start()`.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::Search;
    ///
    /// let search = Search::new("foobar");
    /// assert_eq!(0, search.start());
    ///
    /// let search = Search::new("foobar").span(2..4);
    /// assert_eq!(2, search.start());
    /// ```
    #[inline]
    pub fn start(&self) -> usize {
        self.get_span().start
    }

    /// Return the end position of this search.
    ///
    /// This is a convenience routine for `search.get_span().end()`.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::Search;
    ///
    /// let search = Search::new("foobar");
    /// assert_eq!(6, search.end());
    ///
    /// let search = Search::new("foobar").span(2..4);
    /// assert_eq!(4, search.end());
    /// ```
    #[inline]
    pub fn end(&self) -> usize {
        self.get_span().end
    }

    /// Return the span for this search configuration.
    ///
    /// If one was not explicitly set, then the span corresponds to the entire
    /// range of the haystack.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{Search, Span};
    ///
    /// let search = Search::new("foobar");
    /// assert_eq!(Span { start: 0, end: 6 }, search.get_span());
    /// ```
    #[inline]
    pub fn get_span(&self) -> Span {
        self.span
    }

    /// Return the span as a range for this search configuration.
    ///
    /// If one was not explicitly set, then the span corresponds to the entire
    /// range of the haystack.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::Search;
    ///
    /// let search = Search::new("foobar");
    /// assert_eq!(0..6, search.get_range());
    /// ```
    #[inline]
    pub fn get_range(&self) -> Range<usize> {
        self.get_span().range()
    }

    /// Return the pattern ID for this search configuration, if one was set.
    ///
    /// When no pattern is set, the regex engine should look for matches for
    /// any of the patterns that are in the regex object.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::Search;
    ///
    /// let search = Search::new("foobar");
    /// assert_eq!(None, search.get_pattern());
    /// ```
    #[inline]
    pub fn get_pattern(&self) -> Option<PatternID> {
        self.pattern
    }

    #[inline]
    pub fn get_prefilter(&self) -> Option<&'p dyn Prefilter> {
        self.prefilter
    }

    /// Return whether this search should execute in "earliest" mode.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::Search;
    ///
    /// let search = Search::new("foobar");
    /// assert!(!search.get_earliest());
    /// ```
    #[inline]
    pub fn get_earliest(&self) -> bool {
        self.earliest
    }

    /// Return whether this search should execute in UTF-8 mode.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::Search;
    ///
    /// let search = Search::new("foobar");
    /// assert!(search.get_utf8());
    /// ```
    #[inline]
    pub fn get_utf8(&self) -> bool {
        self.utf8
    }

    /// Return true if and only if this search can never return any other
    /// matches.
    ///
    /// For example, if the start position of this search is greater than the
    /// end position of the search.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::Search;
    ///
    /// let mut search = Search::new("foobar");
    /// assert!(!search.is_done());
    /// search.set_start(6);
    /// assert!(!search.is_done());
    /// search.set_start(7);
    /// assert!(search.is_done());
    /// ```
    #[inline]
    pub fn is_done(&self) -> bool {
        self.get_span().start > self.get_span().end
    }

    /// Returns true if and only if the given offset in this search's haystack
    /// falls on a valid UTF-8 encoded codepoint boundary.
    ///
    /// If the haystack is not valid UTF-8, then the behavior of this routine
    /// is unspecified.
    ///
    /// # Example
    ///
    /// This shows where codepoint bounardies do and don't exist in valid
    /// UTF-8.
    ///
    /// ```
    /// use regex_automata::Search;
    ///
    /// let search = Search::new("☃");
    /// assert!(search.is_char_boundary(0));
    /// assert!(!search.is_char_boundary(1));
    /// assert!(!search.is_char_boundary(2));
    /// assert!(search.is_char_boundary(3));
    /// assert!(!search.is_char_boundary(4));
    /// ```
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
        F: FnMut(&Search<'_, '_>) -> Result<Option<Match>, MatchError>,
    {
        if !self.get_utf8() || !m.is_empty() {
            return Ok(Some(m));
        }
        let mut search = self.clone();
        while m.is_empty() && !search.is_char_boundary(m.end()) {
            search.set_start(search.start().checked_add(1).unwrap());
            m = match find(&search)? {
                None => return Ok(None),
                Some(m) => m,
            };
        }
        Ok(Some(m))
    }
}

impl<'h, 'p> core::fmt::Debug for Search<'h, 'p> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use crate::util::escape::DebugHaystack;

        f.debug_struct("Search")
            .field("haystack", &DebugHaystack(self.haystack()))
            .field("span", &self.span)
            .field("prefilter", &self.prefilter)
            .field("pattern", &self.pattern)
            .field("earliest", &self.earliest)
            .field("utf8", &self.utf8)
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
