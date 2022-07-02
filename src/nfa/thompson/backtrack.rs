/*!
An NFA backed bounded backtracker for executing regex searches with capturing
groups.

This module provides a [`BoundedBacktracker`] that works by simulating an NFA
using the classical backtracking algorithm with a twist: it avoids redoing
work that it has done before and thereby avoids worst case exponential time.
In exchange, it can only be used on "short" haystacks. Its advantage is that
is can be faster than the [`PikeVM`](thompson::pikevm::PikeVM) in many cases
because it does less book-keeping.
*/

use alloc::sync::Arc;

use crate::{
    nfa::thompson::{self, Captures, State, NFA},
    util::{id::StateID, iter, nonmax::NonMaxUsize, prefilter::Prefilter},
    Input, Match, MatchError, MatchKind,
};

/// Returns the minimum visited capacity for the given haystack.
///
/// This function can be used as the argument to [`Config::visited_capacity`]
/// in order to guarantee that a backtracking search for the
/// given `input.haystack()` won't return an error when using a
/// [`BoundedBacktracker`] built from the given `NFA`.
///
/// This routine exists primarily as a way to test that the bounded backtracker
/// works correctly when its capacity is set to the smallest possible amount.
/// Still, it may be useful in cases where you know you want to use the bounded
/// backtracker for a specific input, and just need to know what visited
/// capacity to provide to make it work.
///
/// Be warned that this number could be quite large as it is multiplicative in
/// the size the given NFA and haystack.
pub fn min_visited_capacity(nfa: &NFA, input: &Input<'_, '_>) -> usize {
    div_ceil(nfa.states().len() * (input.haystack().len() + 1), 8)
}

/// The configuration used for building a bounded backtracker.
///
/// A bounded backtracker configuration is a simple data object that is
/// typically used with [`Builder::configure`].
#[derive(Clone, Debug, Default)]
pub struct Config {
    anchored: Option<bool>,
    utf8: Option<bool>,
    pre: Option<Option<Arc<dyn Prefilter>>>,
    visited_capacity: Option<usize>,
}

impl Config {
    /// Return a new default regex configuration.
    pub fn new() -> Config {
        Config::default()
    }

    /// Set whether matching must be anchored at the beginning of the input.
    ///
    /// When enabled, a match must begin at the start of a search. When
    /// disabled (the default), the regex will act as if the pattern started
    /// with a `(?s:.)*?`, which enables a match to appear anywhere.
    ///
    /// By default this is disabled.
    ///
    /// **WARNING:** this is subtly different than using a `^` at the start of
    /// your regex. A `^` forces a regex to match exclusively at the start of
    /// input, regardless of where you begin your search. In contrast, enabling
    /// this option will allow your regex to match anywhere in your input,
    /// but the match must start at the beginning of a search. (Most of the
    /// higher level convenience search routines make "start of input" and
    /// "start of search" equivalent, but some routines allow treating these as
    /// orthogonal.)
    ///
    /// For example, consider the haystack `aba` and the following searches:
    ///
    /// 1. The regex `^a` is compiled with `anchored=false` and searches
    ///    `aba` starting at position `2`. Since `^` requires the match to
    ///    start at the beginning of the input and `2 > 0`, no match is found.
    /// 2. The regex `a` is compiled with `anchored=true` and searches `aba`
    ///    starting at position `2`. This reports a match at `[2, 3]` since
    ///    the match starts where the search started. Since there is no `^`,
    ///    there is no requirement for the match to start at the beginning of
    ///    the input.
    /// 3. The regex `a` is compiled with `anchored=true` and searches `aba`
    ///    starting at position `1`. Since `b` corresponds to position `1` and
    ///    since the regex is anchored, it finds no match.
    /// 4. The regex `a` is compiled with `anchored=false` and searches `aba`
    ///    startting at position `1`. Since the regex is neither anchored nor
    ///    starts with `^`, the regex is compiled with an implicit `(?s:.)*?`
    ///    prefix that permits it to match anywhere. Thus, it reports a match
    ///    at `[2, 3]`.
    ///
    /// # Example
    ///
    /// This demonstrates the differences between an anchored search and
    /// a pattern that begins with `^` (as described in the above warning
    /// message).
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::backtrack::BoundedBacktracker,
    ///     Match, Input,
    /// };
    ///
    /// let haystack = "aba";
    ///
    /// let re = BoundedBacktracker::builder()
    ///     .configure(BoundedBacktracker::config().anchored(false)) // default
    ///     .build(r"^a")?;
    /// let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    /// re.try_search(&mut cache, &Input::new(haystack).span(2..3), &mut caps);
    /// // No match is found because 2 is not the beginning of the haystack,
    /// // which is what ^ requires.
    /// let expected = None;
    /// assert_eq!(expected, caps.get_match());
    ///
    /// let re = BoundedBacktracker::builder()
    ///     .configure(BoundedBacktracker::config().anchored(true))
    ///     .build(r"a")?;
    /// let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    /// re.try_search(&mut cache, &Input::new(haystack).span(2..3), &mut caps);
    /// // An anchored search can still match anywhere in the haystack, it just
    /// // must begin at the start of the search which is '2' in this case.
    /// let expected = Some(Match::must(0, 2..3));
    /// assert_eq!(expected, caps.get_match());
    ///
    /// let re = BoundedBacktracker::builder()
    ///     .configure(BoundedBacktracker::config().anchored(true))
    ///     .build(r"a")?;
    /// let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    /// re.try_search(&mut cache, &Input::new(haystack).span(1..3), &mut caps);
    /// // No match is found since we start searching at offset 1 which
    /// // corresponds to 'b'. Since there is no '(?s:.)*?' prefix, no match
    /// // is found.
    /// let expected = None;
    /// assert_eq!(expected, caps.get_match());
    ///
    /// let re = BoundedBacktracker::builder()
    ///     .configure(BoundedBacktracker::config().anchored(false))
    ///     .build(r"a")?;
    /// let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    /// re.try_search(&mut cache, &Input::new(haystack).span(1..3), &mut caps);
    /// // Since anchored=false, an implicit '(?s:.)*?' prefix was added to the
    /// // pattern. Even though the search starts at 'b', the 'match anything'
    /// // prefix allows the search to match 'a'.
    /// let expected = Some(Match::must(0, 2..3));
    /// assert_eq!(expected, caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn anchored(mut self, yes: bool) -> Config {
        self.anchored = Some(yes);
        self
    }

    /// Whether to enable UTF-8 mode or not.
    ///
    /// When UTF-8 mode is enabled (the default) and an empty match is seen,
    /// the search APIs of [`BoundedBacktracker`] will always start the next
    /// search at the next UTF-8 encoded codepoint when searching valid UTF-8.
    /// When UTF-8 mode is disabled, such searches are begun at the next byte
    /// offset.
    ///
    /// If this mode is enabled and invalid UTF-8 is given to search, then
    /// behavior is unspecified.
    ///
    /// Generally speaking, one should enable this when
    /// [`SyntaxConfig::utf8`](crate::SyntaxConfig::utf8) is enabled, and
    /// disable it otherwise.
    ///
    /// # Example
    ///
    /// This example demonstrates the differences between when this option
    /// is enabled and disabled. The differences only arise when the
    /// regex can return matches of length zero.
    ///
    /// In this first snippet, we show the results when UTF-8 mode is disabled.
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::backtrack::BoundedBacktracker,
    ///     Match,
    /// };
    ///
    /// let re = BoundedBacktracker::builder()
    ///     .configure(BoundedBacktracker::config().utf8(false))
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
    /// use regex_automata::{
    ///     nfa::thompson::backtrack::BoundedBacktracker,
    ///     Match,
    /// };
    ///
    /// let re = BoundedBacktracker::builder()
    ///     .configure(BoundedBacktracker::config().utf8(true))
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
    /// The given prefilter is automatically applied to every search, except
    /// for the lower level routines that accept a prefilter parameter from the
    /// caller (via [`Input::prefilter`]).
    pub fn prefilter(mut self, pre: Option<Arc<dyn Prefilter>>) -> Config {
        self.pre = Some(pre);
        self
    }

    /// Set the visited capacity used to bound backtracking.
    ///
    /// The visited capacity represents the amount of heap memory (in bytes) to
    /// allocate toward tracking which parts of the backtracking search have
    /// been done before. The heap memory needed for any particular search is
    /// proportional to `haystack.len() * nfa.states().len()`, whichc an be
    /// quite large. Therefore, the bounded backtracker is typically only able
    /// to run on shorter haystacks.
    ///
    /// For a given regex, increasing the visited capacity means that the
    /// maximum haystack length that can be searched is increased. The
    /// [`BoundedBacktracker::max_haystack_len`] method returns that maximum.
    ///
    /// The default capacity is a reasonable but empirically chosen size.
    ///
    /// # Example
    ///
    /// As with other regex engines, Unicode is what tends to make the bounded
    /// backtracker less useful by making the maximum haystack length quite
    /// small. If necessary, increasing the visited capacity using this routine
    /// will increase the maximum haystack length at the cost of using more
    /// memory.
    ///
    /// Note though that the specific maximum values here are not an API
    /// guarantee. The default visited capacity is subject to change and not
    /// covered by semver.
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::backtrack::BoundedBacktracker,
    ///     Match, MatchError,
    /// };
    ///
    /// // Unicode inflates the size of the underlying NFA quite a bit, and
    /// // thus means that the backtracker can only handle smaller haystacks,
    /// // assuming that the visited capacity remains unchanged.
    /// let re = BoundedBacktracker::new(r"\w+")?;
    /// assert_eq!(re.max_haystack_len(), 7_036);
    /// // But we can increase the visited capacity to handle bigger haystacks!
    /// let re = BoundedBacktracker::builder()
    ///     .configure(BoundedBacktracker::config().visited_capacity(1<<20))
    ///     .build(r"\w+")?;
    /// assert_eq!(re.max_haystack_len(), 28_148);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn visited_capacity(mut self, capacity: usize) -> Config {
        self.visited_capacity = Some(capacity);
        self
    }

    /// Returns whether this configuration has enabled anchored searches.
    pub fn get_anchored(&self) -> bool {
        self.anchored.unwrap_or(false)
    }

    /// Returns true if and only if this configuration has UTF-8 mode enabled.
    ///
    /// When UTF-8 mode is enabled and an empty match is seen, the regex will
    /// always start the next search at the next UTF-8 encoded codepoint.
    /// When UTF-8 mode is disabled, such searches are begun at the next byte
    /// offset.
    pub fn get_utf8(&self) -> bool {
        self.utf8.unwrap_or(true)
    }

    pub fn get_prefilter(&self) -> Option<&dyn Prefilter> {
        self.pre.as_ref().unwrap_or(&None).as_deref()
    }

    /// Returns the configured visited capacity.
    ///
    /// Note that the actual capacity used may be slightly bigger than the
    /// configured capacity.
    pub fn get_visited_capacity(&self) -> usize {
        const DEFAULT: usize = 256 * (1 << 10); // 256 KB
        self.visited_capacity.unwrap_or(DEFAULT)
    }

    /// Overwrite the default configuration such that the options in `o` are
    /// always used. If an option in `o` is not set, then the corresponding
    /// option in `self` is used. If it's not set in `self` either, then it
    /// remains not set.
    pub(crate) fn overwrite(&self, o: Config) -> Config {
        Config {
            anchored: o.anchored.or(self.anchored),
            utf8: o.utf8.or(self.utf8),
            pre: o.pre.or_else(|| self.pre.clone()),
            visited_capacity: o.visited_capacity.or(self.visited_capacity),
        }
    }
}

/// A builder for a bounded backtracker.
///
/// This builder permits configuring options for the syntax of a pattern, the
/// NFA construction and the `BoundedBacktracker` construction. This builder
/// is different from a general purpose regex builder in that it permits fine
/// grain configuration of the construction process. The trade off for this is
/// complexity, and the possibility of setting a configuration that might not
/// make sense. For example, there are two different UTF-8 modes:
///
/// * [`SyntaxConfig::utf8`](crate::SyntaxConfig::utf8) controls whether the
/// pattern itself can contain sub-expressions that match invalid UTF-8.
/// * [`Config::utf8`] controls how the regex iterators themselves advance
/// the starting position of the next search when a match with zero length is
/// found.
///
/// Generally speaking, callers will want to either enable all of these or
/// disable all of these.
///
/// # Example
///
/// This example shows how to disable UTF-8 mode in the syntax and the regex
/// itself. This is generally what you want for matching on arbitrary bytes.
///
/// ```
/// use regex_automata::{
///     nfa::thompson::backtrack::BoundedBacktracker,
///     Match, SyntaxConfig,
/// };
///
/// let re = BoundedBacktracker::builder()
///     .configure(BoundedBacktracker::config().utf8(false))
///     .syntax(SyntaxConfig::new().utf8(false))
///     .build(r"foo(?-u:[^b])ar.*")?;
/// let mut cache = re.create_cache();
///
/// let haystack = b"\xFEfoo\xFFarzz\xE2\x98\xFF\n";
/// let expected = Some(Match::must(0, 1..9));
/// let got = re.find_iter(&mut cache, haystack).next();
/// assert_eq!(expected, got);
/// // Notice that `(?-u:[^b])` matches invalid UTF-8,
/// // but the subsequent `.*` does not! Disabling UTF-8
/// // on the syntax permits this.
/// //
/// // N.B. This example does not show the impact of
/// // disabling UTF-8 mode on a BoundedBacktracker Config, since that
/// // only impacts regexes that can produce matches of
/// // length 0.
/// assert_eq!(b"foo\xFFarzz", &haystack[got.unwrap().range()]);
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug)]
pub struct Builder {
    config: Config,
    thompson: thompson::Compiler,
}

impl Builder {
    /// Create a new BoundedBacktracker builder with its default configuration.
    pub fn new() -> Builder {
        Builder {
            config: Config::default(),
            thompson: thompson::Compiler::new(),
        }
    }

    /// Build a `BoundedBacktracker` from the given pattern.
    ///
    /// If there was a problem parsing or compiling the pattern, then an error
    /// is returned.
    pub fn build(
        &self,
        pattern: &str,
    ) -> Result<BoundedBacktracker, thompson::Error> {
        self.build_many(&[pattern])
    }

    /// Build a `BoundedBacktracker` from the given patterns.
    pub fn build_many<P: AsRef<str>>(
        &self,
        patterns: &[P],
    ) -> Result<BoundedBacktracker, thompson::Error> {
        let nfa = self.thompson.build_many(patterns)?;
        self.build_from_nfa(nfa)
    }

    /// Build a `BoundedBacktracker` directly from its NFA.
    ///
    /// Note that when using this method, any configuration that applies to the
    /// construction of the NFA itself will of course be ignored, since the NFA
    /// given here is already built.
    pub fn build_from_nfa(
        &self,
        nfa: NFA,
    ) -> Result<BoundedBacktracker, thompson::Error> {
        // If the NFA has no captures, then the backtracker doesn't work since
        // it relies on them in order to report match locations. However, in
        // the special case of an NFA with no patterns, it is allowed, since
        // no matches can ever be produced. And importantly, an NFA with no
        // patterns has no capturing groups anyway, so this is necessary to
        // permit the backtracker to work with regexes with zero patterns.
        if !nfa.has_capture() && nfa.pattern_len() > 0 {
            return Err(thompson::Error::missing_captures());
        }
        if !cfg!(feature = "syntax") {
            if nfa.has_word_boundary_unicode() {
                return Err(thompson::Error::unicode_word_unavailable());
            }
        }
        Ok(BoundedBacktracker { config: self.config.clone(), nfa })
    }

    /// Apply the given `BoundedBacktracker` configuration options to this
    /// builder.
    pub fn configure(&mut self, config: Config) -> &mut Builder {
        self.config = self.config.overwrite(config);
        self
    }

    /// Set the syntax configuration for this builder using
    /// [`SyntaxConfig`](crate::SyntaxConfig).
    ///
    /// This permits setting things like case insensitivity, Unicode and multi
    /// line mode.
    ///
    /// These settings only apply when constructing a `BoundedBacktracker`
    /// directly from a pattern.
    pub fn syntax(
        &mut self,
        config: crate::util::syntax::SyntaxConfig,
    ) -> &mut Builder {
        self.thompson.syntax(config);
        self
    }

    /// Set the Thompson NFA configuration for this builder using
    /// [`nfa::thompson::Config`](crate::nfa::thompson::Config).
    ///
    /// This permits setting things like if additional time should be spent
    /// shrinking the size of the NFA.
    ///
    /// These settings only apply when constructing a `BoundedBacktracker`
    /// directly from a pattern.
    pub fn thompson(&mut self, config: thompson::Config) -> &mut Builder {
        self.thompson.configure(config);
        self
    }
}

/// A backtracking regex engine that bounds its execution to avoid exponential
/// blow-up.
///
/// This regex engine only implements leftmost-first match semantics and
/// only supports leftmost searches. It effectively does the same thing as a
/// [`PikeVM`](thompson::pikevm::PikeVM), but typically does it faster because
/// it doesn't have to worry about copying capturing group spans for most NFA
/// states. Instead, the backtracker can maintain one set of captures (provided
/// by the caller) and never needs to copy them. In exchange, the backtracker
/// bounds itself to ensure it doesn't exhibit worst case exponential time.
/// This results in the backtracker only being able to handle short haystacks
/// given reasonable memory usage.
///
/// # Searches may return an error!
///
/// By design, this backtracking regex engine is bounded. This bound is
/// implemented by not visiting any combination of NFA state ID and position
/// in a haystack. Thus, the total memory required to bound backtracking is
/// proportional to `haystack.len() * nfa.states().len()`. This can obviously
/// get quite large, since large haystacks aren't terribly uncommon. To avoid
/// using exorbitant memory, the capacity is bounded by a fixed limit set via
/// [`Config::visited_capacity`]. Thus, if the total capacity required for a
/// particular regex and a haystack exceeds this capacity, then the search
/// routine will return an error.
///
/// Unlike other regex engines that may return an error at search time (like
/// the DFA or the hybrid NFA/DFA), there is no way to guarantee that a bounded
/// backtracker will work for every haystack. Therefore, it is strongly advised
/// to use the fallible search APIs (methods beginning with `try_`).
///
/// If you do want to use the infallible search APIs, the only way to do so
/// without it potentially panicking is to ensure that your haystack's length
/// does not exceed [`BoundedBacktracker::max_haystack_len`].
///
/// # Example: Unicode word boundaries
///
/// This example shows that the bounded backtracker implements Unicode word
/// boundaries correctly by default.
///
/// ```
/// use regex_automata::{nfa::thompson::backtrack::BoundedBacktracker, Match};
///
/// let re = BoundedBacktracker::new(r"\b\w+\b")?;
/// let mut cache = re.create_cache();
///
/// let mut it = re.find_iter(&mut cache, "Шерлок Холмс");
/// assert_eq!(Some(Match::must(0, 0..12)), it.next());
/// assert_eq!(Some(Match::must(0, 13..23)), it.next());
/// assert_eq!(None, it.next());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Example: multiple regex patterns
///
/// The bounded backtracker supports searching for multiple patterns
/// simultaneously, just like other regex engines. Note though that because it
/// uses a backtracking strategy, this regex engine is unlikely to scale well
/// as more patterns are added. But then again, as more patterns are added, the
/// maximum haystack length allowed will also shorten (assuming the visited
/// capacity remains invariant).
///
/// ```
/// use regex_automata::{nfa::thompson::backtrack::BoundedBacktracker, Match};
///
/// let re = BoundedBacktracker::new_many(&["[a-z]+", "[0-9]+"])?;
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
#[derive(Clone, Debug)]
pub struct BoundedBacktracker {
    config: Config,
    nfa: NFA,
}

impl BoundedBacktracker {
    /// Parse the given regular expression using the default configuration and
    /// return the corresponding `BoundedBacktracker`.
    ///
    /// If you want a non-default configuration, then use the [`Builder`] to
    /// set your own configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::backtrack::BoundedBacktracker,
    ///     Match,
    /// };
    ///
    /// let re = BoundedBacktracker::new("foo[0-9]+bar")?;
    /// let mut cache = re.create_cache();
    /// assert_eq!(
    ///     Some(Match::must(0, 3..14)),
    ///     re.find_iter(&mut cache, "zzzfoo12345barzzz").next(),
    /// );
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(pattern: &str) -> Result<BoundedBacktracker, thompson::Error> {
        BoundedBacktracker::builder().build(pattern)
    }

    /// Like `new`, but parses multiple patterns into a single "multi regex."
    /// This similarly uses the default regex configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::backtrack::BoundedBacktracker,
    ///     Match,
    /// };
    ///
    /// let re = BoundedBacktracker::new_many(&["[a-z]+", "[0-9]+"])?;
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
    ) -> Result<BoundedBacktracker, thompson::Error> {
        BoundedBacktracker::builder().build_many(patterns)
    }

    /// # Example
    ///
    /// This shows how to hand assemble a regular expression via its HIR,
    /// compile an NFA from it and build a BoundedBacktracker from the NFA.
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::{NFA, backtrack::BoundedBacktracker},
    ///     Match,
    /// };
    /// use regex_syntax::hir::{Hir, Class, ClassBytes, ClassBytesRange};
    ///
    /// let hir = Hir::class(Class::Bytes(ClassBytes::new(vec![
    ///     ClassBytesRange::new(b'0', b'9'),
    ///     ClassBytesRange::new(b'A', b'Z'),
    ///     ClassBytesRange::new(b'_', b'_'),
    ///     ClassBytesRange::new(b'a', b'z'),
    /// ])));
    ///
    /// let config = NFA::config().nfa_size_limit(Some(1_000));
    /// let nfa = NFA::compiler().configure(config).build_from_hir(&hir)?;
    ///
    /// let re = BoundedBacktracker::new_from_nfa(nfa)?;
    /// let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    /// let expected = Some(Match::must(0, 3..4));
    /// re.find(&mut cache, "!@#A#@!", &mut caps);
    /// assert_eq!(expected, caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new_from_nfa(
        nfa: NFA,
    ) -> Result<BoundedBacktracker, thompson::Error> {
        BoundedBacktracker::builder().build_from_nfa(nfa)
    }

    /// Create a new `BoundedBacktracker` that matches every input.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::backtrack::BoundedBacktracker,
    ///     Match,
    /// };
    ///
    /// let re = BoundedBacktracker::always_match()?;
    /// let mut cache = re.create_cache();
    ///
    /// let expected = Some(Match::must(0, 0..0));
    /// assert_eq!(expected, re.find_iter(&mut cache, "").next());
    /// assert_eq!(expected, re.find_iter(&mut cache, "foo").next());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn always_match() -> Result<BoundedBacktracker, thompson::Error> {
        let nfa = thompson::NFA::always_match();
        BoundedBacktracker::new_from_nfa(nfa)
    }

    /// Create a new `BoundedBacktracker` that never matches any input.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::nfa::thompson::backtrack::BoundedBacktracker;
    ///
    /// let re = BoundedBacktracker::never_match()?;
    /// let mut cache = re.create_cache();
    ///
    /// assert_eq!(None, re.find_iter(&mut cache, "").next());
    /// assert_eq!(None, re.find_iter(&mut cache, "foo").next());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn never_match() -> Result<BoundedBacktracker, thompson::Error> {
        let nfa = thompson::NFA::never_match();
        BoundedBacktracker::new_from_nfa(nfa)
    }

    /// Return a default configuration for a `BoundedBacktracker`.
    ///
    /// This is a convenience routine to avoid needing to import the `Config`
    /// type when customizing the construction of a `BoundedBacktracker`.
    ///
    /// # Example
    ///
    /// This example shows how to disable UTF-8 mode for `BoundedBacktracker`
    /// searches. When UTF-8 mode is disabled, the position immediately
    /// following an empty match is where the next search begins, instead of
    /// the next position of a UTF-8 encoded codepoint.
    ///
    /// In the code below, notice that `""` is permitted to match positions
    /// that split the encoding of a codepoint. When the [`Config::utf8`]
    /// option is disabled, those positions are not reported.
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::backtrack::BoundedBacktracker,
    ///     Match,
    /// };
    ///
    /// let re = BoundedBacktracker::builder()
    ///     .configure(BoundedBacktracker::config().utf8(false))
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

    /// Return a builder for configuring the construction of a
    /// `BoundedBacktracker`.
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
    ///     nfa::thompson::backtrack::BoundedBacktracker,
    ///     Match, SyntaxConfig,
    /// };
    ///
    /// let re = BoundedBacktracker::builder()
    ///     .configure(BoundedBacktracker::config().utf8(false))
    ///     .syntax(SyntaxConfig::new().utf8(false))
    ///     .build(r"foo(?-u:[^b])ar.*")?;
    /// let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    ///
    /// let haystack = b"\xFEfoo\xFFarzz\xE2\x98\xFF\n";
    /// let expected = Some(Match::must(0, 1..9));
    /// re.find(&mut cache, haystack, &mut caps);
    /// assert_eq!(expected, caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn builder() -> Builder {
        Builder::new()
    }

    /// Create a new `Input` for the given haystack.
    ///
    /// The `Input` returned is configured to match the configuration of this
    /// `BoundedBacktracker`. For example, if this `BoundedBacktracker` was
    /// built with [`Config::utf8`] enabled, then the `Input` returned will
    /// also have its [`Input::utf8`] knob enabled.
    ///
    /// This routine is useful when using the lower-level
    /// [`BoundedBacktracker::try_search`] API.
    pub fn create_input<'h, 'p, H: ?Sized + AsRef<[u8]>>(
        &'p self,
        haystack: &'h H,
    ) -> Input<'h, 'p> {
        let c = self.get_config();
        Input::new(haystack.as_ref())
            .prefilter(c.get_prefilter())
            .utf8(c.get_utf8())
    }

    /// Create a new cache for this regex.
    ///
    /// The cache returned should only be used for searches for this
    /// regex. If you want to reuse the cache for another regex, then you
    /// must call [`Cache::reset`] with that regex (or, equivalently,
    /// [`BoundedBacktracker::reset_cache`]).
    pub fn create_cache(&self) -> Cache {
        Cache::new(self)
    }

    /// Create a new empty set of capturing groups that is guaranteed to be
    /// valid for the search APIs on this `BoundedBacktracker`.
    ///
    /// A `Captures` value created for a specific `BoundedBacktracker` cannot
    /// be used with any other `BoundedBacktracker`.
    ///
    /// See the [`Captures`] documentation for an explanation of its
    /// alternative constructors that permit the `BoundedBacktracker` to do
    /// less work during a search, and thus might make it faster.
    pub fn create_captures(&self) -> Captures {
        Captures::new(self.get_nfa().clone())
    }

    /// Reset the given cache such that it can be used for searching with the
    /// this `BoundedBacktracker` (and only this `BoundedBacktracker`).
    ///
    /// A cache reset permits reusing memory already allocated in this cache
    /// with a different `BoundedBacktracker`.
    ///
    /// # Example
    ///
    /// This shows how to re-purpose a cache for use with a different
    /// `BoundedBacktracker`.
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::backtrack::BoundedBacktracker,
    ///     Match,
    /// };
    ///
    /// let re1 = BoundedBacktracker::new(r"\w")?;
    /// let re2 = BoundedBacktracker::new(r"\W")?;
    ///
    /// let mut cache = re1.create_cache();
    /// assert_eq!(
    ///     Some(Match::must(0, 0..2)),
    ///     re1.find_iter(&mut cache, "Δ").next(),
    /// );
    ///
    /// // Using 'cache' with re2 is not allowed. It may result in panics or
    /// // incorrect results. In order to re-purpose the cache, we must reset
    /// // it with the BoundedBacktracker we'd like to use it with.
    /// //
    /// // Similarly, after this reset, using the cache with 're1' is also not
    /// // allowed.
    /// cache.reset(&re2);
    /// assert_eq!(
    ///     Some(Match::must(0, 0..3)),
    ///     re2.find_iter(&mut cache, "☃").next(),
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reset_cache(&self, cache: &mut Cache) {
        cache.reset(self);
    }

    /// Returns the total number of patterns compiled into this
    /// `BoundedBacktracker`.
    ///
    /// In the case of a `BoundedBacktracker` that contains no patterns, this
    /// returns `0`.
    ///
    /// # Example
    ///
    /// This example shows the pattern length for a `BoundedBacktracker` that
    /// never matches:
    ///
    /// ```
    /// use regex_automata::nfa::thompson::backtrack::BoundedBacktracker;
    ///
    /// let re = BoundedBacktracker::never_match()?;
    /// assert_eq!(re.pattern_len(), 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// And another example for a `BoundedBacktracker` that matches at every
    /// position:
    ///
    /// ```
    /// use regex_automata::nfa::thompson::backtrack::BoundedBacktracker;
    ///
    /// let re = BoundedBacktracker::always_match()?;
    /// assert_eq!(re.pattern_len(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// And finally, a `BoundedBacktracker` that was constructed from multiple
    /// patterns:
    ///
    /// ```
    /// use regex_automata::nfa::thompson::backtrack::BoundedBacktracker;
    ///
    /// let re = BoundedBacktracker::new_many(&["[0-9]+", "[a-z]+", "[A-Z]+"])?;
    /// assert_eq!(re.pattern_len(), 3);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn pattern_len(&self) -> usize {
        self.nfa.pattern_len()
    }

    /// Return the config for this `BoundedBacktracker`.
    #[inline]
    pub fn get_config(&self) -> &Config {
        &self.config
    }

    /// Returns a reference to the underlying NFA.
    #[inline]
    pub fn get_nfa(&self) -> &NFA {
        &self.nfa
    }

    /// Returns the maximum haystack length supported by this backtracker.
    ///
    /// This routine is a function of both [`Config::visited_capacity`] and the
    /// internal size of the backtracker's NFA.
    ///
    /// # Example
    ///
    /// This example shows how the maximum haystack length can vary depending
    /// on the size of the regex itself. Note though that the specific maximum
    /// values here are not an API guarantee. The default visited capacity is
    /// subject to change and not covered by semver.
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::backtrack::BoundedBacktracker,
    ///     Match, MatchError,
    /// };
    ///
    /// // If you're only using ASCII, you get a big budget.
    /// let re = BoundedBacktracker::new(r"(?-u)\w+")?;
    /// let mut cache = re.create_cache();
    /// assert_eq!(re.max_haystack_len(), 299_592);
    /// // Things work up to the max.
    /// let mut haystack = "a".repeat(299_592);
    /// let expected = Some(Match::must(0, 0..299_592));
    /// assert_eq!(expected, re.find_iter(&mut cache, &haystack).next());
    /// // But you'll get an error if you provide a haystack that's too big.
    /// // Notice that we use the 'try_find_iter' routine instead, which
    /// // yields Result<Match, MatchError> instead of Match.
    /// haystack.push('a');
    /// let expected = Some(Err(MatchError::HaystackTooLong { len: 299_593 }));
    /// assert_eq!(expected, re.try_find_iter(&mut cache, &haystack).next());
    ///
    /// // Unicode inflates the size of the underlying NFA quite a bit, and
    /// // thus means that the backtracker can only handle smaller haystacks,
    /// // assuming that the visited capacity remains unchanged.
    /// let re = BoundedBacktracker::new(r"\w+")?;
    /// assert_eq!(re.max_haystack_len(), 7_036);
    /// // But we can increase the visited capacity to handle bigger haystacks!
    /// let re = BoundedBacktracker::builder()
    ///     .configure(BoundedBacktracker::config().visited_capacity(1<<20))
    ///     .build(r"\w+")?;
    /// assert_eq!(re.max_haystack_len(), 28_148);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn max_haystack_len(&self) -> usize {
        // The capacity given in the config is "bytes of heap memory," but the
        // capacity we use here is "number of bits." So conver the capacity in
        // bytes to the capacity in bits.
        let capacity = 8 * self.get_config().get_visited_capacity();
        let blocks = div_ceil(capacity, Visited::BLOCK_SIZE);
        let real_capacity = blocks * Visited::BLOCK_SIZE;
        (real_capacity / self.nfa.states().len()) - 1
    }
}

impl BoundedBacktracker {
    /// Returns true if and only if this regex matches the given haystack.
    ///
    /// In the case of a backtracking regex engine, and unlike most other
    /// regex engines in this crate, short circuiting isn't possible. However,
    /// this routine may still be faster because it instructs backtracking to
    /// not keep track of any capturing groups.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::nfa::thompson::backtrack::BoundedBacktracker;
    ///
    /// let re = BoundedBacktracker::new("foo[0-9]+bar")?;
    /// let mut cache = re.create_cache();
    ///
    /// assert!(re.is_match(&mut cache, "foo12345bar"));
    /// assert!(!re.is_match(&mut cache, "foobar"));
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

    /// Executes a leftmost forward search and writes the spans of capturing
    /// groups that participated in a match into the provided [`Captures`]
    /// value. If no match was found, then [`Captures::is_match`] is guaranteed
    /// to return `false`.
    ///
    /// For more control over the input parameters, see
    /// [`BoundedBacktracker::try_search`].
    ///
    /// # Example
    ///
    /// Leftmost first match semantics corresponds to the match with the
    /// smallest starting offset, but where the end offset is determined by
    /// preferring earlier branches in the original regular expression. For
    /// example, `Sam|Samwise` will match `Sam` in `Samwise`, but `Samwise|Sam`
    /// will match `Samwise` in `Samwise`.
    ///
    /// Generally speaking, the "leftmost first" match is how most backtracking
    /// regular expressions tend to work. This is in contrast to POSIX-style
    /// regular expressions that yield "leftmost longest" matches. Namely,
    /// both `Sam|Samwise` and `Samwise|Sam` match `Samwise` when using
    /// leftmost longest semantics. (This crate does not currently support
    /// leftmost longest semantics, and this backtracking regex engine will
    /// likely never support it.)
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::backtrack::BoundedBacktracker,
    ///     Match,
    /// };
    ///
    /// let re = BoundedBacktracker::new("foo[0-9]+")?;
    /// let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    /// let expected = Match::must(0, 0..8);
    /// re.find(&mut cache, "foo12345", &mut caps);
    /// assert_eq!(Some(expected), caps.get_match());
    ///
    /// // Even though a match is found after reading the first byte (`a`),
    /// // the leftmost first match semantics demand that we find the earliest
    /// // match that prefers earlier parts of the pattern over later parts.
    /// let re = BoundedBacktracker::new("abc|a")?;
    /// let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    /// let expected = Match::must(0, 0..3);
    /// re.find(&mut cache, "abc", &mut caps);
    /// assert_eq!(Some(expected), caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn find<H: AsRef<[u8]>>(
        &self,
        cache: &mut Cache,
        haystack: H,
        caps: &mut Captures,
    ) {
        self.try_find(cache, haystack.as_ref(), caps).unwrap();
    }

    /// Returns an iterator over all non-overlapping leftmost matches in the
    /// given bytes. If no match exists, then the iterator yields no elements.
    ///
    /// If the regex engine returns an error at any point, then the iterator
    /// will panic.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::backtrack::BoundedBacktracker,
    ///     Match,
    /// };
    ///
    /// let re = BoundedBacktracker::new("foo[0-9]+")?;
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
    pub fn find_iter<'r, 'c, 'h, H: AsRef<[u8]> + ?Sized>(
        &'r self,
        cache: &'c mut Cache,
        haystack: &'h H,
    ) -> FindMatches<'r, 'c, 'h> {
        let input = self.create_input(haystack.as_ref());
        let caps = Captures::new_for_matches_only(self.get_nfa().clone());
        let it = iter::Searcher::new(input);
        FindMatches { re: self, cache, caps, it }
    }

    /// Returns an iterator over all non-overlapping `Captures` values. If no
    /// match exists, then the iterator yields no elements.
    ///
    /// This yields the same matches as [`BoundedBacktracker::find_iter`], but
    /// it includes the spans of all capturing groups that participate in each
    /// match.
    ///
    /// If the regex engine returns an error at any point, then the iterator
    /// will panic.
    ///
    /// **Tip:** See [`util::iter::Searcher`](crate::util::iter::Searcher) for
    /// how to correctly iterate over all matches in a haystack while avoiding
    /// the creation of a new `Captures` value for every match. (Which you are
    /// forced to do with an `Iterator`.)
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::backtrack::BoundedBacktracker,
    ///     Span,
    /// };
    ///
    /// let re = BoundedBacktracker::new("foo(?P<numbers>[0-9]+)")?;
    /// let mut cache = re.create_cache();
    ///
    /// let text = "foo1 foo12 foo123";
    /// let matches: Vec<Span> = re
    ///     .captures_iter(&mut cache, text)
    ///     // The unwrap is OK since 'numbers' matches if the pattern matches.
    ///     .map(|caps| caps.get_group_by_name("numbers").unwrap())
    ///     .collect();
    /// assert_eq!(matches, vec![
    ///     Span::from(3..4),
    ///     Span::from(8..10),
    ///     Span::from(14..17),
    /// ]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn captures_iter<'r, 'c, 'h, H: AsRef<[u8]> + ?Sized>(
        &'r self,
        cache: &'c mut Cache,
        haystack: &'h H,
    ) -> CapturesMatches<'r, 'c, 'h> {
        let input = self.create_input(haystack.as_ref());
        let caps = self.create_captures();
        let it = iter::Searcher::new(input);
        CapturesMatches { re: self, cache, caps, it }
    }
}

impl BoundedBacktracker {
    /// Returns true if and only if this regex matches the given haystack.
    ///
    /// In the case of a backtracking regex engine, and unlike most other
    /// regex engines in this crate, short circuiting isn't possible. However,
    /// this routine may still be faster because it instructs backtracking to
    /// not keep track of any capturing groups.
    ///
    /// # Errors
    ///
    /// This routine only errors if the search could not complete. For this
    /// backtracking regex engine, this only occurs when the haystack length
    /// exceeds [`BoundedBacktracker::max_haystack_len`].
    ///
    /// When a search cannot complete, callers cannot know whether a match
    /// exists or not.
    ///
    /// The infallible (panics on error) version of this routine is
    /// [`is_match`](BoundedBacktracker::is_match).
    #[inline]
    pub fn try_is_match<H: AsRef<[u8]>>(
        &self,
        cache: &mut Cache,
        haystack: H,
    ) -> Result<bool, MatchError> {
        let input = self.create_input(haystack.as_ref()).earliest(true);
        let mut caps = Captures::empty(self.nfa.clone());
        self.try_search(cache, &input, &mut caps)?;
        Ok(caps.is_match())
    }

    /// Executes a leftmost forward search and writes the spans of capturing
    /// groups that participated in a match into the provided [`Captures`]
    /// value. If no match was found, then [`Captures::is_match`] is guaranteed
    /// to return `false`.
    ///
    /// # Errors
    ///
    /// This routine only errors if the search could not complete. For this
    /// backtracking regex engine, this only occurs when the haystack length
    /// exceeds [`BoundedBacktracker::max_haystack_len`].
    ///
    /// When a search cannot complete, callers cannot know whether a match
    /// exists or not.
    ///
    /// The infallible (panics on error) version of this routine is
    /// [`find`](BoundedBacktracker::find).
    #[inline]
    pub fn try_find<H: AsRef<[u8]>>(
        &self,
        cache: &mut Cache,
        haystack: H,
        caps: &mut Captures,
    ) -> Result<(), MatchError> {
        let input = self.create_input(haystack.as_ref());
        self.try_search(cache, &input, caps)
    }

    /// Returns an iterator over all non-overlapping leftmost matches in the
    /// given bytes. If no match exists, then the iterator yields no elements.
    ///
    /// If the regex engine returns an error at any point, then the iterator
    /// will yield that error.
    #[inline]
    pub fn try_find_iter<'r, 'c, 'h, H: AsRef<[u8]> + ?Sized>(
        &'r self,
        cache: &'c mut Cache,
        haystack: &'h H,
    ) -> TryFindMatches<'r, 'c, 'h> {
        let input = self.create_input(haystack.as_ref());
        let caps = Captures::new_for_matches_only(self.get_nfa().clone());
        let it = iter::Searcher::new(input);
        TryFindMatches { re: self, cache, caps, it }
    }

    /// Returns an iterator over all non-overlapping `Captures` values. If no
    /// match exists, then the iterator yields no elements.
    ///
    /// This yields the same matches as [`BoundedBacktracker::try_find_iter`],
    /// but it includes the spans of all capturing groups that participate in
    /// each match.
    ///
    /// If the regex engine returns an error at any point, then the iterator
    /// will yield that error.
    ///
    /// **Tip:** See [`util::iter::Searcher`](crate::util::iter::Searcher) for
    /// how to correctly iterate over all matches in a haystack while avoiding
    /// the creation of a new `Captures` value for every match. (Which you are
    /// forced to do with an `Iterator`.)
    #[inline]
    pub fn try_captures_iter<'r, 'c, 'h, H: AsRef<[u8]> + ?Sized>(
        &'r self,
        cache: &'c mut Cache,
        haystack: &'h H,
    ) -> TryCapturesMatches<'r, 'c, 'h> {
        let input = self.create_input(haystack.as_ref());
        let caps = self.create_captures();
        let it = iter::Searcher::new(input);
        TryCapturesMatches { re: self, cache, caps, it }
    }
}

impl BoundedBacktracker {
    /// Executes a leftmost forward search and writes the spans of capturing
    /// groups that participated in a match into the provided [`Captures`]
    /// value. If no match was found, then [`Captures::is_match`] is guaranteed
    /// to return `false`.
    ///
    /// This is like [`BoundedBacktracker::find`], except it provides some
    /// additional control over how the search is executed. Those parameters
    /// are configured via a [`Input`].
    ///
    /// The examples below demonstrate each of these additional parameters.
    ///
    /// # Example: prefilter
    ///
    /// This example shows how to provide a prefilter for a pattern where all
    /// matches start with a `z` byte.
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::backtrack::BoundedBacktracker,
    ///     util::prefilter::{Candidate, Prefilter},
    ///     Match, Input, Span,
    /// };
    ///
    /// #[derive(Debug)]
    /// pub struct ZPrefilter;
    ///
    /// impl Prefilter for ZPrefilter {
    ///     fn find(
    ///         &self,
    ///         haystack: &[u8],
    ///         span: Span,
    ///     ) -> Candidate {
    ///         // Try changing b'z' to b'q' and observe this test fail since
    ///         // the prefilter will skip right over the match.
    ///         match haystack[span].iter().position(|&b| b == b'z') {
    ///             None => Candidate::None,
    ///             Some(i) => {
    ///                 let start = span.start + i;
    ///                 let span = Span::from(start..start + 1);
    ///                 Candidate::PossibleMatch(span)
    ///             }
    ///         }
    ///     }
    ///
    ///     fn memory_usage(&self) -> usize {
    ///         0
    ///     }
    /// }
    ///
    /// let re = BoundedBacktracker::new("z[0-9]{3}")?;
    /// let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    /// let input = Input::new("foobar z123 q123")
    ///     .prefilter(Some(&ZPrefilter));
    /// let expected = Some(Match::must(0, 7..11));
    /// re.try_search(&mut cache, &input, &mut caps)?;
    /// assert_eq!(expected, caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: specific pattern search
    ///
    /// This example shows how to build a multi bounded backtracker that
    /// permits searching for specific patterns.
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::backtrack::BoundedBacktracker,
    ///     Match, PatternID, Input,
    /// };
    ///
    /// let re = BoundedBacktracker::new_many(&[
    ///     "[a-z0-9]{6}",
    ///     "[a-z][a-z0-9]{5}",
    /// ])?;
    /// let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    /// let haystack = "foo123";
    ///
    /// // Since we are using the default leftmost-first match and both
    /// // patterns match at the same starting position, only the first pattern
    /// // will be returned in this case when doing a search for any of the
    /// // patterns.
    /// let expected = Some(Match::must(0, 0..6));
    /// re.try_search(&mut cache, &Input::new(haystack), &mut caps)?;
    /// assert_eq!(expected, caps.get_match());
    ///
    /// // But if we want to check whether some other pattern matches, then we
    /// // can provide its pattern ID.
    /// let expected = Some(Match::must(1, 0..6));
    /// re.try_search(
    ///     &mut cache,
    ///     &Input::new(haystack).pattern(Some(PatternID::must(1))),
    ///     &mut caps,
    /// )?;
    /// assert_eq!(expected, caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: specifying the bounds of a search
    ///
    /// This example shows how providing the bounds of a search can produce
    /// different results than simply sub-slicing the haystack.
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::backtrack::BoundedBacktracker,
    ///     Match, Input,
    /// };
    ///
    /// let re = BoundedBacktracker::new(r"\b[0-9]{3}\b")?;
    /// let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    /// let haystack = "foo123bar";
    ///
    /// // Since we sub-slice the haystack, the search doesn't know about
    /// // the larger context and assumes that `123` is surrounded by word
    /// // boundaries. And of course, the match position is reported relative
    /// // to the sub-slice as well, which means we get `0..3` instead of
    /// // `3..6`.
    /// let expected = Some(Match::must(0, 0..3));
    /// re.try_search(&mut cache, &Input::new(&haystack[3..6]), &mut caps)?;
    /// assert_eq!(expected, caps.get_match());
    ///
    /// // But if we provide the bounds of the search within the context of the
    /// // entire haystack, then the search can take the surrounding context
    /// // into account. (And if we did find a match, it would be reported
    /// // as a valid offset into `haystack` instead of its sub-slice.)
    /// let expected = None;
    /// re.try_search(
    ///     &mut cache, &Input::new(haystack).range(3..6), &mut caps,
    /// )?;
    /// assert_eq!(expected, caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn try_search(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        caps: &mut Captures,
    ) -> Result<(), MatchError> {
        self.search_imp(cache, input, caps)?;
        let m = match caps.get_match() {
            None => return Ok(()),
            Some(m) => m,
        };
        if m.is_empty() {
            input.skip_empty_utf8_splits(m, |search| {
                self.search_imp(cache, search, caps)?;
                Ok(caps.get_match())
            })?;
        }
        Ok(())
    }

    /// The implementation of standard leftmost backtracking search.
    ///
    /// Capturing group spans are written to 'caps', but only if requested.
    /// 'caps' can be one of three things: 1) totally empty, in which case, we
    /// only report the pattern that matched or 2) only has slots for recording
    /// the overall match offsets for any pattern or 3) has all slots available
    /// for recording the spans of any groups participating in a match.
    fn search_imp(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        caps: &mut Captures,
    ) -> Result<(), MatchError> {
        // Unlike in the PikeVM, we write our capturing group spans directly
        // into the caller's captures groups. So we have to make sure we're
        // starting with a blank slate first. In the PikeVM, we avoid this
        // by construction: the spans that are copied to every slot in the
        // 'Captures' value account for presence/absence.
        caps.clear();
        cache.setup_search(&self.nfa, input)?;
        if input.is_done() {
            return Ok(());
        }
        let anchored = self.config.get_anchored()
            || self.nfa.is_always_start_anchored()
            || input.get_pattern().is_some();
        let start_id = match input.get_pattern() {
            // We always use the anchored starting state here, even if doing an
            // unanchored search. The "unanchored" part of it is implemented
            // in the loop below, by computing the epsilon closure from the
            // anchored starting state whenever the current state set list is
            // empty.
            None => self.nfa.start_anchored(),
            Some(pid) => self.nfa.start_pattern(pid),
        };

        if anchored {
            self.backtrack(cache, input, input.start(), start_id, caps);
            return Ok(());
        }
        for at in input.start()..=input.end() {
            self.backtrack(cache, input, at, start_id, caps);
            if caps.is_match() {
                return Ok(());
            }
        }
        Ok(())
    }

    /// Look for a match starting at `at` in `input` and write the matching
    /// pattern ID and group spans to `caps`. The search uses `start_id` as its
    /// starting state in the underlying NFA.
    ///
    /// If no match was found, then the caller should increment `at` and try
    /// at the next position.
    #[inline(always)]
    fn backtrack(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        at: usize,
        start_id: StateID,
        caps: &mut Captures,
    ) {
        cache.stack.push(Frame::Step { sid: start_id, at });
        while let Some(frame) = cache.stack.pop() {
            match frame {
                Frame::Step { sid, at } => {
                    self.step(cache, input, sid, at, caps);
                    if caps.is_match() {
                        return;
                    }
                }
                Frame::RestoreCapture { slot, offset } => {
                    caps.set_slot(slot, offset.map(|o| o.get()));
                }
            }
        }
    }

    // LAMENTATION: The actual backtracking search is implemented in about
    // 75 lines below. Yet this file is over 2,000 lines long. What have I
    // done?

    /// Execute a "step" in the backtracing algorithm.
    ///
    /// A "step" is somewhat of a misnomer, because this routine keeps going
    /// until it either runs out of things to try or fins a match. In the
    /// former case, it may have pushed some things on to the backtracking
    /// stack, in which case, those will be tried next as part of the
    /// 'backtrack' routine above.
    #[inline(always)]
    fn step(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        mut sid: StateID,
        mut at: usize,
        caps: &mut Captures,
    ) {
        loop {
            if !cache.visited.insert(sid, at) {
                return;
            }
            match *self.nfa.state(sid) {
                State::ByteRange { ref trans } => {
                    if trans.matches(input.haystack(), at) {
                        sid = trans.next;
                        at += 1;
                    }
                }
                State::Sparse(ref sparse) => {
                    if let Some(next) = sparse.matches(input.haystack(), at) {
                        sid = next;
                        at += 1;
                    }
                }
                State::Look { look, next } => {
                    if !look.matches(input.haystack(), at) {
                        return;
                    }
                    sid = next;
                }
                State::Union { ref alternates } => {
                    sid = match alternates.get(0) {
                        None => return,
                        Some(&sid) => sid,
                    };
                    cache.stack.extend(
                        alternates[1..]
                            .iter()
                            .copied()
                            .rev()
                            .map(|sid| Frame::Step { sid, at }),
                    );
                }
                State::BinaryUnion { alt1, alt2 } => {
                    sid = alt1;
                    cache.stack.push(Frame::Step { sid: alt2, at });
                }
                State::Capture { next, slot } => {
                    if slot < caps.slot_len() {
                        cache.stack.push(Frame::RestoreCapture {
                            slot,
                            offset: caps
                                .get_slot(slot)
                                .and_then(NonMaxUsize::new),
                        });
                        caps.set_slot(slot, Some(at));
                    }
                    sid = next;
                }
                State::Fail => return,
                State::Match { pattern_id } => {
                    caps.set_pattern(Some(pattern_id));
                    return;
                }
            }
        }
    }
}

/// An iterator over all non-overlapping matches for an infallible search.
///
/// The iterator yields a [`Match`] value until no more matches could be found.
/// If the underlying regex engine returns an error, then a panic occurs.
///
/// The lifetime parameters are as follows:
///
/// * `'r` represents the lifetime of the BoundedBacktracker.
/// * `'c` represents the lifetime of the BoundedBacktracker's cache.
/// * `'h` represents the lifetime of the haystack being searched.
///
/// This iterator can be created with the [`BoundedBacktracker::find_iter`]
/// method.
#[derive(Debug)]
pub struct FindMatches<'r, 'c, 'h> {
    re: &'r BoundedBacktracker,
    cache: &'c mut Cache,
    caps: Captures,
    it: iter::Searcher<'h, 'r>,
}

impl<'r, 'c, 'h> Iterator for FindMatches<'r, 'c, 'h> {
    type Item = Match;

    #[inline]
    fn next(&mut self) -> Option<Match> {
        // Splitting 'self' apart seems necessary to appease borrowck.
        let FindMatches { re, ref mut cache, ref mut caps, ref mut it } =
            *self;
        it.advance(|input| {
            re.try_search(cache, input, caps)?;
            Ok(caps.get_match())
        })
    }
}

/// An iterator over all non-overlapping leftmost matches, with their capturing
/// groups, for an infallible search.
///
/// The iterator yields a [`Captures`] value until no more matches could be
/// found. If the underlying search returns an error, then this panics.
///
/// The lifetime parameters are as follows:
///
/// * `'r` represents the lifetime of the BoundedBacktracker.
/// * `'c` represents the lifetime of the BoundedBacktracker's cache.
/// * `'h` represents the lifetime of the haystack being searched.
///
/// This iterator can be created with the [`BoundedBacktracker::captures_iter`]
/// method.
#[derive(Debug)]
pub struct CapturesMatches<'r, 'c, 'h> {
    re: &'r BoundedBacktracker,
    cache: &'c mut Cache,
    caps: Captures,
    it: iter::Searcher<'h, 'r>,
}

impl<'r, 'c, 'h> Iterator for CapturesMatches<'r, 'c, 'h> {
    type Item = Captures;

    #[inline]
    fn next(&mut self) -> Option<Captures> {
        // Splitting 'self' apart seems necessary to appease borrowck.
        let CapturesMatches { re, ref mut cache, ref mut caps, ref mut it } =
            *self;
        it.advance(|input| {
            re.try_search(cache, input, caps)?;
            Ok(caps.get_match())
        });
        if caps.is_match() {
            Some(caps.clone())
        } else {
            None
        }
    }
}

/// An iterator over all non-overlapping matches for a fallible search.
///
/// The iterator yields a `Result<Match, MatchError` value until no more
/// matches could be found.
///
/// The lifetime parameters are as follows:
///
/// * `'r` represents the lifetime of the BoundedBacktracker.
/// * `'c` represents the lifetime of the BoundedBacktracker's cache.
/// * `'h` represents the lifetime of the haystack being searched.
///
/// This iterator can be created with the [`BoundedBacktracker::try_find_iter`]
/// method.
#[derive(Debug)]
pub struct TryFindMatches<'r, 'c, 'h> {
    re: &'r BoundedBacktracker,
    cache: &'c mut Cache,
    caps: Captures,
    it: iter::Searcher<'h, 'r>,
}

impl<'r, 'c, 'h> Iterator for TryFindMatches<'r, 'c, 'h> {
    type Item = Result<Match, MatchError>;

    #[inline]
    fn next(&mut self) -> Option<Result<Match, MatchError>> {
        // Splitting 'self' apart seems necessary to appease borrowck.
        let TryFindMatches { re, ref mut cache, ref mut caps, ref mut it } =
            *self;
        it.try_advance(|input| {
            re.try_search(cache, input, caps)?;
            Ok(caps.get_match())
        })
        .transpose()
    }
}

/// An iterator over all non-overlapping leftmost matches, with their capturing
/// groups, for a fallible search.
///
/// The iterator yields a `Result<Captures, MatchError>` value until no more
/// matches could be found.
///
/// The lifetime parameters are as follows:
///
/// * `'r` represents the lifetime of the BoundedBacktracker.
/// * `'c` represents the lifetime of the BoundedBacktracker's cache.
/// * `'h` represents the lifetime of the haystack being searched.
///
/// This iterator can be created with the
/// [`BoundedBacktracker::try_captures_iter`] method.
#[derive(Debug)]
pub struct TryCapturesMatches<'r, 'c, 'h> {
    re: &'r BoundedBacktracker,
    cache: &'c mut Cache,
    caps: Captures,
    it: iter::Searcher<'h, 'r>,
}

impl<'r, 'c, 'h> Iterator for TryCapturesMatches<'r, 'c, 'h> {
    type Item = Result<Captures, MatchError>;

    #[inline]
    fn next(&mut self) -> Option<Result<Captures, MatchError>> {
        // Splitting 'self' apart seems necessary to appease borrowck.
        let TryCapturesMatches { re, ref mut cache, ref mut caps, ref mut it } =
            *self;
        it.try_advance(|input| {
            re.try_search(cache, input, caps)?;
            Ok(caps.get_match())
        })
        .transpose()?;
        if caps.is_match() {
            Some(Ok(caps.clone()))
        } else {
            None
        }
    }
}

/// A cache represents mutable state that a [`BoundedBacktracker`] requires
/// during a search.
///
/// For a given [`BoundedBacktracker`], its corresponding cache may be created
/// either via [`BoundedBacktracker::create_cache`], or via [`Cache::new`].
/// They are equivalent in every way, except the former does not require
/// explicitly importing `Cache`.
///
/// A particular `Cache` is coupled with the [`BoundedBacktracker`] from which
/// it was created. It may only be used with that `BoundedBacktracker`. A cache
/// and its allocations may be re-purposed via [`Cache::reset`], in which case,
/// it can only be used with the new `BoundedBacktracker` (and not the old
/// one).
#[derive(Clone, Debug)]
pub struct Cache {
    /// Stack used on the heap for doing backtracking instead of the
    /// traditional recursive approach. We don't want recursion because then
    /// we're likely to hit a stack overflow for bigger regexes.
    stack: Vec<Frame>,
    /// The set of (StateID, HaystackOffset) pairs that have been visited
    /// by the backtracker within a single search. If such a pair has been
    /// visited, then we avoid doing the work for that pair again. This is
    /// what "bounds" the backtracking and prevents it from having worst case
    /// exponential time.
    visited: Visited,
}

impl Cache {
    /// Create a new [`BoundedBacktracker`] cache.
    ///
    /// A potentially more convenient routine to create a cache is
    /// [`BoundedBacktracker::create_cache`], as it does not require also
    /// importing the `Cache` type.
    ///
    /// If you want to reuse the returned `Cache` with some other
    /// `BoundedBacktracker`, then you must call [`Cache::reset`] with the
    /// desired `BoundedBacktracker`.
    pub fn new(re: &BoundedBacktracker) -> Cache {
        Cache { stack: vec![], visited: Visited::new(re) }
    }

    /// Reset this cache such that it can be used for searching with different
    /// [`BoundedBacktracker`].
    ///
    /// A cache reset permits reusing memory already allocated in this cache
    /// with a different `BoundedBacktracker`.
    ///
    /// # Example
    ///
    /// This shows how to re-purpose a cache for use with a different
    /// `BoundedBacktracker`.
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::backtrack::BoundedBacktracker, Match};
    ///
    /// let re1 = BoundedBacktracker::new(r"\w")?;
    /// let re2 = BoundedBacktracker::new(r"\W")?;
    ///
    /// let mut cache = re1.create_cache();
    /// assert_eq!(
    ///     Some(Match::must(0, 0..2)),
    ///     re1.find_iter(&mut cache, "Δ").next(),
    /// );
    ///
    /// // Using 'cache' with re2 is not allowed. It may result in panics or
    /// // incorrect results. In order to re-purpose the cache, we must reset
    /// // it with the BoundedBacktracker we'd like to use it with.
    /// //
    /// // Similarly, after this reset, using the cache with 're1' is also not
    /// // allowed.
    /// cache.reset(&re2);
    /// assert_eq!(
    ///     Some(Match::must(0, 0..3)),
    ///     re2.find_iter(&mut cache, "☃").next(),
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reset(&mut self, re: &BoundedBacktracker) {
        self.visited.reset(re);
    }

    /// Returns the heap memory usage, in bytes, of this cache.
    ///
    /// This does **not** include the stack size used up by this cache. To
    /// compute that, use `std::mem::size_of::<Cache>()`.
    pub fn memory_usage(&self) -> usize {
        self.stack.len() * core::mem::size_of::<Frame>()
            + self.visited.memory_usage()
    }

    /// Clears this cache. This should be called at the start of every search
    /// to ensure we start with a clean slate.
    ///
    /// This also sets the length of the capturing groups used in the current
    /// search. This permits an optimization where by 'SlotTable::for_state'
    /// only returns the number of slots equivalent to the number of slots
    /// given in the 'Captures' value. This may be less than the total number
    /// of possible slots, e.g., when one only wants to track overall match
    /// offsets. This in turn permits less copying of capturing group spans
    /// in the BoundedBacktracker.
    fn setup_search(
        &mut self,
        nfa: &NFA,
        input: &Input<'_, '_>,
    ) -> Result<(), MatchError> {
        self.stack.clear();
        self.visited.setup_search(nfa, input)?;
        Ok(())
    }
}

/// Represents a stack frame on the heap while doing backtracking.
///
/// Instead of using explicit recursion for backtracking, we use a stack on
/// the heap to keep track of things that we want to explore if the current
/// backtracking branch turns out to not lead to a match.
#[derive(Clone, Debug)]
enum Frame {
    /// Look for a match starting at `sid` and the given position in the
    /// haystack.
    Step { sid: StateID, at: usize },
    /// Reset the given `slot` to the given `offset` (which might be `None`).
    /// This effectively gives a "scope" to capturing groups, such that an
    /// offset for a particular group only gets returned if the match goes
    /// through that capturing group. If backtracking ends up going down a
    /// different branch that results in a different offset (or perhaps none at
    /// all), then this "restore capture" frame will cause the offset to get
    /// reset.
    RestoreCapture { slot: usize, offset: Option<NonMaxUsize> },
}

/// A bitset that keeps track of whether a particular (StateID, offset) has
/// been considered during backtracking. If it has already been visited, then
/// backtracking skips it. This is what gives backtracking its "bound."
#[derive(Clone, Debug)]
struct Visited {
    /// The actual underlying bitset. Each element in the bitset corresponds
    /// to a particular (StateID, offset) pair. States correspond to the rows
    /// and the offsets correspond to the columns.
    ///
    /// If our underlying NFA has N states and the haystack we're searching
    /// has M bytes, then we have N*(M+1) entries in our bitset table. The
    /// M+1 occurs because our matches are delayed by one byte (to support
    /// look-around), and so we need to handle the end position itself rather
    /// than stopping just before the end. (If there is no end position, then
    /// it's treated as "end-of-input," which is matched by things like '$'.)
    ///
    /// Given BITS=N*(M+1), we wind up with div_ceil(BITS, sizeof(usize))
    /// blocks.
    ///
    /// We use 'usize' to represent our blocks because it makes some of the
    /// arithmetic in 'insert' a bit nicer. For example, if we used 'u32' for
    /// our block, we'd either need to cast u32s to usizes or usizes to u32s.
    bitset: Vec<usize>,
    /// The stride represents one plus length of the haystack we're searching
    /// (as described above). The stride must be initialized for each search.
    stride: usize,
}

impl Visited {
    /// The size of each block, in bits.
    const BLOCK_SIZE: usize = 8 * core::mem::size_of::<usize>();

    /// Create a new visited set for the given backtracker.
    ///
    /// The set is ready to use, but must be setup at the beginning of each
    /// search by calling `setup_search`.
    fn new(re: &BoundedBacktracker) -> Visited {
        let mut visited = Visited { bitset: vec![], stride: 0 };
        visited.reset(re);
        visited
    }

    /// Insert the given (StateID, offset) pair into this set. If it already
    /// exists, then this is a no-op and it returns false. Otherwise this
    /// returns true.
    fn insert(&mut self, sid: StateID, at: usize) -> bool {
        let table_index = sid.as_usize() * self.stride + at;
        let block_index = table_index / Visited::BLOCK_SIZE;
        let bit = table_index % Visited::BLOCK_SIZE;
        let block_with_bit = 1 << bit;
        if self.bitset[block_index] & block_with_bit != 0 {
            return false;
        }
        self.bitset[block_index] |= block_with_bit;
        true
    }

    /// Returns the capacity of this visited set in terms of the number of bits
    /// it has to track (StateID, offset) pairs.
    fn capacity(&self) -> usize {
        self.bitset.len() * Visited::BLOCK_SIZE
    }

    /// Reset this visited set to work with the given bounded backtracker.
    fn reset(&mut self, re: &BoundedBacktracker) {
        // The capacity given in the config is "bytes of heap memory," but the
        // capacity we use here is "number of bits." So conver the capacity in
        // bytes to the capacity in bits.
        let capacity = 8 * re.get_config().get_visited_capacity();
        let blocks = div_ceil(capacity, Visited::BLOCK_SIZE);
        self.bitset.resize(blocks, 0);
        // N.B. 'stride' is set in 'setup_search', since it isn't known until
        // we know the length of the haystack. (That is also when we return an
        // error if the haystack is too big.)
    }

    /// Setup this visited set to work for a search using the given NFA
    /// and input configuration. The NFA must be the same NFA used by the
    /// BoundedBacktracker given to Visited::reset. Failing to call this might
    /// result in panics or silently incorrect search behavior.
    fn setup_search(
        &mut self,
        nfa: &NFA,
        input: &Input<'_, '_>,
    ) -> Result<(), MatchError> {
        let haylen = input.haystack().len();
        let err = || MatchError::HaystackTooLong { len: haylen };
        // Our stride is one more than the length of the input because our main
        // search loop includes the position at input.haystack().len(). (And
        // it does this because matches are delayed by one byte to account for
        // look-around.)
        self.stride = haylen + 1;
        let capacity = match nfa.states().len().checked_mul(self.stride) {
            None => return Err(err()),
            Some(capacity) => capacity,
        };
        if capacity > self.capacity() {
            return Err(err());
        }
        // We only need to zero out our desired capacity, not our total
        // capacity in this set.
        let blocks = div_ceil(capacity, Visited::BLOCK_SIZE);
        for block in self.bitset.iter_mut().take(blocks) {
            *block = 0;
        }
        Ok(())
    }

    /// Return the heap memory usage, in bytes, of this visited set.
    fn memory_usage(&self) -> usize {
        self.bitset.len() * core::mem::size_of::<usize>()
    }
}

/// Integer division, but rounds up instead of down.
fn div_ceil(lhs: usize, rhs: usize) -> usize {
    if lhs % rhs == 0 {
        lhs / rhs
    } else {
        (lhs / rhs) + 1
    }
}