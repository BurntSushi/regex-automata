/*!
An NFA backed Pike VM for executing regex searches with capturing groups.

This module provides a [`PikeVM`] that works by simulating an NFA and
resolving all spans of capturing groups that participate in a match.
*/

use core::cell::RefCell;

use alloc::{
    sync::{Arc, Weak},
    vec,
    vec::Vec,
};

use crate::{
    nfa::thompson::{self, Captures, Error, State, NFA},
    util::{
        id::{PatternID, StateID},
        iter,
        nonmax::NonMaxUsize,
        prefilter::Prefilter,
        search::{Input, Match, MatchError, MatchKind, PatternSet},
        sparse_set::SparseSet,
    },
};

/// A simple macro for conditionally executing instrumentation logic when
/// the 'trace' log level is enabled. This is a compile-time no-op when the
/// 'instrument-pikevm' feature isn't enabled. The intent here is that this
/// makes it easier to avoid doing extra work when instrumentation isn't
/// enabled.
///
/// This macro accepts a closure of type `|&mut Counters|`. The closure can
/// then increment counters (or whatever) in accordance with what one wants
/// to track.
macro_rules! instrument {
    ($fun:expr) => {
        #[cfg(feature = "instrument-pikevm")]
        {
            // let fun: impl FnMut(&mut Counters) = $fun;
            let mut fun: &mut dyn FnMut(&mut Counters) = &mut $fun;
            COUNTERS.with(|c: &RefCell<Counters>| fun(&mut *c.borrow_mut()));
        }
    };
}

/// Effectively global state used to keep track of instrumentation counters.
/// The "proper" way to do this is to thread it through the PikeVM, but it
/// makes the code quite icky. Since this is just a debugging feature, we're
/// content to relegate it to thread local state. When instrumentation is
/// enabled, the counters are reset at the beginning of every search and
/// printed (with the 'trace' log level) at the end of every search.
#[cfg(feature = "instrument-pikevm")]
thread_local! {
    static COUNTERS: RefCell<Counters> = RefCell::new(Counters::empty());
}

/// The configuration used for building a `PikeVM`.
///
/// A `PikeVM` configuration is a simple data object that is typically used
/// with [`Builder::configure`].
#[derive(Clone, Debug, Default)]
pub struct Config {
    anchored: Option<bool>,
    match_kind: Option<MatchKind>,
    utf8: Option<bool>,
    pre: Option<Option<Arc<dyn Prefilter>>>,
}

impl Config {
    /// Return a new default PikeVM configuration.
    pub fn new() -> Config {
        Config::default()
    }

    /// Set whether matching must be anchored at the beginning of the input.
    ///
    /// When enabled, a match must begin at the start of a search. When
    /// disabled (the default), the `PikeVM` will act as if the pattern started
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
    /// use regex_automata::{nfa::thompson::pikevm::PikeVM, Match, Input};
    ///
    /// let haystack = "aba";
    ///
    /// let vm = PikeVM::builder()
    ///     .configure(PikeVM::config().anchored(false)) // default
    ///     .build(r"^a")?;
    /// let (mut cache, mut caps) = (vm.create_cache(), vm.create_captures());
    /// vm.search(&mut cache, &Input::new(haystack).span(2..3), &mut caps);
    /// // No match is found because 2 is not the beginning of the haystack,
    /// // which is what ^ requires.
    /// let expected = None;
    /// assert_eq!(expected, caps.get_match());
    ///
    /// let vm = PikeVM::builder()
    ///     .configure(PikeVM::config().anchored(true))
    ///     .build(r"a")?;
    /// let (mut cache, mut caps) = (vm.create_cache(), vm.create_captures());
    /// vm.search(&mut cache, &Input::new(haystack).span(2..3), &mut caps);
    /// // An anchored search can still match anywhere in the haystack, it just
    /// // must begin at the start of the search which is '2' in this case.
    /// let expected = Some(Match::must(0, 2..3));
    /// assert_eq!(expected, caps.get_match());
    ///
    /// let vm = PikeVM::builder()
    ///     .configure(PikeVM::config().anchored(true))
    ///     .build(r"a")?;
    /// let (mut cache, mut caps) = (vm.create_cache(), vm.create_captures());
    /// vm.search(&mut cache, &Input::new(haystack).span(1..3), &mut caps);
    /// // No match is found since we start searching at offset 1 which
    /// // corresponds to 'b'. Since there is no '(?s:.)*?' prefix, no match
    /// // is found.
    /// let expected = None;
    /// assert_eq!(expected, caps.get_match());
    ///
    /// let vm = PikeVM::builder()
    ///     .configure(PikeVM::config().anchored(false))
    ///     .build(r"a")?;
    /// let (mut cache, mut caps) = (vm.create_cache(), vm.create_captures());
    /// vm.search(&mut cache, &Input::new(haystack).span(1..3), &mut caps);
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

    /// Set the desired match semantics.
    ///
    /// The default is [`MatchKind::LeftmostFirst`], which corresponds to the
    /// match semantics of Perl-like regex engines. That is, when multiple
    /// patterns would match at the same leftmost position, the pattern that
    /// appears first in the concrete syntax is chosen.
    ///
    /// Currently, the only other kind of match semantics supported is
    /// [`MatchKind::All`]. This corresponds to "classical DFA" construction
    /// where all possible matches are visited in the NFA by the `PikeVM`.
    ///
    /// Typically, `All` is used when one wants to execute an overlapping
    /// search and `LeftmostFirst` otherwise. In particular, it rarely makes
    /// sense to use `All` with the various "leftmost" find routines, since the
    /// leftmost routines depend on the `LeftmostFirst` automata construction
    /// strategy. Specifically, `LeftmostFirst` results in the `PikeVM`
    /// simulating dead states as a way to terminate the search and report a
    /// match. `LeftmostFirst` also supports non-greedy matches using this
    /// strategy where as `All` does not.
    pub fn match_kind(mut self, kind: MatchKind) -> Config {
        self.match_kind = Some(kind);
        self
    }

    /// Whether to enable UTF-8 mode or not.
    ///
    /// When UTF-8 mode is enabled (the default) and an empty match is seen,
    /// the search APIs of [`PikeVM`] will always start the next search at the
    /// next UTF-8 encoded codepoint when searching valid UTF-8. When UTF-8
    /// mode is disabled, such searches are begun at the next byte offset.
    ///
    /// If this mode is enabled and invalid UTF-8 is given to search, then
    /// behavior is unspecified.
    ///
    /// Generally speaking, one should enable this when
    /// [`SyntaxConfig::utf8`](crate::SyntaxConfig::utf8)
    /// is enabled, and disable it otherwise.
    ///
    /// # Example
    ///
    /// This example demonstrates the differences between when this option is
    /// enabled and disabled. The differences only arise when the `PikeVM` can
    /// return matches of length zero.
    ///
    /// In this first snippet, we show the results when UTF-8 mode is disabled.
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::pikevm::PikeVM, Match};
    ///
    /// let vm = PikeVM::builder()
    ///     .configure(PikeVM::config().utf8(false))
    ///     .build(r"")?;
    /// let mut cache = vm.create_cache();
    ///
    /// let haystack = "a☃z";
    /// let mut it = vm.find_iter(&mut cache, haystack);
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
    /// use regex_automata::{nfa::thompson::pikevm::PikeVM, Match};
    ///
    /// let vm = PikeVM::builder()
    ///     .configure(PikeVM::config().utf8(true))
    ///     .build(r"")?;
    /// let mut cache = vm.create_cache();
    ///
    /// let haystack = "a☃z";
    /// let mut it = vm.find_iter(&mut cache, haystack);
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
    /// The given prefilter is automatically applied to every search done by a
    /// `PikeVM`, except for the lower level routines that accept a prefilter
    /// parameter from the caller.
    pub fn prefilter(mut self, pre: Option<Arc<dyn Prefilter>>) -> Config {
        self.pre = Some(pre);
        self
    }

    /// Returns whether this configuration has enabled anchored searches.
    pub fn get_anchored(&self) -> bool {
        self.anchored.unwrap_or(false)
    }

    /// Returns the match semantics set in this configuration.
    pub fn get_match_kind(&self) -> MatchKind {
        self.match_kind.unwrap_or(MatchKind::LeftmostFirst)
    }

    /// Returns true if and only if this configuration has UTF-8 mode enabled.
    ///
    /// When UTF-8 mode is enabled and an empty match is seen, [`PikeVM`] will
    /// always start the next search at the next UTF-8 encoded codepoint.
    /// When UTF-8 mode is disabled, such searches are begun at the next byte
    /// offset.
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
            anchored: o.anchored.or(self.anchored),
            match_kind: o.match_kind.or(self.match_kind),
            utf8: o.utf8.or(self.utf8),
            pre: o.pre.or_else(|| self.pre.clone()),
        }
    }
}

/// A builder for a `PikeVM`.
///
/// This builder permits configuring options for the syntax of a pattern,
/// the NFA construction and the `PikeVM` construction. This builder is
/// different from a general purpose regex builder in that it permits fine
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
/// use regex_automata::{nfa::thompson::pikevm::PikeVM, Match, SyntaxConfig};
///
/// let vm = PikeVM::builder()
///     .configure(PikeVM::config().utf8(false))
///     .syntax(SyntaxConfig::new().utf8(false))
///     .build(r"foo(?-u:[^b])ar.*")?;
/// let mut cache = vm.create_cache();
///
/// let haystack = b"\xFEfoo\xFFarzz\xE2\x98\xFF\n";
/// let expected = Some(Match::must(0, 1..9));
/// let got = vm.find_iter(&mut cache, haystack).next();
/// assert_eq!(expected, got);
/// // Notice that `(?-u:[^b])` matches invalid UTF-8,
/// // but the subsequent `.*` does not! Disabling UTF-8
/// // on the syntax permits this.
/// //
/// // N.B. This example does not show the impact of
/// // disabling UTF-8 mode on a PikeVM Config, since that
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
    /// Create a new PikeVM builder with its default configuration.
    pub fn new() -> Builder {
        Builder {
            config: Config::default(),
            thompson: thompson::Compiler::new(),
        }
    }

    /// Build a `PikeVM` from the given pattern.
    ///
    /// If there was a problem parsing or compiling the pattern, then an error
    /// is returned.
    pub fn build(&self, pattern: &str) -> Result<PikeVM, Error> {
        self.build_many(&[pattern])
    }

    /// Build a `PikeVM` from the given patterns.
    pub fn build_many<P: AsRef<str>>(
        &self,
        patterns: &[P],
    ) -> Result<PikeVM, Error> {
        let nfa = self.thompson.build_many(patterns)?;
        self.build_from_nfa(nfa)
    }

    /// Build a `PikeVM` directly from its NFA.
    ///
    /// Note that when using this method, any configuration that applies to the
    /// construction of the NFA itself will of course be ignored, since the NFA
    /// given here is already built.
    pub fn build_from_nfa(&self, nfa: NFA) -> Result<PikeVM, Error> {
        // TODO: Check that this is correct.
        // if !cfg!(all(
        // feature = "dfa",
        // feature = "syntax",
        // feature = "unicode-perl"
        // )) {
        // If the NFA has no captures, then the PikeVM doesn't work since it
        // relies on them in order to report match locations. However, in
        // the special case of an NFA with no patterns, it is allowed, since
        // no matches can ever be produced. And importantly, an NFA with no
        // patterns has no capturing groups anyway, so this is necessary to
        // permit the PikeVM to work with regexes with zero patterns.
        if !nfa.has_capture() && nfa.pattern_len() > 0 {
            return Err(Error::missing_captures());
        }
        if !cfg!(feature = "syntax") {
            if nfa.has_word_boundary_unicode() {
                return Err(Error::unicode_word_unavailable());
            }
        }
        Ok(PikeVM { config: self.config.clone(), nfa })
    }

    /// Apply the given `PikeVM` configuration options to this builder.
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
    /// These settings only apply when constructing a PikeVM directly from a
    /// pattern.
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
    /// These settings only apply when constructing a PikeVM directly from a
    /// pattern.
    pub fn thompson(&mut self, config: thompson::Config) -> &mut Builder {
        self.thompson.configure(config);
        self
    }
}

/// A virtual machine for executing regex searches with capturing groups.
///
/// # Advice
///
/// The `PikeVM` is generally the most "powerful" regex engine in this crate.
/// "Powerful" in this context means that it can handle any regular expression
/// that is parseable by `regex-syntax` and any size haystack. Regretably,
/// the `PikeVM` is also simultaneously often the _slowest_ regex engine in
/// practice. This results in an annoying situation where one generally tries
/// to pick any other regex engine (or perhaps none at all) before being
/// forced to fall back to a `PikeVM`.
///
/// For example, a common strategy for dealing with capturing groups is to
/// actually look for the overall match of the regex using a faster regex
/// engine, like a [lazy DFA](crate::hybrid::regex::Regex). Once the overall
/// match is found, one can then run the `PikeVM` on just the match span to
/// find the spans of the capturing groups. In this way, the faster regex
/// engine does the majority of the work, while the `PikeVM` only lends its
/// power in a more limited role.
///
/// Unfortunately, this isn't always possible because the faster regex engines
/// don't support all of the regex features in `regex-syntax`. This notably
/// includes (and is currently limited to) Unicode word boundaries. So if your
/// pattern has Unicode word boundaries, you typically can't use a DFA-based
/// regex engine at all (unless you
/// [enable heuristic support for it](crate::hybrid::dfa::Config::unicode_word_boundary)).
///
/// # Example
///
/// This example shows that the `PikeVM` implements Unicode word boundaries
/// correctly by default.
///
/// ```
/// use regex_automata::{nfa::thompson::pikevm::PikeVM, Match};
///
/// let vm = PikeVM::new(r"\b\w+\b")?;
/// let mut cache = vm.create_cache();
///
/// let mut it = vm.find_iter(&mut cache, "Шерлок Холмс");
/// assert_eq!(Some(Match::must(0, 0..12)), it.next());
/// assert_eq!(Some(Match::must(0, 13..23)), it.next());
/// assert_eq!(None, it.next());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug)]
pub struct PikeVM {
    config: Config,
    nfa: NFA,
}

impl PikeVM {
    /// Parse the given regular expression using the default configuration and
    /// return the corresponding `PikeVM`.
    ///
    /// If you want a non-default configuration, then use the [`Builder`] to
    /// set your own configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::pikevm::PikeVM, Match};
    ///
    /// let vm = PikeVM::new("foo[0-9]+bar")?;
    /// let mut cache = vm.create_cache();
    /// assert_eq!(
    ///     Some(Match::must(0, 3..14)),
    ///     vm.find_iter(&mut cache, "zzzfoo12345barzzz").next(),
    /// );
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(pattern: &str) -> Result<PikeVM, Error> {
        PikeVM::builder().build(pattern)
    }

    /// Like `new`, but parses multiple patterns into a single "multi regex."
    /// This similarly uses the default regex configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::pikevm::PikeVM, Match};
    ///
    /// let vm = PikeVM::new_many(&["[a-z]+", "[0-9]+"])?;
    /// let mut cache = vm.create_cache();
    ///
    /// let mut it = vm.find_iter(&mut cache, "abc 1 foo 4567 0 quux");
    /// assert_eq!(Some(Match::must(0, 0..3)), it.next());
    /// assert_eq!(Some(Match::must(1, 4..5)), it.next());
    /// assert_eq!(Some(Match::must(0, 6..9)), it.next());
    /// assert_eq!(Some(Match::must(1, 10..14)), it.next());
    /// assert_eq!(Some(Match::must(1, 15..16)), it.next());
    /// assert_eq!(Some(Match::must(0, 17..21)), it.next());
    /// assert_eq!(None, it.next());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new_many<P: AsRef<str>>(patterns: &[P]) -> Result<PikeVM, Error> {
        PikeVM::builder().build_many(patterns)
    }

    /// # Example
    ///
    /// This shows how to hand assemble a regular expression via its HIR,
    /// compile an NFA from it and build a PikeVM from the NFA.
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::{NFA, pikevm::PikeVM}, Match};
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
    /// let vm = PikeVM::new_from_nfa(nfa)?;
    /// let (mut cache, mut caps) = (vm.create_cache(), vm.create_captures());
    /// let expected = Some(Match::must(0, 3..4));
    /// vm.find(&mut cache, "!@#A#@!", &mut caps);
    /// assert_eq!(expected, caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new_from_nfa(nfa: NFA) -> Result<PikeVM, Error> {
        PikeVM::builder().build_from_nfa(nfa)
    }

    /// Create a new `PikeVM` that matches every input.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::pikevm::PikeVM, Match};
    ///
    /// let vm = PikeVM::always_match()?;
    /// let mut cache = vm.create_cache();
    ///
    /// let expected = Match::must(0, 0..0);
    /// assert_eq!(Some(expected), vm.find_iter(&mut cache, "").next());
    /// assert_eq!(Some(expected), vm.find_iter(&mut cache, "foo").next());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn always_match() -> Result<PikeVM, Error> {
        let nfa = thompson::NFA::always_match();
        PikeVM::new_from_nfa(nfa)
    }

    /// Create a new `PikeVM` that never matches any input.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::nfa::thompson::pikevm::PikeVM;
    ///
    /// let vm = PikeVM::never_match()?;
    /// let mut cache = vm.create_cache();
    ///
    /// assert_eq!(None, vm.find_iter(&mut cache, "").next());
    /// assert_eq!(None, vm.find_iter(&mut cache, "foo").next());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn never_match() -> Result<PikeVM, Error> {
        let nfa = thompson::NFA::never_match();
        PikeVM::new_from_nfa(nfa)
    }

    /// Return a default configuration for a `PikeVM`.
    ///
    /// This is a convenience routine to avoid needing to import the `Config`
    /// type when customizing the construction of a `PikeVM`.
    ///
    /// # Example
    ///
    /// This example shows how to disable UTF-8 mode for `PikeVM` searches.
    /// When UTF-8 mode is disabled, the position immediately following an
    /// empty match is where the next search begins, instead of the next
    /// position of a UTF-8 encoded codepoint.
    ///
    /// In the code below, notice that `""` is permitted to match positions
    /// that split the encoding of a codepoint. When the [`Config::utf8`]
    /// option is disabled, those positions are not reported.
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::pikevm::PikeVM, Match};
    ///
    /// let vm = PikeVM::builder()
    ///     .configure(PikeVM::config().utf8(false))
    ///     .build(r"")?;
    /// let mut cache = vm.create_cache();
    ///
    /// let haystack = "a☃z";
    /// let mut it = vm.find_iter(&mut cache, haystack);
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

    /// Return a builder for configuring the construction of a `PikeVM`.
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
    ///     nfa::thompson::pikevm::PikeVM,
    ///     Match, SyntaxConfig,
    /// };
    ///
    /// let vm = PikeVM::builder()
    ///     .configure(PikeVM::config().utf8(false))
    ///     .syntax(SyntaxConfig::new().utf8(false))
    ///     .build(r"foo(?-u:[^b])ar.*")?;
    /// let (mut cache, mut caps) = (vm.create_cache(), vm.create_captures());
    ///
    /// let haystack = b"\xFEfoo\xFFarzz\xE2\x98\xFF\n";
    /// let expected = Some(Match::must(0, 1..9));
    /// vm.find(&mut cache, haystack, &mut caps);
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
    /// `PikeVM`. For example, if this `PikeVM` was built with [`Config::utf8`]
    /// enabled, then the `Input` returned will also have its [`Input::utf8`]
    /// knob enabled.
    ///
    /// This routine is useful when using the lower-level [`PikeVM::search`]
    /// API.
    pub fn create_input<'h, 'p, H: ?Sized + AsRef<[u8]>>(
        &'p self,
        haystack: &'h H,
    ) -> Input<'h, 'p> {
        let c = self.get_config();
        Input::new(haystack.as_ref())
            .prefilter(c.get_prefilter())
            .utf8(c.get_utf8())
    }

    /// Create a new cache for this `PikeVM`.
    ///
    /// The cache returned should only be used for searches for this
    /// `PikeVM`. If you want to reuse the cache for another `PikeVM`, then
    /// you must call [`Cache::reset`] with that `PikeVM` (or, equivalently,
    /// [`PikeVM::reset_cache`]).
    pub fn create_cache(&self) -> Cache {
        Cache::new(self)
    }

    /// Create a new empty set of capturing groups that is guaranteed to be
    /// valid for the search APIs on this `PikeVM`.
    ///
    /// A `Captures` value created for a specific `PikeVM` cannot be used with
    /// any other `PikeVM`.
    ///
    /// See the [`Captures`] documentation for an explanation of its
    /// alternative constructors that permit the `PikeVM` to do less work
    /// during a search, and thus might make it faster.
    pub fn create_captures(&self) -> Captures {
        Captures::new(self.get_nfa().clone())
    }

    /// Reset the given cache such that it can be used for searching with the
    /// this `PikeVM` (and only this `PikeVM`).
    ///
    /// A cache reset permits reusing memory already allocated in this cache
    /// with a different `PikeVM`.
    ///
    /// # Example
    ///
    /// This shows how to re-purpose a cache for use with a different `PikeVM`.
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::pikevm::PikeVM, Match};
    ///
    /// let re1 = PikeVM::new(r"\w")?;
    /// let re2 = PikeVM::new(r"\W")?;
    ///
    /// let mut cache = re1.create_cache();
    /// assert_eq!(
    ///     Some(Match::must(0, 0..2)),
    ///     re1.find_iter(&mut cache, "Δ").next(),
    /// );
    ///
    /// // Using 'cache' with re2 is not allowed. It may result in panics or
    /// // incorrect results. In order to re-purpose the cache, we must reset
    /// // it with the PikeVM we'd like to use it with.
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

    /// Returns the total number of patterns compiled into this `PikeVM`.
    ///
    /// In the case of a `PikeVM` that contains no patterns, this returns `0`.
    ///
    /// # Example
    ///
    /// This example shows the pattern length for a `PikeVM` that never
    /// matches:
    ///
    /// ```
    /// use regex_automata::nfa::thompson::pikevm::PikeVM;
    ///
    /// let vm = PikeVM::never_match()?;
    /// assert_eq!(vm.pattern_len(), 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// And another example for a `PikeVM` that matches at every position:
    ///
    /// ```
    /// use regex_automata::nfa::thompson::pikevm::PikeVM;
    ///
    /// let vm = PikeVM::always_match()?;
    /// assert_eq!(vm.pattern_len(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// And finally, a `PikeVM` that was constructed from multiple patterns:
    ///
    /// ```
    /// use regex_automata::nfa::thompson::pikevm::PikeVM;
    ///
    /// let vm = PikeVM::new_many(&["[0-9]+", "[a-z]+", "[A-Z]+"])?;
    /// assert_eq!(vm.pattern_len(), 3);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn pattern_len(&self) -> usize {
        self.nfa.pattern_len()
    }

    /// Return the config for this `PikeVM`.
    #[inline]
    pub fn get_config(&self) -> &Config {
        &self.config
    }

    /// Returns a reference to the underlying NFA.
    #[inline]
    pub fn get_nfa(&self) -> &NFA {
        &self.nfa
    }
}

impl PikeVM {
    /// Returns true if and only if this `PikeVM` matches the given haystack.
    ///
    /// This routine may short circuit if it knows that scanning future
    /// input will never lead to a different result. In particular, if the
    /// underlying NFA enters a match state, then this routine will return
    /// `true` immediately without inspecting any future input. (Consider how
    /// this might make a difference given the regex `a+` on the haystack
    /// `aaaaaaaaaaaaaaa`. This routine can stop after it sees the first `a`,
    /// but routines like `find` need to continue searching because `+` is
    /// greedy by default.)
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::nfa::thompson::pikevm::PikeVM;
    ///
    /// let vm = PikeVM::new("foo[0-9]+bar")?;
    /// let mut cache = vm.create_cache();
    ///
    /// assert!(vm.is_match(&mut cache, "foo12345bar"));
    /// assert!(!vm.is_match(&mut cache, "foobar"));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn is_match<H: AsRef<[u8]>>(
        &self,
        cache: &mut Cache,
        haystack: H,
    ) -> bool {
        let input = self.create_input(haystack.as_ref()).earliest(true);
        let mut caps = Captures::empty(self.nfa.clone());
        self.search(cache, &input, &mut caps);
        caps.is_match()
    }

    /// Executes a leftmost forward search and writes the spans of capturing
    /// groups that participated in a match into the provided [`Captures`]
    /// value. If no match was found, then [`Captures::is_match`] is guaranteed
    /// to return `false`.
    ///
    /// For more control over the input parameters, see [`PikeVM::search`].
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
    /// leftmost longest semantics.)
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::pikevm::PikeVM, Match};
    ///
    /// let vm = PikeVM::new("foo[0-9]+")?;
    /// let (mut cache, mut caps) = (vm.create_cache(), vm.create_captures());
    /// let expected = Match::must(0, 0..8);
    /// vm.find(&mut cache, "foo12345", &mut caps);
    /// assert_eq!(Some(expected), caps.get_match());
    ///
    /// // Even though a match is found after reading the first byte (`a`),
    /// // the leftmost first match semantics demand that we find the earliest
    /// // match that prefers earlier parts of the pattern over later parts.
    /// let vm = PikeVM::new("abc|a")?;
    /// let (mut cache, mut caps) = (vm.create_cache(), vm.create_captures());
    /// let expected = Match::must(0, 0..3);
    /// vm.find(&mut cache, "abc", &mut caps);
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
        let input = self.create_input(haystack.as_ref());
        self.search(cache, &input, caps)
    }

    /// Returns an iterator over all non-overlapping leftmost matches in the
    /// given bytes. If no match exists, then the iterator yields no elements.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::pikevm::PikeVM, Match};
    ///
    /// let vm = PikeVM::new("foo[0-9]+")?;
    /// let mut cache = vm.create_cache();
    ///
    /// let text = "foo1 foo12 foo123";
    /// let matches: Vec<Match> = vm.find_iter(&mut cache, text).collect();
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
    /// This yields the same matches as [`PikeVM::find_iter`], but it includes
    /// the spans of all capturing groups that participate in each match.
    ///
    /// **Tip:** See [`util::iter::Searcher`](crate::util::iter::Searcher) for
    /// how to correctly iterate over all matches in a haystack while avoiding
    /// the creation of a new `Captures` value for every match. (Which you are
    /// forced to do with an `Iterator`.)
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::pikevm::PikeVM, Span};
    ///
    /// let vm = PikeVM::new("foo(?P<numbers>[0-9]+)")?;
    /// let mut cache = vm.create_cache();
    ///
    /// let text = "foo1 foo12 foo123";
    /// let matches: Vec<Span> = vm
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

    /// Executes a leftmost forward search and writes the spans of capturing
    /// groups that participated in a match into the provided [`Captures`]
    /// value. If no match was found, then [`Captures::is_match`] is guaranteed
    /// to return `false`.
    ///
    /// This is like [`PikeVM::find`], except it provides some additional
    /// control over how the search is executed. Those parameters are
    /// configured via a [`Input`].
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
    ///     nfa::thompson::pikevm::PikeVM,
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
    /// let vm = PikeVM::new("z[0-9]{3}")?;
    /// let (mut cache, mut caps) = (vm.create_cache(), vm.create_captures());
    /// let input = Input::new("foobar z123 q123")
    ///     .prefilter(Some(&ZPrefilter));
    /// let expected = Some(Match::must(0, 7..11));
    /// vm.search(&mut cache, &input, &mut caps);
    /// assert_eq!(expected, caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: specific pattern search
    ///
    /// This example shows how to build a multi-PikeVM that permits searching
    /// for specific patterns.
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::pikevm::PikeVM,
    ///     Match, PatternID, Input,
    /// };
    ///
    /// let vm = PikeVM::new_many(&["[a-z0-9]{6}", "[a-z][a-z0-9]{5}"])?;
    /// let (mut cache, mut caps) = (vm.create_cache(), vm.create_captures());
    /// let haystack = "foo123";
    ///
    /// // Since we are using the default leftmost-first match and both
    /// // patterns match at the same starting position, only the first pattern
    /// // will be returned in this case when doing a search for any of the
    /// // patterns.
    /// let expected = Some(Match::must(0, 0..6));
    /// vm.search(&mut cache, &Input::new(haystack), &mut caps);
    /// assert_eq!(expected, caps.get_match());
    ///
    /// // But if we want to check whether some other pattern matches, then we
    /// // can provide its pattern ID.
    /// let expected = Some(Match::must(1, 0..6));
    /// vm.search(
    ///     &mut cache,
    ///     &Input::new(haystack).pattern(Some(PatternID::must(1))),
    ///     &mut caps,
    /// );
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
    /// use regex_automata::{nfa::thompson::pikevm::PikeVM, Match, Input};
    ///
    /// let vm = PikeVM::new(r"\b[0-9]{3}\b")?;
    /// let (mut cache, mut caps) = (vm.create_cache(), vm.create_captures());
    /// let haystack = "foo123bar";
    ///
    /// // Since we sub-slice the haystack, the search doesn't know about
    /// // the larger context and assumes that `123` is surrounded by word
    /// // boundaries. And of course, the match position is reported relative
    /// // to the sub-slice as well, which means we get `0..3` instead of
    /// // `3..6`.
    /// let expected = Some(Match::must(0, 0..3));
    /// vm.search(&mut cache, &Input::new(&haystack[3..6]), &mut caps);
    /// assert_eq!(expected, caps.get_match());
    ///
    /// // But if we provide the bounds of the search within the context of the
    /// // entire haystack, then the search can take the surrounding context
    /// // into account. (And if we did find a match, it would be reported
    /// // as a valid offset into `haystack` instead of its sub-slice.)
    /// let expected = None;
    /// vm.search(&mut cache, &Input::new(haystack).range(3..6), &mut caps);
    /// assert_eq!(expected, caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn search(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        caps: &mut Captures,
    ) {
        self.search_imp(cache, input, caps);
        let m = match caps.get_match() {
            None => return,
            Some(m) => m,
        };
        if m.is_empty() {
            input
                .skip_empty_utf8_splits(m, |search| {
                    self.search_imp(cache, search, caps);
                    Ok(caps.get_match())
                })
                .unwrap();
        }
    }

    /// Writes the set of patterns that match anywhere in the given search
    /// configuration to `patset`. If multiple patterns match at the same
    /// position and this `PikeVM` was configured with [`MatchKind::All`]
    /// semantics, then all matching patterns are written to the given set.
    ///
    /// Unless all of the patterns in this `PikeVM` are anchored, then
    /// generally speaking, this will visit every byte in the haystack.
    ///
    /// This search routine *does not* clear the pattern set. This gives some
    /// flexibility to the caller (e.g., running multiple searches with the
    /// same pattern set), but does make the API bug-prone if you're reusing
    /// the same pattern set for multiple searches but intended them to be
    /// independent.
    ///
    /// # Panics
    ///
    /// This routine may panic if the given [`PatternSet`] has insufficient
    /// capacity to hold all matching pattern IDs.
    ///
    /// # Example
    ///
    /// This example shows how to find all matching patterns in a haystack,
    /// even when some patterns match at the same position as other patterns.
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::pikevm::PikeVM,
    ///     MatchKind, PatternSet, Input,
    /// };
    ///
    /// let patterns = &[
    ///     r"\w+", r"\d+", r"\pL+", r"foo", r"bar", r"barfoo", r"foobar",
    /// ];
    /// let vm = PikeVM::builder()
    ///     .configure(PikeVM::config().match_kind(MatchKind::All))
    ///     .build_many(patterns)?;
    /// let mut cache = vm.create_cache();
    ///
    /// let input = Input::new("foobar");
    /// let mut patset = PatternSet::new(vm.pattern_len());
    /// vm.which_overlapping_matches(&mut cache, &input, &mut patset);
    /// let expected = vec![0, 2, 3, 4, 6];
    /// let got: Vec<usize> = patset.iter().map(|p| p.as_usize()).collect();
    /// assert_eq!(expected, got);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn which_overlapping_matches(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        patset: &mut PatternSet,
    ) {
        self.which_overlapping_imp(cache, input, patset)
    }
}

impl PikeVM {
    /// The implementation of standard leftmost search.
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
    ) {
        caps.set_pattern(None);
        cache.setup_search(caps.slot_len());
        if input.is_done() {
            return;
        }
        // Why do we even care about this? Well, in our 'Captures'
        // representation, we use usize::MAX as a sentinel to indicate "no
        // match." This isn't problematic so long as our haystack doesn't have
        // a maximal length. Byte slices are guaranteed by Rust to have a
        // length that fits into isize, and so this assert should always pass.
        // But we put it here to make our assumption explicit.
        assert!(
            input.haystack().len() < core::usize::MAX,
            "byte slice lengths must be less than usize MAX",
        );
        instrument!(|c| c.reset(&self.nfa));

        // Whether we want to visit all match states instead of emulating the
        // 'leftmost' semantics of typical backtracking regex engines.
        let allmatches =
            self.config.get_match_kind().continue_past_first_match();
        // Whether the search is anchored or not. This can be from
        // configuration or from how the pattern itself is constructed. As a
        // special case, searches that have specified a specific pattern to
        // search for are automatically anchored, since there is no unanchored
        // prefix for each individual pattern, only for the entire NFA as a
        // whole.
        let anchored = self.config.get_anchored()
            || self.nfa.is_always_start_anchored()
            || input.get_pattern().is_some();
        // Where to start. It's either the anchored starting state for the
        // entire NFA, or it's the anchored starting state for a specific
        // pattern.
        let start_id = match input.get_pattern() {
            // We always use the anchored starting state here, even if doing an
            // unanchored search. The "unanchored" part of it is implemented
            // in the loop below, by computing the epsilon closure from the
            // anchored starting state whenever the current state set list is
            // empty.
            None => self.nfa.start_anchored(),
            Some(pid) => self.nfa.start_pattern(pid),
        };

        let Cache { ref mut stack, ref mut curr, ref mut next } = cache;
        // Yes, our search doesn't end at input.end(), but includes it. This
        // is necessary because matches are delayed by one byte, just like
        // how the DFA engines work. The delay is used to handle look-behind
        // assertions. In the case of the PikeVM, the delay is implemented
        // by not considering a match to exist until it is visited in
        // 'steps'. Technically, we know a match exists in the previous
        // iteration via 'epsilon_closure'. (It's the same thing in NFA-to-DFA
        // determinization. We don't mark a DFA state as a match state if it
        // contains an NFA match state, but rather, whether the DFA state was
        // generated by a transition from a DFA state that contains an NFA
        // match state.)
        for at in input.start()..=input.end() {
            if curr.set.is_empty() {
                // We have a match and we haven't been instructed to continue
                // on even after finding a match, so we can quit.
                if caps.is_match() && !allmatches {
                    break;
                }
                // If we're running an anchored search and we've advanced
                // beyond the start position, then we will never observe a
                // match and thus must stop.
                if anchored && at > input.start() {
                    break;
                }
            }
            // Instead of using the NFA's unanchored start state, we actually
            // always use its anchored starting state. As a result, when doing
            // an unanchored search, we need to simulate our own '(?s:.)*?'
            // prefix, to permit a match to appear anywhere.
            //
            // Now, we don't *have* to do things this way. We could use the
            // NFA's unanchored starting state and do one 'epsilon_closure'
            // call from that starting state before the main loop here. And
            // that is just as correct. However, it turns out to be slower
            // than our approach here because it slightly increases the cost
            // of processing each byte by requiring us to visit more NFA
            // states to deal with the additional NFA states in the unanchored
            // prefix. By simulating it explicitly here, we lower those costs
            // substantially. The cost is itself small, but it adds up for
            // large haystacks.
            //
            // In order to simulate the '(?s:.)*?' prefix---which is not
            // greedy---we are careful not to perform an epsilon closure on
            // the start state if we already have a match. Namely, if we
            // did otherwise, we would never reach a terminating condition
            // because there would always be additional states to process.
            // In effect, the exclusion of running 'epsilon_closure' when
            // we have a match corresponds to the "dead" states we have in
            // our DFA regex engines. Namely, in a DFA, match states merely
            // instruct the search execution to record the current offset as
            // the most recently seen match. It is the dead state that actually
            // indicates when to stop the search (other than EOF or quit
            // states).
            //
            // However, when 'allmatches' is true, the caller has asked us to
            // leave in every possible match state. This tends not to make a
            // whole lot of sense in unanchored searches, because it means the
            // search really cannot terminate until EOF. And often, in that
            // case, you wind up skipping over a bunch of matches and are left
            // with the "last" match. Arguably, it just doesn't make a lot of
            // sense to run a 'leftmost' search (which is what this routine is)
            // with 'allmatches' set to true. But the DFAs support it and this
            // matches their behavior. (Generally, 'allmatches' is useful for
            // overlapping searches or leftmost anchored searches to find the
            // longest possible match by ignoring match priority.)
            if !caps.is_match() || allmatches {
                // Since we are adding to the 'curr' active states and since
                // this is for the start ID, we use a slots slice that is
                // guaranteed to have the right length but where event element
                // is absent. This is exactly what we want, because this
                // epsilon closure is responsible for simulating an unanchored
                // '(?s:.)*?' prefix. It is specifically outside of any
                // capturing groups, and thus, using slots that are always
                // absent is correct.
                //
                // Note though that we can't just use '&mut []' here, since
                // this epsilon closure may traverse through 'Captures' epsilon
                // transitions, and thus must be able to write offsets to the
                // slots given which are later copied to slot values in 'curr'.
                let slots = next.slot_table.all_absent();
                self.epsilon_closure(stack, slots, curr, input, at, start_id);
            }
            self.nexts(stack, curr, next, input, at, caps);
            // Unless the caller asked us to return early, we need to mush on
            // to see if we can extend our match. (But note that 'nexts' will
            // quit right after seeing a match when match_kind==LeftmostFirst,
            // as is consist with leftmost-first match priority.)
            if input.get_earliest() && caps.is_match() {
                break;
            }
            core::mem::swap(curr, next);
            next.set.clear();
        }
        instrument!(|c| c.eprint(&self.nfa));
    }

    /// The implementation for the 'which_overlapping_matches' API. Basically,
    /// we do a single scan through the entire haystack (unless our regex
    /// or search is anchored) and record every pattern that matched. In
    /// particular, when MatchKind::All is used, this supports overlapping
    /// matches. So if we have the regexes 'sam' and 'samwise', they will
    /// *both* be reported in the pattern set when searching the haystack
    /// 'samwise'.
    fn which_overlapping_imp(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        patset: &mut PatternSet,
    ) {
        // NOTE: This is effectively a copy of 'search_imp' above, but with no
        // captures support and instead writes patterns that matched directly
        // to 'patset'. See that routine for better commentary about what's
        // going on in this routine. We probably could unify the routines using
        // generics or more helper routines, but I'm not sure it's worth it.
        //
        // NOTE: We somewhat go out of our way here to support things like
        // 'input.get_earliest()' and 'leftmost-first' match semantics. Neither
        // of those seem particularly relevant to this routine, but they are
        // both supported by the DFA analogs of this routine by construction
        // and composition, so it seems like good sense to have the PikeVM
        // match that behavior.

        cache.setup_search(0);
        if input.is_done() {
            return;
        }
        assert!(
            input.haystack().len() < core::usize::MAX,
            "byte slice lengths must be less than usize MAX",
        );
        instrument!(|c| c.reset(&self.nfa));

        let allmatches =
            self.config.get_match_kind().continue_past_first_match();
        let anchored = self.config.get_anchored()
            || self.nfa.is_always_start_anchored()
            || input.get_pattern().is_some();
        let start_id = match input.get_pattern() {
            None => self.nfa.start_anchored(),
            Some(pid) => self.nfa.start_pattern(pid),
        };

        let Cache { ref mut stack, ref mut curr, ref mut next } = cache;
        for at in input.start()..=input.end() {
            let any_matches = !patset.is_empty();
            if curr.set.is_empty() {
                if any_matches && !allmatches {
                    break;
                }
                if anchored && at > input.start() {
                    break;
                }
            }
            if !any_matches || allmatches {
                let slots = &mut [];
                self.epsilon_closure(stack, slots, curr, input, at, start_id);
            }
            self.nexts_overlapping(stack, curr, next, input, at, patset);
            // If we found a match and filled our set, then there is no more
            // additional info that we can provide. Thus, we can quit. We also
            // quit if the caller asked us to stop at the earliest point that
            // we know a match exists.
            if patset.is_full() || input.get_earliest() {
                break;
            }
            core::mem::swap(curr, next);
            next.set.clear();
        }
        instrument!(|c| c.eprint(&self.nfa));
    }

    /// Process the active states in 'curr' to find the states (written to
    /// 'next') we should process for the next byte in the haystack.
    ///
    /// 'stack' is used to perform a depth first traversal of the NFA when
    /// computing an epsilon closure.
    ///
    /// When a match is found, the slots for that match state (in 'curr') are
    /// copied to 'caps'. Moreover, once a match is seen, processing for 'curr'
    /// stops (unless the PikeVM was configured with MatchKind::All semantics).
    #[inline(always)]
    fn nexts(
        &self,
        stack: &mut Vec<FollowEpsilon>,
        curr: &mut ActiveStates,
        next: &mut ActiveStates,
        input: &Input<'_, '_>,
        at: usize,
        caps: &mut Captures,
    ) {
        let slot_len = caps.slot_len();
        instrument!(|c| c.record_state_set(&curr.set));
        let ActiveStates { ref set, ref mut slot_table } = *curr;
        for sid in set.iter() {
            let pid = match self.next(stack, slot_table, next, input, at, sid)
            {
                None => continue,
                Some(pid) => pid,
            };
            let slots = slot_table.for_state(sid);
            copy_to_captures(pid, slots, caps);
            if !self.config.get_match_kind().continue_past_first_match() {
                break;
            }
        }
    }

    /// Like 'nexts', but for the overlapping case. This doesn't write any
    /// slots, and instead just writes which pattern matched in 'patset'.
    #[inline(always)]
    fn nexts_overlapping(
        &self,
        stack: &mut Vec<FollowEpsilon>,
        curr: &mut ActiveStates,
        next: &mut ActiveStates,
        input: &Input<'_, '_>,
        at: usize,
        patset: &mut PatternSet,
    ) {
        instrument!(|c| c.record_state_set(&curr.set));
        let ActiveStates { ref set, ref mut slot_table } = *curr;
        for sid in set.iter() {
            let pid = match self.next(stack, slot_table, next, input, at, sid)
            {
                None => continue,
                Some(pid) => pid,
            };
            patset.insert(pid);
            if !self.config.get_match_kind().continue_past_first_match() {
                break;
            }
        }
    }

    /// Starting from 'sid', if the position 'at' in the 'input' haystack has a
    /// transition defined out of 'sid', then add the state transitioned to and
    /// its epsilon closure to the 'next' set of states to explore.
    ///
    /// 'stack' is used by the epsilon closure computation to perform a depth
    /// first traversal of the NFA.
    ///
    /// 'curr_slot_table' should be the table of slots for the current set of
    /// states being explored. If there is a transition out of 'sid', then
    /// sid's row in the slot table is used to perform the epsilon closure.
    #[inline(always)]
    fn next(
        &self,
        stack: &mut Vec<FollowEpsilon>,
        curr_slot_table: &mut SlotTable,
        next: &mut ActiveStates,
        input: &Input<'_, '_>,
        at: usize,
        sid: StateID,
    ) -> Option<PatternID> {
        instrument!(|c| c.record_step(sid));
        match *self.nfa.state(sid) {
            State::Fail
            | State::Look { .. }
            | State::Union { .. }
            | State::BinaryUnion { .. }
            | State::Capture { .. } => None,
            State::ByteRange { ref trans } => {
                if trans.matches(input.haystack(), at) {
                    let slots = curr_slot_table.for_state(sid);
                    // OK because 'at <= haystack.len() < usize::MAX', so
                    // adding 1 will never wrap.
                    let at = at.wrapping_add(1);
                    self.epsilon_closure(
                        stack, slots, next, input, at, trans.next,
                    );
                }
                None
            }
            State::Sparse(ref sparse) => {
                if let Some(next_sid) = sparse.matches(input.haystack(), at) {
                    let slots = curr_slot_table.for_state(sid);
                    // OK because 'at <= haystack.len() < usize::MAX', so
                    // adding 1 will never wrap.
                    let at = at.wrapping_add(1);
                    self.epsilon_closure(
                        stack, slots, next, input, at, next_sid,
                    );
                }
                None
            }
            State::Match { pattern_id } => Some(pattern_id),
        }
    }

    /// Compute the epsilon closure of 'sid', writing the closure into 'next'
    /// while copying slot values from 'curr_slots' into corresponding states
    /// in 'next'. 'curr_slots' should be the slot values corresponding to
    /// 'sid'.
    ///
    /// The given 'stack' is used to perform a depth first traversal of the
    /// NFA by recursively following all epsilon transitions out of 'sid'.
    /// Conditional epsilon transitions are followed if and only if they are
    /// satisfied for the position 'at' in the 'input' haystack.
    ///
    /// While this routine may write to 'curr_slots', once it returns, any
    /// writes are undone and the original values (even if absent) are
    /// restored.
    #[inline(always)]
    fn epsilon_closure(
        &self,
        stack: &mut Vec<FollowEpsilon>,
        curr_slots: &mut [Option<NonMaxUsize>],
        next: &mut ActiveStates,
        input: &Input<'_, '_>,
        at: usize,
        sid: StateID,
    ) {
        instrument!(|c| {
            c.record_closure(sid);
            c.record_stack_push(sid);
        });
        stack.push(FollowEpsilon::Explore(sid));
        while let Some(frame) = stack.pop() {
            match frame {
                FollowEpsilon::RestoreCapture { slot, offset: pos } => {
                    curr_slots[slot] = pos;
                }
                FollowEpsilon::Explore(sid) => {
                    self.epsilon_closure_explore(
                        stack, curr_slots, next, input, at, sid,
                    );
                }
            }
        }
    }

    /// Explore all of the epsilon transitions out of 'sid'. This is mostly
    /// split out from 'epsilon_closure' in order to clearly delineate
    /// the actual work of computing an epsilon closure from the stack
    /// book-keeping.
    ///
    /// This will push any additional explorations needed on to 'stack'.
    ///
    /// 'curr_slots' should refer to the slots for the currently active NFA
    /// state. That is, the current state we are stepping through. These
    /// slots are mutated in place as new 'Captures' states are traversed
    /// during epsilon closure, but the slots are restored to their original
    /// values once the full epsilon closure is completed. The ultimate use of
    /// 'curr_slots' is to copy them to the corresponding 'next_slots', so that
    /// the capturing group spans are forwarded from the currently active state
    /// to the next.
    ///
    /// 'next' refers to the next set of active states. Computing an epsilon
    /// closure may increase the next set of active states.
    ///
    /// 'input' refers to the caller's input configuration and 'at' refers to
    /// the current position in the haystack. These are used to check whether
    /// conditional epsilon transitions (like look-around) are satisfied at
    /// the current position. If they aren't, then the epsilon closure won't
    /// include them.
    #[inline(always)]
    fn epsilon_closure_explore(
        &self,
        stack: &mut Vec<FollowEpsilon>,
        curr_slots: &mut [Option<NonMaxUsize>],
        next: &mut ActiveStates,
        input: &Input<'_, '_>,
        at: usize,
        mut sid: StateID,
    ) {
        // We can avoid pushing some state IDs on to our stack in precisely
        // the cases where a 'push(x)' would be immediately followed by a 'x
        // = pop()'. This is achieved by this outer-loop. We simply set 'sid'
        // to be the next state ID we want to explore once we're done with
        // our initial exploration. In practice, this avoids a lot of stack
        // thrashing.
        loop {
            instrument!(|c| c.record_set_insert(sid));
            // Record this state as part of our next set of active states. If
            // we've already explored it, then no need to do it again.
            if !next.set.insert(sid) {
                return;
            }
            match *self.nfa.state(sid) {
                State::Fail
                | State::Match { .. }
                | State::ByteRange { .. }
                | State::Sparse { .. } => {
                    next.slot_table.for_state(sid).copy_from_slice(curr_slots);
                    return;
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
                    instrument!(|c| {
                        for &alt in &alternates[1..] {
                            c.record_stack_push(alt);
                        }
                    });
                    stack.extend(
                        alternates[1..]
                            .iter()
                            .copied()
                            .rev()
                            .map(FollowEpsilon::Explore),
                    );
                }
                State::BinaryUnion { alt1, alt2 } => {
                    sid = alt1;
                    instrument!(|c| c.record_stack_push(sid));
                    stack.push(FollowEpsilon::Explore(alt2));
                }
                State::Capture { next, slot } => {
                    // There's no need to do anything with slots that
                    // ultimately won't be copied into the caller-provided
                    // 'Captures' value. So we just skip dealing with them at
                    // all.
                    if slot < curr_slots.len() {
                        instrument!(|c| c.record_stack_push(sid));
                        stack.push(FollowEpsilon::RestoreCapture {
                            slot,
                            offset: curr_slots[slot],
                        });
                        // OK because length of a slice must fit into an isize.
                        curr_slots[slot] = Some(NonMaxUsize::new(at).unwrap());
                    }
                    sid = next;
                }
            }
        }
    }
}

/// An iterator over all non-overlapping matches for a particular search.
///
/// The iterator yields a [`Match`] value until no more matches could be found.
/// If the underlying regex engine returns an error, then a panic occurs.
///
/// The lifetime parameters are as follows:
///
/// * `'r` represents the lifetime of the PikeVM.
/// * `'c` represents the lifetime of the PikeVM's cache.
/// * `'h` represents the lifetime of the haystack being searched.
///
/// This iterator can be created with the [`PikeVM::find_iter`] method.
#[derive(Debug)]
pub struct FindMatches<'r, 'c, 'h> {
    re: &'r PikeVM,
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
            re.search(cache, input, caps);
            Ok(caps.get_match())
        })
    }
}

/// An iterator over all non-overlapping leftmost matches, with their capturing
/// groups, for a particular search.
///
/// The iterator yields a [`Captures`] value until no more matches could be
/// found. If the underlying search returns an error, then this panics.
///
/// The lifetime parameters are as follows:
///
/// * `'r` represents the lifetime of the PikeVM.
/// * `'c` represents the lifetime of the PikeVM's cache.
/// * `'h` represents the lifetime of the haystack being searched.
///
/// This iterator can be created with the [`PikeVM::captures_iter`] method.
#[derive(Debug)]
pub struct CapturesMatches<'r, 'c, 'h> {
    re: &'r PikeVM,
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
            re.search(cache, input, caps);
            Ok(caps.get_match())
        });
        if caps.is_match() {
            Some(caps.clone())
        } else {
            None
        }
    }
}

/// A cache represents mutable state that a [`PikeVM`] requires during a
/// search.
///
/// For a given [`PikeVM`], its corresponding cache may be created either via
/// [`PikeVM::create_cache`], or via [`Cache::new`]. They are equivalent in
/// every way, except the former does not require explicitly importing `Cache`.
///
/// A particular `Cache` is coupled with the [`PikeVM`] from which it
/// was created. It may only be used with that `PikeVM`. A cache and its
/// allocations may be re-purposed via [`Cache::reset`], in which case, it can
/// only be used with the new `PikeVM` (and not the old one).
#[derive(Clone, Debug)]
pub struct Cache {
    /// Stack used while computing epsilon closure. This effectively lets us
    /// move what is more naturally expressed through recursion to a stack
    /// on the heap.
    stack: Vec<FollowEpsilon>,
    /// The current active states being explored for the current byte in the
    /// haystack.
    curr: ActiveStates,
    /// The next set of states we're building that will be explored for the
    /// next byte in the haystack.
    next: ActiveStates,
}

impl Cache {
    /// Create a new [`PikeVM`] cache.
    ///
    /// A potentially more convenient routine to create a cache is
    /// [`PikeVM::create_cache`], as it does not require also importing the
    /// `Cache` type.
    ///
    /// If you want to reuse the returned `Cache` with some other `PikeVM`,
    /// then you must call [`Cache::reset`] with the desired `PikeVM`.
    pub fn new(vm: &PikeVM) -> Cache {
        Cache {
            stack: vec![],
            curr: ActiveStates::new(vm),
            next: ActiveStates::new(vm),
        }
    }

    /// Reset this cache such that it can be used for searching with different
    /// [`PikeVM`].
    ///
    /// A cache reset permits reusing memory already allocated in this cache
    /// with a different `PikeVM`.
    ///
    /// # Example
    ///
    /// This shows how to re-purpose a cache for use with a different `PikeVM`.
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::pikevm::PikeVM, Match};
    ///
    /// let re1 = PikeVM::new(r"\w")?;
    /// let re2 = PikeVM::new(r"\W")?;
    ///
    /// let mut cache = re1.create_cache();
    /// assert_eq!(
    ///     Some(Match::must(0, 0..2)),
    ///     re1.find_iter(&mut cache, "Δ").next(),
    /// );
    ///
    /// // Using 'cache' with re2 is not allowed. It may result in panics or
    /// // incorrect results. In order to re-purpose the cache, we must reset
    /// // it with the PikeVM we'd like to use it with.
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
    pub fn reset(&mut self, vm: &PikeVM) {
        self.curr.reset(vm);
        self.next.reset(vm);
    }

    /// Returns the heap memory usage, in bytes, of this cache.
    ///
    /// This does **not** include the stack size used up by this cache. To
    /// compute that, use `std::mem::size_of::<Cache>()`.
    pub fn memory_usage(&self) -> usize {
        use core::mem::size_of;
        (self.stack.len() * size_of::<FollowEpsilon>())
            + self.curr.memory_usage()
            + self.next.memory_usage()
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
    /// in the PikeVM.
    fn setup_search(&mut self, captures_slot_len: usize) {
        self.stack.clear();
        self.curr.setup_search(captures_slot_len);
        self.next.setup_search(captures_slot_len);
    }
}

/// A set of active states used to "simulate" the execution of an NFA via the
/// PikeVM.
///
/// There are two sets of these used during NFA simulation. One set corresponds
/// to the "current" set of states being traversed for the current position
/// in a haystack. The other set corresponds to the "next" set of states being
/// built, which will become the new "current" set for the next position in the
/// haystack. These two sets correspond to CLIST and NLIST in Thompson's
/// original paper regexes: https://dl.acm.org/doi/pdf/10.1145/363347.363387
///
/// In addition to representing a set of NFA states, this also maintains slot
/// values for each state. These slot values are what turn the NFA simulation
/// into the "Pike VM." Namely, they track capturing group values for each
/// state. During the computation of epsilon closure, we copy slot values from
/// states in the "current" set to the "next" set. Eventually, once a match
/// is found, the slot values for that match state are what we write to the
/// caller provided 'Captures' value.
#[derive(Clone, Debug)]
struct ActiveStates {
    /// The set of active NFA states. This set preserves insertion order, which
    /// is critical for simulating the match semantics of backtracking regex
    /// engines.
    set: SparseSet,
    /// The slots for every NFA state, where each slot stores a (possibly
    /// absent) offset. Every capturing group has two slots. One for a start
    /// offset and one for an end offset.
    slot_table: SlotTable,
}

impl ActiveStates {
    /// Create a new set of active states for the given PikeVM. The active
    /// states returned may only be used with the given PikeVM. (Use 'reset'
    /// to re-purpose the allocation for a different PikeVM.)
    fn new(vm: &PikeVM) -> ActiveStates {
        let mut active = ActiveStates {
            set: SparseSet::new(0),
            slot_table: SlotTable::new(),
        };
        active.reset(vm);
        active
    }

    /// Reset this set of active states such that it can be used with the given
    /// PikeVM (and only that PikeVM).
    fn reset(&mut self, vm: &PikeVM) {
        let nfa = vm.get_nfa();
        self.set.resize(vm.get_nfa().states().len());
        self.slot_table.reset(vm);
    }

    /// Return the heap memory usage, in bytes, used by this set of active
    /// states.
    ///
    /// This does not include the stack size of this value.
    fn memory_usage(&self) -> usize {
        self.set.memory_usage() + self.slot_table.memory_usage()
    }

    /// Setup this set of active states for a new search. The given slot
    /// length should be the number of slots in a caller provided 'Captures'
    /// (and may be zero).
    fn setup_search(&mut self, captures_slot_len: usize) {
        self.set.clear();
        self.slot_table.setup_search(captures_slot_len);
    }
}

/// A table of slots, where each row represent a state in an NFA. Thus, the
/// table has room for storing slots for every single state in an NFA.
///
/// This table is represented with a single contiguous allocation. In general,
/// the notion of "capturing group" doesn't really exist at this level of
/// abstraction, hence the name "slot" instead. (Indeed, every capturing group
/// maps to a pair of slots, one for the start offset and one for the end
/// offset.) Slots are indexed by the 'Captures' NFA state.
///
/// N.B. Not every state actually needs a row of slots. Namely, states that
/// only have epsilon transitions currently never have anything written to
/// their rows in this table. Thus, the table is somewhat wasteful in its heap
/// usage. However, it is important to maintain fast random access by state
/// ID, which means one giant table tends to work well. RE2 takes a different
/// approach here and allocates each row as its own reference counted thing.
/// I explored such a strategy at one point here, but couldn't get it to work
/// well using entirely safe code. (To the ambitious reader: I encourage you to
/// re-litigate that experiment.) I very much wanted to stick to safe code, but
/// could be convinced otherwise if there was a solid argument and the safety
/// was encapsulated well.
#[derive(Clone, Debug)]
struct SlotTable {
    /// The actual table of offsets.
    table: Vec<Option<NonMaxUsize>>,
    /// The number of slots per state, i.e., the table's stride or the length
    /// of each row.
    slots_per_state: usize,
    /// The number of slots in the caller-provided 'Captures' value for the
    /// current search. Setting this to 'slots_per_state' is always correct,
    /// but may be wasteful.
    slots_for_captures: usize,
}

impl SlotTable {
    /// Create a new slot table.
    ///
    /// One should call 'reset' with the corresponding PikeVM before use.
    fn new() -> SlotTable {
        SlotTable { table: vec![], slots_for_captures: 0, slots_per_state: 0 }
    }

    /// Reset this slot table such that it can be used with the given PikeVM
    /// (and only that PikeVM).
    fn reset(&mut self, vm: &PikeVM) {
        let nfa = vm.get_nfa();
        self.slots_per_state = nfa.capture_slot_len();
        // This is always correct, but may be reduced for a particular search
        // if a 'Captures' has fewer slots, e.g., none at all or only slots
        // for tracking the overall match instead of all slots for every
        // group.
        self.slots_for_captures = nfa.capture_slot_len();
        let len = nfa
            .states()
            .len()
            // We add 1 so that our last row is always empty. We use it as
            // "scratch" space for computing the epsilon closure off of the
            // starting state.
            .checked_add(1)
            .and_then(|x| x.checked_mul(self.slots_per_state))
            // It seems like this could actually panic on legitimate inputs on
            // 32-bit targets, and very likely to panic on 16-bit. Should we
            // somehow convert this to an error? What about something similar
            // for the lazy DFA cache?
            .expect("slot table length doesn't overflow");
        self.table.resize(len, None);
    }

    /// Return the heap memory usage, in bytes, used by this slot table.
    ///
    /// This does not include the stack size of this value.
    fn memory_usage(&self) -> usize {
        self.table.len() * core::mem::size_of::<Option<NonMaxUsize>>()
    }

    /// Perform any per-search setup for this slot table.
    ///
    /// In particular, this sets the length of the number of slots used in the
    /// 'Captures' given by the caller (if any at all). This number may be
    /// smaller than the total number of slots available, e.g., when the caller
    /// is only interested in tracking the overall match and not the spans of
    /// every matching capturing group. Only tracking the overall match can
    /// save a substantial amount of time copying capturing spans during a
    /// search.
    fn setup_search(&mut self, captures_slot_len: usize) {
        self.slots_for_captures = captures_slot_len;
    }

    /// Return a mutable slice of the slots for the given state.
    ///
    /// Note that the length of the slice returned may be less than the total
    /// number of slots available for this state. In particular, the length
    /// always matches the number of slots indicated via 'setup_search'.
    fn for_state(&mut self, sid: StateID) -> &mut [Option<NonMaxUsize>] {
        let i = sid.as_usize() * self.slots_per_state;
        &mut self.table[i..i + self.slots_for_captures]
    }

    /// Return a slice of slots of appropriate length where every slot offset
    /// is guaranteed to be absent. This is useful in cases where you need to
    /// compute an epsilon closure outside of the user supplied regex, and thus
    /// never want it to have any capturing slots set.
    fn all_absent(&mut self) -> &mut [Option<NonMaxUsize>] {
        let i = self.table.len() - self.slots_per_state;
        &mut self.table[i..i + self.slots_for_captures]
    }
}

/// Represents a stack frame for use while computing an epsilon closure.
///
/// (An "epsilon closure" refers to the set of reachable NFA states from a
/// single state without consuming any input. That is, the set of all epsilon
/// transitions not only from that single state, but from every other state
/// reachable by an epsilon transition as well. This is why it's called a
/// "closure." Computing an epsilon closure is also done during DFA
/// determinization! Compare and contrast the epsilon closure here in this
/// PikeVM and the one used for determinization in crate::util::determinize.)
///
/// Computing the epsilon closure in a Thompson NFA proceeds via a depth
/// first traversal over all epsilon transitions from a particular state.
/// (A depth first traversal is important because it emulates the same priority
/// of matches that is typically found in backtracking regex engines.) This
/// depth first traversal is naturally expressed using recursion, but to avoid
/// a call stack size proportional to the size of a regex, we put our stack on
/// the heap instead.
///
/// This stack thus consists of call frames. The typical call frame is
/// `Explore`, which instructs epsilon closure to explore the epsilon
/// transitions from that state. (Subsequent epsilon transitions are then
/// pushed on to the stack as more `Explore` frames.) If the state ID being
/// explored has no epsilon transitions, then the capturing group slots are
/// copied from the original state that sparked the epsilon closure (from the
/// 'step' routine) to the state ID being explored. This way, capturing group
/// slots are forwarded from the previous state to the next.
///
/// The other stack frame, `RestoreCaptures`, instructs the epsilon closure to
/// set the position for a particular slot back to some particular offset. This
/// frame is pushed when `Explore` sees a `Capture` transition. `Explore` will
/// set the offset of the slot indicated in `Capture` to the current offset,
/// and then push the old offset on to the stack as a `RestoreCapture` frame.
/// Thus, the new offset is only used until the epsilon closure reverts back to
/// the `RestoreCapture` frame. In effect, this gives the `Capture` epsilon
/// transition its "scope" to only states that come "after" it during depth
/// first traversal.
#[derive(Clone, Debug)]
enum FollowEpsilon {
    /// Explore the epsilon transitions from a state ID.
    Explore(StateID),
    /// Reset the given `slot` to the given `offset` (which might be `None`).
    RestoreCapture { slot: usize, offset: Option<NonMaxUsize> },
}

/// Write the given pattern ID and slot values to the given `Captures`.
///
/// This is generally what you want to use once a match has been found. This
/// copies the internal slot data to the public `Captures` type.
fn copy_to_captures(
    pid: PatternID,
    slots: &[Option<NonMaxUsize>],
    caps: &mut Captures,
) {
    caps.set_pattern(Some(pid));
    for (i, &slot) in slots.iter().enumerate() {
        caps.set_slot(i, slot.map(|s| s.get()));
    }
}

/// A set of counters that "instruments" a PikeVM search. To enable this, you
/// must enable the 'instrument-pikevm' feature. Then run your Rust program
/// with RUST_LOG=regex_automata::nfa::thompson::pikevm=trace set in the
/// environment. The metrics collected will be dumped automatically for every
/// search executed by the PikeVM.
///
/// NOTE: When 'instrument-pikevm' is enabled, it will likely cause an absolute
/// decrease in wall-clock performance, even if the 'trace' log level isn't
/// enabled. (Although, we do try to avoid extra costs when 'trace' isn't
/// enabled.) The main point of instrumentation is to get counts of various
/// events that occur during the PikeVM's execution.
///
/// This is a somewhat hacked together collection of metrics that are useful
/// to gather from a PikeVM search. In particular, it lets us scrutinize the
/// performance profile of a search beyond what general purpose profiling tools
/// give us. Namely, we orient the profiling data around the specific states of
/// the NFA.
///
/// In other words, this lets us see which parts of the NFA graph are most
/// frequently activated. This then provides direction for optimization
/// opportunities.
///
/// The really sad part about this is that it absolutely clutters up the PikeVM
/// implementation. :'( Another approach would be to just manually add this
/// code in whenever I want this kind of profiling data, but it's complicated
/// and tedious enough that I went with this approach... for now.
///
/// When instrumentation is enabled (which also turns on 'logging'), then a
/// `Counters` is initialized for every search and `trace`'d just before the
/// search returns to the caller.
///
/// Tip: When debugging performance problems with the PikeVM, it's best to try
/// to work with an NFA that is as small as possible. Otherwise the state graph
/// is likely to be too big to digest.
#[cfg(feature = "instrument-pikevm")]
#[derive(Clone, Debug)]
struct Counters {
    /// The number of times the NFA is in a particular permutation of states.
    state_sets: alloc::collections::BTreeMap<Vec<StateID>, u64>,
    /// The number of times 'step' is called for a particular state ID (which
    /// indexes this array).
    steps: Vec<u64>,
    /// The number of times an epsilon closure was computed for a state.
    closures: Vec<u64>,
    /// The number of times a particular state ID is pushed on to a stack while
    /// computing an epsilon closure.
    stack_pushes: Vec<u64>,
    /// The number of times a particular state ID is inserted into a sparse set
    /// while computing an epsilon closure.
    set_inserts: Vec<u64>,
}

#[cfg(feature = "instrument-pikevm")]
impl Counters {
    fn empty() -> Counters {
        Counters {
            state_sets: alloc::collections::BTreeMap::new(),
            steps: vec![],
            closures: vec![],
            stack_pushes: vec![],
            set_inserts: vec![],
        }
    }

    fn reset(&mut self, nfa: &NFA) {
        let len = nfa.states().len();

        self.state_sets.clear();

        self.steps.clear();
        self.steps.resize(len, 0);

        self.closures.clear();
        self.closures.resize(len, 0);

        self.stack_pushes.clear();
        self.stack_pushes.resize(len, 0);

        self.set_inserts.clear();
        self.set_inserts.resize(len, 0);
    }

    fn eprint(&self, nfa: &NFA) {
        trace!("===== START PikeVM Instrumentation Output =====");
        // We take the top-K most occurring state sets. Otherwise the output
        // is likely to be overwhelming. And we probably only care about the
        // most frequently occuring ones anyway.
        const LIMIT: usize = 20;
        let mut set_counts =
            self.state_sets.iter().collect::<Vec<(&Vec<StateID>, &u64)>>();
        set_counts.sort_by_key(|(_, &count)| core::cmp::Reverse(count));
        trace!("## PikeVM frequency of state sets (top {})", LIMIT);
        for (set, count) in set_counts.iter().take(LIMIT) {
            trace!("{:?}: {}", set, count);
        }
        if set_counts.len() > LIMIT {
            trace!(
                "... {} sets omitted (out of {} total)",
                set_counts.len() - LIMIT,
                set_counts.len(),
            );
        }

        trace!("");
        trace!("## PikeVM total frequency of events");
        trace!(
            "steps: {}, closures: {}, stack-pushes: {}, set-inserts: {}",
            self.steps.iter().copied().sum::<u64>(),
            self.closures.iter().copied().sum::<u64>(),
            self.stack_pushes.iter().copied().sum::<u64>(),
            self.set_inserts.iter().copied().sum::<u64>(),
        );

        trace!("");
        trace!("## PikeVM frequency of events broken down by state");
        for sid in 0..self.steps.len() {
            trace!(
                "{:06}: steps: {}, closures: {}, \
                 stack-pushes: {}, set-inserts: {}",
                sid,
                self.steps[sid],
                self.closures[sid],
                self.stack_pushes[sid],
                self.set_inserts[sid],
            );
        }

        trace!("");
        trace!("## NFA debug display");
        trace!("{:?}", nfa);
        trace!("===== END PikeVM Instrumentation Output =====");
    }

    fn record_state_set(&mut self, set: &SparseSet) {
        let set = set.iter().collect::<Vec<StateID>>();
        *self.state_sets.entry(set).or_insert(0) += 1;
    }

    fn record_step(&mut self, sid: StateID) {
        self.steps[sid] += 1;
    }

    fn record_closure(&mut self, sid: StateID) {
        self.closures[sid] += 1;
    }

    fn record_stack_push(&mut self, sid: StateID) {
        self.stack_pushes[sid] += 1;
    }

    fn record_set_insert(&mut self, sid: StateID) {
        self.set_inserts[sid] += 1;
    }
}
