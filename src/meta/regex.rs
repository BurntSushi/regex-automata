use core::borrow::Borrow;

use alloc::{boxed::Box, sync::Arc, vec, vec::Vec};

use regex_syntax::{
    ast,
    hir::{self, Hir},
};

use crate::{
    meta::{
        error::BuildError,
        strategy::{self, Strategy},
        wrappers,
    },
    nfa::thompson::pikevm,
    util::{
        captures::{Captures, GroupInfo},
        iter,
        pool::{Pool, PoolGuard},
        prefilter::{self, Prefilter},
        primitives::{NonMaxUsize, PatternID},
        search::{HalfMatch, Input, Match, MatchError, MatchKind, PatternSet},
    },
};

/// # Example
///
/// ```
/// use regex_automata::{meta::Regex, Anchored, Input, PatternID};
///
/// let re = Regex::new(r"[a-z]+")?;
/// assert!(re.is_match("123 abc"));
///
/// let input = Input::new("123 abc").anchored(Anchored::Yes);
/// assert!(!re.is_match(input));
///
/// let input = Input::new("123 abc").anchored(Anchored::Yes).range(4..);
/// assert!(re.is_match(input));
///
/// let input = Input::new("123 abc")
///     .anchored(Anchored::Pattern(PatternID::ZERO))
///     .range(4..);
/// assert!(re.is_match(input));
///
/// let input = Input::new("123 abc")
///     .anchored(Anchored::Pattern(PatternID::must(1)))
///     .range(4..);
/// assert!(!re.is_match(input));
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug)]
pub struct Regex(Arc<RegexI>);

/// The internal implementation of `Regex`, split out so that it can be wrapped
/// in an `Arc`.
#[derive(Debug)]
struct RegexI {
    /// The core matching engine.
    ///
    /// Why is this reference counted when RegexI is already wrapped in an Arc?
    /// Well, we need to capture this in a closure to our `Pool` below in order
    /// to create new `Cache` values when needed. So since it needs to be in
    /// two places, we make it reference counted.
    ///
    /// We make `RegexI` itself reference counted too so that `Regex` itself
    /// stays extremely small and very cheap to clone.
    strat: Arc<dyn Strategy>,
    /// Metadata about the regexes driving the strategy. The metadata is also
    /// usually stored inside the strategy too, but we put it here as well
    /// so that we can get quick access to it (without virtual calls) before
    /// executing the regex engine. For example, we use this metadata to
    /// detect a subset of cases where we know a match is impossible, and can
    /// thus avoid calling into the strategy at all.
    ///
    /// Since `RegexInfo` is stored in multiple places, it is also reference
    /// counted.
    info: RegexInfo,
    pool: Pool<Cache, Box<dyn Fn() -> Cache>>,
}

impl Regex {
    pub fn new(pattern: &str) -> Result<Regex, BuildError> {
        Regex::builder().build(pattern)
    }

    pub fn new_many<P: AsRef<str>>(
        patterns: &[P],
    ) -> Result<Regex, BuildError> {
        Regex::builder().build_many(patterns)
    }

    pub fn always_match() -> Result<Regex, BuildError> {
        Regex::new("")
    }

    pub fn never_match() -> Result<Regex, BuildError> {
        Regex::new(r"[a&&b]")
    }

    pub fn config() -> Config {
        Config::new()
    }

    pub fn builder() -> Builder {
        Builder::new()
    }

    pub fn create_captures(&self) -> Captures {
        Captures::all(self.group_info().clone())
    }

    pub fn create_cache(&self) -> Cache {
        self.0.strat.create_cache()
    }

    pub fn reset_cache(&self, cache: &mut Cache) {
        self.0.strat.reset_cache(cache)
    }

    pub fn pattern_len(&self) -> usize {
        self.0.info.pattern_len()
    }

    /// Returns the total number of capturing groups.
    ///
    /// This includes the implicit capturing group corresponding to the
    /// entire match. Therefore, the minimum value returned is `1`.
    ///
    /// # Example
    ///
    /// This shows a few patterns and how many capture groups they have.
    ///
    /// ```
    /// use regex_automata::meta::Regex;
    ///
    /// let len = |pattern| {
    ///     Regex::new(pattern).map(|re| re.captures_len())
    /// };
    ///
    /// assert_eq!(1, len("a")?);
    /// assert_eq!(2, len("(a)")?);
    /// assert_eq!(3, len("(a)|(b)")?);
    /// assert_eq!(5, len("(a)(b)|(c)(d)")?);
    /// assert_eq!(2, len("(a)|b")?);
    /// assert_eq!(2, len("a|(b)")?);
    /// assert_eq!(2, len("(b)*")?);
    /// assert_eq!(2, len("(b)+")?);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: multiple patterns
    ///
    /// This routine also works for multiple patterns. The total number is
    /// the sum of the capture groups of each pattern.
    ///
    /// ```
    /// use regex_automata::meta::Regex;
    ///
    /// let len = |patterns| {
    ///     Regex::new_many(patterns).map(|re| re.captures_len())
    /// };
    ///
    /// assert_eq!(2, len(&["a", "b"])?);
    /// assert_eq!(4, len(&["(a)", "(b)"])?);
    /// assert_eq!(6, len(&["(a)|(b)", "(c)|(d)"])?);
    /// assert_eq!(8, len(&["(a)(b)|(c)(d)", "(x)(y)"])?);
    /// assert_eq!(3, len(&["(a)", "b"])?);
    /// assert_eq!(3, len(&["a", "(b)"])?);
    /// assert_eq!(4, len(&["(a)", "(b)*"])?);
    /// assert_eq!(4, len(&["(a)+", "(b)+"])?);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn captures_len(&self) -> usize {
        self.0
            .info
            .props_union()
            .explicit_captures_len()
            .saturating_add(self.pattern_len())
    }

    /// Returns the total number of capturing groups that appear in every
    /// possible match.
    ///
    /// If the number of capture groups can vary depending on the match, then
    /// this returns `None`. That is, a value is only returned when the number
    /// of matching groups is invariant or "static."
    ///
    /// Note that like [`Regex::captures_len`], this **does** include the
    /// implicit capturing group corresponding to the entire match. Therefore,
    /// when a non-None value is returned, it is guaranteed to be at least `1`.
    /// Stated differently, a return value of `Some(0)` is impossible.
    ///
    /// # Example
    ///
    /// This shows a few cases where a static number of capture groups is
    /// available and a few cases where it is not.
    ///
    /// ```
    /// use regex_automata::meta::Regex;
    ///
    /// let len = |pattern| {
    ///     Regex::new(pattern).map(|re| re.static_captures_len())
    /// };
    ///
    /// assert_eq!(Some(1), len("a")?);
    /// assert_eq!(Some(2), len("(a)")?);
    /// assert_eq!(Some(2), len("(a)|(b)")?);
    /// assert_eq!(Some(3), len("(a)(b)|(c)(d)")?);
    /// assert_eq!(None, len("(a)|b")?);
    /// assert_eq!(None, len("a|(b)")?);
    /// assert_eq!(None, len("(b)*")?);
    /// assert_eq!(Some(2), len("(b)+")?);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: multiple patterns
    ///
    /// This property extends to regexes with multiple patterns as well. In
    /// order for their to be a static number of capture groups in this case,
    /// every pattern must have the same static number.
    ///
    /// ```
    /// use regex_automata::meta::Regex;
    ///
    /// let len = |patterns| {
    ///     Regex::new_many(patterns).map(|re| re.static_captures_len())
    /// };
    ///
    /// assert_eq!(Some(1), len(&["a", "b"])?);
    /// assert_eq!(Some(2), len(&["(a)", "(b)"])?);
    /// assert_eq!(Some(2), len(&["(a)|(b)", "(c)|(d)"])?);
    /// assert_eq!(Some(3), len(&["(a)(b)|(c)(d)", "(x)(y)"])?);
    /// assert_eq!(None, len(&["(a)", "b"])?);
    /// assert_eq!(None, len(&["a", "(b)"])?);
    /// assert_eq!(None, len(&["(a)", "(b)*"])?);
    /// assert_eq!(Some(2), len(&["(a)+", "(b)+"])?);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn static_captures_len(&self) -> Option<usize> {
        self.0
            .info
            .props_union()
            .static_explicit_captures_len()
            .map(|len| len.saturating_add(1))
    }

    #[inline]
    pub fn group_info(&self) -> &GroupInfo {
        self.0.strat.group_info()
    }

    pub fn get_config(&self) -> &Config {
        self.0.info.config()
    }

    pub fn memory_usage(&self) -> usize {
        0
    }
}

impl Regex {
    /// ```
    /// use regex_automata::meta::Regex;
    ///
    /// let re = Regex::new(r"\w+$").unwrap();
    /// assert!(re.is_match("foo"));
    /// ```
    #[inline]
    pub fn is_match<'h, I: Into<Input<'h>>>(&self, input: I) -> bool {
        let input = input.into().earliest(true);
        self.search_half(&input).is_some()
    }

    #[inline]
    pub fn find<'h, I: Into<Input<'h>>>(&self, input: I) -> Option<Match> {
        self.search(&input.into())
    }

    #[inline]
    pub fn captures<'h, I: Into<Input<'h>>>(
        &self,
        input: I,
        caps: &mut Captures,
    ) {
        self.search_captures(&input.into(), caps)
    }

    #[inline]
    pub fn find_iter<'r, 'h, I: Into<Input<'h>>>(
        &'r self,
        input: I,
    ) -> FindMatches<'r, 'h> {
        let cache = self.0.pool.get();
        let it = iter::Searcher::new(input.into());
        FindMatches { re: self, cache, it }
    }

    #[inline]
    pub fn captures_iter<'r, 'h, I: Into<Input<'h>>>(
        &'r self,
        input: I,
    ) -> CapturesMatches<'r, 'h> {
        let cache = self.0.pool.get();
        let caps = self.create_captures();
        let it = iter::Searcher::new(input.into());
        CapturesMatches { re: self, cache, caps, it }
    }
}

impl Regex {
    #[inline]
    pub fn search(&self, input: &Input<'_>) -> Option<Match> {
        if self.0.info.is_impossible(input) {
            return None;
        }
        self.search_with(&mut self.0.pool.get(), input)
    }

    #[inline]
    pub fn search_half(&self, input: &Input<'_>) -> Option<HalfMatch> {
        if self.0.info.is_impossible(input) {
            return None;
        }
        self.search_half_with(&mut self.0.pool.get(), input)
    }

    #[inline]
    pub fn search_captures(&self, input: &Input<'_>, caps: &mut Captures) {
        self.search_captures_with(&mut self.0.pool.get(), input, caps);
    }

    #[inline]
    pub fn search_slots(
        &self,
        input: &Input<'_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Option<PatternID> {
        if self.0.info.is_impossible(input) {
            return None;
        }
        self.search_slots_with(&mut self.0.pool.get(), input, slots)
    }

    #[inline]
    pub fn which_overlapping_matches(
        &self,
        input: &Input<'_>,
        patset: &mut PatternSet,
    ) {
        if self.0.info.is_impossible(input) {
            return;
        }
        self.which_overlapping_matches_with(
            &mut self.0.pool.get(),
            input,
            patset,
        )
    }
}

impl Regex {
    #[inline]
    pub fn search_with(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
    ) -> Option<Match> {
        if self.0.info.is_impossible(input) {
            return None;
        }
        self.0.strat.search(cache, input)
    }

    #[inline]
    pub fn search_half_with(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
    ) -> Option<HalfMatch> {
        if self.0.info.is_impossible(input) {
            return None;
        }
        self.0.strat.search_half(cache, input)
    }

    #[inline]
    pub fn search_captures_with(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
        caps: &mut Captures,
    ) {
        caps.set_pattern(None);
        let pid = self.search_slots_with(cache, input, caps.slots_mut());
        caps.set_pattern(pid);
    }

    #[inline]
    pub fn search_slots_with(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Option<PatternID> {
        if self.0.info.is_impossible(input) {
            return None;
        }
        self.0.strat.search_slots(cache, input, slots)
    }

    #[inline]
    pub fn which_overlapping_matches_with(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
        patset: &mut PatternSet,
    ) {
        if self.0.info.is_impossible(input) {
            return;
        }
        self.0.strat.which_overlapping_matches(cache, input, patset)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RegexInfo(Arc<RegexInfoI>);

#[derive(Clone, Debug)]
struct RegexInfoI {
    config: Config,
    props: Vec<hir::Properties>,
    props_union: hir::Properties,
}

impl RegexInfo {
    fn new(config: Config, hirs: &[&Hir]) -> RegexInfo {
        // Collect all of the properties from each of the HIRs, and also
        // union them into one big set of properties representing all HIRs
        // as if they were in one big alternation.
        let mut props = vec![];
        for hir in hirs.iter() {
            props.push(hir.properties().clone());
        }
        let props_union = hir::Properties::union(&props);

        RegexInfo(Arc::new(RegexInfoI { config, props, props_union }))
    }

    pub(crate) fn config(&self) -> &Config {
        &self.0.config
    }

    pub(crate) fn props(&self) -> &[hir::Properties] {
        &self.0.props
    }

    pub(crate) fn props_union(&self) -> &hir::Properties {
        &self.0.props_union
    }

    pub(crate) fn pattern_len(&self) -> usize {
        self.props().len()
    }

    /// Returns true when the search is guaranteed to be anchored. That is,
    /// when a match is reported, its offset is guaranteed to correspond to
    /// the start of the search.
    ///
    /// This includes returning true when `input` _isn't_ anchored but the
    /// underlying regex is.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn is_anchored_start(&self, input: &Input<'_>) -> bool {
        input.get_anchored().is_anchored() || self.is_always_anchored_start()
    }

    /// Returns true when this regex is always anchored to the start of a
    /// search. And in particular, that regardless of an `Input` configuration,
    /// if any match is reported it must start at `0`.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn is_always_anchored_start(&self) -> bool {
        use regex_syntax::hir::Look;
        self.props_union().look_set_prefix().contains(Look::Start)
    }

    /// Returns true when this regex is always anchored to the end of a
    /// search. And in particular, that regardless of an `Input` configuration,
    /// if any match is reported it must end at the end of the haystack.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn is_always_anchored_end(&self) -> bool {
        use regex_syntax::hir::Look;
        self.props_union().look_set_suffix().contains(Look::End)
    }

    /// Returns true if and only if it is known that a match is impossible
    /// for the given input. This is useful for short-circuiting and avoiding
    /// running the regex engine if it's known no match can be reported.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn is_impossible(&self, input: &Input<'_>) -> bool {
        // Input has been exhausted, nothing left to do.
        if input.is_done() {
            return true;
        }
        // The underlying regex is anchored, so if we don't start the search
        // at position 0, a match is impossible, because the anchor can only
        // match at position 0.
        if input.start() > 0 && self.is_always_anchored_start() {
            return true;
        }
        // Same idea, but for the end anchor.
        if input.end() < input.haystack().len()
            && self.is_always_anchored_end()
        {
            return true;
        }
        // If the haystack is smaller than the minimum length required, then
        // we know there can be no match.
        let minlen = match self.props_union().minimum_len() {
            None => return false,
            Some(minlen) => minlen,
        };
        if input.get_span().len() < minlen {
            return true;
        }
        // Same idea as minimum, but for maximum. This is trickier. We can
        // only apply the maximum when we know the entire span that we're
        // searching *has* to match according to the regex (and possibly the
        // input configuration). If we know there is too much for the regex
        // to match, we can bail early.
        //
        // I don't think we can apply the maximum otherwise unfortunately.
        if self.is_anchored_start(input) && self.is_always_anchored_end() {
            let maxlen = match self.props_union().maximum_len() {
                None => return false,
                Some(maxlen) => maxlen,
            };
            if input.get_span().len() > maxlen {
                return true;
            }
        }
        false
    }
}

#[derive(Debug)]
pub struct FindMatches<'r, 'h> {
    re: &'r Regex,
    cache: PoolGuard<'r, Cache, Box<dyn Fn() -> Cache>>,
    it: iter::Searcher<'h>,
}

impl<'r, 'h> Iterator for FindMatches<'r, 'h> {
    type Item = Match;

    #[inline]
    fn next(&mut self) -> Option<Match> {
        let FindMatches { re, ref mut cache, ref mut it } = *self;
        it.advance(|input| Ok(re.search_with(cache, input)))
    }

    #[inline]
    fn count(self) -> usize {
        // If all we care about is a count of matches, then we only need to
        // find the end position of each match. This can give us a 2x perf
        // boost in some cases, because it avoids needing to do a reverse scan
        // to find the start of a match.
        let FindMatches { re, mut cache, it } = self;
        it.into_half_matches_iter(|input| {
            Ok(re.search_half_with(&mut cache, input))
        })
        .count()
    }
}

#[derive(Debug)]
pub struct CapturesMatches<'r, 'h> {
    re: &'r Regex,
    cache: PoolGuard<'r, Cache, Box<dyn Fn() -> Cache>>,
    caps: Captures,
    it: iter::Searcher<'h>,
}

impl<'r, 'h> Iterator for CapturesMatches<'r, 'h> {
    type Item = Captures;

    #[inline]
    fn next(&mut self) -> Option<Captures> {
        // Splitting 'self' apart seems necessary to appease borrowck.
        let CapturesMatches { re, ref mut cache, ref mut caps, ref mut it } =
            *self;
        let _ = it.advance(|input| {
            re.search_captures_with(cache, input, caps);
            Ok(caps.get_match())
        });
        if caps.is_match() {
            Some(caps.clone())
        } else {
            None
        }
    }

    #[inline]
    fn count(self) -> usize {
        let CapturesMatches { re, mut cache, it, .. } = self;
        it.into_half_matches_iter(|input| {
            Ok(re.search_half_with(&mut cache, input))
        })
        .count()
    }
}

#[derive(Debug, Clone)]
pub struct Cache {
    pub(crate) capmatches: Captures,
    pub(crate) pikevm: wrappers::PikeVMCache,
    pub(crate) backtrack: wrappers::BoundedBacktrackerCache,
    pub(crate) onepass: wrappers::OnePassCache,
    pub(crate) hybrid: wrappers::HybridCache,
    pub(crate) revhybrid: wrappers::ReverseHybridCache,
}

impl Cache {
    pub fn new(re: &Regex) -> Cache {
        re.create_cache()
    }

    pub fn reset(&mut self, re: &Regex) {
        re.reset_cache(self)
    }

    pub fn memory_usage(&self) -> usize {
        let mut bytes = 0;
        bytes += self.pikevm.memory_usage();
        bytes += self.backtrack.memory_usage();
        bytes += self.onepass.memory_usage();
        bytes += self.hybrid.memory_usage();
        bytes += self.revhybrid.memory_usage();
        bytes
    }
}

#[derive(Clone, Debug, Default)]
pub struct Config {
    // As with other configuration types in this crate, we put all our knobs
    // in options so that we can distinguish between "default" and "not set."
    // This makes it possible to easily combine multiple configurations
    // without default values overwriting explicitly specified values. See the
    // 'overwrite' method.
    //
    // For docs on the fields below, see the corresponding method setters.
    match_kind: Option<MatchKind>,
    utf8: Option<bool>,
    autopre: Option<bool>,
    pre: Option<Option<Prefilter>>,
    nfa_size_limit: Option<Option<usize>>,
    onepass_size_limit: Option<Option<usize>>,
    hybrid_cache_capacity: Option<usize>,
    hybrid: Option<bool>,
    dfa: Option<bool>,
    dfa_size_limit: Option<Option<usize>>,
    dfa_state_limit: Option<Option<usize>>,
    onepass: Option<bool>,
    backtrack: Option<bool>,
    byte_classes: Option<bool>,
    line_terminator: Option<u8>,
}

impl Config {
    pub fn new() -> Config {
        Config::default()
    }

    pub fn match_kind(self, kind: MatchKind) -> Config {
        Config { match_kind: Some(kind), ..self }
    }

    pub fn utf8(self, yes: bool) -> Config {
        Config { utf8: Some(yes), ..self }
    }

    pub fn auto_prefilter(self, yes: bool) -> Config {
        Config { autopre: Some(yes), ..self }
    }

    pub fn prefilter(self, pre: Option<Prefilter>) -> Config {
        Config { pre: Some(pre), ..self }
    }

    pub fn nfa_size_limit(self, limit: Option<usize>) -> Config {
        Config { nfa_size_limit: Some(limit), ..self }
    }

    pub fn onepass_size_limit(self, limit: Option<usize>) -> Config {
        Config { onepass_size_limit: Some(limit), ..self }
    }

    pub fn hybrid_cache_capacity(self, limit: usize) -> Config {
        Config { hybrid_cache_capacity: Some(limit), ..self }
    }

    pub fn dfa_size_limit(self, limit: Option<usize>) -> Config {
        Config { dfa_size_limit: Some(limit), ..self }
    }

    pub fn dfa_state_limit(self, limit: Option<usize>) -> Config {
        Config { dfa_state_limit: Some(limit), ..self }
    }

    pub fn hybrid(self, yes: bool) -> Config {
        Config { hybrid: Some(yes), ..self }
    }

    pub fn dfa(self, yes: bool) -> Config {
        Config { dfa: Some(yes), ..self }
    }

    pub fn onepass(self, yes: bool) -> Config {
        Config { onepass: Some(yes), ..self }
    }

    pub fn backtrack(self, yes: bool) -> Config {
        Config { backtrack: Some(yes), ..self }
    }

    pub fn byte_classes(self, yes: bool) -> Config {
        Config { byte_classes: Some(yes), ..self }
    }

    pub fn line_terminator(self, byte: u8) -> Config {
        Config { line_terminator: Some(byte), ..self }
    }

    pub fn get_match_kind(&self) -> MatchKind {
        self.match_kind.unwrap_or(MatchKind::LeftmostFirst)
    }

    pub fn get_utf8(&self) -> bool {
        self.utf8.unwrap_or(true)
    }

    pub fn get_auto_prefilter(&self) -> bool {
        self.autopre.unwrap_or(true)
    }

    pub fn get_prefilter(&self) -> Option<&Prefilter> {
        self.pre.as_ref().unwrap_or(&None).as_ref()
    }

    pub fn get_nfa_size_limit(&self) -> Option<usize> {
        self.nfa_size_limit.unwrap_or(Some(10 * (1 << 20)))
    }

    pub fn get_onepass_size_limit(&self) -> Option<usize> {
        self.onepass_size_limit.unwrap_or(Some(1 * (1 << 20)))
    }

    pub fn get_hybrid_cache_capacity(&self) -> usize {
        self.hybrid_cache_capacity.unwrap_or(2 * (1 << 20))
    }

    pub fn get_dfa_size_limit(&self) -> Option<usize> {
        // The default for this is VERY small because building a full DFA is
        // ridiculously costly. But for regexes that are very small, it can be
        // beneficial to use a full DFA. In particular, a full DFA can enable
        // additional optimizations via something called "accelerated" states.
        // Namely, when there's a state with only a few outgoing transitions,
        // we can temporary suspend walking the transition table and use memchr
        // for just those outgoing transitions to skip ahead very quickly.
        //
        // Generally speaking, if Unicode is enabled in your regex and you're
        // using some kind of Unicode feature, then it's going to blow this
        // size limit. Moreover, Unicode tends to defeat the "accelerated"
        // state optimization too, so it's a double whammy.
        //
        // We also use a limit on the number of NFA states to avoid even
        // starting the DFA construction process. Namely, DFA construction
        // itself could make lots of initial allocs proportional to the size
        // of the NFA, and if the NFA is large, it doesn't make sense to pay
        // that cost if we know it's likely to be blown by a large margin.
        self.dfa_size_limit.unwrap_or(Some(40 * (1 << 10)))
    }

    pub fn get_dfa_state_limit(&self) -> Option<usize> {
        // Again, as with the size limit, we keep this very small.
        self.dfa_state_limit.unwrap_or(Some(30))
    }

    pub fn get_hybrid(&self) -> bool {
        #[cfg(feature = "hybrid")]
        {
            self.hybrid.unwrap_or(true)
        }
        #[cfg(not(feature = "hybrid"))]
        {
            false
        }
    }

    pub fn get_dfa(&self) -> bool {
        #[cfg(feature = "dfa-build")]
        {
            self.dfa.unwrap_or(true)
        }
        #[cfg(not(feature = "dfa-build"))]
        {
            false
        }
    }

    pub fn get_onepass(&self) -> bool {
        #[cfg(feature = "dfa-onepass")]
        {
            self.onepass.unwrap_or(true)
        }
        #[cfg(not(feature = "dfa-onepass"))]
        {
            false
        }
    }

    pub fn get_backtrack(&self) -> bool {
        #[cfg(feature = "nfa-backtrack")]
        {
            self.backtrack.unwrap_or(true)
        }
        #[cfg(not(feature = "nfa-backtrack"))]
        {
            false
        }
    }

    pub fn get_byte_classes(&self) -> bool {
        self.byte_classes.unwrap_or(true)
    }

    pub fn get_line_terminator(&self) -> u8 {
        self.line_terminator.unwrap_or(b'\n')
    }

    /// Overwrite the default configuration such that the options in `o` are
    /// always used. If an option in `o` is not set, then the corresponding
    /// option in `self` is used. If it's not set in `self` either, then it
    /// remains not set.
    pub(crate) fn overwrite(&self, o: Config) -> Config {
        Config {
            match_kind: o.match_kind.or(self.match_kind),
            utf8: o.utf8.or(self.utf8),
            autopre: o.autopre.or(self.autopre),
            pre: o.pre.or_else(|| self.pre.clone()),
            nfa_size_limit: o.nfa_size_limit.or(self.nfa_size_limit),
            onepass_size_limit: o
                .onepass_size_limit
                .or(self.onepass_size_limit),
            hybrid_cache_capacity: o
                .hybrid_cache_capacity
                .or(self.hybrid_cache_capacity),
            hybrid: o.hybrid.or(self.hybrid),
            dfa: o.dfa.or(self.dfa),
            dfa_size_limit: o.dfa_size_limit.or(self.dfa_size_limit),
            dfa_state_limit: o.dfa_state_limit.or(self.dfa_state_limit),
            onepass: o.onepass.or(self.onepass),
            backtrack: o.backtrack.or(self.backtrack),
            byte_classes: o.byte_classes.or(self.byte_classes),
            line_terminator: o.line_terminator.or(self.line_terminator),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Builder {
    config: Config,
    ast: ast::parse::ParserBuilder,
    hir: hir::translate::TranslatorBuilder,
}

impl Builder {
    pub fn new() -> Builder {
        Builder {
            config: Config::default(),
            ast: ast::parse::ParserBuilder::new(),
            hir: hir::translate::TranslatorBuilder::new(),
        }
    }

    pub fn build(&self, pattern: &str) -> Result<Regex, BuildError> {
        self.build_many(&[pattern])
    }

    pub fn build_many<P: AsRef<str>>(
        &self,
        patterns: &[P],
    ) -> Result<Regex, BuildError> {
        use crate::util::primitives::IteratorIndexExt;
        log! {
            debug!("building meta regex with {} patterns:", patterns.len());
            for (pid, p) in patterns.iter().with_pattern_ids() {
                let p = p.as_ref();
                // We might split a grapheme with this truncation logic, but
                // that's fine. We at least avoid splitting a codepoint.
                let maxoff = p
                    .char_indices()
                    .map(|(i, ch)| i + ch.len_utf8())
                    .take(1000)
                    .last()
                    .unwrap_or(0);
                if maxoff < p.len() {
                    debug!("{:?}: {}[... snip ...]", pid, &p[..maxoff]);
                } else {
                    debug!("{:?}: {}", pid, p);
                }
            }
        }
        let (mut asts, mut hirs) = (vec![], vec![]);
        for (pid, p) in patterns.iter().with_pattern_ids() {
            let ast = self
                .ast
                .build()
                .parse(p.as_ref())
                .map_err(|err| BuildError::ast(pid, err))?;
            asts.push(ast);
        }
        for ((pid, p), ast) in
            patterns.iter().with_pattern_ids().zip(asts.iter())
        {
            let hir = self
                .hir
                .build()
                .translate(p.as_ref(), ast)
                .map_err(|err| BuildError::hir(pid, err))?;
            hirs.push(hir);
        }
        self.build_many_from_hir(&hirs)
    }

    pub fn build_from_hir(&self, hir: &Hir) -> Result<Regex, BuildError> {
        self.build_many_from_hir(&[hir])
    }

    pub fn build_many_from_hir<H: Borrow<Hir>>(
        &self,
        hirs: &[H],
    ) -> Result<Regex, BuildError> {
        let config = self.config.clone();
        // We collect the HIRs into a vec so we can write internal routines
        // with '&[&Hir]'. i.e., Don't use generics everywhere to keep code
        // bloat down..
        let hirs: Vec<&Hir> = hirs.iter().map(|hir| hir.borrow()).collect();
        let info = RegexInfo::new(config, &hirs);
        let strat = strategy::new(&info, &hirs)?;
        let pool = {
            let strat = Arc::clone(&strat);
            let create: Box<dyn Fn() -> Cache> =
                Box::new(move || strat.create_cache());
            Pool::new(create)
        };
        Ok(Regex(Arc::new(RegexI { strat, info, pool })))
    }

    pub fn configure(&mut self, config: Config) -> &mut Builder {
        self.config = self.config.overwrite(config);
        self
    }

    pub fn syntax(
        &mut self,
        config: crate::util::syntax::Config,
    ) -> &mut Builder {
        config.apply_ast(&mut self.ast);
        config.apply_hir(&mut self.hir);
        self
    }
}
