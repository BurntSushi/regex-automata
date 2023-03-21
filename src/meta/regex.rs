use core::{
    borrow::Borrow,
    panic::{RefUnwindSafe, UnwindSafe},
};

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

/// A type alias for our pool of meta::Cache that fixes the type parameters to
/// what we use for the meta regex below.
type CachePool = Pool<Cache, CachePoolFn>;

/// Same as above, but for the guard returned by a pool.
type CachePoolGuard<'a> = PoolGuard<'a, Cache, CachePoolFn>;

/// The type of the closure we use to create new caches. We need to spell out
/// all of the marker traits or else we risk leaking !MARKER impls.
type CachePoolFn =
    Box<dyn Fn() -> Cache + Send + Sync + UnwindSafe + RefUnwindSafe>;

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
#[derive(Debug)]
pub struct Regex {
    /// The actual regex implementation.
    imp: Arc<RegexI>,
    /// A thread safe pool of caches.
    ///
    /// For the higher level search APIs, a `Cache` is automatically plucked
    /// from this pool before running a search. The lower level `with` methods
    /// permit the caller to provide their own cache, thereby bypassing
    /// accesses to this pool.
    ///
    /// Note that we put this outside the `Arc` so that cloning a `Regex`
    /// results in creating a fresh `CachePool`. This in turn permits callers
    /// to clone regexes into separate threads where each such regex gets
    /// the pool's "thread owner" optimization. Otherwise, if one shares the
    /// `Regex` directly, then the pool will go through a slower mutex path for
    /// all threads except for the "owner."
    pool: CachePool,
}

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
        self.imp.strat.create_cache()
    }

    pub fn reset_cache(&self, cache: &mut Cache) {
        self.imp.strat.reset_cache(cache)
    }

    pub fn pattern_len(&self) -> usize {
        self.imp.info.pattern_len()
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
        self.imp
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
        self.imp
            .info
            .props_union()
            .static_explicit_captures_len()
            .map(|len| len.saturating_add(1))
    }

    #[inline]
    pub fn group_info(&self) -> &GroupInfo {
        self.imp.strat.group_info()
    }

    pub fn get_config(&self) -> &Config {
        self.imp.info.config()
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
        let cache = self.pool.get();
        let it = iter::Searcher::new(input.into());
        FindMatches { re: self, cache, it }
    }

    #[inline]
    pub fn captures_iter<'r, 'h, I: Into<Input<'h>>>(
        &'r self,
        input: I,
    ) -> CapturesMatches<'r, 'h> {
        let cache = self.pool.get();
        let caps = self.create_captures();
        let it = iter::Searcher::new(input.into());
        CapturesMatches { re: self, cache, caps, it }
    }
}

impl Regex {
    #[inline]
    pub fn search(&self, input: &Input<'_>) -> Option<Match> {
        if self.imp.info.is_impossible(input) {
            return None;
        }
        let mut guard = self.pool.get();
        let result = self.imp.strat.search(&mut guard, input);
        // We do this dance with the guard and explicitly put it back in the
        // pool because it seems to result in better codegen. If we let the
        // guard's Drop impl put it back in the pool, then functions like
        // ptr::drop_in_place get called and they *don't* get inlined. This
        // isn't usually a big deal, but in latency sensitive benchmarks the
        // extra function call can matter.
        //
        // I used `rebar measure -f '^grep/every-line$' -e meta` to measure
        // the effects here.
        //
        // Note that this doesn't eliminate the latency effects of using the
        // pool. There is still some (minor) cost for the "thread owner" of the
        // pool. (i.e., The thread that first calls a regex search routine.)
        // However, for other threads using the regex, the pool access can be
        // quite expensive as it goes through a mutex. Callers can avoid this
        // by either cloning the Regex (which creates a distinct copy of the
        // pool), or callers can use the lower level APIs that accept a 'Cache'
        // directly and do their own handling.
        PoolGuard::put(guard);
        result
    }

    #[inline]
    pub fn search_half(&self, input: &Input<'_>) -> Option<HalfMatch> {
        if self.imp.info.is_impossible(input) {
            return None;
        }
        let mut guard = self.pool.get();
        let result = self.imp.strat.search_half(&mut guard, input);
        // See 'Regex::search' for why we put the guard back explicitly.
        PoolGuard::put(guard);
        result
    }

    #[inline]
    pub fn search_captures(&self, input: &Input<'_>, caps: &mut Captures) {
        caps.set_pattern(None);
        let pid = self.search_slots(input, caps.slots_mut());
        caps.set_pattern(pid);
    }

    #[inline]
    pub fn search_slots(
        &self,
        input: &Input<'_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Option<PatternID> {
        if self.imp.info.is_impossible(input) {
            return None;
        }
        let mut guard = self.pool.get();
        let result = self.imp.strat.search_slots(&mut guard, input, slots);
        // See 'Regex::search' for why we put the guard back explicitly.
        PoolGuard::put(guard);
        result
    }

    #[inline]
    pub fn which_overlapping_matches(
        &self,
        input: &Input<'_>,
        patset: &mut PatternSet,
    ) {
        if self.imp.info.is_impossible(input) {
            return;
        }
        let mut guard = self.pool.get();
        let result = self
            .imp
            .strat
            .which_overlapping_matches(&mut guard, input, patset);
        // See 'Regex::search' for why we put the guard back explicitly.
        PoolGuard::put(guard);
        result
    }
}

impl Regex {
    #[inline]
    pub fn search_with(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
    ) -> Option<Match> {
        if self.imp.info.is_impossible(input) {
            return None;
        }
        self.imp.strat.search(cache, input)
    }

    #[inline]
    pub fn search_half_with(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
    ) -> Option<HalfMatch> {
        if self.imp.info.is_impossible(input) {
            return None;
        }
        self.imp.strat.search_half(cache, input)
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
        if self.imp.info.is_impossible(input) {
            return None;
        }
        self.imp.strat.search_slots(cache, input, slots)
    }

    #[inline]
    pub fn which_overlapping_matches_with(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
        patset: &mut PatternSet,
    ) {
        if self.imp.info.is_impossible(input) {
            return;
        }
        self.imp.strat.which_overlapping_matches(cache, input, patset)
    }
}

impl Clone for Regex {
    fn clone(&self) -> Regex {
        let imp = Arc::clone(&self.imp);
        let pool = {
            let strat = Arc::clone(&imp.strat);
            let create: CachePoolFn = Box::new(move || strat.create_cache());
            Pool::new(create)
        };
        Regex { imp, pool }
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
    cache: CachePoolGuard<'r>,
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
        // This does the deref for PoolGuard once instead of every iter.
        let cache = &mut *cache;
        it.into_half_matches_iter(
            |input| Ok(re.search_half_with(cache, input)),
        )
        .count()
    }
}

#[derive(Debug)]
pub struct CapturesMatches<'r, 'h> {
    re: &'r Regex,
    cache: CachePoolGuard<'r>,
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
        // This does the deref for PoolGuard once instead of every iter.
        let cache = &mut *cache;
        it.into_half_matches_iter(
            |input| Ok(re.search_half_with(cache, input)),
        )
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

/// An object describing the configuration of a [`Regex`].
///
/// This configuration only includes options for the
/// non-syntax behavior of a `Regex`, and can be applied via the
/// [`Builder::configure`] method. For configuring the syntax options, see
/// [`util::syntax::Config`](crate::util::syntax::Config).
///
/// # Example: lower the NFA size limit
///
/// In some cases, the default size limit might be too big. The size limit can
/// be lowered, which will prevent large regex patterns from compiling.
///
/// ```
/// use regex_automata::meta::Regex;
///
/// let result = Regex::builder()
///     .configure(Regex::config().nfa_size_limit(Some(20 * (1<<10))))
///     // Not even 20KB is enough to build a single large Unicode class!
///     .build(r"\pL");
/// assert!(result.is_err());
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
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
    utf8_empty: Option<bool>,
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
    /// Create a new configuration object for a `Regex`.
    pub fn new() -> Config {
        Config::default()
    }

    /// Set the match semantics for a `Regex`.
    ///
    /// The default value is [`MatchKind::LeftmostFirst`].
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{meta::Regex, Match, MatchKind};
    ///
    /// // By default, leftmost-first semantics are used, which
    /// // disambiguates matches at the same position by selecting
    /// // the one that corresponds earlier in the pattern.
    /// let re = Regex::new("sam|samwise")?;
    /// assert_eq!(Some(Match::must(0, 0..3)), re.find("samwise"));
    ///
    /// // But with 'all' semantics, match priority is ignored
    /// // and all match states are included. When coupled with
    /// // a leftmost search, the search will report the last
    /// // possible match.
    /// let re = Regex::builder()
    ///     .configure(Regex::config().match_kind(MatchKind::All))
    ///     .build("sam|samwise")?;
    /// assert_eq!(Some(Match::must(0, 0..7)), re.find("samwise"));
    /// // Beware that this can lead to skipping matches!
    /// // Usually 'all' is used for anchored reverse searches
    /// // only, or for overlapping searches.
    /// assert_eq!(Some(Match::must(0, 4..11)), re.find("sam samwise"));
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn match_kind(self, kind: MatchKind) -> Config {
        Config { match_kind: Some(kind), ..self }
    }

    /// Toggles whether empty matches are permitted to occur between the code
    /// units of a UTF-8 encoded codepoint.
    ///
    /// This should generally be enabled when search a `&str` or anything that
    /// you otherwise know is valid UTF-8. It should be disabled in all other
    /// cases. Namely, if the haystack is not valid UTF-8 and this is enabled,
    /// then behavior is unspecified.
    ///
    /// By default, this is enabled.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{meta::Regex, Match};
    ///
    /// let re = Regex::new("")?;
    /// let got: Vec<Match> = re.find_iter("☃").collect();
    /// // Matches only occur at the beginning and end of the snowman.
    /// assert_eq!(got, vec![
    ///     Match::must(0, 0..0),
    ///     Match::must(0, 3..3),
    /// ]);
    ///
    /// let re = Regex::builder()
    ///     .configure(Regex::config().utf8_empty(false))
    ///     .build("")?;
    /// let got: Vec<Match> = re.find_iter("☃").collect();
    /// // Matches now occur at every position!
    /// assert_eq!(got, vec![
    ///     Match::must(0, 0..0),
    ///     Match::must(0, 1..1),
    ///     Match::must(0, 2..2),
    ///     Match::must(0, 3..3),
    /// ]);
    ///
    /// Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn utf8_empty(self, yes: bool) -> Config {
        Config { utf8_empty: Some(yes), ..self }
    }

    /// Toggles whether automatic prefilter support is enabled.
    ///
    /// If this is disabled and [`Config::prefilter`] is not set, then the
    /// meta regex engine will not use any prefilters. This can sometimes
    /// be beneficial in cases where you know (or have measured) that the
    /// prefilter leads to overall worse search performance.
    ///
    /// By default, this is enabled.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{meta::Regex, Match};
    ///
    /// let re = Regex::builder()
    ///     .configure(Regex::config().auto_prefilter(false))
    ///     .build(r"Bruce \w+")?;
    /// let hay = "Hello Bruce Springsteen!";
    /// assert_eq!(Some(Match::must(0, 6..23)), re.find(hay));
    ///
    /// Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn auto_prefilter(self, yes: bool) -> Config {
        Config { autopre: Some(yes), ..self }
    }

    /// Overrides and sets the prefilter to use inside a `Regex`.
    ///
    /// This permits one to forcefully set a prefilter in cases where the
    /// caller knows better than whatever the automatic prefilter logic is
    /// capable of.
    ///
    /// By default, this is set to `None` and an automatic prefilter will be
    /// used if one could be built. (Assuming [`Config::auto_prefilter`] is
    /// enabled, which it is by default.)
    ///
    /// # Example
    ///
    /// This example shows how to set your own prefilter. In the case of a
    /// pattern like `Bruce \w+`, the automatic prefilter is likely to be
    /// constructed in a way that it will look for occurrences of `Bruce `.
    /// In most cases, this is the best choice. But in some cases, it may be
    /// the case that running `memchr` on `B` is the best choice. One can
    /// achieve that behavior by overriding the automatic prefilter logic
    /// and providing a prefilter that just matches `B`.
    ///
    /// ```
    /// use regex_automata::{
    ///     meta::Regex,
    ///     util::prefilter::Prefilter,
    ///     Match, MatchKind,
    /// };
    ///
    /// let pre = Prefilter::new(MatchKind::LeftmostFirst, &["B"])
    ///     .expect("a prefilter");
    /// let re = Regex::builder()
    ///     .configure(Regex::config().prefilter(Some(pre)))
    ///     .build(r"Bruce \w+")?;
    /// let hay = "Hello Bruce Springsteen!";
    /// assert_eq!(Some(Match::must(0, 6..23)), re.find(hay));
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: incorrect prefilters can lead to incorrect results!
    ///
    /// Be warned that setting an incorrect prefilter can lead to missed
    /// matches. So if you use this option, ensure your prefilter can _never_
    /// report false negatives. (A false positive is, on the other hand, quite
    /// okay and generally unavoidable.)
    ///
    /// ```
    /// use regex_automata::{
    ///     meta::Regex,
    ///     util::prefilter::Prefilter,
    ///     Match, MatchKind,
    /// };
    ///
    /// let pre = Prefilter::new(MatchKind::LeftmostFirst, &["Z"])
    ///     .expect("a prefilter");
    /// let re = Regex::builder()
    ///     .configure(Regex::config().prefilter(Some(pre)))
    ///     .build(r"Bruce \w+")?;
    /// let hay = "Hello Bruce Springsteen!";
    /// // Oops! No match found, but there should be one!
    /// assert_eq!(None, re.find(hay));
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn prefilter(self, pre: Option<Prefilter>) -> Config {
        Config { pre: Some(pre), ..self }
    }

    /// Sets the size limit to enforce on the construction of every NFA build
    /// by the meta regex engine.
    ///
    /// Note that this limit is applied to _each_ NFA built, and if any of
    /// them excceed the limit, then construction will fail. This limit does
    /// _not_ correspond to the total memory used by all NFAs in the meta regex
    /// engine.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::meta::Regex;
    ///
    /// let result = Regex::builder()
    ///     .configure(Regex::config().nfa_size_limit(Some(20 * (1<<10))))
    ///     // Not even 20KB is enough to build a single large Unicode class!
    ///     .build(r"\pL");
    /// assert!(result.is_err());
    ///
    /// // But notice that building such a regex with the exact same limit
    /// // can succeed depending on other aspects of the configuration. For
    /// // example, a single *forward* NFA will (at time of writing) fit into
    /// // the 20KB limit, but a *reverse* NFA of the same pattern will not.
    /// // So if one configures a meta regex such that a reverse NFA is never
    /// // needed and thus never built, then the 20KB limit will be enough for
    /// // a pattern like \pL!
    /// let result = Regex::builder()
    ///     .configure(Regex::config()
    ///         .nfa_size_limit(Some(20 * (1<<10)))
    ///         // The DFAs are the only thing that (currently) need a reverse
    ///         // NFA. So if both are disabled, the meta regex engine will
    ///         // skip building the reverse NFA. Note that this isn't an API
    ///         // guarantee. A future semver compatible version may introduce
    ///         // new use cases for a reverse NFA.
    ///         .hybrid(false)
    ///         .dfa(false)
    ///     )
    ///     // Not even 20KB is enough to build a single large Unicode class!
    ///     .build(r"\pL");
    /// assert!(result.is_ok());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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

    pub fn byte_classes(self, yes: bool) -> Config {
        Config { byte_classes: Some(yes), ..self }
    }

    pub fn line_terminator(self, byte: u8) -> Config {
        Config { line_terminator: Some(byte), ..self }
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

    pub fn get_match_kind(&self) -> MatchKind {
        self.match_kind.unwrap_or(MatchKind::LeftmostFirst)
    }

    pub fn get_utf8_empty(&self) -> bool {
        self.utf8_empty.unwrap_or(true)
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

    pub fn get_byte_classes(&self) -> bool {
        self.byte_classes.unwrap_or(true)
    }

    pub fn get_line_terminator(&self) -> u8 {
        self.line_terminator.unwrap_or(b'\n')
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

    /// Overwrite the default configuration such that the options in `o` are
    /// always used. If an option in `o` is not set, then the corresponding
    /// option in `self` is used. If it's not set in `self` either, then it
    /// remains not set.
    pub(crate) fn overwrite(&self, o: Config) -> Config {
        Config {
            match_kind: o.match_kind.or(self.match_kind),
            utf8_empty: o.utf8_empty.or(self.utf8_empty),
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

/// A builder for configuring and constructing a [`Regex`].
///
/// The builder permits configuring two different aspects of a `Regex`:
///
/// * [`Builder::configure`] will set high-level configuration options as
/// described by a [`Config`].
/// * [`Builder::syntax`] will set the syntax level configuration options
/// as described by a [`util::syntax::Config`](crate::util::syntax::Config).
/// This only applies when building a `Regex` from pattern strings.
///
/// Once configured, the builder can then be used to construct a `Regex` from
/// one of 4 different inputs:
///
/// * [`Builder::build`] creates a regex from a single pattern string.
/// * [`Builder::build_many`] creates a regex from many pattern strings.
/// * [`Builder::build_from_hir`] creates a regex from a
/// [`regex-syntax::Hir`](Hir) expression.
/// * [`Builder::build_many_from_hir`] creates a regex from many
/// [`regex-syntax::Hir`](Hir) expressions.
///
/// The latter two methods in particular provide a way to construct a fully
/// feature regular expression matcher directly from an `Hir` expression
/// without having to first convert it to a string. (This is in contrast to the
/// top-level `regex` crate which intentionally provides no such API in order
/// to avoid making `regex-syntax` a public dependency.)
///
/// As a convenience, this builder may be created via [`Regex::builder`], which
/// may help avoid an extra import.
///
/// # Example: change the line terminator
///
/// This example shows how to enable multi-line mode by default and change the
/// line terminator to the NUL byte:
///
/// ```
/// use regex_automata::{meta::Regex, util::syntax, Match};
///
/// let re = Regex::builder()
///     .syntax(syntax::Config::new().multi_line(true))
///     .configure(Regex::config().line_terminator(b'\x00'))
///     .build(r"^foo$")?;
/// let hay = "\x00foo\x00";
/// assert_eq!(Some(Match::must(0, 1..4)), re.find(hay));
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Example: disable UTF-8 requirement
///
/// By default, regex patterns are required to match UTF-8. This includes
/// regex patterns that can produce matches of length zero. In the case of an
/// empty match, by default, matches will not appear between the code units of
/// a UTF-8 encoded codepoint.
///
/// However, it can be useful to disable this requirement, particularly if
/// you're searching things like `&[u8]` that are not known to be valid UTF-8.
///
/// ```
/// use regex_automata::{meta::Regex, util::syntax, Match};
///
/// let mut builder = Regex::builder();
/// // Disables the requirement that non-empty matches match UTF-8.
/// builder.syntax(syntax::Config::new().utf8(false));
/// // Disables the requirement that empty matches match UTF-8 boundaries.
/// builder.configure(Regex::config().utf8_empty(false));
///
/// // We can match raw bytes via \xZZ syntax, but we need to disable
/// // Unicode mode to do that. We could disable it everywhere, or just
/// // selectively, as shown here.
/// let re = builder.build(r"(?-u:\xFF)foo(?-u:\xFF)")?;
/// let hay = b"\xFFfoo\xFF";
/// assert_eq!(Some(Match::must(0, 0..5)), re.find(hay));
///
/// // We can also match between code units.
/// let re = builder.build(r"")?;
/// let hay = "☃";
/// assert_eq!(re.find_iter(hay).collect::<Vec<Match>>(), vec![
///     Match::must(0, 0..0),
///     Match::must(0, 1..1),
///     Match::must(0, 2..2),
///     Match::must(0, 3..3),
/// ]);
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug)]
pub struct Builder {
    config: Config,
    ast: ast::parse::ParserBuilder,
    hir: hir::translate::TranslatorBuilder,
}

impl Builder {
    /// Creates a new builder for configuring and constructing a [`Regex`].
    pub fn new() -> Builder {
        Builder {
            config: Config::default(),
            ast: ast::parse::ParserBuilder::new(),
            hir: hir::translate::TranslatorBuilder::new(),
        }
    }

    /// Builds a `Regex` from a single pattern string.
    ///
    /// If there was a problem parsing the pattern or a problem turning it into
    /// a regex matcher, then an error is returned.
    ///
    /// # Example
    ///
    /// This example shows how to configure syntax options.
    ///
    /// ```
    /// use regex_automata::{meta::Regex, util::syntax, Match};
    ///
    /// let re = Regex::builder()
    ///     .syntax(syntax::Config::new().crlf(true).multi_line(true))
    ///     .build(r"^foo$")?;
    /// let hay = "\r\nfoo\r\n";
    /// assert_eq!(Some(Match::must(0, 2..5)), re.find(hay));
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn build(&self, pattern: &str) -> Result<Regex, BuildError> {
        self.build_many(&[pattern])
    }

    /// Builds a `Regex` from a many pattern strings.
    ///
    /// If there was a problem parsing any of the patterns or a problem turning
    /// them into a regex matcher, then an error is returned.
    ///
    /// # Example: finding the pattern that caused an error
    ///
    /// When a syntax error occurs, it is possible to ask which pattern
    /// caused the syntax error.
    ///
    /// ```
    /// use regex_automata::{meta::Regex, PatternID};
    ///
    /// let err = Regex::builder()
    ///     .build_many(&["a", "b", r"\p{Foo}", "c"])
    ///     .unwrap_err();
    /// assert_eq!(Some(PatternID::must(2)), err.pattern());
    /// ```
    ///
    /// # Example: zero patterns is a valid number
    ///
    /// Building a regex with zero patterns results in a regex that never
    /// matches anything. Because of the generics, passing an empty slice
    /// usually requires a turbo-fish (or something else to help type
    /// inference).
    ///
    /// ```
    /// use regex_automata::{meta::Regex, util::syntax, Match};
    ///
    /// let re = Regex::builder()
    ///     .build_many::<&str>(&[])?;
    /// assert_eq!(None, re.find(""));
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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

    /// Builds a `Regex` directly from an `Hir` expression.
    ///
    /// This is useful if you needed to parse a pattern string into an `Hir`
    /// for other reasons (such as analysis or transformations). This routine
    /// permits building a `Regex` directly from the `Hir` expression instead
    /// of first converting the `Hir` back to a pattern string.
    ///
    /// When using this method, any options set via [`Builder::syntax`] are
    /// ignored. Namely, the syntax options only apply when parsing a pattern
    /// string, which isn't relevant here.
    ///
    /// If there was a problem building the underlying regex matcher for the
    /// given `Hir`, then an error is returned.
    ///
    /// # Example
    ///
    /// This example shows how one can hand-construct an `Hir` expression and
    /// build a regex from it without doing any parsing at all.
    ///
    /// ```
    /// use {
    ///     regex_automata::{meta::Regex, Match},
    ///     regex_syntax::hir::{Hir, Look},
    /// };
    ///
    /// // (?Rm)^foo$
    /// let hir = Hir::concat(vec![
    ///     Hir::look(Look::StartCRLF),
    ///     Hir::literal("foo".as_bytes()),
    ///     Hir::look(Look::EndCRLF),
    /// ]);
    /// let re = Regex::builder()
    ///     .build_from_hir(&hir)?;
    /// let hay = "\r\nfoo\r\n";
    /// assert_eq!(Some(Match::must(0, 2..5)), re.find(hay));
    ///
    /// Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn build_from_hir(&self, hir: &Hir) -> Result<Regex, BuildError> {
        self.build_many_from_hir(&[hir])
    }

    /// Builds a `Regex` directly from many `Hir` expressions.
    ///
    /// This is useful if you needed to parse pattern strings into `Hir`
    /// expressions for other reasons (such as analysis or transformations).
    /// This routine permits building a `Regex` directly from the `Hir`
    /// expressions instead of first converting the `Hir` expressions back to
    /// pattern strings.
    ///
    /// When using this method, any options set via [`Builder::syntax`] are
    /// ignored. Namely, the syntax options only apply when parsing a pattern
    /// string, which isn't relevant here.
    ///
    /// If there was a problem building the underlying regex matcher for the
    /// given `Hir` expressions, then an error is returned.
    ///
    /// Note that unlike [`Builder::build_many`], this can only fail as a
    /// result of building the underlying matcher. In that case, there is
    /// no single `Hir` expression that can be isolated as a reason for the
    /// failure. So if this routine fails, it's not possible to determine which
    /// `Hir` expression caused the failure.
    ///
    /// # Example
    ///
    /// This example shows how one can hand-construct multiple `Hir`
    /// expressions and build a single regex from them without doing any
    /// parsing at all.
    ///
    /// ```
    /// use {
    ///     regex_automata::{meta::Regex, Match},
    ///     regex_syntax::hir::{Hir, Look},
    /// };
    ///
    /// // (?Rm)^foo$
    /// let hir1 = Hir::concat(vec![
    ///     Hir::look(Look::StartCRLF),
    ///     Hir::literal("foo".as_bytes()),
    ///     Hir::look(Look::EndCRLF),
    /// ]);
    /// // (?Rm)^bar$
    /// let hir2 = Hir::concat(vec![
    ///     Hir::look(Look::StartCRLF),
    ///     Hir::literal("bar".as_bytes()),
    ///     Hir::look(Look::EndCRLF),
    /// ]);
    /// let re = Regex::builder()
    ///     .build_many_from_hir(&[&hir1, &hir2])?;
    /// let hay = "\r\nfoo\r\nbar";
    /// let got: Vec<Match> = re.find_iter(hay).collect();
    /// let expected = vec![
    ///     Match::must(0, 2..5),
    ///     Match::must(1, 7..10),
    /// ];
    /// assert_eq!(expected, got);
    ///
    /// Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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
            let create: CachePoolFn = Box::new(move || strat.create_cache());
            Pool::new(create)
        };
        Ok(Regex { imp: Arc::new(RegexI { strat, info }), pool })
    }

    /// Configure the behavior of a `Regex`.
    ///
    /// This configuration controls non-syntax options related to the behavior
    /// of a `Regex`. This includes things like whether empty matches can split
    /// a codepoint, prefilters, line terminators and a long list of options
    /// for configuring which regex engines the meta regex engine will be able
    /// to use internally.
    ///
    /// # Example
    ///
    /// This example shows how to disable UTF-8 empty mode. This will permit
    /// empty matches to occur between the UTF-8 encoding of a codepoint.
    ///
    /// ```
    /// use regex_automata::{meta::Regex, Match};
    ///
    /// let re = Regex::new("")?;
    /// let got: Vec<Match> = re.find_iter("☃").collect();
    /// // Matches only occur at the beginning and end of the snowman.
    /// assert_eq!(got, vec![
    ///     Match::must(0, 0..0),
    ///     Match::must(0, 3..3),
    /// ]);
    ///
    /// let re = Regex::builder()
    ///     .configure(Regex::config().utf8_empty(false))
    ///     .build("")?;
    /// let got: Vec<Match> = re.find_iter("☃").collect();
    /// // Matches now occur at every position!
    /// assert_eq!(got, vec![
    ///     Match::must(0, 0..0),
    ///     Match::must(0, 1..1),
    ///     Match::must(0, 2..2),
    ///     Match::must(0, 3..3),
    /// ]);
    ///
    /// Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn configure(&mut self, config: Config) -> &mut Builder {
        self.config = self.config.overwrite(config);
        self
    }

    /// Configure the syntax options when parsing a pattern string while
    /// building a `Regex`.
    ///
    /// These options _only_ apply when [`Builder::build`] or [`Builder::build_many`]
    /// are used. The other build methods accept `Hir` values, which have
    /// already been parsed.
    ///
    /// # Example
    ///
    /// This example shows how to enable case insensitive mode.
    ///
    /// ```
    /// use regex_automata::{meta::Regex, util::syntax, Match};
    ///
    /// let re = Regex::builder()
    ///     .syntax(syntax::Config::new().case_insensitive(true))
    ///     .build(r"δ")?;
    /// assert_eq!(Some(Match::must(0, 0..2)), re.find(r"Δ"));
    ///
    /// Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn syntax(
        &mut self,
        config: crate::util::syntax::Config,
    ) -> &mut Builder {
        config.apply_ast(&mut self.ast);
        config.apply_hir(&mut self.hir);
        self
    }
}
