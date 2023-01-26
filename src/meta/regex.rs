use core::borrow::Borrow;

use alloc::{sync::Arc, vec, vec::Vec};

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
        captures::Captures,
        iter,
        prefilter::{self, Prefilter},
        primitives::{NonMaxUsize, PatternID},
        search::{HalfMatch, Input, Match, MatchError, MatchKind, PatternSet},
    },
};

#[derive(Clone, Debug)]
pub struct Regex {
    /// The core matching engine.
    strat: Arc<dyn Strategy>,
    /// Metadata about the regexes driving the strategy. The metadata is also
    /// usually stored inside the strategy too, but we put it here as well
    /// so that we can get quick access to it (without virtual calls) before
    /// executing the regex engine. For example, we use this metadata to
    /// detect a subset of cases where we know a match is impossible, and can
    /// thus avoid calling into the strategy at all.
    ///
    /// This is also why a RegexInfo itself is internally ref-counted so clones
    /// are cheap.
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
        self.strat.create_captures()
    }

    pub fn create_cache(&self) -> Cache {
        self.strat.create_cache()
    }

    pub fn reset_cache(&self, cache: &mut Cache) {
        self.strat.reset_cache(cache)
    }

    pub fn pattern_len(&self) -> usize {
        self.info.props().len()
    }

    pub fn get_config(&self) -> &Config {
        self.info.config()
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
    /// let mut cache = re.create_cache();
    /// assert!(re.is_match(&mut cache, "foo"));
    /// ```
    #[inline]
    pub fn is_match<'h, I: Into<Input<'h>>>(
        &self,
        cache: &mut Cache,
        input: I,
    ) -> bool {
        let input = input.into().earliest(true);
        self.search_half(cache, &input).is_some()
    }

    #[inline]
    pub fn find<'h, I: Into<Input<'h>>>(
        &self,
        cache: &mut Cache,
        input: I,
    ) -> Option<Match> {
        self.search(cache, &input.into())
    }

    #[inline]
    pub fn captures<'h, I: Into<Input<'h>>>(
        &self,
        cache: &mut Cache,
        input: I,
        caps: &mut Captures,
    ) {
        self.search_captures(cache, &input.into(), caps)
    }

    #[inline]
    pub fn find_iter<'r, 'c, 'h, I: Into<Input<'h>>>(
        &'r self,
        cache: &'c mut Cache,
        input: I,
    ) -> FindMatches<'r, 'c, 'h> {
        let it = iter::Searcher::new(input.into());
        FindMatches { re: self, cache, it }
    }

    #[inline]
    pub fn captures_iter<'r, 'c, 'h, I: Into<Input<'h>>>(
        &'r self,
        cache: &'c mut Cache,
        input: I,
    ) -> CapturesMatches<'r, 'c, 'h> {
        let caps = self.create_captures();
        let it = iter::Searcher::new(input.into());
        CapturesMatches { re: self, cache, caps, it }
    }
}

impl Regex {
    #[inline]
    pub fn search(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
    ) -> Option<Match> {
        if self.info.is_impossible(input) {
            return None;
        }
        self.strat.search(cache, input)
    }

    #[inline]
    pub fn search_half(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
    ) -> Option<HalfMatch> {
        if self.info.is_impossible(input) {
            return None;
        }
        self.strat.search_half(cache, input)
    }

    #[inline]
    pub fn search_captures(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
        caps: &mut Captures,
    ) {
        caps.set_pattern(None);
        let pid = self.search_slots(cache, input, caps.slots_mut());
        caps.set_pattern(pid);
    }

    #[inline]
    pub fn search_slots(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Option<PatternID> {
        if self.info.is_impossible(input) {
            return None;
        }
        self.strat.search_slots(cache, input, slots)
    }

    #[inline]
    pub fn which_overlapping_matches(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
        patset: &mut PatternSet,
    ) {
        if self.info.is_impossible(input) {
            return;
        }
        self.strat.which_overlapping_matches(cache, input, patset)
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

    /// Returns true when the search is guaranteed to be anchored. That is,
    /// when a match is reported, its offset is guaranteed to correspond to
    /// the start of the search.
    ///
    /// This includes returning true when `input` _isn't_ anchored but the
    /// underlying regex is.
    #[inline(always)]
    pub(crate) fn is_anchored_start(&self, input: &Input<'_>) -> bool {
        input.get_anchored().is_anchored() || self.is_always_anchored_start()
    }

    /// Returns true when this regex is always anchored to the start of a
    /// search. And in particular, that regardless of an `Input` configuration,
    /// if any match is reported it must start at `0`.
    #[inline(always)]
    pub(crate) fn is_always_anchored_start(&self) -> bool {
        use regex_syntax::hir::Look;
        self.props_union().look_set_prefix().contains(Look::Start)
    }

    /// Returns true when this regex is always anchored to the end of a
    /// search. And in particular, that regardless of an `Input` configuration,
    /// if any match is reported it must end at the end of the haystack.
    #[inline(always)]
    pub(crate) fn is_always_anchored_end(&self) -> bool {
        use regex_syntax::hir::Look;
        self.props_union().look_set_suffix().contains(Look::End)
    }

    /// Returns true if and only if it is known that a match is impossible
    /// for the given input. This is useful for short-circuiting and avoiding
    /// running the regex engine if it's known no match can be reported.
    #[inline(always)]
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
        // Same idea at minimum, but for maximum. This is trickier. We can
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
pub struct FindMatches<'r, 'c, 'h> {
    re: &'r Regex,
    cache: &'c mut Cache,
    it: iter::Searcher<'h>,
}

impl<'r, 'c, 'h> Iterator for FindMatches<'r, 'c, 'h> {
    type Item = Match;

    #[inline]
    fn next(&mut self) -> Option<Match> {
        let FindMatches { re, ref mut cache, ref mut it } = *self;
        it.advance(|input| Ok(re.search(cache, input)))
    }
}

#[derive(Debug)]
pub struct CapturesMatches<'r, 'c, 'h> {
    re: &'r Regex,
    cache: &'c mut Cache,
    caps: Captures,
    it: iter::Searcher<'h>,
}

impl<'r, 'c, 'h> Iterator for CapturesMatches<'r, 'c, 'h> {
    type Item = Captures;

    #[inline]
    fn next(&mut self) -> Option<Captures> {
        // Splitting 'self' apart seems necessary to appease borrowck.
        let CapturesMatches { re, ref mut cache, ref mut caps, ref mut it } =
            *self;
        let _ = it.advance(|input| {
            re.search_captures(cache, input, caps);
            Ok(caps.get_match())
        });
        if caps.is_match() {
            Some(caps.clone())
        } else {
            None
        }
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
        log! {
            use crate::util::primitives::IteratorIndexExt;

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
        for p in patterns.iter() {
            asts.push(
                self.ast.build().parse(p.as_ref()).map_err(BuildError::ast)?,
            );
        }
        for (p, ast) in patterns.iter().zip(asts.iter()) {
            let hir = self
                .hir
                .build()
                .translate(p.as_ref(), ast)
                .map_err(BuildError::hir)?;
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
        Ok(Regex { strat, info })
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
