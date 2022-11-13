// Our base requirements are 'syntax' and the PikeVM. We could in theory only
// require an NFA and at least one (any) regex engine, but that seems likely to
// lead to a quite complicated internal setup. (And things are already going to
// be complicated.) We really want 'syntax' so we can do literal extraction and
// also control how the regex engines are built. For example, we might not want
// to build an NFA at all, but instead just build an Aho-Corasick automaton.
//
// So I guess the meta regex engine has two different decision trees: one at
// build time and one at search time.
//
// Build time
// ----------
//
//
// Search time
// -----------

#![allow(warnings)]

use core::borrow::Borrow;

use alloc::{sync::Arc, vec, vec::Vec};

use regex_syntax::{
    ast,
    hir::{self, Hir},
};

use crate::{
    meta::strategy::Strategy,
    nfa::thompson::pikevm,
    util::{
        captures::Captures,
        iter,
        primitives::{NonMaxUsize, PatternID},
        search::{Input, Match, MatchError, MatchKind, PatternSet},
    },
};

#[cfg(feature = "dfa-onepass")]
use crate::dfa::onepass;
#[cfg(feature = "hybrid")]
use crate::hybrid;
#[cfg(feature = "nfa-backtrack")]
use crate::nfa::thompson::backtrack;

pub use self::error::BuildError;

mod error;
mod strategy;
mod wrappers;

#[derive(Clone, Debug)]
pub struct Regex {
    info: RegexInfo,
    strat: Arc<dyn Strategy>,
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

    #[inline]
    pub fn create_input<'h, 'p, H: ?Sized + AsRef<[u8]>>(
        &'p self,
        haystack: &'h H,
    ) -> Input<'h, 'p> {
        let c = self.get_config();
        Input::new(haystack.as_ref())
            // .prefilter(c.get_prefilter())
            .utf8(c.get_utf8())
    }

    pub fn create_captures(&self) -> Captures {
        self.strat.create_captures()
    }

    pub fn create_cache(&self) -> Cache {
        Cache::new(self)
    }

    pub fn reset_cache(&self, cache: &mut Cache) {
        cache.reset(self)
    }

    pub fn pattern_len(&self) -> usize {
        self.info.props.len()
    }

    pub fn get_config(&self) -> &Config {
        &self.info.config
    }

    pub fn memory_usage(&self) -> usize {
        0
    }
}

impl Regex {
    #[inline]
    pub fn try_is_match<H: AsRef<[u8]>>(
        &self,
        cache: &mut Cache,
        haystack: H,
    ) -> Result<bool, MatchError> {
        let input = self.create_input(haystack.as_ref()).earliest(true);
        self.strat.try_is_match(cache, &input)
    }

    #[inline]
    pub fn try_find<H: AsRef<[u8]>>(
        &self,
        cache: &mut Cache,
        haystack: H,
    ) -> Result<Option<Match>, MatchError> {
        let input = self.create_input(haystack.as_ref());
        self.try_search(cache, &input)
    }

    #[inline]
    pub fn try_find_iter<'r, 'c, 'h, H: AsRef<[u8]> + ?Sized>(
        &'r self,
        cache: &'c mut Cache,
        haystack: &'h H,
    ) -> TryFindMatches<'r, 'c, 'h> {
        let input = self.create_input(haystack.as_ref());
        let it = iter::Searcher::new(input);
        TryFindMatches { re: self, cache, it }
    }

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

impl Regex {
    #[inline]
    pub fn try_search(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
    ) -> Result<Option<Match>, MatchError> {
        self.strat.try_find(cache, input)
    }

    #[inline]
    pub fn try_search_captures(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        caps: &mut Captures,
    ) -> Result<(), MatchError> {
        caps.set_pattern(None);
        let pid = self.try_search_slots(cache, input, caps.slots_mut())?;
        caps.set_pattern(pid);
        Ok(())
    }

    #[inline]
    pub fn try_search_slots(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Result<Option<PatternID>, MatchError> {
        self.strat.try_slots(cache, input, slots)
    }

    #[inline]
    pub fn try_which_overlapping_matches(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        patset: &mut PatternSet,
    ) -> Result<(), MatchError> {
        self.strat.try_which_overlapping_matches(cache, input, patset)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RegexInfo {
    pub(crate) config: Config,
    pub(crate) props: Vec<hir::Properties>,
    pub(crate) props_union: hir::Properties,
}

#[derive(Debug)]
pub struct TryFindMatches<'r, 'c, 'h> {
    re: &'r Regex,
    cache: &'c mut Cache,
    it: iter::Searcher<'h, 'r>,
}

impl<'r, 'c, 'h> Iterator for TryFindMatches<'r, 'c, 'h> {
    type Item = Result<Match, MatchError>;

    #[inline]
    fn next(&mut self) -> Option<Result<Match, MatchError>> {
        let TryFindMatches { re, ref mut cache, ref mut it } = *self;
        it.try_advance(|input| re.try_search(cache, input)).transpose()
    }
}

#[derive(Debug)]
pub struct TryCapturesMatches<'r, 'c, 'h> {
    re: &'r Regex,
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
        let _ = it
            .try_advance(|input| {
                re.try_search_captures(cache, input, caps)?;
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

#[derive(Debug, Clone)]
pub struct Cache {
    capmatches: Captures,
    pikevm: wrappers::PikeVMCache,
    backtrack: wrappers::BoundedBacktrackerCache,
    onepass: wrappers::OnePassCache,
    hybrid: wrappers::HybridCache,
}

impl Cache {
    pub fn new(re: &Regex) -> Cache {
        re.strat.create_cache()
    }

    pub fn reset(&mut self, re: &Regex) {
        re.strat.reset_cache(self)
    }

    pub fn memory_usage(&self) -> usize {
        let mut bytes = 0;
        bytes += self.pikevm.memory_usage();
        bytes += self.backtrack.memory_usage();
        bytes += self.onepass.memory_usage();
        bytes += self.hybrid.memory_usage();
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
    nfa_size_limit: Option<Option<usize>>,
    onepass_size_limit: Option<Option<usize>>,
    hybrid_cache_capacity: Option<usize>,
    hybrid: Option<bool>,
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

    pub fn nfa_size_limit(self, limit: Option<usize>) -> Config {
        Config { nfa_size_limit: Some(limit), ..self }
    }

    pub fn onepass_size_limit(self, limit: Option<usize>) -> Config {
        Config { onepass_size_limit: Some(limit), ..self }
    }

    pub fn hybrid_cache_capacity(self, limit: usize) -> Config {
        Config { hybrid_cache_capacity: Some(limit), ..self }
    }

    pub fn hybrid(self, yes: bool) -> Config {
        Config { hybrid: Some(yes), ..self }
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

    pub fn get_nfa_size_limit(&self) -> Option<usize> {
        self.nfa_size_limit.unwrap_or(Some(10 * (1 << 20)))
    }

    pub fn get_onepass_size_limit(&self) -> Option<usize> {
        self.onepass_size_limit.unwrap_or(Some(500 * (1 << 10)))
    }

    pub fn get_hybrid_cache_capacity(&self) -> usize {
        self.hybrid_cache_capacity.unwrap_or(2 * (1 << 20))
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
            nfa_size_limit: o.nfa_size_limit.or(self.nfa_size_limit),
            onepass_size_limit: o
                .onepass_size_limit
                .or(self.onepass_size_limit),
            hybrid_cache_capacity: o
                .hybrid_cache_capacity
                .or(self.hybrid_cache_capacity),
            hybrid: o.hybrid.or(self.hybrid),
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
        // We collect the HIRs into a vec so we can write internal routines
        // with '&[&Hir]'. i.e., Don't use generics everywhere to keep code
        // bloat down..
        let hirs: Vec<&Hir> = hirs.iter().map(|hir| hir.borrow()).collect();
        let config = self.config.clone();
        let mut props = vec![];
        for hir in hirs.iter() {
            props.push(hir.properties().clone());
        }
        let props_union = hir::Properties::union(&props);
        let info = RegexInfo { config, props, props_union };
        let strat = self::strategy::new(&info, &hirs)?;
        Ok(Regex { info, strat })
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
