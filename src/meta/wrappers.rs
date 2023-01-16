use crate::{
    meta::{BuildError, Config, RegexInfo},
    nfa::thompson::{pikevm, NFA},
    util::primitives::NonMaxUsize,
    HalfMatch, Input, Match, MatchError, MatchKind, PatternID, PatternSet,
};

#[cfg(feature = "dfa-build")]
use crate::dfa;
#[cfg(feature = "dfa-onepass")]
use crate::dfa::onepass;
#[cfg(feature = "hybrid")]
use crate::hybrid;
#[cfg(feature = "nfa-backtrack")]
use crate::nfa::thompson::backtrack;

#[derive(Debug)]
pub(crate) struct PikeVM(Option<PikeVMEngine>);

impl PikeVM {
    pub(crate) fn none() -> PikeVM {
        PikeVM(None)
    }

    pub(crate) fn new(
        info: &RegexInfo,
        nfa: &NFA,
    ) -> Result<PikeVM, BuildError> {
        PikeVMEngine::new(info, nfa).map(PikeVM)
    }

    pub(crate) fn create_cache(&self) -> PikeVMCache {
        PikeVMCache::new(self)
    }

    #[inline(always)]
    pub(crate) fn get(&self) -> Option<&PikeVMEngine> {
        self.0.as_ref()
    }
}

#[derive(Debug)]
pub(crate) struct PikeVMEngine(pikevm::PikeVM);

impl PikeVMEngine {
    pub(crate) fn new(
        info: &RegexInfo,
        nfa: &NFA,
    ) -> Result<Option<PikeVMEngine>, BuildError> {
        let pikevm_config =
            pikevm::Config::new().match_kind(info.config.get_match_kind());
        let engine = pikevm::Builder::new()
            .configure(pikevm_config)
            .build_from_nfa(nfa.clone())
            .map_err(BuildError::nfa)?;
        debug!("PikeVM built");
        Ok(Some(PikeVMEngine(engine)))
    }

    #[inline(always)]
    pub(crate) fn try_slots(
        &self,
        cache: &mut PikeVMCache,
        input: &Input<'_, '_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Result<Option<PatternID>, MatchError> {
        self.0.try_search_slots(cache.0.as_mut().unwrap(), input, slots)
    }

    #[inline(always)]
    pub(crate) fn which_overlapping_matches(
        &self,
        cache: &mut PikeVMCache,
        input: &Input<'_, '_>,
        patset: &mut PatternSet,
    ) -> Result<(), MatchError> {
        self.0.which_overlapping_matches(
            cache.0.as_mut().unwrap(),
            input,
            patset,
        )
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PikeVMCache(Option<pikevm::Cache>);

impl PikeVMCache {
    pub(crate) fn none() -> PikeVMCache {
        PikeVMCache(None)
    }

    pub(crate) fn new(builder: &PikeVM) -> PikeVMCache {
        PikeVMCache(builder.0.as_ref().map(|e| e.0.create_cache()))
    }

    pub(crate) fn reset(&mut self, builder: &PikeVM) {
        if let Some(ref e) = builder.0 {
            self.0.as_mut().unwrap().reset(&e.0);
        }
    }

    pub(crate) fn memory_usage(&self) -> usize {
        self.0.as_ref().map_or(0, |c| c.memory_usage())
    }
}

#[derive(Debug)]
pub(crate) struct BoundedBacktracker(Option<BoundedBacktrackerEngine>);

impl BoundedBacktracker {
    pub(crate) fn none() -> BoundedBacktracker {
        BoundedBacktracker(None)
    }

    pub(crate) fn new(
        info: &RegexInfo,
        nfa: &NFA,
    ) -> Result<BoundedBacktracker, BuildError> {
        BoundedBacktrackerEngine::new(info, nfa).map(BoundedBacktracker)
    }

    pub(crate) fn create_cache(&self) -> BoundedBacktrackerCache {
        BoundedBacktrackerCache::new(self)
    }

    #[inline(always)]
    pub(crate) fn get(
        &self,
        input: &Input<'_, '_>,
    ) -> Option<&BoundedBacktrackerEngine> {
        let engine = self.0.as_ref()?;
        // It is difficult to make the backtracker give up early if it is
        // guaranteed to eventually wind up in a match state. This is because
        // of the greedy nature of a backtracker: it just blindly mushes
        // forward. Every other regex engine is able to give up more quickly,
        // so even if the backtracker might be able to zip through faster than
        // (say) the PikeVM, we prefer the theoretical benefit that some other
        // engine might be able to scan much less of the haystack than the
        // backtracker.
        //
        // Now, if the haystack is really short already, then we allow the
        // backtracker to run. (This hasn't been litigated quantitatively with
        // benchmarks. Just a hunch.)
        if input.get_earliest() && input.haystack().len() > 128 {
            return None;
        }
        // If the backtracker is just going to return an error because the
        // haystack is too long, then obviously do not use it.
        if input.get_span().len() > engine.max_haystack_len() {
            return None;
        }
        Some(engine)
    }
}

#[derive(Debug)]
pub(crate) struct BoundedBacktrackerEngine(
    #[cfg(feature = "nfa-backtrack")] backtrack::BoundedBacktracker,
    #[cfg(not(feature = "nfa-backtrack"))] (),
);

impl BoundedBacktrackerEngine {
    pub(crate) fn new(
        info: &RegexInfo,
        nfa: &NFA,
    ) -> Result<Option<BoundedBacktrackerEngine>, BuildError> {
        #[cfg(feature = "nfa-backtrack")]
        {
            if !info.config.get_backtrack()
                || info.config.get_match_kind() != MatchKind::LeftmostFirst
            {
                return Ok(None);
            }
            let backtrack_config = backtrack::Config::new();
            let engine = backtrack::Builder::new()
                .configure(backtrack_config)
                .build_from_nfa(nfa.clone())
                .map_err(BuildError::nfa)?;
            debug!("BoundedBacktracker built");
            Ok(Some(BoundedBacktrackerEngine(engine)))
        }
        #[cfg(not(feature = "nfa-backtrack"))]
        {
            Ok(None)
        }
    }

    #[inline(always)]
    pub(crate) fn try_slots(
        &self,
        cache: &mut BoundedBacktrackerCache,
        input: &Input<'_, '_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Result<Option<PatternID>, MatchError> {
        #[cfg(feature = "nfa-backtrack")]
        {
            self.0.try_search_slots(cache.0.as_mut().unwrap(), input, slots)
        }
        #[cfg(not(feature = "nfa-backtrack"))]
        {
            // Impossible to reach because this engine is never constructed
            // if the requisite features aren't enabled.
            unreachable!()
        }
    }

    #[inline(always)]
    fn max_haystack_len(&self) -> usize {
        #[cfg(feature = "nfa-backtrack")]
        {
            self.0.max_haystack_len()
        }
        #[cfg(not(feature = "nfa-backtrack"))]
        {
            // Impossible to reach because this engine is never constructed
            // if the requisite features aren't enabled.
            unreachable!()
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct BoundedBacktrackerCache(
    #[cfg(feature = "nfa-backtrack")] Option<backtrack::Cache>,
    #[cfg(not(feature = "nfa-backtrack"))] (),
);

impl BoundedBacktrackerCache {
    pub(crate) fn none() -> BoundedBacktrackerCache {
        BoundedBacktrackerCache(None)
    }

    pub(crate) fn new(
        builder: &BoundedBacktracker,
    ) -> BoundedBacktrackerCache {
        #[cfg(feature = "nfa-backtrack")]
        {
            BoundedBacktrackerCache(
                builder.0.as_ref().map(|e| e.0.create_cache()),
            )
        }
        #[cfg(not(feature = "nfa-backtrack"))]
        {
            BoundedBacktrackerCache(())
        }
    }

    pub(crate) fn reset(&mut self, builder: &BoundedBacktracker) {
        #[cfg(feature = "nfa-backtrack")]
        if let Some(ref e) = builder.0 {
            self.0.as_mut().unwrap().reset(&e.0);
        }
    }

    pub(crate) fn memory_usage(&self) -> usize {
        #[cfg(feature = "nfa-backtrack")]
        {
            self.0.as_ref().map_or(0, |c| c.memory_usage())
        }
        #[cfg(not(feature = "nfa-backtrack"))]
        {
            0
        }
    }
}

#[derive(Debug)]
pub(crate) struct Hybrid(Option<HybridEngine>);

impl Hybrid {
    pub(crate) fn none() -> Hybrid {
        Hybrid(None)
    }

    pub(crate) fn new(info: &RegexInfo, nfa: &NFA, nfarev: &NFA) -> Hybrid {
        Hybrid(HybridEngine::new(info, nfa, nfarev))
    }

    pub(crate) fn create_cache(&self) -> HybridCache {
        HybridCache::new(self)
    }

    #[inline(always)]
    pub(crate) fn get(&self, input: &Input<'_, '_>) -> Option<&HybridEngine> {
        let engine = self.0.as_ref()?;
        Some(engine)
    }
}

#[derive(Debug)]
pub(crate) struct HybridEngine(
    #[cfg(feature = "hybrid")] hybrid::regex::Regex,
    #[cfg(not(feature = "hybrid"))] (),
);

impl HybridEngine {
    pub(crate) fn new(
        info: &RegexInfo,
        nfa: &NFA,
        nfarev: &NFA,
    ) -> Option<HybridEngine> {
        #[cfg(feature = "hybrid")]
        {
            if !info.config.get_hybrid() {
                return None;
            }
            let dfa_config = hybrid::dfa::Config::new()
                .match_kind(info.config.get_match_kind())
                // Unconditionally enabling this seems fine for a meta regex
                // engine. It isn't too costly. At worst, it just uses a bit
                // more memory for lazy DFAs (and DFAs, but we don't use those
                // yet in the meta engine), and makes the API more flexible by
                // always supporting anchored searches for any of the patterns
                // in the regex.
                .starts_for_each_pattern(true)
                .byte_classes(info.config.get_byte_classes())
                .unicode_word_boundary(true)
                // TODO: Only set this to true if we have a prefilter
                .specialize_start_states(true)
                // .cache_capacity(info.config.get_hybrid_cache_capacity())
                .cache_capacity(info.config.get_hybrid_cache_capacity())
                // This makes it possible for building a lazy DFA to
                // fail even though the NFA has already been built. Namely,
                // if the cache capacity is too small to fit some minimum
                // number of states (which is small, like 4 or 5), then the
                // DFA will refuse to build.
                //
                // We shouldn't enable this to make building always work, since
                // this could cause the allocation of a cache bigger than the
                // provided capacity amount.
                //
                // This is effectively the only reason why building a lazy DFA
                // could fail. If it does, then we simply suppress the error
                // and return None.
                .skip_cache_capacity_check(false)
                // This and enabling heuristic Unicode word boundary support
                // above make it so the lazy DFA can quit at match time.
                .minimum_cache_clear_count(Some(10));
            let result = hybrid::dfa::Builder::new()
                .configure(dfa_config.clone())
                .build_from_nfa(nfa.clone());
            let fwd = match result {
                Ok(fwd) => fwd,
                Err(err) => {
                    debug!("forward lazy DFA failed to build: {}", err);
                    return None;
                }
            };
            let result = hybrid::dfa::Builder::new()
                .configure(dfa_config.clone().match_kind(MatchKind::All))
                .build_from_nfa(nfarev.clone());
            let rev = match result {
                Ok(rev) => rev,
                Err(err) => {
                    debug!("reverse lazy DFA failed to build: {}", err);
                    return None;
                }
            };
            let hybrid_config = hybrid::regex::Config::new();
            let engine = hybrid::regex::Builder::new()
                .configure(hybrid_config)
                .build_from_dfas(fwd, rev);
            debug!("lazy DFA built");
            Some(HybridEngine(engine))
        }
        #[cfg(not(feature = "hybrid"))]
        {
            None
        }
    }

    #[inline(always)]
    pub(crate) fn try_find_earliest(
        &self,
        cache: &mut HybridCache,
        input: &Input<'_, '_>,
    ) -> Result<Option<HalfMatch>, MatchError> {
        #[cfg(feature = "hybrid")]
        {
            let fwd = self.0.forward();
            let mut fwdcache = cache.0.as_mut().unwrap().as_parts_mut().0;
            fwd.try_search_fwd(&mut fwdcache, input)
        }
        #[cfg(not(feature = "hybrid"))]
        {
            // Impossible to reach because this engine is never constructed
            // if the requisite features aren't enabled.
            unreachable!()
        }
    }

    #[inline(always)]
    pub(crate) fn try_find(
        &self,
        cache: &mut HybridCache,
        input: &Input<'_, '_>,
    ) -> Result<Option<Match>, MatchError> {
        #[cfg(feature = "hybrid")]
        {
            self.0.try_search(cache.0.as_mut().unwrap(), input)
        }
        #[cfg(not(feature = "hybrid"))]
        {
            // Impossible to reach because this engine is never constructed
            // if the requisite features aren't enabled.
            unreachable!()
        }
    }

    #[inline]
    pub(crate) fn try_which_overlapping_matches(
        &self,
        cache: &mut HybridCache,
        input: &Input<'_, '_>,
        patset: &mut PatternSet,
    ) -> Result<(), MatchError> {
        #[cfg(feature = "hybrid")]
        {
            let fwd = self.0.forward();
            let mut fwdcache = cache.0.as_mut().unwrap().as_parts_mut().0;
            fwd.try_which_overlapping_matches(&mut fwdcache, input, patset)
        }
        #[cfg(not(feature = "hybrid"))]
        {
            // Impossible to reach because this engine is never constructed
            // if the requisite features aren't enabled.
            unreachable!()
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct HybridCache(
    #[cfg(feature = "hybrid")] Option<hybrid::regex::Cache>,
    #[cfg(not(feature = "hybrid"))] (),
);

impl HybridCache {
    pub(crate) fn none() -> HybridCache {
        HybridCache(None)
    }

    pub(crate) fn new(builder: &Hybrid) -> HybridCache {
        #[cfg(feature = "hybrid")]
        {
            HybridCache(builder.0.as_ref().map(|e| e.0.create_cache()))
        }
        #[cfg(not(feature = "hybrid"))]
        {
            HybridCache(())
        }
    }

    pub(crate) fn reset(&mut self, builder: &Hybrid) {
        #[cfg(feature = "hybrid")]
        if let Some(ref e) = builder.0 {
            self.0.as_mut().unwrap().reset(&e.0);
        }
    }

    pub(crate) fn memory_usage(&self) -> usize {
        #[cfg(feature = "hybrid")]
        {
            self.0.as_ref().map_or(0, |c| c.memory_usage())
        }
        #[cfg(not(feature = "hybrid"))]
        {
            0
        }
    }
}

#[derive(Debug)]
pub(crate) struct DFA(Option<DFAEngine>);

impl DFA {
    pub(crate) fn none() -> DFA {
        DFA(None)
    }

    pub(crate) fn new(info: &RegexInfo, nfa: &NFA, nfarev: &NFA) -> DFA {
        DFA(DFAEngine::new(info, nfa, nfarev))
    }

    #[inline(always)]
    pub(crate) fn get(&self, input: &Input<'_, '_>) -> Option<&DFAEngine> {
        let engine = self.0.as_ref()?;
        Some(engine)
    }

    pub(crate) fn is_some(&self) -> bool {
        self.0.is_some()
    }
}

#[derive(Debug)]
pub(crate) struct DFAEngine(
    #[cfg(feature = "dfa-build")] dfa::regex::Regex,
    #[cfg(not(feature = "dfa-build"))] (),
);

impl DFAEngine {
    pub(crate) fn new(
        info: &RegexInfo,
        nfa: &NFA,
        nfarev: &NFA,
    ) -> Option<DFAEngine> {
        #[cfg(feature = "dfa-build")]
        {
            if !info.config.get_dfa() {
                return None;
            }
            // If our NFA is anything but small, don't even bother with a DFA.
            if let Some(state_limit) = info.config.get_dfa_state_limit() {
                if nfa.states().len() > state_limit {
                    return None;
                }
            }
            // We cut the size limit in four because the total heap used by
            // DFA construction is determinization aux memory and the DFA
            // itself, and those things are configured independently in the
            // lower level DFA builder API. And then split that in two because
            // of forward and reverse DFAs.
            let size_limit = info.config.get_dfa_size_limit().map(|n| n / 4);
            let dfa_config = dfa::dense::Config::new()
                .match_kind(info.config.get_match_kind())
                // Unconditionally enabling this seems fine for a meta regex
                // engine. It can be most costly for DFAs than lazy DFAs,
                // because it means computing the transition table for both
                // unanchored and anchored starting states. But since we keep
                // a VERY TIGHT bound on the total size of the DFA, this just
                // in practice means we don't use the DFA as much as we might
                // otherwise if we only had to build anchored or unanchored
                // support instead of both.
                .starts_for_each_pattern(true)
                .byte_classes(info.config.get_byte_classes())
                .unicode_word_boundary(true)
                // TODO: Only set this to true if we have a prefilter
                .specialize_start_states(true)
                .determinize_size_limit(size_limit)
                .dfa_size_limit(size_limit);
            let fwd = dfa::dense::Builder::new()
                .configure(dfa_config.clone())
                .build_from_nfa(&nfa)
                .ok()?;
            let rev = dfa::dense::Builder::new()
                .configure(dfa_config.clone().match_kind(MatchKind::All))
                .build_from_nfa(&nfarev)
                .ok()?;
            let regex_config = dfa::regex::Config::new();
            let engine = dfa::regex::Builder::new()
                .configure(regex_config)
                .build_from_dfas(fwd, rev);
            debug!("fully compiled DFA built");
            Some(DFAEngine(engine))
        }
        #[cfg(not(feature = "dfa-build"))]
        {
            None
        }
    }

    #[inline(always)]
    pub(crate) fn try_find_earliest(
        &self,
        input: &Input<'_, '_>,
    ) -> Result<Option<HalfMatch>, MatchError> {
        #[cfg(feature = "dfa-build")]
        {
            use crate::dfa::Automaton;
            self.0.forward().try_search_fwd(input)
        }
        #[cfg(not(feature = "dfa-build"))]
        {
            // Impossible to reach because this engine is never constructed
            // if the requisite features aren't enabled.
            unreachable!()
        }
    }

    #[inline(always)]
    pub(crate) fn try_find(
        &self,
        input: &Input<'_, '_>,
    ) -> Result<Option<Match>, MatchError> {
        #[cfg(feature = "dfa-build")]
        {
            self.0.try_search(input)
        }
        #[cfg(not(feature = "dfa-build"))]
        {
            // Impossible to reach because this engine is never constructed
            // if the requisite features aren't enabled.
            unreachable!()
        }
    }

    #[inline]
    pub(crate) fn try_which_overlapping_matches(
        &self,
        input: &Input<'_, '_>,
        patset: &mut PatternSet,
    ) -> Result<(), MatchError> {
        #[cfg(feature = "hybrid")]
        {
            use crate::dfa::Automaton;
            self.0.forward().try_which_overlapping_matches(input, patset)
        }
        #[cfg(not(feature = "dfa-build"))]
        {
            // Impossible to reach because this engine is never constructed
            // if the requisite features aren't enabled.
            unreachable!()
        }
    }
}

#[derive(Debug)]
pub(crate) struct OnePass(Option<OnePassEngine>);

impl OnePass {
    pub(crate) fn none() -> OnePass {
        OnePass(None)
    }

    pub(crate) fn new(info: &RegexInfo, nfa: &NFA) -> OnePass {
        OnePass(OnePassEngine::new(info, nfa))
    }

    pub(crate) fn create_cache(&self) -> OnePassCache {
        OnePassCache::new(self)
    }

    #[inline(always)]
    pub(crate) fn get(&self, input: &Input<'_, '_>) -> Option<&OnePassEngine> {
        let engine = self.0.as_ref()?;
        if !input.get_anchored().is_anchored() {
            return None;
        }
        Some(engine)
    }
}

#[derive(Debug)]
pub(crate) struct OnePassEngine(
    #[cfg(feature = "dfa-onepass")] onepass::DFA,
    #[cfg(not(feature = "dfa-onepass"))] (),
);

impl OnePassEngine {
    pub(crate) fn new(info: &RegexInfo, nfa: &NFA) -> Option<OnePassEngine> {
        #[cfg(feature = "dfa-onepass")]
        {
            use regex_syntax::hir::Look;

            if !info.config.get_onepass() {
                return None;
            }
            // In order to even attempt building a one-pass DFA, we require
            // that we either have at least one explicit capturing group or
            // there's a Unicode word boundary somewhere. If we don't have
            // either of these things, then the lazy DFA will almost certainly
            // be useable and be much faster. The only case where it might
            // not is if the lazy DFA isn't utilizing its cache effectively,
            // but in those cases, the underlying regex is almost certainly
            // not one-pass or is too big to fit within the current one-pass
            // implementation limits.
            if info.props_union.captures_len() == 0
                && !info.props_union.look_set().contains_word_unicode()
            {
                debug!("not building OnePass because it isn't worth it");
                return None;
            }
            let onepass_config = onepass::Config::new()
                .match_kind(info.config.get_match_kind())
                .utf8(info.config.get_utf8())
                // Like for the lazy DFA, we unconditionally enable this
                // because it doesn't cost much and makes the API more
                // flexible.
                .starts_for_each_pattern(true)
                .byte_classes(info.config.get_byte_classes())
                .size_limit(info.config.get_onepass_size_limit());
            let result = onepass::Builder::new()
                .configure(onepass_config)
                .build_from_nfa(nfa.clone());
            let engine = match result {
                Ok(engine) => engine,
                Err(err) => {
                    debug!("OnePass failed to build: {}", err);
                    return None;
                }
            };
            debug!("OnePass built");
            Some(OnePassEngine(engine))
        }
        #[cfg(not(feature = "dfa-onepass"))]
        {
            None
        }
    }

    #[inline(always)]
    pub(crate) fn try_slots(
        &self,
        cache: &mut OnePassCache,
        input: &Input<'_, '_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Result<Option<PatternID>, MatchError> {
        #[cfg(feature = "dfa-onepass")]
        {
            self.0.try_search_slots(cache.0.as_mut().unwrap(), input, slots)
        }
        #[cfg(not(feature = "dfa-onepass"))]
        {
            // Impossible to reach because this engine is never constructed
            // if the requisite features aren't enabled.
            unreachable!()
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct OnePassCache(
    #[cfg(feature = "dfa-onepass")] Option<onepass::Cache>,
    #[cfg(not(feature = "dfa-onepass"))] (),
);

impl OnePassCache {
    pub(crate) fn none() -> OnePassCache {
        OnePassCache(None)
    }

    pub(crate) fn new(builder: &OnePass) -> OnePassCache {
        #[cfg(feature = "dfa-onepass")]
        {
            OnePassCache(builder.0.as_ref().map(|e| e.0.create_cache()))
        }
        #[cfg(not(feature = "dfa-onepass"))]
        {
            OnePassCache(())
        }
    }

    pub(crate) fn reset(&mut self, builder: &OnePass) {
        #[cfg(feature = "dfa-onepass")]
        if let Some(ref e) = builder.0 {
            self.0.as_mut().unwrap().reset(&e.0);
        }
    }

    pub(crate) fn memory_usage(&self) -> usize {
        #[cfg(feature = "dfa-onepass")]
        {
            self.0.as_ref().map_or(0, |c| c.memory_usage())
        }
        #[cfg(not(feature = "dfa-onepass"))]
        {
            0
        }
    }
}
