use core::{
    borrow::Borrow,
    fmt::Debug,
    panic::{RefUnwindSafe, UnwindSafe},
};

use alloc::sync::Arc;

use regex_syntax::hir::{self, Hir};

use crate::{
    meta::{self, wrappers, BuildError},
    nfa::thompson::{self, pikevm::PikeVM, NFA},
    util::{
        captures::Captures,
        primitives::{NonMaxUsize, PatternID},
        search::{Input, Match, MatchError, PatternSet},
    },
    MatchKind,
};

#[cfg(feature = "dfa-onepass")]
use crate::dfa::onepass;
#[cfg(feature = "hybrid")]
use crate::hybrid;
#[cfg(feature = "nfa-backtrack")]
use crate::nfa::thompson::backtrack;

// BREADCRUMBS:
//
// This whole 'Strategy' trait just doesn't feel right... At first I thought I
// would have a whole bunch of impls, but the trait has a lot of surface area
// and having a lot of impls would be really quite annoying.
//
// So maybe we need to list out what the actual strategies are? The other issue
// is that we might want to change strategies if one proves ineffective.
//
// Maybe another way to think about this is to split strategies up into
// different components and then compose them. Like one component could be
// "call this to extract captures." And that routine knows whether to use
// onpass, backtracking or the PikeVM.
//
// Issues crop up with unbridled composition though. For example, if someone
// asks for captures, we would normally want to run the lazy DFA to find
// the match bounds and *then* ask for captures. But if the lazy DFA is
// unavailable, then we'd probably just want to just run an unanchored search
// with backtracking or the PikeVM directly instead of first trying to find the
// match bounds. OK, let's try to write this out in pseudo-code for each of the
// 3 fundamental questions: has a match? where is it? where are the submatches?
//
// definitions
// -----------
// anchored_at_end:
//   props.look_set_suffix().contains(Look::End)
// ahocorasick:
//   cfg(perf-literal-multisubstring)
//     && props.captures_len() == 0
//     && props.is_alternation_literal()
// substring:
//   cfg(perf-literal-substring)
//     && props.captures_len() == 0
//     && props.is_literal()
// lazydfa:
//   cfg(hybrid) && config.get_hybrid()
// reversesuffix:
//   lazydfa
//     && len(suffixes.longest_common_suffix()) >= 3
// onepass
//   cfg(onepass) && config.get_onepass() && OneBuild::new().is_ok()
// backtrack
//   cfg(backtrack)
//
// is_match
// --------
// if anchored_at_end && lazydfa:
//   if let Ok(matched) = lazydfa.reverse().run():
//      return matched
// elif substring:
//   return substring.run()
// elif ahocorasick:
//   return ahocorasick.run()
// elif reversesuffix:
//   if let Ok(matched) = reversesuffix.run():
//      return matched
// elif lazydfa:
//   if let Ok(matched) = lazydfa.forward().run():
//      return matched
// if onepass:
//   return onepass.is_match()
// if backtrack:
//   return backtrack.is_match()
// return pikevm.is_match()

pub(super) trait Strategy:
    Debug + Send + Sync + RefUnwindSafe + UnwindSafe
{
    fn create_captures(&self) -> Captures;

    fn create_cache(&self) -> meta::Cache;

    fn reset_cache(&self, cache: &mut meta::Cache);

    fn try_is_match(
        &self,
        cache: &mut meta::Cache,
        input: &Input<'_, '_>,
    ) -> Result<bool, MatchError>;

    fn try_find(
        &self,
        cache: &mut meta::Cache,
        input: &Input<'_, '_>,
    ) -> Result<Option<Match>, MatchError>;

    fn try_slots(
        &self,
        cache: &mut meta::Cache,
        input: &Input<'_, '_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Result<Option<PatternID>, MatchError>;

    fn try_which_overlapping_matches(
        &self,
        cache: &mut meta::Cache,
        input: &Input<'_, '_>,
        patset: &mut PatternSet,
    ) -> Result<(), MatchError>;
}

pub(super) fn new(
    config: &meta::Config,
    props: &[hir::Properties],
    props_union: &hir::Properties,
    hirs: &[&Hir],
) -> Result<Arc<dyn Strategy>, BuildError> {
    Core::new(config, hirs)
}

#[derive(Debug)]
struct Core {
    nfa: NFA,
    nfarev: Option<NFA>,
    pikevm: wrappers::PikeVM,
    backtrack: wrappers::BoundedBacktracker,
    onepass: wrappers::OnePass,
    hybrid: wrappers::Hybrid,
}

impl Core {
    fn new(
        config: &meta::Config,
        hirs: &[&Hir],
    ) -> Result<Arc<dyn Strategy>, BuildError> {
        let thompson_config = thompson::Config::new()
            .nfa_size_limit(config.get_nfa_size_limit())
            .shrink(false)
            .captures(true);
        let nfa = thompson::Compiler::new()
            .configure(thompson_config.clone())
            .build_many_from_hir(hirs)
            .map_err(BuildError::nfa)?;
        let (nfarev, hybrid) = if !config.get_hybrid() {
            (None, wrappers::Hybrid::none())
        } else {
            let nfarev = thompson::Compiler::new()
                // Currently, reverse NFAs don't support capturing groups, so
                // we MUST disable them. But even if we didn't have to, we
                // would, because nothing in this crate does anything useful
                // with capturing groups in reverse. And of course, the lazy
                // DFA ignores capturing groups in all cases.
                .configure(
                    thompson_config.clone().captures(false).reverse(true),
                )
                .build_many_from_hir(hirs)
                .map_err(BuildError::nfa)?;
            let hybrid = wrappers::Hybrid::new(config, &nfa, &nfarev);
            (Some(nfarev), hybrid)
        };
        let pikevm = wrappers::PikeVM::new(config, &nfa)?;
        let backtrack = wrappers::BoundedBacktracker::new(config, &nfa)?;
        let onepass = wrappers::OnePass::new(config, &nfa);
        Ok(Arc::new(Core { nfa, nfarev, pikevm, backtrack, onepass, hybrid }))
    }
}

impl Strategy for Core {
    fn create_captures(&self) -> Captures {
        Captures::all(self.nfa.group_info().clone())
    }

    fn create_cache(&self) -> meta::Cache {
        meta::Cache {
            capmatches: self.create_captures(),
            pikevm: self.pikevm.create_cache(),
            backtrack: self.backtrack.create_cache(),
            onepass: self.onepass.create_cache(),
            hybrid: self.hybrid.create_cache(),
        }
    }

    fn reset_cache(&self, cache: &mut meta::Cache) {
        cache.pikevm.reset(&self.pikevm);
        cache.backtrack.reset(&self.backtrack);
        cache.onepass.reset(&self.onepass);
        cache.hybrid.reset(&self.hybrid);
    }

    fn try_is_match(
        &self,
        cache: &mut meta::Cache,
        input: &Input<'_, '_>,
    ) -> Result<bool, MatchError> {
        if let Some(ref e) = self.backtrack.get(input) {
            return e
                .try_slots(&mut cache.backtrack, input, &mut [])
                .map(|p| p.is_some());
        }
        self.pikevm
            .get()
            .expect("PikeVM is always available")
            .try_slots(&mut cache.pikevm, input, &mut [])
            .map(|pid| pid.is_some())
    }

    fn try_find(
        &self,
        cache: &mut meta::Cache,
        input: &Input<'_, '_>,
    ) -> Result<Option<Match>, MatchError> {
        let caps = &mut cache.capmatches;
        caps.set_pattern(None);
        let pid = if let Some(ref e) = self.backtrack.get(input) {
            e.try_slots(&mut cache.backtrack, input, caps.slots_mut())?
        } else {
            let e = self.pikevm.get().expect("PikeVM is always available");
            e.try_slots(&mut cache.pikevm, input, caps.slots_mut())?
        };
        caps.set_pattern(pid);
        Ok(caps.get_match())
    }

    fn try_slots(
        &self,
        cache: &mut meta::Cache,
        input: &Input<'_, '_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Result<Option<PatternID>, MatchError> {
        if let Some(ref e) = self.backtrack.get(input) {
            e.try_slots(&mut cache.backtrack, input, slots)
        } else {
            let e = self.pikevm.get().expect("PikeVM is always available");
            e.try_slots(&mut cache.pikevm, input, slots)
        }
    }

    fn try_which_overlapping_matches(
        &self,
        cache: &mut meta::Cache,
        input: &Input<'_, '_>,
        patset: &mut PatternSet,
    ) -> Result<(), MatchError> {
        let e = self.pikevm.get().expect("PikeVM is always available");
        e.which_overlapping_matches(&mut cache.pikevm, input, patset)
    }
}
