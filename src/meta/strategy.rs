use core::{
    borrow::Borrow,
    fmt::Debug,
    panic::{RefUnwindSafe, UnwindSafe},
};

use alloc::sync::Arc;

use regex_syntax::hir::{self, Hir};

use crate::{
    meta::{self, BuildError},
    nfa::thompson::{self, pikevm::PikeVM, NFA},
    util::{
        captures::Captures,
        primitives::{NonMaxUsize, PatternID},
        search::{Input, Match, MatchError, PatternSet},
    },
};

#[cfg(feature = "dfa-onepass")]
use crate::dfa::onepass;
#[cfg(feature = "hybrid")]
use crate::hybrid;
#[cfg(feature = "nfa-backtrack")]
use crate::nfa::thompson::backtrack::BoundedBacktracker;

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
    pikevm: PikeVM,
    #[cfg(feature = "nfa-backtrack")]
    backtrack: Option<BoundedBacktracker>,
    #[cfg(feature = "dfa-onepass")]
    onepass: Option<onepass::DFA>,
    #[cfg(feature = "hybrid")]
    hybrid: Option<hybrid::regex::Regex>,
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
            .configure(thompson_config)
            .build_many_from_hir(hirs)
            .map_err(meta::BuildError::nfa)?;
        let pikevm = PikeVM::builder()
            .configure(
                PikeVM::config()
                    .match_kind(config.get_match_kind())
                    .utf8(config.get_utf8()),
            )
            .build_from_nfa(nfa)
            .map_err(meta::BuildError::nfa)?;
        Ok(Arc::new(Core {
            pikevm,
            #[cfg(feature = "nfa-backtrack")]
            backtrack: None,
            #[cfg(feature = "dfa-onepass")]
            onepass: None,
            #[cfg(feature = "hybrid")]
            hybrid: None,
        }))
    }
}

impl Strategy for Core {
    fn create_captures(&self) -> Captures {
        self.pikevm.create_captures()
    }

    fn create_cache(&self) -> meta::Cache {
        meta::Cache {
            capmatches: self.create_captures(),
            pikevm: Some(self.pikevm.create_cache()),
            #[cfg(feature = "nfa-backtrack")]
            backtrack: None,
            #[cfg(feature = "dfa-onepass")]
            onepass: None,
            #[cfg(feature = "hybrid")]
            hybrid: None,
        }
    }

    fn reset_cache(&self, cache: &mut meta::Cache) {
        cache.pikevm.as_mut().unwrap().reset(&self.pikevm);
    }

    fn try_is_match(
        &self,
        cache: &mut meta::Cache,
        input: &Input<'_, '_>,
    ) -> Result<bool, MatchError> {
        let cache = cache.pikevm.as_mut().unwrap();
        self.pikevm
            .try_search_slots(cache, input, &mut [])
            .map(|pid| pid.is_some())
    }

    fn try_find(
        &self,
        cache: &mut meta::Cache,
        input: &Input<'_, '_>,
    ) -> Result<Option<Match>, MatchError> {
        let caps = &mut cache.capmatches;
        let cache = cache.pikevm.as_mut().unwrap();
        caps.set_pattern(None);
        let pid =
            self.pikevm.try_search_slots(cache, input, caps.slots_mut())?;
        caps.set_pattern(pid);
        Ok(caps.get_match())
    }

    fn try_slots(
        &self,
        cache: &mut meta::Cache,
        input: &Input<'_, '_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Result<Option<PatternID>, MatchError> {
        let cache = cache.pikevm.as_mut().unwrap();
        self.pikevm.try_search_slots(cache, input, slots)
    }

    fn try_which_overlapping_matches(
        &self,
        cache: &mut meta::Cache,
        input: &Input<'_, '_>,
        patset: &mut PatternSet,
    ) -> Result<(), MatchError> {
        let cache = cache.pikevm.as_mut().unwrap();
        self.pikevm.which_overlapping_matches(cache, input, patset)
    }
}
