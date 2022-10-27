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
    Basic::new(config, hirs)
}

#[derive(Debug)]
struct Basic {
    pikevm: PikeVM,
}

impl Basic {
    fn new(
        config: &meta::Config,
        hirs: &[&Hir],
    ) -> Result<Arc<dyn Strategy>, BuildError> {
        let nfa = thompson::Compiler::new()
            .build_many_from_hir(hirs)
            .map_err(meta::BuildError::nfa)?;
        let pikevm = PikeVM::builder()
            .configure(PikeVM::config().match_kind(config.get_match_kind()))
            .build_from_nfa(nfa)
            .map_err(meta::BuildError::nfa)?;
        Ok(Arc::new(Basic { pikevm }))
    }
}

impl Strategy for Basic {
    fn create_captures(&self) -> Captures {
        self.pikevm.create_captures()
    }

    fn create_cache(&self) -> meta::Cache {
        meta::Cache {
            capmatches: self.create_captures(),
            pikevm: Some(self.pikevm.create_cache()),
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
