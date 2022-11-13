use core::{
    borrow::Borrow,
    fmt::Debug,
    panic::{RefUnwindSafe, UnwindSafe},
};

use alloc::sync::Arc;

use regex_syntax::hir::{self, Hir};

use crate::{
    meta::{self, wrappers, BuildError, RegexInfo},
    nfa::thompson::{self, pikevm::PikeVM, NFA},
    util::{
        captures::Captures,
        primitives::{NonMaxUsize, PatternID},
        search::{Anchored, Input, Match, MatchError, MatchKind, PatternSet},
    },
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
    info: &RegexInfo,
    hirs: &[&Hir],
) -> Result<Arc<dyn Strategy>, BuildError> {
    Core::new(info, hirs)
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
        info: &RegexInfo,
        hirs: &[&Hir],
    ) -> Result<Arc<dyn Strategy>, BuildError> {
        let thompson_config = thompson::Config::new()
            .nfa_size_limit(info.config.get_nfa_size_limit())
            .shrink(false)
            .captures(true);
        let nfa = thompson::Compiler::new()
            .configure(thompson_config.clone())
            .build_many_from_hir(hirs)
            .map_err(BuildError::nfa)?;
        // It's possible for the PikeVM or the BB to fail to build, even though
        // at this point, we already have a full NFA in hand. They can fail
        // when a Unicode word boundary is used but where Unicode word boundary
        // support is disabled at compile time, thus making it impossible to
        // match. (Construction can also fail if the NFA was compiled without
        // captures, but we always enable that above.)
        let pikevm = wrappers::PikeVM::new(info, &nfa)?;
        let backtrack = wrappers::BoundedBacktracker::new(info, &nfa)?;
        let onepass = wrappers::OnePass::new(info, &nfa);
        // We try to encapsulate whether a particular regex engine should
        // be used within each respective wrapper, but the lazy DFA needs a
        // reverse NFA to build itself, and we really do not want to build a
        // reverse NFA if we know we aren't going to use the lazy DFA. So we do
        // a config check up front, which is in practice the only way we won't
        // try to use the lazy DFA.
        let (nfarev, hybrid) = if !info.config.get_hybrid() {
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
            let hybrid = wrappers::Hybrid::new(info, &nfa, &nfarev);
            (Some(nfarev), hybrid)
        };
        Ok(Arc::new(Core { nfa, nfarev, pikevm, backtrack, onepass, hybrid }))
    }

    fn try_find_no_hybrid(
        &self,
        cache: &mut meta::Cache,
        input: &Input<'_, '_>,
    ) -> Result<Option<Match>, MatchError> {
        let caps = &mut cache.capmatches;
        caps.set_pattern(None);
        // We manually inline 'try_slots_no_hybrid' here because we need to
        // borrow from 'cache.capmatches' in this method, but if we do, then
        // we can't pass 'cache' wholesale to to 'try_slots_no_hybrid'. It's a
        // classic example of how the borrow checker inhibits decomposition.
        // There are of course work-arounds (more types and/or interior
        // mutability), but that's more annoying than this IMO.
        let pid = if let Some(ref e) = self.onepass.get(input) {
            trace!("using OnePass for basic search");
            e.try_slots(&mut cache.onepass, input, caps.slots_mut())
        } else if let Some(ref e) = self.backtrack.get(input) {
            trace!("using BoundedBacktracker for basic search");
            e.try_slots(&mut cache.backtrack, input, caps.slots_mut())
        } else {
            trace!("using PikeVM for basic search");
            let e = self.pikevm.get().expect("PikeVM is always available");
            e.try_slots(&mut cache.pikevm, input, caps.slots_mut())
        }?;
        caps.set_pattern(pid);
        Ok(caps.get_match())
    }

    fn try_slots_no_hybrid(
        &self,
        cache: &mut meta::Cache,
        input: &Input<'_, '_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Result<Option<PatternID>, MatchError> {
        if let Some(ref e) = self.onepass.get(input) {
            trace!("using OnePass for capture search");
            e.try_slots(&mut cache.onepass, input, slots)
        } else if let Some(ref e) = self.backtrack.get(input) {
            trace!("using BoundedBacktracker for capture search");
            e.try_slots(&mut cache.backtrack, input, slots)
        } else {
            trace!("using PikeVM for capture search");
            let e = self.pikevm.get().expect("PikeVM is always available");
            e.try_slots(&mut cache.pikevm, input, slots)
        }
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
        if let Some(e) = self.hybrid.get(input) {
            trace!("using lazy DFA for 'is match'");
            let err = match e.try_is_match(&mut cache.hybrid, input) {
                Ok(matched) => return Ok(matched),
                Err(err) => err,
            };
            if !is_err_quit_or_gaveup(&err) {
                return Err(err);
            }
            trace!("lazy DFA failed for 'is match', using fallback");
            // Fallthrough to the fallback.
        }
        self.try_slots_no_hybrid(cache, input, &mut [])
            .map(|pid| pid.is_some())
    }

    fn try_find(
        &self,
        cache: &mut meta::Cache,
        input: &Input<'_, '_>,
    ) -> Result<Option<Match>, MatchError> {
        if let Some(e) = self.hybrid.get(input) {
            trace!("using lazy DFA for basic search");
            let err = match e.try_find(&mut cache.hybrid, input) {
                Ok(m) => return Ok(m),
                Err(err) => err,
            };
            if !is_err_quit_or_gaveup(&err) {
                return Err(err);
            }
            trace!("lazy DFA failed in basic search, using fallback");
            // Fallthrough to the fallback.
        }
        self.try_find_no_hybrid(cache, input)
    }

    fn try_slots(
        &self,
        cache: &mut meta::Cache,
        input: &Input<'_, '_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Result<Option<PatternID>, MatchError> {
        // Even if the regex has explicit capture groups, if the caller didn't
        // provide any explicit slots, then it doesn't make sense to try and do
        // extra work to get offsets for those slots. Ideally the caller should
        // realize this and not call this routine in the first place, but alas,
        // we try to save the caller from themselves if they do.
        if slots.len() <= self.nfa.group_info().explicit_slot_len() {
            trace!("asked for slots unnecessarily, diverting to 'find'");
            let m = match self.try_find(cache, input)? {
                None => return Ok(None),
                Some(m) => m,
            };
            let slot_start = m.pattern().as_usize() * 2;
            let slot_end = slot_start + 1;
            if slot_start < slots.len() {
                slots[slot_start] = NonMaxUsize::new(m.start());
                if slot_end < slots.len() {
                    slots[slot_end] = NonMaxUsize::new(m.end());
                }
            }
            return Ok(Some(m.pattern()));
        }
        if let Some(e) = self.hybrid.get(input) {
            trace!("using lazy DFA for capture search");
            match e.try_find(&mut cache.hybrid, input) {
                Ok(None) => return Ok(None),
                Ok(Some(m)) => {
                    // At this point, now that we've found the bounds of the
                    // match, we need to re-run something that can resolve
                    // capturing groups. But we only need to run on it on the
                    // match bounds and not the entire haystack.
                    trace!(
                        "match found at {}..{}, \
                         using another engine to find captures",
                        m.start(),
                        m.end(),
                    );
                    let input = input
                        .clone()
                        .span(m.start()..m.end())
                        .anchored(Anchored::Yes);
                    return self.try_slots_no_hybrid(cache, &input, slots);
                }
                Err(err) => {
                    if !is_err_quit_or_gaveup(&err) {
                        return Err(err);
                    }
                    trace!(
                        "lazy DFA failed in capture search, using fallback"
                    );
                    // Otherwise fallthrough to the fallback below.
                }
            };
        }
        self.try_slots_no_hybrid(cache, input, slots)
    }

    fn try_which_overlapping_matches(
        &self,
        cache: &mut meta::Cache,
        input: &Input<'_, '_>,
        patset: &mut PatternSet,
    ) -> Result<(), MatchError> {
        if let Some(e) = self.hybrid.get(input) {
            trace!("using lazy DFA for overlapping search");
            let err = match e.try_which_overlapping_matches(
                &mut cache.hybrid,
                input,
                patset,
            ) {
                Ok(m) => return Ok(m),
                Err(err) => err,
            };
            if !is_err_quit_or_gaveup(&err) {
                return Err(err);
            }
            trace!("lazy DFA failed in overlapping search, using fallback");
            // Fallthrough to the fallback.
        }
        let e = self.pikevm.get().expect("PikeVM is always available");
        e.which_overlapping_matches(&mut cache.pikevm, input, patset)
    }
}

/// Returns true only when the given error corresponds to a search that failed
/// quit because it saw a specific byte, or gave up because it thought itself
/// to be too slow.
///
/// This is useful for checking whether an error returned by the lazy DFA
/// should be bubbled up or if it should result in running another regex
/// engine. Errors like "invalid pattern ID" should get bubbled up, while
/// quitting or giving up should result in trying a different engine.
fn is_err_quit_or_gaveup(err: &MatchError) -> bool {
    use crate::MatchErrorKind::*;
    matches!(*err.kind(), Quit { .. } | GaveUp { .. })
}
