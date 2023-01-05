use core::{
    borrow::Borrow,
    fmt::Debug,
    panic::{RefUnwindSafe, UnwindSafe},
};

use alloc::{sync::Arc, vec};

use regex_syntax::hir::{self, literal, Hir};

use crate::{
    meta::{wrappers, BuildError, Cache, RegexInfo},
    nfa::thompson::{self, pikevm::PikeVM, NFA},
    util::{
        captures::{Captures, GroupInfo},
        prefilter::{self, Prefilter},
        primitives::{NonMaxUsize, PatternID},
        search::{
            Anchored, HalfMatch, Input, Match, MatchError, MatchKind,
            PatternSet,
        },
        syntax::Literals,
    },
};

#[cfg(feature = "dfa-onepass")]
use crate::dfa::onepass;
#[cfg(feature = "hybrid")]
use crate::hybrid;
#[cfg(feature = "nfa-backtrack")]
use crate::nfa::thompson::backtrack;

pub(crate) trait Strategy:
    Debug + Send + Sync + RefUnwindSafe + UnwindSafe + 'static
{
    fn create_captures(&self) -> Captures;

    fn create_cache(&self) -> Cache;

    fn reset_cache(&self, cache: &mut Cache);

    fn try_find(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
    ) -> Result<Option<Match>, MatchError>;

    fn try_find_earliest(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
    ) -> Result<Option<HalfMatch>, MatchError> {
        Ok(self
            .try_find(cache, input)?
            .map(|m| HalfMatch::new(m.pattern(), m.end())))
    }

    fn try_slots(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Result<Option<PatternID>, MatchError>;

    fn try_which_overlapping_matches(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        patset: &mut PatternSet,
    ) -> Result<(), MatchError>;
}

// Implement strategy for anything that implements prefilter.
//
// Note that this must only be used for regexes of length 1. Multi-regexes
// don't work here. The prefilter interface only provides the span of a match
// and not the pattern ID. (I did consider making it more expressive, but I
// couldn't figure out how to tie everything together elegantly.) Thus, so long
// as the regex only contains one pattern, we can simply assume that a match
// corresponds to PatternID::ZERO. And indeed, that's what we do here.
//
// In practice, since this impl is used to report matches directly and thus
// completely bypasses the regex engine, we only wind up using this under the
// following restrictions:
//
// * There must be only one pattern. As explained above.
// * The literal sequence must be finite and only contain exact literals.
// * There must not be any look-around assertions. If there are, the literals
// extracted might be exact, but a match doesn't necessarily imply an overall
// match. As a trivial example, 'foo\bbar' does not match 'foobar'.
// * The pattern must not have any explicit capturing groups. If it does, the
// caller might expect them to be resolved. e.g., 'foo(bar)'.
//
// So when all of those things are true, we use a prefilter directly as a
// strategy.
//
// In the case where the number of patterns is more than 1, we don't use this
// but do use a special Aho-Corasick strategy if all of the regexes are just
// simple literals or alternations of literals. (We also use the Aho-Corasick
// strategy when len(patterns)==1 if the number of literals is large. In that
// case, literal extraction gives up and will return an infinite set.)
impl<T: Prefilter> Strategy for T {
    fn create_captures(&self) -> Captures {
        // The only thing we support here is the start and end of the overall
        // match for a single pattern. In other words, exactly one implicit
        // capturing group. In theory, capturing groups should never be used
        // for this regex because the only way this impl gets used is if there
        // are no explicit capturing groups. Thus, asking to resolve capturing
        // groups is always wasteful.
        let info = GroupInfo::new(vec![vec![None::<&str>]]).unwrap();
        Captures::matches(info)
    }

    fn create_cache(&self) -> Cache {
        Cache {
            capmatches: self.create_captures(),
            pikevm: wrappers::PikeVMCache::none(),
            backtrack: wrappers::BoundedBacktrackerCache::none(),
            onepass: wrappers::OnePassCache::none(),
            hybrid: wrappers::HybridCache::none(),
        }
    }

    fn reset_cache(&self, cache: &mut Cache) {}

    fn try_find(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
    ) -> Result<Option<Match>, MatchError> {
        if input.is_done() {
            return Ok(None);
        }
        if input.get_anchored().is_anchored() {
            return Ok(self
                .prefix(input.haystack(), input.get_span())
                .map(|sp| Match::new(PatternID::ZERO, sp)));
        }
        Ok(self
            .find(input.haystack(), input.get_span())
            .map(|sp| Match::new(PatternID::ZERO, sp)))
    }

    fn try_slots(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Result<Option<PatternID>, MatchError> {
        let m = match self.try_find(cache, input)? {
            None => return Ok(None),
            Some(m) => m,
        };
        if slots.len() >= 1 {
            slots[0] = NonMaxUsize::new(m.start());
            if slots.len() >= 2 {
                slots[1] = NonMaxUsize::new(m.end());
            }
        }
        Ok(Some(m.pattern()))
    }

    fn try_which_overlapping_matches(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        patset: &mut PatternSet,
    ) -> Result<(), MatchError> {
        if self.try_find(cache, input)?.is_some() {
            patset.insert(PatternID::ZERO);
        }
        Ok(())
    }
}

pub(super) fn new(
    info: &RegexInfo,
    hirs: &[&Hir],
) -> Result<(Arc<dyn Strategy>, Option<Arc<dyn Prefilter>>), BuildError> {
    let lits = Literals::new(hirs);
    // Check to see if our prefixes are exact, which means we might be able to
    // bypass the regex engine entirely and just rely on literal searches. We
    // need to meet a few criteria that basically lets us implement the full
    // regex API. So for example, we can implement "ask for capturing groups"
    // so long as they are no capturing groups in the regex.
    //
    // We also require that we have a single regex pattern. Namely, we reuse
    // the prefilter infrastructure to implement search and prefilters only
    // report spans. Prefilters don't know about pattern IDs. The multi-regex
    // case isn't a lost cause, we might still use Aho-Corasick and we might
    // still just use a regular prefilter, but that's done below.
    //
    // If we do have only one pattern, then we also require that it has zero
    // look-around assertions. Namely, literal extraction treats look-around
    // assertions as if they match *every* empty string. But of course, that
    // isn't true. So for example, 'foo\bquux' never matches anything, but
    // 'fooquux' is extracted from that as an exact literal. Such cases should
    // just run the regex engine. 'fooquux' will be used as a normal prefilter,
    // and then the regex engine will try to look for an actual match.
    //
    // Finally, currently, our prefilters are all oriented around
    // leftmost-first match semantics, so don't try to use them if the caller
    // asked for anything else.
    //
    // This seems like a lot of requirements to meet, but it applies to a lot
    // of cases. 'foo', '[abc][123]' and 'foo|bar|quux' all meet the above
    // criteria, for example.
    //
    // Note that this is effectively a latency optimization. If we didn't
    // do this, then the extracted literals would still get bundled into a
    // prefilter, and every regex engine capable of running unanchored searches
    // supports prefilters. So this optimization merely sidesteps having to run
    // the regex engine at all to confirm the match. Thus, it decreases the
    // latency of a match.
    if lits.prefixes().is_exact()
        && hirs.len() == 1
        && info.props[0].look_set().is_empty()
        && info.props[0].captures_len() == 0
        // We require this because our prefilters currently assume
        // leftmost-first semantics.
        && info.config.get_match_kind() == MatchKind::LeftmostFirst
    {
        // OK because we know the set is exact and thus finite.
        let prefixes = lits.prefixes().literals().unwrap();
        debug!(
            "trying to bypass regex engine by creating prefilter from: {:?}",
            prefixes
        );
        if let Some(pre) = prefilter::new_as_strategy(prefixes) {
            return Ok((pre, None));
        }
        debug!("regex bypass failed because no prefilter could be extracted");
    }

    let pre = if let Some(Some(ref pre)) = info.config.pre {
        Some(Arc::clone(pre))
    } else if info.props_union.look_set_prefix().contains(hir::Look::Start) {
        None
    } else if info.config.get_auto_prefilter() {
        lits.prefixes().literals().and_then(|strings| {
            debug!("creating prefilter from {:?}", strings);
            prefilter::new(strings)
        })
    } else {
        None
    };
    let strat = Core::new(info, hirs)?;
    Ok((strat, pre))
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
        // The onepass engine can of course fail to build, but we expect it to
        // fail in many cases because it is an optimization that doesn't apply
        // to all regexes. The 'OnePass' wrapper encapsulates this failure (and
        // logs a message if it occurs).
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
        cache: &mut Cache,
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
            trace!("using OnePass for basic search at {:?}", input.get_span());
            e.try_slots(&mut cache.onepass, input, caps.slots_mut())
        } else if let Some(ref e) = self.backtrack.get(input) {
            trace!(
                "using BoundedBacktracker for basic search at {:?}",
                input.get_span()
            );
            e.try_slots(&mut cache.backtrack, input, caps.slots_mut())
        } else {
            trace!("using PikeVM for basic search at {:?}", input.get_span());
            let e = self.pikevm.get().expect("PikeVM is always available");
            e.try_slots(&mut cache.pikevm, input, caps.slots_mut())
        }?;
        caps.set_pattern(pid);
        Ok(caps.get_match())
    }

    fn try_slots_no_hybrid(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Result<Option<PatternID>, MatchError> {
        if let Some(ref e) = self.onepass.get(input) {
            trace!(
                "using OnePass for capture search at {:?}",
                input.get_span()
            );
            e.try_slots(&mut cache.onepass, input, slots)
        } else if let Some(ref e) = self.backtrack.get(input) {
            trace!(
                "using BoundedBacktracker for capture search at {:?}",
                input.get_span()
            );
            e.try_slots(&mut cache.backtrack, input, slots)
        } else {
            trace!(
                "using PikeVM for capture search at {:?}",
                input.get_span()
            );
            let e = self.pikevm.get().expect("PikeVM is always available");
            e.try_slots(&mut cache.pikevm, input, slots)
        }
    }
}

impl Strategy for Core {
    fn create_captures(&self) -> Captures {
        Captures::all(self.nfa.group_info().clone())
    }

    fn create_cache(&self) -> Cache {
        Cache {
            capmatches: self.create_captures(),
            pikevm: self.pikevm.create_cache(),
            backtrack: self.backtrack.create_cache(),
            onepass: self.onepass.create_cache(),
            hybrid: self.hybrid.create_cache(),
        }
    }

    fn reset_cache(&self, cache: &mut Cache) {
        cache.pikevm.reset(&self.pikevm);
        cache.backtrack.reset(&self.backtrack);
        cache.onepass.reset(&self.onepass);
        cache.hybrid.reset(&self.hybrid);
    }

    fn try_find(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
    ) -> Result<Option<Match>, MatchError> {
        if let Some(e) = self.hybrid.get(input) {
            trace!(
                "using lazy DFA for basic search at {:?}",
                input.get_span()
            );
            let err = match e.try_find(&mut cache.hybrid, input) {
                Ok(m) => return Ok(m),
                Err(err) => err,
            };
            if !is_err_quit_or_gaveup(&err) {
                return Err(err);
            }
            trace!("lazy DFA failed in basic search, using fallback: {}", err);
            // Fallthrough to the fallback.
        }
        self.try_find_no_hybrid(cache, input)
    }

    fn try_find_earliest(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
    ) -> Result<Option<HalfMatch>, MatchError> {
        if let Some(e) = self.hybrid.get(input) {
            trace!("using lazy DFA for 'is match' at {:?}", input.get_span());
            // This is the main difference with 'try_find'. If we only need a
            // HalfMatch, then the lazy DFA can simply do a forward scan and
            // return the end offset and skip the reverse scan since the start
            // offset has not been requested.
            let err = match e.try_find_earliest(&mut cache.hybrid, input) {
                Ok(matched) => return Ok(matched),
                Err(err) => err,
            };
            if !is_err_quit_or_gaveup(&err) {
                return Err(err);
            }
            trace!("lazy DFA failed for 'is match', using fallback: {}", err);
            // Fallthrough to the fallback.
        }
        // Only the lazy DFA returns half-matches, since the DFA requires
        // a reverse scan to find the start position. These fallback regex
        // engines can find the start and end in a single pass, so we just do
        // that and throw away the start offset.
        Ok(self
            .try_find_no_hybrid(cache, input)?
            .map(|m| HalfMatch::new(m.pattern(), m.end())))
    }

    fn try_slots(
        &self,
        cache: &mut Cache,
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
            trace!(
                "using lazy DFA for capture search at {:?}",
                input.get_span()
            );
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
                        "lazy DFA failed in capture search, using fallback: {}",
                        err,
                    );
                    // Otherwise fallthrough to the fallback below.
                }
            };
        }
        self.try_slots_no_hybrid(cache, input, slots)
    }

    fn try_which_overlapping_matches(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        patset: &mut PatternSet,
    ) -> Result<(), MatchError> {
        if let Some(e) = self.hybrid.get(input) {
            trace!(
                "using lazy DFA for overlapping search at {:?}",
                input.get_span()
            );
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
            trace!(
                "lazy DFA failed in overlapping search, using fallback: {}",
                err
            );
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
