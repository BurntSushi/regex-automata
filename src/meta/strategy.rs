use core::{
    borrow::Borrow,
    fmt::Debug,
    panic::{RefUnwindSafe, UnwindSafe},
};

use alloc::{sync::Arc, vec, vec::Vec};

use regex_syntax::hir::{self, literal, Hir};

use crate::{
    meta::{
        error::{BuildError, RetryError, RetryFailError, RetryQuadraticError},
        regex::{Cache, RegexInfo},
        reverse_inner, wrappers,
    },
    nfa::thompson::{self, pikevm::PikeVM, NFA},
    util::{
        captures::{Captures, GroupInfo},
        look::LookMatcher,
        prefilter::{self, Prefilter, PrefilterI},
        primitives::{NonMaxUsize, PatternID},
        search::{
            Anchored, HalfMatch, Input, Match, MatchError, MatchKind,
            PatternSet, Span,
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

    fn search(&self, cache: &mut Cache, input: &Input<'_>) -> Option<Match>;

    fn search_half(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
    ) -> Option<HalfMatch>;

    fn search_slots(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Option<PatternID>;

    fn which_overlapping_matches(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
        patset: &mut PatternSet,
    );
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
impl<T: PrefilterI> Strategy for T {
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
            revhybrid: wrappers::ReverseHybridCache::none(),
        }
    }

    fn reset_cache(&self, cache: &mut Cache) {}

    fn search(&self, cache: &mut Cache, input: &Input<'_>) -> Option<Match> {
        if input.is_done() {
            return None;
        }
        if input.get_anchored().is_anchored() {
            return self
                .prefix(input.haystack(), input.get_span())
                .map(|sp| Match::new(PatternID::ZERO, sp));
        }
        self.find(input.haystack(), input.get_span())
            .map(|sp| Match::new(PatternID::ZERO, sp))
    }

    fn search_half(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
    ) -> Option<HalfMatch> {
        self.search(cache, input).map(|m| HalfMatch::new(m.pattern(), m.end()))
    }

    fn search_slots(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Option<PatternID> {
        let m = self.search(cache, input)?;
        if let Some(slot) = slots.get_mut(0) {
            *slot = NonMaxUsize::new(m.start());
        }
        if let Some(slot) = slots.get_mut(1) {
            *slot = NonMaxUsize::new(m.end());
        }
        Some(m.pattern())
    }

    fn which_overlapping_matches(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
        patset: &mut PatternSet,
    ) {
        if self.search(cache, input).is_some() {
            patset.insert(PatternID::ZERO);
        }
    }
}

pub(super) fn new(
    info: &RegexInfo,
    hirs: &[&Hir],
) -> Result<Arc<dyn Strategy>, BuildError> {
    let kind = info.config().get_match_kind();
    let lits = Literals::new(kind, hirs);
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
        && info.props()[0].look_set().is_empty()
        && info.props()[0].captures_len() == 0
        // We require this because our prefilters can't currently handle
        // assuming the responsibility of being the regex engine in all
        // cases. For example, when running a leftmost search with 'All'
        // match semantics for the regex 'foo|foobar', the prefilter will
        // currently report 'foo' as a match against 'foobar'. 'foo' is a
        // correct candidate, but it is not the correct leftmost match in this
        // circumstance, since the 'all' semantic demands that the search
        // continue until a dead state is reached.
        && info.config().get_match_kind() == MatchKind::LeftmostFirst
    {
        // OK because we know the set is exact and thus finite.
        let prefixes = lits.prefixes().literals().unwrap();
        debug!(
            "trying to bypass regex engine by creating \
             prefilter from {} literals: {:?}",
            prefixes.len(),
            prefixes,
        );
        if let Some(pre) = prefilter::new_as_strategy(kind, prefixes) {
            return Ok(pre);
        }
        debug!("regex bypass failed because no prefilter could be built");
    }
    // This now attempts another short-circuit of the regex engine: if we
    // have a huge alternation of just plain literals, then we can just use
    // Aho-Corasick for that and avoid the regex engine entirely.
    #[cfg(feature = "perf-literal-multisubstring")]
    if let Some(ac) = alternation_literals_to_aho_corasick(info, hirs) {
        debug!(
            "found plain alternation of literals, \
             avoiding regex engine entirely and using Aho-Corasick"
        );
        return Ok(ac);
    }

    // At this point, we're committed to a regex engine of some kind. So pull
    // out a prefilter if we can, which will feed to each of the constituent
    // regex engines.
    let pre = if info.is_always_anchored_start() {
        // TODO: I'm not sure we necessarily want to do this... We may want to
        // run a prefilter for quickly rejecting in some cases. This might mean
        // having a notion of whether a prefilter is "fast"? Or maybe it just
        // depends on haystack length? Or both?
        debug!("discarding prefixes (if any) since regex is anchored");
        None
    } else if let Some(pre) = info.config().get_prefilter() {
        debug!(
            "discarding extracted prefixes (if any) \
             since the caller provided a prefilter"
        );
        Some(pre.clone())
    } else if info.config().get_auto_prefilter() {
        lits.prefixes().literals().and_then(|strings| {
            debug!(
                "creating prefilter from {} literals: {:?}",
                strings.len(),
                strings,
            );
            Prefilter::new(kind, strings)
        })
    } else {
        debug!("discarding prefixes (if any) since prefilters were disabled");
        None
    };
    let presuf = if !info.config().get_auto_prefilter() {
        None
    } else {
        match lits.suffixes().longest_common_suffix() {
            None => None,
            Some(suffix) => {
                debug!(
                    "creating suffix prefilter from {} literals: {:?}",
                    1,
                    literal::Seq::new(&[suffix]),
                );
                Prefilter::new(kind, &[suffix])
            }
        }
    };
    let mut core = Core::new(info.clone(), pre.clone(), hirs)?;
    // Now that we have our core regex engines built, there are a few cases
    // where we can do a little bit better than just a normal "search forward
    // and maybe use a prefilter when in a start state." However, these cases
    // may not always work or otherwise build on top of the Core searcher.
    // For example, the reverse anchored optimization seems like it might
    // always work, but only the DFAs support reverse searching and the DFAs
    // might give up or quit for reasons. If we had, e.g., a PikeVM that
    // supported reverse searching, then we could avoid building a full Core
    // engine for this case.
    core = match ReverseAnchored::new(core) {
        Err(core) => core,
        Ok(ra) => {
            debug!("using reverse anchored strategy");
            return Ok(Arc::new(ra));
        }
    };
    core = match ReverseSuffix::new(core, presuf.clone()) {
        Err(core) => core,
        Ok(rs) => {
            debug!("using reverse suffix strategy");
            return Ok(Arc::new(rs));
        }
    };
    core = match ReverseInner::new(hirs, core) {
        Err(core) => core,
        Ok(ri) => {
            debug!("using reverse inner strategy");
            return Ok(Arc::new(ri));
        }
    };
    debug!("using core strategy");
    Ok(Arc::new(core))
}

#[derive(Debug)]
struct Core {
    info: RegexInfo,
    pre: Option<Prefilter>,
    nfa: NFA,
    nfarev: Option<NFA>,
    pikevm: wrappers::PikeVM,
    backtrack: wrappers::BoundedBacktracker,
    onepass: wrappers::OnePass,
    hybrid: wrappers::Hybrid,
    dfa: wrappers::DFA,
}

impl Core {
    fn new(
        info: RegexInfo,
        pre: Option<Prefilter>,
        hirs: &[&Hir],
    ) -> Result<Core, BuildError> {
        let mut lookm = LookMatcher::new();
        lookm.set_line_terminator(info.config().get_line_terminator());
        let thompson_config = thompson::Config::new()
            .utf8(info.config().get_utf8())
            .nfa_size_limit(info.config().get_nfa_size_limit())
            .shrink(false)
            .captures(true)
            .look_matcher(lookm);
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
        let pikevm = wrappers::PikeVM::new(&info, pre.clone(), &nfa)?;
        let backtrack =
            wrappers::BoundedBacktracker::new(&info, pre.clone(), &nfa)?;
        // The onepass engine can of course fail to build, but we expect it to
        // fail in many cases because it is an optimization that doesn't apply
        // to all regexes. The 'OnePass' wrapper encapsulates this failure (and
        // logs a message if it occurs).
        let onepass = wrappers::OnePass::new(&info, &nfa);
        // We try to encapsulate whether a particular regex engine should be
        // used within each respective wrapper, but the DFAs need a reverse NFA
        // to build itself, and we really do not want to build a reverse NFA if
        // we know we aren't going to use the lazy DFA. So we do a config check
        // up front, which is in practice the only way we won't try to use the
        // DFA.
        let (nfarev, hybrid, dfa) =
            if !info.config().get_hybrid() && !info.config().get_dfa() {
                (None, wrappers::Hybrid::none(), wrappers::DFA::none())
            } else {
                // FIXME: Technically, we don't quite yet KNOW that we need
                // a reverse NFA. It's possible for the DFAs below to both
                // fail to build just based on the forward NFA. In which case,
                // building the reverse NFA was totally wasted work. But...
                // fixing this requires breaking DFA construction apart into
                // two pieces: one for the forward part and another for the
                // reverse part. Quite annoying. Making it worse, when building
                // both DFAs fails, it's quite likely that the NFA is large and
                // that it will take quite some time to build the reverse NFA
                // too. So... it's really probably worth it to do this!
                let nfarev = thompson::Compiler::new()
                    // Currently, reverse NFAs don't support capturing groups,
                    // so we MUST disable them. But even if we didn't have to,
                    // we would, because nothing in this crate does anything
                    // useful with capturing groups in reverse. And of course,
                    // the lazy DFA ignores capturing groups in all cases.
                    .configure(
                        thompson_config.clone().captures(false).reverse(true),
                    )
                    .build_many_from_hir(hirs)
                    .map_err(BuildError::nfa)?;
                let dfa = if !info.config().get_dfa() {
                    wrappers::DFA::none()
                } else {
                    wrappers::DFA::new(&info, pre.clone(), &nfa, &nfarev)
                };
                let hybrid = if !info.config().get_hybrid() {
                    wrappers::Hybrid::none()
                } else if dfa.is_some() {
                    debug!("skipping lazy DFA because we have a full DFA");
                    wrappers::Hybrid::none()
                } else {
                    wrappers::Hybrid::new(&info, pre.clone(), &nfa, &nfarev)
                };
                (Some(nfarev), hybrid, dfa)
            };
        Ok(Core {
            info,
            pre,
            nfa,
            nfarev,
            pikevm,
            backtrack,
            onepass,
            hybrid,
            dfa,
        })
    }

    #[inline(always)]
    fn try_search_mayfail(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
    ) -> Option<Result<Option<Match>, RetryFailError>> {
        if let Some(e) = self.dfa.get(input) {
            trace!("using full DFA for search at {:?}", input.get_span());
            Some(e.try_search(input))
        } else if let Some(e) = self.hybrid.get(input) {
            trace!("using lazy DFA for search at {:?}", input.get_span());
            Some(e.try_search(&mut cache.hybrid, input))
        } else {
            None
        }
    }

    fn search_nofail(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
    ) -> Option<Match> {
        let caps = &mut cache.capmatches;
        caps.set_pattern(None);
        // We manually inline 'try_search_slots_nofail' here because we need to
        // borrow from 'cache.capmatches' in this method, but if we do, then
        // we can't pass 'cache' wholesale to to 'try_slots_no_hybrid'. It's a
        // classic example of how the borrow checker inhibits decomposition.
        // There are of course work-arounds (more types and/or interior
        // mutability), but that's more annoying than this IMO.
        let pid = if let Some(ref e) = self.onepass.get(input) {
            trace!("using OnePass for search at {:?}", input.get_span());
            e.search_slots(&mut cache.onepass, input, caps.slots_mut())
        } else if let Some(ref e) = self.backtrack.get(input) {
            trace!(
                "using BoundedBacktracker for search at {:?}",
                input.get_span()
            );
            e.search_slots(&mut cache.backtrack, input, caps.slots_mut())
        } else {
            trace!("using PikeVM for search at {:?}", input.get_span());
            let e = self.pikevm.get().expect("PikeVM is always available");
            e.search_slots(&mut cache.pikevm, input, caps.slots_mut())
        };
        caps.set_pattern(pid);
        caps.get_match()
    }

    fn search_half_nofail(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
    ) -> Option<HalfMatch> {
        // Only the lazy/full DFA returns half-matches, since the DFA requires
        // a reverse scan to find the start position. These fallback regex
        // engines can find the start and end in a single pass, so we just do
        // that and throw away the start offset to conform to the API.
        let m = self.search_nofail(cache, input)?;
        Some(HalfMatch::new(m.pattern(), m.end()))
    }

    fn search_slots_nofail(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Option<PatternID> {
        if let Some(ref e) = self.onepass.get(input) {
            trace!(
                "using OnePass for capture search at {:?}",
                input.get_span()
            );
            e.search_slots(&mut cache.onepass, input, slots)
        } else if let Some(ref e) = self.backtrack.get(input) {
            trace!(
                "using BoundedBacktracker for capture search at {:?}",
                input.get_span()
            );
            e.search_slots(&mut cache.backtrack, input, slots)
        } else {
            trace!(
                "using PikeVM for capture search at {:?}",
                input.get_span()
            );
            let e = self.pikevm.get().expect("PikeVM is always available");
            e.search_slots(&mut cache.pikevm, input, slots)
        }
    }

    fn is_capture_search_needed(&self, slots_len: usize) -> bool {
        slots_len > self.nfa.group_info().implicit_slot_len()
    }
}

impl Strategy for Core {
    #[inline(always)]
    fn create_captures(&self) -> Captures {
        Captures::all(self.nfa.group_info().clone())
    }

    #[inline(always)]
    fn create_cache(&self) -> Cache {
        Cache {
            capmatches: self.create_captures(),
            pikevm: self.pikevm.create_cache(),
            backtrack: self.backtrack.create_cache(),
            onepass: self.onepass.create_cache(),
            hybrid: self.hybrid.create_cache(),
            revhybrid: wrappers::ReverseHybridCache::none(),
        }
    }

    #[inline(always)]
    fn reset_cache(&self, cache: &mut Cache) {
        cache.pikevm.reset(&self.pikevm);
        cache.backtrack.reset(&self.backtrack);
        cache.onepass.reset(&self.onepass);
        cache.hybrid.reset(&self.hybrid);
    }

    #[inline(always)]
    fn search(&self, cache: &mut Cache, input: &Input<'_>) -> Option<Match> {
        // We manually inline try_search_mayfail here because letting the
        // compiler do it seems to produce pretty crappy codegen.
        return if let Some(e) = self.dfa.get(input) {
            trace!("using full DFA for search at {:?}", input.get_span());
            match e.try_search(input) {
                Ok(x) => x,
                Err(err) => {
                    trace!("full DFA search failed: {}", err);
                    self.search_nofail(cache, input)
                }
            }
        } else if let Some(e) = self.hybrid.get(input) {
            trace!("using lazy DFA for search at {:?}", input.get_span());
            match e.try_search(&mut cache.hybrid, input) {
                Ok(x) => x,
                Err(err) => {
                    trace!("lazy DFA search failed: {}", err);
                    self.search_nofail(cache, input)
                }
            }
        } else {
            self.search_nofail(cache, input)
        };
    }

    #[inline(always)]
    fn search_half(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
    ) -> Option<HalfMatch> {
        // The main difference with 'search' is that if we're using a DFA, we
        // can use a single forward scan without needing to run the reverse
        // DFA.
        return if let Some(e) = self.dfa.get(input) {
            trace!("using full DFA for search at {:?}", input.get_span());
            match e.try_search_half_fwd(input) {
                Ok(x) => x,
                Err(err) => {
                    trace!("full DFA half search failed: {}", err);
                    self.search_half_nofail(cache, input)
                }
            }
        } else if let Some(e) = self.hybrid.get(input) {
            trace!("using lazy DFA for search at {:?}", input.get_span());
            match e.try_search_half_fwd(&mut cache.hybrid, input) {
                Ok(x) => x,
                Err(err) => {
                    trace!("lazy DFA half search failed: {}", err);
                    self.search_half_nofail(cache, input)
                }
            }
        } else {
            self.search_half_nofail(cache, input)
        };
    }

    #[inline(always)]
    fn search_slots(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Option<PatternID> {
        // Even if the regex has explicit capture groups, if the caller didn't
        // provide any explicit slots, then it doesn't make sense to try and do
        // extra work to get offsets for those slots. Ideally the caller should
        // realize this and not call this routine in the first place, but alas,
        // we try to save the caller from themselves if they do.
        if !self.is_capture_search_needed(slots.len()) {
            trace!("asked for slots unnecessarily, trying fast path");
            let m = self.search(cache, input)?;
            copy_match_to_slots(m, slots);
            return Some(m.pattern());
        }
        // If the onepass DFA is available for this search (which only happens
        // when it's anchored), then skip running a fallible DFA. The onepass
        // DFA isn't as fast as a full or lazy DFA, but it is typically quite
        // a bit faster than the backtracker or the PikeVM. So it isn't as
        // advantageous to try and do a full/lazy DFA scan first.
        //
        // We still theorize that it's better to do a full/lazy DFA scan, even
        // when it's anchored, because it's usually much faster and permits us
        // to say "no match" much more quickly. This does hurt the case of,
        // say, parsing each line in a log file into capture groups, because
        // in that case, the line always matches. So the lazy DFA scan is
        // usually just wasted work. But, the lazy DFA is usually quite fast
        // and doesn't cost too much here.
        if self.onepass.get(&input).is_some() {
            return self.search_slots_nofail(cache, &input, slots);
        }
        let m = match self.try_search_mayfail(cache, input) {
            Some(Ok(Some(m))) => m,
            Some(Ok(None)) => return None,
            Some(Err(err)) => {
                trace!("fast capture search failed: {}", err);
                return self.search_slots_nofail(cache, input, slots);
            }
            None => {
                return self.search_slots_nofail(cache, input, slots);
            }
        };
        // At this point, now that we've found the bounds of the
        // match, we need to re-run something that can resolve
        // capturing groups. But we only need to run on it on the
        // match bounds and not the entire haystack.
        trace!(
            "match found at {}..{} in capture search, \
		  	 using another engine to find captures",
            m.start(),
            m.end(),
        );
        let input = input
            .clone()
            .span(m.start()..m.end())
            .anchored(Anchored::Pattern(m.pattern()));
        Some(
            self.search_slots_nofail(cache, &input, slots)
                .expect("should find a match"),
        )
    }

    #[inline(always)]
    fn which_overlapping_matches(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
        patset: &mut PatternSet,
    ) {
        if let Some(e) = self.dfa.get(input) {
            trace!(
                "using full DFA for overlapping search at {:?}",
                input.get_span()
            );
            let err = match e.try_which_overlapping_matches(input, patset) {
                Ok(()) => return,
                Err(err) => err,
            };
            trace!("fast overlapping search failed: {}", err);
        } else if let Some(e) = self.hybrid.get(input) {
            trace!(
                "using lazy DFA for overlapping search at {:?}",
                input.get_span()
            );
            let err = match e.try_which_overlapping_matches(
                &mut cache.hybrid,
                input,
                patset,
            ) {
                Ok(()) => return,
                Err(err) => err,
            };
            trace!("fast overlapping search failed: {}", err);
        }
        trace!(
            "using PikeVM for overlapping search at {:?}",
            input.get_span()
        );
        let e = self.pikevm.get().expect("PikeVM is always available");
        e.which_overlapping_matches(&mut cache.pikevm, input, patset)
    }
}

#[derive(Debug)]
struct ReverseAnchored {
    core: Core,
}

impl ReverseAnchored {
    fn new(core: Core) -> Result<ReverseAnchored, Core> {
        if !core.info.is_always_anchored_end() {
            debug!(
                "skipping reverse anchored optimization because \
				 the regex is not always anchored at the end"
            );
            return Err(core);
        }
        if core.info.is_always_anchored_start() {
            debug!(
                "skipping reverse anchored optimization because \
				 the regex is also anchored at the start"
            );
            return Err(core);
        }
        // Only DFAs can do reverse searches (currently), so we need one of
        // them in order to do this optimization. It's possible (although
        // pretty unlikely) that we have neither and need to give up.
        if !core.hybrid.is_some() && !core.dfa.is_some() {
            debug!(
                "skipping reverse anchored optimization because \
				 we don't have a lazy DFA or a full DFA"
            );
            return Err(core);
        }
        Ok(ReverseAnchored { core })
    }

    #[inline(always)]
    fn try_search_half_anchored_rev(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
    ) -> Result<Option<HalfMatch>, RetryFailError> {
        // We of course always want an anchored search. In theory, the
        // underlying regex engines should automatically enable anchored
        // searches since the regex is itself anchored, but this more clearly
        // expresses intent and is always correct.
        let input = input.clone().anchored(Anchored::Yes);
        if let Some(e) = self.core.dfa.get(&input) {
            trace!(
                "using full DFA for reverse anchored search at {:?}",
                input.get_span()
            );
            e.try_search_half_rev(&input)
        } else if let Some(e) = self.core.hybrid.get(&input) {
            trace!(
                "using lazy DFA for reverse anchored search at {:?}",
                input.get_span()
            );
            e.try_search_half_rev(&mut cache.hybrid, &input)
        } else {
            unreachable!("ReverseAnchored always has a DFA")
        }
    }
}

// Note that in this impl, we don't check that 'input.end() ==
// input.haystack().len()'. In particular, when that condition is false, a
// match is always impossible because we know that the regex is always anchored
// at the end (or else 'ReverseAnchored' won't be built). We don't check that
// here because the 'Regex' wrapper actually does that for us in all cases.
// Thus, in this impl, we can actually assume that the end position in 'input'
// is equivalent to the length of the haystack.
impl Strategy for ReverseAnchored {
    #[inline(always)]
    fn create_captures(&self) -> Captures {
        self.core.create_captures()
    }

    #[inline(always)]
    fn create_cache(&self) -> Cache {
        self.core.create_cache()
    }

    #[inline(always)]
    fn reset_cache(&self, cache: &mut Cache) {
        self.core.reset_cache(cache);
    }

    #[inline(always)]
    fn search(&self, cache: &mut Cache, input: &Input<'_>) -> Option<Match> {
        match self.try_search_half_anchored_rev(cache, input) {
            Err(err) => {
                trace!("fast reverse anchored search failed: {}", err);
                self.core.search_nofail(cache, input)
            }
            Ok(None) => None,
            Ok(Some(hm)) => {
                Some(Match::new(hm.pattern(), hm.offset()..input.end()))
            }
        }
    }

    #[inline(always)]
    fn search_half(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
    ) -> Option<HalfMatch> {
        match self.try_search_half_anchored_rev(cache, input) {
            Err(err) => {
                trace!("fast reverse anchored search failed: {}", err);
                self.core.search_half_nofail(cache, input)
            }
            Ok(None) => None,
            Ok(Some(hm)) => {
                // Careful here! 'try_search_half' is a *forward* search that
                // only cares about the *end* position of a match. But
                // 'hm.offset()' is actually the start of the match. So we
                // actually just throw that away here and, since we know we
                // have a match, return the only possible position at which a
                // match can occur: input.end().
                Some(HalfMatch::new(hm.pattern(), input.end()))
            }
        }
    }

    #[inline(always)]
    fn search_slots(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Option<PatternID> {
        match self.try_search_half_anchored_rev(cache, input) {
            Err(err) => {
                trace!("fast reverse anchored search failed: {}", err);
                self.core.search_slots_nofail(cache, input, slots)
            }
            Ok(None) => None,
            Ok(Some(hm)) => {
                if !self.core.is_capture_search_needed(slots.len()) {
                    trace!("asked for slots unnecessarily, skipping captures");
                    let m = Match::new(hm.pattern(), hm.offset()..input.end());
                    copy_match_to_slots(m, slots);
                    return Some(m.pattern());
                }
                let start = hm.offset();
                let input = input
                    .clone()
                    .span(start..input.end())
                    .anchored(Anchored::Pattern(hm.pattern()));
                self.core.search_slots_nofail(cache, &input, slots)
            }
        }
    }

    #[inline(always)]
    fn which_overlapping_matches(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
        patset: &mut PatternSet,
    ) {
        // It seems like this could probably benefit from a reverse anchored
        // optimization, perhaps by doing an overlapping reverse search (which
        // the DFAs do support). I haven't given it much thought though, and
        // I'm currently focus more on the single pattern case.
        self.core.which_overlapping_matches(cache, input, patset)
    }
}

#[derive(Debug)]
struct ReverseSuffix {
    core: Core,
    presuf: Prefilter,
}

impl ReverseSuffix {
    fn new(
        core: Core,
        presuf: Option<Prefilter>,
    ) -> Result<ReverseSuffix, Core> {
        // Like the reverse inner optimization, we don't do this for regexes
        // that are always anchored. It could lead to scanning too much, but
        // could say "no match" much more quickly than running the regex
        // engine if the initial literal scan doesn't match. With that said,
        // the reverse suffix optimization has lower overhead, since it only
        // requires a reverse scan after a literal match to confirm or reject
        // the match. (Although, in the case of confirmation, it then needs to
        // do another forward scan to find the end position.)
        if core.info.is_always_anchored_start() {
            debug!(
                "skipping reverse suffix optimization because \
				 the regex is always anchored at the start",
            );
            return Err(core);
        }
        // Only DFAs can do reverse searches (currently), so we need one of
        // them in order to do this optimization. It's possible (although
        // pretty unlikely) that we have neither and need to give up.
        if !core.hybrid.is_some() && !core.dfa.is_some() {
            debug!(
                "skipping reverse suffix optimization because \
				 we don't have a lazy DFA or a full DFA"
            );
            return Err(core);
        }
        if core.pre.as_ref().map_or(false, |p| p.is_fast()) {
            debug!(
                "skipping reverse suffix optimization because \
				 we already have a prefilter that we think is fast"
            );
            return Err(core);
        }
        let presuf = match presuf {
            Some(presuf) => presuf,
            None => {
                debug!(
                    "skipping reverse suffix optimization because \
					 we don't have a suffix prefilter"
                );
                return Err(core);
            }
        };
        if !presuf.is_fast() {
            debug!(
                "skipping reverse suffix optimization because \
				 while we have a suffix prefilter, it is not \
				 believed to be 'fast'"
            );
        }
        Ok(ReverseSuffix { core, presuf })
    }

    #[inline(always)]
    fn try_search_half_start(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
    ) -> Result<Option<(HalfMatch, Span)>, RetryError> {
        let mut span = input.get_span();
        let mut min_start = 0;
        loop {
            let litmatch = match self.presuf.find(input.haystack(), span) {
                None => return Ok(None),
                Some(span) => span,
            };
            trace!("reverse suffix scan found suffix match at {:?}", litmatch);
            let revinput = input
                .clone()
                .anchored(Anchored::Yes)
                .span(input.start()..litmatch.end);
            match self
                .try_search_half_rev_limited(cache, &revinput, min_start)?
            {
                None => {
                    if span.start >= span.end {
                        break;
                    }
                    span.start = litmatch.start.checked_add(1).unwrap();
                }
                Some(hm) => return Ok(Some((hm, litmatch))),
            }
            min_start = litmatch.end;
        }
        Ok(None)
    }

    #[inline(always)]
    fn try_search_half_fwd(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
    ) -> Result<Option<HalfMatch>, RetryFailError> {
        if let Some(e) = self.core.dfa.get(&input) {
            trace!(
                "using full DFA for forward reverse suffix search at {:?}",
                input.get_span()
            );
            e.try_search_half_fwd(&input)
        } else if let Some(e) = self.core.hybrid.get(&input) {
            trace!(
                "using lazy DFA for forward reverse suffix search at {:?}",
                input.get_span()
            );
            e.try_search_half_fwd(&mut cache.hybrid, &input)
        } else {
            unreachable!("ReverseSuffix always has a DFA")
        }
    }

    #[inline(always)]
    fn try_search_half_rev_limited(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
        min_start: usize,
    ) -> Result<Option<HalfMatch>, RetryError> {
        if let Some(e) = self.core.dfa.get(&input) {
            trace!(
                "using full DFA for reverse suffix search at {:?}, \
                 but will be stopped at {} to avoid quadratic behavior",
                input.get_span(),
                min_start,
            );
            e.try_search_half_rev_limited(&input, min_start)
        } else if let Some(e) = self.core.hybrid.get(&input) {
            trace!(
                "using lazy DFA for reverse inner search at {:?}, \
                 but will be stopped at {} to avoid quadratic behavior",
                input.get_span(),
                min_start,
            );
            e.try_search_half_rev_limited(&mut cache.hybrid, &input, min_start)
        } else {
            unreachable!("ReverseSuffix always has a DFA")
        }
    }
}

impl Strategy for ReverseSuffix {
    #[inline(always)]
    fn create_captures(&self) -> Captures {
        self.core.create_captures()
    }

    #[inline(always)]
    fn create_cache(&self) -> Cache {
        self.core.create_cache()
    }

    #[inline(always)]
    fn reset_cache(&self, cache: &mut Cache) {
        self.core.reset_cache(cache);
    }

    #[inline(always)]
    fn search(&self, cache: &mut Cache, input: &Input<'_>) -> Option<Match> {
        match self.try_search_half_start(cache, input) {
            Err(RetryError::Quadratic(err)) => {
                trace!("reverse suffix optimization failed: {}", err);
                self.core.search(cache, input)
            }
            Err(RetryError::Fail(err)) => {
                trace!("reverse suffix reverse fast search failed: {}", err);
                self.core.search_nofail(cache, input)
            }
            Ok(None) => None,
            Ok(Some((hm_start, _))) => {
                let fwdinput = input
                    .clone()
                    .anchored(Anchored::Pattern(hm_start.pattern()))
                    .span(hm_start.offset()..input.end());
                match self.try_search_half_fwd(cache, &fwdinput) {
                    Err(err) => {
                        trace!(
                            "reverse suffix forward fast search failed: {}",
                            err
                        );
                        self.core.search_nofail(cache, input)
                    }
                    Ok(None) => {
                        unreachable!(
                            "suffix match plus reverse match implies \
						     there must be a match",
                        )
                    }
                    Ok(Some(hm_end)) => Some(Match::new(
                        hm_start.pattern(),
                        hm_start.offset()..hm_end.offset(),
                    )),
                }
            }
        }
    }

    #[inline(always)]
    fn search_half(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
    ) -> Option<HalfMatch> {
        match self.try_search_half_start(cache, input) {
            Err(RetryError::Quadratic(err)) => {
                trace!("reverse suffix half optimization failed: {}", err);
                self.core.search_half(cache, input)
            }
            Err(RetryError::Fail(err)) => {
                trace!(
                    "reverse suffix reverse fast half search failed: {}",
                    err
                );
                self.core.search_half_nofail(cache, input)
            }
            Ok(None) => None,
            Ok(Some((hm_start, pre_span))) => {
                Some(HalfMatch::new(hm_start.pattern(), pre_span.end))
            }
        }
    }

    #[inline(always)]
    fn search_slots(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Option<PatternID> {
        if !self.core.is_capture_search_needed(slots.len()) {
            trace!("asked for slots unnecessarily, trying fast path");
            let m = self.search(cache, input)?;
            copy_match_to_slots(m, slots);
            return Some(m.pattern());
        }
        let hm_start = match self.try_search_half_start(cache, input) {
            Err(RetryError::Quadratic(err)) => {
                trace!("reverse suffix captures optimization failed: {}", err);
                return self.core.search_slots(cache, input, slots);
            }
            Err(RetryError::Fail(err)) => {
                trace!(
                    "reverse suffix reverse fast captures search failed: {}",
                    err
                );
                return self.core.search_slots_nofail(cache, input, slots);
            }
            Ok(None) => return None,
            Ok(Some((hm_start, _))) => hm_start,
        };
        trace!(
            "match found at {}..{} in capture search, \
		  	 using another engine to find captures",
            hm_start.offset(),
            input.end(),
        );
        let start = hm_start.offset();
        let input = input
            .clone()
            .span(start..input.end())
            .anchored(Anchored::Pattern(hm_start.pattern()));
        self.core.search_slots_nofail(cache, &input, slots)
    }

    #[inline(always)]
    fn which_overlapping_matches(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
        patset: &mut PatternSet,
    ) {
        self.core.which_overlapping_matches(cache, input, patset)
    }
}

#[derive(Debug)]
struct ReverseInner {
    core: Core,
    preinner: Prefilter,
    nfarev: NFA,
    hybrid: wrappers::ReverseHybrid,
    dfa: wrappers::ReverseDFA,
}

impl ReverseInner {
    fn new(hirs: &[&Hir], core: Core) -> Result<ReverseInner, Core> {
        if !core.info.config().get_auto_prefilter() {
            debug!(
                "skipping reverse inner optimization because \
                 automatic prefilters are disabled"
            );
            return Err(core);
        }
        // Currently we hard-code the assumption of leftmost-first match
        // semantics. This isn't a huge deal because 'all' semantics tend to
        // only be used for forward overlapping searches with multiple regexes,
        // and this optimization only supports a single pattern at the moment.
        if core.info.config().get_match_kind() != MatchKind::LeftmostFirst {
            debug!(
                "skipping reverse inner optimization because \
				 match kind is {:?} but this only supports leftmost-first",
                core.info.config().get_match_kind(),
            );
            return Err(core);
        }
        // It's likely that a reverse inner scan has too much overhead for it
        // to be worth it when the regex is anchored at the start. It is
        // possible for it to be quite a bit faster if the initial literal
        // scan fails to detect a match, in which case, we can say "no match"
        // very quickly. But this could be undesirable, e.g., scanning too far
        // or when the literal scan matches. If it matches, then confirming the
        // match requires a reverse scan followed by a forward scan to confirm
        // or reject, which is a fair bit of work.
        if core.info.is_always_anchored_start() {
            debug!(
                "skipping reverse inner optimization because \
				 the regex is always anchored at the start",
            );
            return Err(core);
        }
        // Only DFAs can do reverse searches (currently), so we need one of
        // them in order to do this optimization. It's possible (although
        // pretty unlikely) that we have neither and need to give up.
        if !core.hybrid.is_some() && !core.dfa.is_some() {
            debug!(
                "skipping reverse inner optimization because \
				 we don't have a lazy DFA or a full DFA"
            );
            return Err(core);
        }
        if core.pre.as_ref().map_or(false, |p| p.is_fast()) {
            debug!(
                "skipping reverse inner optimization because \
				 we already have a prefilter that we think is fast"
            );
            return Err(core);
        }
        let (concat_prefix, preinner) = match reverse_inner::extract(hirs) {
            Some(x) => x,
            // N.B. the 'extract' function emits debug messages explaining
            // why we bailed out here.
            None => return Err(core),
        };
        debug!("building reverse NFA for prefix before inner literal");
        let mut lookm = LookMatcher::new();
        lookm.set_line_terminator(core.info.config().get_line_terminator());
        let thompson_config = thompson::Config::new()
            .reverse(true)
            .utf8(core.info.config().get_utf8())
            .nfa_size_limit(core.info.config().get_nfa_size_limit())
            .shrink(false)
            .captures(false)
            .look_matcher(lookm);
        let result = thompson::Compiler::new()
            .configure(thompson_config)
            .build_from_hir(&concat_prefix);
        let nfarev = match result {
            Ok(nfarev) => nfarev,
            Err(err) => {
                debug!(
                    "skipping reverse inner optimization because the \
					 reverse NFA failed to build: {}",
                    err,
                );
                return Err(core);
            }
        };
        debug!("building reverse DFA for prefix before inner literal");
        let hybrid = wrappers::ReverseHybrid::none();
        let dfa = wrappers::ReverseDFA::none();
        let dfa = if !core.info.config().get_dfa() {
            wrappers::ReverseDFA::none()
        } else {
            wrappers::ReverseDFA::new(&core.info, &nfarev)
        };
        let hybrid = if !core.info.config().get_hybrid() {
            wrappers::ReverseHybrid::none()
        } else if dfa.is_some() {
            debug!(
                "skipping lazy DFA for reverse inner optimization \
				 because we have a full DFA"
            );
            wrappers::ReverseHybrid::none()
        } else {
            wrappers::ReverseHybrid::new(&core.info, &nfarev)
        };
        Ok(ReverseInner { core, preinner, nfarev, hybrid, dfa })
    }

    #[inline(always)]
    fn try_search_full(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
    ) -> Result<Option<Match>, RetryError> {
        let mut span = input.get_span();
        let mut min_match_start = 0;
        let mut min_pre_start = 0;
        loop {
            let litmatch = match self.preinner.find(input.haystack(), span) {
                None => return Ok(None),
                Some(span) => span,
            };
            if litmatch.start < min_pre_start {
                trace!(
                    "found inner prefilter match at {:?}, which starts \
					 before the end of the last forward scan at {}, \
					 quitting to avoid quadratic behavior",
                    litmatch,
                    min_pre_start,
                );
                return Err(RetryError::Quadratic(RetryQuadraticError::new()));
            }
            trace!("reverse inner scan found inner match at {:?}", litmatch);
            let revinput = input
                .clone()
                .anchored(Anchored::Yes)
                .span(input.start()..litmatch.start);
            match self.try_search_half_rev_limited(
                cache,
                &revinput,
                min_match_start,
            )? {
                None => {
                    if span.start >= span.end {
                        break;
                    }
                    span.start = litmatch.start.checked_add(1).unwrap();
                }
                Some(hm_start) => {
                    let fwdinput = input
                        .clone()
                        .anchored(Anchored::Pattern(hm_start.pattern()))
                        .span(hm_start.offset()..input.end());
                    match self.try_search_half_fwd_stopat(cache, &fwdinput)? {
                        Err(stopat) => {
                            min_pre_start = stopat;
                            span.start =
                                litmatch.start.checked_add(1).unwrap();
                        }
                        Ok(hm_end) => {
                            return Ok(Some(Match::new(
                                hm_start.pattern(),
                                hm_start.offset()..hm_end.offset(),
                            )))
                        }
                    }
                }
            }
            min_match_start = litmatch.end;
        }
        Ok(None)
    }

    #[inline(always)]
    fn try_search_half_fwd_stopat(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
    ) -> Result<Result<HalfMatch, usize>, RetryFailError> {
        if let Some(e) = self.core.dfa.get(&input) {
            trace!(
                "using full DFA for forward reverse inner search at {:?}",
                input.get_span()
            );
            e.try_search_half_fwd_stopat(&input)
        } else if let Some(e) = self.core.hybrid.get(&input) {
            trace!(
                "using lazy DFA for forward reverse inner search at {:?}",
                input.get_span()
            );
            e.try_search_half_fwd_stopat(&mut cache.hybrid, &input)
        } else {
            unreachable!("ReverseInner always has a DFA")
        }
    }

    #[inline(always)]
    fn try_search_half_rev_limited(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
        min_start: usize,
    ) -> Result<Option<HalfMatch>, RetryError> {
        if let Some(e) = self.dfa.get(&input) {
            trace!(
                "using full DFA for reverse inner search at {:?}, \
                 but will be stopped at {} to avoid quadratic behavior",
                input.get_span(),
                min_start,
            );
            e.try_search_half_rev_limited(&input, min_start)
        } else if let Some(e) = self.hybrid.get(&input) {
            trace!(
                "using lazy DFA for reverse inner search at {:?}, \
                 but will be stopped at {} to avoid quadratic behavior",
                input.get_span(),
                min_start,
            );
            e.try_search_half_rev_limited(
                &mut cache.revhybrid,
                &input,
                min_start,
            )
        } else {
            unreachable!("ReverseInner always has a DFA")
        }
    }
}

impl Strategy for ReverseInner {
    #[inline(always)]
    fn create_captures(&self) -> Captures {
        self.core.create_captures()
    }

    #[inline(always)]
    fn create_cache(&self) -> Cache {
        let mut cache = self.core.create_cache();
        cache.revhybrid = self.hybrid.create_cache();
        cache
    }

    #[inline(always)]
    fn reset_cache(&self, cache: &mut Cache) {
        self.core.reset_cache(cache);
    }

    #[inline(always)]
    fn search(&self, cache: &mut Cache, input: &Input<'_>) -> Option<Match> {
        match self.try_search_full(cache, input) {
            Err(RetryError::Quadratic(err)) => {
                trace!("reverse inner optimization failed: {}", err);
                self.core.search(cache, input)
            }
            Err(RetryError::Fail(err)) => {
                trace!("reverse inner fast search failed: {}", err);
                self.core.search_nofail(cache, input)
            }
            Ok(matornot) => matornot,
        }
    }

    #[inline(always)]
    fn search_half(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
    ) -> Option<HalfMatch> {
        match self.try_search_full(cache, input) {
            Err(RetryError::Quadratic(err)) => {
                trace!("reverse inner half optimization failed: {}", err);
                self.core.search_half(cache, input)
            }
            Err(RetryError::Fail(err)) => {
                trace!("reverse inner fast half search failed: {}", err);
                self.core.search_half_nofail(cache, input)
            }
            Ok(None) => None,
            Ok(Some(m)) => Some(HalfMatch::new(m.pattern(), m.end())),
        }
    }

    #[inline(always)]
    fn search_slots(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Option<PatternID> {
        if !self.core.is_capture_search_needed(slots.len()) {
            trace!("asked for slots unnecessarily, trying fast path");
            let m = self.search(cache, input)?;
            copy_match_to_slots(m, slots);
            return Some(m.pattern());
        }
        let m = match self.try_search_full(cache, input) {
            Err(RetryError::Quadratic(err)) => {
                trace!("reverse inner captures optimization failed: {}", err);
                return self.core.search_slots(cache, input, slots);
            }
            Err(RetryError::Fail(err)) => {
                trace!("reverse inner fast captures search failed: {}", err);
                return self.core.search_slots_nofail(cache, input, slots);
            }
            Ok(None) => return None,
            Ok(Some(m)) => m,
        };
        trace!(
            "match found at {}..{} in capture search, \
		  	 using another engine to find captures",
            m.start(),
            m.end(),
        );
        let input = input
            .clone()
            .span(m.start()..m.end())
            .anchored(Anchored::Pattern(m.pattern()));
        self.core.search_slots_nofail(cache, &input, slots)
    }

    #[inline(always)]
    fn which_overlapping_matches(
        &self,
        cache: &mut Cache,
        input: &Input<'_>,
        patset: &mut PatternSet,
    ) {
        self.core.which_overlapping_matches(cache, input, patset)
    }
}

/// Attempts to extract an alternation of literals, and if it's deemed worth
/// doing, returns an Aho-Corasick prefilter as a strategy.
///
/// And currently, this only returns something when 'hirs.len() == 1'. This
/// could in theory do something if there are multiple HIRs where all of them
/// are alternation of literals, but I haven't had the time to go down that
/// path yet.
#[cfg(feature = "perf-literal-multisubstring")]
fn alternation_literals_to_aho_corasick(
    info: &RegexInfo,
    hirs: &[&Hir],
) -> Option<Arc<dyn Strategy>> {
    use crate::util::prefilter::AhoCorasick;

    let lits = alternation_literals(info, hirs)?;
    let ac = AhoCorasick::new(MatchKind::LeftmostFirst, &lits)?;
    Some(Arc::new(ac))
}

/// Pull out an alternation of literals from the given sequence of HIR
/// expressions.
///
/// There are numerous ways for this to fail. Generally, this only applies
/// to regexes of the form 'foo|bar|baz|...|quux'. It can also fail if there
/// are "too few" alternates, in which case, the regex engine is likely faster.
///
/// And currently, this only returns something when 'hirs.len() == 1'.
#[cfg(feature = "perf-literal-multisubstring")]
fn alternation_literals(
    info: &RegexInfo,
    hirs: &[&Hir],
) -> Option<Vec<Vec<u8>>> {
    use regex_syntax::hir::{HirKind, Literal};

    // This is pretty hacky, but basically, if `is_alternation_literal` is
    // true, then we can make several assumptions about the structure of our
    // HIR. This is what justifies the `unreachable!` statements below.
    //
    // This code should be refactored once we overhaul this crate's
    // optimization pipeline, because this is a terribly inflexible way to go
    // about things.
    if hirs.len() != 1
        || !info.props()[0].look_set().is_empty()
        || info.props()[0].captures_len() > 0
        || !info.props()[0].is_alternation_literal()
        || info.config().get_match_kind() != MatchKind::LeftmostFirst
    {
        return None;
    }
    let hir = &hirs[0];
    let alts = match *hir.kind() {
        HirKind::Alternation(ref alts) => alts,
        _ => return None, // one literal isn't worth it
    };

    let mut lits = vec![];
    for alt in alts {
        let mut lit = vec![];
        match *alt.kind() {
            HirKind::Literal(Literal(ref bytes)) => {
                lit.extend_from_slice(bytes)
            }
            HirKind::Concat(ref exprs) => {
                for e in exprs {
                    match *e.kind() {
                        HirKind::Literal(Literal(ref bytes)) => {
                            lit.extend_from_slice(bytes);
                        }
                        _ => unreachable!("expected literal, got {:?}", e),
                    }
                }
            }
            _ => unreachable!("expected literal or concat, got {:?}", alt),
        }
        lits.push(lit);
    }
    // Why do this? Well, when the number of literals is small, it's likely
    // that we'll use the lazy DFA which is in turn likely to be faster than
    // Aho-Corasick in such cases. Primarily because Aho-Corasick doesn't have
    // a "lazy DFA" but either a contiguous NFA or a full DFA. We rarely use
    // the latter because it is so hungry (in time and space), and the former
    // is decently fast, but not as fast as a well oiled lazy DFA.
    //
    // However, once the number starts getting large, the lazy DFA is likely
    // to start thrashing because of the modest default cache size. When
    // exactly does this happen? Dunno. But at whatever point that is (we make
    // a guess below based on ad hoc benchmarking), we'll want to cut over to
    // Aho-Corasick, where even the contiguous NFA is likely to do much better.
    if lits.len() < 3000 {
        debug!("skipping Aho-Corasick because there are too few literals");
        return None;
    }
    Some(lits)
}

#[inline(always)]
fn copy_match_to_slots(m: Match, slots: &mut [Option<NonMaxUsize>]) {
    let slot_start = m.pattern().as_usize() * 2;
    let slot_end = slot_start + 1;
    if let Some(slot) = slots.get_mut(slot_start) {
        *slot = NonMaxUsize::new(m.start());
    }
    if let Some(slot) = slots.get_mut(slot_end) {
        *slot = NonMaxUsize::new(m.end());
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
