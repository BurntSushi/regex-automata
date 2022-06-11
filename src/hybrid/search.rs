use crate::{
    hybrid::{
        dfa::{Cache, DFA},
        id::{LazyStateID, OverlappingState},
    },
    nfa::thompson,
    util::{
        id::PatternID,
        prefilter::Prefilter,
        search::{HalfMatch, MatchError, Search, Span},
    },
};

#[inline(never)]
pub(crate) fn find_fwd(
    dfa: &DFA,
    cache: &mut Cache,
    search: &Search<'_, '_>,
) -> Result<Option<HalfMatch>, MatchError> {
    if search.is_done() {
        return Ok(None);
    }
    // So what we do here is specialize four different versions of 'find_fwd':
    // one for each of the combinations for 'has prefilter' and 'is earliest
    // search'. The reason for doing this is that both of these things require
    // branches and special handling in some code that can be very hot,
    // and shaving off as much as we can when we don't need it tends to be
    // beneficial in ad hoc benchmarks. To see these differences, you often
    // need a query with a high match count. In other words, specializing these
    // four routines *tends* to help latency more than throughput.
    if search.get_prefilter().is_some() && search.get_pattern().is_none() {
        // Searching with a pattern ID is always anchored, so we should never
        // use a prefilter.
        if search.get_earliest() {
            find_fwd_imp(dfa, cache, search, search.get_prefilter(), true)
        } else {
            find_fwd_imp(dfa, cache, search, search.get_prefilter(), false)
        }
    } else {
        if search.get_earliest() {
            find_fwd_imp(dfa, cache, search, None, true)
        } else {
            find_fwd_imp(dfa, cache, search, None, false)
        }
    }
}

#[inline(always)]
fn find_fwd_imp(
    dfa: &DFA,
    cache: &mut Cache,
    search: &Search<'_, '_>,
    pre: Option<&'_ dyn Prefilter>,
    earliest: bool,
) -> Result<Option<HalfMatch>, MatchError> {
    let mut sid = init_fwd(dfa, cache, search)?;
    let mut last_match = None;
    let mut at = search.start();
    // This could just be a closure, but then I think it would be unsound
    // because it would need to be safe to invoke. This way, the lack of safety
    // is clearer in the code below.
    macro_rules! next_unchecked {
        ($sid:expr, $at:expr) => {{
            let byte = *search.haystack().get_unchecked($at);
            dfa.next_state_untagged_unchecked(cache, $sid, byte)
        }};
    }

    if let Some(ref pre) = pre {
        let span = Span::from(at..search.end());
        // If a prefilter doesn't report false positives, then we don't need to
        // touch the DFA at all. However, since all matches include the pattern
        // ID, and the prefilter infrastructure doesn't report pattern IDs, we
        // limit this optimization to cases where there is exactly one pattern.
        // In that case, any match must be the 0th pattern.
        if dfa.pattern_len() == 1 && !pre.reports_false_positives() {
            // TODO: This looks wrong? Shouldn't offset be the END of a match?
            return Ok(pre.find(search.haystack(), span).into_option().map(
                |offset| HalfMatch { pattern: PatternID::ZERO, offset },
            ));
        } else {
            match pre.find(search.haystack(), span).into_option() {
                None => return Ok(None),
                Some(i) => {
                    at = i;
                }
            }
        }
    }
    while at < search.end() {
        if sid.is_tagged() {
            sid = dfa
                .next_state(cache, sid, search.haystack()[at])
                .map_err(|_| gave_up(at))?;
        } else {
            // SAFETY: There are two safety invariants we need to uphold
            // here in the loops below: that 'sid' and 'prev_sid' are valid
            // state IDs for this DFA, and that 'at' is a valid index into
            // 'haystack'. For the former, we rely on the invariant that
            // next_state* and start_state_forward always returns a valid state
            // ID (given a valid state ID in the former case), and that we are
            // only at this place in the code if 'sid' is untagged. Moreover,
            // every call to next_state_untagged_unchecked below is guarded by
            // a check that sid is untagged. For the latter safety invariant,
            // we always guard unchecked access with a check that 'at' is less
            // than 'end', where 'end <= haystack.len()'. In the unrolled loop
            // below, we ensure that 'at' is always in bounds.
            //
            // PERF: For justification of omitting bounds checks, it gives us a
            // ~10% bump in search time. This was used for a benchmark:
            //
            //     regex-cli find hybrid dfa @bigfile '(?m)^.+$' -UBb
            //
            // PERF: For justification for the loop unrolling, we use a few
            // different tests:
            //
            //     regex-cli find hybrid dfa @$bigfile '\w{50}' -UBb
            //     regex-cli find hybrid dfa @$bigfile '(?m)^.+$' -UBb
            //     regex-cli find hybrid dfa @$bigfile 'ZQZQZQZQ' -UBb
            //
            // And there are three different configurations:
            //
            //     nounroll: this entire 'else' block vanishes and we just
            //               always use 'dfa.next_state(..)'.
            //      unroll1: just the outer loop below
            //      unroll2: just the inner loop below
            //      unroll3: both the outer and inner loops below
            //
            // This results in a matrix of timings for each of the above
            // regexes with each of the above unrolling configurations:
            //
            //              '\w{50}'   '(?m)^.+$'   'ZQZQZQZQ'
            //   nounroll   1.51s      2.34s        1.51s
            //    unroll1   1.53s      2.32s        1.56s
            //    unroll2   2.22s      1.50s        0.61s
            //    unroll3   1.67s      1.45s        0.61s
            //
            // Ideally we'd be able to find a configuration they yields the
            // best time for all regexes, but alas we settle for unroll3 that
            // gives us *almost* the best for '\w{50}' and the best for the
            // other two regexes.
            //
            // So what exactly is going on here? The first unrolling (grouping
            // together runs of untagged transitions) specifically targets
            // our choice of representation. The second unrolling (grouping
            // together runs of self-transitions) specifically targets a common
            // DFA topology. Let's dig in a little bit by looking at our
            // regexes:
            //
            // '\w{50}': This regex spends a lot of time outside of the DFA's
            // start state matching some part of the '\w' repetition. This
            // means that it's a bit of a worst case for loop unrolling that
            // targets self-transitions since the self-transitions in '\w{50}'
            // are not particularly active for this haystack. However, the
            // first unrolling (grouping together untagged transitions)
            // does apply quite well here since very few transitions hit
            // match/dead/quit/unknown states. It is however worth mentioning
            // that if start states are configured to be tagged (which you
            // typically want to do if you have a prefilter), then this regex
            // actually slows way down because it is constantly ping-ponging
            // out of the unrolled loop and into the handling of a tagged start
            // state below. But when start states aren't tagged, the unrolled
            // loop stays hot. (This is why it's imperative that start state
            // tagging be disabled when there isn't a prefilter!)
            //
            // '(?m)^.+$': There are two important aspects of this regex: 1)
            // on this haystack, its match count is very high, much higher
            // than the other two regex and 2) it spends the vast majority
            // of its time matching '.+'. Since Unicode mode is disabled,
            // this corresponds to repeatedly following self transitions for
            // the vast majority of the input. This does benefit from the
            // untagged unrolling since most of the transitions will be to
            // untagged states, but the untagged unrolling does more work than
            // what is actually required. Namely, it has to keep track of the
            // previous and next state IDs, which I guess requires a bit more
            // shuffling. This is supported by the fact that nounroll+unroll1
            // are both slower than unroll2+unroll3, where the latter has a
            // loop unrolling that specifically targets self-transitions.
            //
            // 'ZQZQZQZQ': This one is very similar to '(?m)^.+$' because it
            // spends the vast majority of its time in self-transitions for
            // the (implicit) unanchored prefix. The main difference with
            // '(?m)^.+$' is that it has a much lower match count. So there
            // isn't much time spent in the overhead of reporting matches. This
            // is the primary explainer in the perf difference here. We include
            // this regex and the former to make sure we have comparison points
            // with high and low match counts.
            //
            // NOTE: I used 'OpenSubtitles2018.raw.sample.en' for 'bigfile'.
            let mut prev_sid = sid;
            while at < search.end() {
                prev_sid = unsafe { next_unchecked!(sid, at) };
                if prev_sid.is_tagged() || at + 3 >= search.end() {
                    core::mem::swap(&mut prev_sid, &mut sid);
                    break;
                }
                at += 1;

                sid = unsafe { next_unchecked!(prev_sid, at) };
                if sid.is_tagged() {
                    break;
                }
                at += 1;

                prev_sid = unsafe { next_unchecked!(sid, at) };
                if prev_sid.is_tagged() {
                    core::mem::swap(&mut prev_sid, &mut sid);
                    break;
                }
                at += 1;

                sid = unsafe { next_unchecked!(prev_sid, at) };
                if sid.is_tagged() {
                    break;
                }
                at += 1;

                if prev_sid == sid {
                    while at + 4 < search.end() {
                        let next = unsafe { next_unchecked!(sid, at) };
                        if sid != next {
                            break;
                        }
                        at += 1;

                        let next = unsafe { next_unchecked!(sid, at) };
                        if sid != next {
                            break;
                        }
                        at += 1;

                        let next = unsafe { next_unchecked!(sid, at) };
                        if sid != next {
                            break;
                        }
                        at += 1;

                        let next = unsafe { next_unchecked!(sid, at) };
                        if sid != next {
                            break;
                        }
                        at += 1;
                    }
                }
            }
            // If we quit out of the code above with an unknown state ID at
            // any point, then we need to re-compute that transition using
            // 'next_state', which will do NFA powerset construction for us.
            if sid.is_unknown() {
                sid = dfa
                    .next_state(cache, prev_sid, search.haystack()[at])
                    .map_err(|_| gave_up(at))?;
            }
        }
        if sid.is_tagged() {
            if sid.is_start() {
                if let Some(ref pre) = pre {
                    let span = Span::from(at..search.end());
                    match pre.find(search.haystack(), span).into_option() {
                        // TODO: This looks like a bug to me. We should
                        // return 'Ok(last_match)', i.e., treat it like a
                        // dead state. But don't 'fix' it until we can
                        // write a regression test.
                        None => return Ok(None),
                        Some(i) => {
                            at = i;
                            // We want to skip any update to 'at' below
                            // at the end of this iteration and just
                            // jump immediately back to the next state
                            // transition at the leading position of the
                            // candidate match.
                            continue;
                        }
                    }
                }
            } else if sid.is_match() {
                let pattern = dfa.match_pattern(cache, sid, 0);
                // Since slice ranges are inclusive at the beginning and
                // exclusive at the end, and since forward searches report
                // the end, we can return 'at' as-is. This only works because
                // matches are delayed by 1 byte. So by the time we observe a
                // match, 'at' has already been set to 1 byte past the actual
                // match location, which is precisely the exclusive ending
                // bound of the match.
                last_match = Some(HalfMatch { pattern, offset: at });
                if earliest {
                    return Ok(last_match);
                }
            } else if sid.is_dead() {
                return Ok(last_match);
            } else if sid.is_quit() {
                if last_match.is_some() {
                    return Ok(last_match);
                }
                return Err(MatchError::Quit {
                    byte: search.haystack()[at],
                    offset: at,
                });
            } else {
                debug_assert!(sid.is_unknown());
                unreachable!("sid being unknown is a bug");
            }
        }
        at += 1;
    }
    Ok(eoi_fwd(dfa, cache, search, &mut sid)?.or(last_match))
}

#[inline(never)]
pub(crate) fn find_rev(
    dfa: &DFA,
    cache: &mut Cache,
    search: &Search<'_, '_>,
) -> Result<Option<HalfMatch>, MatchError> {
    if search.is_done() {
        return Ok(None);
    }
    if search.get_earliest() {
        find_rev_imp(dfa, cache, search, true)
    } else {
        find_rev_imp(dfa, cache, search, false)
    }
}

#[inline(always)]
fn find_rev_imp(
    dfa: &DFA,
    cache: &mut Cache,
    search: &Search<'_, '_>,
    earliest: bool,
) -> Result<Option<HalfMatch>, MatchError> {
    let mut sid = init_rev(dfa, cache, search)?;
    // In reverse search, the loop below can't handle the case of searching an
    // empty slice. Ideally we could write something congruent to the forward
    // search, i.e., 'while at >= start', but 'start' might be 0. Since we use
    // an unsigned offset, 'at >= 0' is trivially always true. We could avoid
    // this extra case handling by using a signed offset, but Rust makes it
    // annoying to do. So... We just handle the empty case separately.
    if search.start() == search.end() {
        return Ok(eoi_rev(dfa, cache, search, &mut sid)?);
    }

    let mut last_match = None;
    let mut at = search.end() - 1;
    macro_rules! next_unchecked {
        ($sid:expr, $at:expr) => {{
            let byte = *search.haystack().get_unchecked($at);
            dfa.next_state_untagged_unchecked(cache, $sid, byte)
        }};
    }
    loop {
        if sid.is_tagged() {
            sid = dfa
                .next_state(cache, sid, search.haystack()[at])
                .map_err(|_| gave_up(at))?;
        } else {
            // SAFETY: See comments in 'find_fwd' for a safety argument.
            //
            // PERF: The comments in 'find_fwd' also provide a justification
            // from a performance perspective as to 1) why we elide bounds
            // checks and 2) why we do a specialized version of unrolling
            // below. The reverse search does have a slightly different
            // consideration in that most reverse searches tend to be
            // anchored and on shorter haystacks. However, this still makes a
            // difference. Take this command for example:
            //
            //     regex-cli find hybrid regex @$bigfile '(?m)^.+$' -UBb
            //
            // (Notice that we use 'find hybrid regex', not 'find hybrid dfa'
            // like in the justification for the forward direction. The 'regex'
            // sub-command will find start-of-match and thus run the reverse
            // direction.)
            //
            // Without unrolling below, the above command takes around 3.76s.
            // But with the unrolling below, we get down to 2.55s. If we keep
            // the unrolling but add in bounds checks, then we get 2.86s.
            //
            // NOTE: I used 'OpenSubtitles2018.raw.sample.en' for 'bigfile'.
            let mut prev_sid = sid;
            while at >= search.start() {
                prev_sid = unsafe { next_unchecked!(sid, at) };
                if prev_sid.is_tagged()
                    || at <= search.start().saturating_add(3)
                {
                    core::mem::swap(&mut prev_sid, &mut sid);
                    break;
                }
                at -= 1;

                sid = unsafe { next_unchecked!(prev_sid, at) };
                if sid.is_tagged() {
                    break;
                }
                at -= 1;

                prev_sid = unsafe { next_unchecked!(sid, at) };
                if prev_sid.is_tagged() {
                    core::mem::swap(&mut prev_sid, &mut sid);
                    break;
                }
                at -= 1;

                sid = unsafe { next_unchecked!(prev_sid, at) };
                if sid.is_tagged() {
                    break;
                }
                at -= 1;

                if prev_sid == sid {
                    while at > search.start().saturating_add(3) {
                        let next = unsafe { next_unchecked!(sid, at) };
                        if sid != next {
                            break;
                        }
                        at -= 1;

                        let next = unsafe { next_unchecked!(sid, at) };
                        if sid != next {
                            break;
                        }
                        at -= 1;

                        let next = unsafe { next_unchecked!(sid, at) };
                        if sid != next {
                            break;
                        }
                        at -= 1;

                        let next = unsafe { next_unchecked!(sid, at) };
                        if sid != next {
                            break;
                        }
                        at -= 1;
                    }
                }
            }
            // If we quit out of the code above with an unknown state ID at
            // any point, then we need to re-compute that transition using
            // 'next_state', which will do NFA powerset construction for us.
            if sid.is_unknown() {
                sid = dfa
                    .next_state(cache, prev_sid, search.haystack()[at])
                    .map_err(|_| gave_up(at))?;
            }
        }
        if sid.is_tagged() {
            if sid.is_start() {
                continue;
            } else if sid.is_match() {
                last_match = Some(HalfMatch {
                    pattern: dfa.match_pattern(cache, sid, 0),
                    // Since reverse searches report the beginning of a match
                    // and the beginning is inclusive (not exclusive like the
                    // end of a match), we add 1 to make it inclusive.
                    offset: at + 1,
                });
                if earliest {
                    return Ok(last_match);
                }
            } else if sid.is_dead() {
                return Ok(last_match);
            } else if sid.is_quit() {
                if last_match.is_some() {
                    return Ok(last_match);
                }
                return Err(MatchError::Quit {
                    byte: search.haystack()[at],
                    offset: at,
                });
            } else {
                debug_assert!(sid.is_unknown());
                unreachable!("sid being unknown is a bug");
            }
        }
        if at == search.start() {
            break;
        }
        at -= 1;
    }
    Ok(eoi_rev(dfa, cache, search, &mut sid)?.or(last_match))
}

#[inline(never)]
pub(crate) fn find_overlapping_fwd(
    dfa: &DFA,
    cache: &mut Cache,
    search: &Search<'_, '_>,
    state: &mut OverlappingState,
) -> Result<Option<HalfMatch>, MatchError> {
    if search.is_done() {
        return Ok(None);
    }
    // Searching with a pattern ID is always anchored, so we should never use
    // a prefilter.
    if search.get_prefilter().is_some() && search.get_pattern().is_none() {
        let pre = search.get_prefilter();
        find_overlapping_fwd_imp(dfa, cache, search, pre, state)
    } else {
        find_overlapping_fwd_imp(dfa, cache, search, None, state)
    }
}

#[inline(always)]
fn find_overlapping_fwd_imp(
    dfa: &DFA,
    cache: &mut Cache,
    search: &Search<'_, '_>,
    pre: Option<&'_ dyn Prefilter>,
    state: &mut OverlappingState,
) -> Result<Option<HalfMatch>, MatchError> {
    let mut sid = match state.id {
        None => {
            state.at = search.start();
            init_fwd(dfa, cache, search)?
        }
        Some(sid) => {
            if let Some(match_index) = state.next_match_index {
                let match_len = dfa.match_len(cache, sid);
                if match_index < match_len {
                    let m = HalfMatch {
                        pattern: dfa.match_pattern(cache, sid, match_index),
                        offset: state.at,
                    };
                    state.next_match_index = Some(match_index + 1);
                    return Ok(Some(m));
                }
            }
            // Once we've reported all matches at a given position, we need to
            // advance the search to the next position.
            state.at += 1;
            sid
        }
    };

    // NOTE: We don't optimize the crap out of this routine primarily because
    // it seems like most overlapping searches will have higher match counts,
    // and thus, throughput is perhaps not as important. But if you have a use
    // case for something faster, feel free to file an issue.
    while state.at < search.end() {
        sid = dfa
            .next_state(cache, sid, search.haystack()[state.at])
            .map_err(|_| gave_up(state.at))?;
        if sid.is_tagged() {
            state.id = Some(sid);
            if sid.is_start() {
                if let Some(ref pre) = pre {
                    let span = Span::from(state.at..search.end());
                    match pre.find(search.haystack(), span).into_option() {
                        None => return Ok(None),
                        Some(i) => {
                            state.at = i;
                            continue;
                        }
                    }
                }
            } else if sid.is_match() {
                state.next_match_index = Some(1);
                return Ok(Some(HalfMatch {
                    pattern: dfa.match_pattern(cache, sid, 0),
                    offset: state.at,
                }));
            } else if sid.is_dead() {
                return Ok(None);
            } else if sid.is_quit() {
                return Err(MatchError::Quit {
                    byte: search.haystack()[state.at],
                    offset: state.at,
                });
            } else {
                debug_assert!(sid.is_unknown());
                unreachable!("sid being unknown is a bug");
            }
        }
        state.at += 1;
    }

    let result = eoi_fwd(dfa, cache, search, &mut sid);
    state.id = Some(sid);
    if let Ok(Some(ref last_match)) = result {
        // '1' is always correct here since if we get to this point, this
        // always corresponds to the first (index '0') match discovered at
        // this position. So the next match to report at this position (if
        // it exists) is at index '1'.
        state.next_match_index = Some(1);
    }
    result
}

#[inline(never)]
pub(crate) fn find_overlapping_rev(
    dfa: &DFA,
    cache: &mut Cache,
    search: &Search<'_, '_>,
    state: &mut OverlappingState,
) -> Result<Option<HalfMatch>, MatchError> {
    if search.is_done() {
        return Ok(None);
    }
    let mut sid = match state.id {
        None => {
            state.id = Some(init_rev(dfa, cache, search)?);
            if search.start() == search.end() {
                state.rev_eoi = true;
            } else {
                state.at = search.end() - 1;
            }
            state.id.unwrap()
        }
        Some(sid) => {
            if let Some(match_index) = state.next_match_index {
                let match_len = dfa.match_len(cache, sid);
                if match_index < match_len {
                    let m = HalfMatch {
                        pattern: dfa.match_pattern(cache, sid, match_index),
                        offset: state.at,
                    };
                    state.next_match_index = Some(match_index + 1);
                    return Ok(Some(m));
                }
            }
            // Once we've reported all matches at a given position, we need
            // to advance the search to the next position. However, if we've
            // already followed the EOI transition, then we know we're done
            // with the search and there cannot be any more matches to report.
            if state.rev_eoi {
                return Ok(None);
            } else if state.at == search.start() {
                // At this point, we should follow the EOI transition. This
                // will cause us the skip the main loop below and fall through
                // to the final 'eoi_rev' transition.
                state.rev_eoi = true;
            } else {
                // We haven't hit the end of the search yet, so move on.
                state.at -= 1;
            }
            sid
        }
    };
    while !state.rev_eoi {
        sid = dfa
            .next_state(cache, sid, search.haystack()[state.at])
            .map_err(|_| gave_up(state.at))?;
        if sid.is_tagged() {
            state.id = Some(sid);
            if sid.is_start() {
                continue;
            } else if sid.is_match() {
                state.next_match_index = Some(1);
                return Ok(Some(HalfMatch {
                    pattern: dfa.match_pattern(cache, sid, 0),
                    offset: state.at + 1,
                }));
            } else if sid.is_dead() {
                return Ok(None);
            } else if sid.is_quit() {
                return Err(MatchError::Quit {
                    byte: search.haystack()[state.at],
                    offset: state.at,
                });
            } else {
                debug_assert!(sid.is_unknown());
                unreachable!("sid being unknown is a bug");
            }
        }
        if state.at == search.start() {
            break;
        }
        state.at -= 1;
    }

    state.rev_eoi = true;
    let result = eoi_rev(dfa, cache, search, &mut sid);
    state.id = Some(sid);
    if let Ok(Some(ref last_match)) = result {
        // '1' is always correct here since if we get to this point, this
        // always corresponds to the first (index '0') match discovered at
        // this position. So the next match to report at this position (if
        // it exists) is at index '1'.
        state.next_match_index = Some(1);
    }
    result
}

#[inline(always)]
fn init_fwd(
    dfa: &DFA,
    cache: &mut Cache,
    search: &Search<'_, '_>,
) -> Result<LazyStateID, MatchError> {
    let sid = dfa
        .start_state_forward(cache, search)
        .map_err(|_| gave_up(search.start()))?;
    // Start states can never be match states, since all matches are delayed
    // by 1 byte.
    debug_assert!(!sid.is_match());
    Ok(sid)
}

#[inline(always)]
fn init_rev(
    dfa: &DFA,
    cache: &mut Cache,
    search: &Search<'_, '_>,
) -> Result<LazyStateID, MatchError> {
    let sid = dfa
        .start_state_reverse(cache, search)
        .map_err(|_| gave_up(search.end()))?;
    // Start states can never be match states, since all matches are delayed
    // by 1 byte.
    debug_assert!(!sid.is_match());
    Ok(sid)
}

#[inline(always)]
fn eoi_fwd(
    dfa: &DFA,
    cache: &mut Cache,
    search: &Search<'_, '_>,
    sid: &mut LazyStateID,
) -> Result<Option<HalfMatch>, MatchError> {
    let sp = search.get_span();
    match search.haystack().get(sp.end) {
        Some(&b) => {
            *sid =
                dfa.next_state(cache, *sid, b).map_err(|_| gave_up(sp.end))?;
            if sid.is_match() {
                Ok(Some(HalfMatch {
                    pattern: dfa.match_pattern(cache, *sid, 0),
                    offset: sp.end,
                }))
            } else {
                Ok(None)
            }
        }
        None => {
            *sid = dfa
                .next_eoi_state(cache, *sid)
                .map_err(|_| gave_up(search.haystack().len()))?;
            if sid.is_match() {
                Ok(Some(HalfMatch {
                    pattern: dfa.match_pattern(cache, *sid, 0),
                    offset: search.haystack().len(),
                }))
            } else {
                Ok(None)
            }
        }
    }
}

#[inline(always)]
fn eoi_rev(
    dfa: &DFA,
    cache: &mut Cache,
    search: &Search<'_, '_>,
    sid: &mut LazyStateID,
) -> Result<Option<HalfMatch>, MatchError> {
    let sp = search.get_span();
    if sp.start > 0 {
        *sid = dfa
            .next_state(cache, *sid, search.haystack()[sp.start - 1])
            .map_err(|_| gave_up(sp.start))?;
        if sid.is_match() {
            Ok(Some(HalfMatch {
                pattern: dfa.match_pattern(cache, *sid, 0),
                offset: sp.start,
            }))
        } else {
            Ok(None)
        }
    } else {
        *sid =
            dfa.next_eoi_state(cache, *sid).map_err(|_| gave_up(sp.start))?;
        if sid.is_match() {
            Ok(Some(HalfMatch {
                pattern: dfa.match_pattern(cache, *sid, 0),
                offset: 0,
            }))
        } else {
            Ok(None)
        }
    }
}

/// A convenience routine for constructing a "gave up" match error.
#[inline(always)]
fn gave_up(offset: usize) -> MatchError {
    MatchError::GaveUp { offset }
}
