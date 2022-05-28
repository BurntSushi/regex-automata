use crate::{
    hybrid::{
        dfa::{Cache, DFA},
        id::{LazyStateID, OverlappingState, StateMatch},
    },
    nfa::thompson,
    util::{
        id::PatternID,
        prefilter,
        search::{HalfMatch, MatchError, Search, Span, MATCH_OFFSET},
    },
};

#[inline(never)]
pub(crate) fn find_fwd(
    dfa: &DFA,
    cache: &mut Cache,
    pre: Option<&mut prefilter::Scanner>,
    search: &Search<'_>,
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
    if pre.is_some() && search.get_pattern().is_none() {
        // Searching with a pattern ID is always anchored, so we should never
        // use a prefilter.
        if search.get_earliest() {
            find_fwd_imp(dfa, cache, pre, search, true)
        } else {
            find_fwd_imp(dfa, cache, pre, search, false)
        }
    } else {
        if search.get_earliest() {
            find_fwd_imp(dfa, cache, None, search, true)
        } else {
            find_fwd_imp(dfa, cache, None, search, false)
        }
    }
}

#[inline(always)]
fn find_fwd_imp(
    dfa: &DFA,
    cache: &mut Cache,
    mut pre: Option<&mut prefilter::Scanner>,
    search: &Search<'_>,
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

    if let Some(ref mut pre) = pre {
        let span = Span::new(at, search.end());
        // If a prefilter doesn't report false positives, then we don't need to
        // touch the DFA at all. However, since all matches include the pattern
        // ID, and the prefilter infrastructure doesn't report pattern IDs, we
        // limit this optimization to cases where there is exactly one pattern.
        // In that case, any match must be the 0th pattern.
        if dfa.pattern_count() == 1 && !pre.reports_false_positives() {
            // TODO: This looks wrong? Shouldn't offset be the END of a match?
            return Ok(pre.find(search.haystack(), span).into_option().map(
                |offset| HalfMatch { pattern: PatternID::ZERO, offset },
            ));
        } else if pre.is_effective(at) {
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
                if let Some(ref mut pre) = pre {
                    if pre.is_effective(at) {
                        let span = Span::new(at, search.end());
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
    search: &Search<'_>,
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
    search: &Search<'_>,
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
        return Ok(eoi_rev(dfa, cache, search, sid)?);
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
                    // Since slice ranges are inclusive at the beginning and
                    // exclusive at the end, and since reverse searches report
                    // the beginning, we need to offset the position by our
                    // MATCH_OFFSET (because matches are delayed by 1 byte in
                    // the DFA).
                    offset: at + MATCH_OFFSET,
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
    Ok(eoi_rev(dfa, cache, search, sid)?.or(last_match))
}

#[inline(never)]
pub(crate) fn find_overlapping_fwd(
    dfa: &DFA,
    cache: &mut Cache,
    pre: Option<&mut prefilter::Scanner>,
    search: &Search<'_>,
    state: &mut OverlappingState,
) -> Result<Option<HalfMatch>, MatchError> {
    if search.is_done() {
        return Ok(None);
    }
    // Searching with a pattern ID is always anchored, so we should never use
    // a prefilter.
    if pre.is_some() && search.get_pattern().is_none() {
        find_overlapping_fwd_imp(dfa, cache, pre, search, state)
    } else {
        find_overlapping_fwd_imp(dfa, cache, None, search, state)
    }
}

/// The implementation for forward overlapping search.
///
/// We do not have a corresponding reverse overlapping search. We can actually
/// reuse the existing non-overlapping reverse search to find start-of-match
/// for overlapping results. (Because the start-of-match aspect already
/// knows which pattern matched.) We might want a reverse overlapping search
/// though if we want to support proper reverse searching instead of just
/// start-of-match handling for forward searches.
#[inline(always)]
fn find_overlapping_fwd_imp(
    dfa: &DFA,
    cache: &mut Cache,
    mut pre: Option<&mut prefilter::Scanner>,
    search: &Search<'_>,
    state: &mut OverlappingState,
) -> Result<Option<HalfMatch>, MatchError> {
    // BREADCRUMBS: Consider 'ab|b' matched against 'ab'. Technically, the full
    // set of overlapping matches is [ab, b]. But right now, we only report
    // 'ab'. It's not clear that reporting 'b' is even possible. Consider: how
    // would the reverse search know to stop matching after seeing the 'b'?
    // It wouldn't. The only way to make something like this work is to split
    // 'ab|b' into two patterns 'ab' and 'b'. Then, since you have distinct
    // patterns, the pattern IDs force both matches to be reported and the
    // reverse search knows which pattern it's looking for and thus knows to
    // stop after seeing 'b'.
    //
    // So I think we need to somehow document this as a limitation. But it just
    // turns out to be real weird when, for example, matching 'samwise|wise'
    // against 'samwise' reports only [samwise], but matching 'samwise|wis'
    // reports [samwise, wis]. Like, wat. So we can yield all overlapping
    // matches... except for matches that end at the same location?
    //
    // Well... Unless... Our overlapping iterator only advanced by adding 1 to
    // the previous starting position. (Well, the overlapping iterator needs to
    // be pushed down into this routine.) If we did that, then we would indeed
    // find 'wise' because 'samwise' would stop matching. So maybe that's all
    // that's needed?
    //
    // And don't forget, it seems like matching '\w+' against 'bar' should
    // yield [b, ba, bar, a, ar, r], yet right now it yields [b, ba, bar]. I
    // guess that's also handled by the same trick above, ain't it?
    //
    // Turns out the "trick" above does not really have an obvious way to be
    // implemented. How do you know when to increment the starting position,
    // for example? Do you wait until you reach the EOI? Or until you find a
    // match after the initial starting position?
    //
    // No, actually, I believe the answer here is that our overlapping search
    // needs to be overlapping not only in the forwards direction, but in the
    // reverse direction as well! Take 'ab|b' against 'ab' for example. The
    // forward search will find 'ab'. A non-overlapping reverse search will
    // only find 'ab', since it's instructed to go as far as possible. But an
    // *overlapping* reverse search will find 'b' and 'ab'. Interestingly, the
    // results will then be [b, ab], since the 'b' will be found first. This is
    // QUITE weird. But you still get all of the results you want.
    //
    // This also fixes the 'samwise|wise' regex. The forward search on
    // 'samwise' finds 'samwise', then a reverse overlapping search will find
    // 'wise' and then 'samwise'.
    //
    // What about '\w+' against 'bar'? We get [b, a, ba, r, ar, bar]. Again,
    // the ordering is quite weird.
    //
    // What if we collected results returned by the reverse overlapping search
    // in-memory and then reversed them before yielding each? For '\w+' against
    // 'bar', we'd get [b, ba, a, bar, ar, r]. Which is also still pretty
    // weird, particularly since 'bar' follows 'a', despite 'bar' being more
    // leftmost than 'a'.
    //
    // Can we even describe this ordering in a coherent way?
    //
    // Also, how does a search make progress? I think that we always start our
    // search at the position following the end of the last search.
    //
    // What are the bounds of the reverse search? I think the bounds are just
    // always the same: the start of the current context. The only thing that
    // changes is the current position in the haystack at which to resume the
    // search because conceptually, iteration is one giant search unlike for
    // leftmost non-overlapping searches.

    let mut at = search.start();
    let mut sid = match state.id() {
        None => init_fwd(dfa, cache, search)?,
        Some(sid) => {
            if let Some(last) = state.last_match() {
                let match_count = dfa.match_count(cache, sid);
                if last.match_index < match_count {
                    let m = HalfMatch {
                        pattern: dfa.match_pattern(
                            cache,
                            sid,
                            last.match_index,
                        ),
                        offset: last.offset,
                    };
                    last.match_index += 1;
                    return Ok(Some(m));
                }
            }

            // This is a subtle but critical detail. If the caller provides a
            // non-None state ID, then it must be the case that the state ID
            // corresponds to one set by this function. The state ID therefore
            // corresponds to a match state, a dead state or some other state.
            // However, "some other" state _only_ occurs when the input has
            // been exhausted because the only way to stop before then is to
            // see a match or a dead/quit state.
            //
            // If the input is exhausted or if it's a dead state, then
            // incrementing the starting position has no relevance on
            // correctness, since the loop below will either not execute
            // at all or will immediately stop due to being in a dead state.
            // (Once in a dead state it is impossible to leave it.)
            //
            // Therefore, the only case we need to consider is when state
            // is a match state. In this case, since our machines support
            // the ability to delay a match by a certain number of bytes (to
            // support look-around), it follows that we actually consumed that
            // many additional bytes on our previous search. When the caller
            // resumes their search to find subsequent matches, they will use
            // the ending location from the previous match as the next starting
            // point, which is `match_offset` bytes PRIOR to where we scanned
            // to on the previous search. Therefore, we need to compensate by
            // bumping `start` up by `MATCH_OFFSET` bytes.
            at += MATCH_OFFSET;
            sid
        }
    };

    // NOTE: We don't optimize the crap out of this routine primarily because
    // it seems like most find_overlapping searches will have higher match
    // counts, and thus, throughput is perhaps not as important. But if you
    // have a use case for something faster, feel free to file an issue.
    while at < search.end() {
        sid = dfa
            .next_state(cache, sid, search.haystack()[at])
            .map_err(|_| gave_up(at))?;
        if sid.is_tagged() {
            state.set_id(sid);
            if sid.is_start() {
                if let Some(ref mut pre) = pre {
                    if pre.is_effective(at) {
                        let span = Span::new(at, search.end());
                        match pre.find(search.haystack(), span).into_option() {
                            None => return Ok(None),
                            Some(i) => {
                                at = i;
                                continue;
                            }
                        }
                    }
                }
            } else if sid.is_match() {
                state
                    .set_last_match(StateMatch { match_index: 1, offset: at });
                return Ok(Some(HalfMatch {
                    pattern: dfa.match_pattern(cache, sid, 0),
                    offset: at,
                }));
            } else if sid.is_dead() {
                return Ok(None);
            } else if sid.is_quit() {
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

    let result = eoi_fwd(dfa, cache, search, &mut sid);
    state.set_id(sid);
    if let Ok(Some(ref last_match)) = result {
        state.set_last_match(StateMatch {
            // '1' is always correct here since if we get to this point, this
            // always corresponds to the first (index '0') match discovered at
            // this position. So the next match to report at this position (if
            // it exists) is at index '1'.
            match_index: 1,
            offset: last_match.offset(),
        });
    }
    result
}

#[inline(always)]
fn init_fwd(
    dfa: &DFA,
    cache: &mut Cache,
    search: &Search<'_>,
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
    search: &Search<'_>,
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
    search: &Search<'_>,
    sid: &mut LazyStateID,
) -> Result<Option<HalfMatch>, MatchError> {
    match search.haystack().get(search.end()) {
        Some(&b) => {
            *sid = dfa
                .next_state(cache, *sid, b)
                .map_err(|_| gave_up(search.end()))?;
            if sid.is_match() {
                Ok(Some(HalfMatch {
                    pattern: dfa.match_pattern(cache, *sid, 0),
                    offset: search.end(),
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
    search: &Search<'_>,
    state: LazyStateID,
) -> Result<Option<HalfMatch>, MatchError> {
    if search.start() > 0 {
        let sid = dfa
            .next_state(cache, state, search.haystack()[search.start() - 1])
            .map_err(|_| gave_up(search.start()))?;
        if sid.is_match() {
            Ok(Some(HalfMatch {
                pattern: dfa.match_pattern(cache, sid, 0),
                offset: search.start(),
            }))
        } else {
            Ok(None)
        }
    } else {
        let sid = dfa
            .next_eoi_state(cache, state)
            .map_err(|_| gave_up(search.start()))?;
        if sid.is_match() {
            Ok(Some(HalfMatch {
                pattern: dfa.match_pattern(cache, sid, 0),
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
