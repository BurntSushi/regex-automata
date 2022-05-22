use crate::{
    dfa::{
        accel,
        automaton::{Automaton, OverlappingState, StateMatch},
    },
    util::{
        id::{PatternID, StateID},
        prefilter,
        search::{HalfMatch, Search, Span, MATCH_OFFSET},
    },
    MatchError,
};

#[inline(never)]
pub fn find_fwd<A: Automaton + ?Sized>(
    dfa: &A,
    pre: Option<&mut prefilter::Scanner>,
    search: &Search<'_>,
) -> Result<Option<HalfMatch>, MatchError> {
    if search.is_done() {
        return Ok(None);
    }
    // Searching with a pattern ID is always anchored, so we should never use
    // a prefilter.
    if pre.is_some() && search.get_pattern().is_none() {
        if search.get_earliest() {
            find_fwd_imp(dfa, pre, search, true)
        } else {
            find_fwd_imp(dfa, pre, search, false)
        }
    } else {
        if search.get_earliest() {
            find_fwd_imp(dfa, None, search, true)
        } else {
            find_fwd_imp(dfa, None, search, false)
        }
    }
}

#[inline(always)]
fn find_fwd_imp<A: Automaton + ?Sized>(
    dfa: &A,
    mut pre: Option<&mut prefilter::Scanner>,
    search: &Search<'_>,
    earliest: bool,
) -> Result<Option<HalfMatch>, MatchError> {
    let mut sid = init_fwd(dfa, search)?;
    let mut last_match = None;
    let mut at = search.start();
    // This could just be a closure, but then I think it would be unsound
    // because it would need to be safe to invoke. This way, the lack of safety
    // is clearer in the code below.
    macro_rules! next_unchecked {
        ($sid:expr, $at:expr) => {{
            let byte = *search.haystack().get_unchecked($at);
            dfa.next_state_unchecked($sid, byte)
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
        // SAFETY: There are two safety invariants we need to uphold here in
        // the loops below: that 'sid' and 'prev_sid' are valid state IDs
        // for this DFA, and that 'at' is a valid index into 'haystack'.
        // For the former, we rely on the invariant that next_state* and
        // start_state_forward always returns a valid state ID (given a valid
        // state ID in the former case). For the latter safety invariant, we
        // always guard unchecked access with a check that 'at' is less than
        // 'end', where 'end <= haystack.len()'. In the unrolled loop below, we
        // ensure that 'at' is always in bounds.
        //
        // PERF: See a similar comment in src/hybrid/search.rs that justifies
        // this extra work to make the search loop fast. The same reasoning and
        // benchmarks apply here.
        let mut prev_sid = sid;
        while at < search.end() {
            prev_sid = unsafe { next_unchecked!(sid, at) };
            if dfa.is_special_state(prev_sid) || at + 3 >= search.end() {
                core::mem::swap(&mut prev_sid, &mut sid);
                break;
            }
            at += 1;

            sid = unsafe { next_unchecked!(prev_sid, at) };
            if dfa.is_special_state(sid) {
                break;
            }
            at += 1;

            prev_sid = unsafe { next_unchecked!(sid, at) };
            if dfa.is_special_state(prev_sid) {
                core::mem::swap(&mut prev_sid, &mut sid);
                break;
            }
            at += 1;

            sid = unsafe { next_unchecked!(prev_sid, at) };
            if dfa.is_special_state(sid) {
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
        if dfa.is_special_state(sid) {
            if dfa.is_start_state(sid) {
                if let Some(ref mut pre) = pre {
                    if pre.is_effective(at) {
                        let span = Span::new(at, search.end());
                        match pre.find(search.haystack(), span).into_option() {
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
                } else if dfa.is_accel_state(sid) {
                    let needles = dfa.accelerator(sid);
                    at = accel::find_fwd(needles, search.haystack(), at)
                        .unwrap_or(search.end());
                    continue;
                }
            } else if dfa.is_match_state(sid) {
                let pattern = dfa.match_pattern(sid, 0);
                last_match = Some(HalfMatch { pattern, offset: at });
                if earliest {
                    return Ok(last_match);
                }
                if dfa.is_accel_state(sid) {
                    let needles = dfa.accelerator(sid);
                    at = accel::find_fwd(needles, search.haystack(), at)
                        .unwrap_or(search.end());
                    continue;
                }
            } else if dfa.is_accel_state(sid) {
                let needs = dfa.accelerator(sid);
                at = accel::find_fwd(needs, search.haystack(), at)
                    .unwrap_or(search.end());
                continue;
            } else if dfa.is_dead_state(sid) {
                return Ok(last_match);
            } else {
                debug_assert!(dfa.is_quit_state(sid));
                if last_match.is_some() {
                    return Ok(last_match);
                }
                return Err(MatchError::Quit {
                    byte: search.haystack()[at],
                    offset: at,
                });
            }
        }
        at += 1;
    }
    Ok(eoi_fwd(dfa, search, &mut sid)?.or(last_match))
}

#[inline(never)]
pub fn find_rev<A: Automaton + ?Sized>(
    dfa: &A,
    search: &Search<'_>,
) -> Result<Option<HalfMatch>, MatchError> {
    if search.is_done() {
        return Ok(None);
    }
    if search.get_earliest() {
        find_rev_imp(dfa, search, true)
    } else {
        find_rev_imp(dfa, search, false)
    }
}

#[inline(always)]
fn find_rev_imp<A: Automaton + ?Sized>(
    dfa: &A,
    search: &Search<'_>,
    earliest: bool,
) -> Result<Option<HalfMatch>, MatchError> {
    let mut sid = init_rev(dfa, search)?;
    // In reverse search, the loop below can't handle the case of searching an
    // empty slice. Ideally we could write something congruent to the forward
    // search, i.e., 'while at >= start', but 'start' might be 0. Since we use
    // an unsigned offset, 'at >= 0' is trivially always true. We could avoid
    // this extra case handling by using a signed offset, but Rust makes it
    // annoying to do. So... We just handle the empty case separately.
    if search.start() == search.end() {
        return Ok(eoi_rev(dfa, search, sid)?);
    }

    let mut last_match = None;
    let mut at = search.end() - 1;
    macro_rules! next_unchecked {
        ($sid:expr, $at:expr) => {{
            let byte = *search.haystack().get_unchecked($at);
            dfa.next_state_unchecked($sid, byte)
        }};
    }
    loop {
        // SAFETY: See comments in 'find_fwd' for a safety argument.
        let mut prev_sid = sid;
        while at >= search.start() {
            prev_sid = unsafe { next_unchecked!(sid, at) };
            if dfa.is_special_state(prev_sid)
                || at <= search.start().saturating_add(3)
            {
                core::mem::swap(&mut prev_sid, &mut sid);
                break;
            }
            at -= 1;

            sid = unsafe { next_unchecked!(prev_sid, at) };
            if dfa.is_special_state(sid) {
                break;
            }
            at -= 1;

            prev_sid = unsafe { next_unchecked!(sid, at) };
            if dfa.is_special_state(prev_sid) {
                core::mem::swap(&mut prev_sid, &mut sid);
                break;
            }
            at -= 1;

            sid = unsafe { next_unchecked!(prev_sid, at) };
            if dfa.is_special_state(sid) {
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
        if dfa.is_special_state(sid) {
            if dfa.is_start_state(sid) {
                if dfa.is_accel_state(sid) {
                    let needles = dfa.accelerator(sid);
                    at = accel::find_rev(needles, search.haystack(), at)
                        .map(|i| i + 1)
                        .unwrap_or(search.start());
                }
            } else if dfa.is_match_state(sid) {
                last_match = Some(HalfMatch {
                    pattern: dfa.match_pattern(sid, 0),
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
                if dfa.is_accel_state(sid) {
                    let needles = dfa.accelerator(sid);
                    at = accel::find_rev(needles, search.haystack(), at)
                        .map(|i| i + 1)
                        .unwrap_or(search.start());
                }
            } else if dfa.is_accel_state(sid) {
                let needles = dfa.accelerator(sid);
                // If the accelerator returns nothing, why don't we quit the
                // search? Well, if the accelerator doesn't find anything, that
                // doesn't mean we don't have a match. It just means that we
                // can't leave the current state given one of the 255 possible
                // byte values. However, there might be an EOI transition. So
                // we set 'at' to the end of the haystack, which will cause
                // this loop to stop and fall down into the EOI transition.
                at = accel::find_rev(needles, search.haystack(), at)
                    .map(|i| i + 1)
                    .unwrap_or(search.start());
            } else if dfa.is_dead_state(sid) {
                return Ok(last_match);
            } else {
                debug_assert!(dfa.is_quit_state(sid));
                if last_match.is_some() {
                    return Ok(last_match);
                }
                return Err(MatchError::Quit {
                    byte: search.haystack()[at],
                    offset: at,
                });
            }
        }
        if at == search.start() {
            break;
        }
        at -= 1;
    }
    Ok(eoi_rev(dfa, search, sid)?.or(last_match))
}

#[inline(never)]
pub fn find_overlapping_fwd<A: Automaton + ?Sized>(
    dfa: &A,
    pre: Option<&mut prefilter::Scanner>,
    search: &Search<'_>,
    state: &mut OverlappingState,
) -> Result<Option<HalfMatch>, MatchError> {
    if search.is_done() {
        return Ok(None);
    }
    // Searching with a pattern ID is always anchored, so we should only ever
    // use a prefilter when no pattern ID is given.
    if pre.is_some() && search.get_pattern().is_none() {
        find_overlapping_fwd_imp(dfa, pre, search, state)
    } else {
        find_overlapping_fwd_imp(dfa, None, search, state)
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
fn find_overlapping_fwd_imp<A: Automaton + ?Sized>(
    dfa: &A,
    mut pre: Option<&mut prefilter::Scanner>,
    search: &Search<'_>,
    state: &mut OverlappingState,
) -> Result<Option<HalfMatch>, MatchError> {
    let mut at = search.start();
    let mut sid = match state.id() {
        None => init_fwd(dfa, search)?,
        Some(sid) => {
            if let Some(last) = state.last_match() {
                let match_count = dfa.match_count(sid);
                if last.match_index < match_count {
                    let m = HalfMatch {
                        pattern: dfa.match_pattern(sid, last.match_index),
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
            // point, which is `MATCH_OFFSET` bytes PRIOR to where we scanned
            // to on the previous search. Therefore, we need to compensate by
            // bumping `start` up by `MATCH_OFFSET` bytes.
            //
            // Incidentally, since MATCH_OFFSET is non-zero, this also makes
            // dealing with empty matches convenient. Namely, callers needn't
            // special case them when implementing an iterator. Instead, this
            // ensures that forward progress is always made.
            at += MATCH_OFFSET;
            sid
        }
    };

    // NOTE: We don't optimize the crap out of this routine primarily because
    // it seems like most find_overlapping searches will have higher match
    // counts, and thus, throughput is perhaps not as important. But if you
    // have a use case for something faster, feel free to file an issue.
    while at < search.end() {
        sid = dfa.next_state(sid, search.haystack()[at]);
        if dfa.is_special_state(sid) {
            state.set_id(sid);
            if dfa.is_start_state(sid) {
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
                } else if dfa.is_accel_state(sid) {
                    let needles = dfa.accelerator(sid);
                    at = accel::find_fwd(needles, search.haystack(), at)
                        .unwrap_or(search.end());
                    continue;
                }
            } else if dfa.is_match_state(sid) {
                state
                    .set_last_match(StateMatch { match_index: 1, offset: at });
                return Ok(Some(HalfMatch {
                    pattern: dfa.match_pattern(sid, 0),
                    offset: at,
                }));
            } else if dfa.is_accel_state(sid) {
                let needs = dfa.accelerator(sid);
                // If the accelerator returns nothing, why don't we quit the
                // search? Well, if the accelerator doesn't find anything, that
                // doesn't mean we don't have a match. It just means that we
                // can't leave the current state given one of the 255 possible
                // byte values. However, there might be an EOI transition. So
                // we set 'at' to the end of the haystack, which will cause
                // this loop to stop and fall down into the EOI transition.
                at = accel::find_fwd(needs, search.haystack(), at)
                    .unwrap_or(search.end());
                continue;
            } else if dfa.is_dead_state(sid) {
                return Ok(None);
            } else {
                debug_assert!(dfa.is_quit_state(sid));
                return Err(MatchError::Quit {
                    byte: search.haystack()[at],
                    offset: at,
                });
            }
        }
        at += 1;
    }

    let result = eoi_fwd(dfa, search, &mut sid);
    state.set_id(sid);
    if let Ok(Some(ref last_match)) = result {
        state.set_last_match(StateMatch {
            match_index: 1,
            offset: last_match.offset(),
        });
    }
    result
}

#[inline(always)]
fn init_fwd<A: Automaton + ?Sized>(
    dfa: &A,
    search: &Search<'_>,
) -> Result<StateID, MatchError> {
    let state = dfa.start_state_forward(search);
    // Start states can never be match states, since all matches are delayed
    // by 1 byte.
    assert!(!dfa.is_match_state(state));
    Ok(state)
}

#[inline(always)]
fn init_rev<A: Automaton + ?Sized>(
    dfa: &A,
    search: &Search<'_>,
) -> Result<StateID, MatchError> {
    let state = dfa.start_state_reverse(search);
    // Start states can never be match states, since all matches are delayed
    // by 1 byte.
    assert!(!dfa.is_match_state(state));
    Ok(state)
}

#[inline(always)]
fn eoi_fwd<A: Automaton + ?Sized>(
    dfa: &A,
    search: &Search<'_>,
    sid: &mut StateID,
) -> Result<Option<HalfMatch>, MatchError> {
    match search.haystack().get(search.end()) {
        Some(&b) => {
            *sid = dfa.next_state(*sid, b);
            if dfa.is_match_state(*sid) {
                Ok(Some(HalfMatch {
                    pattern: dfa.match_pattern(*sid, 0),
                    offset: search.end(),
                }))
            } else {
                Ok(None)
            }
        }
        None => {
            *sid = dfa.next_eoi_state(*sid);
            if dfa.is_match_state(*sid) {
                Ok(Some(HalfMatch {
                    pattern: dfa.match_pattern(*sid, 0),
                    offset: search.end(),
                }))
            } else {
                Ok(None)
            }
        }
    }
}

#[inline(always)]
fn eoi_rev<A: Automaton + ?Sized>(
    dfa: &A,
    search: &Search<'_>,
    sid: StateID,
) -> Result<Option<HalfMatch>, MatchError> {
    if search.start() > 0 {
        let sid = dfa.next_state(sid, search.haystack()[search.start() - 1]);
        if dfa.is_match_state(sid) {
            Ok(Some(HalfMatch {
                pattern: dfa.match_pattern(sid, 0),
                offset: search.start(),
            }))
        } else {
            Ok(None)
        }
    } else {
        let sid = dfa.next_eoi_state(sid);
        if dfa.is_match_state(sid) {
            Ok(Some(HalfMatch {
                pattern: dfa.match_pattern(sid, 0),
                offset: 0,
            }))
        } else {
            Ok(None)
        }
    }
}
