use crate::{
    hybrid::{
        id::{LazyStateID, OverlappingState, StateMatch},
        lazy::DFA,
    },
    nfa::thompson,
    util::{
        id::PatternID,
        matchtypes::{HalfMatch, MatchError},
        prefilter, MATCH_OFFSET,
    },
};

#[inline(never)]
pub(crate) fn find_earliest_fwd(
    pre: Option<&mut prefilter::Scanner>,
    mut dfa: DFA<'_, '_>,
    pattern_id: Option<PatternID>,
    bytes: &[u8],
    start: usize,
    end: usize,
) -> Result<Option<HalfMatch>, MatchError> {
    // Searching with a pattern ID is always anchored, so we should never use
    // a prefilter.
    if pre.is_some() && pattern_id.is_none() {
        find_fwd(pre, true, dfa, pattern_id, bytes, start, end)
    } else {
        find_fwd(None, true, dfa, pattern_id, bytes, start, end)
    }
}

#[inline(never)]
pub(crate) fn find_leftmost_fwd(
    pre: Option<&mut prefilter::Scanner>,
    mut dfa: DFA<'_, '_>,
    pattern_id: Option<PatternID>,
    bytes: &[u8],
    start: usize,
    end: usize,
) -> Result<Option<HalfMatch>, MatchError> {
    // Searching with a pattern ID is always anchored, so we should never use
    // a prefilter.
    if pre.is_some() && pattern_id.is_none() {
        find_fwd(pre, false, dfa, pattern_id, bytes, start, end)
    } else {
        find_fwd(None, false, dfa, pattern_id, bytes, start, end)
    }
}

#[inline(always)]
fn find_fwd(
    mut pre: Option<&mut prefilter::Scanner>,
    earliest: bool,
    mut dfa: DFA<'_, '_>,
    pattern_id: Option<PatternID>,
    haystack: &[u8],
    start: usize,
    end: usize,
) -> Result<Option<HalfMatch>, MatchError> {
    assert!(start <= end);
    assert!(start <= haystack.len());
    assert!(end <= haystack.len());

    // Why do this? This lets 'scoped_bytes[at]' work without bounds checks
    // below. It seems the assert on 'end <= haystack.len()' above is otherwise
    // not enough. Why not just make 'bytes' scoped this way anyway? Well,
    // 'eoi_fwd' (below) might actually want to try to access the byte at 'end'
    // for resolving look-ahead.
    let bytes = &haystack[..end];

    let mut sid =
        init_fwd(dfa.as_ref_mut(), pattern_id, haystack, start, end)?;
    let mut last_match = None;
    let mut at = start;
    while at < end {
        if !sid.is_tagged() {
            let mut prev_sid = sid;
            while at < end && !sid.is_tagged() {
                prev_sid = sid;
                sid = dfa.next_state_untagged(sid, bytes[at]);
                at += 1;
            }
            if sid.is_unknown() {
                sid = dfa
                    .next_state(prev_sid, bytes[at - 1])
                    .map_err(|_| gave_up(at - 1))?;
            }
        } else {
            sid = dfa.next_state(sid, bytes[at]).map_err(|_| gave_up(at))?;
            at += 1;
        }
        if sid.is_tagged() {
            if sid.is_start() {
                if let Some(ref mut pre) = pre {
                    if pre.is_effective(at) {
                        match pre.next_candidate(bytes, at).into_option() {
                            None => return Ok(None),
                            Some(i) => {
                                at = i;
                            }
                        }
                    }
                }
            } else if sid.is_match() {
                last_match = Some(HalfMatch {
                    pattern: dfa.match_pattern(sid, 0),
                    offset: at - MATCH_OFFSET,
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
                let offset = at - 1;
                return Err(MatchError::Quit { byte: bytes[offset], offset });
            } else {
                debug_assert!(sid.is_unknown());
                unreachable!("sid being unknown is a bug");
            }
        }
    }
    // We are careful to use 'haystack' here, which contains the full context
    // that we might want to inspect.
    Ok(eoi_fwd(dfa, haystack, end, &mut sid)?.or(last_match))
}

#[inline(never)]
pub(crate) fn find_earliest_rev(
    mut dfa: DFA<'_, '_>,
    pattern_id: Option<PatternID>,
    bytes: &[u8],
    start: usize,
    end: usize,
) -> Result<Option<HalfMatch>, MatchError> {
    find_rev(true, dfa, pattern_id, bytes, start, end)
}

#[inline(never)]
pub(crate) fn find_leftmost_rev(
    mut dfa: DFA<'_, '_>,
    pattern_id: Option<PatternID>,
    bytes: &[u8],
    start: usize,
    end: usize,
) -> Result<Option<HalfMatch>, MatchError> {
    find_rev(false, dfa, pattern_id, bytes, start, end)
}

#[inline(always)]
fn find_rev(
    earliest: bool,
    mut dfa: DFA<'_, '_>,
    pattern_id: Option<PatternID>,
    bytes: &[u8],
    start: usize,
    end: usize,
) -> Result<Option<HalfMatch>, MatchError> {
    assert!(start <= end);
    assert!(start <= bytes.len());
    assert!(end <= bytes.len());

    let mut sid = init_rev(dfa.as_ref_mut(), pattern_id, bytes, start, end)?;
    let mut last_match = None;
    let mut at = end;
    while at > start {
        at -= 1;
        let byte = bytes[at];
        sid = dfa.next_state(sid, byte).map_err(|_| gave_up(at))?;
        if sid.is_tagged() {
            if sid.is_start() {
                continue;
            } else if sid.is_match() {
                last_match = Some(HalfMatch {
                    pattern: dfa.match_pattern(sid, 0),
                    offset: at + MATCH_OFFSET,
                });
                if earliest {
                    return Ok(last_match);
                }
            } else if sid.is_dead() {
                return Ok(last_match);
            } else {
                debug_assert!(sid.is_quit());
                if last_match.is_some() {
                    return Ok(last_match);
                }
                return Err(MatchError::Quit { byte, offset: at });
            }
        }
    }
    Ok(eoi_rev(dfa, bytes, start, sid)?.or(last_match))
}

#[inline(never)]
pub(crate) fn find_overlapping_fwd(
    pre: Option<&mut prefilter::Scanner>,
    mut dfa: DFA<'_, '_>,
    pattern_id: Option<PatternID>,
    bytes: &[u8],
    start: usize,
    end: usize,
    caller_state: &mut OverlappingState,
) -> Result<Option<HalfMatch>, MatchError> {
    // Searching with a pattern ID is always anchored, so we should only ever
    // use a prefilter when no pattern ID is given.
    if pre.is_some() && pattern_id.is_none() {
        find_overlapping_fwd_imp(
            pre,
            dfa,
            pattern_id,
            bytes,
            start,
            end,
            caller_state,
        )
    } else {
        find_overlapping_fwd_imp(
            None,
            dfa,
            pattern_id,
            bytes,
            start,
            end,
            caller_state,
        )
    }
}

#[inline(always)]
fn find_overlapping_fwd_imp(
    mut pre: Option<&mut prefilter::Scanner>,
    mut dfa: DFA<'_, '_>,
    pattern_id: Option<PatternID>,
    bytes: &[u8],
    mut start: usize,
    end: usize,
    caller_state: &mut OverlappingState,
) -> Result<Option<HalfMatch>, MatchError> {
    assert!(start <= end);
    assert!(start <= bytes.len());
    assert!(end <= bytes.len());

    let mut sid = match caller_state.id() {
        None => init_fwd(dfa.as_ref_mut(), pattern_id, bytes, start, end)?,
        Some(sid) => {
            if let Some(last) = caller_state.last_match() {
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
            // Therefore, the only case we need to consider is when
            // caller_state is a match state. In this case, since our machines
            // support the ability to delay a match by a certain number of
            // bytes (to support look-around), it follows that we actually
            // consumed that many additional bytes on our previous search. When
            // the caller resumes their search to find subsequent matches, they
            // will use the ending location from the previous match as the next
            // starting point, which is `match_offset` bytes PRIOR to where
            // we scanned to on the previous search. Therefore, we need to
            // compensate by bumping `start` up by `MATCH_OFFSET` bytes.
            //
            // Incidentally, since MATCH_OFFSET is non-zero, this also makes
            // dealing with empty matches convenient. Namely, callers needn't
            // special case them when implementing an iterator. Instead, this
            // ensures that forward progress is always made.
            start += MATCH_OFFSET;
            sid
        }
    };

    let mut at = start;
    while at < end {
        let byte = bytes[at];
        sid = dfa.next_state(sid, byte).map_err(|_| gave_up(at))?;
        at += 1;
        if sid.is_tagged() {
            caller_state.set_id(sid);
            if sid.is_start() {
                if let Some(ref mut pre) = pre {
                    if pre.is_effective(at) {
                        match pre.next_candidate(bytes, at).into_option() {
                            None => return Ok(None),
                            Some(i) => {
                                at = i;
                            }
                        }
                    }
                }
            } else if sid.is_match() {
                let offset = at - MATCH_OFFSET;
                caller_state
                    .set_last_match(StateMatch { match_index: 1, offset });
                return Ok(Some(HalfMatch {
                    pattern: dfa.match_pattern(sid, 0),
                    offset,
                }));
            } else if sid.is_dead() {
                return Ok(None);
            } else {
                debug_assert!(sid.is_quit());
                return Err(MatchError::Quit { byte, offset: at - 1 });
            }
        }
    }

    let result = eoi_fwd(dfa, bytes, end, &mut sid);
    caller_state.set_id(sid);
    if let Ok(Some(ref last_match)) = result {
        caller_state.set_last_match(StateMatch {
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

fn init_fwd(
    mut dfa: DFA<'_, '_>,
    pattern_id: Option<PatternID>,
    bytes: &[u8],
    start: usize,
    end: usize,
) -> Result<LazyStateID, MatchError> {
    let sid = dfa
        .start_state_forward(pattern_id, bytes, start, end)
        .map_err(|_| gave_up(start))?;
    // Start states can never be match states, since all matches are delayed
    // by 1 byte.
    assert!(!sid.is_match());
    Ok(sid)
}

fn init_rev(
    mut dfa: DFA<'_, '_>,
    pattern_id: Option<PatternID>,
    bytes: &[u8],
    start: usize,
    end: usize,
) -> Result<LazyStateID, MatchError> {
    let sid = dfa
        .start_state_reverse(pattern_id, bytes, start, end)
        .map_err(|_| gave_up(end))?;
    // Start states can never be match states, since all matches are delayed
    // by 1 byte.
    assert!(!sid.is_match());
    Ok(sid)
}

fn eoi_fwd(
    mut dfa: DFA<'_, '_>,
    bytes: &[u8],
    end: usize,
    sid: &mut LazyStateID,
) -> Result<Option<HalfMatch>, MatchError> {
    match bytes.get(end) {
        Some(&b) => {
            *sid = dfa.next_state(*sid, b).map_err(|_| gave_up(end))?;
            if sid.is_match() {
                Ok(Some(HalfMatch {
                    pattern: dfa.match_pattern(*sid, 0),
                    offset: end,
                }))
            } else {
                Ok(None)
            }
        }
        None => {
            *sid =
                dfa.next_eoi_state(*sid).map_err(|_| gave_up(bytes.len()))?;
            if sid.is_match() {
                Ok(Some(HalfMatch {
                    pattern: dfa.match_pattern(*sid, 0),
                    offset: bytes.len(),
                }))
            } else {
                Ok(None)
            }
        }
    }
}

fn eoi_rev(
    mut dfa: DFA<'_, '_>,
    bytes: &[u8],
    start: usize,
    state: LazyStateID,
) -> Result<Option<HalfMatch>, MatchError> {
    if start > 0 {
        let sid = dfa
            .next_state(state, bytes[start - 1])
            .map_err(|_| gave_up(start))?;
        if sid.is_match() {
            Ok(Some(HalfMatch {
                pattern: dfa.match_pattern(sid, 0),
                offset: start,
            }))
        } else {
            Ok(None)
        }
    } else {
        let sid = dfa.next_eoi_state(state).map_err(|_| gave_up(start))?;
        if sid.is_match() {
            Ok(Some(HalfMatch {
                pattern: dfa.match_pattern(sid, 0),
                offset: 0,
            }))
        } else {
            Ok(None)
        }
    }
}

/// A convenience routine for constructing a "gave up" match error.
fn gave_up(offset: usize) -> MatchError {
    MatchError::GaveUp { offset }
}
