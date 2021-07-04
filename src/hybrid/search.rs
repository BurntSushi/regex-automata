use crate::{
    dfa::HalfMatch,
    hybrid::{id::LazyStateID, lazy::DFA},
    nfa::thompson,
    util::{id::PatternID, matchtypes::MatchError},
};

// TODO: Unify this with same constant in crate::dfa::automaton.
const MATCH_OFFSET: usize = 1;

#[inline(always)]
pub(crate) fn find_fwd<'i, 'c>(
    earliest: bool,
    dfa: &mut DFA<'i, 'c>,
    pattern_id: Option<PatternID>,
    bytes: &[u8],
    start: usize,
    end: usize,
) -> Result<Option<HalfMatch>, MatchError> {
    assert!(start <= end);
    assert!(start <= bytes.len());
    assert!(end <= bytes.len());

    let mut sid = init_fwd(dfa, pattern_id, bytes, start, end)?;
    let mut last_match = None;
    let mut at = start;
    while at < end {
        let byte = bytes[at];
        sid = dfa.next_state(sid, byte).map_err(|_| gave_up(at))?;
        at += 1;
        if !sid.is_unmasked() {
            if sid.is_start() {
                continue;
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
            } else {
                debug_assert!(sid.is_quit());
                if last_match.is_some() {
                    return Ok(last_match);
                }
                return Err(MatchError::Quit { byte, offset: at - 1 });
            }
        }
    }
    Ok(eoi_fwd(dfa, bytes, end, &mut sid)?.or(last_match))
}

#[inline(always)]
pub(crate) fn find_rev<'i, 'c>(
    earliest: bool,
    dfa: &mut DFA<'i, 'c>,
    pattern_id: Option<PatternID>,
    bytes: &[u8],
    start: usize,
    end: usize,
) -> Result<Option<HalfMatch>, MatchError> {
    assert!(start <= end);
    assert!(start <= bytes.len());
    assert!(end <= bytes.len());

    let mut sid = init_rev(dfa, pattern_id, bytes, start, end)?;
    let mut last_match = None;
    let mut at = end;
    while at > start {
        at -= 1;
        let byte = bytes[at];
        sid = dfa.next_state(sid, byte).map_err(|_| gave_up(at))?;
        if !sid.is_unmasked() {
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

fn init_fwd<'i, 'c>(
    dfa: &mut DFA<'i, 'c>,
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

fn init_rev<'i, 'c>(
    dfa: &mut DFA<'i, 'c>,
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

fn eoi_fwd<'i, 'c>(
    dfa: &mut DFA<'i, 'c>,
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

fn eoi_rev<'i, 'c>(
    dfa: &mut DFA<'i, 'c>,
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
