use crate::{
    dfa::HalfMatch,
    hybrid::{id::LazyStateID, lazy::DFA},
    nfa::thompson,
    util::{id::PatternID, matchtypes::MatchError},
};

// TODO: Unify this with same constant in crate::dfa::automaton.
const MATCH_OFFSET: usize = 1;

#[inline(always)]
fn find_fwd<'a>(
    earliest: bool,
    dfa: &mut DFA<'a, 'a, &'a thompson::NFA>,
    pattern_id: Option<PatternID>,
    bytes: &[u8],
    start: usize,
    end: usize,
) -> Result<Option<HalfMatch>, MatchError> {
    assert!(start <= end);
    assert!(start <= bytes.len());
    assert!(end <= bytes.len());

    let mut sid = init_fwd(dfa, pattern_id, bytes, start, end)?;
    if !sid.is_unmasked() {
        debug_assert!(sid.is_dead());
        return Ok(None);
    }
    let mut last_match = None;
    let mut at = start;
    while at < end {
        let byte = bytes[at];
        sid = dfa.next_state(sid, byte).map_err(|_| gave_up(at))?;
        at += 1;
        if !sid.is_unmasked() {
            if sid.is_match() {
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
        while at < end
            && dfa.next_state(sid, bytes[at]).map_err(|_| gave_up(at))? == sid
        {
            at += 1;
        }
    }
    Ok(eoi_fwd(dfa, bytes, end, &mut sid)?.or(last_match))
}

fn init_fwd<'a>(
    dfa: &mut DFA<'a, 'a, &'a thompson::NFA>,
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

fn eoi_fwd<'a>(
    dfa: &mut DFA<'a, 'a, &'a thompson::NFA>,
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

/// A convenience routine for constructing a "gave up" match error.
fn gave_up(offset: usize) -> MatchError {
    MatchError::GaveUp { offset }
}
