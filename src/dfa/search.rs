use crate::dfa::accel;
use crate::dfa::automaton::{Automaton, HalfMatch, State, StateMatch};
use crate::prefilter::{self, Prefilter};
use crate::NoMatch;

#[inline(never)]
pub fn find_earliest_fwd<A: Automaton + ?Sized>(
    mut pre: Option<&mut prefilter::Scanner>,
    dfa: &A,
    bytes: &[u8],
    start: usize,
    end: usize,
) -> Result<Option<HalfMatch>, NoMatch> {
    if pre.is_some() {
        find_fwd(pre, true, dfa, bytes, start, end)
    } else {
        find_fwd(None, true, dfa, bytes, start, end)
    }
}

#[inline(never)]
pub fn find_leftmost_fwd<A: Automaton + ?Sized>(
    mut pre: Option<&mut prefilter::Scanner>,
    dfa: &A,
    bytes: &[u8],
    start: usize,
    end: usize,
) -> Result<Option<HalfMatch>, NoMatch> {
    if pre.is_some() {
        find_fwd(pre, false, dfa, bytes, start, end)
    } else {
        find_fwd(None, false, dfa, bytes, start, end)
    }
}

/// This is marked as `inline(always)` specifically because it supports
/// multiple modes of searching. Namely, the 'pre' and 'earliest' parameters
/// getting inlined eliminate some critical branches. To avoid bloating binary
/// size, we only call this function in a fixed number of places.
#[inline(always)]
fn find_fwd<A: Automaton + ?Sized>(
    mut pre: Option<&mut prefilter::Scanner>,
    earliest: bool,
    dfa: &A,
    bytes: &[u8],
    start: usize,
    end: usize,
) -> Result<Option<HalfMatch>, NoMatch> {
    assert!(start <= end);
    assert!(start <= bytes.len());
    assert!(end <= bytes.len());

    let (mut state, mut last_match) = init_fwd(dfa, bytes, start, end)?;
    if earliest && last_match.is_some() {
        return Ok(last_match);
    }

    let mut at = start;
    if let Some(ref mut pre) = pre {
        if !pre.reports_false_positives() {
            return Ok(pre.next_candidate(bytes, at).into_option().map(
                |offset| HalfMatch {
                    pattern: dfa.match_pattern(state, 0),
                    offset,
                },
            ));
        } else if pre.is_effective(at) {
            match pre.next_candidate(bytes, at).into_option() {
                None => return Ok(None),
                Some(i) => {
                    at = i;
                }
            }
        }
    }
    while at < end {
        let byte = bytes[at];
        state = dfa.next_state(state, byte);
        at += 1;
        if dfa.is_special_state(state) {
            if dfa.is_start_state(state) {
                if let Some(ref mut pre) = pre {
                    if pre.is_effective(at) {
                        match pre.next_candidate(bytes, at).into_option() {
                            None => return Ok(None),
                            Some(i) => {
                                at = i;
                            }
                        }
                    }
                } else if dfa.is_accel_state(state) {
                    let needles = dfa.accelerator(state);
                    if !needles.is_empty() {
                        at = accel::find_fwd(needles, bytes, at)
                            .unwrap_or(bytes.len());
                    }
                }
            } else if dfa.is_match_state(state) {
                last_match = Some(HalfMatch {
                    pattern: dfa.match_pattern(state, 0),
                    offset: at - dfa.match_offset(),
                });
                if earliest {
                    return Ok(last_match);
                }
                if dfa.is_accel_state(state) {
                    let needles = dfa.accelerator(state);
                    if !needles.is_empty() {
                        at = accel::find_fwd(needles, bytes, at)
                            .unwrap_or(bytes.len());
                    }
                }
            } else if dfa.is_accel_state(state) {
                let needs = dfa.accelerator(state);
                at = accel::find_fwd(needs, bytes, at).unwrap_or(bytes.len());
            } else if dfa.is_dead_state(state) {
                return Ok(last_match);
            } else {
                debug_assert!(dfa.is_quit_state(state));
                return Err(NoMatch::Quit { byte, offset: at - 1 });
            }
        }
        while at < end && dfa.next_state(state, bytes[at]) == state {
            at += 1;
        }
    }
    Ok(eof_fwd(dfa, bytes, end, &mut state)?.or(last_match))
}

#[inline(never)]
pub fn find_earliest_rev<A: Automaton + ?Sized>(
    dfa: &A,
    bytes: &[u8],
    start: usize,
    end: usize,
) -> Result<Option<HalfMatch>, NoMatch> {
    find_rev(true, dfa, bytes, start, end)
}

#[inline(never)]
pub fn find_leftmost_rev<A: Automaton + ?Sized>(
    dfa: &A,
    bytes: &[u8],
    start: usize,
    end: usize,
) -> Result<Option<HalfMatch>, NoMatch> {
    find_rev(false, dfa, bytes, start, end)
}

/// This is marked as `inline(always)` specifically because it supports
/// multiple modes of searching. Namely, the 'earliest' boolean getting inlined
/// permits eliminating a few crucial branches.
#[inline(always)]
fn find_rev<A: Automaton + ?Sized>(
    earliest: bool,
    dfa: &A,
    bytes: &[u8],
    start: usize,
    end: usize,
) -> Result<Option<HalfMatch>, NoMatch> {
    assert!(start <= end);
    assert!(start <= bytes.len());
    assert!(end <= bytes.len());

    let (mut state, mut last_match) = init_rev(dfa, bytes, start, end)?;
    if earliest && last_match.is_some() {
        return Ok(last_match);
    }

    let mut at = end;
    while at > start {
        at -= 1;
        while at > start && dfa.next_state(state, bytes[at]) == state {
            at -= 1;
        }

        let byte = bytes[at];
        state = dfa.next_state(state, byte);
        if dfa.is_special_state(state) {
            if dfa.is_start_state(state) {
                if dfa.is_accel_state(state) {
                    let needles = dfa.accelerator(state);
                    if !needles.is_empty() {
                        at = accel::find_rev(needles, bytes, at)
                            .map(|i| i + 1)
                            .unwrap_or(0);
                    }
                }
            } else if dfa.is_match_state(state) {
                last_match = Some(HalfMatch {
                    pattern: dfa.match_pattern(state, 0),
                    offset: at + dfa.match_offset(),
                });
                if earliest {
                    return Ok(last_match);
                }
                if dfa.is_accel_state(state) {
                    let needles = dfa.accelerator(state);
                    if !needles.is_empty() {
                        at = accel::find_rev(needles, bytes, at)
                            .map(|i| i + 1)
                            .unwrap_or(0);
                    }
                }
            } else if dfa.is_accel_state(state) {
                let needles = dfa.accelerator(state);
                at = accel::find_rev(needles, bytes, at)
                    .map(|i| i + 1)
                    .unwrap_or(0);
            } else if dfa.is_dead_state(state) {
                return Ok(last_match);
            } else {
                debug_assert!(dfa.is_quit_state(state));
                return Err(NoMatch::Quit { byte, offset: at });
            }
        }
    }
    Ok(eof_rev(dfa, state, bytes, start)?.or(last_match))
}

#[inline(never)]
pub fn find_overlapping_fwd<A: Automaton + ?Sized>(
    mut pre: Option<&mut prefilter::Scanner>,
    dfa: &A,
    bytes: &[u8],
    mut start: usize,
    end: usize,
    caller_state: &mut State<A::ID>,
) -> Result<Option<HalfMatch>, NoMatch> {
    if pre.is_some() {
        find_overlapping_fwd_imp(pre, dfa, bytes, start, end, caller_state)
    } else {
        find_overlapping_fwd_imp(None, dfa, bytes, start, end, caller_state)
    }
}

#[inline(always)]
fn find_overlapping_fwd_imp<A: Automaton + ?Sized>(
    mut pre: Option<&mut prefilter::Scanner>,
    dfa: &A,
    bytes: &[u8],
    mut start: usize,
    end: usize,
    caller_state: &mut State<A::ID>,
) -> Result<Option<HalfMatch>, NoMatch> {
    assert!(start <= end);
    assert!(start <= bytes.len());
    assert!(end <= bytes.len());

    let (mut state, mut last_match) = match caller_state.id() {
        None => init_fwd(dfa, bytes, start, end)?,
        Some(id) => {
            if let Some(last) = caller_state.last_match() {
                let match_count = dfa.match_count(id);
                if last.match_index < match_count {
                    let m = HalfMatch {
                        pattern: dfa.match_pattern(id, last.match_index),
                        offset: last.offset,
                    };
                    last.match_index += 1;
                    return Ok(Some(m));
                }
                caller_state.clear_last_match();
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
            // compensate by bumping `start` up by `match_offset` bytes.
            start += dfa.match_offset();
            // Since match_offset could be any arbitrary value, it's possible
            // that we are at EOF. So check that now.
            if start > end {
                return Ok(None);
            }
            (id, None)
        }
    };
    if let Some(last_match) = last_match {
        caller_state.set_id(state);
        caller_state.set_last_match(StateMatch {
            match_index: 1,
            offset: last_match.offset(),
        });
        return Ok(Some(last_match));
    }
    caller_state.clear_last_match();

    let mut at = start;
    while at < end {
        let byte = bytes[at];
        state = dfa.next_state(state, byte);
        at += 1;
        if dfa.is_special_state(state) {
            caller_state.set_id(state);
            if dfa.is_start_state(state) {
                if let Some(ref mut pre) = pre {
                    if pre.is_effective(at) {
                        match pre.next_candidate(bytes, at).into_option() {
                            None => return Ok(None),
                            Some(i) => {
                                at = i;
                            }
                        }
                    }
                } else if dfa.is_accel_state(state) {
                    let needles = dfa.accelerator(state);
                    if !needles.is_empty() {
                        at = accel::find_fwd(needles, bytes, at)
                            .unwrap_or(bytes.len());
                    }
                }
            } else if dfa.is_match_state(state) {
                let offset = at - dfa.match_offset();
                caller_state
                    .set_last_match(StateMatch { match_index: 1, offset });
                return Ok(Some(HalfMatch {
                    pattern: dfa.match_pattern(state, 0),
                    offset,
                }));
            } else if dfa.is_accel_state(state) {
                let needs = dfa.accelerator(state);
                at = accel::find_fwd(needs, bytes, at).unwrap_or(bytes.len());
            } else if dfa.is_dead_state(state) {
                return Ok(None);
            } else {
                debug_assert!(dfa.is_quit_state(state));
                return Err(NoMatch::Quit { byte, offset: at - 1 });
            }
        }
    }

    let result = eof_fwd(dfa, bytes, end, &mut state);
    caller_state.set_id(state);
    if let Ok(Some(ref last_match)) = result {
        caller_state.set_last_match(StateMatch {
            match_index: 1,
            offset: last_match.offset(),
        });
    }
    result
}

fn init_fwd<A: Automaton + ?Sized>(
    dfa: &A,
    bytes: &[u8],
    start: usize,
    end: usize,
) -> Result<(A::ID, Option<HalfMatch>), NoMatch> {
    let state = dfa.start_state_forward(bytes, start, end);
    if dfa.is_match_state(state) {
        let m = HalfMatch {
            pattern: dfa.match_pattern(state, 0),
            offset: start - dfa.match_offset(),
        };
        Ok((state, Some(m)))
    } else {
        Ok((state, None))
    }
}

fn init_rev<A: Automaton + ?Sized>(
    dfa: &A,
    bytes: &[u8],
    start: usize,
    end: usize,
) -> Result<(A::ID, Option<HalfMatch>), NoMatch> {
    let state = dfa.start_state_reverse(bytes, start, end);
    if dfa.is_match_state(state) {
        let m = HalfMatch {
            pattern: dfa.match_pattern(state, 0),
            offset: end + dfa.match_offset(),
        };
        Ok((state, Some(m)))
    } else {
        Ok((state, None))
    }
}

fn eof_fwd<A: Automaton + ?Sized>(
    dfa: &A,
    bytes: &[u8],
    end: usize,
    state: &mut A::ID,
) -> Result<Option<HalfMatch>, NoMatch> {
    match bytes.get(end) {
        Some(&b) => {
            *state = dfa.next_state(*state, b);
            if dfa.is_match_state(*state) {
                Ok(Some(HalfMatch {
                    pattern: dfa.match_pattern(*state, 0),
                    offset: end,
                }))
            } else {
                Ok(None)
            }
        }
        None => {
            *state = dfa.next_eof_state(*state);
            if dfa.is_match_state(*state) {
                Ok(Some(HalfMatch {
                    pattern: dfa.match_pattern(*state, 0),
                    offset: bytes.len(),
                }))
            } else {
                Ok(None)
            }
        }
    }
}

fn eof_rev<A: Automaton + ?Sized>(
    dfa: &A,
    state: A::ID,
    bytes: &[u8],
    start: usize,
) -> Result<Option<HalfMatch>, NoMatch> {
    if start > 0 {
        let state = dfa.next_state(state, bytes[start - 1]);
        if dfa.is_match_state(state) {
            Ok(Some(HalfMatch {
                pattern: dfa.match_pattern(state, 0),
                offset: start,
            }))
        } else {
            Ok(None)
        }
    } else {
        let state = dfa.next_eof_state(state);
        if dfa.is_match_state(state) {
            Ok(Some(HalfMatch {
                pattern: dfa.match_pattern(state, 0),
                offset: 0,
            }))
        } else {
            Ok(None)
        }
    }
}

/// Returns the distance between the given pointer and the start of `bytes`.
/// This assumes that the given pointer points to somewhere in the `bytes`
/// slice given.
fn offset(bytes: &[u8], p: *const u8) -> usize {
    debug_assert!(bytes.as_ptr() <= p);
    debug_assert!(bytes[bytes.len()..].as_ptr() >= p);
    ((p as isize) - (bytes.as_ptr() as isize)) as usize
}
