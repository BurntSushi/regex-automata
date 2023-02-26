use crate::{meta::error::RetryFailError, HalfMatch, Input, MatchError};

#[cfg(feature = "dfa-build")]
pub(crate) fn dfa_try_search_half_fwd(
    dfa: &crate::dfa::dense::DFA<alloc::vec::Vec<u32>>,
    input: &Input<'_>,
) -> Result<Result<HalfMatch, usize>, RetryFailError> {
    use crate::dfa::{accel, Automaton};

    let mut mat = None;
    let mut sid = match dfa.start_state_forward(input)? {
        None => return Ok(Err(input.start())),
        Some(sid) => sid,
    };
    let mut at = input.start();
    while at < input.end() {
        sid = dfa.next_state(sid, input.haystack()[at]);
        if dfa.is_special_state(sid) {
            if dfa.is_match_state(sid) {
                let pattern = dfa.match_pattern(sid, 0);
                mat = Some(HalfMatch::new(pattern, at));
                if input.get_earliest() {
                    return Ok(mat.ok_or(at));
                }
                if dfa.is_accel_state(sid) {
                    let needs = dfa.accelerator(sid);
                    at = accel::find_fwd(needs, input.haystack(), at)
                        .unwrap_or(input.end());
                    continue;
                }
            } else if dfa.is_accel_state(sid) {
                let needs = dfa.accelerator(sid);
                at = accel::find_fwd(needs, input.haystack(), at)
                    .unwrap_or(input.end());
                continue;
            } else if dfa.is_dead_state(sid) {
                return Ok(mat.ok_or(at));
            } else if dfa.is_quit_state(sid) {
                if mat.is_some() {
                    return Ok(mat.ok_or(at));
                }
                return Err(MatchError::quit(input.haystack()[at], at).into());
            }
        }
        at += 1;
    }
    dfa_eoi_fwd(dfa, input, &mut sid, &mut mat)?;
    Ok(mat.ok_or(at))
}

#[cfg(feature = "hybrid")]
pub(crate) fn hybrid_try_search_half_fwd(
    dfa: &crate::hybrid::dfa::DFA,
    cache: &mut crate::hybrid::dfa::Cache,
    input: &Input<'_>,
) -> Result<Result<HalfMatch, usize>, RetryFailError> {
    let mut mat = None;
    let mut sid = match dfa.start_state_forward(cache, input)? {
        None => return Ok(Err(input.start())),
        Some(sid) => sid,
    };
    let mut at = input.start();
    while at < input.end() {
        sid = dfa
            .next_state(cache, sid, input.haystack()[at])
            .map_err(|_| MatchError::gave_up(at))?;
        if sid.is_tagged() {
            if sid.is_match() {
                let pattern = dfa.match_pattern(cache, sid, 0);
                mat = Some(HalfMatch::new(pattern, at));
                if input.get_earliest() {
                    return Ok(mat.ok_or(at));
                }
            } else if sid.is_dead() {
                return Ok(mat.ok_or(at));
            } else if sid.is_quit() {
                if mat.is_some() {
                    return Ok(mat.ok_or(at));
                }
                return Err(MatchError::quit(input.haystack()[at], at).into());
            } else {
                debug_assert!(sid.is_unknown());
                unreachable!("sid being unknown is a bug");
            }
        }
        at += 1;
    }
    hybrid_eoi_fwd(dfa, cache, input, &mut sid, &mut mat)?;
    Ok(mat.ok_or(at))
}

#[cfg(feature = "dfa-build")]
#[inline(always)]
fn dfa_eoi_fwd(
    dfa: &crate::dfa::dense::DFA<alloc::vec::Vec<u32>>,
    input: &Input<'_>,
    sid: &mut crate::util::primitives::StateID,
    mat: &mut Option<HalfMatch>,
) -> Result<(), MatchError> {
    use crate::dfa::Automaton;

    let sp = input.get_span();
    match input.haystack().get(sp.end) {
        Some(&b) => {
            *sid = dfa.next_state(*sid, b);
            if dfa.is_match_state(*sid) {
                let pattern = dfa.match_pattern(*sid, 0);
                *mat = Some(HalfMatch::new(pattern, sp.end));
            } else if dfa.is_quit_state(*sid) {
                if mat.is_some() {
                    return Ok(());
                }
                return Err(MatchError::quit(b, sp.end));
            }
        }
        None => {
            *sid = dfa.next_eoi_state(*sid);
            if dfa.is_match_state(*sid) {
                let pattern = dfa.match_pattern(*sid, 0);
                *mat = Some(HalfMatch::new(pattern, input.haystack().len()));
            }
            // N.B. We don't have to check 'is_quit' here because the EOI
            // transition can never lead to a quit state.
            debug_assert!(!dfa.is_quit_state(*sid));
        }
    }
    Ok(())
}

#[cfg(feature = "hybrid")]
#[inline(always)]
fn hybrid_eoi_fwd(
    dfa: &crate::hybrid::dfa::DFA,
    cache: &mut crate::hybrid::dfa::Cache,
    input: &Input<'_>,
    sid: &mut crate::hybrid::LazyStateID,
    mat: &mut Option<HalfMatch>,
) -> Result<(), MatchError> {
    let sp = input.get_span();
    match input.haystack().get(sp.end) {
        Some(&b) => {
            *sid = dfa
                .next_state(cache, *sid, b)
                .map_err(|_| MatchError::gave_up(sp.end))?;
            if sid.is_match() {
                let pattern = dfa.match_pattern(cache, *sid, 0);
                *mat = Some(HalfMatch::new(pattern, sp.end));
            } else if sid.is_quit() {
                if mat.is_some() {
                    return Ok(());
                }
                return Err(MatchError::quit(b, sp.end));
            }
        }
        None => {
            *sid = dfa
                .next_eoi_state(cache, *sid)
                .map_err(|_| MatchError::gave_up(input.haystack().len()))?;
            if sid.is_match() {
                let pattern = dfa.match_pattern(cache, *sid, 0);
                *mat = Some(HalfMatch::new(pattern, input.haystack().len()));
            }
            // N.B. We don't have to check 'is_quit' here because the EOI
            // transition can never lead to a quit state.
            debug_assert!(!sid.is_quit());
        }
    }
    Ok(())
}
