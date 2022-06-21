#[cfg(not(miri))]
use regex_automata::nfa::thompson::Captures;

mod backtrack;
mod pikevm;

/// Convert Thompson captures into the test suite's capture values.
///
/// The given Thompson captures must represent a valid match, where the first
/// capturing group has a non-None span. Otherwise this panics.
#[cfg(not(miri))]
fn testify_captures(caps: &Captures) -> ret::Captures {
    assert!(caps.is_match(), "expected captures to represent a match");
    let spans = caps
        .iter()
        .map(|group| group.map(|m| ret::Span { start: m.start, end: m.end }));
    // These unwraps are OK because we assume our 'caps' represents a match,
    // and a match always gives a non-zero number of groups with the first
    // group being non-None.
    ret::Captures::new(caps.pattern().unwrap().as_usize(), spans).unwrap()
}
