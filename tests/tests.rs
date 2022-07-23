mod dfa;
mod hybrid;
mod nfa;
mod util;

#[cfg(not(miri))]
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[cfg(not(miri))]
fn suite() -> Result<ret::RegexTests> {
    let mut tests = ret::RegexTests::new();
    macro_rules! load {
        ($name:expr) => {{
            const DATA: &[u8] =
                include_bytes!(concat!("data/", $name, ".toml"));
            tests.load_slice($name, DATA)?;
        }};
    }

    load!("anchored");
    load!("bytes");
    load!("crazy");
    load!("earliest");
    load!("empty");
    load!("expensive");
    load!("flags");
    load!("iter");
    load!("leftmost-all");
    load!("misc");
    load!("multiline");
    load!("no-unicode");
    load!("overlapping");
    load!("regression");
    load!("set");
    load!("unicode");
    load!("word-boundary");
    load!("fowler/basic");
    load!("fowler/nullsubexpr");
    load!("fowler/repetition");

    Ok(tests)
}

/// Convert Thompson captures into the test suite's capture values.
///
/// The given Thompson captures must represent a valid match, where the first
/// capturing group has a non-None span. Otherwise this panics.
#[cfg(not(miri))]
fn testify_captures(
    caps: &regex_automata::util::captures::Captures,
) -> ret::Captures {
    assert!(caps.is_match(), "expected captures to represent a match");
    let spans = caps
        .iter()
        .map(|group| group.map(|m| ret::Span { start: m.start, end: m.end }));
    // These unwraps are OK because we assume our 'caps' represents a match,
    // and a match always gives a non-zero number of groups with the first
    // group being non-None.
    ret::Captures::new(caps.pattern().unwrap().as_usize(), spans).unwrap()
}
