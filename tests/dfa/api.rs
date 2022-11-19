use std::error::Error;

use regex_automata::{
    dfa::{dense, regex::Regex, Automaton, OverlappingState},
    nfa::thompson,
    HalfMatch, Input, Match, MatchError,
};

use crate::util::{BunkPrefilter, SubstringPrefilter};

// Tests that quit bytes in the forward direction work correctly.
#[test]
fn quit_fwd() -> Result<(), Box<dyn Error>> {
    let dfa = dense::Builder::new()
        .configure(dense::Config::new().quit(b'x', true))
        .build("[[:word:]]+$")?;

    assert_eq!(dfa.try_find_fwd(b"abcxyz"), Err(MatchError::quit(b'x', 3)),);
    assert_eq!(
        dfa.try_search_overlapping_fwd(
            &Input::new(b"abcxyz"),
            &mut OverlappingState::start()
        ),
        Err(MatchError::quit(b'x', 3)),
    );

    Ok(())
}

// Tests that quit bytes in the reverse direction work correctly.
#[test]
fn quit_rev() -> Result<(), Box<dyn Error>> {
    let dfa = dense::Builder::new()
        .configure(dense::Config::new().quit(b'x', true))
        .thompson(thompson::Config::new().reverse(true))
        .build("^[[:word:]]+")?;

    assert_eq!(dfa.try_find_rev(b"abcxyz"), Err(MatchError::quit(b'x', 3)),);

    Ok(())
}

// Tests that if we heuristically enable Unicode word boundaries but then
// instruct that a non-ASCII byte should NOT be a quit byte, then the builder
// will panic.
#[test]
#[should_panic]
fn quit_panics() {
    dense::Config::new().unicode_word_boundary(true).quit(b'\xFF', false);
}

// This tests an intesting case where even if the Unicode word boundary option
// is disabled, setting all non-ASCII bytes to be quit bytes will cause Unicode
// word boundaries to be enabled.
#[test]
fn unicode_word_implicitly_works() -> Result<(), Box<dyn Error>> {
    let mut config = dense::Config::new();
    for b in 0x80..=0xFF {
        config = config.quit(b, true);
    }
    let dfa = dense::Builder::new().configure(config).build(r"\b")?;
    let expected = HalfMatch::must(0, 1);
    assert_eq!(dfa.try_find_fwd(b" a"), Ok(Some(expected)));
    Ok(())
}

// Tests that we can provide a prefilter to a Regex, and the search reports
// correct results.
#[test]
fn prefilter_works() -> Result<(), Box<dyn Error>> {
    let re = Regex::new(r"a[0-9]+")
        .unwrap()
        .with_prefilter(Some(SubstringPrefilter::new("a")));
    let text = b"foo abc foo a1a2a3 foo a123 bar aa456";
    let matches: Vec<(usize, usize)> =
        re.find_iter(text).map(|m| (m.start(), m.end())).collect();
    assert_eq!(
        matches,
        vec![(12, 14), (14, 16), (16, 18), (23, 27), (33, 37),]
    );
    Ok(())
}

// This test confirms that a prefilter is active by using a prefilter that
// reports false negatives.
#[test]
fn prefilter_is_active() -> Result<(), Box<dyn Error>> {
    let re = Regex::new(r"a[0-9]+")
        .unwrap()
        .with_prefilter(Some(SubstringPrefilter::new("a")));
    assert_eq!(re.find(b"za123"), Some(Match::must(0, 1..5)));
    assert_eq!(re.find(b"a123"), Some(Match::must(0, 0..4)));
    let re = re.with_prefilter(Some(BunkPrefilter::new()));
    assert_eq!(re.find(b"za123"), None);
    // This checks that the prefilter is used when first starting the search,
    // instead of waiting until at least one transition has occurred.
    assert_eq!(re.find(b"a123"), None);
    Ok(())
}
