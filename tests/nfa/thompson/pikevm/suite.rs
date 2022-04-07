use regex_automata::{
    nfa::thompson::{
        self,
        pikevm::{self, PikeVM},
    },
    SyntaxConfig,
};

use regex_test::{
    bstr::{BString, ByteSlice},
    Captures as TestCaptures, CompiledRegex, Match, RegexTest,
    SearchKind as TestSearchKind, Span, TestResult, TestRunner,
};

use crate::{suite, Result};

/// Tests the default configuration of the hybrid NFA/DFA.
#[test]
fn default() -> Result<()> {
    let builder = PikeVM::builder();
    TestRunner::new()?.test_iter(suite()?.iter(), compiler(builder)).assert();
    Ok(())
}

fn compiler(
    mut builder: pikevm::Builder,
) -> impl FnMut(&RegexTest, &[BString]) -> Result<CompiledRegex> {
    move |test, regexes| {
        let regexes = regexes
            .iter()
            .map(|r| r.to_str().map(|s| s.to_string()))
            .collect::<std::result::Result<Vec<String>, _>>()?;
        if !configure_pikevm_builder(test, &mut builder) {
            return Ok(CompiledRegex::skip());
        }
        let re = builder.build_many(&regexes)?;
        let mut cache = re.create_cache();
        Ok(CompiledRegex::compiled(move |test| -> Vec<TestResult> {
            run_test(&re, &mut cache, test)
        }))
    }
}

fn run_test(
    re: &PikeVM,
    cache: &mut pikevm::Cache,
    test: &RegexTest,
) -> Vec<TestResult> {
    let is_match = if re.is_match(cache, test.input()) {
        TestResult::matched()
    } else {
        TestResult::no_match()
    };
    let is_match = is_match.name("is_match");

    let find = match test.search_kind() {
        TestSearchKind::Earliest => {
            TestResult::skip().name("find_earliest_iter")
        }
        TestSearchKind::Leftmost => {
            let it = re
                .find_leftmost_iter(cache, test.input())
                .take(test.match_limit().unwrap_or(std::usize::MAX))
                .map(|m| Match {
                    id: m.pattern().as_usize(),
                    span: Span { start: m.start(), end: m.end() },
                });
            TestResult::matches(it).name("find_leftmost_iter")
        }
        TestSearchKind::Overlapping => {
            TestResult::skip().name("find_overlapping_iter")
        }
    };

    let captures = match test.search_kind() {
        TestSearchKind::Earliest => {
            TestResult::skip().name("captures_earliest_iter")
        }
        TestSearchKind::Leftmost => {
            let it = re
                .captures_leftmost_iter(cache, test.input())
                .take(test.match_limit().unwrap_or(std::usize::MAX))
                .map(|caps| {
                    // We wouldn't get a non-matching captures in an iterator.
                    assert!(caps.is_match());
                    let testcaps = caps.iter().map(|m| {
                        m.map(|m| Span { start: m.start(), end: m.end() })
                    });
                    // This unwrap is OK because we know captures is a match,
                    // and all matches always have the first group set.
                    TestCaptures::new(caps.pattern().as_usize(), testcaps)
                        .unwrap()
                });
            TestResult::captures(it).name("captures_leftmost_iter")
        }
        TestSearchKind::Overlapping => {
            TestResult::skip().name("captures_overlapping_iter")
        }
    };

    vec![is_match, find, captures]
}

/// Configures the given regex builder with all relevant settings on the given
/// regex test.
///
/// If the regex test has a setting that is unsupported, then this returns
/// false (implying the test should be skipped).
fn configure_pikevm_builder(
    test: &RegexTest,
    builder: &mut pikevm::Builder,
) -> bool {
    // TODO: I think we need to support All in the PikeVM? Maybe not. But seems
    // closely related to Overlapping search...
    // let match_kind = match test.match_kind() {
    // TestMatchKind::All => MatchKind::All,
    // TestMatchKind::LeftmostFirst => MatchKind::LeftmostFirst,
    // TestMatchKind::LeftmostLongest => return false,
    // };
    let pikevm_config =
        PikeVM::config().anchored(test.anchored()).utf8(test.utf8());
    builder
        .configure(pikevm_config)
        .syntax(config_syntax(test))
        .thompson(config_thompson(test));
    true
}

/// Configuration of a Thompson NFA compiler from a regex test.
fn config_thompson(test: &RegexTest) -> thompson::Config {
    thompson::Config::new().utf8(test.utf8())
}

/// Configuration of the regex parser from a regex test.
fn config_syntax(test: &RegexTest) -> SyntaxConfig {
    SyntaxConfig::new()
        .case_insensitive(test.case_insensitive())
        .unicode(test.unicode())
        .utf8(test.utf8())
}
