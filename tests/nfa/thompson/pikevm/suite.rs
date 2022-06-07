use std::{cell::RefCell, rc::Rc};

use regex_automata::{
    nfa::thompson::{
        self,
        pikevm::{self, PikeVM},
    },
    util::iter,
    MatchKind, PatternSet, Search, SyntaxConfig,
};

use ret::{
    bstr::{BString, ByteSlice},
    CompiledRegex, RegexTest, TestResult, TestRunner,
};

use crate::{nfa::thompson::testify_captures, suite, Result};

/// Tests the default configuration of the hybrid NFA/DFA.
#[test]
fn default() -> Result<()> {
    let builder = PikeVM::builder();
    let mut runner = TestRunner::new()?;
    runner.expand(&["is_match", "find", "captures"], |test| test.compiles());
    runner.test_iter(suite()?.iter(), compiler(builder)).assert();
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
        Ok(CompiledRegex::compiled(move |test| -> TestResult {
            run_test(&re, &mut cache, test)
        }))
    }
}

fn run_test(
    re: &PikeVM,
    cache: &mut pikevm::Cache,
    test: &RegexTest,
) -> TestResult {
    match test.additional_name() {
        "is_match" => TestResult::matched(re.is_match(cache, test.input())),
        "find" => match test.search_kind() {
            ret::SearchKind::Earliest => {
                let mut caps = re.create_captures();
                let it = iter::TryMatches::new(
                    Search::new(test.input().as_bytes())
                        .earliest(true)
                        .utf8(test.utf8()),
                    move |search| {
                        re.search(cache, None, search, &mut caps);
                        Ok(caps.get_match())
                    },
                )
                .infallible()
                .take(test.match_limit().unwrap_or(std::usize::MAX))
                .map(|m| ret::Match {
                    id: m.pattern().as_usize(),
                    span: ret::Span { start: m.start(), end: m.end() },
                });
                TestResult::matches(it)
            }
            ret::SearchKind::Leftmost => {
                let it = re
                    .find_iter(cache, test.input())
                    .take(test.match_limit().unwrap_or(std::usize::MAX))
                    .map(|m| ret::Match {
                        id: m.pattern().as_usize(),
                        span: ret::Span { start: m.start(), end: m.end() },
                    });
                TestResult::matches(it)
            }
            ret::SearchKind::Overlapping => {
                let mut matset = PatternSet::new(re.get_nfa().pattern_len());
                let search =
                    Search::new(test.input()).utf8(re.get_config().get_utf8());
                re.which_overlapping_matches(
                    cache,
                    None,
                    &search,
                    &mut matset,
                );
                TestResult::which(matset.iter().map(|p| p.as_usize()))
            }
        },
        "captures" => {
            match test.search_kind() {
                ret::SearchKind::Earliest => {
                    // This is pretty messy. There is no 'earliest' iterator,
                    // so we've got to roll our own. We do reuse the iterator
                    // 'TryMatches' helper, but since those helpers don't
                    // support capturing groups directly, we've got to smuggle
                    // it through using a RefCell.
                    let caps = Rc::new(RefCell::new(re.create_captures()));
                    let it = iter::TryMatches::new(
                        Search::new(test.input().as_bytes())
                            .earliest(true)
                            .utf8(test.utf8()),
                        {
                            let caps = Rc::clone(&caps);
                            move |search| {
                                let caps = &mut *caps.borrow_mut();
                                re.search(cache, None, search, caps);
                                Ok(caps.get_match())
                            }
                        },
                    )
                    .infallible()
                    .take(test.match_limit().unwrap_or(std::usize::MAX))
                    .map(|_| testify_captures(&caps.borrow()));
                    TestResult::captures(it)
                }
                ret::SearchKind::Leftmost => {
                    let it = re
                        .captures_iter(cache, test.input())
                        .take(test.match_limit().unwrap_or(std::usize::MAX))
                        .map(|caps| testify_captures(&caps));
                    TestResult::captures(it)
                }
                ret::SearchKind::Overlapping => {
                    let mut matset =
                        PatternSet::new(re.get_nfa().pattern_len());
                    let search = Search::new(test.input())
                        .utf8(re.get_config().get_utf8());
                    re.which_overlapping_matches(
                        cache,
                        None,
                        &search,
                        &mut matset,
                    );
                    TestResult::which(matset.iter().map(|p| p.as_usize()))
                }
            }
        }
        name => TestResult::fail(&format!("unrecognized test name: {}", name)),
    }
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
    let match_kind = match test.match_kind() {
        ret::MatchKind::All => MatchKind::All,
        ret::MatchKind::LeftmostFirst => MatchKind::LeftmostFirst,
        ret::MatchKind::LeftmostLongest => return false,
    };
    let pikevm_config = PikeVM::config()
        .anchored(test.anchored())
        .match_kind(match_kind)
        .utf8(test.utf8());
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
