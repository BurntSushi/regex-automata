use regex_automata::{
    nfa::thompson::{
        self,
        backtrack::{self, BoundedBacktracker},
        NFA,
    },
    util::syntax,
    Input,
};

use ret::{
    bstr::{BString, ByteSlice},
    CompiledRegex, RegexTest, TestResult, TestRunner,
};

use crate::{nfa::thompson::testify_captures, suite, Result};

/// Tests the default configuration of the bounded backtracker.
#[test]
fn default() -> Result<()> {
    let builder = BoundedBacktracker::builder();
    let mut runner = TestRunner::new()?;
    runner.expand(&["is_match", "find", "captures"], |test| test.compiles());
    // At the time of writing, every regex search in the test suite fits
    // into the backtracker's default visited capacity (except for the
    // blacklisted one below). If regexes are added that blow that capacity,
    // then they should be blacklisted here. A tempting alternative is to
    // automatically skip them by checking the haystack length against
    // BoundedBacktracker::max_haystack_len, but that could wind up hiding
    // interesting failure modes. e.g., If the visited capacity is somehow
    // wrong or smaller than it should be.
    runner.blacklist("expensive/backtrack-blow-visited-capacity");
    runner.test_iter(suite()?.iter(), compiler(builder)).assert();
    Ok(())
}

/// Tests the bounded backtracker when its visited capacity is set to its
/// minimum amount.
#[test]
fn min_visited_capacity() -> Result<()> {
    let mut runner = TestRunner::new()?;
    runner.expand(&["is_match", "find", "captures"], |test| test.compiles());
    runner
        .test_iter(suite()?.iter(), move |test, regexes| {
            let regexes = regexes
                .iter()
                .map(|r| r.to_str().map(|s| s.to_string()))
                .collect::<std::result::Result<Vec<String>, _>>()?;
            let nfa = NFA::compiler()
                .configure(config_thompson(test))
                .syntax(config_syntax(test))
                .build_many(&regexes)?;
            let mut builder = BoundedBacktracker::builder();
            if !configure_backtrack_builder(test, &mut builder) {
                return Ok(CompiledRegex::skip());
            }
            // Setup the bounded backtracker so that its visited capacity is
            // the absolute minimum required for the test's haystack.
            builder.configure(BoundedBacktracker::config().visited_capacity(
                backtrack::min_visited_capacity(
                    &nfa,
                    &Input::new(test.input()),
                ),
            ));

            let re = builder.build_from_nfa(nfa)?;
            let mut cache = re.create_cache();
            Ok(CompiledRegex::compiled(move |test| -> TestResult {
                run_test(&re, &mut cache, test)
            }))
        })
        .assert();
    Ok(())
}

fn compiler(
    mut builder: backtrack::Builder,
) -> impl FnMut(&RegexTest, &[BString]) -> Result<CompiledRegex> {
    move |test, regexes| {
        let regexes = regexes
            .iter()
            .map(|r| r.to_str().map(|s| s.to_string()))
            .collect::<std::result::Result<Vec<String>, _>>()?;
        if !configure_backtrack_builder(test, &mut builder) {
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
    re: &BoundedBacktracker,
    cache: &mut backtrack::Cache,
    test: &RegexTest,
) -> TestResult {
    match test.additional_name() {
        "is_match" => match test.search_kind() {
            ret::SearchKind::Earliest | ret::SearchKind::Overlapping => {
                TestResult::skip()
            }
            ret::SearchKind::Leftmost => TestResult::matched(
                re.try_is_match(cache, test.input()).unwrap(),
            ),
        },
        "find" => match test.search_kind() {
            ret::SearchKind::Earliest | ret::SearchKind::Overlapping => {
                TestResult::skip()
            }
            ret::SearchKind::Leftmost => {
                let it = re
                    .try_find_iter(cache, test.input())
                    .take(test.match_limit().unwrap_or(std::usize::MAX))
                    .map(|result| {
                        let m = result.unwrap();
                        ret::Match {
                            id: m.pattern().as_usize(),
                            span: ret::Span { start: m.start(), end: m.end() },
                        }
                    });
                TestResult::matches(it)
            }
        },
        "captures" => match test.search_kind() {
            ret::SearchKind::Earliest | ret::SearchKind::Overlapping => {
                TestResult::skip()
            }
            ret::SearchKind::Leftmost => {
                let it = re
                    .try_captures_iter(cache, test.input())
                    .take(test.match_limit().unwrap_or(std::usize::MAX))
                    .map(|result| testify_captures(&result.unwrap()));
                TestResult::captures(it)
            }
        },
        name => TestResult::fail(&format!("unrecognized test name: {}", name)),
    }
}

/// Configures the given regex builder with all relevant settings on the given
/// regex test.
///
/// If the regex test has a setting that is unsupported, then this returns
/// false (implying the test should be skipped).
fn configure_backtrack_builder(
    test: &RegexTest,
    builder: &mut backtrack::Builder,
) -> bool {
    match (test.search_kind(), test.match_kind()) {
        // For testing the standard search APIs. This is the only supported
        // configuration for the backtracker.
        (ret::SearchKind::Leftmost, ret::MatchKind::LeftmostFirst) => {}
        // Overlapping APIs not supported at all for backtracker.
        (ret::SearchKind::Overlapping, _) => return false,
        // Backtracking doesn't really support the notion of 'earliest'.
        // Namely, backtracking already works by returning as soon as it knows
        // it has found a match. It just so happens that this corresponds to
        // the standard 'leftmost' formulation.
        //
        // The 'earliest' definition in this crate does indeed permit this
        // behavior, so this is "fine," but our test suite specifically looks
        // for the earliest position at which a match is known, which our
        // finite automata based regex engines have no problem providing. So
        // for backtracking, we just skip these tests.
        (ret::SearchKind::Earliest, _) => return false,
        // For backtracking, 'all' semantics don't really make sense.
        (_, ret::MatchKind::All) => return false,
        // Not supported at all in regex-automata.
        (_, ret::MatchKind::LeftmostLongest) => return false,
    };
    let backtrack_config = BoundedBacktracker::config()
        .anchored(test.anchored())
        .utf8(test.utf8());
    builder
        .configure(backtrack_config)
        .syntax(config_syntax(test))
        .thompson(config_thompson(test));
    true
}

/// Configuration of a Thompson NFA compiler from a regex test.
fn config_thompson(_test: &RegexTest) -> thompson::Config {
    thompson::Config::new()
}

/// Configuration of the regex parser from a regex test.
fn config_syntax(test: &RegexTest) -> syntax::Config {
    syntax::Config::new()
        .case_insensitive(test.case_insensitive())
        .unicode(test.unicode())
        .utf8(test.utf8())
}
