use regex_automata::{
    meta::{self, Regex},
    util::{iter, syntax},
    MatchKind, PatternSet,
};

use ret::{
    bstr::{BString, ByteSlice},
    CompiledRegex, RegexTest, TestResult, TestRunner,
};

use crate::{create_input, suite, testify_captures, Result};

const BLACKLIST: &[&str] = &[
    // These 'earliest' tests are blacklisted because the meta searcher doesn't
    // give the same offsets that the test expects. This is legal because the
    // 'earliest' routines don't guarantee a particular match offset other
    // than "the earliest the regex engine can report a match." Some regex
    // engines will quit earlier than others. The backtracker, for example,
    // can't really quit before finding the full leftmost-first match. Many of
    // the literal searchers also don't have the ability to quit fully or it's
    // otherwise not worth doing. (A literal searcher not quitting as early as
    // possible usually means looking at a few more bytes. That's no biggie.)
    //
    // Other 'earliest' tests can be added here if the need arises.
    "earliest/no-leftmost-first-100/",
    "earliest/no-leftmost-first-200/",
];

/// Tests the default configuration of the hybrid NFA/DFA.
#[test]
fn default() -> Result<()> {
    let builder = Regex::builder();
    let mut runner = TestRunner::new()?;
    runner
        .expand(&["is_match", "find", "captures"], |test| test.compiles())
        .blacklist_iter(BLACKLIST)
        .test_iter(suite()?.iter(), compiler(builder))
        .assert();
    Ok(())
}

fn compiler(
    mut builder: meta::Builder,
) -> impl FnMut(&RegexTest, &[BString]) -> Result<CompiledRegex> {
    move |test, regexes| {
        let regexes = regexes
            .iter()
            .map(|r| r.to_str().map(|s| s.to_string()))
            .collect::<std::result::Result<Vec<String>, _>>()?;
        if !configure_meta_builder(test, &mut builder) {
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
    re: &Regex,
    cache: &mut meta::Cache,
    test: &RegexTest,
) -> TestResult {
    let input = create_input(test);
    match test.additional_name() {
        "is_match" => TestResult::matched(
            re.try_search_slots(cache, &input.earliest(true), &mut [])
                .unwrap()
                .is_some(),
        ),
        "find" => match test.search_kind() {
            ret::SearchKind::Earliest => {
                let input = input.earliest(true);
                let it = iter::Searcher::new(input)
                    .into_matches_iter(|input| re.try_search(cache, input))
                    .infallible()
                    .take(test.match_limit().unwrap_or(std::usize::MAX))
                    .map(|m| ret::Match {
                        id: m.pattern().as_usize(),
                        span: ret::Span { start: m.start(), end: m.end() },
                    });
                TestResult::matches(it)
            }
            ret::SearchKind::Leftmost => {
                let it = iter::Searcher::new(input)
                    .into_matches_iter(|input| re.try_search(cache, input))
                    .infallible()
                    .take(test.match_limit().unwrap_or(std::usize::MAX))
                    .map(|m| ret::Match {
                        id: m.pattern().as_usize(),
                        span: ret::Span { start: m.start(), end: m.end() },
                    });
                TestResult::matches(it)
            }
            ret::SearchKind::Overlapping => {
                let mut patset = PatternSet::new(re.pattern_len());
                re.try_which_overlapping_matches(cache, &input, &mut patset)
                    .unwrap();
                TestResult::which(patset.iter().map(|p| p.as_usize()))
            }
        },
        "captures" => match test.search_kind() {
            ret::SearchKind::Earliest => {
                let input = input.earliest(true);
                let it = iter::Searcher::new(input)
                    .into_captures_iter(re.create_captures(), |input, caps| {
                        re.try_search_captures(cache, input, caps)
                    })
                    .infallible()
                    .take(test.match_limit().unwrap_or(std::usize::MAX))
                    .map(|caps| testify_captures(&caps));
                TestResult::captures(it)
            }
            ret::SearchKind::Leftmost => {
                let it = iter::Searcher::new(input)
                    .into_captures_iter(re.create_captures(), |input, caps| {
                        re.try_search_captures(cache, input, caps)
                    })
                    .infallible()
                    .take(test.match_limit().unwrap_or(std::usize::MAX))
                    .map(|caps| testify_captures(&caps));
                TestResult::captures(it)
            }
            ret::SearchKind::Overlapping => {
                // There is no overlapping regex API that supports captures.
                TestResult::skip()
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
fn configure_meta_builder(
    test: &RegexTest,
    builder: &mut meta::Builder,
) -> bool {
    let match_kind = match test.match_kind() {
        ret::MatchKind::All => MatchKind::All,
        ret::MatchKind::LeftmostFirst => MatchKind::LeftmostFirst,
        ret::MatchKind::LeftmostLongest => return false,
    };
    let meta_config = Regex::config().match_kind(match_kind).utf8(test.utf8());
    builder.configure(meta_config).syntax(config_syntax(test));
    true
}

/// Configuration of the regex parser from a regex test.
fn config_syntax(test: &RegexTest) -> syntax::Config {
    syntax::Config::new()
        .case_insensitive(test.case_insensitive())
        .unicode(test.unicode())
        .utf8(test.utf8())
}
