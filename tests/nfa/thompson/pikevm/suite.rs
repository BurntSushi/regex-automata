use regex_automata::{
    nfa::thompson::{
        self,
        pikevm::{self, PikeVM},
    },
    util::{iter, prefilter::Prefilter, syntax},
    PatternSet,
};

use ret::{
    bstr::{BString, ByteSlice},
    CompiledRegex, RegexTest, TestResult, TestRunner,
};

use crate::{create_input, suite, testify_captures, untestify_kind, Result};

/// Tests the default configuration of the hybrid NFA/DFA.
#[test]
fn default() -> Result<()> {
    let builder = PikeVM::builder();
    let mut runner = TestRunner::new()?;
    runner.expand(&["is_match", "find", "captures"], |test| test.compiles());
    runner.test_iter(suite()?.iter(), compiler(builder)).assert();
    Ok(())
}

/// Tests the PikeVM with prefilters enabled.
#[test]
fn prefilter() -> Result<()> {
    let my_compiler = |test: &RegexTest, regexes: &[BString]| {
        // Parse regexes as HIRs so we can get literals to build a prefilter.
        let mut hirs = vec![];
        for pattern in regexes.iter() {
            let pattern = pattern.to_str()?;
            hirs.push(syntax::parse(&config_syntax(test), pattern)?);
        }
        let kind = match untestify_kind(test.match_kind()) {
            None => return Ok(CompiledRegex::skip()),
            Some(kind) => kind,
        };
        let pre = Prefilter::from_hirs(kind, &hirs);
        let mut builder = PikeVM::builder();
        builder.configure(PikeVM::config().prefilter(pre));
        compiler(builder)(test, regexes)
    };
    let mut runner = TestRunner::new()?;
    runner.expand(&["is_match", "find", "captures"], |test| test.compiles());
    runner.test_iter(suite()?.iter(), my_compiler).assert();
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
    let input = create_input(test);
    match test.additional_name() {
        "is_match" => {
            // TODO: Use the actual 'is_match' method once it is generic on
            // 'Into<Input>'.
            let mut caps = re.create_captures();
            TestResult::matched(
                re.try_search_slots(
                    cache,
                    &input.earliest(true),
                    caps.slots_mut(),
                )
                .unwrap()
                .is_some(),
            )
        }
        "find" => match test.search_kind() {
            ret::SearchKind::Earliest => {
                let input = input.earliest(true);
                let mut caps = re.create_captures();
                let it = iter::Searcher::new(input)
                    .into_matches_iter(|input| {
                        re.try_search(cache, input, &mut caps)?;
                        Ok(caps.get_match())
                    })
                    .infallible()
                    .take(test.match_limit().unwrap_or(std::usize::MAX))
                    .map(|m| ret::Match {
                        id: m.pattern().as_usize(),
                        span: ret::Span { start: m.start(), end: m.end() },
                    });
                TestResult::matches(it)
            }
            ret::SearchKind::Leftmost => {
                let mut caps = re.create_captures();
                let it = iter::Searcher::new(input)
                    .into_matches_iter(|input| {
                        re.try_search(cache, input, &mut caps)?;
                        Ok(caps.get_match())
                    })
                    .infallible()
                    .take(test.match_limit().unwrap_or(std::usize::MAX))
                    .map(|m| ret::Match {
                        id: m.pattern().as_usize(),
                        span: ret::Span { start: m.start(), end: m.end() },
                    });
                TestResult::matches(it)
            }
            ret::SearchKind::Overlapping => {
                let mut patset = PatternSet::new(re.get_nfa().pattern_len());
                re.which_overlapping_matches(cache, &input, &mut patset)
                    .unwrap();
                TestResult::which(patset.iter().map(|p| p.as_usize()))
            }
        },
        "captures" => match test.search_kind() {
            ret::SearchKind::Earliest => {
                let input = input.earliest(true);
                let it = iter::Searcher::new(input)
                    .into_captures_iter(re.create_captures(), |input, caps| {
                        re.try_search(cache, input, caps)
                    })
                    .infallible()
                    .take(test.match_limit().unwrap_or(std::usize::MAX))
                    .map(|caps| testify_captures(&caps));
                TestResult::captures(it)
            }
            ret::SearchKind::Leftmost => {
                let it = iter::Searcher::new(input)
                    .into_captures_iter(re.create_captures(), |input, caps| {
                        re.try_search(cache, input, caps)
                    })
                    .infallible()
                    .take(test.match_limit().unwrap_or(std::usize::MAX))
                    .map(|caps| testify_captures(&caps));
                TestResult::captures(it)
            }
            ret::SearchKind::Overlapping => {
                // There is no overlapping PikeVM API that supports captures.
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
fn configure_pikevm_builder(
    test: &RegexTest,
    builder: &mut pikevm::Builder,
) -> bool {
    let match_kind = match untestify_kind(test.match_kind()) {
        None => return false,
        Some(k) => k,
    };
    let pikevm_config = PikeVM::config().match_kind(match_kind);
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
fn config_syntax(test: &RegexTest) -> syntax::Config {
    syntax::Config::new()
        .case_insensitive(test.case_insensitive())
        .unicode(test.unicode())
        .utf8(test.utf8())
}
