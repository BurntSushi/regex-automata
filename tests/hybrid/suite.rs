use regex_automata::{
    hybrid::{
        dfa::DFA,
        regex::{self, Regex},
    },
    nfa::thompson,
    util::iter,
    MatchKind, MatchSet, Search, SyntaxConfig,
};

use ret::{
    bstr::{BString, ByteSlice},
    CompiledRegex, RegexTest, TestResult, TestRunner,
};

use crate::{suite, Result};

/// Tests the default configuration of the hybrid NFA/DFA.
#[test]
fn default() -> Result<()> {
    let builder = Regex::builder();
    TestRunner::new()?
        .expand(&["is_match", "find"], |t| t.compiles())
        // Without NFA shrinking, this test blows the default cache capacity.
        .blacklist("expensive/regression-many-repeat-no-stack-overflow")
        .test_iter(suite()?.iter(), compiler(builder))
        .assert();
    Ok(())
}

/// Tests the hybrid NFA/DFA with NFA shrinking enabled.
///
/// This is *usually* not the configuration one wants for a lazy DFA. NFA
/// shrinking is mostly only advantageous when building a full DFA since it
/// can sharply decrease the amount of time determinization takes. But NFA
/// shrinking is itself otherwise fairly expensive currently. Since a lazy DFA
/// has no compilation time (other than for building the NFA of course) before
/// executing a search, it's usually worth it to forgo NFA shrinking.
///
/// Nevertheless, we test to make sure everything is OK with NFA shrinking. As
/// a bonus, there are some tests we don't need to skip because they now fit in
/// the default cache capacity.
#[test]
fn nfa_shrink() -> Result<()> {
    let mut builder = Regex::builder();
    builder.thompson(thompson::Config::new().shrink(true));
    TestRunner::new()?
        .expand(&["is_match", "find"], |t| t.compiles())
        .test_iter(suite()?.iter(), compiler(builder))
        .assert();
    Ok(())
}

/// Tests the hybrid NFA/DFA when 'starts_for_each_pattern' is enabled.
#[test]
fn starts_for_each_pattern() -> Result<()> {
    let mut builder = Regex::builder();
    builder.dfa(DFA::config().starts_for_each_pattern(true));
    TestRunner::new()?
        .expand(&["is_match", "find"], |t| t.compiles())
        // Without NFA shrinking, this test blows the default cache capacity.
        .blacklist("expensive/regression-many-repeat-no-stack-overflow")
        .test_iter(suite()?.iter(), compiler(builder))
        .assert();
    Ok(())
}

/// Tests the hybrid NFA/DFA when byte classes are disabled.
///
/// N.B. Disabling byte classes doesn't avoid any indirection at search time.
/// All it does is cause every byte value to be its own distinct equivalence
/// class.
#[test]
fn no_byte_classes() -> Result<()> {
    let mut builder = Regex::builder();
    builder.dfa(DFA::config().byte_classes(false));
    TestRunner::new()?
        .expand(&["is_match", "find"], |t| t.compiles())
        // Without NFA shrinking, this test blows the default cache capacity.
        .blacklist("expensive/regression-many-repeat-no-stack-overflow")
        .test_iter(suite()?.iter(), compiler(builder))
        .assert();
    Ok(())
}

/// Tests that hybrid NFA/DFA never clears its cache for any test with the
/// default capacity.
///
/// N.B. If a regex suite test is added that causes the cache to be cleared,
/// then this should just skip that test. (Which can be done by calling the
/// 'blacklist' method on 'TestRunner'.)
#[test]
fn no_cache_clearing() -> Result<()> {
    let mut builder = Regex::builder();
    builder.dfa(DFA::config().minimum_cache_clear_count(Some(0)));
    TestRunner::new()?
        .expand(&["is_match", "find"], |t| t.compiles())
        // Without NFA shrinking, this test blows the default cache capacity.
        .blacklist("expensive/regression-many-repeat-no-stack-overflow")
        .test_iter(suite()?.iter(), compiler(builder))
        .assert();
    Ok(())
}

/// Tests the hybrid NFA/DFA when the minimum cache capacity is set.
#[test]
fn min_cache_capacity() -> Result<()> {
    let mut builder = Regex::builder();
    builder
        .dfa(DFA::config().cache_capacity(0).skip_cache_capacity_check(true));
    TestRunner::new()?
        .expand(&["is_match", "find"], |t| t.compiles())
        .test_iter(suite()?.iter(), compiler(builder))
        .assert();
    Ok(())
}

fn compiler(
    mut builder: regex::Builder,
) -> impl FnMut(&RegexTest, &[BString]) -> Result<CompiledRegex> {
    move |test, regexes| {
        let regexes = regexes
            .iter()
            .map(|r| r.to_str().map(|s| s.to_string()))
            .collect::<std::result::Result<Vec<String>, _>>()?;

        // Check if our regex contains things that aren't supported by DFAs.
        // That is, Unicode word boundaries when searching non-ASCII text.
        let mut thompson = thompson::Compiler::new();
        thompson.syntax(config_syntax(test)).configure(config_thompson(test));
        if let Ok(nfa) = thompson.build_many(&regexes) {
            let non_ascii = test.input().iter().any(|&b| !b.is_ascii());
            if nfa.has_word_boundary_unicode() && non_ascii {
                return Ok(CompiledRegex::skip());
            }
        }
        if !configure_regex_builder(test, &mut builder) {
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
    cache: &mut regex::Cache,
    test: &RegexTest,
) -> TestResult {
    match test.additional_name() {
        "is_match" => TestResult::matched(re.is_match(cache, test.input())),
        "find" => match test.search_kind() {
            ret::SearchKind::Earliest => {
                let mut scanner = re.scanner();
                let it = iter::TryMatches::new(
                    Search::new(test.input().as_bytes())
                        .earliest(true)
                        .utf8(test.utf8()),
                    move |search| {
                        re.try_search(cache, scanner.as_mut(), search)
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
                let patlen = re.forward().get_nfa().pattern_len();
                let mut matset = MatchSet::new(patlen);
                let search =
                    Search::new(test.input()).utf8(re.get_config().get_utf8());
                re.forward()
                    .which_overlapping_matches(
                        cache.as_parts_mut().0,
                        None,
                        &search,
                        &mut matset,
                    )
                    .unwrap();
                TestResult::which(matset.iter().map(|p| p.as_usize()))
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
fn configure_regex_builder(
    test: &RegexTest,
    builder: &mut regex::Builder,
) -> bool {
    let match_kind = match test.match_kind() {
        ret::MatchKind::All => MatchKind::All,
        ret::MatchKind::LeftmostFirst => MatchKind::LeftmostFirst,
        ret::MatchKind::LeftmostLongest => return false,
    };

    let dense_config = DFA::config()
        .anchored(test.anchored())
        .match_kind(match_kind)
        .unicode_word_boundary(true);
    let regex_config = Regex::config().utf8(test.utf8());
    builder
        .configure(regex_config)
        .syntax(config_syntax(test))
        .thompson(config_thompson(test))
        .dfa(dense_config);
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
