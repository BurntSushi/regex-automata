use regex_automata::{
    hybrid::{
        self,
        regex::{self, Regex},
    },
    nfa::thompson,
    MatchKind, SyntaxConfig,
};
use regex_syntax as syntax;

use regex_test::{
    bstr::{BString, ByteSlice},
    CompiledRegex, Match, MatchKind as TestMatchKind, RegexTest, RegexTests,
    SearchKind as TestSearchKind, TestResult, TestRunner,
};

use crate::{suite, Result};

#[test]
fn default() -> Result<()> {
    let builder = Regex::builder();
    TestRunner::new()?.test_iter(suite()?.iter(), compiler2(builder)).assert();
    Ok(())
}

fn compiler2(
    builder: hybrid::regex::Builder,
) -> impl FnMut(&RegexTest, &[BString]) -> Result<CompiledRegex> {
    compiler(builder, |_, re, mut cache| {
        Ok(CompiledRegex::compiled(move |test| -> Vec<TestResult> {
            run_test(&re, &mut cache, test)
        }))
    })
}

fn compiler(
    mut builder: hybrid::regex::Builder,
    mut create_matcher: impl FnMut(
        &hybrid::regex::Builder,
        Regex,
        regex::Cache,
    ) -> Result<CompiledRegex>,
) -> impl FnMut(&RegexTest, &[BString]) -> Result<CompiledRegex> {
    move |test, regexes| {
        let regexes = regexes
            .iter()
            .map(|r| r.to_str().map(|s| s.to_string()))
            .collect::<std::result::Result<Vec<String>, _>>()?;

        // Check if our regex contains things that aren't supported by DFAs.
        // That is, Unicode word boundaries when searching non-ASCII text.
        let mut thompson = thompson::Builder::new();
        thompson.configure(config_thompson(test));
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
        let cache = re.new_cache();
        create_matcher(&builder, re, cache)
    }
}

fn run_test(
    re: &Regex,
    cache: &mut regex::Cache,
    test: &RegexTest,
) -> Vec<TestResult> {
    // let is_match = if re.is_match(cache, test.input()) {
    // TestResult::matched()
    // } else {
    // TestResult::no_match()
    // };
    // let is_match = is_match.name("is_match");

    let find_matches = match test.search_kind() {
        TestSearchKind::Earliest => {
            let it = re
                .find_earliest_iter(cache, test.input())
                .take(test.match_limit().unwrap_or(std::usize::MAX))
                .map(|m| Match {
                    id: m.pattern().as_usize(),
                    start: m.start(),
                    end: m.end(),
                });
            TestResult::matches(it).name("find_earliest_iter")
        }
        TestSearchKind::Leftmost => {
            let it = re
                .find_leftmost_iter(cache, test.input())
                .take(test.match_limit().unwrap_or(std::usize::MAX))
                .map(|m| Match {
                    id: m.pattern().as_usize(),
                    start: m.start(),
                    end: m.end(),
                });
            TestResult::matches(it).name("find_leftmost_iter")
        }
        TestSearchKind::Overlapping => {
            let it = re
                .find_overlapping_iter(cache, test.input())
                .take(test.match_limit().unwrap_or(std::usize::MAX))
                .map(|m| Match {
                    id: m.pattern().as_usize(),
                    start: m.start(),
                    end: m.end(),
                });
            TestResult::matches(it).name("find_overlapping_iter")
        }
    };

    // vec![is_match, find_matches]
    vec![find_matches]
}

fn configure_regex_builder(
    test: &RegexTest,
    builder: &mut hybrid::regex::Builder,
) -> bool {
    let match_kind = match test.match_kind() {
        TestMatchKind::All => MatchKind::All,
        TestMatchKind::LeftmostFirst => MatchKind::LeftmostFirst,
        TestMatchKind::LeftmostLongest => return false,
    };

    let syntax_config = SyntaxConfig::new()
        .case_insensitive(test.case_insensitive())
        .unicode(test.unicode())
        .utf8(test.utf8());
    let lazy_config = hybrid::Config::new()
        .anchored(test.anchored())
        .match_kind(match_kind)
        .byte_classes(false)
        .unicode_word_boundary(true);
    let regex_config = Regex::config().utf8(test.utf8());

    builder
        .configure(regex_config)
        .syntax(syntax_config)
        .thompson(config_thompson(test))
        .lazy(lazy_config);
    true
}

fn config_thompson(test: &RegexTest) -> thompson::Config {
    thompson::Config::new().utf8(test.utf8())
}
