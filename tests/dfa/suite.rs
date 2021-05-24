use regex_automata::{
    dfa::{dense, /*sparse,*/ Automaton, Regex, RegexBuilder, RegexConfig,},
    nfa::thompson,
    MatchKind, SyntaxConfig,
};
use regex_syntax as syntax;

use regex_test::bstr::{BString, ByteSlice};
use regex_test::{
    CompiledRegex, Match, MatchKind as TestMatchKind, RegexTest, RegexTests,
    SearchKind as TestSearchKind, TestResult, TestRunner,
};

use crate::{suite, Result};

#[test]
fn unminimized_default() -> Result<()> {
    let builder = RegexBuilder::new();
    TestRunner::new()?
        .test_iter(suite()?.iter(), dense_compiler(builder))
        .assert();
    Ok(())
}

#[test]
fn unminimized_no_byte_class() -> Result<()> {
    let mut builder = RegexBuilder::new();
    builder.dense(dense::Config::new().byte_classes(false));

    TestRunner::new()?
        .test_iter(suite()?.iter(), dense_compiler(builder))
        .assert();
    Ok(())
}

#[test]
fn unminimized_no_nfa_shrink_default() -> Result<()> {
    let mut builder = RegexBuilder::new();
    builder.thompson(thompson::Config::new().shrink(false));

    TestRunner::new()?
        .test_iter(suite()?.iter(), dense_compiler(builder))
        .assert();
    Ok(())
}

#[test]
fn minimized_default() -> Result<()> {
    let mut builder = RegexBuilder::new();
    builder.dense(dense::Config::new().minimize(true));
    TestRunner::new()?
        .blacklist("expensive")
        .test_iter(suite()?.iter(), dense_compiler(builder))
        .assert();
    Ok(())
}

#[test]
fn minimized_no_byte_class() -> Result<()> {
    let mut builder = RegexBuilder::new();
    builder.dense(dense::Config::new().minimize(true).byte_classes(false));

    TestRunner::new()?
        .blacklist("expensive")
        .test_iter(suite()?.iter(), dense_compiler(builder))
        .assert();
    Ok(())
}

/*
#[test]
fn sparse_unminimized_default() -> Result<()> {
    let builder = RegexBuilder::new();
    TestRunner::new()?
        .test_iter(suite()?.iter(), sparse_compiler(builder))
        .assert();
    Ok(())
}
*/

// Another basic sanity test that checks we can serialize and then deserialize
// a regex, and that the resulting regex can be used for searching correctly.
#[test]
fn serialization_unminimized_default() -> Result<()> {
    let builder = RegexBuilder::new();
    let my_compiler = |builder| {
        compiler(builder, |builder, re| {
            let builder = builder.clone();
            let (fwd_bytes, _) = re.forward().to_bytes_native_endian();
            let (rev_bytes, _) = re.reverse().to_bytes_native_endian();
            Ok(CompiledRegex::compiled(move |test| -> Vec<TestResult> {
                let fwd: dense::DFA<&[u32], &[u8]> =
                    dense::DFA::from_bytes(&fwd_bytes).unwrap().0;
                let rev: dense::DFA<&[u32], &[u8]> =
                    dense::DFA::from_bytes(&rev_bytes).unwrap().0;
                let re = builder.build_from_dfas(fwd, rev);

                run_test(&re, test)
            }))
        })
    };
    TestRunner::new()?
        .test_iter(suite()?.iter(), my_compiler(builder))
        .assert();
    Ok(())
}

/*
// A basic sanity test that checks we can serialize and then deserialize a
// regex using sparse DFAs, and that the resulting regex can be used for
// searching correctly.
#[test]
fn sparse_serialization_unminimized_default() -> Result<()> {
    let builder = RegexBuilder::new();
    let my_compiler = |builder| {
        compiler(builder, |builder, re| {
            let builder = builder.clone();
            let fwd_bytes = re.forward().to_sparse()?.to_bytes_native_endian();
            let rev_bytes = re.reverse().to_sparse()?.to_bytes_native_endian();
            Ok(CompiledRegex::compiled(move |test| -> Vec<TestResult> {
                let fwd: sparse::DFA<&[u8]> =
                    sparse::DFA::from_bytes(&fwd_bytes).unwrap().0;
                let rev: sparse::DFA<&[u8]> =
                    sparse::DFA::from_bytes(&rev_bytes).unwrap().0;
                let re = builder.build_from_dfas(fwd, rev);

                run_test(&re, test)
            }))
        })
    };
    TestRunner::new()?
        .test_iter(suite()?.iter(), my_compiler(builder))
        .assert();
    Ok(())
}
*/

fn dense_compiler(
    builder: RegexBuilder,
) -> impl FnMut(&RegexTest, &[BString]) -> Result<CompiledRegex> {
    compiler(builder, |_, re| {
        Ok(CompiledRegex::compiled(move |test| -> Vec<TestResult> {
            run_test(&re, test)
        }))
    })
}

/*
fn sparse_compiler(
    builder: RegexBuilder,
) -> impl FnMut(&RegexTest, &[BString]) -> Result<CompiledRegex> {
    compiler(builder, |builder, re| {
        let fwd = re.forward().to_sparse()?;
        let rev = re.reverse().to_sparse()?;
        let re = builder.build_from_dfas(fwd, rev);
        Ok(CompiledRegex::compiled(move |test| -> Vec<TestResult> {
            run_test(&re, test)
        }))
    })
}
*/

fn compiler(
    mut builder: RegexBuilder,
    mut create_matcher: impl FnMut(&RegexBuilder, Regex) -> Result<CompiledRegex>,
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
        create_matcher(&builder, builder.build_many(&regexes)?)
    }
}

fn run_test<A: Automaton>(re: &Regex<A>, test: &RegexTest) -> Vec<TestResult> {
    let is_match = if re.is_match(test.input()) {
        TestResult::matched()
    } else {
        TestResult::no_match()
    };
    let is_match = is_match.name("is_match");

    let find_matches = match test.search_kind() {
        TestSearchKind::Earliest => {
            let it = re
                .find_earliest_iter(test.input())
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
                .find_leftmost_iter(test.input())
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
                .find_overlapping_iter(test.input())
                .take(test.match_limit().unwrap_or(std::usize::MAX))
                .map(|m| Match {
                    id: m.pattern().as_usize(),
                    start: m.start(),
                    end: m.end(),
                });
            TestResult::matches(it).name("find_overlapping_iter")
        }
    };

    vec![is_match, find_matches]
}

fn configure_regex_builder(
    test: &RegexTest,
    builder: &mut RegexBuilder,
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
    let dense_config = dense::Config::new()
        .anchored(test.anchored())
        .match_kind(match_kind)
        .unicode_word_boundary(true);
    let regex_config = RegexConfig::new().utf8(test.utf8());

    builder
        .configure(regex_config)
        .syntax(syntax_config)
        .thompson(config_thompson(test))
        .dense(dense_config);
    true
}

fn config_thompson(test: &RegexTest) -> thompson::Config {
    thompson::Config::new().utf8(test.utf8())
}
