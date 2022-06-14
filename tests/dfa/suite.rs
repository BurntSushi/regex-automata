use regex_automata::{
    dfa::{self, dense, regex::Regex, sparse, Automaton, OverlappingState},
    nfa::thompson,
    util::iter,
    Input, MatchKind, PatternSet, SyntaxConfig,
};

use ret::{
    bstr::{BString, ByteSlice},
    CompiledRegex, RegexTest, TestResult, TestRunner,
};

use crate::{suite, Result};

const EXPANSIONS: &[&str] = &["is_match", "find", "which"];

/// Runs the test suite with the default configuration.
#[test]
fn unminimized_default() -> Result<()> {
    let builder = Regex::builder();
    TestRunner::new()?
        .expand(EXPANSIONS, |t| t.compiles())
        .test_iter(suite()?.iter(), dense_compiler(builder))
        .assert();
    Ok(())
}

/// Runs the test suite with byte classes disabled.
#[test]
fn unminimized_no_byte_class() -> Result<()> {
    let mut builder = Regex::builder();
    builder.dense(dense::Config::new().byte_classes(false));

    TestRunner::new()?
        .expand(EXPANSIONS, |t| t.compiles())
        .test_iter(suite()?.iter(), dense_compiler(builder))
        .assert();
    Ok(())
}

/// Runs the test suite with NFA shrinking enabled.
#[test]
fn unminimized_nfa_shrink() -> Result<()> {
    let mut builder = Regex::builder();
    builder.thompson(thompson::Config::new().shrink(true));

    TestRunner::new()?
        .expand(EXPANSIONS, |t| t.compiles())
        .test_iter(suite()?.iter(), dense_compiler(builder))
        .assert();
    Ok(())
}

/// Runs the test suite on a minimized DFA with an otherwise default
/// configuration.
#[test]
fn minimized_default() -> Result<()> {
    let mut builder = Regex::builder();
    builder.dense(dense::Config::new().minimize(true));
    TestRunner::new()?
        .expand(EXPANSIONS, |t| t.compiles())
        // These regexes tend to be too big. Minimization takes... forever.
        .blacklist("expensive")
        .test_iter(suite()?.iter(), dense_compiler(builder))
        .assert();
    Ok(())
}

/// Runs the test suite on a minimized DFA with byte classes disabled.
#[test]
fn minimized_no_byte_class() -> Result<()> {
    let mut builder = Regex::builder();
    builder.dense(dense::Config::new().minimize(true).byte_classes(false));

    TestRunner::new()?
        .expand(EXPANSIONS, |t| t.compiles())
        // These regexes tend to be too big. Minimization takes... forever.
        .blacklist("expensive")
        .test_iter(suite()?.iter(), dense_compiler(builder))
        .assert();
    Ok(())
}

/// Runs the test suite on a sparse unminimized DFA.
#[test]
fn sparse_unminimized_default() -> Result<()> {
    let builder = Regex::builder();
    TestRunner::new()?
        .expand(EXPANSIONS, |t| t.compiles())
        .test_iter(suite()?.iter(), sparse_compiler(builder))
        .assert();
    Ok(())
}

/// Another basic sanity test that checks we can serialize and then deserialize
/// a regex, and that the resulting regex can be used for searching correctly.
#[test]
fn serialization_unminimized_default() -> Result<()> {
    let builder = Regex::builder();
    let my_compiler = |builder| {
        compiler(builder, |builder, re| {
            let builder = builder.clone();
            let (fwd_bytes, _) = re.forward().to_bytes_native_endian();
            let (rev_bytes, _) = re.reverse().to_bytes_native_endian();
            Ok(CompiledRegex::compiled(move |test| -> TestResult {
                let fwd: dense::DFA<&[u32]> =
                    dense::DFA::from_bytes(&fwd_bytes).unwrap().0;
                let rev: dense::DFA<&[u32]> =
                    dense::DFA::from_bytes(&rev_bytes).unwrap().0;
                let re = builder.build_from_dfas(fwd, rev);

                run_test(&re, test)
            }))
        })
    };
    TestRunner::new()?
        .expand(EXPANSIONS, |t| t.compiles())
        .test_iter(suite()?.iter(), my_compiler(builder))
        .assert();
    Ok(())
}

/// A basic sanity test that checks we can serialize and then deserialize a
/// regex using sparse DFAs, and that the resulting regex can be used for
/// searching correctly.
#[test]
fn sparse_serialization_unminimized_default() -> Result<()> {
    let builder = Regex::builder();
    let my_compiler = |builder| {
        compiler(builder, |builder, re| {
            let builder = builder.clone();
            let fwd_bytes = re.forward().to_sparse()?.to_bytes_native_endian();
            let rev_bytes = re.reverse().to_sparse()?.to_bytes_native_endian();
            Ok(CompiledRegex::compiled(move |test| -> TestResult {
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
        .expand(EXPANSIONS, |t| t.compiles())
        .test_iter(suite()?.iter(), my_compiler(builder))
        .assert();
    Ok(())
}

fn dense_compiler(
    builder: dfa::regex::Builder,
) -> impl FnMut(&RegexTest, &[BString]) -> Result<CompiledRegex> {
    compiler(builder, |_, re| {
        Ok(CompiledRegex::compiled(move |test| -> TestResult {
            run_test(&re, test)
        }))
    })
}

fn sparse_compiler(
    builder: dfa::regex::Builder,
) -> impl FnMut(&RegexTest, &[BString]) -> Result<CompiledRegex> {
    compiler(builder, |builder, re| {
        let fwd = re.forward().to_sparse()?;
        let rev = re.reverse().to_sparse()?;
        let re = builder.build_from_dfas(fwd, rev);
        Ok(CompiledRegex::compiled(move |test| -> TestResult {
            run_test(&re, test)
        }))
    })
}

fn compiler(
    mut builder: dfa::regex::Builder,
    mut create_matcher: impl FnMut(
        &dfa::regex::Builder,
        Regex,
    ) -> Result<CompiledRegex>,
) -> impl FnMut(&RegexTest, &[BString]) -> Result<CompiledRegex> {
    move |test, regexes| {
        let regexes = regexes
            .iter()
            .map(|r| r.to_str().map(|s| s.to_string()))
            .collect::<std::result::Result<Vec<String>, _>>()?;

        // Check if our regex contains things that aren't supported by DFAs.
        // That is, Unicode word boundaries when searching non-ASCII text.
        let mut thompson = thompson::Compiler::new();
        thompson.configure(config_thompson(test));
        // TODO: Modify Hir to report facts like this, instead of needing to
        // build an NFA to do it.
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

fn run_test<A: Automaton>(re: &Regex<A>, test: &RegexTest) -> TestResult {
    match test.additional_name() {
        "is_match" => TestResult::matched(re.is_match(test.input())),
        "find" => match test.search_kind() {
            ret::SearchKind::Earliest => {
                let search = re.create_input(test.input()).earliest(true);
                let it = iter::TryMatches::new(search, move |search| {
                    re.try_search(search)
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
                let it = re
                    .find_iter(test.input())
                    .take(test.match_limit().unwrap_or(std::usize::MAX))
                    .map(|m| ret::Match {
                        id: m.pattern().as_usize(),
                        span: ret::Span { start: m.start(), end: m.end() },
                    });
                TestResult::matches(it)
            }
            ret::SearchKind::Overlapping => {
                let search = re.create_input(test.input());
                try_search_overlapping(re, &search).unwrap()
            }
        },
        "which" => match test.search_kind() {
            ret::SearchKind::Earliest | ret::SearchKind::Leftmost => {
                // There are no "which" APIs for standard searches. So this is
                // technically redundant, but we produce a result anyway.
                let mut pids: Vec<usize> = re
                    .find_iter(test.input())
                    .map(|m| m.pattern().as_usize())
                    .collect();
                pids.sort();
                pids.dedup();
                TestResult::which(pids)
            }
            ret::SearchKind::Overlapping => {
                let dfa = re.forward();
                let mut patset = PatternSet::new(dfa.pattern_len());
                let search = re.create_input(test.input());
                dfa.try_which_overlapping_matches(&search, &mut patset)
                    .unwrap();
                TestResult::which(patset.iter().map(|p| p.as_usize()))
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
    builder: &mut dfa::regex::Builder,
) -> bool {
    let match_kind = match test.match_kind() {
        ret::MatchKind::All => MatchKind::All,
        ret::MatchKind::LeftmostFirst => MatchKind::LeftmostFirst,
        ret::MatchKind::LeftmostLongest => return false,
    };

    let syntax_config = SyntaxConfig::new()
        .case_insensitive(test.case_insensitive())
        .unicode(test.unicode())
        .utf8(test.utf8());
    let mut dense_config = dense::Config::new()
        .anchored(test.anchored())
        .match_kind(match_kind)
        .unicode_word_boundary(true);
    // When doing an overlapping search, we might try to find the start of each
    // match with a custom search routine. In that case, we need to tell the
    // reverse search (for the start offset) which pattern to look for. The
    // only way that API works is when anchored starting states are compiled
    // for each pattern. This does technically also enable it for the forward
    // DFA, but we're okay with that.
    if test.search_kind() == ret::SearchKind::Overlapping {
        dense_config = dense_config.starts_for_each_pattern(true);
    }
    let regex_config = Regex::config().utf8(test.utf8());

    builder
        .configure(regex_config)
        .syntax(syntax_config)
        .thompson(config_thompson(test))
        .dense(dense_config);
    true
}

/// Configuration of a Thompson NFA compiler from a regex test.
fn config_thompson(test: &RegexTest) -> thompson::Config {
    thompson::Config::new().utf8(test.utf8())
}

/// Execute an overlapping search, and for each match found, also find its
/// overlapping starting positions.
///
/// N.B. This routine used to be part of the crate API, but 1) it wasn't clear
/// to me how useful it was and 2) it wasn't clear to me what its semantics
/// should be. In particular, a potentially surprising footgun of this routine
/// that it is worst case *quadratic* in the size of the haystack. Namely, it's
/// possible to report a match at every position, and for every such position,
/// scan all the way to the beginning of the haystack to find the starting
/// position. Typical leftmost non-overlapping searches don't suffer from this
/// because, well, matches can't overlap. So subsequent searches after a match
/// is found don't revisit previously scanned parts of the haystack.
///
/// Its semantics can be strange for other reasons too. For example, given
/// the regex '.*' and the haystack 'zz', the full set of overlapping matches
/// is: [0, 0], [1, 1], [0, 1], [2, 2], [1, 2], [0, 2]. The ordering of
/// those matches is quite strange, but makes sense when you think about the
/// implementation: an end offset is found left-to-right, and then one or more
/// starting offsets are found right-to-left.
///
/// Nevertheless, we provide this routine in our test suite because it's
/// useful to test the low level DFA overlapping search and our test suite
/// is written in a way that requires starting offsets.
fn try_search_overlapping<A: Automaton>(
    re: &Regex<A>,
    input: &Input<'_, '_>,
) -> Result<TestResult> {
    let mut matches = vec![];
    let mut fwd_state = OverlappingState::start();
    let (fwd_dfa, rev_dfa) = (re.forward(), re.reverse());
    while let Some(end) =
        fwd_dfa.try_search_overlapping_fwd(input, &mut fwd_state)?
    {
        let revsearch = input
            .clone()
            .pattern(Some(end.pattern()))
            .earliest(false)
            .range(input.start()..end.offset());
        let mut rev_state = OverlappingState::start();
        while let Some(start) =
            rev_dfa.try_search_overlapping_rev(&revsearch, &mut rev_state)?
        {
            // let start = rev_dfa
            // .try_search_rev(rev_cache, &revsearch)?
            // .expect("reverse search must match if forward search does");
            let span = ret::Span { start: start.offset(), end: end.offset() };
            // Some tests check that we don't yield matches that split a
            // codepoint when UTF-8 mode is enabled, so skip those here.
            if input.get_utf8()
                && span.start == span.end
                && !input.is_char_boundary(span.end)
            {
                continue;
            }
            let mat = ret::Match { id: end.pattern().as_usize(), span };
            dbg!(&mat);
            matches.push(mat);
        }
    }
    Ok(TestResult::matches(matches))
}
