use regex_automata::{
    dfa::{
        self, dense, regex::Regex, sparse, Automaton, OverlappingState,
        StartKind,
    },
    nfa::thompson,
    util::{iter, prefilter::Prefilter, syntax},
    Anchored, Input, PatternSet,
};
use regex_syntax::hir;

use ret::{
    bstr::{BString, ByteSlice},
    CompiledRegex, RegexTest, TestResult, TestRunner,
};

use crate::{create_input, suite, untestify_kind, Result};

const EXPANSIONS: &[&str] = &["is_match", "find", "which"];

/// Runs the test suite with the default configuration.
#[test]
fn unminimized_default() -> Result<()> {
    let builder = Regex::builder();
    TestRunner::new()?
        .expand(EXPANSIONS, |t| t.compiles())
        .blacklist("expensive")
        .test_iter(suite()?.iter(), dense_compiler(builder))
        .assert();
    Ok(())
}

/// Runs the test suite with the default configuration and a prefilter enabled,
/// if one can be built.
#[test]
fn unminimized_prefilter() -> Result<()> {
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
        let mut builder = Regex::builder();
        builder.dense(dense::DFA::config().prefilter(pre));
        compiler(builder, |_, _, re| {
            Ok(CompiledRegex::compiled(move |test| -> TestResult {
                run_test(&re, test)
            }))
        })(test, regexes)
    };
    TestRunner::new()?
        .expand(EXPANSIONS, |t| t.compiles())
        .blacklist("expensive")
        .test_iter(suite()?.iter(), my_compiler)
        .assert();
    Ok(())
}

/// Runs the test suite with start states specialized.
#[test]
fn unminimized_specialized_start_states() -> Result<()> {
    let mut builder = Regex::builder();
    builder.dense(dense::Config::new().specialize_start_states(true));

    TestRunner::new()?
        .expand(EXPANSIONS, |t| t.compiles())
        .blacklist("expensive")
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
        .blacklist("expensive")
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
        .blacklist("expensive")
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
        .blacklist("expensive")
        .test_iter(suite()?.iter(), sparse_compiler(builder))
        .assert();
    Ok(())
}

/// Runs the test suite on a sparse unminimized DFA with prefilters enabled.
#[test]
fn sparse_unminimized_prefilter() -> Result<()> {
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
        let mut builder = Regex::builder();
        builder.dense(dense::DFA::config().prefilter(pre));
        compiler(builder, |builder, _, re| {
            let fwd = re.forward().to_sparse()?;
            let rev = re.reverse().to_sparse()?;
            let re = builder.build_from_dfas(fwd, rev);
            Ok(CompiledRegex::compiled(move |test| -> TestResult {
                run_test(&re, test)
            }))
        })(test, regexes)
    };
    TestRunner::new()?
        .expand(EXPANSIONS, |t| t.compiles())
        .blacklist("expensive")
        .test_iter(suite()?.iter(), my_compiler)
        .assert();
    Ok(())
}

/// Another basic sanity test that checks we can serialize and then deserialize
/// a regex, and that the resulting regex can be used for searching correctly.
#[test]
fn serialization_unminimized_default() -> Result<()> {
    let builder = Regex::builder();
    let my_compiler = |builder| {
        compiler(builder, |builder, _, re| {
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
        .blacklist("expensive")
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
        compiler(builder, |builder, _, re| {
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
        .blacklist("expensive")
        .test_iter(suite()?.iter(), my_compiler(builder))
        .assert();
    Ok(())
}

fn dense_compiler(
    builder: dfa::regex::Builder,
) -> impl FnMut(&RegexTest, &[BString]) -> Result<CompiledRegex> {
    compiler(builder, |_, _, re| {
        Ok(CompiledRegex::compiled(move |test| -> TestResult {
            run_test(&re, test)
        }))
    })
}

fn sparse_compiler(
    builder: dfa::regex::Builder,
) -> impl FnMut(&RegexTest, &[BString]) -> Result<CompiledRegex> {
    compiler(builder, |builder, _, re| {
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
        Option<Prefilter>,
        Regex,
    ) -> Result<CompiledRegex>,
) -> impl FnMut(&RegexTest, &[BString]) -> Result<CompiledRegex> {
    move |test, regexes| {
        let regexes = regexes
            .iter()
            .map(|r| r.to_str().map(|s| s.to_string()))
            .collect::<std::result::Result<Vec<String>, _>>()?;

        // Parse regexes as HIRs for some analysis below.
        let mut hirs = vec![];
        for pattern in regexes.iter() {
            hirs.push(syntax::parse(&config_syntax(test), pattern)?);
        }

        // Get a prefilter in case the test wants it.
        let kind = match untestify_kind(test.match_kind()) {
            None => return Ok(CompiledRegex::skip()),
            Some(kind) => kind,
        };
        let pre = Prefilter::from_hirs(kind, &hirs);

        // Check if our regex contains things that aren't supported by DFAs.
        // That is, Unicode word boundaries when searching non-ASCII text.
        if !test.input().is_ascii() {
            for hir in hirs.iter() {
                let looks = hir.properties().look_set();
                if looks.contains(hir::Look::WordUnicode)
                    || looks.contains(hir::Look::WordUnicodeNegate)
                {
                    return Ok(CompiledRegex::skip());
                }
            }
        }
        if !configure_regex_builder(test, &mut builder) {
            return Ok(CompiledRegex::skip());
        }
        create_matcher(&builder, pre, builder.build_many(&regexes)?)
    }
}

fn run_test<A: Automaton>(re: &Regex<A>, test: &RegexTest) -> TestResult {
    let input = create_input(test, |h| re.create_input(h));
    match test.additional_name() {
        "is_match" => {
            // TODO: Use 'is_match' API once it accetps 'Into<Input>'.
            let input = input.earliest(true);
            TestResult::matched(
                re.forward().try_search_fwd(&input).unwrap().is_some(),
            )
        }
        "find" => match test.search_kind() {
            ret::SearchKind::Earliest | ret::SearchKind::Leftmost => {
                let input = input
                    .earliest(test.search_kind() == ret::SearchKind::Earliest);
                let it = iter::Searcher::new(input)
                    .into_matches_iter(|input| re.try_search(input))
                    .infallible()
                    .take(test.match_limit().unwrap_or(std::usize::MAX))
                    .map(|m| ret::Match {
                        id: m.pattern().as_usize(),
                        span: ret::Span { start: m.start(), end: m.end() },
                    });
                TestResult::matches(it)
            }
            ret::SearchKind::Overlapping => {
                try_search_overlapping(re, &input).unwrap()
            }
        },
        "which" => match test.search_kind() {
            ret::SearchKind::Earliest | ret::SearchKind::Leftmost => {
                // There are no "which" APIs for standard searches.
                TestResult::skip()
            }
            ret::SearchKind::Overlapping => {
                let dfa = re.forward();
                let mut patset = PatternSet::new(dfa.pattern_len());
                dfa.try_which_overlapping_matches(&input, &mut patset)
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
    let match_kind = match untestify_kind(test.match_kind()) {
        None => return false,
        Some(k) => k,
    };

    let starts = if test.anchored() {
        StartKind::Anchored
    } else {
        StartKind::Unanchored
    };
    let mut dense_config = dense::Config::new()
        .start_kind(starts)
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

    builder
        .syntax(config_syntax(test))
        .thompson(config_thompson(test))
        .dense(dense_config);
    true
}

/// Configuration of a Thompson NFA compiler from a regex test.
fn config_thompson(test: &RegexTest) -> thompson::Config {
    thompson::Config::new().utf8(test.utf8())
}

/// Configuration of the regex syntax from a regex test.
fn config_syntax(test: &RegexTest) -> syntax::Config {
    syntax::Config::new()
        .case_insensitive(test.case_insensitive())
        .unicode(test.unicode())
        .utf8(test.utf8())
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
    while let Some(end) = {
        fwd_dfa.try_search_overlapping_fwd(input, &mut fwd_state)?;
        fwd_state.get_match()
    } {
        let revsearch = input
            .clone()
            .range(input.start()..end.offset())
            .anchored(Anchored::Pattern(end.pattern()))
            .earliest(false);
        let mut rev_state = OverlappingState::start();
        while let Some(start) = {
            rev_dfa.try_search_overlapping_rev(&revsearch, &mut rev_state)?;
            rev_state.get_match()
        } {
            // let start = rev_dfa
            // .try_search_rev(rev_cache, &revsearch)?
            // .expect("reverse search must match if forward search does");
            let span = ret::Span { start: start.offset(), end: end.offset() };
            let mat = ret::Match { id: end.pattern().as_usize(), span };
            matches.push(mat);
        }
    }
    Ok(TestResult::matches(matches))
}
