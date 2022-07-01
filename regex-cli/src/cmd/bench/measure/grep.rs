use std::convert::TryFrom;

use bstr::ByteSlice;

use automata::SyntaxConfig;

use super::{Benchmark, Results};

pub(super) fn run(b: &Benchmark) -> anyhow::Result<Results> {
    match &*b.engine {
        "regex/api" => regex_api(b),
        "regex/automata/dfa/dense" => regex_automata_dfa_dense(b),
        "regex/automata/dfa/sparse" => regex_automata_dfa_sparse(b),
        "regex/automata/hybrid" => regex_automata_hybrid(b),
        "regex/automata/backtrack" => regex_automata_backtrack(b),
        "regex/automata/pikevm" => regex_automata_pikevm(b),
        name => anyhow::bail!("unknown regex engine '{}'", name),
    }
}

fn verify(b: &Benchmark, count: usize) -> anyhow::Result<()> {
    let count = u64::try_from(count).expect("too many benchmark iterations");
    anyhow::ensure!(
        b.def.line_count.unwrap() == count,
        "line count mismatch: expected {} but got {}",
        b.def.line_count.unwrap(),
        count,
    );
    Ok(())
}

fn regex_api(b: &Benchmark) -> anyhow::Result<Results> {
    use regex::bytes::RegexBuilder;

    let haystack = &*b.haystack;
    let re = RegexBuilder::new(&b.def.regex)
        .unicode(b.def.unicode)
        .case_insensitive(b.def.case_insensitive)
        .build()?;
    b.run(verify, || {
        let mut count = 0;
        for line in haystack.lines() {
            if re.is_match(line) {
                count += 1;
            }
        }
        Ok(count)
    })
}

fn regex_automata_dfa_dense(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::dfa::regex::Regex;

    let haystack = &*b.haystack;
    let re = Regex::builder()
        .configure(Regex::config().utf8(false))
        .syntax(syntax_config(b))
        .build(&b.def.regex)?;
    b.run(verify, || {
        let mut count = 0;
        for line in haystack.lines() {
            if re.is_match(line) {
                count += 1;
            }
        }
        Ok(count)
    })
}

fn regex_automata_dfa_sparse(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::dfa::regex::Regex;

    let haystack = &*b.haystack;
    let re = Regex::builder()
        .configure(Regex::config().utf8(false))
        .syntax(syntax_config(b))
        .build_sparse(&b.def.regex)?;
    b.run(verify, || {
        let mut count = 0;
        for line in haystack.lines() {
            if re.is_match(line) {
                count += 1;
            }
        }
        Ok(count)
    })
}

fn regex_automata_hybrid(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::hybrid::{dfa::DFA, regex::Regex};

    let haystack = &*b.haystack;
    let re = Regex::builder()
        .configure(Regex::config().utf8(false))
        .dfa(DFA::config().skip_cache_capacity_check(true))
        .syntax(syntax_config(b))
        .build(&b.def.regex)?;
    let mut cache = re.create_cache();
    b.run(verify, || {
        let mut count = 0;
        for line in haystack.lines() {
            if re.is_match(&mut cache, line) {
                count += 1;
            }
        }
        Ok(count)
    })
}

fn regex_automata_backtrack(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::nfa::thompson::backtrack::BoundedBacktracker;

    let haystack = &*b.haystack;
    let re = BoundedBacktracker::builder()
        .configure(BoundedBacktracker::config().utf8(false))
        .syntax(syntax_config(b))
        .build(&b.def.regex)?;
    let mut cache = re.create_cache();
    b.run(verify, || {
        let mut count = 0;
        for line in haystack.lines() {
            if re.try_is_match(&mut cache, line)? {
                count += 1;
            }
        }
        Ok(count)
    })
}

fn regex_automata_pikevm(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::nfa::thompson::pikevm::PikeVM;

    let haystack = &*b.haystack;
    let re = PikeVM::builder()
        .configure(PikeVM::config().utf8(false))
        .syntax(syntax_config(b))
        .build(&b.def.regex)?;
    let mut cache = re.create_cache();
    b.run(verify, || {
        let mut count = 0;
        for line in haystack.lines() {
            if re.is_match(&mut cache, line) {
                count += 1;
            }
        }
        Ok(count)
    })
}

/// For regex-automata based regex engines, this builds a syntax configuration
/// from a benchmark definition.
fn syntax_config(b: &Benchmark) -> SyntaxConfig {
    SyntaxConfig::new()
        .utf8(false)
        .unicode(b.def.unicode)
        .case_insensitive(b.def.case_insensitive)
}
