use std::convert::TryFrom;

use automata::SyntaxConfig;

use super::{Benchmark, Results};

pub(super) fn run(b: &Benchmark) -> anyhow::Result<Results> {
    match &*b.engine {
        "regex/api" => regex_api(b),
        "regex/automata/dfa/dense" => regex_automata_dfa_dense(b),
        "regex/automata/dfa/sparse" => regex_automata_dfa_sparse(b),
        "regex/automata/hybrid" => regex_automata_hybrid(b),
        "regex/automata/pikevm" => regex_automata_pikevm(b),
        name => anyhow::bail!("unknown regex engine '{}'", name),
    }
}

fn verify(
    b: &Benchmark,
    mut findall: Box<dyn FnMut(&[u8]) -> anyhow::Result<usize>>,
) -> anyhow::Result<()> {
    let count = u64::try_from(findall(&b.haystack)?)
        .expect("too many benchmark iterations");
    anyhow::ensure!(
        b.def.match_count.unwrap() == count,
        "count mismatch: expected {} but got {}",
        b.def.match_count.unwrap(),
        count,
    );
    Ok(())
}

fn regex_api(b: &Benchmark) -> anyhow::Result<Results> {
    use regex::bytes::RegexBuilder;

    b.run(verify, || {
        let re = RegexBuilder::new(&b.def.regex)
            .unicode(b.def.unicode)
            .case_insensitive(b.def.case_insensitive)
            .build()?;
        Ok(Box::new(move |h| Ok(re.find_iter(h).count())))
    })
}

fn regex_automata_dfa_dense(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::dfa::regex::Regex;

    b.run(verify, || {
        let re = Regex::builder()
            .configure(Regex::config().utf8(false))
            .syntax(syntax_config(b))
            .build(&b.def.regex)?;
        Ok(Box::new(move |h| Ok(re.find_iter(h).count())))
    })
}

fn regex_automata_dfa_sparse(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::dfa::regex::Regex;

    b.run(verify, || {
        let re = Regex::builder()
            .configure(Regex::config().utf8(false))
            .syntax(syntax_config(b))
            .build_sparse(&b.def.regex)?;
        Ok(Box::new(move |h| Ok(re.find_iter(h).count())))
    })
}

fn regex_automata_hybrid(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::hybrid::{dfa::DFA, regex::Regex};

    b.run(verify, || {
        let re = Regex::builder()
            .configure(Regex::config().utf8(false))
            .dfa(DFA::config().skip_cache_capacity_check(true))
            .syntax(syntax_config(b))
            .build(&b.def.regex)?;
        let mut cache = re.create_cache();
        Ok(Box::new(move |h| Ok(re.find_iter(&mut cache, h).count())))
    })
}

fn regex_automata_pikevm(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::nfa::thompson::pikevm::PikeVM;

    b.run(verify, || {
        let re = PikeVM::builder()
            .configure(PikeVM::config().utf8(false))
            .syntax(syntax_config(b))
            .build(&b.def.regex)?;
        let mut cache = re.create_cache();
        Ok(Box::new(move |h| Ok(re.find_iter(&mut cache, h).count())))
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
