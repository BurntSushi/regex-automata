use std::convert::TryFrom;

use super::{Benchmark, Results};

fn verify(b: &Benchmark, count: usize) -> anyhow::Result<()> {
    let count = u64::try_from(count).expect("too many benchmark iterations");
    anyhow::ensure!(
        b.def.match_count.unwrap() == count,
        "match count mismatch: expected {} but got {}",
        b.def.match_count.unwrap(),
        count,
    );
    Ok(())
}

pub(super) fn regex_api(b: &Benchmark) -> anyhow::Result<Results> {
    use regex::bytes::Regex;

    let haystack = &*b.haystack;
    let re = Regex::new(&b.def.regex)?;
    b.run(verify, || Ok(re.find_iter(haystack).count()))
}

pub(super) fn regex_automata_dfa_dense(
    b: &Benchmark,
) -> anyhow::Result<Results> {
    use automata::dfa::regex::Regex;

    let haystack = &*b.haystack;
    let re = Regex::new(&b.def.regex)?;
    b.run(verify, || Ok(re.find_iter(haystack).count()))
}

pub(super) fn regex_automata_dfa_sparse(
    b: &Benchmark,
) -> anyhow::Result<Results> {
    use automata::dfa::regex::Regex;

    let haystack = &*b.haystack;
    let re = Regex::new_sparse(&b.def.regex)?;
    b.run(verify, || Ok(re.find_iter(haystack).count()))
}

pub(super) fn regex_automata_hybrid(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::hybrid::regex::Regex;

    let haystack = &*b.haystack;
    let re = Regex::new(&b.def.regex)?;
    let mut cache = re.create_cache();
    b.run(verify, || Ok(re.find_iter(&mut cache, haystack).count()))
}

pub(super) fn regex_automata_pikevm(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::nfa::thompson::pikevm::PikeVM;

    let haystack = &*b.haystack;
    let re = PikeVM::new(&b.def.regex)?;
    let mut cache = re.create_cache();
    b.run(verify, || Ok(re.find_iter(&mut cache, haystack).count()))
}

pub(super) fn memchr_memmem(b: &Benchmark) -> anyhow::Result<Results> {
    use memchr::memmem;

    let haystack = &*b.haystack;
    let finder = memmem::Finder::new(&b.def.regex);
    b.run(verify, || Ok(finder.find_iter(haystack).count()))
}
