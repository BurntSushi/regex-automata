use std::convert::TryFrom;

use bstr::ByteSlice;

use super::{new, Benchmark, Results};

pub(super) fn run(b: &Benchmark) -> anyhow::Result<Results> {
    match &*b.engine {
        "regex/api" => regex_api(b),
        "regex/automata/dfa/dense" => regex_automata_dfa_dense(b),
        "regex/automata/dfa/sparse" => regex_automata_dfa_sparse(b),
        "regex/automata/hybrid" => regex_automata_hybrid(b),
        "regex/automata/backtrack" => regex_automata_backtrack(b),
        "regex/automata/pikevm" => regex_automata_pikevm(b),
        "regex/automata/onepass" => regex_automata_onepass(b),
        #[cfg(feature = "extre-re2")]
        "re2/api" => re2_api(b),
        #[cfg(feature = "extre-pcre2")]
        "pcre2/api/jit" => pcre2_api_jit(b),
        #[cfg(feature = "extre-pcre2")]
        "pcre2/api/nojit" => pcre2_api_nojit(b),
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
    let haystack = &*b.haystack;
    let re = new::regex_api(b)?;
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
    let haystack = &*b.haystack;
    let re = new::regex_automata_dfa_dense(b)?;
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
    let haystack = &*b.haystack;
    let re = new::regex_automata_dfa_sparse(b)?;
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
    let haystack = &*b.haystack;
    let re = new::regex_automata_hybrid(b)?;
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
    let haystack = &*b.haystack;
    let re = new::regex_automata_backtrack(b)?;
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
    let haystack = &*b.haystack;
    let re = new::regex_automata_pikevm(b)?;
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

fn regex_automata_onepass(b: &Benchmark) -> anyhow::Result<Results> {
    let haystack = &*b.haystack;
    let re = new::regex_automata_onepass(b)?;
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

#[cfg(feature = "extre-re2")]
fn re2_api(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::Input;

    let haystack = &*b.haystack;
    let re = new::re2_api(b)?;
    b.run(verify, || {
        let mut count = 0;
        for line in haystack.lines() {
            if re.is_match(&Input::new(line)) {
                count += 1;
            }
        }
        Ok(count)
    })
}

#[cfg(feature = "extre-pcre2")]
fn pcre2_api_jit(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::Input;

    let haystack = &*b.haystack;
    let re = new::pcre2_api_jit(b)?;
    let mut md = re.create_match_data_for_matches_only();
    b.run(verify, || {
        let mut count = 0;
        for line in haystack.lines() {
            if re.try_find(&Input::new(line), &mut md)? {
                count += 1;
            }
        }
        Ok(count)
    })
}

#[cfg(feature = "extre-pcre2")]
fn pcre2_api_nojit(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::Input;

    let haystack = &*b.haystack;
    let re = new::pcre2_api_nojit(b)?;
    let mut md = re.create_match_data_for_matches_only();
    b.run(verify, || {
        let mut count = 0;
        for line in haystack.lines() {
            if re.try_find(&Input::new(line), &mut md)? {
                count += 1;
            }
        }
        Ok(count)
    })
}
