use super::{new, Benchmark, Results};

pub(super) fn run(b: &Benchmark) -> anyhow::Result<Results> {
    match &*b.engine {
        "regex/api" => regex_api(b),
        "regex/automata/dense" => regex_automata_dfa_dense(b),
        "regex/automata/sparse" => regex_automata_dfa_sparse(b),
        "regex/automata/hybrid" => regex_automata_hybrid(b),
        "regex/automata/backtrack" => regex_automata_backtrack(b),
        "regex/automata/pikevm" => regex_automata_pikevm(b),
        "memchr/memmem" => memchr_memmem(b),
        "aho-corasick/dfa" => aho_corasick_dfa(b),
        "aho-corasick/nfa" => aho_corasick_nfa(b),
        #[cfg(feature = "old-regex-crate")]
        "regexold/api" => regexold_api(b),
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
        b.def.match_count.unwrap() == count,
        "match count mismatch: expected {} but got {}",
        b.def.match_count.unwrap(),
        count,
    );
    Ok(())
}

fn regex_api(b: &Benchmark) -> anyhow::Result<Results> {
    let haystack = &*b.haystack;
    let re = new::regex_api(b)?;
    b.run(verify, || Ok(re.find_iter(haystack).count()))
}

fn regex_automata_dfa_dense(b: &Benchmark) -> anyhow::Result<Results> {
    let haystack = &*b.haystack;
    let re = new::regex_automata_dfa_dense(b)?;
    b.run(verify, || Ok(re.find_iter(haystack).count()))
}

fn regex_automata_dfa_sparse(b: &Benchmark) -> anyhow::Result<Results> {
    let haystack = &*b.haystack;
    let re = new::regex_automata_dfa_sparse(b)?;
    b.run(verify, || Ok(re.find_iter(haystack).count()))
}

fn regex_automata_hybrid(b: &Benchmark) -> anyhow::Result<Results> {
    let haystack = &*b.haystack;
    let re = new::regex_automata_hybrid(b)?;
    let mut cache = re.create_cache();
    b.run(verify, || Ok(re.find_iter(&mut cache, haystack).count()))
}

fn regex_automata_backtrack(b: &Benchmark) -> anyhow::Result<Results> {
    let haystack = &*b.haystack;
    let re = new::regex_automata_backtrack(b)?;
    let mut cache = re.create_cache();
    b.run(verify, || {
        // We could check the haystack length against
        // 'backtrack::min_visited_capacity' and return an error before running
        // our benchmark, but handling the error at search time is probably
        // more consistent with real world usage. Some brief experiments don't
        // seem to show much of a difference between this and the panicking
        // APIs.
        let mut count = 0;
        for result in re.try_find_iter(&mut cache, haystack) {
            result?;
            count += 1;
        }
        Ok(count)
    })
}

fn regex_automata_pikevm(b: &Benchmark) -> anyhow::Result<Results> {
    let haystack = &*b.haystack;
    let re = new::regex_automata_pikevm(b)?;
    let mut cache = re.create_cache();
    b.run(verify, || Ok(re.find_iter(&mut cache, haystack).count()))
}

fn memchr_memmem(b: &Benchmark) -> anyhow::Result<Results> {
    let haystack = &*b.haystack;
    let finder = new::memchr_memmem(b)?;
    b.run(verify, || Ok(finder.find_iter(haystack).count()))
}

fn aho_corasick_dfa(b: &Benchmark) -> anyhow::Result<Results> {
    let haystack = &*b.haystack;
    let finder = new::aho_corasick_dfa(b)?;
    b.run(verify, || Ok(finder.find_iter(haystack).count()))
}

fn aho_corasick_nfa(b: &Benchmark) -> anyhow::Result<Results> {
    let haystack = &*b.haystack;
    let finder = new::aho_corasick_nfa(b)?;
    b.run(verify, || Ok(finder.find_iter(haystack).count()))
}

#[cfg(feature = "old-regex-crate")]
fn regexold_api(b: &Benchmark) -> anyhow::Result<Results> {
    let haystack = &*b.haystack;
    let re = new::regexold_api(b)?;
    b.run(verify, || Ok(re.find_iter(haystack).count()))
}

#[cfg(feature = "extre-re2")]
fn re2_api(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::Input;

    let haystack = &*b.haystack;
    let re = new::re2_api(b)?;
    b.run(verify, || Ok(re.find_iter(Input::new(haystack)).count()))
}

#[cfg(feature = "extre-pcre2")]
fn pcre2_api_jit(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::Input;

    let haystack = &*b.haystack;
    let re = new::pcre2_api_jit(b)?;
    let mut md = re.create_match_data_for_matches_only();
    b.run(verify, || {
        // We do the same thing for our bounded backtracker. Namely, we check
        // that no error has occurred on every search. In general, I expect
        // this to match real world conditions since it is difficult to predict
        // when a pure backtracking regex engine will return an error.
        let mut count = 0;
        for result in re.try_find_iter(Input::new(haystack), &mut md) {
            result?;
            count += 1;
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
        // We do the same thing for our bounded backtracker. Namely, we check
        // that no error has occurred on every search. In general, I expect
        // this to match real world conditions since it is difficult to predict
        // when a pure backtracking regex engine will return an error.
        let mut count = 0;
        for result in re.try_find_iter(Input::new(haystack), &mut md) {
            result?;
            count += 1;
        }
        Ok(count)
    })
}
