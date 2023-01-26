use std::convert::TryFrom;

use super::{new, Benchmark, Results};

pub(super) fn run(b: &Benchmark) -> anyhow::Result<Results> {
    match &*b.engine {
        "regex/api" => regex_api(b),
        "regex/automata/meta" => regex_automata_meta(b),
        "regex/automata/backtrack" => regex_automata_backtrack(b),
        "regex/automata/pikevm" => regex_automata_pikevm(b),
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
        b.def.capture_count.unwrap() == count,
        "capture count mismatch: expected {} but got {}",
        b.def.capture_count.unwrap(),
        count,
    );
    Ok(())
}

fn regex_api(b: &Benchmark) -> anyhow::Result<Results> {
    let haystack = &*b.haystack;
    let re = new::regex_api(b)?;
    let mut caps = re.capture_locations();
    b.run(verify, || {
        let mut at = 0;
        let mut count = 0;
        while let Some(m) = re.captures_read_at(&mut caps, haystack, at) {
            for i in 0..caps.len() {
                if caps.get(i).is_some() {
                    count += 1;
                }
            }
            // Benchmark definition says we may assume empty matches are
            // impossible.
            at = m.end();
        }
        Ok(count)
    })
}

fn regex_automata_meta(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::Input;

    let mut input = Input::new(&b.haystack);
    let re = new::regex_automata_meta(b)?;
    let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    b.run(verify, || {
        input.set_start(0);
        let mut count = 0;
        while let Some(m) = {
            re.search_captures(&mut cache, &input, &mut caps);
            caps.get_match()
        } {
            for i in 0..caps.group_len() {
                if caps.get_group(i).is_some() {
                    count += 1;
                }
            }
            // Benchmark definition says we may assume empty matches are
            // impossible.
            input.set_start(m.end());
        }
        Ok(count)
    })
}

fn regex_automata_backtrack(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::Input;

    let mut input = Input::new(&b.haystack);
    let re = new::regex_automata_backtrack(b)?;
    let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    b.run(verify, || {
        input.set_start(0);
        let mut count = 0;
        while let Some(m) = {
            re.try_search(&mut cache, &input, &mut caps)?;
            caps.get_match()
        } {
            for i in 0..caps.group_len() {
                if caps.get_group(i).is_some() {
                    count += 1;
                }
            }
            // Benchmark definition says we may assume empty matches are
            // impossible.
            input.set_start(m.end());
        }
        Ok(count)
    })
}

fn regex_automata_pikevm(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::Input;

    let mut input = Input::new(&b.haystack);
    let re = new::regex_automata_pikevm(b)?;
    let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    b.run(verify, || {
        input.set_start(0);
        let mut count = 0;
        while let Some(m) = {
            re.try_search(&mut cache, &input, &mut caps)?;
            caps.get_match()
        } {
            for i in 0..caps.group_len() {
                if caps.get_group(i).is_some() {
                    count += 1;
                }
            }
            // Benchmark definition says we may assume empty matches are
            // impossible.
            input.set_start(m.end());
        }
        Ok(count)
    })
}

#[cfg(feature = "old-regex-crate")]
fn regexold_api(b: &Benchmark) -> anyhow::Result<Results> {
    let haystack = &*b.haystack;
    let re = new::regexold_api(b)?;
    let mut caps = re.capture_locations();
    b.run(verify, || {
        let mut at = 0;
        let mut count = 0;
        while let Some(m) = re.captures_read_at(&mut caps, haystack, at) {
            for i in 0..caps.len() {
                if caps.get(i).is_some() {
                    count += 1;
                }
            }
            // Benchmark definition says we may assume empty matches are
            // impossible.
            at = m.end();
        }
        Ok(count)
    })
}

#[cfg(feature = "extre-re2")]
fn re2_api(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::Input;

    let mut input = Input::new(&b.haystack);
    let re = new::re2_api(b)?;
    let mut caps = re.create_captures();
    b.run(verify, || {
        input.set_start(0);
        let mut count = 0;
        while let Some(m) = {
            re.captures(&input, &mut caps);
            caps.get_match()
        } {
            for i in 0..caps.group_len() {
                if caps.get_group(i).is_some() {
                    count += 1;
                }
            }
            // Benchmark definition says we may assume empty matches are
            // impossible.
            input.set_start(m.end());
        }
        Ok(count)
    })
}

#[cfg(feature = "extre-pcre2")]
fn pcre2_api_jit(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::Input;

    let mut input = Input::new(&b.haystack);
    let re = new::pcre2_api_jit(b)?;
    let mut md = re.create_match_data();
    b.run(verify, || {
        input.set_start(0);
        let mut count = 0;
        while let Some(m) = {
            re.try_find(&input, &mut md)?;
            md.get_match()
        } {
            for i in 0..md.group_len() {
                if md.get_group(i).is_some() {
                    count += 1;
                }
            }
            // Benchmark definition says we may assume empty matches are
            // impossible.
            input.set_start(m.end());
        }
        Ok(count)
    })
}

#[cfg(feature = "extre-pcre2")]
fn pcre2_api_nojit(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::Input;

    let mut input = Input::new(&b.haystack);
    let re = new::pcre2_api_nojit(b)?;
    let mut md = re.create_match_data();
    b.run(verify, || {
        input.set_start(0);
        let mut count = 0;
        while let Some(m) = {
            re.try_find(&input, &mut md)?;
            md.get_match()
        } {
            for i in 0..md.group_len() {
                if md.get_group(i).is_some() {
                    count += 1;
                }
            }
            // Benchmark definition says we may assume empty matches are
            // impossible.
            input.set_start(m.end());
        }
        Ok(count)
    })
}
