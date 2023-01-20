use bstr::ByteSlice;

use automata::{Anchored, Input};

use super::{new, Benchmark, Results};

pub(super) fn run(b: &Benchmark) -> anyhow::Result<Results> {
    match &*b.engine {
        "regex/api" => regex_api(b),
        "regex/automata/meta" => regex_automata_meta(b),
        "regex/automata/backtrack" => regex_automata_backtrack(b),
        "regex/automata/pikevm" => regex_automata_pikevm(b),
        "regex/automata/onepass" => regex_automata_onepass(b),
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

fn verify(
    b: &Benchmark,
    (line_count, capture_count): (u64, u64),
) -> anyhow::Result<()> {
    anyhow::ensure!(
        b.def.line_count.unwrap() == line_count,
        "line count mismatch: expected {} but got {}",
        b.def.line_count.unwrap(),
        line_count,
    );
    anyhow::ensure!(
        b.def.capture_count.unwrap() == capture_count,
        "capture count mismatch: expected {} but got {}",
        b.def.capture_count.unwrap(),
        capture_count,
    );
    Ok(())
}

fn regex_api(b: &Benchmark) -> anyhow::Result<Results> {
    let haystack = &*b.haystack;
    let re = new::regex_api(b)?;
    let mut caps = re.capture_locations();
    b.run(verify, || {
        let (mut line_count, mut capture_count) = (0, 0);
        for line in haystack.lines() {
            let mut at = 0;
            let mut count = 0;
            while let Some(m) = re.captures_read_at(&mut caps, line, at) {
                for i in 0..caps.len() {
                    if caps.get(i).is_some() {
                        count += 1;
                    }
                }
                // Benchmark definition says we may assume empty matches are
                // impossible.
                at = m.end();
            }
            capture_count += count;
            if count > 0 {
                line_count += 1;
            }
        }
        Ok((line_count, capture_count))
    })
}

fn regex_automata_meta(b: &Benchmark) -> anyhow::Result<Results> {
    let haystack = &*b.haystack;
    let re = new::regex_automata_meta(b)?;
    let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    b.run(verify, || {
        let (mut line_count, mut capture_count) = (0, 0);
        for line in haystack.lines() {
            let mut count = 0;
            let mut input = Input::new(line);
            while let Some(m) = {
                re.try_search_captures(&mut cache, &input, &mut caps)?;
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
            capture_count += count;
            if count > 0 {
                line_count += 1;
            }
        }
        Ok((line_count, capture_count))
    })
}

fn regex_automata_backtrack(b: &Benchmark) -> anyhow::Result<Results> {
    let haystack = &*b.haystack;
    let re = new::regex_automata_backtrack(b)?;
    let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    b.run(verify, || {
        let (mut line_count, mut capture_count) = (0, 0);
        for line in haystack.lines() {
            let mut count = 0;
            let mut input = Input::new(line);
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
            capture_count += count;
            if count > 0 {
                line_count += 1;
            }
        }
        Ok((line_count, capture_count))
    })
}

fn regex_automata_pikevm(b: &Benchmark) -> anyhow::Result<Results> {
    let haystack = &*b.haystack;
    let re = new::regex_automata_pikevm(b)?;
    let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    b.run(verify, || {
        let (mut line_count, mut capture_count) = (0, 0);
        for line in haystack.lines() {
            let mut count = 0;
            let mut input = Input::new(line);
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
            capture_count += count;
            if count > 0 {
                line_count += 1;
            }
        }
        Ok((line_count, capture_count))
    })
}

fn regex_automata_onepass(b: &Benchmark) -> anyhow::Result<Results> {
    let haystack = &*b.haystack;
    let re = new::regex_automata_onepass(b)?;
    let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    b.run(verify, || {
        let (mut line_count, mut capture_count) = (0, 0);
        for line in haystack.lines() {
            let mut count = 0;
            let mut input = Input::new(line).anchored(Anchored::Yes);
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
            capture_count += count;
            if count > 0 {
                line_count += 1;
            }
        }
        Ok((line_count, capture_count))
    })
}

#[cfg(feature = "old-regex-crate")]
fn regexold_api(b: &Benchmark) -> anyhow::Result<Results> {
    let haystack = &*b.haystack;
    let re = new::regexold_api(b)?;
    let mut caps = re.capture_locations();
    b.run(verify, || {
        let (mut line_count, mut capture_count) = (0, 0);
        for line in haystack.lines() {
            let mut at = 0;
            let mut count = 0;
            while let Some(m) = re.captures_read_at(&mut caps, line, at) {
                for i in 0..caps.len() {
                    if caps.get(i).is_some() {
                        count += 1;
                    }
                }
                // Benchmark definition says we may assume empty matches are
                // impossible.
                at = m.end();
            }
            capture_count += count;
            if count > 0 {
                line_count += 1;
            }
        }
        Ok((line_count, capture_count))
    })
}

#[cfg(feature = "extre-re2")]
fn re2_api(b: &Benchmark) -> anyhow::Result<Results> {
    let haystack = &*b.haystack;
    let re = new::re2_api(b)?;
    let mut caps = re.create_captures();
    b.run(verify, || {
        let (mut line_count, mut capture_count) = (0, 0);
        for line in haystack.lines() {
            let mut count = 0;
            let mut input = Input::new(line);
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
            capture_count += count;
            if count > 0 {
                line_count += 1;
            }
        }
        Ok((line_count, capture_count))
    })
}

#[cfg(feature = "extre-pcre2")]
fn pcre2_api_jit(b: &Benchmark) -> anyhow::Result<Results> {
    let haystack = &*b.haystack;
    let re = new::pcre2_api_jit(b)?;
    let mut md = re.create_match_data();
    b.run(verify, || {
        let (mut line_count, mut capture_count) = (0, 0);
        for line in haystack.lines() {
            let mut count = 0;
            let mut input = Input::new(line);
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
            capture_count += count;
            if count > 0 {
                line_count += 1;
            }
        }
        Ok((line_count, capture_count))
    })
}

#[cfg(feature = "extre-pcre2")]
fn pcre2_api_nojit(b: &Benchmark) -> anyhow::Result<Results> {
    let haystack = &*b.haystack;
    let re = new::pcre2_api_nojit(b)?;
    let mut md = re.create_match_data();
    b.run(verify, || {
        let (mut line_count, mut capture_count) = (0, 0);
        for line in haystack.lines() {
            let mut count = 0;
            let mut input = Input::new(line);
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
            capture_count += count;
            if count > 0 {
                line_count += 1;
            }
        }
        Ok((line_count, capture_count))
    })
}
