use std::convert::TryFrom;

use automata::SyntaxConfig;

use super::{Benchmark, Results};

pub(super) fn run(b: &Benchmark) -> anyhow::Result<Results> {
    match &*b.engine {
        "regex/api" => regex_api(b),
        "regex/automata/backtrack" => regex_automata_backtrack(b),
        "regex/automata/pikevm" => regex_automata_pikevm(b),
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
    use regex::bytes::RegexBuilder;

    let haystack = &*b.haystack;
    let re = RegexBuilder::new(&b.def.regex)
        .unicode(b.def.unicode)
        .case_insensitive(b.def.case_insensitive)
        .build()?;
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

fn regex_automata_backtrack(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::{nfa::thompson::backtrack::BoundedBacktracker, Input};

    let haystack = &*b.haystack;
    let mut input = Input::new(haystack);
    let re = BoundedBacktracker::builder()
        .configure(BoundedBacktracker::config().utf8(false))
        .syntax(syntax_config(b))
        .build(&b.def.regex)?;
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
    use automata::{nfa::thompson::pikevm::PikeVM, Input};

    let haystack = &*b.haystack;
    let mut input = Input::new(haystack);
    let re = PikeVM::builder()
        .configure(PikeVM::config().utf8(false))
        .syntax(syntax_config(b))
        .build(&b.def.regex)?;
    let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    b.run(verify, || {
        input.set_start(0);
        let mut count = 0;
        while let Some(m) = {
            re.search(&mut cache, &input, &mut caps);
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

/// For regex-automata based regex engines, this builds a syntax configuration
/// from a benchmark definition.
fn syntax_config(b: &Benchmark) -> SyntaxConfig {
    SyntaxConfig::new()
        .utf8(false)
        .unicode(b.def.unicode)
        .case_insensitive(b.def.case_insensitive)
}
