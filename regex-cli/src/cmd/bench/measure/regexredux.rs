use std::fmt::Write;

use super::{new, Benchmark, Results};

pub(super) fn run(b: &Benchmark) -> anyhow::Result<Results> {
    match &*b.engine {
        "regex/api" => regex_api(b),
        "regex/automata/dfa/dense" => regex_automata_dfa_dense(b),
        "regex/automata/hybrid" => regex_automata_hybrid(b),
        "regex/automata/pikevm" => regex_automata_pikevm(b),
        #[cfg(feature = "extre-re2")]
        "re2/api" => re2_api(b),
        #[cfg(feature = "extre-pcre2")]
        "pcre2/api/jit" => pcre2_api_jit(b),
        #[cfg(feature = "extre-pcre2")]
        "pcre2/api/nojit" => pcre2_api_nojit(b),
        name => anyhow::bail!("unknown regex engine '{}'", name),
    }
}

fn verify(_b: &Benchmark, output: String) -> anyhow::Result<()> {
    let expected = "\
agggtaaa|tttaccct 6
[cgt]gggtaaa|tttaccc[acg] 26
a[act]ggtaaa|tttacc[agt]t 86
ag[act]gtaaa|tttac[agt]ct 58
agg[act]taaa|ttta[agt]cct 113
aggg[acg]aaa|ttt[cgt]ccct 31
agggt[cgt]aa|tt[acg]accct 31
agggta[cgt]a|t[acg]taccct 32
agggtaa[cgt]|[acg]ttaccct 43

1016745
1000000
547899
";
    anyhow::ensure!(
        expected == &*output,
        "output did not match what was expected",
    );
    Ok(())
}

fn regex_api(b: &Benchmark) -> anyhow::Result<Results> {
    use regex::bytes::RegexBuilder;

    let compile = |pattern: &str| -> anyhow::Result<RegexFn> {
        let re = RegexBuilder::new(pattern)
            .unicode(b.def.unicode)
            .case_insensitive(b.def.case_insensitive)
            .build()?;
        let find =
            move |h: &[u8]| Ok(re.find(h).map(|m| (m.start(), m.end())));
        Ok(Box::new(find))
    };
    b.run(verify, || generic_regex_redux(&b.haystack, compile))
}

fn regex_automata_dfa_dense(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::dfa::regex::Regex;

    let compile = |pattern: &str| -> anyhow::Result<RegexFn> {
        let re = Regex::builder()
            .configure(Regex::config().utf8(false))
            .syntax(new::automata_syntax_config(b))
            .build(pattern)?;
        let find = move |h: &[u8]| -> anyhow::Result<Option<(usize, usize)>> {
            Ok(re.find(h).map(|m| (m.start(), m.end())))
        };
        Ok(Box::new(find))
    };
    b.run(verify, || generic_regex_redux(&b.haystack, compile))
}

fn regex_automata_hybrid(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::hybrid::{dfa::DFA, regex::Regex};

    let compile = |pattern: &str| -> anyhow::Result<RegexFn> {
        let re = Regex::builder()
            .configure(Regex::config().utf8(false))
            .dfa(DFA::config().skip_cache_capacity_check(true))
            .syntax(new::automata_syntax_config(b))
            .build(pattern)?;
        let mut cache = re.create_cache();
        let find = move |h: &[u8]| -> anyhow::Result<Option<(usize, usize)>> {
            Ok(re.find(&mut cache, h).map(|m| (m.start(), m.end())))
        };
        Ok(Box::new(find))
    };
    b.run(verify, || generic_regex_redux(&b.haystack, compile))
}

fn regex_automata_pikevm(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::{nfa::thompson::pikevm::PikeVM, util::captures::Captures};

    let compile = |pattern: &str| -> anyhow::Result<RegexFn> {
        let re = PikeVM::builder()
            .configure(PikeVM::config().utf8(false))
            .syntax(new::automata_syntax_config(b))
            .build(pattern)?;
        let mut cache = re.create_cache();
        let mut caps =
            Captures::new_for_matches_only(re.get_nfa().group_info().clone());
        let find = move |h: &[u8]| -> anyhow::Result<Option<(usize, usize)>> {
            re.find(&mut cache, h, &mut caps);
            Ok(caps.get_match().map(|m| (m.start(), m.end())))
        };
        Ok(Box::new(find))
    };
    b.run(verify, || generic_regex_redux(&b.haystack, compile))
}

#[cfg(feature = "extre-re2")]
fn re2_api(b: &Benchmark) -> anyhow::Result<Results> {
    use crate::ffi::re2::Regex;
    use automata::Input;

    let compile = |pattern: &str| -> anyhow::Result<RegexFn> {
        let re = Regex::new(pattern, new::re2_options(b))?;
        let find = move |h: &[u8]| {
            Ok(re.find(&Input::new(h)).map(|m| (m.start(), m.end())))
        };
        Ok(Box::new(find))
    };
    b.run(verify, || generic_regex_redux(&b.haystack, compile))
}

#[cfg(feature = "extre-pcre2")]
fn pcre2_api_jit(b: &Benchmark) -> anyhow::Result<Results> {
    use crate::ffi::pcre2::Regex;
    use automata::Input;

    let compile = |pattern: &str| -> anyhow::Result<RegexFn> {
        let re = Regex::new(pattern, new::pcre2_options(b))?;
        let mut md = re.create_match_data_for_matches_only();
        let find = move |h: &[u8]| {
            re.try_find(&Input::new(h), &mut md)?;
            Ok(md.get_match().map(|m| (m.start(), m.end())))
        };
        Ok(Box::new(find))
    };
    b.run(verify, || generic_regex_redux(&b.haystack, compile))
}

#[cfg(feature = "extre-pcre2")]
fn pcre2_api_nojit(b: &Benchmark) -> anyhow::Result<Results> {
    use crate::ffi::pcre2::Regex;
    use automata::Input;

    let compile = |pattern: &str| -> anyhow::Result<RegexFn> {
        let mut opts = new::pcre2_options(b);
        opts.jit = false;
        let re = Regex::new(pattern, opts)?;
        let mut md = re.create_match_data_for_matches_only();
        let find = move |h: &[u8]| {
            re.try_find(&Input::new(h), &mut md)?;
            Ok(md.get_match().map(|m| (m.start(), m.end())))
        };
        Ok(Box::new(find))
    };
    b.run(verify, || generic_regex_redux(&b.haystack, compile))
}

type RegexFn = Box<dyn FnMut(&[u8]) -> anyhow::Result<Option<(usize, usize)>>>;

fn generic_regex_redux(
    haystack: &[u8],
    mut compile: impl FnMut(&str) -> anyhow::Result<RegexFn>,
) -> anyhow::Result<String> {
    let mut out = String::new();
    let mut seq = haystack.to_vec();
    let ilen = seq.len();

    let flatten = compile(r">[^\n]*\n|\n")?;
    seq = replace_all(&seq, "", flatten)?;
    let clen = seq.len();

    let variants = vec![
        r"agggtaaa|tttaccct",
        r"[cgt]gggtaaa|tttaccc[acg]",
        r"a[act]ggtaaa|tttacc[agt]t",
        r"ag[act]gtaaa|tttac[agt]ct",
        r"agg[act]taaa|ttta[agt]cct",
        r"aggg[acg]aaa|ttt[cgt]ccct",
        r"agggt[cgt]aa|tt[acg]accct",
        r"agggta[cgt]a|t[acg]taccct",
        r"agggtaa[cgt]|[acg]ttaccct",
    ];
    for variant in variants {
        let re = compile(variant)?;
        writeln!(out, "{} {}", variant, count(&seq, re)?)?;
    }

    let substs = vec![
        (compile(r"tHa[Nt]")?, "<4>"),
        (compile(r"aND|caN|Ha[DS]|WaS")?, "<3>"),
        (compile(r"a[NSt]|BY")?, "<2>"),
        (compile(r"<[^>]*>")?, "|"),
        (compile(r"\|[^|][^|]*\|")?, "-"),
    ];
    for (re, replacement) in substs.into_iter() {
        seq = replace_all(&seq, replacement, re)?;
    }
    writeln!(out, "\n{}\n{}\n{}", ilen, clen, seq.len())?;
    Ok(out)
}

fn count(
    mut haystack: &[u8],
    mut find: impl FnMut(&[u8]) -> anyhow::Result<Option<(usize, usize)>>,
) -> anyhow::Result<usize> {
    let mut count = 0;
    // This type of iteration only works in cases where there isn't any
    // look-around and there aren't any empty matches. Which is the case
    // for this benchmark.
    while let Some((_, end)) = find(haystack)? {
        haystack = &haystack[end..];
        count += 1;
    }
    Ok(count)
}

fn replace_all(
    mut haystack: &[u8],
    replacement: &str,
    mut find: impl FnMut(&[u8]) -> anyhow::Result<Option<(usize, usize)>>,
) -> anyhow::Result<Vec<u8>> {
    let mut new = Vec::with_capacity(haystack.len());
    // This type of iteration only works in cases where there isn't any
    // look-around and there aren't any empty matches. Which is the case
    // for this benchmark.
    while let Some((start, end)) = find(haystack)? {
        new.extend_from_slice(&haystack[..start]);
        new.extend_from_slice(replacement.as_bytes());
        haystack = &haystack[end..];
    }
    new.extend_from_slice(haystack);
    Ok(new)
}
