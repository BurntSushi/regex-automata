use std::convert::TryFrom;

use super::{new, Benchmark, Results};

pub(super) fn run(b: &Benchmark) -> anyhow::Result<Results> {
    match &*b.engine {
        "regex/api" => regex_api(b),
        "regex/ast" => regex_ast(b),
        "regex/hir" => regex_hir(b),
        "regex/nfa" => regex_nfa(b),
        "regex/automata/dense" => regex_automata_dfa_dense(b),
        "regex/automata/sparse" => regex_automata_dfa_sparse(b),
        "regex/automata/hybrid" => regex_automata_hybrid(b),
        "regex/automata/pikevm" => regex_automata_pikevm(b),
        "regex/automata/onepass" => regex_automata_onepass(b),
        "aho-corasick/dfa" => aho_corasick_dfa(b),
        "aho-corasick/nfa" => aho_corasick_nfa(b),
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
    b.run(verify, || {
        let re = new::regex_api(b)?;
        Ok(Box::new(move |h| Ok(re.find_iter(h).count())))
    })
}

fn regex_ast(b: &Benchmark) -> anyhow::Result<Results> {
    use syntax::ast::{parse::ParserBuilder, Ast};

    // We don't bother "verifying" the AST since it is already implicitly
    // verified via regex/api.
    #[inline(never)]
    fn verify(_: &Benchmark, _: Ast) -> anyhow::Result<()> {
        Ok(())
    }
    b.run(verify, || {
        let mut parser = ParserBuilder::new().build();
        let ast = parser.parse(&b.regex)?;
        Ok(ast)
    })
}

fn regex_hir(b: &Benchmark) -> anyhow::Result<Results> {
    use syntax::{
        ast::parse::ParserBuilder,
        hir::{translate::TranslatorBuilder, Hir},
    };

    // We don't bother "verifying" the HIR since it is already implicitly
    // verified via regex/api.
    #[inline(never)]
    fn verify(_: &Benchmark, _: Hir) -> anyhow::Result<()> {
        Ok(())
    }

    let ast = ParserBuilder::new().build().parse(&b.regex)?;
    let mut translator = TranslatorBuilder::new()
        .allow_invalid_utf8(true)
        .unicode(b.def.unicode)
        .case_insensitive(b.def.case_insensitive)
        .build();
    b.run(verify, || {
        let hir = translator.translate(&b.regex, &ast)?;
        Ok(hir)
    })
}

fn regex_nfa(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::nfa::thompson::{Compiler, NFA};
    use syntax::ParserBuilder;

    // We don't bother "verifying" the NFA since it is already implicitly
    // verified via regex/api.
    #[inline(never)]
    fn verify(_: &Benchmark, _: NFA) -> anyhow::Result<()> {
        Ok(())
    }

    let hir = ParserBuilder::new()
        .allow_invalid_utf8(true)
        .unicode(b.def.unicode)
        .case_insensitive(b.def.case_insensitive)
        .build()
        .parse(&b.regex)?;
    b.run(verify, || {
        let nfa = Compiler::new().build_from_hir(&hir)?;
        Ok(nfa)
    })
}

fn regex_automata_dfa_dense(b: &Benchmark) -> anyhow::Result<Results> {
    b.run(verify, || {
        let re = new::regex_automata_dfa_dense(b)?;
        Ok(Box::new(move |h| Ok(re.find_iter(h).count())))
    })
}

fn regex_automata_dfa_sparse(b: &Benchmark) -> anyhow::Result<Results> {
    b.run(verify, || {
        let re = new::regex_automata_dfa_sparse(b)?;
        Ok(Box::new(move |h| Ok(re.find_iter(h).count())))
    })
}

fn regex_automata_hybrid(b: &Benchmark) -> anyhow::Result<Results> {
    b.run(verify, || {
        let re = new::regex_automata_hybrid(b)?;
        let mut cache = re.create_cache();
        Ok(Box::new(move |h| Ok(re.find_iter(&mut cache, h).count())))
    })
}

fn regex_automata_pikevm(b: &Benchmark) -> anyhow::Result<Results> {
    b.run(verify, || {
        let re = new::regex_automata_pikevm(b)?;
        let mut cache = re.create_cache();
        Ok(Box::new(move |h| Ok(re.find_iter(&mut cache, h).count())))
    })
}

fn regex_automata_onepass(b: &Benchmark) -> anyhow::Result<Results> {
    b.run(verify, || {
        let re = new::regex_automata_onepass(b)?;
        let mut cache = re.create_cache();
        Ok(Box::new(move |h| {
            use automata::util::iter::Searcher;

            // The one-pass DFA only does anchored searches, so it doesn't
            // provide an iterator API. Technically though, we can still report
            // multiple matches if the regex matches are directly adjacent. So
            // we just build our own iterator.
            let mut caps = re.create_captures();
            let it = Searcher::new(re.create_input(h))
                .into_matches_iter(|input| {
                    re.try_search(&mut cache, input, &mut caps)?;
                    Ok(caps.get_match())
                })
                .infallible();
            Ok(it.count())
        }))
    })
}

fn aho_corasick_dfa(b: &Benchmark) -> anyhow::Result<Results> {
    b.run(verify, || {
        let re = new::aho_corasick_dfa(b)?;
        Ok(Box::new(move |h| Ok(re.find_iter(h).count())))
    })
}

fn aho_corasick_nfa(b: &Benchmark) -> anyhow::Result<Results> {
    b.run(verify, || {
        let re = new::aho_corasick_nfa(b)?;
        Ok(Box::new(move |h| Ok(re.find_iter(h).count())))
    })
}

#[cfg(feature = "extre-re2")]
fn re2_api(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::Input;

    b.run(verify, || {
        let re = new::re2_api(b)?;
        Ok(Box::new(move |h| Ok(re.find_iter(Input::new(h)).count())))
    })
}

#[cfg(feature = "extre-pcre2")]
fn pcre2_api_jit(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::Input;

    b.run(verify, || {
        let re = new::pcre2_api_jit(b)?;
        let mut md = re.create_match_data_for_matches_only();
        Ok(Box::new(move |h| {
            Ok(re.try_find_iter(Input::new(h), &mut md).count())
        }))
    })
}

#[cfg(feature = "extre-pcre2")]
fn pcre2_api_nojit(b: &Benchmark) -> anyhow::Result<Results> {
    use automata::Input;

    b.run(verify, || {
        let re = new::pcre2_api_nojit(b)?;
        let mut md = re.create_match_data_for_matches_only();
        Ok(Box::new(move |h| {
            Ok(re.try_find_iter(Input::new(h), &mut md).count())
        }))
    })
}
