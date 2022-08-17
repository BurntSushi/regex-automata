use std::io::{stdout, Write};

use crate::{
    app::{self, App, Args},
    config,
    util::{self, Table},
};

use automata::{
    dfa::{self, Automaton},
    hybrid,
    nfa::thompson::{
        backtrack::{self, BoundedBacktracker},
        pikevm::{self, PikeVM},
    },
    util::{captures::Captures, iter},
    Input,
};

const ABOUT: &'static str = "\
Counts all occurrences of a regex in a file.

This is principally useful for ad hoc benchmarking. It never prints any of the
matches, and instead just counts the number of occurrences. Files are memory
mapped to reduce I/O latency. When benchmarking, files should be big enough
such that searches take longer than a few tens of milliseconds.
";

pub fn define() -> App {
    app::command("matches")
        .about("Count the number of matches of a regex in a file.")
        .before_help(ABOUT)
        .subcommand(define_api())
        .subcommand(define_dfa())
        .subcommand(define_hybrid())
        .subcommand(define_nfa())
}

pub fn run(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(args, define, |cmd, args| match cmd {
        "api" => run_api(args),
        "dfa" => run_dfa(args),
        "hybrid" => run_hybrid(args),
        "nfa" => run_nfa(args),
        _ => Err(util::UnrecognizedCommandError.into()),
    })
}

fn define_api() -> App {
    let mut regex = app::leaf("regex").about("Search using a 'Regex'.");
    regex = config::Haystack::define(regex);
    regex = config::Patterns::define(regex);
    regex = config::Syntax::define(regex);
    regex = config::RegexAPI::define(regex);
    regex = config::Find::define(regex);

    app::command("api")
        .about("Search using a top-level 'regex' crate API.")
        .subcommand(regex)
}

fn define_dfa() -> App {
    let mut dense = app::leaf("dense").about("Search using a dense DFA.");
    dense = config::Haystack::define(dense);
    dense = config::Input::define(dense);
    dense = config::Patterns::define(dense);
    dense = config::Syntax::define(dense);
    dense = config::Thompson::define(dense);
    dense = config::Dense::define(dense);
    dense = config::Overlapping::define(dense);
    dense = config::Find::define(dense);

    let mut sparse = app::leaf("sparse").about("Search using a sparse DFA.");
    sparse = config::Haystack::define(sparse);
    sparse = config::Input::define(sparse);
    sparse = config::Patterns::define(sparse);
    sparse = config::Syntax::define(sparse);
    sparse = config::Thompson::define(sparse);
    sparse = config::Dense::define(sparse);
    sparse = config::Overlapping::define(sparse);
    sparse = config::Find::define(sparse);

    let mut onepass =
        app::leaf("onepass").about("Search using a one-pass DFA.");
    onepass = config::Haystack::define(onepass);
    onepass = config::Input::define(onepass);
    onepass = config::Patterns::define(onepass);
    onepass = config::Syntax::define(onepass);
    onepass = config::Thompson::define(onepass);
    onepass = config::OnePass::define(onepass);
    onepass = config::Find::define(onepass);

    app::command("dfa")
        .about("Search using a DFA.")
        .before_help("Search using a DFA.")
        .subcommand(dense)
        .subcommand(sparse)
        .subcommand(onepass)
        .subcommand(define_dfa_regex())
}

fn define_dfa_regex() -> App {
    let mut dense = app::leaf("dense").about("Search using a dense DFA.");
    dense = config::Haystack::define(dense);
    dense = config::Input::define(dense);
    dense = config::Patterns::define(dense);
    dense = config::Syntax::define(dense);
    dense = config::Thompson::define(dense);
    dense = config::Dense::define(dense);
    dense = config::RegexDFA::define(dense);
    dense = config::Find::define(dense);

    let mut sparse = app::leaf("sparse").about("Search using a sparse DFA.");
    sparse = config::Haystack::define(sparse);
    sparse = config::Input::define(sparse);
    sparse = config::Patterns::define(sparse);
    sparse = config::Syntax::define(sparse);
    sparse = config::Thompson::define(sparse);
    sparse = config::Dense::define(sparse);
    sparse = config::RegexDFA::define(sparse);
    sparse = config::Find::define(sparse);

    app::command("regex")
        .about("Search using a regex DFA.")
        .subcommand(dense)
        .subcommand(sparse)
}

fn define_hybrid() -> App {
    let mut dfa = app::leaf("dfa").about("Search using a lazy DFA object.");
    dfa = config::Haystack::define(dfa);
    dfa = config::Input::define(dfa);
    dfa = config::Patterns::define(dfa);
    dfa = config::Syntax::define(dfa);
    dfa = config::Thompson::define(dfa);
    dfa = config::Hybrid::define(dfa);
    dfa = config::Overlapping::define(dfa);
    dfa = config::Find::define(dfa);

    let mut regex =
        app::leaf("regex").about("Search using a lazy regex object.");
    regex = config::Haystack::define(regex);
    regex = config::Input::define(regex);
    regex = config::Patterns::define(regex);
    regex = config::Syntax::define(regex);
    regex = config::Thompson::define(regex);
    regex = config::Hybrid::define(regex);
    regex = config::RegexHybrid::define(regex);
    regex = config::Find::define(regex);

    app::command("hybrid")
        .about("Search using a hybrid NFA/DFA object.")
        .subcommand(dfa)
        .subcommand(regex)
}

fn define_nfa() -> App {
    app::command("nfa")
        .about("Search using an NFA.")
        .subcommand(define_nfa_thompson())
}

fn define_nfa_thompson() -> App {
    let mut backtrack =
        app::leaf("backtrack").about("Search using a bounded backtracker.");
    backtrack = config::Haystack::define(backtrack);
    backtrack = config::Input::define(backtrack);
    backtrack = config::Patterns::define(backtrack);
    backtrack = config::Syntax::define(backtrack);
    backtrack = config::Thompson::define(backtrack);
    backtrack = config::Backtrack::define(backtrack);
    backtrack = config::Find::define(backtrack);

    let mut pikevm = app::leaf("pikevm").about("Search using a Pike VM.");
    pikevm = config::Haystack::define(pikevm);
    pikevm = config::Input::define(pikevm);
    pikevm = config::Patterns::define(pikevm);
    pikevm = config::Syntax::define(pikevm);
    pikevm = config::Thompson::define(pikevm);
    pikevm = config::PikeVM::define(pikevm);
    pikevm = config::Find::define(pikevm);

    app::command("thompson")
        .about("Search using a Thompson NFA.")
        .subcommand(backtrack)
        .subcommand(pikevm)
}

fn run_api(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(args, define_api, |cmd, args| match cmd {
        "regex" => run_api_regex(args),
        _ => Err(util::UnrecognizedCommandError.into()),
    })
}

fn run_api_regex(args: &Args) -> anyhow::Result<()> {
    let mut table = Table::empty();

    let csyntax = config::Syntax::get(args)?;
    let cregex = config::RegexAPI::get(args)?;
    let chaystack = config::Haystack::get(args)?;
    let cpatterns = config::Patterns::get(args)?;
    let cfind = config::Find::get(args)?;

    let re =
        cregex.from_patterns(&mut table, &csyntax, &cregex, &cpatterns)?;
    chaystack.with_bytes(|bytes| {
        let mut buf = String::new();
        let (count, time) =
            util::timeitr(|| search_api_regex(&re, &cfind, bytes, &mut buf))?;
        table.add("search time", time);
        table.add("count", count);
        table.print(stdout())?;
        if !buf.is_empty() {
            write!(stdout(), "\n{}", buf)?;
        }
        Ok(())
    })
}

fn run_dfa(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(args, define_dfa, |cmd, args| match cmd {
        "dense" => run_dfa_dense(args),
        "sparse" => run_dfa_sparse(args),
        "onepass" => run_dfa_onepass(args),
        "regex" => run_dfa_regex(args),
        _ => Err(util::UnrecognizedCommandError.into()),
    })
}

fn run_dfa_dense(args: &Args) -> anyhow::Result<()> {
    let mut table = Table::empty();

    let csyntax = config::Syntax::get(args)?;
    let cthompson = config::Thompson::get(args)?;
    let cdense = config::Dense::get(args)?;
    let chaystack = config::Haystack::get(args)?;
    let cinput = config::Input::get(args)?;
    let cpatterns = config::Patterns::get(args)?;
    let coverlapping = config::Overlapping::get(args)?;
    let cfind = config::Find::get(args)?;

    let dfa = cdense.from_patterns_dense(
        &mut table, &csyntax, &cthompson, &cdense, &cpatterns,
    )?;
    cinput.with_input(&chaystack, |input| {
        let mut buf = String::new();
        let (counts, time) = util::timeitr(|| {
            search_dfa_automaton(&dfa, &coverlapping, &cfind, input, &mut buf)
        })?;
        table.add("search time", time);
        table.add("counts", counts);
        table.print(stdout())?;
        if !buf.is_empty() {
            write!(stdout(), "\n{}", buf)?;
        }
        Ok(())
    })
}

fn run_dfa_sparse(args: &Args) -> anyhow::Result<()> {
    let mut table = Table::empty();

    let csyntax = config::Syntax::get(args)?;
    let cthompson = config::Thompson::get(args)?;
    let cdense = config::Dense::get(args)?;
    let chaystack = config::Haystack::get(args)?;
    let cinput = config::Input::get(args)?;
    let cpatterns = config::Patterns::get(args)?;
    let coverlapping = config::Overlapping::get(args)?;
    let cfind = config::Find::get(args)?;

    let dfa = cdense.from_patterns_sparse(
        &mut table, &csyntax, &cthompson, &cdense, &cpatterns,
    )?;
    cinput.with_input(&chaystack, |input| {
        let mut buf = String::new();
        let (counts, time) = util::timeitr(|| {
            search_dfa_automaton(&dfa, &coverlapping, &cfind, input, &mut buf)
        })?;
        table.add("search time", time);
        table.add("counts", counts);
        table.print(stdout())?;
        if !buf.is_empty() {
            write!(stdout(), "\n{}", buf)?;
        }
        Ok(())
    })
}

fn run_dfa_onepass(args: &Args) -> anyhow::Result<()> {
    let mut table = Table::empty();

    let csyntax = config::Syntax::get(args)?;
    let cthompson = config::Thompson::get(args)?;
    let conepass = config::OnePass::get(args)?;
    let chaystack = config::Haystack::get(args)?;
    let cinput = config::Input::get(args)?;
    let cpatterns = config::Patterns::get(args)?;
    let cfind = config::Find::get(args)?;

    let re = conepass
        .from_patterns(&mut table, &csyntax, &cthompson, &cpatterns)?;

    let (mut cache, time) = util::timeit(|| re.create_cache());
    table.add("create regex cache time", time);
    table.add("cache size", cache.memory_usage());

    cinput.with_input(&chaystack, |input| {
        let mut buf = String::new();
        let (counts, time) = util::timeitr(|| {
            search_dfa_onepass(&re, &mut cache, &cfind, input, &mut buf)
        })?;
        table.add("search time", time);
        table.add("counts", counts);
        table.print(stdout())?;
        if !buf.is_empty() {
            write!(stdout(), "\n{}", buf)?;
        }
        Ok(())
    })
}

fn run_dfa_regex(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(args, define_dfa_regex, |cmd, args| match cmd {
        "dense" => run_dfa_regex_dense(args),
        "sparse" => run_dfa_regex_sparse(args),
        _ => Err(util::UnrecognizedCommandError.into()),
    })
}

fn run_dfa_regex_dense(args: &Args) -> anyhow::Result<()> {
    let mut table = Table::empty();

    let csyntax = config::Syntax::get(args)?;
    let cthompson = config::Thompson::get(args)?;
    let cdense = config::Dense::get(args)?;
    let cregex = config::RegexDFA::get(args)?;
    let chaystack = config::Haystack::get(args)?;
    let cinput = config::Input::get(args)?;
    let cpatterns = config::Patterns::get(args)?;
    let cfind = config::Find::get(args)?;

    let re = cregex.from_patterns_dense(
        &mut table, &csyntax, &cthompson, &cdense, &cpatterns,
    )?;
    cinput.with_input(&chaystack, |input| {
        let mut buf = String::new();
        let (counts, time) =
            util::timeitr(|| search_dfa_regex(&re, &cfind, input, &mut buf))?;
        table.add("search time", time);
        table.add("counts", counts);
        table.print(stdout())?;
        if !buf.is_empty() {
            write!(stdout(), "\n{}", buf)?;
        }
        Ok(())
    })
}

fn run_dfa_regex_sparse(args: &Args) -> anyhow::Result<()> {
    let mut table = Table::empty();

    let csyntax = config::Syntax::get(args)?;
    let cthompson = config::Thompson::get(args)?;
    let cdense = config::Dense::get(args)?;
    let cregex = config::RegexDFA::get(args)?;
    let chaystack = config::Haystack::get(args)?;
    let cinput = config::Input::get(args)?;
    let cpatterns = config::Patterns::get(args)?;
    let cfind = config::Find::get(args)?;

    let re = cregex.from_patterns_sparse(
        &mut table, &csyntax, &cthompson, &cdense, &cpatterns,
    )?;
    cinput.with_input(&chaystack, |input| {
        let mut buf = String::new();
        let (counts, time) =
            util::timeitr(|| search_dfa_regex(&re, &cfind, input, &mut buf))?;
        table.add("search time", time);
        table.add("counts", counts);
        table.print(stdout())?;
        if !buf.is_empty() {
            write!(stdout(), "\n{}", buf)?;
        }
        Ok(())
    })
}

fn run_hybrid(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(args, define_hybrid, |cmd, args| match cmd {
        "dfa" => run_hybrid_dfa(args),
        "regex" => run_hybrid_regex(args),
        _ => Err(util::UnrecognizedCommandError.into()),
    })
}

fn run_hybrid_dfa(args: &Args) -> anyhow::Result<()> {
    let mut table = Table::empty();

    let csyntax = config::Syntax::get(args)?;
    let cthompson = config::Thompson::get(args)?;
    let cdfa = config::Hybrid::get(args)?;
    let chaystack = config::Haystack::get(args)?;
    let cinput = config::Input::get(args)?;
    let cpatterns = config::Patterns::get(args)?;
    let coverlapping = config::Overlapping::get(args)?;
    let cfind = config::Find::get(args)?;

    let dfa =
        cdfa.from_patterns(&mut table, &csyntax, &cthompson, &cpatterns)?;

    let (mut cache, time) = util::timeit(|| dfa.create_cache());
    table.add("create cache time", time);
    table.add("cache size", cache.memory_usage());

    cinput.with_input(&chaystack, |input| {
        let mut buf = String::new();
        let (counts, time) = util::timeitr(|| {
            search_hybrid_dfa(
                &dfa,
                &mut cache,
                &coverlapping,
                &cfind,
                input,
                &mut buf,
            )
        })?;
        table.add("search time", time);
        table.add("cache clear count", cache.clear_count());
        table.add("counts", counts);
        table.print(stdout())?;
        if !buf.is_empty() {
            write!(stdout(), "\n{}", buf)?;
        }
        Ok(())
    })
}

fn run_hybrid_regex(args: &Args) -> anyhow::Result<()> {
    let mut table = Table::empty();

    let csyntax = config::Syntax::get(args)?;
    let cthompson = config::Thompson::get(args)?;
    let cdfa = config::Hybrid::get(args)?;
    let cregex = config::RegexHybrid::get(args)?;
    let chaystack = config::Haystack::get(args)?;
    let cinput = config::Input::get(args)?;
    let cpatterns = config::Patterns::get(args)?;
    let cfind = config::Find::get(args)?;

    let re = cregex
        .from_patterns(&mut table, &csyntax, &cthompson, &cdfa, &cpatterns)?;

    let (mut cache, time) = util::timeit(|| re.create_cache());
    table.add("create regex cache time", time);
    table.add("cache size", cache.memory_usage());

    cinput.with_input(&chaystack, |input| {
        let mut buf = String::new();
        let (counts, time) = util::timeitr(|| {
            search_hybrid_regex(&re, &mut cache, &cfind, input, &mut buf)
        })?;
        table.add("search time", time);
        let (cache_fwd, cache_rev) = cache.as_parts();
        table.add("cache clear count (forward)", cache_fwd.clear_count());
        table.add("cache clear count (reverse)", cache_rev.clear_count());
        table.add("counts", counts);
        table.print(stdout())?;
        if !buf.is_empty() {
            write!(stdout(), "\n{}", buf)?;
        }
        Ok(())
    })
}

fn run_nfa(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(args, define_nfa, |cmd, args| match cmd {
        "thompson" => run_nfa_thompson(args),
        _ => Err(util::UnrecognizedCommandError.into()),
    })
}

fn run_nfa_thompson(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(args, define_nfa_thompson, |cmd, args| match cmd {
        "backtrack" => run_nfa_thompson_backtrack(args),
        "pikevm" => run_nfa_thompson_pikevm(args),
        _ => Err(util::UnrecognizedCommandError.into()),
    })
}

fn run_nfa_thompson_backtrack(args: &Args) -> anyhow::Result<()> {
    let mut table = Table::empty();

    let csyntax = config::Syntax::get(args)?;
    let cthompson = config::Thompson::get(args)?;
    let cbacktrack = config::Backtrack::get(args)?;
    let chaystack = config::Haystack::get(args)?;
    let cinput = config::Input::get(args)?;
    let cpatterns = config::Patterns::get(args)?;
    let cfind = config::Find::get(args)?;

    let re = cbacktrack
        .from_patterns(&mut table, &csyntax, &cthompson, &cpatterns)?;

    let (mut cache, time) = util::timeit(|| re.create_cache());
    table.add("create cache time", time);
    table.add("cache size", cache.memory_usage());
    table.add("max haystack length", re.max_haystack_len());

    cinput.with_input(&chaystack, |input| {
        let mut buf = String::new();
        let (counts, time) = util::timeitr(|| {
            search_backtrack(&re, &mut cache, &cfind, input, &mut buf)
        })?;
        table.add("search time", time);
        table.add("counts", counts);
        table.print(stdout())?;
        if !buf.is_empty() {
            write!(stdout(), "\n{}", buf)?;
        }
        Ok(())
    })
}

fn run_nfa_thompson_pikevm(args: &Args) -> anyhow::Result<()> {
    let mut table = Table::empty();

    let csyntax = config::Syntax::get(args)?;
    let cthompson = config::Thompson::get(args)?;
    let cpikevm = config::PikeVM::get(args)?;
    let chaystack = config::Haystack::get(args)?;
    let cinput = config::Input::get(args)?;
    let cpatterns = config::Patterns::get(args)?;
    let cfind = config::Find::get(args)?;

    let re =
        cpikevm.from_patterns(&mut table, &csyntax, &cthompson, &cpatterns)?;

    let (mut cache, time) = util::timeit(|| re.create_cache());
    table.add("create cache time", time);
    table.add("cache size", cache.memory_usage());

    cinput.with_input(&chaystack, |input| {
        let mut buf = String::new();
        let (counts, time) = util::timeitr(|| {
            search_pikevm(&re, &mut cache, &cfind, input, &mut buf)
        })?;
        table.add("search time", time);
        table.add("counts", counts);
        table.print(stdout())?;
        if !buf.is_empty() {
            write!(stdout(), "\n{}", buf)?;
        }
        Ok(())
    })
}

fn search_api_regex(
    re: &regex::bytes::Regex,
    find: &config::Find,
    haystack: &[u8],
    buf: &mut String,
) -> anyhow::Result<u64> {
    let mut count = 0;
    for m in re.find_iter(haystack) {
        count += 1;
        if find.matches() {
            write_api_match(m, buf);
        }
    }
    Ok(count)
}

fn search_dfa_automaton<A: Automaton>(
    dfa: A,
    overlapping: &config::Overlapping,
    find: &config::Find,
    input: &Input<'_, '_>,
    buf: &mut String,
) -> anyhow::Result<Vec<u64>> {
    let mut counts = vec![0u64; dfa.pattern_len()];
    if overlapping.enabled() {
        let mut state = dfa::OverlappingState::start();
        while let Some(m) = {
            dfa.try_search_overlapping_fwd(input, &mut state)?;
            state.get_match()
        } {
            counts[m.pattern()] += 1;
            if find.matches() {
                write_half_match(m, buf);
            }
        }
    } else {
        let it = iter::Searcher::new(input.clone())
            .into_half_matches_iter(|input| dfa.try_search_fwd(input));
        for result in it {
            let m = result?;
            counts[m.pattern()] += 1;
            if find.matches() {
                write_half_match(m, buf);
            }
        }
    }
    Ok(counts)
}

fn search_dfa_regex<A: Automaton>(
    re: &dfa::regex::Regex<A>,
    find: &config::Find,
    input: &Input<'_, '_>,
    buf: &mut String,
) -> anyhow::Result<Vec<u64>> {
    let mut counts = vec![0u64; re.pattern_len()];
    let it = iter::Searcher::new(input.clone())
        .into_matches_iter(|input| re.try_search(input));
    for result in it {
        let m = result?;
        counts[m.pattern()] += 1;
        if find.matches() {
            write_multi_match(m, buf);
        }
    }
    Ok(counts)
}

fn search_dfa_onepass(
    re: &dfa::onepass::DFA,
    cache: &mut dfa::onepass::Cache,
    find: &config::Find,
    input: &Input<'_, '_>,
    buf: &mut String,
) -> anyhow::Result<Vec<u64>> {
    let mut counts = vec![0u64; re.pattern_len()];
    let mut caps = Captures::matches(re.get_nfa().group_info().clone());
    let it = iter::Searcher::new(input.clone()).into_matches_iter(|input| {
        re.search(cache, input, &mut caps);
        Ok(caps.get_match())
    });
    for result in it {
        let m = result?;
        counts[m.pattern()] += 1;
        if find.matches() {
            write_multi_match(m, buf);
        }
    }
    Ok(counts)
}

fn search_hybrid_dfa<'i, 'c>(
    dfa: &hybrid::dfa::DFA,
    cache: &mut hybrid::dfa::Cache,
    overlapping: &config::Overlapping,
    find: &config::Find,
    input: &Input<'_, '_>,
    buf: &mut String,
) -> anyhow::Result<Vec<u64>> {
    let mut counts = vec![0u64; dfa.pattern_len()];
    if overlapping.enabled() {
        let mut state = hybrid::dfa::OverlappingState::start();
        while let Some(m) = {
            dfa.try_search_overlapping_fwd(cache, input, &mut state)?;
            state.get_match()
        } {
            counts[m.pattern()] += 1;
            if find.matches() {
                write_half_match(m, buf);
            }
        }
    } else {
        let it = iter::Searcher::new(input.clone())
            .into_half_matches_iter(|input| dfa.try_search_fwd(cache, input));
        for result in it {
            let m = result?;
            counts[m.pattern()] += 1;
            if find.matches() {
                write_half_match(m, buf);
            }
        }
    }
    Ok(counts)
}

fn search_hybrid_regex(
    re: &hybrid::regex::Regex,
    cache: &mut hybrid::regex::Cache,
    find: &config::Find,
    input: &Input<'_, '_>,
    buf: &mut String,
) -> anyhow::Result<Vec<u64>> {
    let mut counts = vec![0u64; re.pattern_len()];
    let it = iter::Searcher::new(input.clone())
        .into_matches_iter(|input| re.try_search(cache, input));
    for result in it {
        let m = result?;
        counts[m.pattern()] += 1;
        if find.matches() {
            write_multi_match(m, buf);
        }
    }
    Ok(counts)
}

fn search_backtrack(
    re: &BoundedBacktracker,
    cache: &mut backtrack::Cache,
    find: &config::Find,
    input: &Input<'_, '_>,
    buf: &mut String,
) -> anyhow::Result<Vec<u64>> {
    let mut counts = vec![0u64; re.pattern_len()];
    let mut caps = Captures::matches(re.get_nfa().group_info().clone());
    let mut it = iter::Searcher::new(input.clone());
    loop {
        it.try_advance(|input| {
            re.try_search(cache, input, &mut caps)?;
            Ok(caps.get_match())
        })?;
        let m = match caps.get_match() {
            None => break,
            Some(m) => m,
        };
        counts[m.pattern()] += 1;
        if find.matches() {
            write_multi_match(m, buf);
        }
    }
    Ok(counts)
}

fn search_pikevm(
    re: &PikeVM,
    cache: &mut pikevm::Cache,
    find: &config::Find,
    input: &Input<'_, '_>,
    buf: &mut String,
) -> anyhow::Result<Vec<u64>> {
    let mut counts = vec![0u64; re.pattern_len()];
    let mut caps = Captures::matches(re.get_nfa().group_info().clone());
    let mut it = iter::Searcher::new(input.clone());
    loop {
        it.advance(|input| {
            re.search(cache, input, &mut caps);
            Ok(caps.get_match())
        });
        let m = match caps.get_match() {
            None => break,
            Some(m) => m,
        };
        counts[m.pattern()] += 1;
        if find.matches() {
            write_multi_match(m, buf);
        }
    }
    Ok(counts)
}

fn write_api_match(m: regex::bytes::Match, buf: &mut String) {
    use std::fmt::Write;

    writeln!(buf, "[{:?}, {:?})", m.start(), m.end()).unwrap();
}

fn write_multi_match(m: automata::Match, buf: &mut String) {
    use std::fmt::Write;

    writeln!(buf, "{:?}: [{:?}, {:?})", m.pattern(), m.start(), m.end())
        .unwrap();
}

fn write_half_match(m: automata::HalfMatch, buf: &mut String) {
    use std::fmt::Write;

    writeln!(buf, "{:?}: {:?}", m.pattern(), m.offset()).unwrap();
}
