use std::cmp;
use std::io::{stdout, Write};

use crate::{
    app::{self, App, Args},
    config,
    util::{self, Table},
};

use anyhow::Context;
use automata::{
    dfa::{self, Automaton},
    hybrid,
};

const ABOUT: &'static str = "\
Finds all occurrences of a regex in a file.

This is principally useful for ad hoc benchmarking. It never prints any of the
matches, and instead just counts the number of occurrences. Files are memory
mapped to reduce I/O latency. When benchmarking, files should be big enough
such that searches take longer than a few tens of milliseconds.
";

pub fn define() -> App {
    app::command("find")
        .about("Find the number of occurrences of a regex in a file.")
        .before_help(ABOUT)
        .subcommand(define_dfa())
        .subcommand(define_hybrid())
}

pub fn run(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(args, define, |cmd, args| match cmd {
        "dfa" => run_dfa(args),
        "hybrid" => run_hybrid(args),
        _ => Err(util::UnrecognizedCommandError.into()),
    })
}

fn define_dfa() -> App {
    let mut dense = app::leaf("dense").about("Search using a dense DFA.");
    dense = config::Input::define(dense);
    dense = config::Patterns::define(dense);
    dense = config::Syntax::define(dense);
    dense = config::Thompson::define(dense);
    dense = config::Dense::define(dense);
    dense = config::Find::define(dense);

    let mut sparse = app::leaf("sparse").about("Search using a sparse DFA.");
    sparse = config::Input::define(sparse);
    sparse = config::Patterns::define(sparse);
    sparse = config::Syntax::define(sparse);
    sparse = config::Thompson::define(sparse);
    sparse = config::Dense::define(sparse);
    sparse = config::Find::define(sparse);

    app::command("dfa")
        .about("Search using a DFA.")
        .subcommand(dense)
        .subcommand(sparse)
        .subcommand(define_dfa_regex())
}

fn define_dfa_regex() -> App {
    let mut dense = app::leaf("dense").about("Search using a dense DFA.");
    dense = config::Input::define(dense);
    dense = config::Patterns::define(dense);
    dense = config::Syntax::define(dense);
    dense = config::Thompson::define(dense);
    dense = config::Dense::define(dense);
    dense = config::RegexDFA::define(dense);
    dense = config::Find::define(dense);

    let mut sparse = app::leaf("sparse").about("Search using a sparse DFA.");
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
    dfa = config::Input::define(dfa);
    dfa = config::Patterns::define(dfa);
    dfa = config::Syntax::define(dfa);
    dfa = config::Thompson::define(dfa);
    dfa = config::Hybrid::define(dfa);
    dfa = config::Find::define(dfa);

    let mut regex =
        app::leaf("regex").about("Search using a lazy regex object.");
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

fn run_dfa(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(args, define, |cmd, args| match cmd {
        "dense" => run_dfa_dense(args),
        "sparse" => run_dfa_sparse(args),
        "regex" => run_dfa_regex(args),
        _ => Err(util::UnrecognizedCommandError.into()),
    })
}

fn run_dfa_dense(args: &Args) -> anyhow::Result<()> {
    let mut table = Table::empty();

    let csyntax = config::Syntax::get(args)?;
    let cthompson = config::Thompson::get(args)?;
    let cdense = config::Dense::get(args)?;
    let input = config::Input::get(args)?;
    let patterns = config::Patterns::get(args)?;
    let find = config::Find::get(args)?;

    let dfa = cdense.from_patterns_dense(
        &mut table, &csyntax, &cthompson, &cdense, &patterns,
    )?;
    input.with_mmap(|haystack| {
        let mut buf = String::new();
        let (counts, time) = util::timeitr(|| {
            search_dfa_automaton(&dfa, &find, &*haystack, &mut buf)
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
    let input = config::Input::get(args)?;
    let patterns = config::Patterns::get(args)?;
    let find = config::Find::get(args)?;

    let dfa = cdense.from_patterns_sparse(
        &mut table, &csyntax, &cthompson, &cdense, &patterns,
    )?;

    input.with_mmap(|haystack| {
        let mut buf = String::new();
        let (counts, time) = util::timeitr(|| {
            search_dfa_automaton(&dfa, &find, &*haystack, &mut buf)
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
    util::run_subcommand(args, define, |cmd, args| match cmd {
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
    let input = config::Input::get(args)?;
    let patterns = config::Patterns::get(args)?;
    let find = config::Find::get(args)?;

    let re = cregex.from_patterns_dense(
        &mut table, &csyntax, &cthompson, &cdense, &patterns,
    )?;

    input.with_mmap(|haystack| {
        let mut buf = String::new();
        let (counts, time) = util::timeitr(|| {
            search_dfa_regex(&re, &find, &*haystack, &mut buf)
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

fn run_dfa_regex_sparse(args: &Args) -> anyhow::Result<()> {
    let mut table = Table::empty();

    let csyntax = config::Syntax::get(args)?;
    let cthompson = config::Thompson::get(args)?;
    let cdense = config::Dense::get(args)?;
    let cregex = config::RegexDFA::get(args)?;
    let input = config::Input::get(args)?;
    let patterns = config::Patterns::get(args)?;
    let find = config::Find::get(args)?;

    let re = cregex.from_patterns_sparse(
        &mut table, &csyntax, &cthompson, &cdense, &patterns,
    )?;

    input.with_mmap(|haystack| {
        let mut buf = String::new();
        let (counts, time) = util::timeitr(|| {
            search_dfa_regex(&re, &find, &*haystack, &mut buf)
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

fn run_hybrid(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(args, define, |cmd, args| match cmd {
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
    let input = config::Input::get(args)?;
    let patterns = config::Patterns::get(args)?;
    let find = config::Find::get(args)?;

    let idfa =
        cdfa.from_patterns(&mut table, &csyntax, &cthompson, &patterns)?;

    let (mut cache, time) = util::timeit(|| idfa.create_cache());
    table.add("create cache time", time);
    let (mut dfa, time) = util::timeit(|| idfa.dfa(&mut cache));
    table.add("create dfa time", time);

    input.with_mmap(|haystack| {
        let mut buf = String::new();
        let (counts, time) = util::timeitr(|| {
            search_hybrid_dfa(&mut dfa, &find, &*haystack, &mut buf)
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

fn run_hybrid_regex(args: &Args) -> anyhow::Result<()> {
    let mut table = Table::empty();

    let csyntax = config::Syntax::get(args)?;
    let cthompson = config::Thompson::get(args)?;
    let cdfa = config::Hybrid::get(args)?;
    let cregex = config::RegexHybrid::get(args)?;
    let input = config::Input::get(args)?;
    let patterns = config::Patterns::get(args)?;
    let find = config::Find::get(args)?;

    let re = cregex
        .from_patterns(&mut table, &csyntax, &cthompson, &cdfa, &patterns)?;

    let (mut cache, time) = util::timeit(|| re.create_cache());
    table.add("create regex cache time", time);

    input.with_mmap(|haystack| {
        let mut buf = String::new();
        let (counts, time) = util::timeitr(|| {
            search_hybrid_regex(&re, &mut cache, &find, &*haystack, &mut buf)
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

fn search_dfa_automaton<A: Automaton>(
    dfa: A,
    find: &config::Find,
    haystack: &[u8],
    buf: &mut String,
) -> anyhow::Result<Vec<u64>> {
    let mut counts = vec![0u64; dfa.pattern_count()];
    let mut at = 0;
    match find.kind() {
        config::FindKind::Earliest => {
            while at < haystack.len() {
                let result =
                    dfa.find_earliest_fwd(&haystack[at..]).with_context(
                        || format!("failed to find match at {}", at),
                    )?;
                let end = match result {
                    None => break,
                    Some(end) => end,
                };
                // Always advance one byte, in the case of an zero-width match.
                at = cmp::max(at + 1, at + end.offset());
                counts[end.pattern()] += 1;
                if find.matches() {
                    write_half_match(end, buf);
                }
            }
        }
        config::FindKind::Leftmost => {
            while at < haystack.len() {
                let result =
                    dfa.find_leftmost_fwd(&haystack[at..]).with_context(
                        || format!("failed to find match at {}", at),
                    )?;
                let end = match result {
                    None => break,
                    Some(end) => end,
                };
                // Always advance one byte, in the case of an zero-width match.
                at = cmp::max(at + 1, at + end.offset());
                counts[end.pattern()] += 1;
                if find.matches() {
                    write_half_match(end, buf);
                }
            }
        }
        config::FindKind::Overlapping => {
            let mut state = dfa::OverlappingState::start();
            while at < haystack.len() {
                let result = dfa
                    .find_overlapping_fwd_at(
                        None,
                        None,
                        haystack,
                        at,
                        haystack.len(),
                        &mut state,
                    )
                    .with_context(|| {
                        format!("failed to find match at {}", at)
                    })?;
                let end = match result {
                    None => {
                        break;
                    }
                    Some(end) => end,
                };
                // Unlike the non-overlapping case, we're OK with empty matches
                // at this level. In particular, the overlapping search
                // algorithm is itself responsible for ensuring that progress
                // is always made. (The starting position of the search is
                // incremented by 1 whenever a non-None state ID is given.)
                at = end.offset();
                counts[end.pattern()] += 1;
                if find.matches() {
                    write_half_match(end, buf);
                }
            }
        }
    }
    Ok(counts)
}

fn search_dfa_regex<A: Automaton>(
    re: &dfa::regex::Regex<A>,
    find: &config::Find,
    haystack: &[u8],
    buf: &mut String,
) -> anyhow::Result<Vec<u64>> {
    let mut count = 0;
    let mut counts = vec![0u64; re.pattern_count()];
    match find.kind() {
        config::FindKind::Earliest => {
            for result in re.try_find_earliest_iter(haystack) {
                let m = result.with_context(|| {
                    format!("search failure after {} matches", count)
                })?;
                count += 1;
                counts[m.pattern()] += 1;
                if find.matches() {
                    write_multi_match(m, buf);
                }
            }
        }
        config::FindKind::Leftmost => {
            for result in re.try_find_leftmost_iter(haystack) {
                let m = result.with_context(|| {
                    format!("search failure after {} matches", count)
                })?;
                count += 1;
                counts[m.pattern()] += 1;
                if find.matches() {
                    write_multi_match(m, buf);
                }
            }
        }
        config::FindKind::Overlapping => {
            for result in re.try_find_overlapping_iter(haystack) {
                let m = result.with_context(|| {
                    format!("search failure after {} matches", count)
                })?;
                count += 1;
                counts[m.pattern()] += 1;
                if find.matches() {
                    write_multi_match(m, buf);
                }
            }
        }
    }
    Ok(counts)
}

fn search_hybrid_dfa<'i, 'c>(
    dfa: &mut hybrid::DFA<'i, 'c>,
    find: &config::Find,
    haystack: &[u8],
    buf: &mut String,
) -> anyhow::Result<Vec<u64>> {
    let mut counts = vec![0u64; dfa.pattern_count()];
    let mut at = 0;
    match find.kind() {
        config::FindKind::Earliest => {
            while at < haystack.len() {
                let result =
                    dfa.find_earliest_fwd(&haystack[at..]).with_context(
                        || format!("failed to find match at {}", at),
                    )?;
                let end = match result {
                    None => break,
                    Some(end) => end,
                };
                // Always advance one byte, in the case of an zero-width match.
                at = cmp::max(at + 1, at + end.offset());
                counts[end.pattern()] += 1;
                if find.matches() {
                    write_half_match(end, buf);
                }
            }
        }
        config::FindKind::Leftmost => {
            while at < haystack.len() {
                let result =
                    dfa.find_leftmost_fwd(&haystack[at..]).with_context(
                        || format!("failed to find match at {}", at),
                    )?;
                let end = match result {
                    None => break,
                    Some(end) => end,
                };
                // Always advance one byte, in the case of an zero-width match.
                at = cmp::max(at + 1, at + end.offset());
                counts[end.pattern()] += 1;
                if find.matches() {
                    write_half_match(end, buf);
                }
            }
        }
        config::FindKind::Overlapping => {
            let mut state = hybrid::OverlappingState::start();
            while at < haystack.len() {
                let result = dfa
                    .find_overlapping_fwd_at(
                        None,
                        haystack,
                        at,
                        haystack.len(),
                        &mut state,
                    )
                    .with_context(|| {
                        format!("failed to find match at {}", at)
                    })?;
                let end = match result {
                    None => {
                        break;
                    }
                    Some(end) => end,
                };
                // Unlike the non-overlapping case, we're OK with empty matches
                // at this level. In particular, the overlapping search
                // algorithm is itself responsible for ensuring that progress
                // is always made. (The starting position of the search is
                // incremented by 1 whenever a non-None state ID is given.)
                at = end.offset();
                counts[end.pattern()] += 1;
                if find.matches() {
                    write_half_match(end, buf);
                }
            }
        }
    }
    Ok(counts)
}

fn search_hybrid_regex(
    re: &hybrid::regex::Regex,
    cache: &mut hybrid::regex::Cache,
    find: &config::Find,
    haystack: &[u8],
    buf: &mut String,
) -> anyhow::Result<Vec<u64>> {
    let mut count = 0;
    let mut counts = vec![0u64; re.pattern_count()];
    match find.kind() {
        config::FindKind::Earliest => {
            for result in re.try_find_earliest_iter(cache, haystack) {
                let m = result.with_context(|| {
                    format!("search failure after {} matches", count)
                })?;
                count += 1;
                counts[m.pattern()] += 1;
                if find.matches() {
                    write_multi_match(m, buf);
                }
            }
        }
        config::FindKind::Leftmost => {
            for result in re.try_find_leftmost_iter(cache, haystack) {
                let m = result.with_context(|| {
                    format!("search failure after {} matches", count)
                })?;
                count += 1;
                counts[m.pattern()] += 1;
                if find.matches() {
                    write_multi_match(m, buf);
                }
            }
        }
        config::FindKind::Overlapping => {
            for result in re.try_find_overlapping_iter(cache, haystack) {
                let m = result.with_context(|| {
                    format!("search failure after {} matches", count)
                })?;
                count += 1;
                counts[m.pattern()] += 1;
                if find.matches() {
                    write_multi_match(m, buf);
                }
            }
        }
    }
    Ok(counts)
}

fn write_multi_match(m: automata::MultiMatch, buf: &mut String) {
    use std::fmt::Write;

    writeln!(buf, "{:?}: [{:?}, {:?})", m.pattern(), m.start(), m.end())
        .unwrap();
}

fn write_half_match(m: automata::HalfMatch, buf: &mut String) {
    use std::fmt::Write;

    writeln!(buf, "{:?}: {:?}", m.pattern(), m.offset()).unwrap();
}
