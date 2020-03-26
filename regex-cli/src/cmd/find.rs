use std::cmp;
use std::io::{stdout, Write};

use crate::{
    app::{self, App, Args},
    config,
    util::{self, Table},
};

use anyhow::Context;
use automata::{
    dfa::{self, Automaton, Regex},
    StateID,
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
}

pub fn run(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(args, define, |cmd, args| match cmd {
        "dfa" => run_dfa(args),
        _ => Err(util::UnrecognizedCommandError.into()),
    })
}

fn define_dfa() -> App {
    let mut dense = app::leaf("dense").about("Search using a dense DFA.");
    dense = config::File::define(dense);
    dense = config::Patterns::define(dense);
    dense = config::Syntax::define(dense);
    dense = config::Thompson::define(dense);
    dense = config::Dense::define(dense);
    dense = config::Find::define(dense);

    let mut sparse = app::leaf("sparse").about("Search using a sparse DFA.");
    sparse = config::File::define(sparse);
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
    dense = config::File::define(dense);
    dense = config::Patterns::define(dense);
    dense = config::Syntax::define(dense);
    dense = config::Thompson::define(dense);
    dense = config::Dense::define(dense);
    dense = config::RegexDFA::define(dense);
    dense = config::Find::define(dense);

    let mut sparse = app::leaf("sparse").about("Search using a sparse DFA.");
    sparse = config::File::define(sparse);
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

fn run_dfa(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(args, define, |cmd, args| {
        let state_id = config::get_state_id_size(args)?;
        match cmd {
            "dense" => each_state_size!(state_id, run_dfa_dense, args),
            "sparse" => each_state_size!(state_id, run_dfa_sparse, args),
            "regex" => run_dfa_regex(args),
            _ => Err(util::UnrecognizedCommandError.into()),
        }
    })
}

fn run_dfa_dense<S: StateID>(args: &Args) -> anyhow::Result<()> {
    let mut table = Table::empty();

    let csyntax = config::Syntax::get(args)?;
    let cthompson = config::Thompson::get(args)?;
    let cdense = config::Dense::get(args)?;
    let file = config::File::get(args)?;
    let patterns = config::Patterns::get(args)?;
    let find = config::Find::get(args)?;

    let dfa = cdense.from_patterns_dense::<S>(
        &mut table, &csyntax, &cthompson, &cdense, &patterns,
    )?;

    let haystack = unsafe { file.mmap()? };
    let mut buf = String::new();
    let (count, time) =
        util::timeitr(|| search_automaton(&dfa, &find, &*haystack, &mut buf))?;
    table.add("search time", time);
    table.add("count", count);
    table.print(stdout())?;
    if !buf.is_empty() {
        write!(stdout(), "\n{}", buf)?;
    }
    Ok(())
}

fn run_dfa_sparse<S: StateID>(args: &Args) -> anyhow::Result<()> {
    let mut table = Table::empty();

    let csyntax = config::Syntax::get(args)?;
    let cthompson = config::Thompson::get(args)?;
    let cdense = config::Dense::get(args)?;
    let file = config::File::get(args)?;
    let patterns = config::Patterns::get(args)?;
    let find = config::Find::get(args)?;

    let dfa = cdense.from_patterns_sparse::<S>(
        &mut table, &csyntax, &cthompson, &cdense, &patterns,
    )?;

    let haystack = unsafe { file.mmap()? };
    let mut buf = String::new();
    let (count, time) =
        util::timeitr(|| search_automaton(&dfa, &find, &*haystack, &mut buf))?;
    table.add("search time", time);
    table.add("count", count);
    table.print(stdout())?;
    if !buf.is_empty() {
        write!(stdout(), "\n{}", buf)?;
    }
    Ok(())
}

fn run_dfa_regex(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(args, define, |cmd, args| {
        let state_id = config::get_state_id_size(args)?;
        match cmd {
            "dense" => each_state_size!(state_id, run_dfa_regex_dense, args),
            "sparse" => each_state_size!(state_id, run_dfa_regex_sparse, args),
            _ => Err(util::UnrecognizedCommandError.into()),
        }
    })
}

fn run_dfa_regex_dense<S: StateID>(args: &Args) -> anyhow::Result<()> {
    let mut table = Table::empty();

    let csyntax = config::Syntax::get(args)?;
    let cthompson = config::Thompson::get(args)?;
    let cdense = config::Dense::get(args)?;
    let cregex = config::RegexDFA::get(args)?;
    let file = config::File::get(args)?;
    let patterns = config::Patterns::get(args)?;
    let find = config::Find::get(args)?;

    let re = cregex.from_patterns_dense::<S>(
        &mut table, &csyntax, &cthompson, &cdense, &patterns,
    )?;

    let haystack = unsafe { file.mmap()? };
    let mut buf = String::new();
    let (count, time) =
        util::timeitr(|| search_regex(&re, &find, &*haystack, &mut buf))?;
    table.add("search time", time);
    table.add("count", count);
    table.print(stdout())?;
    if !buf.is_empty() {
        write!(stdout(), "\n{}", buf)?;
    }
    Ok(())
}

fn run_dfa_regex_sparse<S: StateID>(args: &Args) -> anyhow::Result<()> {
    let mut table = Table::empty();

    let csyntax = config::Syntax::get(args)?;
    let cthompson = config::Thompson::get(args)?;
    let cdense = config::Dense::get(args)?;
    let cregex = config::RegexDFA::get(args)?;
    let file = config::File::get(args)?;
    let patterns = config::Patterns::get(args)?;
    let find = config::Find::get(args)?;

    let re = cregex.from_patterns_sparse::<S>(
        &mut table, &csyntax, &cthompson, &cdense, &patterns,
    )?;

    let haystack = unsafe { file.mmap()? };
    let mut buf = String::new();
    let (count, time) =
        util::timeitr(|| search_regex(&re, &find, &*haystack, &mut buf))?;
    table.add("search time", time);
    table.add("count", count);
    table.print(stdout())?;
    if !buf.is_empty() {
        write!(stdout(), "\n{}", buf)?;
    }
    Ok(())
}

fn search_automaton<A: Automaton>(
    dfa: A,
    find: &config::Find,
    haystack: &[u8],
    buf: &mut String,
) -> anyhow::Result<Vec<u64>> {
    let mut counts = vec![0u64; dfa.patterns()];
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
                counts[end.pattern() as usize] += 1;
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
                counts[end.pattern() as usize] += 1;
                if find.matches() {
                    write_half_match(end, buf);
                }
            }
        }
        config::FindKind::Overlapping => {
            let mut state = dfa::State::start();
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
                counts[end.pattern() as usize] += 1;
                if find.matches() {
                    write_half_match(end, buf);
                }
            }
        }
    }
    Ok(counts)
}

fn search_regex<A: Automaton>(
    re: &Regex<A>,
    find: &config::Find,
    haystack: &[u8],
    buf: &mut String,
) -> anyhow::Result<Vec<u64>> {
    let mut count = 0;
    let mut counts = vec![0u64; re.patterns()];
    match find.kind() {
        config::FindKind::Earliest => {
            for result in re.try_find_earliest_iter(haystack) {
                let m = result.with_context(|| {
                    format!("search failure after {} matches", count)
                })?;
                count += 1;
                counts[m.pattern() as usize] += 1;
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
                counts[m.pattern() as usize] += 1;
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
                counts[m.pattern() as usize] += 1;
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

    writeln!(buf, "{}: [{}, {})", m.pattern(), m.start(), m.end()).unwrap();
}

fn write_half_match(m: automata::dfa::HalfMatch, buf: &mut String) {
    use std::fmt::Write;

    writeln!(buf, "{}: {}", m.pattern(), m.offset()).unwrap();
}
