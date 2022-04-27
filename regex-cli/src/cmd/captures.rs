use std::{
    cmp,
    collections::BTreeMap,
    io::{stdout, Write},
};

use crate::{
    app::{self, App, Args},
    config,
    util::{self, Table},
};

use anyhow::Context;
use automata::{
    dfa::{self, Automaton},
    hybrid,
    nfa::thompson::pikevm::{self, PikeVM},
    PatternID,
};

const ABOUT_SHORT: &'static str = "\
Finds occurrences of a regex and its capturing groups in a haystack.
";

const ABOUT_LONG: &'static str = "\
Finds occurrences of a regex and its capturing groups in a haystack.
";

pub fn define() -> App {
    app::command("captures")
        .about(ABOUT_SHORT)
        .before_help(ABOUT_LONG)
        .subcommand(define_api())
        .subcommand(define_nfa())
}

pub fn run(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(args, define, |cmd, args| match cmd {
        "api" => run_api(args),
        "nfa" => run_nfa(args),
        _ => Err(util::UnrecognizedCommandError.into()),
    })
}

fn define_api() -> App {
    let mut regex = app::leaf("regex").about("Search using a 'Regex'.");
    regex = config::Input::define(regex);
    regex = config::Patterns::define(regex);
    regex = config::Syntax::define(regex);
    regex = config::RegexAPI::define(regex);
    regex = config::Captures::define(regex);

    app::command("api")
        .about("Search using a top-level 'regex' crate API.")
        .subcommand(regex)
}

fn define_nfa() -> App {
    app::command("nfa")
        .about("Search using an NFA.")
        .subcommand(define_nfa_thompson())
}

fn define_nfa_thompson() -> App {
    let mut pikevm = app::leaf("pikevm").about("Search using a Pike VM.");
    pikevm = config::Input::define(pikevm);
    pikevm = config::Patterns::define(pikevm);
    pikevm = config::Syntax::define(pikevm);
    pikevm = config::Thompson::define(pikevm);
    pikevm = config::PikeVM::define(pikevm);
    pikevm = config::Captures::define(pikevm);

    app::command("thompson")
        .about("Search using a Thompson NFA.")
        .subcommand(pikevm)
}

fn run_api(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(args, define, |cmd, args| match cmd {
        "regex" => run_api_regex(args),
        _ => Err(util::UnrecognizedCommandError.into()),
    })
}

fn run_api_regex(args: &Args) -> anyhow::Result<()> {
    let mut table = Table::empty();

    let csyntax = config::Syntax::get(args)?;
    let cthompson = config::Thompson::get(args)?;
    let cregex = config::RegexAPI::get(args)?;
    let input = config::Input::get(args)?;
    let patterns = config::Patterns::get(args)?;
    let captures = config::Captures::get(args)?;

    let re = cregex.from_patterns(&mut table, &csyntax, &cregex, &patterns)?;
    let index_to_name: Vec<Option<&str>> = re.capture_names().collect();
    input.with_mmap(|haystack| {
        let mut buf = String::new();
        let (counts, time) = util::timeitr(|| {
            search_api_regex(&re, &captures, &*haystack, &mut buf)
        })?;
        table.add("search time", time);
        let nicecaps = format_capture_counts(&counts, |i| {
            index_to_name[i].map(|n| n.to_string())
        });
        table.add("counts", nicecaps);
        table.print(stdout())?;
        if !buf.is_empty() {
            write!(stdout(), "\n{}", buf)?;
        }
        Ok(())
    })
}

fn run_nfa(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(args, define, |cmd, args| match cmd {
        "thompson" => run_nfa_thompson(args),
        _ => Err(util::UnrecognizedCommandError.into()),
    })
}

fn run_nfa_thompson(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(args, define, |cmd, args| match cmd {
        "pikevm" => run_nfa_thompson_pikevm(args),
        _ => Err(util::UnrecognizedCommandError.into()),
    })
}

fn run_nfa_thompson_pikevm(args: &Args) -> anyhow::Result<()> {
    let mut table = Table::empty();

    let csyntax = config::Syntax::get(args)?;
    let cthompson = config::Thompson::get(args)?;
    let cvm = config::PikeVM::get(args)?;
    let input = config::Input::get(args)?;
    let patterns = config::Patterns::get(args)?;
    let captures = config::Captures::get(args)?;

    let vm = cvm.from_patterns(&mut table, &csyntax, &cthompson, &patterns)?;

    let (mut cache, time) = util::timeit(|| vm.create_cache());
    table.add("create cache time", time);

    input.with_mmap(|haystack| {
        let mut buf = String::new();
        let (counts, time) = util::timeitr(|| {
            search_pikevm(&vm, &mut cache, &captures, &*haystack, &mut buf)
        })?;
        table.add("search time", time);
        for (pid, groups) in counts.iter().enumerate() {
            let pid = PatternID::must(pid);
            let nicecaps = format_capture_counts(groups, |i| {
                vm.nfa().capture_index_to_name(pid, i).map(|n| n.to_string())
            });
            table.add(&format!("counts({})", pid.as_usize()), nicecaps);
        }
        table.print(stdout())?;
        if !buf.is_empty() {
            write!(stdout(), "\n{}", buf)?;
        }
        Ok(())
    })
}

fn search_api_regex(
    re: &regex::bytes::Regex,
    captures: &config::Captures,
    haystack: &[u8],
    buf: &mut String,
) -> anyhow::Result<Vec<u64>> {
    let mut counts = vec![0; re.captures_len()];
    match captures.kind() {
        config::SearchKind::Earliest => {
            anyhow::bail!("API 'Regex' only supports leftmost searching");
        }
        config::SearchKind::Leftmost => {
            for caps in re.captures_iter(haystack) {
                for (group_index, m) in caps.iter().enumerate() {
                    if m.is_some() {
                        counts[group_index] += 1;
                    }
                }
                if captures.matches() {
                    write_api_captures(&caps, buf);
                }
            }
        }
        config::SearchKind::Overlapping => {
            anyhow::bail!("API 'Regex' only supports leftmost searching");
        }
    }
    Ok(counts)
}

fn search_pikevm(
    vm: &PikeVM,
    cache: &mut pikevm::Cache,
    captures: &config::Captures,
    haystack: &[u8],
    buf: &mut String,
) -> anyhow::Result<Vec<Vec<u64>>> {
    let mut counts = vec![vec![]; vm.nfa().pattern_len()];
    for pid in vm.nfa().patterns() {
        counts[pid] = vec![0; vm.nfa().pattern_capture_len(pid)];
    }
    match captures.kind() {
        config::SearchKind::Earliest => {
            for caps in vm.captures_earliest_iter(cache, haystack) {
                let pid = caps.pattern().unwrap();
                for (group_index, m) in caps.iter().enumerate() {
                    if m.is_some() {
                        counts[pid][group_index] += 1;
                    }
                }
                if captures.matches() {
                    write_thompson_captures(vm.nfa(), &caps, buf);
                }
            }
        }
        config::SearchKind::Leftmost => {
            for caps in vm.captures_leftmost_iter(cache, haystack) {
                let pid = caps.pattern().unwrap();
                for (group_index, m) in caps.iter().enumerate() {
                    if m.is_some() {
                        counts[pid][group_index] += 1;
                    }
                }
                if captures.matches() {
                    write_thompson_captures(vm.nfa(), &caps, buf);
                }
            }
        }
        config::SearchKind::Overlapping => {
            for caps in vm.captures_overlapping_iter(cache, haystack) {
                let pid = caps.pattern().unwrap();
                for (group_index, m) in caps.iter().enumerate() {
                    if m.is_some() {
                        counts[pid][group_index] += 1;
                    }
                }
                if captures.matches() {
                    write_thompson_captures(vm.nfa(), &caps, buf);
                }
            }
        }
    }
    Ok(counts)
}

fn format_capture_counts(
    caps: &[u64],
    mut get_name: impl FnMut(usize) -> Option<String>,
) -> String {
    use std::fmt::Write;

    let mut buf = String::new();
    write!(buf, "{{").unwrap();
    for (group_index, &count) in caps.iter().enumerate() {
        if group_index > 0 {
            write!(buf, ", ").unwrap();
        }
        write!(buf, "{}", group_index).unwrap();
        if let Some(name) = get_name(group_index) {
            write!(buf, "/{}", name).unwrap();
        }
        write!(buf, ": {}", count).unwrap();
    }
    write!(buf, "}}").unwrap();
    buf
}

fn write_api_captures(caps: &regex::bytes::Captures, buf: &mut String) {
    use std::fmt::Write;

    write!(buf, "0: {{").unwrap();
    for (group_index, m) in caps.iter().enumerate() {
        if group_index > 0 {
            write!(buf, ", ").unwrap();
        }
        write!(buf, "{}", group_index).unwrap();
        match m {
            None => write!(buf, ": ()").unwrap(),
            Some(m) => write!(buf, ": ({}, {})", m.start(), m.end()).unwrap(),
        }
    }
    write!(buf, "}}\n").unwrap();
}

fn write_thompson_captures(
    nfa: &automata::nfa::thompson::NFA,
    caps: &automata::nfa::thompson::Captures,
    buf: &mut String,
) {
    use std::fmt::Write;

    let pid = caps.pattern().unwrap();
    write!(buf, "{:?}: {{", pid).unwrap();
    for (group_index, m) in caps.iter().enumerate() {
        if group_index > 0 {
            write!(buf, ", ").unwrap();
        }
        write!(buf, "{}", group_index).unwrap();
        if let Some(name) = nfa.capture_index_to_name(pid, group_index) {
            write!(buf, "/{}", name).unwrap();
        }
        match m {
            None => write!(buf, ": ()").unwrap(),
            Some(m) => write!(buf, ": ({}, {})", m.start(), m.end()).unwrap(),
        }
    }
    write!(buf, "}}\n").unwrap();
}
