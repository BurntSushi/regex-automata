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
    PatternID, Search,
};

const ABOUT_SHORT: &'static str = "\
Counts occurrences of a regex and its capturing groups in a haystack.
";

const ABOUT_LONG: &'static str = "\
Counts occurrences of a regex and its capturing groups in a haystack.
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
                vm.get_nfa()
                    .capture_index_to_name(pid, i)
                    .map(|n| n.to_string())
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
            anyhow::bail!("earliest searches not supported");
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
            anyhow::bail!("overlapping searches not supported");
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
    let mut counts = vec![vec![]; vm.get_nfa().pattern_len()];
    for pid in vm.get_nfa().patterns() {
        counts[pid] = vec![0; vm.get_nfa().pattern_capture_len(pid)];
    }
    match captures.kind() {
        config::SearchKind::Earliest => {
            // The PikeVM has no 'earliest' captures iter, and using the
            // generic iterators is a little strained since they don't support
            // 'Captures' directly. So we just hand-write our own iterator.
            let mut search = Search::new(haystack)
                .earliest(true)
                .utf8(vm.get_config().get_utf8());
            let mut caps = vm.create_captures();
            let mut last_match_end: Option<usize> = None;
            loop {
                vm.search(cache, None, &search, &mut caps);
                let m = match caps.get_match() {
                    None => break,
                    Some(m) => m,
                };
                search.set_start(m.end());
                if m.is_empty() {
                    // After every empty match, we forcefully step forward,
                    // since we know we'll otherwise run the search again
                    // at the same bounds, get the same result and then hit
                    // the 'last_match_end == Some(m.end())' case below. So
                    // doing this step for all empty matches isn't needed for
                    // correctness, but it avoids an additional search call in
                    // some common cases (e.g., for the empty regex).
                    search.step_byte();
                    // If we see an empty match that overlaps with the previous
                    // match, we skip this one and continue searching at the
                    // next byte.
                    //
                    // Because of the optimization above, this case is only
                    // triggered when a non-empty match is followed by an empty
                    // match.
                    if last_match_end == Some(m.end()) {
                        continue;
                    }
                }
                last_match_end = Some(m.end());

                for (group_index, subm) in caps.iter().enumerate() {
                    if subm.is_some() {
                        counts[m.pattern()][group_index] += 1;
                    }
                }
                if captures.matches() {
                    write_thompson_captures(vm.get_nfa(), &caps, buf);
                }
            }
        }
        config::SearchKind::Leftmost => {
            for caps in vm.captures_iter(cache, haystack) {
                let pid = caps.pattern().unwrap();
                for (group_index, m) in caps.iter().enumerate() {
                    if m.is_some() {
                        counts[pid][group_index] += 1;
                    }
                }
                if captures.matches() {
                    write_thompson_captures(vm.get_nfa(), &caps, buf);
                }
            }
        }
        config::SearchKind::Overlapping => {
            anyhow::bail!("overlapping searches not supported");
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
