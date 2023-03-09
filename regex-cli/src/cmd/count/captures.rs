use std::io::{stdout, Write};

use crate::{
    app::{self, App, Args},
    config,
    util::{self, Table},
};

use automata::{
    dfa::onepass::{self, DFA},
    meta,
    nfa::thompson::{
        backtrack::{self, BoundedBacktracker},
        pikevm::{self, PikeVM},
    },
    util::iter,
    Input, PatternID,
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
        .subcommand(define_dfa())
        .subcommand(define_meta())
        .subcommand(define_nfa())
}

pub fn run(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(args, define, |cmd, args| match cmd {
        "api" => run_api(args),
        "dfa" => run_dfa(args),
        "meta" => run_meta(args),
        "nfa" => run_nfa(args),
        _ => Err(util::UnrecognizedCommandError.into()),
    })
}

fn define_api() -> App {
    let mut regex = app::leaf("regex").about("Search using a 'Regex'.");
    regex = config::Haystack::define(regex);
    regex = config::Input::define(regex);
    regex = config::Patterns::define(regex);
    regex = config::Syntax::define(regex);
    regex = config::RegexAPI::define(regex);
    regex = config::Captures::define(regex);

    app::command("api")
        .about("Search using a top-level 'regex' crate API.")
        .subcommand(regex)
}

fn define_dfa() -> App {
    let mut onepass =
        app::leaf("onepass").about("Search using a one-pass DFA.");
    onepass = config::Haystack::define(onepass);
    onepass = config::Input::define(onepass);
    onepass = config::Patterns::define(onepass);
    onepass = config::Syntax::define(onepass);
    onepass = config::Thompson::define(onepass);
    onepass = config::OnePass::define(onepass);
    onepass = config::Find::define(onepass);

    app::command("dfa").about("Search using a DFA.").subcommand(onepass)
}

fn define_meta() -> App {
    let mut meta =
        app::leaf("meta").about("Search using the meta regex engine.");
    meta = config::Haystack::define(meta);
    meta = config::Input::define(meta);
    meta = config::Patterns::define(meta);
    meta = config::Syntax::define(meta);
    meta = config::Meta::define(meta);
    meta = config::Find::define(meta);
    meta
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
    backtrack = config::PikeVM::define(backtrack);
    backtrack = config::Captures::define(backtrack);

    let mut pikevm = app::leaf("pikevm").about("Search using a Pike VM.");
    pikevm = config::Haystack::define(pikevm);
    pikevm = config::Input::define(pikevm);
    pikevm = config::Patterns::define(pikevm);
    pikevm = config::Syntax::define(pikevm);
    pikevm = config::Thompson::define(pikevm);
    pikevm = config::PikeVM::define(pikevm);
    pikevm = config::Captures::define(pikevm);

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
    let cinput = config::Input::get(args)?;
    let cpatterns = config::Patterns::get(args)?;
    let ccaptures = config::Captures::get(args)?;

    let re =
        cregex.from_patterns(&mut table, &csyntax, &cregex, &cpatterns)?;
    let index_to_name: Vec<Option<&str>> = re.capture_names().collect();
    cinput.with_input(&chaystack, |input| {
        let mut buf = String::new();
        let (counts, time) = util::timeitr(|| {
            search_api_regex(&re, &ccaptures, input, &mut buf)
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

fn run_dfa(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(args, define_dfa, |cmd, args| match cmd {
        "onepass" => run_dfa_onepass(args),
        _ => Err(util::UnrecognizedCommandError.into()),
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
    let ccaptures = config::Captures::get(args)?;

    let re = conepass
        .from_patterns(&mut table, &csyntax, &cthompson, &cpatterns)?;

    let (mut cache, time) = util::timeit(|| re.create_cache());
    table.add("create cache time", time);

    cinput.with_input(&chaystack, |input| {
        let mut buf = String::new();
        let (counts, time) = util::timeitr(|| {
            search_onepass(&re, &mut cache, &ccaptures, input, &mut buf)
        })?;
        table.add("search time", time);
        for (pid, groups) in counts.iter().enumerate() {
            let pid = PatternID::must(pid);
            let nicecaps = format_capture_counts(groups, |i| {
                re.get_nfa()
                    .group_info()
                    .to_name(pid, i)
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

fn run_meta(args: &Args) -> anyhow::Result<()> {
    let mut table = Table::empty();

    let csyntax = config::Syntax::get(args)?;
    let cmeta = config::Meta::get(args)?;
    let chaystack = config::Haystack::get(args)?;
    let cinput = config::Input::get(args)?;
    let cpatterns = config::Patterns::get(args)?;
    let ccaptures = config::Captures::get(args)?;

    let re = cmeta.from_patterns(&mut table, &csyntax, &cpatterns)?;

    let (mut cache, time) = util::timeit(|| re.create_cache());
    table.add("create cache time", time);

    cinput.with_input(&chaystack, |input| {
        let mut buf = String::new();
        let (counts, time) = util::timeitr(|| {
            search_meta(&re, &mut cache, &ccaptures, input, &mut buf)
        })?;
        table.add("search time", time);
        let caps = re.create_captures();
        for (pid, groups) in counts.iter().enumerate() {
            let pid = PatternID::must(pid);
            let nicecaps = format_capture_counts(groups, |i| {
                caps.group_info().to_name(pid, i).map(|n| n.to_string())
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
    let ccaptures = config::Captures::get(args)?;

    let re = cbacktrack
        .from_patterns(&mut table, &csyntax, &cthompson, &cpatterns)?;

    let (mut cache, time) = util::timeit(|| re.create_cache());
    table.add("create cache time", time);

    cinput.with_input(&chaystack, |input| {
        let mut buf = String::new();
        let (counts, time) = util::timeitr(|| {
            search_backtrack(&re, &mut cache, &ccaptures, input, &mut buf)
        })?;
        table.add("search time", time);
        for (pid, groups) in counts.iter().enumerate() {
            let pid = PatternID::must(pid);
            let nicecaps = format_capture_counts(groups, |i| {
                re.get_nfa()
                    .group_info()
                    .to_name(pid, i)
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

fn run_nfa_thompson_pikevm(args: &Args) -> anyhow::Result<()> {
    let mut table = Table::empty();

    let csyntax = config::Syntax::get(args)?;
    let cthompson = config::Thompson::get(args)?;
    let cpikevm = config::PikeVM::get(args)?;
    let chaystack = config::Haystack::get(args)?;
    let cinput = config::Input::get(args)?;
    let cpatterns = config::Patterns::get(args)?;
    let ccaptures = config::Captures::get(args)?;

    let re =
        cpikevm.from_patterns(&mut table, &csyntax, &cthompson, &cpatterns)?;

    let (mut cache, time) = util::timeit(|| re.create_cache());
    table.add("create cache time", time);

    cinput.with_input(&chaystack, |input| {
        let mut buf = String::new();
        let (counts, time) = util::timeitr(|| {
            search_pikevm(&re, &mut cache, &ccaptures, input, &mut buf)
        })?;
        table.add("search time", time);
        for (pid, groups) in counts.iter().enumerate() {
            let pid = PatternID::must(pid);
            let nicecaps = format_capture_counts(groups, |i| {
                re.get_nfa()
                    .group_info()
                    .to_name(pid, i)
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
    input: &Input<'_>,
    buf: &mut String,
) -> anyhow::Result<Vec<u64>> {
    let mut counts = vec![0; re.captures_len()];
    for caps in re.captures_iter(input.haystack()) {
        for (group_index, m) in caps.iter().enumerate() {
            if m.is_some() {
                counts[group_index] += 1;
            }
        }
        if captures.matches() {
            write_api_captures(&caps, buf);
        }
    }
    Ok(counts)
}

fn search_meta(
    re: &meta::Regex,
    cache: &mut meta::Cache,
    captures: &config::Captures,
    input: &Input<'_>,
    buf: &mut String,
) -> anyhow::Result<Vec<Vec<u64>>> {
    let mut caps = re.create_captures();
    let mut counts = vec![vec![]; re.pattern_len()];
    for pid in 0..re.pattern_len() {
        let pid = PatternID::must(pid);
        counts[pid] = vec![0; caps.group_info().group_len(pid)];
    }
    // The standard iterators alloc a new 'Captures' for each match, so
    // we use a slightly less convenient API to reuse 'Captures' for
    // each match. Overall, this should result in zero amortized allocs
    // per match.
    let mut it = iter::Searcher::new(input.clone());
    loop {
        it.advance(|input| {
            re.search_captures(cache, input, &mut caps);
            Ok(caps.get_match())
        });
        let m = match caps.get_match() {
            None => break,
            Some(m) => m,
        };
        for (group_index, subm) in caps.iter().enumerate() {
            if subm.is_some() {
                counts[m.pattern()][group_index] += 1;
            }
        }
        if captures.matches() {
            write_automata_captures(&caps, buf);
        }
    }
    Ok(counts)
}

fn search_onepass(
    re: &DFA,
    cache: &mut onepass::Cache,
    captures: &config::Captures,
    input: &Input<'_>,
    buf: &mut String,
) -> anyhow::Result<Vec<Vec<u64>>> {
    let mut counts = vec![vec![]; re.get_nfa().pattern_len()];
    for pid in re.get_nfa().patterns() {
        counts[pid] = vec![0; re.get_nfa().group_info().group_len(pid)];
    }
    // The standard iterators alloc a new 'Captures' for each match, so
    // we use a slightly less convenient API to reuse 'Captures' for
    // each match. Overall, this should result in zero amortized allocs
    // per match.
    let mut caps = re.create_captures();
    let mut it = iter::Searcher::new(input.clone());
    loop {
        it.advance(|input| {
            re.try_search(cache, input, &mut caps)?;
            Ok(caps.get_match())
        });
        let m = match caps.get_match() {
            None => break,
            Some(m) => m,
        };
        for (group_index, subm) in caps.iter().enumerate() {
            if subm.is_some() {
                counts[m.pattern()][group_index] += 1;
            }
        }
        if captures.matches() {
            write_automata_captures(&caps, buf);
        }
    }
    Ok(counts)
}

fn search_backtrack(
    re: &BoundedBacktracker,
    cache: &mut backtrack::Cache,
    captures: &config::Captures,
    input: &Input<'_>,
    buf: &mut String,
) -> anyhow::Result<Vec<Vec<u64>>> {
    let mut counts = vec![vec![]; re.get_nfa().pattern_len()];
    for pid in re.get_nfa().patterns() {
        counts[pid] = vec![0; re.get_nfa().group_info().group_len(pid)];
    }
    // The standard iterators alloc a new 'Captures' for each match, so
    // we use a slightly less convenient API to reuse 'Captures' for
    // each match. Overall, this should result in zero amortized allocs
    // per match.
    let mut caps = re.create_captures();
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
        for (group_index, subm) in caps.iter().enumerate() {
            if subm.is_some() {
                counts[m.pattern()][group_index] += 1;
            }
        }
        if captures.matches() {
            write_automata_captures(&caps, buf);
        }
    }
    Ok(counts)
}

fn search_pikevm(
    re: &PikeVM,
    cache: &mut pikevm::Cache,
    captures: &config::Captures,
    input: &Input<'_>,
    buf: &mut String,
) -> anyhow::Result<Vec<Vec<u64>>> {
    let mut counts = vec![vec![]; re.get_nfa().pattern_len()];
    for pid in re.get_nfa().patterns() {
        counts[pid] = vec![0; re.get_nfa().group_info().group_len(pid)];
    }
    // The standard iterators alloc a new 'Captures' for each match, so
    // we use a slightly less convenient API to reuse 'Captures' for
    // each match. Overall, this should result in zero amortized allocs
    // per match.
    let mut caps = re.create_captures();
    let mut it = iter::Searcher::new(input.clone());
    loop {
        it.try_advance(|input| {
            re.search(cache, input, &mut caps);
            Ok(caps.get_match())
        })?;
        let m = match caps.get_match() {
            None => break,
            Some(m) => m,
        };
        for (group_index, subm) in caps.iter().enumerate() {
            if subm.is_some() {
                counts[m.pattern()][group_index] += 1;
            }
        }
        if captures.matches() {
            write_automata_captures(&caps, buf);
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

fn write_automata_captures(
    caps: &automata::util::captures::Captures,
    buf: &mut String,
) {
    use std::fmt::Write;

    let pid = caps.pattern().unwrap();
    write!(buf, "{:?}: {{", pid).unwrap();
    for (group_index, span) in caps.iter().enumerate() {
        if group_index > 0 {
            write!(buf, ", ").unwrap();
        }
        write!(buf, "{}", group_index).unwrap();
        if let Some(name) = caps.group_info().to_name(pid, group_index) {
            write!(buf, "/{}", name).unwrap();
        }
        match span {
            None => write!(buf, ": ()").unwrap(),
            Some(span) => {
                write!(buf, ": ({}, {})", span.start, span.end).unwrap()
            }
        }
    }
    write!(buf, "}}\n").unwrap();
}
