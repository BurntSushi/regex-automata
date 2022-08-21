use std::io::stdout;

use crate::{
    app::{self, App, Args},
    config,
    util::{self, Table},
};

use automata::{
    dfa::Automaton,
    hybrid,
    nfa::thompson::pikevm::{self, PikeVM},
    Input, PatternSet,
};

const ABOUT_SHORT: &'static str = "\
Shows which patterns match a haystack.
";

const ABOUT_LONG: &'static str = "\
Shows which patterns match a haystack.
";

pub fn define() -> App {
    app::command("which")
        .about(ABOUT_SHORT)
        .before_help(ABOUT_LONG)
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
    let mut regex = app::leaf("regexset").about("Search using a 'Regex'.");
    regex = config::Haystack::define(regex);
    regex = config::Input::define(regex);
    regex = config::Patterns::define(regex);
    regex = config::Syntax::define(regex);
    regex = config::RegexSetAPI::define(regex);
    regex = config::Captures::define(regex);

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
    dense = config::Find::define(dense);

    let mut sparse = app::leaf("sparse").about("Search using a sparse DFA.");
    sparse = config::Haystack::define(sparse);
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
}

fn define_hybrid() -> App {
    let mut dfa = app::leaf("dfa").about("Search using a lazy DFA object.");
    dfa = config::Haystack::define(dfa);
    dfa = config::Input::define(dfa);
    dfa = config::Patterns::define(dfa);
    dfa = config::Syntax::define(dfa);
    dfa = config::Thompson::define(dfa);
    dfa = config::Hybrid::define(dfa);

    app::command("hybrid")
        .about("Search using a hybrid NFA/DFA object.")
        .subcommand(dfa)
}

fn define_nfa() -> App {
    app::command("nfa")
        .about("Search using an NFA.")
        .subcommand(define_nfa_thompson())
}

fn define_nfa_thompson() -> App {
    let mut pikevm = app::leaf("pikevm").about("Search using a Pike VM.");
    pikevm = config::Haystack::define(pikevm);
    pikevm = config::Input::define(pikevm);
    pikevm = config::Patterns::define(pikevm);
    pikevm = config::Syntax::define(pikevm);
    pikevm = config::Thompson::define(pikevm);
    pikevm = config::PikeVM::define(pikevm);

    app::command("thompson")
        .about("Search using a Thompson NFA.")
        .subcommand(pikevm)
}

fn run_api(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(args, define_api, |cmd, args| match cmd {
        "regexset" => run_api_regex(args),
        _ => Err(util::UnrecognizedCommandError.into()),
    })
}

fn run_api_regex(args: &Args) -> anyhow::Result<()> {
    let mut table = Table::empty();

    let csyntax = config::Syntax::get(args)?;
    let cregex = config::RegexSetAPI::get(args)?;
    let chaystack = config::Haystack::get(args)?;
    let cinput = config::Input::get(args)?;
    let cpatterns = config::Patterns::get(args)?;

    let re =
        cregex.from_patterns(&mut table, &csyntax, &cregex, &cpatterns)?;
    cinput.with_input(&chaystack, |input| {
        let (pids, time) = util::timeitr(|| search_api_regex(&re, input))?;
        table.add("search time", time);
        table.add("which", pids);
        table.print(stdout())?;
        Ok(())
    })
}

fn run_dfa(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(args, define_dfa, |cmd, args| match cmd {
        "dense" => run_dfa_dense(args),
        "sparse" => run_dfa_sparse(args),
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

    let dfa = cdense.from_patterns_dense(
        &mut table, &csyntax, &cthompson, &cdense, &cpatterns,
    )?;
    cinput.with_input(&chaystack, |input| {
        let (pids, time) =
            util::timeitr(|| search_dfa_automaton(&dfa, input))?;
        table.add("search time", time);
        table.add("which", pids);
        table.print(stdout())?;
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

    let dfa = cdense.from_patterns_sparse(
        &mut table, &csyntax, &cthompson, &cdense, &cpatterns,
    )?;
    cinput.with_input(&chaystack, |input| {
        let (pids, time) =
            util::timeitr(|| search_dfa_automaton(&dfa, input))?;
        table.add("search time", time);
        table.add("which", pids);
        table.print(stdout())?;
        Ok(())
    })
}

fn run_hybrid(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(args, define_hybrid, |cmd, args| match cmd {
        "dfa" => run_hybrid_dfa(args),
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

    let dfa =
        cdfa.from_patterns(&mut table, &csyntax, &cthompson, &cpatterns)?;

    let (mut cache, time) = util::timeit(|| dfa.create_cache());
    table.add("create cache time", time);

    cinput.with_input(&chaystack, |input| {
        let (pids, time) =
            util::timeitr(|| search_hybrid_dfa(&dfa, &mut cache, input))?;
        table.add("search time", time);
        table.add("cache clear count", cache.clear_count());
        table.add("which", pids);
        table.print(stdout())?;
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
        "pikevm" => run_nfa_thompson_pikevm(args),
        _ => Err(util::UnrecognizedCommandError.into()),
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

    let re =
        cpikevm.from_patterns(&mut table, &csyntax, &cthompson, &cpatterns)?;

    let (mut cache, time) = util::timeit(|| re.create_cache());
    table.add("create cache time", time);

    cinput.with_input(&chaystack, |input| {
        let (pids, time) =
            util::timeitr(|| search_pikevm(&re, &mut cache, input))?;
        table.add("search time", time);
        table.add("which", pids);
        table.print(stdout())?;
        Ok(())
    })
}

fn search_api_regex(
    re: &regex::bytes::RegexSet,
    input: &Input<'_, '_>,
) -> anyhow::Result<Vec<usize>> {
    Ok(re.matches(input.haystack()).into_iter().collect())
}

fn search_dfa_automaton<A: Automaton>(
    dfa: A,
    input: &Input<'_, '_>,
) -> anyhow::Result<Vec<usize>> {
    let mut patset = PatternSet::new(dfa.pattern_len());
    dfa.try_which_overlapping_matches(&input, &mut patset)?;
    Ok(patset.iter().map(|pid| pid.as_usize()).collect())
}

fn search_hybrid_dfa<'i, 'c>(
    dfa: &hybrid::dfa::DFA,
    cache: &mut hybrid::dfa::Cache,
    input: &Input<'_, '_>,
) -> anyhow::Result<Vec<usize>> {
    let mut patset = PatternSet::new(dfa.pattern_len());
    dfa.try_which_overlapping_matches(cache, &input, &mut patset)?;
    Ok(patset.iter().map(|pid| pid.as_usize()).collect())
}

fn search_pikevm(
    re: &PikeVM,
    cache: &mut pikevm::Cache,
    input: &Input<'_, '_>,
) -> anyhow::Result<Vec<usize>> {
    let mut patset = PatternSet::new(re.get_nfa().pattern_len());
    re.which_overlapping_matches(cache, &input, &mut patset)?;
    Ok(patset.iter().map(|pid| pid.as_usize()).collect())
}
