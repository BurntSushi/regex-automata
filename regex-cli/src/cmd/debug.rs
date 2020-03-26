use std::io::{stdout, Write};

use automata::StateID;

use crate::{
    app::{self, App, Args},
    config,
    util::{self, Table},
};

const ABOUT: &'static str = "\
Prints a debug representation of a regex object.

This is principally useful for debugging while doing development on regexes.
It makes it easy to print ASTs, HIRs, NFAs, DFAs and more in a format that is
easy to consume for humans.
";

pub fn define() -> App {
    let mut ast =
        app::leaf("ast").about("Print the AST of one or more regex patterns.");
    ast = config::Patterns::define(ast);
    ast = config::Syntax::define(ast);

    let mut hir =
        app::leaf("hir").about("Print the HIR of one or more regex patterns.");
    hir = config::Patterns::define(hir);
    hir = config::Syntax::define(hir);

    app::command("debug")
        .about("Print debug representation of a regex object.")
        .before_help(ABOUT)
        .subcommand(ast)
        .subcommand(hir)
        .subcommand(define_nfa())
        .subcommand(define_dfa())
}

pub fn run(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(args, define, |cmd, args| match cmd {
        "ast" => run_ast(args),
        "hir" => run_hir(args),
        "nfa" => run_nfa(args),
        "dfa" => run_dfa(args),
        _ => Err(util::UnrecognizedCommandError.into()),
    })
}

fn define_nfa() -> App {
    let mut thompson = app::leaf("thompson")
        .about("Print a debug representation of a Thompson NFA object.");
    thompson = config::Patterns::define(thompson);
    thompson = config::Syntax::define(thompson);
    thompson = config::Thompson::define(thompson);

    app::command("nfa")
        .about("Print a debug representation of an NFA object.")
        .subcommand(thompson)
}

fn define_dfa() -> App {
    let mut dense = app::leaf("dense")
        .about("Print a debug representation of a dense DFA object.");
    dense = config::Patterns::define(dense);
    dense = config::Syntax::define(dense);
    dense = config::Thompson::define(dense);
    dense = config::Dense::define(dense);

    let mut sparse = app::leaf("sparse")
        .about("Print a debug representation of a sparse DFA object.");
    sparse = config::Patterns::define(sparse);
    sparse = config::Syntax::define(sparse);
    sparse = config::Thompson::define(sparse);
    sparse = config::Dense::define(sparse);

    app::command("dfa")
        .about("Print a debug representation of a DFA object.")
        .subcommand(dense)
        .subcommand(sparse)
        .subcommand(define_dfa_regex())
}

fn define_dfa_regex() -> App {
    let mut dense = app::leaf("dense")
        .about("Print a debug representation of a dense Regex DFA object.");
    dense = config::Patterns::define(dense);
    dense = config::Syntax::define(dense);
    dense = config::Thompson::define(dense);
    dense = config::Dense::define(dense);
    dense = config::RegexDFA::define(dense);

    let mut sparse = app::leaf("sparse")
        .about("Print a debug representation of a sparse Regex DFA object.");
    sparse = config::Patterns::define(sparse);
    sparse = config::Syntax::define(sparse);
    sparse = config::Thompson::define(sparse);
    sparse = config::Dense::define(sparse);
    sparse = config::RegexDFA::define(sparse);

    app::command("regex")
        .about("Print a debug representation of a Regex DFA object.")
        .subcommand(dense)
        .subcommand(sparse)
}

fn run_ast(args: &Args) -> anyhow::Result<()> {
    let csyntax = config::Syntax::get(args)?;
    let patterns = config::Patterns::get(args)?;
    for (i, p) in patterns.into_iter().enumerate() {
        if i > 0 {
            writeln!(stdout(), "{}", "~".repeat(79))?;
        }

        util::print_with_underline(stdout(), &p)?;

        let mut table = Table::empty();
        let (ast, time_ast) = util::timeitr(|| csyntax.ast(&p))?;
        table.add("parse time", time_ast);
        table.print(stdout())?;
        if !args.is_present("quiet") {
            writeln!(stdout(), "\n{:#?}", ast)?;
        }
    }
    Ok(())
}

fn run_hir(args: &Args) -> anyhow::Result<()> {
    let csyntax = config::Syntax::get(args)?;
    let patterns = config::Patterns::get(args)?;
    for (i, p) in patterns.into_iter().enumerate() {
        if i > 0 {
            writeln!(stdout(), "{}", "~".repeat(79))?;
        }

        util::print_with_underline(stdout(), &p)?;

        let mut table = Table::empty();
        let (ast, time_ast) = util::timeitr(|| csyntax.ast(&p))?;
        table.add("parse time", time_ast);
        let (hir, time_hir) = util::timeitr(|| csyntax.hir(&p, &ast))?;
        table.add("translate time", time_hir);
        table.print(stdout())?;
        if !args.is_present("quiet") {
            writeln!(stdout(), "\n{:#?}", hir)?;
        }
    }
    Ok(())
}

fn run_nfa(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(args, define, |cmd, args| match cmd {
        "thompson" => run_nfa_thompson(args),
        _ => Err(util::UnrecognizedCommandError.into()),
    })
}

fn run_nfa_thompson(args: &Args) -> anyhow::Result<()> {
    let csyntax = config::Syntax::get(args)?;
    let cthompson = config::Thompson::get(args)?;
    let patterns = config::Patterns::get(args)?;
    let patterns = patterns.as_strings();

    let mut table = Table::empty();
    let (asts, time_ast) = util::timeitr(|| csyntax.asts(patterns))?;
    table.add("parse time", time_ast);
    let (hirs, time_hir) = util::timeitr(|| csyntax.hirs(patterns, &asts))?;
    table.add("translate time", time_hir);
    let (nfa, time_nfa) = util::timeitr(|| cthompson.from_hirs(&hirs))?;
    table.add("compile nfa time", time_nfa);
    table.add("pattern count", nfa.match_len());
    table.print(stdout())?;
    if !args.is_present("quiet") {
        writeln!(stdout(), "\n{:?}", nfa)?;
    }
    Ok(())
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
    let patterns = config::Patterns::get(args)?;

    let dfa = cdense.from_patterns_dense::<S>(
        &mut table, &csyntax, &cthompson, &cdense, &patterns,
    )?;
    table.print(stdout())?;
    if !args.is_present("quiet") {
        writeln!(stdout(), "\n{:?}", dfa)?;
    }
    Ok(())
}

fn run_dfa_sparse<S: StateID>(args: &Args) -> anyhow::Result<()> {
    let mut table = Table::empty();

    let csyntax = config::Syntax::get(args)?;
    let cthompson = config::Thompson::get(args)?;
    let cdense = config::Dense::get(args)?;
    let patterns = config::Patterns::get(args)?;

    let dfa = cdense.from_patterns_sparse::<S>(
        &mut table, &csyntax, &cthompson, &cdense, &patterns,
    )?;
    table.print(stdout())?;
    if !args.is_present("quiet") {
        writeln!(stdout(), "\n{:?}", dfa)?;
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
    let patterns = config::Patterns::get(args)?;

    let re = cregex.from_patterns_dense::<S>(
        &mut table, &csyntax, &cthompson, &cdense, &patterns,
    )?;
    table.print(stdout())?;
    if !args.is_present("quiet") {
        writeln!(stdout(), "\n{:?}", re)?;
    }
    Ok(())
}

fn run_dfa_regex_sparse<S: StateID>(args: &Args) -> anyhow::Result<()> {
    let mut table = Table::empty();

    let csyntax = config::Syntax::get(args)?;
    let cthompson = config::Thompson::get(args)?;
    let cdense = config::Dense::get(args)?;
    let cregex = config::RegexDFA::get(args)?;
    let patterns = config::Patterns::get(args)?;

    let sre = cregex.from_patterns_sparse::<S>(
        &mut table, &csyntax, &cthompson, &cdense, &patterns,
    )?;
    table.print(stdout())?;
    if !args.is_present("quiet") {
        writeln!(stdout(), "\n{:?}", sre)?;
    }
    Ok(())
}
