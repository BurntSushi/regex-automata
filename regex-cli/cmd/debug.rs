use std::io::{stdout, Write};

use regex_automata::dfa::Automaton;

use crate::{
    args::{
        self, common, configure, dfa, hybrid, meta, onepass, patterns, syntax,
        thompson, Configurable,
    },
    util::{self, Table},
};

pub fn run(p: &mut lexopt::Parser) -> anyhow::Result<()> {
    const USAGE: &'static str = "\
Prints the debug representation of various things from regex-automata and
regex-syntax.

USAGE:
    regex-cli debug <command> ...

COMMANDS:
    ast        Print the debug representation of an AST.
    dense      Print the debug representation of a dense DFA.
    hir        Print the debug representation of an HIR.
    literal    Print the debug representation of extracted literals.
    onepass    Print the debug representation of a one-pass DFA.
    sparse     Print the debug representation of a sparse DFA.
    thompson   Print the debug representation of a Thompson NFA.
";

    let cmd = args::next_as_command(USAGE, p)?;
    match &*cmd {
        "ast" => run_ast(p),
        "dense" => run_dense(p),
        "hir" => run_hir(p),
        "literal" => run_literal(p),
        "onepass" => run_onepass(p),
        "sparse" => run_sparse(p),
        "thompson" => run_thompson(p),
        unk => anyhow::bail!("unrecognized command '{}'", unk),
    }
}

fn run_ast(p: &mut lexopt::Parser) -> anyhow::Result<()> {
    const USAGE: &'static str = "\
Prints the debug representation of an abstract syntax tree (AST).

USAGE:
    regex-cli debug ast <pattern>

TIP:
    use -h for short docs and --help for long docs

OPTIONS:
%options%
";

    let mut common = common::Config::default();
    let mut patterns = patterns::Config::positional();
    let mut syntax = syntax::Config::default();
    configure(p, USAGE, &mut [&mut common, &mut patterns, &mut syntax])?;

    let pats = patterns.get()?;
    anyhow::ensure!(
        pats.len() == 1,
        "only one pattern is allowed, but {} were given",
        pats.len(),
    );

    let mut table = Table::empty();
    let (asts, time) = util::timeitr(|| syntax.asts(&pats))?;
    table.add("parse time", time);
    table.print(stdout())?;
    if !common.quiet {
        writeln!(stdout(), "\n{:#?}", &asts[0])?;
    }
    Ok(())
}

fn run_dense(p: &mut lexopt::Parser) -> anyhow::Result<()> {
    const USAGE: &'static str = "\
Prints the debug representation of a dense DFA or a dense DFA regex.

A DFA regex contains two DFAs: a forward DFA for finding the end of a match,
and a reverse DFA for finding the start of a match. These can be compiled
independently using just 'regex-cli debug dense dfa', but using the 'regex'
sub-command will handle it for you and print the debug representation of both
the forward and reverse DFAs.

USAGE:
    regex-cli debug dense <command> ...

COMMANDS:
    dfa    Print the debug representation of a single dense DFA.
    regex  Print the debug representation of a forward and reverse DFA regex.
";

    let cmd = args::next_as_command(USAGE, p)?;
    match &*cmd {
        "dfa" => run_dense_dfa(p),
        "regex" => run_dense_regex(p),
        unk => anyhow::bail!("unrecognized command '{}'", unk),
    }
}

fn run_dense_dfa(p: &mut lexopt::Parser) -> anyhow::Result<()> {
    const USAGE: &'static str = "\
Prints the debug representation of a fully compiled dense DFA.

USAGE:
    regex-cli debug dense dfa [<pattern> ...]

TIP:
    use -h for short docs and --help for long docs

OPTIONS:
%options%
";

    let mut common = common::Config::default();
    let mut patterns = patterns::Config::positional();
    let mut syntax = syntax::Config::default();
    let mut thompson = thompson::Config::default();
    let mut dfa = dfa::Config::default();
    configure(
        p,
        USAGE,
        &mut [
            &mut common,
            &mut patterns,
            &mut syntax,
            &mut thompson,
            &mut dfa,
        ],
    )?;

    let pats = patterns.get()?;
    let mut table = Table::empty();
    let (asts, time) = util::timeitr(|| syntax.asts(&pats))?;
    table.add("parse time", time);
    let (hirs, time) = util::timeitr(|| syntax.hirs(&pats, &asts))?;
    table.add("translate time", time);
    let (nfa, time) = util::timeitr(|| thompson.from_hirs(&hirs))?;
    table.add("compile nfa time", time);
    let (dfa, time) = util::timeitr(|| dfa.from_nfa(&nfa))?;
    table.add("compile dfa time", time);
    table.add("memory", dfa.memory_usage());
    table.add("pattern len", dfa.pattern_len());
    table.add("start kind", dfa.start_kind());
    table.add("alphabet len", dfa.alphabet_len());
    table.add("stride", dfa.stride());
    table.add("has empty?", dfa.has_empty());
    table.add("is utf8?", dfa.is_utf8());
    table.print(stdout())?;
    if !common.quiet {
        writeln!(stdout(), "\n{:?}", dfa)?;
    }
    Ok(())
}

fn run_dense_regex(p: &mut lexopt::Parser) -> anyhow::Result<()> {
    const USAGE: &'static str = "\
Prints the debug representation of a fully compiled dense DFA regex.

This includes both the forward and reverse DFAs that make up a dense DFA regex.

USAGE:
    regex-cli debug dense regex [<pattern> ...]

TIP:
    use -h for short docs and --help for long docs

OPTIONS:
%options%
";

    let mut common = common::Config::default();
    let mut patterns = patterns::Config::positional();
    let mut syntax = syntax::Config::default();
    let mut thompson = thompson::Config::default();
    let mut dfa = dfa::Config::default();
    configure(
        p,
        USAGE,
        &mut [
            &mut common,
            &mut patterns,
            &mut syntax,
            &mut thompson,
            &mut dfa,
        ],
    )?;

    let pats = patterns.get()?;
    let mut table = Table::empty();
    let (asts, time) = util::timeitr(|| syntax.asts(&pats))?;
    table.add("parse time", time);
    let (hirs, time) = util::timeitr(|| syntax.hirs(&pats, &asts))?;
    table.add("translate time", time);

    let (nfafwd, time) = util::timeitr(|| thompson.from_hirs(&hirs))?;
    table.add("compile forward nfa time", time);
    let (dfafwd, time) = util::timeitr(|| dfa.from_nfa(&nfafwd))?;
    table.add("compile forward dfa time", time);

    let (nfarev, time) =
        util::timeitr(|| thompson.reversed().from_hirs(&hirs))?;
    table.add("compile reverse nfa time", time);
    let (dfarev, time) = util::timeitr(|| dfa.reversed().from_nfa(&nfarev))?;
    table.add("compile reverse dfa time", time);

    let (re, time) = util::timeit(|| {
        regex_automata::dfa::regex::Builder::new()
            .build_from_dfas(dfafwd, dfarev)
    });
    table.add("build regex time", time);
    table.add(
        "memory",
        re.forward().memory_usage() + re.reverse().memory_usage(),
    );
    table.add("pattern len", re.pattern_len());
    table.print(stdout())?;
    if !common.quiet {
        writeln!(stdout(), "\n{:?}", re)?;
    }
    Ok(())
}

fn run_hir(p: &mut lexopt::Parser) -> anyhow::Result<()> {
    const USAGE: &'static str = "\
Prints the debug representation of a high-level intermediate representation
(HIR).

USAGE:
    regex-cli debug hir <pattern>

TIP:
    use -h for short docs and --help for long docs

OPTIONS:
%options%
";

    let mut common = common::Config::default();
    let mut patterns = patterns::Config::positional();
    let mut syntax = syntax::Config::default();
    configure(p, USAGE, &mut [&mut common, &mut patterns, &mut syntax])?;

    let pats = patterns.get()?;
    anyhow::ensure!(
        pats.len() == 1,
        "only one pattern is allowed, but {} were given",
        pats.len(),
    );

    let mut table = Table::empty();
    let (asts, time) = util::timeitr(|| syntax.asts(&pats))?;
    table.add("parse time", time);
    let (hirs, time) = util::timeitr(|| syntax.hirs(&pats, &asts))?;
    table.add("translate time", time);
    table.print(stdout())?;
    if !common.quiet {
        writeln!(stdout(), "\n{:#?}", &hirs[0])?;
    }
    Ok(())
}

fn run_literal(p: &mut lexopt::Parser) -> anyhow::Result<()> {
    const USAGE: &'static str = "\
Prints the debug representation of extract literals from a regex pattern.

USAGE:
    regex-cli debug literal <pattern>

TIP:
    use -h for short docs and --help for long docs

OPTIONS:
%options%
";

    let mut common = common::Config::default();
    let mut patterns = patterns::Config::positional();
    let mut syntax = syntax::Config::default();
    configure(p, USAGE, &mut [&mut common, &mut patterns, &mut syntax])?;

    let pats = patterns.get()?;
    anyhow::ensure!(
        pats.len() == 1,
        "only one pattern is allowed, but {} were given",
        pats.len(),
    );

    let mut table = Table::empty();
    let (asts, time) = util::timeitr(|| syntax.asts(&pats))?;
    table.add("parse time", time);
    let (hirs, time) = util::timeitr(|| syntax.hirs(&pats, &asts))?;
    table.add("translate time", time);
    // BREADCRUMBS: extract literals here. Expose options...
    table.print(stdout())?;
    if !common.quiet {
        writeln!(stdout(), "\n{:#?}", &hirs[0])?;
    }
    Ok(())
}

fn run_onepass(p: &mut lexopt::Parser) -> anyhow::Result<()> {
    const USAGE: &'static str = "\
Prints the debug representation of a one-pass DFA.

USAGE:
    regex-cli debug onepass [<pattern> ...]

TIP:
    use -h for short docs and --help for long docs

OPTIONS:
%options%
";

    let mut common = common::Config::default();
    let mut patterns = patterns::Config::positional();
    let mut syntax = syntax::Config::default();
    let mut thompson = thompson::Config::default();
    let mut onepass = onepass::Config::default();
    configure(
        p,
        USAGE,
        &mut [
            &mut common,
            &mut patterns,
            &mut syntax,
            &mut thompson,
            &mut onepass,
        ],
    )?;

    let pats = patterns.get()?;
    let mut table = Table::empty();
    let (asts, time) = util::timeitr(|| syntax.asts(&pats))?;
    table.add("parse time", time);
    let (hirs, time) = util::timeitr(|| syntax.hirs(&pats, &asts))?;
    table.add("translate time", time);
    let (nfa, time) = util::timeitr(|| thompson.from_hirs(&hirs))?;
    table.add("compile nfa time", time);
    let (dfa, time) = util::timeitr(|| onepass.from_nfa(&nfa))?;
    table.add("compile one-pass DFA time", time);
    table.add("memory", dfa.memory_usage());
    table.add("states", dfa.state_len());
    table.add("pattern len", dfa.pattern_len());
    table.add("alphabet len", dfa.alphabet_len());
    table.add("stride", dfa.stride());
    table.print(stdout())?;
    if !common.quiet {
        writeln!(stdout(), "\n{:?}", dfa)?;
    }
    Ok(())
}

fn run_sparse(p: &mut lexopt::Parser) -> anyhow::Result<()> {
    const USAGE: &'static str = "\
Prints the debug representation of a sparse DFA or a sparse DFA regex.

A DFA regex contains two DFAs: a forward DFA for finding the end of a match,
and a reverse DFA for finding the start of a match. These can be compiled
independently using just 'regex-cli debug dense dfa', but using the 'regex'
sub-command will handle it for you and print the debug representation of both
the forward and reverse DFAs.

USAGE:
    regex-cli debug sparse <command> ...

COMMANDS:
    dfa    Print the debug representation of a single sparse DFA.
    regex  Print the debug representation of a forward and reverse DFA regex.
";

    let cmd = args::next_as_command(USAGE, p)?;
    match &*cmd {
        "dfa" => run_sparse_dfa(p),
        "regex" => run_sparse_regex(p),
        unk => anyhow::bail!("unrecognized command '{}'", unk),
    }
}

fn run_sparse_dfa(p: &mut lexopt::Parser) -> anyhow::Result<()> {
    const USAGE: &'static str = "\
Prints the debug representation of a fully compiled sparse DFA.

USAGE:
    regex-cli debug sparse dfa [<pattern> ...]

TIP:
    use -h for short docs and --help for long docs

OPTIONS:
%options%
";

    let mut common = common::Config::default();
    let mut patterns = patterns::Config::positional();
    let mut syntax = syntax::Config::default();
    let mut thompson = thompson::Config::default();
    let mut dfa = dfa::Config::default();
    configure(
        p,
        USAGE,
        &mut [
            &mut common,
            &mut patterns,
            &mut syntax,
            &mut thompson,
            &mut dfa,
        ],
    )?;

    let pats = patterns.get()?;
    let mut table = Table::empty();
    let (asts, time) = util::timeitr(|| syntax.asts(&pats))?;
    table.add("parse time", time);
    let (hirs, time) = util::timeitr(|| syntax.hirs(&pats, &asts))?;
    table.add("translate time", time);
    let (nfa, time) = util::timeitr(|| thompson.from_hirs(&hirs))?;
    table.add("compile nfa time", time);
    let (dfa, time) = util::timeitr(|| dfa.from_nfa_sparse(&nfa))?;
    table.add("compile dfa time", time);
    table.add("memory", dfa.memory_usage());
    table.add("pattern len", dfa.pattern_len());
    table.add("start kind", dfa.start_kind());
    table.add("has empty?", dfa.has_empty());
    table.add("is utf8?", dfa.is_utf8());
    table.print(stdout())?;
    if !common.quiet {
        writeln!(stdout(), "\n{:?}", dfa)?;
    }
    Ok(())
}

fn run_sparse_regex(p: &mut lexopt::Parser) -> anyhow::Result<()> {
    const USAGE: &'static str = "\
Prints the debug representation of a fully compiled sparse DFA regex.

This includes both the forward and reverse DFAs that make up a sparse DFA
regex.

USAGE:
    regex-cli debug sparse regex [<pattern> ...]

TIP:
    use -h for short docs and --help for long docs

OPTIONS:
%options%
";

    let mut common = common::Config::default();
    let mut patterns = patterns::Config::positional();
    let mut syntax = syntax::Config::default();
    let mut thompson = thompson::Config::default();
    let mut dfa = dfa::Config::default();
    configure(
        p,
        USAGE,
        &mut [
            &mut common,
            &mut patterns,
            &mut syntax,
            &mut thompson,
            &mut dfa,
        ],
    )?;

    let pats = patterns.get()?;
    let mut table = Table::empty();
    let (asts, time) = util::timeitr(|| syntax.asts(&pats))?;
    table.add("parse time", time);
    let (hirs, time) = util::timeitr(|| syntax.hirs(&pats, &asts))?;
    table.add("translate time", time);

    let (nfafwd, time) = util::timeitr(|| thompson.from_hirs(&hirs))?;
    table.add("compile forward nfa time", time);
    let (dfafwd, time) = util::timeitr(|| dfa.from_nfa_sparse(&nfafwd))?;
    table.add("compile forward dfa time", time);

    let (nfarev, time) =
        util::timeitr(|| thompson.reversed().from_hirs(&hirs))?;
    table.add("compile reverse nfa time", time);
    let (dfarev, time) =
        util::timeitr(|| dfa.reversed().from_nfa_sparse(&nfarev))?;
    table.add("compile reverse dfa time", time);

    let (re, time) = util::timeit(|| {
        regex_automata::dfa::regex::Builder::new()
            .build_from_dfas(dfafwd, dfarev)
    });
    table.add("build regex time", time);
    table.add(
        "memory",
        re.forward().memory_usage() + re.reverse().memory_usage(),
    );
    table.add("pattern len", re.pattern_len());
    table.print(stdout())?;
    if !common.quiet {
        writeln!(stdout(), "\n{:?}", re)?;
    }
    Ok(())
}

fn run_thompson(p: &mut lexopt::Parser) -> anyhow::Result<()> {
    const USAGE: &'static str = "\
Prints the debug representation of a Thompson NFA.

USAGE:
    regex-cli debug thompson [<pattern> ...]

TIP:
    use -h for short docs and --help for long docs

OPTIONS:
%options%
";

    let mut common = common::Config::default();
    let mut patterns = patterns::Config::positional();
    let mut syntax = syntax::Config::default();
    let mut thompson = thompson::Config::default();
    configure(
        p,
        USAGE,
        &mut [&mut common, &mut patterns, &mut syntax, &mut thompson],
    )?;

    let pats = patterns.get()?;
    let mut table = Table::empty();
    let (asts, time) = util::timeitr(|| syntax.asts(&pats))?;
    table.add("parse time", time);
    let (hirs, time) = util::timeitr(|| syntax.hirs(&pats, &asts))?;
    table.add("translate time", time);
    let (nfa, time) = util::timeitr(|| thompson.from_hirs(&hirs))?;
    table.add("compile nfa time", time);
    table.add("memory", nfa.memory_usage());
    table.add("states", nfa.states().len());
    table.add("pattern len", nfa.pattern_len());
    table.add("capture len", nfa.group_info().all_group_len());
    table.add("has empty?", nfa.has_empty());
    table.add("is utf8?", nfa.is_utf8());
    table.add("is reverse?", nfa.is_reverse());
    table.add(
        "line terminator",
        bstr::BString::from(&[nfa.look_matcher().get_line_terminator()][..]),
    );
    table.add("lookset any", nfa.look_set_any());
    table.add("lookset prefix any", nfa.look_set_prefix_any());
    table.add("lookset prefix all", nfa.look_set_prefix_all());
    table.print(stdout())?;
    if !common.quiet {
        writeln!(stdout(), "\n{:?}", nfa)?;
    }
    Ok(())
}
