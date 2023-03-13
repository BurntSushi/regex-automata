use std::io::{stdout, Write};

use crate::{
    args,
    config::{
        self, common, configure, patterns, syntax, thompson, Configurable,
    },
    util::{self, Table},
};

pub fn run(p: &mut lexopt::Parser) -> anyhow::Result<()> {
    const USAGE: &'static str = "\
Prints the debug representation of various things from regex-automata.

USAGE:
    regex-cli debug ...

COMMANDS:
    nfa    Print the debug representation of NFAs.
    dfa    Print the debug representation of DFAs.
";

    let cmd = args::next_as_command(USAGE, p)?;
    match &*cmd {
        "nfa" => run_nfa(p),
        unk => anyhow::bail!("unrecognized command '{}'", unk),
    }
}

pub fn run_nfa(p: &mut lexopt::Parser) -> anyhow::Result<()> {
    const USAGE: &'static str = "\
Prints the debug representation of NFAs in regex-automata.

USAGE:
    regex-cli debug nfa ...

COMMANDS:
    thompson    Print the debug representation of Thompson NFAs.
";

    let cmd = args::next_as_command(USAGE, p)?;
    match &*cmd {
        "thompson" => run_nfa_thompson(p),
        unk => anyhow::bail!("unrecognized command '{}'", unk),
    }
}

pub fn run_nfa_thompson(p: &mut lexopt::Parser) -> anyhow::Result<()> {
    const USAGE: &'static str = "\
Prints the debug representation of a Thompson NFA.

USAGE:
    regex-cli debug nfa thompson [<pattern> ...]

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
    table.add("nfa memory", nfa.memory_usage());
    table.add("nfa states", nfa.states().len());
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
