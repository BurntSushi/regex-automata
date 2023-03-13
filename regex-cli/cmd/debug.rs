use crate::{
    args,
    config::{configure, patterns, syntax, thompson, Configurable},
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

    let mut patterns = patterns::Config::positional();
    let mut syntax = syntax::Config::default();
    let mut thompson = thompson::Config::default();
    configure(p, USAGE, &mut [&mut patterns, &mut syntax, &mut thompson])?;

    let pats = patterns.get()?;
    let asts = syntax.asts(&pats)?;
    let hirs = syntax.hirs(&pats, &asts)?;
    let nfa = thompson.from_hirs(&hirs)?;

    println!("{:?}", nfa);
    Ok(())
}
