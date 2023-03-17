use crate::args;

mod debug;
mod find;

const USAGE: &'static str = "\
A tool for interacting with Rust's regex crate on the command line.

USAGE:
    regex-cli <command> ...

COMMANDS:
    debug    Print the debug representation of things from regex-automata.
    find     Search haystacks with one of many different regex engines.
";

pub fn run(p: &mut lexopt::Parser) -> anyhow::Result<()> {
    let cmd = args::next_as_command(USAGE, p)?;
    match &*cmd {
        "find" => find::run(p),
        "debug" => debug::run(p),
        unk => anyhow::bail!("unrecognized command '{}'", unk),
    }
}
