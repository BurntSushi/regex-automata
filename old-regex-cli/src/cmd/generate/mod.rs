use crate::{
    app::{self, App, Args},
    util,
};

mod fowler;
mod unicode;

const ABOUT: &'static str = "\
Generate some kind of output.

This command contains a smattering of sub-commands that do some kind of output.
The output might be code (for building regexes to compile into your program)
or it might be test data (for converting the old Glenn Fowler tests into a more
structured format).
";

pub fn define() -> App {
    app::command("generate")
        .about("Generata some kind of output.")
        .before_help(ABOUT)
        .subcommand(self::fowler::define())
        .subcommand(self::unicode::define())
}

pub fn run(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(args, define, |cmd, args| match cmd {
        "fowler" => self::fowler::run(args),
        "unicode" => self::unicode::run(args),
        _ => Err(util::UnrecognizedCommandError.into()),
    })
}
