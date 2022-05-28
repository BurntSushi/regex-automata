mod captures;
mod matches;
mod which;

use crate::{
    app::{self, App, Args},
    util,
};

const ABOUT_SHORT: &'static str = "\
Counts all occurrences of a regex in a file.
";

const ABOUT_LONG: &'static str = "\
Counts all occurrences of a regex in a file.
";

pub fn define() -> App {
    app::command("count")
        .about(ABOUT_SHORT)
        .before_help(ABOUT_LONG)
        .subcommand(captures::define())
        .subcommand(matches::define())
        .subcommand(which::define())
}

pub fn run(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(args, define, |cmd, args| match cmd {
        "captures" => captures::run(args),
        "matches" => matches::run(args),
        "which" => which::run(args),
        _ => Err(util::UnrecognizedCommandError.into()),
    })
}
