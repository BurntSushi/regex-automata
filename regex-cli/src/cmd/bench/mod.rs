use crate::{
    app::{self, App, Args},
    util,
};

mod count;

const ABOUT: &'static str = "\
Run benchmarks.
";

pub fn define() -> App {
    app::command("bench")
        .about("Run benchmarks.")
        .before_help(ABOUT)
        .subcommand(self::count::define())
}

pub fn run(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(&args, define, |cmd, args| match cmd {
        "count" => count::run(args),
        _ => Err(util::UnrecognizedCommandError.into()),
    })
}
