use crate::{
    app::{self, App, Args},
    util::{self, ShortHumanDuration, Throughput},
};

mod measure;

const ABOUT_SHORT: &'static str = "\
Run and compare benchmarks.
";

const ABOUT_LONG: &'static str = "\
Run and compare benchmarks.
";

pub fn define() -> App {
    app::command("bench")
        .about(ABOUT_SHORT)
        .before_help(ABOUT_LONG)
        .subcommand(measure::define())
}

pub fn run(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(&args, define, |cmd, args| match cmd {
        "measure" => measure::run(args),
        _ => Err(util::UnrecognizedCommandError.into()),
    })
}
