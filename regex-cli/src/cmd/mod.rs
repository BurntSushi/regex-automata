mod bench;
mod count;
mod debug;
mod generate;

use crate::{
    app::{self, App, Args},
    util,
};

pub fn define() -> App {
    app::root()
        .subcommand(bench::define())
        .subcommand(count::define())
        .subcommand(debug::define())
        .subcommand(generate::define())
}

pub fn run(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(&args, define, |cmd, args| match cmd {
        "bench" => bench::run(args),
        "count" => count::run(args),
        "debug" => debug::run(args),
        "generate" => generate::run(args),
        _ => Err(util::UnrecognizedCommandError.into()),
    })
}
