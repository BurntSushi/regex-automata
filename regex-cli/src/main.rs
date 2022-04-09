#![allow(warnings)]

mod app;
mod cmd;
mod config;
mod escape;
mod util;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = app::root().get_matches();
    util::run_subcommand(&args, app::root, |cmd, args| match cmd {
        "captures" => cmd::captures::run(args),
        "debug" => cmd::debug::run(args),
        "find" => cmd::find::run(args),
        "generate" => cmd::generate::run(args),
        _ => Err(util::UnrecognizedCommandError.into()),
    })
}
