#[macro_use]
mod macros;

mod app;
mod cmd;
mod config;
mod util;

fn main() -> anyhow::Result<()> {
    let args = app::root().get_matches();
    util::run_subcommand(&args, app::root, |cmd, args| match cmd {
        "debug" => cmd::debug::run(args),
        "find" => cmd::find::run(args),
        _ => Err(util::UnrecognizedCommandError.into()),
    })
}
