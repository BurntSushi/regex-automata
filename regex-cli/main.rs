use std::io::Write;

mod args;
mod cmd;
mod util;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    if let Err(err) = cmd::run(&mut lexopt::Parser::from_env()) {
        if std::env::var("RUST_BACKTRACE").map_or(false, |v| v == "1") {
            writeln!(&mut std::io::stderr(), "{:?}", err).unwrap();
        } else {
            writeln!(&mut std::io::stderr(), "{:#}", err).unwrap();
        }
        std::process::exit(1);
    }
    Ok(())
}