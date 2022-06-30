mod app;
mod cmd;
mod config;
mod escape;
mod util;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = cmd::define().get_matches();
    cmd::run(&args)
}
