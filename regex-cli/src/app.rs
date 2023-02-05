use crate::cmd;

const TEMPLATE_ROOT: &'static str = "\
{bin} {version}
{author}
{about}
USAGE:
    {usage}

TIP:
    use -h for short docs and --help for long docs

SUBCOMMANDS:
{subcommands}

OPTIONS:
{options}";

const TEMPLATE_SUBCOMMAND: &'static str = "\
USAGE:
    {usage}

TIP:
    use -h for short docs and --help for long docs

SUBCOMMANDS:
{subcommands}

OPTIONS:
{options}";

const TEMPLATE_LEAF: &'static str = "\
USAGE:
    {usage}

TIP:
    use -h for short docs and --help for long docs

ARGS:
{positionals}

OPTIONS:
{options}";

const ABOUT: &'static str = "
regex-cli is a tool for interacting with regular expressions on the command
line. It is useful as a debugging aide, an ad hoc benchmarking tool and as a
way to conveniently pre-compile and embed regular expressions into Rust
code.
";

/// Convenience type alias for the Clap app type that we use.
pub type App = clap::Command;

/// Convenience type alias for the Clap argument result type that we use.
pub type Args = clap::ArgMatches;

/// Convenience function for creating a new Clap sub-command.
///
/// This should be used for sub-commands that contain other sub-commands.
pub fn command(name: &'static str) -> App {
    clap::Command::new(name)
        .author(clap::crate_authors!())
        .version(clap::crate_version!())
        .help_template(TEMPLATE_SUBCOMMAND)
}

/// Convenience function for creating a new Clap sub-command.
///
/// This should be used for sub-commands that do NOT contain other
/// sub-commands.
pub fn leaf(name: &'static str) -> App {
    clap::Command::new(name)
        .author(clap::crate_authors!())
        .version(clap::crate_version!())
        .help_template(TEMPLATE_LEAF)
}

/// Convenience function for defining a Clap positional argument with the
/// given name.
pub fn arg(name: &'static str) -> clap::Arg {
    clap::Arg::new(name)
}

/// Convenience function for defining a Clap argument with a long flag name
/// that accepts a single value.
pub fn flag(name: &'static str) -> clap::Arg {
    clap::Arg::new(name).long(name)
}

/// Convenience function for defining a Clap argument with a long flag name
/// that accepts no values. i.e., It is a boolean switch.
pub fn switch(name: &'static str) -> clap::Arg {
    clap::Arg::new(name).long(name).action(clap::ArgAction::SetTrue)
}

/// Build the main Clap application.
pub fn root() -> App {
    clap::Command::new("regex-cli")
        .author(clap::crate_authors!())
        .version(clap::crate_version!())
        .about(ABOUT)
        .help_template(TEMPLATE_ROOT)
        .max_term_width(100)
        .arg(switch("quiet").short('q').global(true).help("Show less output."))
        .subcommand(cmd::debug::define())
        .subcommand(cmd::find::define())
}
