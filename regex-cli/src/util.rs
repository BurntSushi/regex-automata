use std::io::{self, Write};

use unicode_width::UnicodeWidthStr;

use crate::app::{App, Args};

/// An error that indicates that a sub-command was seen that was not
/// recognized.
///
/// This is a sentinel error that is always converted to a panic via
/// run_subcommand. Namely, not handling a defined sub-command is a programmer
/// error.
#[derive(Debug)]
pub struct UnrecognizedCommandError;

impl std::error::Error for UnrecognizedCommandError {}

impl std::fmt::Display for UnrecognizedCommandError {
    fn fmt(&self, _: &mut std::fmt::Formatter) -> std::fmt::Result {
        unreachable!()
    }
}

/// Choose the sub-command of 'args' to run with 'run'. If the sub-command
/// wasn't recognized or is unknown, then an error is returned.
pub fn run_subcommand(
    args: &Args,
    app: impl FnOnce() -> App,
    run: impl FnOnce(&str, &Args) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let (name, args) = args.subcommand();
    if name.is_empty() || args.is_none() {
        app().print_help()?;
        writeln!(std::io::stdout(), "")?;
        return Ok(());
    }
    let err = match run(name, args.unwrap()) {
        Ok(()) => return Ok(()),
        Err(err) => err,
    };
    if err.is::<UnrecognizedCommandError>() {
        // The programmer should handle all defined sub-commands,
        unreachable!("unrecognized command: {}", name);
    }
    Err(err)
}

/// Time an arbitrary operation.
pub fn timeit<T>(run: impl FnOnce() -> T) -> (T, std::time::Duration) {
    let start = std::time::Instant::now();
    let t = run();
    (t, start.elapsed())
}

/// Convenient time an operation that returns a result by packing the duration
/// into the `Ok` variant.
pub fn timeitr<T, E>(
    run: impl FnOnce() -> Result<T, E>,
) -> Result<(T, std::time::Duration), E> {
    let (result, time) = timeit(run);
    let t = result?;
    Ok((t, time))
}

/// Print the given text with an ASCII art underline beneath it.
///
/// If the given text is empty, then '<empty>' is printed.
pub fn print_with_underline<W: io::Write>(
    mut wtr: W,
    text: &str,
) -> io::Result<()> {
    let toprint = if text.is_empty() { "<empty>" } else { text };
    writeln!(wtr, "{}", toprint)?;
    writeln!(wtr, "{}", "-".repeat(toprint.width()))?;
    Ok(())
}

#[derive(Debug)]
pub struct Table {
    pairs: Vec<(&'static str, Box<dyn std::fmt::Debug>)>,
}

impl Table {
    pub fn empty() -> Table {
        Table { pairs: vec![] }
    }

    pub fn add<D: std::fmt::Debug + 'static>(
        &mut self,
        label: &'static str,
        value: D,
    ) {
        self.pairs.push((label, Box::new(value)));
    }

    pub fn print<W: io::Write>(&self, wtr: W) -> io::Result<()> {
        let mut wtr = tabwriter::TabWriter::new(wtr)
            .alignment(tabwriter::Alignment::Right);
        for (label, value) in self.pairs.iter() {
            writeln!(wtr, "{}:\t{:?}", label, value)?;
        }
        wtr.flush()
    }
}
