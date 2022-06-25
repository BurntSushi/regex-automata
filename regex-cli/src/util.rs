use std::{
    io::{self, Write},
    time::Duration,
};

use {
    anyhow::Context, once_cell::sync::Lazy, regex::Regex,
    unicode_width::UnicodeWidthStr,
};

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

/// A somewhat silly little thing that prints an aligned table of key-value
/// pairs. Keys can be any string and values can be anything that implements
/// Debug.
///
/// This table is used to print little bits of useful information about stuff.
#[derive(Debug)]
pub struct Table {
    pairs: Vec<(String, Box<dyn std::fmt::Debug>)>,
}

impl Table {
    pub fn empty() -> Table {
        Table { pairs: vec![] }
    }

    pub fn add<D: std::fmt::Debug + 'static>(
        &mut self,
        label: &str,
        value: D,
    ) {
        self.pairs.push((label.to_string(), Box::new(value)));
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

/// A simple little wrapper type around std::time::Duration that permits
/// serializing and deserializing using a basic human friendly short duration.
///
/// We can get away with being simple here by assuming the duration is short.
/// i.e., No longer than one minute. So all we handle here are seconds,
/// milliseconds, microseconds and nanoseconds.
///
/// This avoids bringing in another crate to do this work (like humantime).
#[derive(Clone)]
pub struct ShortHumanDuration {
    pub dur: Duration,
}

impl std::fmt::Debug for ShortHumanDuration {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl std::fmt::Display for ShortHumanDuration {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let v = self.dur.as_secs_f64();
        if v >= 0.950 {
            write!(f, "{:.2}s", v)
        } else if v >= 0.000_950 {
            write!(f, "{:.2}ms", v * 1_000.0)
        } else if v >= 0.000_000_950 {
            write!(f, "{:.2}us", v * 1_000_000.0)
        } else {
            write!(f, "{:.2}ns", v * 1_000_000_000.0)
        }
    }
}

impl std::str::FromStr for ShortHumanDuration {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<ShortHumanDuration> {
        static RE: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"^(?P<digits>[0-9]+)(?P<units>s|ms|us|ns)$").unwrap()
        });
        let caps = match RE.captures(s) {
            Some(caps) => caps,
            None => anyhow::bail!(
                "duration '{}' not in '[0-9]+(s|ms|us|ns)' format",
                s,
            ),
        };
        let mut value: u64 =
            caps["digits"].parse().context("invalid duration integer")?;
        match &caps["units"] {
            "s" => value *= 1_000_000_000,
            "ms" => value *= 1_000_000,
            "us" => value *= 1_000,
            "ns" => value *= 1,
            unit => unreachable!("impossible unit '{}'", unit),
        }
        Ok(ShortHumanDuration { dur: Duration::from_nanos(value) })
    }
}
