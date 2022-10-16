use std::{
    ffi::OsStr,
    io::{self, Write},
    time::Duration,
};

use {
    anyhow::Context,
    bstr::{BString, ByteVec},
    once_cell::sync::Lazy,
    regex::Regex,
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

/// Filter is the implementation of whitelist/blacklist rules. If there are no
/// rules, everything matches. If there's at least one whitelist rule, then you
/// need at least one whitelist rule to match to get through the filter. If
/// there are no whitelist regexes, then you can't match any of the blacklist
/// regexes.
///
/// This filter also has precedence built into that. That means that the order
/// of rules matters. So for example, if you have a whitelist regex that
/// matches AFTER a blacklist regex matches, then the input is considered to
/// have matched the filter.
#[derive(Clone, Debug)]
pub struct Filter {
    rules: Vec<FilterRule>,
}

/// A single rule in a filter, which is a combination of a regex and whether
/// it's a blacklist rule or not.
#[derive(Clone, Debug)]
struct FilterRule {
    regex: Regex,
    blacklist: bool,
}

impl Filter {
    /// Return a new filter from the given rules. The order of the rules
    /// matters, as the last rule that matches takes precedent over any
    /// previous matching rules.
    pub fn new<'a>(
        rules: impl Iterator<Item = &'a OsStr>,
    ) -> anyhow::Result<Filter> {
        let mut filter = Filter { rules: vec![] };
        for osrule in rules {
            let rule = match osrule.to_str() {
                Some(rule) => rule,
                None => {
                    let raw = BString::from(
                        Vec::from_os_str_lossy(osrule).into_owned(),
                    );
                    anyhow::bail!("regex is not UTF-8: '{}'", raw)
                }
            };
            let (pattern, blacklist) = if rule.starts_with('!') {
                (&rule[1..], true)
            } else {
                (&*rule, false)
            };
            let regex = Regex::new(pattern).context("regex is not valid")?;
            filter.rules.push(FilterRule { regex, blacklist });
        }
        Ok(filter)
    }

    /// Return true if and only if the given subject passes this filter.
    pub fn include(&self, subject: &str) -> bool {
        // If we have no rules, then everything matches.
        if self.rules.is_empty() {
            return true;
        }
        // If we have any whitelist rules, then 'include' starts off as false,
        // as we need at least one whitelist rule in that case to match. If all
        // we have are blacklists though, then we start off with include=true,
        // and we only get excluded if one of those blacklists is matched.
        let mut include = self.rules.iter().all(|r| r.blacklist);
        for rule in &self.rules {
            if rule.regex.is_match(subject) {
                include = !rule.blacklist;
            }
        }
        include
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
/// Hah, incidentally, when I wrote this, I had forgotten that I already had
/// an indirect dependency on 'humantime' via 'env_logger'. I decided to keep
/// this type because it lets us precisely control the format we support.
#[derive(Clone, Copy, Default)]
pub struct ShortHumanDuration(Duration);

impl ShortHumanDuration {
    pub fn serialize_with<S: serde::Serializer>(
        d: &Duration,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        serde::Serialize::serialize(&ShortHumanDuration::from(*d), s)
    }

    pub fn deserialize_with<'de, D: serde::Deserializer<'de>>(
        d: D,
    ) -> Result<Duration, D::Error> {
        let sdur: ShortHumanDuration = serde::Deserialize::deserialize(d)?;
        Ok(Duration::from(sdur))
    }
}

impl From<ShortHumanDuration> for Duration {
    fn from(hdur: ShortHumanDuration) -> Duration {
        hdur.0
    }
}

impl From<Duration> for ShortHumanDuration {
    fn from(dur: Duration) -> ShortHumanDuration {
        ShortHumanDuration(dur)
    }
}

impl std::fmt::Debug for ShortHumanDuration {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl std::fmt::Display for ShortHumanDuration {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let v = self.0.as_secs_f64();
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
            Regex::new(
                r"(?x)
                ^
                (?P<float>[0-9]+(?:\.[0-9]*)?|\.[0-9]+)
                (?P<units>s|ms|us|ns)
                $
            ",
            )
            .unwrap()
        });
        let caps = match RE.captures(s) {
            Some(caps) => caps,
            None => anyhow::bail!(
                "duration '{}' not in '<decimal>(s|ms|us|ns)' format",
                s,
            ),
        };
        let mut value: f64 =
            caps["float"].parse().context("invalid duration decimal")?;
        match &caps["units"] {
            "s" => value /= 1.0,
            "ms" => value /= 1_000.0,
            "us" => value /= 1_000_000.0,
            "ns" => value /= 1_000_000_000.0,
            unit => unreachable!("impossible unit '{}'", unit),
        }
        Ok(ShortHumanDuration(Duration::from_secs_f64(value)))
    }
}

impl serde::Serialize for ShortHumanDuration {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> serde::Deserialize<'de> for ShortHumanDuration {
    fn deserialize<D>(deserializer: D) -> Result<ShortHumanDuration, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct V;

        impl<'de> serde::de::Visitor<'de> for V {
            type Value = ShortHumanDuration;

            fn expecting(
                &self,
                f: &mut std::fmt::Formatter,
            ) -> std::fmt::Result {
                write!(f, "duration string of the form <decimal>(s|ms|us|ns)")
            }

            fn visit_str<E>(self, s: &str) -> Result<ShortHumanDuration, E>
            where
                E: serde::de::Error,
            {
                s.parse::<ShortHumanDuration>()
                    .map_err(|e| serde::de::Error::custom(e.to_string()))
            }
        }
        deserializer.deserialize_str(V)
    }
}

/// Another little wrapper type for computing, serializing and deserializing
/// throughput.
///
/// We fix our time units for throughput to "per second," but try to show
/// convenient size units, e.g., GB, MB, KB or B.
///
/// The internal representation is always in bytes per second.
#[derive(Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Throughput(f64);

impl Throughput {
    /// Create a new throughput from the given number of bytes and the amount
    /// of time taken to process those bytes.
    pub fn new(bytes: u64, duration: Duration) -> Throughput {
        let bytes_per_second = (bytes as f64) / duration.as_secs_f64();
        Throughput::from_bytes_per_second(bytes_per_second)
    }

    /// If you've already computed a throughput and it is in units of B/sec,
    /// then this permits building a `Throughput` from that raw value.
    pub fn from_bytes_per_second(bytes_per_second: f64) -> Throughput {
        Throughput(bytes_per_second)
    }
}

impl std::fmt::Debug for Throughput {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl std::fmt::Display for Throughput {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        const KB: f64 = (1 << 10) as f64;
        const MB: f64 = (1 << 20) as f64;
        const GB: f64 = (1 << 30) as f64;
        const MIN_KB: f64 = 2.0 * KB;
        const MIN_MB: f64 = 2.0 * MB;
        const MIN_GB: f64 = 2.0 * GB;

        let bytes_per_second = self.0 as f64;
        if bytes_per_second < MIN_KB {
            write!(f, "{} B/s", bytes_per_second as u64)
        } else if bytes_per_second < MIN_MB {
            write!(f, "{:.1} KB/s", bytes_per_second / KB)
        } else if bytes_per_second < MIN_GB {
            write!(f, "{:.1} MB/s", bytes_per_second / MB)
        } else {
            write!(f, "{:.1} GB/s", bytes_per_second / GB)
        }
    }
}

impl std::str::FromStr for Throughput {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Throughput> {
        static RE: Lazy<Regex> = Lazy::new(|| {
            Regex::new(
                r"(?x)
                ^
                (?P<float>[0-9]+(?:\.[0-9]*)?|\.[0-9]+)
                \s*
                (?P<units>B|KB|MB|GB)/s
                $
            ",
            )
            .unwrap()
        });
        let caps = match RE.captures(s) {
            Some(caps) => caps,
            None => anyhow::bail!(
                "throughput '{}' not in '<decimal>(B|KB|MB|GB)/s' format",
                s,
            ),
        };
        let mut bytes_per_second: f64 = caps["float"]
            .parse()
            .context("invalid throughput decimal number")?;
        match &caps["units"] {
            "B" => bytes_per_second *= (1 << 0) as f64,
            "KB" => bytes_per_second *= (1 << 10) as f64,
            "MB" => bytes_per_second *= (1 << 20) as f64,
            "GB" => bytes_per_second *= (1 << 30) as f64,
            unit => unreachable!("impossible unit '{}'", unit),
        }
        Ok(Throughput(bytes_per_second))
    }
}

impl serde::Serialize for Throughput {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> serde::Deserialize<'de> for Throughput {
    fn deserialize<D>(deserializer: D) -> Result<Throughput, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct V;

        impl<'de> serde::de::Visitor<'de> for V {
            type Value = Throughput;

            fn expecting(
                &self,
                f: &mut std::fmt::Formatter,
            ) -> std::fmt::Result {
                write!(
                    f,
                    "throughput string of the form <decimal>(B|KB|MB|GB)/s"
                )
            }

            fn visit_str<E>(self, s: &str) -> Result<Throughput, E>
            where
                E: serde::de::Error,
            {
                s.parse::<Throughput>()
                    .map_err(|e| serde::de::Error::custom(e.to_string()))
            }
        }
        deserializer.deserialize_str(V)
    }
}
