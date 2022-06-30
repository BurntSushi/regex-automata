use std::time::Duration;

use anyhow::Context;

use crate::{
    app::{self, App, Args},
    util::{self, Filter, ShortHumanDuration, Throughput},
};

mod cmp;
mod diff;
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
        .subcommand(cmp::define())
        .subcommand(diff::define())
        .subcommand(measure::define())
}

pub fn run(args: &Args) -> anyhow::Result<()> {
    util::run_subcommand(&args, define, |cmd, args| match cmd {
        "cmp" => cmp::run(args),
        "diff" => diff::run(args),
        "measure" => measure::run(args),
        _ => Err(util::UnrecognizedCommandError.into()),
    })
}

/// Like 'AggregateDuration', but uses throughputs instead of durations. In
/// my opinion, throughput is easier to reason about for regex benchmarks. It
/// gives you the same information, but it also gives you some intuition for
/// how long it will take to search some data. Namely, throughput provides more
/// bits of information when compared to benchmark iteration duration.
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
struct Aggregate {
    full_name: String,
    engine: String,
    haystack_len: u64,
    err: Option<String>,
    iters: u64,
    #[serde(serialize_with = "ShortHumanDuration::serialize_with")]
    #[serde(deserialize_with = "ShortHumanDuration::deserialize_with")]
    total: Duration,
    median: Throughput,
    mean: Throughput,
    stddev: Throughput,
    min: Throughput,
    max: Throughput,
}

impl Aggregate {
    /// Get the corresponding throughput statistic from this aggregate.
    fn throughput(&self, stat: Stat) -> Throughput {
        match stat {
            Stat::Median => self.median,
            Stat::Mean => self.mean,
            Stat::Min => self.min,
            Stat::Max => self.max,
        }
    }

    /// Get the corresponding duration statistic from this aggregate.
    fn duration(&self, stat: Stat) -> Duration {
        self.throughput(stat).duration(self.haystack_len)
    }
}

/// Aggregate statistics for a particular benchmark in terms of durations. This
/// is what we compute directly from the samples collected from a benchmark.
/// But we quickly convert it to an aggregate that uses throughputs instead,
/// based on the belief that they are easier to understand and related to real
/// world use cases.
///
/// This could probably be simplified somewhat by attaching a `Benchmark`
/// to it, but it is an intentionally flattened structure so as to make
/// (de)serializing a bit more convenient.
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
struct AggregateDuration {
    full_name: String,
    engine: String,
    haystack_len: u64,
    err: Option<String>,
    iters: u64,
    #[serde(serialize_with = "ShortHumanDuration::serialize_with")]
    #[serde(deserialize_with = "ShortHumanDuration::deserialize_with")]
    total: Duration,
    #[serde(serialize_with = "ShortHumanDuration::serialize_with")]
    #[serde(deserialize_with = "ShortHumanDuration::deserialize_with")]
    median: Duration,
    #[serde(serialize_with = "ShortHumanDuration::serialize_with")]
    #[serde(deserialize_with = "ShortHumanDuration::deserialize_with")]
    mean: Duration,
    #[serde(serialize_with = "ShortHumanDuration::serialize_with")]
    #[serde(deserialize_with = "ShortHumanDuration::deserialize_with")]
    stddev: Duration,
    #[serde(serialize_with = "ShortHumanDuration::serialize_with")]
    #[serde(deserialize_with = "ShortHumanDuration::deserialize_with")]
    min: Duration,
    #[serde(serialize_with = "ShortHumanDuration::serialize_with")]
    #[serde(deserialize_with = "ShortHumanDuration::deserialize_with")]
    max: Duration,
}

impl AggregateDuration {
    /// Convert this aggregate value from using duration to throughput.
    fn into_throughput(self) -> Aggregate {
        if self.err.is_some() {
            return Aggregate {
                full_name: self.full_name,
                engine: self.engine,
                err: self.err,
                ..Aggregate::default()
            };
        }
        // Getting stddev as a throughput is not quite as straight-forward. I
        // believe the correct thing to do here is to compute the ratio between
        // stddev and mean in terms of duration, then compute the throughput of
        // the mean and then use the ratio on the mean throughput to find the
        // stddev throughput.
        let ratio = self.stddev.as_secs_f64() / self.mean.as_secs_f64();
        let mean = Throughput::new(self.haystack_len, self.mean);
        let stddev =
            Throughput::from_bytes_per_second(mean.bytes_per_second() * ratio);
        Aggregate {
            full_name: self.full_name,
            engine: self.engine,
            haystack_len: self.haystack_len,
            err: self.err,
            iters: self.iters,
            total: self.total,
            median: Throughput::new(self.haystack_len, self.median),
            mean,
            stddev,
            // For throughput, min/max are flipped. Which makes sense, the
            // bigger the throughput, the better. But the smaller the duration,
            // the better.
            min: Throughput::new(self.haystack_len, self.max),
            max: Throughput::new(self.haystack_len, self.min),
        }
    }
}

/// A filter based on benchmark name.
#[derive(Clone, Debug)]
struct FilterByBenchmarkName(Filter);

impl FilterByBenchmarkName {
    /// Define a -f/--filter flag on the given app.
    pub fn define(app: App) -> App {
        const SHORT: &str = "Filter benchmarks by name using regex.";
        const LONG: &str = "\
Filter benchmarks by name using regex.

This flag may be given multiple times. The value can either be a whitelist
regex or a blacklist regex. To make it a blacklist regex, start it with a '~'.
If there is at least one whitelist regex, then a benchmark must match at least
one of them in order to be included. If there are no whitelist regexes, then a
benchmark is only included when it does not match any blacklist regexes. The
last filter regex that matches (whether it be a whitelist or a blacklist) is
what takes precedence. So for example, a whitelist regex that matches after a
blacklist regex matches, that would result in that benchmark being included in
the comparison.

So for example, consider the benchmarks 'foo', 'bar', 'baz' and 'quux'.

* '-f foo' will include 'foo'.
* '-f ~foo' will include 'bar', 'baz' and 'quux'.
* '-f . -f ~ba -f bar' will include 'foo', 'bar' and 'quux'.

Filter regexes are matched on the full name of the benchmark, which takes the
form '{type}/{group}/{name}'.
";
        app.arg(app::mflag("filter").short("f").help(SHORT).long_help(LONG))
    }

    /// Parse out filter from the CLI args given. If no rules were given, then
    /// an empty filter (which matches everything) is returned. If there was
    /// a problem parsing any of the filter rules, then an error is returned.
    pub fn get(args: &Args) -> anyhow::Result<Filter> {
        if let Some(rules) = args.values_of_os("filter") {
            Filter::new(rules).context("-f/--filter")
        } else {
            // OK because an empty filter can always be built.
            Filter::new([].into_iter())
        }
    }
}

/// A filter based on regex engine name.
#[derive(Clone, Debug)]
struct FilterByEngineName(Filter);

impl FilterByEngineName {
    /// Define a -e/--engine flag on the given app.
    pub fn define(app: App) -> App {
        const SHORT: &str =
            "Filter benchmarks by regex engine name using regex.";
        const LONG: &str = "\
Filter benchmarks by regex engine name using regex.

This is just like the -f/--filter flag (with the same whitelist/blacklist
rules), except it applies to which regex engines to include. For example, many
benchmarks list a number of regex engines that it should run with, but this
filter permits specifying a smaller set of regex engines to include.

This filter is applied to every benchmark. It is useful, for example, if you
only want to include benchmarks across two regex engines instead of all regex
engines that were run in any given benchmark.
";
        app.arg(app::mflag("engine").short("e").help(SHORT).long_help(LONG))
    }

    /// Parse out filter from the CLI args given. If no rules were given, then
    /// an empty filter (which matches everything) is returned. If there was
    /// a problem parsing any of the filter rules, then an error is returned.
    pub fn get(args: &Args) -> anyhow::Result<Filter> {
        if let Some(rules) = args.values_of_os("engine") {
            Filter::new(rules).context("-e/--engine")
        } else {
            // OK because an empty filter can always be built.
            Filter::new([].into_iter())
        }
    }
}

/// An integer threshold value that can be used to filter out results whose
/// differences are too small to care about.
#[derive(Clone, Debug)]
pub struct Threshold(f64);

impl Threshold {
    /// Define a -t/--threshold flag on the given app.
    pub fn define(app: App) -> App {
        const SHORT: &str =
            "The minimum threshold measurements must differ by to be shown.";
        const LONG: &str = "\
The minimum threshold measurements must differ by to be shown.

The value given here is a percentage. Only benchmarks containing measurements
with at least a difference of X% will be shown in the comparison output. So
for example, given '-t5', only benchmarks whose minimum and maximum measurement
differ by at least 5% will be shown.

By default, there is no threshold enforced. All benchmarks in the given data
set matching the filters are shown.
";
        app.arg(app::flag("threshold").short("t").help(SHORT).long_help(LONG))
    }

    /// Get the threshold value from the CLI arguments. If the threshold
    /// value is invalid (i.e., not an integer in the range [0, 100]), then
    /// an error is returned. If it isn't present, then `0` is returned (which
    /// is equivalent to no threshold).
    pub fn get(args: &Args) -> anyhow::Result<Threshold> {
        if let Some(threshold) = args.value_of_lossy("threshold") {
            let percent = threshold
                .parse::<u32>()
                .context("invalid integer percent threshold")?;
            anyhow::ensure!(
                percent <= 100,
                "threshold must be a percent integer in the range [0, 100]"
            );
            Ok(Threshold(f64::from(percent)))
        } else {
            Ok(Threshold(0.0))
        }
    }

    /// Returns true if and only if the given difference exceeds or meets this
    /// threshold. When no threshold was given by the user, then a threshold of
    /// 0 is used, which everything exceeds or meets.
    pub fn include(&self, difference: f64) -> bool {
        !(difference < self.0)
    }
}

/// The choice of statistic to use. This is used in the commands for comparing
/// benchmark measurements.
#[derive(Clone, Copy, Debug)]
enum Stat {
    Median,
    Mean,
    Min,
    Max,
}

impl Stat {
    /// Define a -s/--statistic flag on the given app.
    pub fn define(app: App) -> App {
        const SHORT: &str =
            "The aggregate statistic on which to compare (default: median).";
        const LONG: &str = "\
The aggregate statistic on which to compare (default: median).

Comparisons are only performed on the basis of a single statistic. The choices
are: median, mean, min, max.
";
        app.arg(app::flag("statistic").short("s").help(SHORT).long_help(LONG))
    }

    /// Read the selected statistic from the given CLI args. If one was not
    /// specified, then a default is returned. If one was specified but
    /// unrecognized, then an error is returned.
    pub fn get(args: &Args) -> anyhow::Result<Stat> {
        if let Some(statname) = args.value_of_lossy("statistic") {
            statname.parse()
        } else {
            Ok(Stat::Median)
        }
    }
}

impl std::str::FromStr for Stat {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Stat> {
        let stat = match s {
            "median" => Stat::Median,
            "mean" => Stat::Mean,
            "min" => Stat::Min,
            "max" => Stat::Max,
            unknown => {
                anyhow::bail!(
                    "unrecognized statistic name '{}', must be \
                     one of median, mean, min or max.",
                    unknown,
                )
            }
        };
        Ok(stat)
    }
}

/// The choice of units to use when representing an aggregate statistic based
/// on time.
#[derive(Clone, Copy, Debug)]
enum Units {
    Time,
    Throughput,
}

impl Units {
    /// Define a -u/--units flag on the given app.
    pub fn define(app: App) -> App {
        const SHORT: &str =
            "The units to use in comparisons (default: throughput).";
        const LONG: &str = "\
The units to use in comparisons (default: thoughput).

The same units are used in all comparisons. The choices are: time or thoughput.
";
        app.arg(app::flag("units").short("u").help(SHORT).long_help(LONG))
    }

    /// Read the selected units from the given CLI args. If one was not
    /// specified, then a default is returned. If one was specified but
    /// unrecognized, then an error is returned.
    pub fn get(args: &Args) -> anyhow::Result<Units> {
        if let Some(unitname) = args.value_of_lossy("units") {
            unitname.parse()
        } else {
            Ok(Units::Throughput)
        }
    }
}

impl std::str::FromStr for Units {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Units> {
        let stat = match s {
            "time" => Units::Time,
            "throughput" => Units::Throughput,
            unknown => {
                anyhow::bail!(
                    "unrecognized units name '{}', must be \
                     one of time or throughput.",
                    unknown,
                )
            }
        };
        Ok(stat)
    }
}

/// Write the given divider character `width` times to the given writer.
fn write_divider<W: termcolor::WriteColor>(
    mut wtr: W,
    divider: char,
    width: usize,
) -> anyhow::Result<()> {
    let div: String = std::iter::repeat(divider).take(width).collect();
    write!(wtr, "{}", div)?;
    Ok(())
}
