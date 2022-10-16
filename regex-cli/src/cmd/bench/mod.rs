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

/// The in-memory representation of a single benchmark execution for a single
/// engine. It does not include all samples taken (those are thrown away and
/// not recorded anywhere), but does include aggregate statistics about the
/// samples.
///
/// Note that when 'err' is set, most other fields are set to their
/// empty/default values.
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
#[serde(from = "MeasurementWire", into = "MeasurementWire")]
struct Measurement {
    full_name: String,
    engine: String,
    err: Option<String>,
    iters: u64,
    total: Duration,
    aggregate: Aggregate,
}

/// The aggregate statistics computed from samples taken from a benchmark.
///
/// This includes aggregate timings and throughputs, but only the latter when
/// the benchmark includes a non-zero haystack length.
#[derive(Clone, Debug, Default)]
struct Aggregate {
    times: AggregateTimes,
    tputs: Option<AggregateThroughputs>,
}

/// The aggregate timings.
#[derive(Clone, Debug, Default)]
struct AggregateTimes {
    median: Duration,
    mean: Duration,
    stddev: Duration,
    min: Duration,
    max: Duration,
}

/// The aggregate throughputs. The `len` field is guaranteed to be non-zero.
#[derive(Clone, Debug, Default)]
struct AggregateThroughputs {
    len: u64,
    median: Throughput,
    mean: Throughput,
    stddev: Throughput,
    min: Throughput,
    max: Throughput,
}

impl Measurement {
    /// Get the corresponding throughput statistic from this aggregate.
    ///
    /// If this measurement doesn't have any throughputs (i.e., its haystack
    /// length is missing or zero), then this returns `None` regardless of the
    /// value of `stat`.
    fn throughput(&self, stat: Stat) -> Option<Throughput> {
        let tputs = self.aggregate.tputs.as_ref()?;
        Some(match stat {
            Stat::Median => tputs.median,
            Stat::Mean => tputs.mean,
            Stat::Stddev => tputs.stddev,
            Stat::Min => tputs.min,
            Stat::Max => tputs.max,
        })
    }

    /// Get the corresponding duration statistic from this aggregate.
    fn duration(&self, stat: Stat) -> Duration {
        let times = &self.aggregate.times;
        match stat {
            Stat::Median => times.median,
            Stat::Mean => times.mean,
            Stat::Stddev => times.stddev,
            Stat::Min => times.min,
            Stat::Max => times.max,
        }
    }
}

impl Aggregate {
    /// Creates a new set of aggregate statistics.
    ///
    /// If a non-zero haystack length is provided, then the aggregate returned
    /// includes throughputs.
    fn new(times: AggregateTimes, haystack_len: Option<u64>) -> Aggregate {
        let tputs = haystack_len.and_then(|len| {
            // We treat an explicit length of 0 and a totally missing value as
            // the same. In practice, there is no difference. We can't get a
            // meaningful throughput with a zero length haystack.
            if len == 0 {
                return None;
            }
            Some(AggregateThroughputs {
                len,
                median: Throughput::new(len, times.median),
                mean: Throughput::new(len, times.mean),
                stddev: Throughput::new(len, times.stddev),
                min: Throughput::new(len, times.min),
                max: Throughput::new(len, times.max),
            })
        });
        Aggregate { times, tputs }
    }
}

/// The wire Serde type corresponding to a single CSV record in the output
/// of 'regex-cli bench measure'.
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
struct MeasurementWire {
    full_name: String,
    engine: String,
    haystack_len: Option<u64>,
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

impl From<MeasurementWire> for Measurement {
    fn from(w: MeasurementWire) -> Measurement {
        let times = AggregateTimes {
            median: w.median,
            mean: w.mean,
            stddev: w.stddev,
            min: w.min,
            max: w.max,
        };
        let aggregate = Aggregate::new(times, w.haystack_len);
        Measurement {
            full_name: w.full_name,
            engine: w.engine,
            err: w.err,
            iters: w.iters,
            total: w.total,
            aggregate,
        }
    }
}

impl From<Measurement> for MeasurementWire {
    fn from(m: Measurement) -> MeasurementWire {
        MeasurementWire {
            full_name: m.full_name,
            engine: m.engine,
            haystack_len: m.aggregate.tputs.map(|x| x.len),
            err: m.err,
            iters: m.iters,
            total: m.total,
            median: m.aggregate.times.median,
            mean: m.aggregate.times.mean,
            stddev: m.aggregate.times.stddev,
            min: m.aggregate.times.min,
            max: m.aggregate.times.max,
        }
    }
}

/// A filter based on benchmark name.
#[derive(Clone, Debug)]
struct FilterByBenchmarkName(Filter);

impl FilterByBenchmarkName {
    /// Define a -f/--filter flag on the given app.
    fn define(app: App) -> App {
        const SHORT: &str = "Filter benchmarks by name using regex.";
        const LONG: &str = r#"\
Filter benchmarks by name using regex.

This flag may be given multiple times. The value can either be a whitelist
regex or a blacklist regex. To make it a blacklist regex, start it with a '!'.
If there is at least one whitelist regex, then a benchmark must match at least
one of them in order to be included. If there are no whitelist regexes, then a
benchmark is only included when it does not match any blacklist regexes. The
last filter regex that matches (whether it be a whitelist or a blacklist) is
what takes precedence. So for example, a whitelist regex that matches after a
blacklist regex matches, that would result in that benchmark being included in
the comparison.

So for example, consider the benchmarks 'foo', 'bar', 'baz' and 'quux'.

* "-f foo" will include "foo".
* "-f '!foo'" will include "bar", "baz" and "quux".
* "-f . -f '!ba' -f bar" will include "foo", "bar" and "quux".

Filter regexes are matched on the full name of the benchmark, which takes the
form '{type}/{group}/{name}'.
"#;
        app.arg(app::mflag("filter").short("f").help(SHORT).long_help(LONG))
    }

    /// Parse out filter from the CLI args given. If no rules were given, then
    /// an empty filter (which matches everything) is returned. If there was
    /// a problem parsing any of the filter rules, then an error is returned.
    fn get(args: &Args) -> anyhow::Result<Filter> {
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
    fn define(app: App) -> App {
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
    fn get(args: &Args) -> anyhow::Result<Filter> {
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
struct Threshold(f64);

impl Threshold {
    /// Define a -t/--threshold flag on the given app.
    fn define(app: App) -> App {
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
    /// value is invalid (i.e., not a non-negative integer), then an error
    /// is returned. If it isn't present, then `0` is returned (which is
    /// equivalent to no threshold).
    fn get(args: &Args) -> anyhow::Result<Threshold> {
        if let Some(threshold) = args.value_of_lossy("threshold") {
            let percent = threshold
                .parse::<u32>()
                .context("invalid integer percent threshold")?;
            Ok(Threshold(f64::from(percent)))
        } else {
            Ok(Threshold(0.0))
        }
    }

    /// Returns true if and only if the given difference exceeds or meets this
    /// threshold. When no threshold was given by the user, then a threshold of
    /// 0 is used, which everything exceeds or meets.
    fn include(&self, difference: f64) -> bool {
        !(difference < self.0)
    }
}

/// The choice of statistic to use. This is used in the commands for comparing
/// benchmark measurements.
#[derive(Clone, Copy, Debug)]
enum Stat {
    Median,
    Mean,
    Stddev,
    Min,
    Max,
}

impl Stat {
    /// Define a -s/--statistic flag on the given app.
    fn define(app: App) -> App {
        const SHORT: &str =
            "The aggregate statistic on which to compare (default: median).";
        const LONG: &str = "\
The aggregate statistic on which to compare (default: median).

Comparisons are only performed on the basis of a single statistic. The choices
are: median, mean, stddev, min, max.
";
        app.arg(app::flag("statistic").short("s").help(SHORT).long_help(LONG))
    }

    /// Read the selected statistic from the given CLI args. If one was not
    /// specified, then a default is returned. If one was specified but
    /// unrecognized, then an error is returned.
    fn get(args: &Args) -> anyhow::Result<Stat> {
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
            "stddev" => Stat::Stddev,
            "min" => Stat::Min,
            "max" => Stat::Max,
            unknown => {
                anyhow::bail!(
                    "unrecognized statistic name '{}', must be \
                     one of median, mean, stddev, min or max.",
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
    fn define(app: App) -> App {
        const SHORT: &str =
            "The units to use in comparisons (default: throughput).";
        const LONG: &str = "\
The units to use in comparisons (default: thoughput).

The same units are used in all comparisons. The choices are: time or thoughput.

If any particular group of measurements are all missing throughputs (i.e., when
their haystack length is missing or non-sensical), then timings are reported
for that group even if throughput is selected.
";
        app.arg(app::flag("units").short("u").help(SHORT).long_help(LONG))
    }

    /// Read the selected units from the given CLI args. If one was not
    /// specified, then a default is returned. If one was specified but
    /// unrecognized, then an error is returned.
    fn get(args: &Args) -> anyhow::Result<Units> {
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
