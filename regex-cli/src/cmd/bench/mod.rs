use std::time::Duration;

use crate::{
    app::{self, App, Args},
    util::{self, ShortHumanDuration, Throughput},
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
