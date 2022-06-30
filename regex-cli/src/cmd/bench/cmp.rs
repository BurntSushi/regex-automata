use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};

use {anyhow::Context, termcolor::WriteColor};

use crate::{
    app::{self, App, Args},
    cmd::bench::Aggregate,
    util::{Filter, Throughput},
};

const ABOUT_SHORT: &'static str = "\
Compare benchmarks between different regex engines.
";

const ABOUT_LONG: &'static str = "\
Compare benchmarks between different regex engines.

To compare benchmarks for the same regex engine over time, use the 'regex-cli
bench diff' command.
";

pub fn define() -> App {
    let mut app =
        app::command("cmp").about(ABOUT_SHORT).before_help(ABOUT_LONG);
    {
        const SHORT: &str =
            "File paths to CSV data containing benchmark measurements.";
        const LONG: &str = "\
File paths to CSV data containing benchmark measurements.

In general, the CSV data read by this tool is the same CSV data written by the
'regex-cli bench measure' command.

Multiple CSV data sets may be given to this command. However, if after applying
the filters given, there are any duplicate benchmark/engine pairs, then the
command will error.
";
        app = app.arg(
            app::arg("csv-path").multiple(true).help(SHORT).long_help(LONG),
        );
    }
    {
        const SHORT: &str =
            "Filter (using regex) which benchmarks to compare.";
        const LONG: &str = "\
Filter (using regex) which benchmarks to compare.

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
        app = app
            .arg(app::mflag("filter").short("f").help(SHORT).long_help(LONG));
    }
    {
        const SHORT: &str =
            "Filter (using regex) which regex engines to compare.";
        const LONG: &str = "\
Filter (using regex) which regex engines to compare.

This is just like the -f/--filter flag (with the same whitelist/blacklist
rules), except it applies to which regex engines to compare. For example, many
benchmarks list a number of regex engines that it should run with, but this
filter permits specifying a smaller set of regex engines to compare.

This filter is applied to every benchmark. It is useful, for example, if you
only want to compare benchmarks across two regex engines instead of all regex
engines that were run in that benchmark.
";
        app = app
            .arg(app::mflag("engine").short("e").help(SHORT).long_help(LONG));
    }
    {
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
        app = app.arg(
            app::mflag("threshold").short("t").help(SHORT).long_help(LONG),
        );
    }
    {
        const SHORT: &str =
            "The aggregate statistic on which to compare (default: median).";
        const LONG: &str = "\
The aggregate statistic on which to compare (default: median).

Comparisons are only performed on the basis of a single statistic. The choices
are: median, mean, min, max.
";
        app = app.arg(
            app::mflag("statistic").short("s").help(SHORT).long_help(LONG),
        );
    }
    {
        const SHORT: &str = "Whether to use color (default: auto).";
        const LONG: &str = "\
Whether to use color (default: auto).

When enabled, color is used to indicate which regex engine did the best on each
benchmark. The choices are: auto, always, never.
";
        app = app.arg(app::mflag("color").help(SHORT).long_help(LONG));
    }
    app
}

pub fn run(args: &Args) -> anyhow::Result<()> {
    let cmpargs = CmpArgs::new(args)?;
    let aggs = cmpargs.read_aggregates()?;
    let aggs_by_name = AggregatesByBenchmarkName::new(cmpargs.stat, aggs);
    let engines = aggs_by_name.engine_names();

    let mut wtr = cmpargs.elastic_stdout();

    // Write column names.
    write!(wtr, "benchmark")?;
    for engine in engines.iter() {
        write!(wtr, "\t{}", engine)?;
    }
    writeln!(wtr, "")?;

    // Write underlines beneath each column name to give some separation. Note
    // that we use byte length because we require that all names are ASCII.
    write_divider(&mut wtr, '-', "benchmark".len())?;
    for engine in engines.iter() {
        write!(wtr, "\t")?;
        write_divider(&mut wtr, '-', engine.len())?;
    }
    writeln!(wtr, "")?;

    for group in aggs_by_name.groups.iter() {
        if group.biggest_difference(cmpargs.stat) < cmpargs.threshold {
            continue;
        }
        write!(wtr, "{}", group.full_name)?;
        for engine in engines.iter() {
            write!(wtr, "\t")?;
            match group.aggs_by_engine.get(engine) {
                None => {
                    write!(wtr, "-")?;
                }
                Some(agg) => {
                    let best = engine == &*group.best_engine_name;
                    if best {
                        let mut spec = termcolor::ColorSpec::new();
                        spec.set_fg(Some(termcolor::Color::Green))
                            .set_bold(true);
                        wtr.set_color(&spec)?;
                    }
                    write!(wtr, "{}", cmpargs.stat.get(&agg))?;
                    if best {
                        wtr.reset()?;
                    }
                }
            }
        }
        writeln!(wtr, "")?;
    }
    wtr.flush()?;
    Ok(())
}

#[derive(Debug)]
struct CmpArgs {
    csv_paths: Vec<PathBuf>,
    bench_filter: Filter,
    engine_filter: Filter,
    stat: Stat,
    threshold: f64,
    color: Option<bool>,
}

impl CmpArgs {
    /// Parse 'cmp' args from the given CLI args.
    fn new(args: &Args) -> anyhow::Result<CmpArgs> {
        let mut cmpargs = CmpArgs {
            csv_paths: args
                .values_of_os("csv-path")
                .into_iter()
                .flatten()
                .map(PathBuf::from)
                .collect(),
            bench_filter: Filter::new([].into_iter()).unwrap(),
            engine_filter: Filter::new([].into_iter()).unwrap(),
            stat: Stat::Median,
            threshold: 0.0,
            color: None,
            // color: atty::is(atty::Stream::Stdout),
        };
        anyhow::ensure!(
            !cmpargs.csv_paths.is_empty(),
            "no CSV file paths given"
        );
        if let Some(rules) = args.values_of_os("filter") {
            cmpargs.bench_filter =
                Filter::new(rules).context("-f/--filter")?;
        }
        if let Some(rules) = args.values_of_os("engine") {
            cmpargs.engine_filter =
                Filter::new(rules).context("-e/--engine")?;
        }
        if let Some(threshold) = args.value_of_lossy("threshold") {
            let percent = threshold
                .parse::<u32>()
                .context("invalid integer percent threshold")?;
            anyhow::ensure!(
                percent <= 100,
                "threshold must be a percent integer in the range [0, 100]"
            );
            cmpargs.threshold = f64::from(percent);
        }
        if let Some(statname) = args.value_of_lossy("statistic") {
            cmpargs.stat = match &*statname {
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
        }
        if let Some(colorchoice) = args.value_of_lossy("color") {
            cmpargs.color = match &*colorchoice {
                "auto" => None,
                "always" => Some(true),
                "never" => Some(false),
                unknown => {
                    anyhow::bail!(
                        "unrecognized color config '{}', must be \
                         one of auto, always or never.",
                        unknown,
                    )
                }
            }
        }
        Ok(cmpargs)
    }

    /// Return a possible colorable stdout that supports elastic tabstops.
    ///
    /// Currently this only supports writing ANSI escape sequences.
    fn elastic_stdout(&self) -> Box<dyn WriteColor> {
        use {
            tabwriter::TabWriter,
            termcolor::{Ansi, NoColor},
        };

        let isatty = atty::is(atty::Stream::Stdout);
        if self.color == Some(true) || (self.color == None && isatty) {
            Box::new(Ansi::new(TabWriter::new(std::io::stdout())))
        } else {
            Box::new(NoColor::new(TabWriter::new(std::io::stdout())))
        }
    }

    /// Reads all aggregate benchmark measurements from all CSV file paths
    /// given, and returns them as one flattened vector. The filters provided
    /// are applied. If any duplicates are seen (for a given benchmark name and
    /// regex engine pair), then an error is returned.
    fn read_aggregates(&self) -> anyhow::Result<Vec<Aggregate>> {
        let mut aggregates = vec![];
        // A set of (benchmark full name, regex engine name) pairs.
        let mut seen: BTreeSet<(String, String)> = BTreeSet::new();
        for csv_path in self.csv_paths.iter() {
            let mut rdr = csv::Reader::from_path(csv_path)?;
            for result in rdr.deserialize() {
                let agg: Aggregate = result?;
                if !self.bench_filter.include(&agg.full_name) {
                    continue;
                }
                if !self.engine_filter.include(&agg.engine) {
                    continue;
                }
                let pair = (agg.full_name.clone(), agg.engine.clone());
                anyhow::ensure!(
                    !seen.contains(&pair),
                    "duplicate benchmark with name {} and regex engine {}",
                    agg.full_name,
                    agg.engine,
                );
                seen.insert(pair);
                aggregates.push(agg);
            }
        }
        Ok(aggregates)
    }
}

/// The statistic we use to compare aggregates.
#[derive(Clone, Copy, Debug)]
enum Stat {
    Median,
    Mean,
    Min,
    Max,
}

impl Stat {
    /// Get the corresponding throughput statistic for the aggregate given.
    fn get(self, agg: &Aggregate) -> Throughput {
        match self {
            Stat::Median => agg.median,
            Stat::Mean => agg.mean,
            Stat::Min => agg.min,
            Stat::Max => agg.max,
        }
    }
}

/// A grouping of all aggregates into groups where each group corresponds to a
/// single benchmark definition and every aggregate in that group corresponds
/// to a distinct regex engine. That is, the groups are rows in the output of
/// this command and the elements in each group are the columns.
#[derive(Debug)]
struct AggregatesByBenchmarkName {
    groups: Vec<AggregateGroup>,
}

impl AggregatesByBenchmarkName {
    /// Group all of the aggregate given. The 'stat' given is the metric by
    /// which aggregates are compared.
    fn new(stat: Stat, aggs: Vec<Aggregate>) -> AggregatesByBenchmarkName {
        let mut grouped = AggregatesByBenchmarkName { groups: vec![] };
        // Map from benchmark name to all aggregates with that name in 'aggs'.
        let mut name_to_aggs: BTreeMap<String, Vec<Aggregate>> =
            BTreeMap::new();
        for agg in aggs {
            name_to_aggs
                .entry(agg.full_name.clone())
                .or_insert(vec![])
                .push(agg);
        }
        for (_, aggs) in name_to_aggs {
            grouped.groups.push(AggregateGroup::new(stat, aggs));
        }
        grouped
    }

    /// Returns a lexicographically sorted list of all regex engine names in
    /// this collection of aggregates. The order is ascending.
    fn engine_names(&self) -> Vec<String> {
        let mut engine_names = BTreeSet::new();
        for group in self.groups.iter() {
            for agg in group.aggs_by_engine.values() {
                engine_names.insert(agg.engine.clone());
            }
        }
        engine_names.into_iter().collect()
    }
}

/// A group of aggregates for a single benchmark name. Every aggregate in this
/// group represents a distinct regex engine for the same benchmark definition.
#[derive(Debug)]
struct AggregateGroup {
    /// The benchmark definition's "full name," corresponding to all aggregates
    /// in this group. This is mostly just an easy convenience for accessing
    /// the name without having to dig through the map.
    full_name: String,
    /// A map from the benchmark's regex engine to the aggregate statistics.
    /// Every aggregate in this map must have the same benchmark 'full_name'.
    aggs_by_engine: BTreeMap<String, Aggregate>,
    /// The name of the 'best' aggregate in the map above according to the
    /// statistic given to AggregateGroup::new.
    best_engine_name: String,
}

impl AggregateGroup {
    /// Create a new group of aggregates for a single benchmark name. Every
    /// aggregate given must have the same 'full_name'. Each aggregate is
    /// expected to be a measurement for a distinct regex engine.
    fn new(stat: Stat, aggs: Vec<Aggregate>) -> AggregateGroup {
        let mut aggs_by_engine = BTreeMap::new();
        let (full_name, mut best_engine_name, mut best_stat) = {
            let agg = &aggs[0];
            (agg.full_name.clone(), agg.engine.clone(), stat.get(agg))
        };
        for agg in aggs {
            assert_eq!(
                full_name, agg.full_name,
                "expected all aggregates to have name {}, but also found {}",
                full_name, agg.full_name,
            );
            if stat.get(&agg) > best_stat {
                best_engine_name = agg.engine.clone();
                best_stat = stat.get(&agg);
            }
            assert!(
                !aggs_by_engine.contains_key(&agg.engine),
                "duplicate regex engine {} for benchmark {}",
                agg.engine,
                agg.full_name,
            );
            aggs_by_engine.insert(agg.engine.clone(), agg);
        }
        AggregateGroup { full_name, aggs_by_engine, best_engine_name }
    }

    /// Return the biggest difference, percentage wise, between aggregates
    /// in this group. The comparison statistic given is used.
    fn biggest_difference(&self, stat: Stat) -> f64 {
        if self.aggs_by_engine.len() < 2 {
            return 0.0;
        }
        let best = stat.get(&self.aggs_by_engine[&self.best_engine_name]);
        let mut worst = best;
        for agg in self.aggs_by_engine.values() {
            let candidate = stat.get(agg);
            if candidate < worst {
                worst = candidate;
            }
        }
        let best_bps = best.bytes_per_second();
        let worst_bps = worst.bytes_per_second();
        ((best_bps - worst_bps) / best_bps) * 100.0
    }
}

/// Write the given divider character `width` times to the given writer.
fn write_divider<W: WriteColor>(
    mut wtr: W,
    divider: char,
    width: usize,
) -> anyhow::Result<()> {
    let div: String = std::iter::repeat(divider).take(width).collect();
    write!(wtr, "{}", div)?;
    Ok(())
}
