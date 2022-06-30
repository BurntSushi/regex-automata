use std::{
    collections::btree_map::{BTreeMap, Entry},
    path::{Path, PathBuf},
};

use {anyhow::Context, termcolor::WriteColor};

use crate::{
    app::{self, App, Args},
    cmd::bench::Aggregate,
    util::{Filter, Throughput},
};

const ABOUT_SHORT: &'static str = "\
Compare benchmarks across time.
";

const ABOUT_LONG: &'static str = "\
Compare benchmarks across time.

To compare benchmarks between regex engines, use the 'regex-cli bench cmp'
command.
";

pub fn define() -> App {
    let mut app =
        app::command("diff").about(ABOUT_SHORT).before_help(ABOUT_LONG);
    {
        const SHORT: &str =
            "File paths to CSV data containing benchmark measurements.";
        const LONG: &str = "\
File paths to CSV data containing benchmark measurements.

In general, the CSV data read by this tool is the same CSV data written by the
'regex-cli bench measure' command.

Multiple CSV data sets may be given to this command. Each data set corresponds
to a column in the output.
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

When enabled, color is used to indicate which measurement did the best on each
benchmark. The choices are: auto, always, never.
";
        app = app.arg(app::mflag("color").help(SHORT).long_help(LONG));
    }
    app
}

pub fn run(args: &Args) -> anyhow::Result<()> {
    let diffargs = DiffArgs::new(args)?;
    let data_names = diffargs.csv_data_names()?;
    let grouped_aggs = diffargs.read_aggregate_groups()?;

    let mut wtr = diffargs.elastic_stdout();

    // Write column names.
    write!(wtr, "benchmark")?;
    write!(wtr, "\tengine")?;
    for data_name in data_names.iter() {
        write!(wtr, "\t{}", data_name)?;
    }
    writeln!(wtr, "")?;

    // Write underlines beneath each column name to give some separation. Note
    // that we use byte length, which is a little suspect, because file names
    // might have Unicode in them.
    write_divider(&mut wtr, '-', "benchmark".len())?;
    write!(wtr, "\t")?;
    write_divider(&mut wtr, '-', "engine".len())?;
    for data_name in data_names.iter() {
        write!(wtr, "\t")?;
        write_divider(&mut wtr, '-', data_name.len())?;
    }
    writeln!(wtr, "")?;

    for group in grouped_aggs.iter() {
        if group.biggest_difference(diffargs.stat) < diffargs.threshold {
            continue;
        }
        write!(wtr, "{}", group.full_name)?;
        write!(wtr, "\t{}", group.engine)?;
        // We write an entry for every engine we care about, even if the engine
        // isn't in this group. This makes sure everything stays aligned. If
        // an output has too many missing entries, the user can use filters to
        // condense things.
        let best = group.best(diffargs.stat);
        for data_name in data_names.iter() {
            write!(wtr, "\t")?;
            match group.aggs_by_data.get(data_name) {
                None => {
                    write!(wtr, "-")?;
                }
                Some(agg) => {
                    if best == data_name {
                        let mut spec = termcolor::ColorSpec::new();
                        spec.set_fg(Some(termcolor::Color::Green))
                            .set_bold(true);
                        wtr.set_color(&spec)?;
                    }
                    write!(wtr, "{}", diffargs.stat.get(&agg))?;
                    if best == data_name {
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

/// The arguments for this 'diff' command parsed from CLI args.
#[derive(Debug)]
struct DiffArgs {
    /// File paths to CSV files.
    csv_paths: Vec<PathBuf>,
    /// A filter to be applied to benchmark "full names."
    bench_filter: Filter,
    /// A filter to be applied to regex engine names.
    engine_filter: Filter,
    /// The statistic we want to compare.
    stat: Stat,
    /// Defaults to 0, and is a percent. When the biggest difference in a row
    /// is less than this threshold, then we skip writing that row.
    threshold: f64,
    /// 'none' means 'auto', i.e., we only write colors when stdout is a tty.
    color: Option<bool>,
}

impl DiffArgs {
    /// Parse 'diff' args from the given CLI args.
    fn new(args: &Args) -> anyhow::Result<DiffArgs> {
        let mut diffargs = DiffArgs {
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
            !diffargs.csv_paths.is_empty(),
            "no CSV file paths given"
        );
        if let Some(rules) = args.values_of_os("filter") {
            diffargs.bench_filter =
                Filter::new(rules).context("-f/--filter")?;
        }
        if let Some(rules) = args.values_of_os("engine") {
            diffargs.engine_filter =
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
            diffargs.threshold = f64::from(percent);
        }
        if let Some(statname) = args.value_of_lossy("statistic") {
            diffargs.stat = match &*statname {
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
            diffargs.color = match &*colorchoice {
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
        Ok(diffargs)
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
    /// given, and returns them grouped by the data set. That is, each group
    /// represents all measurements found across the data sets given for a
    /// single (benchmark name, engine name) pair. The filters provided are
    /// applied.
    fn read_aggregate_groups(&self) -> anyhow::Result<Vec<AggregateGroup>> {
        // Our groups are just maps from CSV data name to measurements.
        let mut groups: Vec<BTreeMap<String, Aggregate>> = vec![];
        // Map from (benchmark, engine) pair to index in 'groups'. We use the
        // index to find which group to insert each Aggregate into.
        let mut pair2idx: BTreeMap<(String, String), usize> = BTreeMap::new();
        for csv_path in self.csv_paths.iter() {
            let data_name = csv_data_name(csv_path)?;
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
                let idx = match pair2idx.entry(pair) {
                    Entry::Occupied(e) => *e.get(),
                    Entry::Vacant(e) => {
                        let idx = groups.len();
                        groups.push(BTreeMap::new());
                        *e.insert(idx)
                    }
                };
                groups[idx].insert(data_name.clone(), agg);
            }
        }
        Ok(groups.into_iter().map(AggregateGroup::new).collect())
    }

    /// Returns the "nice" CSV data names from the paths given. The names are
    /// just the stems of the file names from each of the paths. The vector
    /// returned contains the names in the same order as given on the CLI.
    /// Duplicates are not removed. If there was a problem exteacting the file
    /// stem from any path, then an error is returned.
    fn csv_data_names(&self) -> anyhow::Result<Vec<String>> {
        self.csv_paths.iter().map(csv_data_name).collect()
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

/// A group of aggregates for a single pair of (benchmark name, engine name).
/// Every aggregate in this group represents a measurement from a given CSV
/// input.
#[derive(Debug)]
struct AggregateGroup {
    /// The benchmark definition's "full name," corresponding to all aggregates
    /// in this group. This is mostly just an easy convenience for accessing
    /// the name without having to dig through the map.
    full_name: String,
    /// Similarly to 'full_name', this is the regex engine corresponding to all
    /// aggregates in this group.
    engine: String,
    /// A map from the data set name to the aggregate statistics. Every
    /// aggregate in this map must have the same benchmark 'full_name' and
    /// 'engine' name.
    aggs_by_data: BTreeMap<String, Aggregate>,
}

impl AggregateGroup {
    /// Create a new group of aggregates for a single (benchmark name, engine
    /// name) pair. Every aggregate given must have the same 'full_name'
    /// and 'engine'. Each aggregate is expected to be a measurement from a
    /// distinct CSV input, where the name of the CSV input is the key in the
    /// map given.
    fn new(aggs_by_data: BTreeMap<String, Aggregate>) -> AggregateGroup {
        let mut it = aggs_by_data.values();
        let (full_name, engine) = {
            let agg = it.next().expect("at least one aggregate");
            (agg.full_name.clone(), agg.engine.clone())
        };
        for agg in it {
            assert_eq!(
                full_name, agg.full_name,
                "expected all aggregates to have name {}, but also found {}",
                full_name, agg.full_name,
            );
            assert_eq!(
                engine, agg.engine,
                "expected all aggregates to have engine {}, but also found {}",
                engine, agg.engine,
            );
        }
        AggregateGroup { full_name, engine, aggs_by_data }
    }

    /// Return the biggest difference, percentage wise, between aggregates
    /// in this group. The comparison statistic given is used. If this group
    /// is a singleton, then 0 is returned. (Which makes sense. There is no
    /// difference at all, so specifying any non-zero threshold should exclude
    /// it.)
    fn biggest_difference(&self, stat: Stat) -> f64 {
        if self.aggs_by_data.len() < 2 {
            // I believe this is a redundant base case.
            return 0.0;
        }
        let best = stat.get(&self.aggs_by_data[self.best(stat)]);
        let worst = stat.get(&self.aggs_by_data[self.worst(stat)]);
        let best_bps = best.bytes_per_second();
        let worst_bps = worst.bytes_per_second();
        ((best_bps - worst_bps) / best_bps) * 100.0
    }

    /// Return the data name of the best measurement in this group. The name
    /// returned is guaranteed to exist in this group.
    fn best(&self, stat: Stat) -> &str {
        let mut it = self.aggs_by_data.iter();
        let mut best_data_name = it.next().unwrap().0;
        for (data_name, agg) in self.aggs_by_data.iter() {
            if stat.get(agg) > stat.get(&self.aggs_by_data[best_data_name]) {
                best_data_name = data_name;
            }
        }
        best_data_name
    }

    /// Return the data name of the worst measurement in this group. The name
    /// returned is guaranteed to exist in this group.
    fn worst(&self, stat: Stat) -> &str {
        let mut it = self.aggs_by_data.iter();
        let mut worst_data_name = it.next().unwrap().0;
        for (data_name, agg) in self.aggs_by_data.iter() {
            if stat.get(agg) < stat.get(&self.aggs_by_data[worst_data_name]) {
                worst_data_name = data_name;
            }
        }
        worst_data_name
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

/// Extract a "data set" name from a given CSV file path.
///
/// If there was a problem getting the name (i.e., the file path is "weird" in
/// some way), then an error is returned.
fn csv_data_name<P: AsRef<Path>>(path: P) -> anyhow::Result<String> {
    let path = path.as_ref();
    let stem = match path.file_stem() {
        Some(stem) => stem,
        None => anyhow::bail!("{}: could not get file stem", path.display()),
    };
    match stem.to_str() {
        Some(name) => Ok(name.to_string()),
        None => anyhow::bail!(
            "{}: path's file name is not valid UTF-8",
            path.display()
        ),
    }
}
