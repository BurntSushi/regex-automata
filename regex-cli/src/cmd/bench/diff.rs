use std::{
    collections::btree_map::{BTreeMap, Entry},
    path::{Path, PathBuf},
};

use crate::{
    app::{self, App, Args},
    cmd::bench::{
        write_divider, Aggregate, FilterByBenchmarkName, FilterByEngineName,
        Stat, Threshold, Units,
    },
    config::Color,
    util::{Filter, ShortHumanDuration},
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
    app = FilterByBenchmarkName::define(app);
    app = FilterByEngineName::define(app);
    app = Threshold::define(app);
    app = Stat::define(app);
    app = Units::define(app);
    app = Color::define(app);

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
    app
}

pub fn run(args: &Args) -> anyhow::Result<()> {
    let diffargs = DiffArgs::new(args)?;
    let data_names = diffargs.csv_data_names()?;
    let grouped_aggs = diffargs.read_aggregate_groups()?;

    let mut wtr = diffargs.color.elastic_stdout();

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
        let diff = group.biggest_difference(diffargs.stat);
        if !diffargs.threshold.include(diff) {
            continue;
        }
        write!(wtr, "{}", group.full_name)?;
        write!(wtr, "\t{}", group.engine)?;
        // We write an entry for every data set given, even if this benchmark
        // doesn't appear in every data set. This makes sure everything stays
        // aligned. If an output has too many missing entries, the user can use
        // filters to condense things.
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
                    match diffargs.units {
                        Units::Time => {
                            let d = agg.duration(diffargs.stat);
                            write!(wtr, "{}", ShortHumanDuration::from(d))?;
                        }
                        Units::Throughput => {
                            write!(wtr, "{}", agg.throughput(diffargs.stat))?;
                        }
                    }
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
    /// The statistical units we want to use in our comparisons.
    units: Units,
    /// Defaults to 0, and is a percent. When the biggest difference in a row
    /// is less than this threshold, then we skip writing that row.
    threshold: Threshold,
    /// The user's color choice. We default to 'Auto'.
    color: Color,
}

impl DiffArgs {
    /// Parse 'diff' args from the given CLI args.
    fn new(args: &Args) -> anyhow::Result<DiffArgs> {
        let diffargs = DiffArgs {
            csv_paths: args
                .values_of_os("csv-path")
                .into_iter()
                .flatten()
                .map(PathBuf::from)
                .collect(),
            bench_filter: FilterByBenchmarkName::get(args)?,
            engine_filter: FilterByEngineName::get(args)?,
            stat: Stat::get(args)?,
            units: Units::get(args)?,
            threshold: Threshold::get(args)?,
            color: Color::get(args)?,
        };
        anyhow::ensure!(
            !diffargs.csv_paths.is_empty(),
            "no CSV file paths given"
        );
        Ok(diffargs)
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
                if let Some(ref err) = agg.err {
                    eprintln!(
                        "{}:{}: skipping because of error: {}",
                        agg.full_name, agg.engine, err
                    );
                    continue;
                }
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
        let best = self.aggs_by_data[self.best(stat)].throughput(stat);
        let worst = self.aggs_by_data[self.worst(stat)].throughput(stat);
        let best_bps = best.bytes_per_second();
        let worst_bps = worst.bytes_per_second();
        ((best_bps - worst_bps) / best_bps) * 100.0
    }

    /// Return the data name of the best measurement in this group. The name
    /// returned is guaranteed to exist in this group.
    fn best(&self, stat: Stat) -> &str {
        let mut it = self.aggs_by_data.iter();
        let mut best_data_name = it.next().unwrap().0;
        for (data_name, candidate) in self.aggs_by_data.iter() {
            let best = &self.aggs_by_data[best_data_name];
            if candidate.throughput(stat) > best.throughput(stat) {
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
        for (data_name, candidate) in self.aggs_by_data.iter() {
            let worst = &self.aggs_by_data[worst_data_name];
            if candidate.throughput(stat) < worst.throughput(stat) {
                worst_data_name = data_name;
            }
        }
        worst_data_name
    }
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
