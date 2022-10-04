use std::{
    collections::btree_map::{BTreeMap, Entry},
    path::{Path, PathBuf},
};

use unicode_width::UnicodeWidthStr;

use crate::{
    app::{self, App, Args},
    cmd::bench::{
        write_divider, FilterByBenchmarkName, FilterByEngineName, Measurement,
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
    let grouped_aggs = diffargs.read_measurement_groups()?;

    let mut wtr = diffargs.color.elastic_stdout();

    // Write column names.
    write!(wtr, "benchmark")?;
    write!(wtr, "\tengine")?;
    for data_name in data_names.iter() {
        write!(wtr, "\t{}", data_name)?;
    }
    writeln!(wtr, "")?;

    // Write underlines beneath each column name to give some separation.
    write_divider(&mut wtr, '-', "benchmark".width())?;
    write!(wtr, "\t")?;
    write_divider(&mut wtr, '-', "engine".width())?;
    for data_name in data_names.iter() {
        write!(wtr, "\t")?;
        write_divider(&mut wtr, '-', data_name.width())?;
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
        let has_throughput = group.any_throughput();
        for data_name in data_names.iter() {
            write!(wtr, "\t")?;
            match group.measurements_by_data.get(data_name) {
                None => {
                    write!(wtr, "-")?;
                }
                Some(m) => {
                    if best == data_name {
                        let mut spec = termcolor::ColorSpec::new();
                        spec.set_fg(Some(termcolor::Color::Green))
                            .set_bold(true);
                        wtr.set_color(&spec)?;
                    }
                    match diffargs.units {
                        Units::Throughput if has_throughput => {
                            if let Some(tput) = m.throughput(diffargs.stat) {
                                write!(wtr, "{}", tput)?;
                            } else {
                                write!(wtr, "NO-THROUGHPUT")?;
                            }
                        }
                        _ => {
                            let d = m.duration(diffargs.stat);
                            write!(wtr, "{}", ShortHumanDuration::from(d))?;
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
    fn read_measurement_groups(
        &self,
    ) -> anyhow::Result<Vec<MeasurementGroup>> {
        // Our groups are just maps from CSV data name to measurements.
        let mut groups: Vec<BTreeMap<String, Measurement>> = vec![];
        // Map from (benchmark, engine) pair to index in 'groups'. We use the
        // index to find which group to insert each measurement into.
        let mut pair2idx: BTreeMap<(String, String), usize> = BTreeMap::new();
        for csv_path in self.csv_paths.iter() {
            let data_name = csv_data_name(csv_path)?;
            let mut rdr = csv::Reader::from_path(csv_path)?;
            for result in rdr.deserialize() {
                let m: Measurement = result?;
                if let Some(ref err) = m.err {
                    eprintln!(
                        "{}:{}: skipping because of error: {}",
                        m.full_name, m.engine, err
                    );
                    continue;
                }
                if !self.bench_filter.include(&m.full_name) {
                    continue;
                }
                if !self.engine_filter.include(&m.engine) {
                    continue;
                }
                let pair = (m.full_name.clone(), m.engine.clone());
                let idx = match pair2idx.entry(pair) {
                    Entry::Occupied(e) => *e.get(),
                    Entry::Vacant(e) => {
                        let idx = groups.len();
                        groups.push(BTreeMap::new());
                        *e.insert(idx)
                    }
                };
                groups[idx].insert(data_name.clone(), m);
            }
        }
        Ok(groups.into_iter().map(MeasurementGroup::new).collect())
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

/// A group of measurements for a single pair of (benchmark name, engine name).
/// Every measurement in this group represents an aggregate group of statistic
/// from a given CSV input.
#[derive(Debug)]
struct MeasurementGroup {
    /// The benchmark definition's "full name," corresponding to all
    /// measurements in this group. This is mostly just an easy convenience for
    /// accessing the name without having to dig through the map.
    full_name: String,
    /// Similarly to 'full_name', this is the regex engine corresponding to all
    /// measurements in this group.
    engine: String,
    /// A map from the data set name to the measurement. Every measurement in
    /// this map must have the same benchmark 'full_name' and 'engine' name.
    measurements_by_data: BTreeMap<String, Measurement>,
}

impl MeasurementGroup {
    /// Create a new group of aggregates for a single (benchmark name, engine
    /// name) pair. Every aggregate given must have the same 'full_name'
    /// and 'engine'. Each aggregate is expected to be a measurement from a
    /// distinct CSV input, where the name of the CSV input is the key in the
    /// map given.
    fn new(
        measurements_by_data: BTreeMap<String, Measurement>,
    ) -> MeasurementGroup {
        let mut it = measurements_by_data.values();
        let (full_name, engine) = {
            let m = it.next().expect("at least one measurement");
            (m.full_name.clone(), m.engine.clone())
        };
        for m in it {
            assert_eq!(
                full_name, m.full_name,
                "expected all measurements to have name {}, \
                 but also found {}",
                full_name, m.full_name,
            );
            assert_eq!(
                engine, m.engine,
                "expected all measurements to have engine {}, \
                 but also found {}",
                engine, m.engine,
            );
        }
        MeasurementGroup { full_name, engine, measurements_by_data }
    }

    /// Return the biggest difference, percentage wise, between aggregates
    /// in this group. The comparison statistic given is used. If this group
    /// is a singleton, then 0 is returned. (Which makes sense. There is no
    /// difference at all, so specifying any non-zero threshold should exclude
    /// it.)
    fn biggest_difference(&self, stat: Stat) -> f64 {
        if self.measurements_by_data.len() < 2 {
            // I believe this is a redundant base case.
            return 0.0;
        }
        let best = self.measurements_by_data[self.best(stat)]
            .duration(stat)
            .as_secs_f64();
        let worst = self.measurements_by_data[self.worst(stat)]
            .duration(stat)
            .as_secs_f64();
        ((best - worst).abs() / best) * 100.0
    }

    /// Return the data name of the best measurement in this group. The name
    /// returned is guaranteed to exist in this group.
    fn best(&self, stat: Stat) -> &str {
        let mut it = self.measurements_by_data.iter();
        let mut best_data_name = it.next().unwrap().0;
        for (data_name, candidate) in self.measurements_by_data.iter() {
            let best = &self.measurements_by_data[best_data_name];
            if candidate.duration(stat) < best.duration(stat) {
                best_data_name = data_name;
            }
        }
        best_data_name
    }

    /// Return the data name of the worst measurement in this group. The name
    /// returned is guaranteed to exist in this group.
    fn worst(&self, stat: Stat) -> &str {
        let mut it = self.measurements_by_data.iter();
        let mut worst_data_name = it.next().unwrap().0;
        for (data_name, candidate) in self.measurements_by_data.iter() {
            let worst = &self.measurements_by_data[worst_data_name];
            if candidate.duration(stat) > worst.duration(stat) {
                worst_data_name = data_name;
            }
        }
        worst_data_name
    }

    /// Returns true if and only if at least one measurement in this group
    /// has throughputs available.
    fn any_throughput(&self) -> bool {
        self.measurements_by_data.values().any(|m| m.aggregate.tputs.is_some())
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
