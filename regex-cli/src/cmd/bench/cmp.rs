use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
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

Multiple CSV data sets may be given to this command. However, if after applying
the filters given, there are any duplicate benchmark/engine pairs, then the
command will error.
";
        app = app.arg(
            app::arg("csv-path").multiple(true).help(SHORT).long_help(LONG),
        );
    }
    app
}

pub fn run(args: &Args) -> anyhow::Result<()> {
    let cmpargs = CmpArgs::new(args)?;
    let measurements = cmpargs.read_measurements()?;
    let measurements_by_name = MeasurementsByBenchmarkName::new(measurements);
    let engines = measurements_by_name.engine_names();

    let mut wtr = cmpargs.color.elastic_stdout();

    // Write column names.
    write!(wtr, "benchmark")?;
    for engine in engines.iter() {
        write!(wtr, "\t{}", engine)?;
    }
    writeln!(wtr, "")?;

    // Write underlines beneath each column name to give some separation.
    write_divider(&mut wtr, '-', "benchmark".width())?;
    for engine in engines.iter() {
        write!(wtr, "\t")?;
        write_divider(&mut wtr, '-', engine.width())?;
    }
    writeln!(wtr, "")?;

    for group in measurements_by_name.groups.iter() {
        let diff = group.biggest_difference(cmpargs.stat);
        if !cmpargs.threshold.include(diff) {
            continue;
        }
        write!(wtr, "{}", group.full_name)?;
        // We write an entry for every engine we care about, even if the engine
        // isn't in this group. This makes sure everything stays aligned. If
        // an output has too many missing entries, the user can use filters to
        // condense things.
        let has_throughput = group.any_throughput();
        for engine in engines.iter() {
            write!(wtr, "\t")?;
            match group.measurements_by_engine.get(engine) {
                None => {
                    write!(wtr, "-")?;
                }
                Some(m) => {
                    if engine == group.best(cmpargs.stat) {
                        let mut spec = termcolor::ColorSpec::new();
                        spec.set_fg(Some(termcolor::Color::Green))
                            .set_bold(true);
                        wtr.set_color(&spec)?;
                    }
                    match cmpargs.units {
                        Units::Throughput if has_throughput => {
                            if let Some(tput) = m.throughput(cmpargs.stat) {
                                write!(wtr, "{}", tput)?;
                            } else {
                                write!(wtr, "NO-THROUGHPUT")?;
                            }
                        }
                        _ => {
                            let d = m.duration(cmpargs.stat);
                            write!(wtr, "{}", ShortHumanDuration::from(d))?;
                        }
                    }
                    if engine == group.best(cmpargs.stat) {
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

/// The arguments for this 'cmp' command parsed from CLI args.
#[derive(Debug)]
struct CmpArgs {
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

impl CmpArgs {
    /// Parse 'cmp' args from the given CLI args.
    fn new(args: &Args) -> anyhow::Result<CmpArgs> {
        let cmpargs = CmpArgs {
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
            !cmpargs.csv_paths.is_empty(),
            "no CSV file paths given"
        );
        Ok(cmpargs)
    }

    /// Reads all aggregate benchmark measurements from all CSV file paths
    /// given, and returns them as one flattened vector. The filters provided
    /// are applied. If any duplicates are seen (for a given benchmark name and
    /// regex engine pair), then an error is returned.
    fn read_measurements(&self) -> anyhow::Result<Vec<Measurement>> {
        let mut measurements = vec![];
        // A set of (benchmark full name, regex engine name) pairs.
        let mut seen: BTreeSet<(String, String)> = BTreeSet::new();
        for csv_path in self.csv_paths.iter() {
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
                anyhow::ensure!(
                    !seen.contains(&pair),
                    "duplicate benchmark with name {} and regex engine {}",
                    m.full_name,
                    m.engine,
                );
                seen.insert(pair);
                measurements.push(m);
            }
        }
        Ok(measurements)
    }
}

/// A grouping of all measurements into groups where each group corresponds
/// to a single benchmark definition and every measurement in that group
/// corresponds to a distinct regex engine. That is, the groups are rows in the
/// output of this command and the elements in each group are the columns.
#[derive(Debug)]
struct MeasurementsByBenchmarkName {
    groups: Vec<MeasurementGroup>,
}

impl MeasurementsByBenchmarkName {
    /// Group all of the aggregate given.
    fn new(measurements: Vec<Measurement>) -> MeasurementsByBenchmarkName {
        let mut grouped = MeasurementsByBenchmarkName { groups: vec![] };
        // Map from benchmark name to all aggregates with that name in 'aggs'.
        let mut name_to_measurements: BTreeMap<String, Vec<Measurement>> =
            BTreeMap::new();
        for m in measurements {
            name_to_measurements
                .entry(m.full_name.clone())
                .or_insert(vec![])
                .push(m);
        }
        for (_, measurements) in name_to_measurements {
            grouped.groups.push(MeasurementGroup::new(measurements));
        }
        grouped
    }

    /// Returns a lexicographically sorted list of all regex engine names in
    /// this collection of aggregates. The order is ascending.
    fn engine_names(&self) -> Vec<String> {
        let mut engine_names = BTreeSet::new();
        for group in self.groups.iter() {
            for agg in group.measurements_by_engine.values() {
                engine_names.insert(agg.engine.clone());
            }
        }
        engine_names.into_iter().collect()
    }
}

/// A group of aggregates for a single benchmark name. Every aggregate in this
/// group represents a distinct regex engine for the same benchmark definition.
#[derive(Debug)]
struct MeasurementGroup {
    /// The benchmark definition's "full name," corresponding to all aggregates
    /// in this group. This is mostly just an easy convenience for accessing
    /// the name without having to dig through the map.
    full_name: String,
    /// A map from the benchmark's regex engine to the aggregate statistics.
    /// Every aggregate in this map must have the same benchmark 'full_name'.
    measurements_by_engine: BTreeMap<String, Measurement>,
}

impl MeasurementGroup {
    /// Create a new group of aggregates for a single benchmark name. Every
    /// aggregate given must have the same 'full_name'. Each aggregate is
    /// expected to be a measurement for a distinct regex engine.
    fn new(measurements: Vec<Measurement>) -> MeasurementGroup {
        let mut measurements_by_engine = BTreeMap::new();
        let full_name = measurements[0].full_name.clone();
        for m in measurements {
            assert_eq!(
                full_name, m.full_name,
                "expected all aggregates to have name {}, but also found {}",
                full_name, m.full_name,
            );
            assert!(
                !measurements_by_engine.contains_key(&m.engine),
                "duplicate regex engine {} for benchmark {}",
                m.engine,
                m.full_name,
            );
            measurements_by_engine.insert(m.engine.clone(), m);
        }
        MeasurementGroup { full_name, measurements_by_engine }
    }

    /// Return the biggest difference, percentage wise, between aggregates
    /// in this group. The comparison statistic given is used. If this group
    /// is a singleton, then 0 is returned. (Which makes sense. There is no
    /// difference at all, so specifying any non-zero threshold should exclude
    /// it.)
    fn biggest_difference(&self, stat: Stat) -> f64 {
        if self.measurements_by_engine.len() < 2 {
            // I believe this is a redundant base case.
            return 0.0;
        }
        let best = self.measurements_by_engine[self.best(stat)]
            .duration(stat)
            .as_secs_f64();
        let worst = self.measurements_by_engine[self.worst(stat)]
            .duration(stat)
            .as_secs_f64();
        ((best - worst).abs() / best) * 100.0
    }

    /// Return the engine name of the best measurement in this group. The name
    /// returned is guaranteed to exist in this group.
    fn best(&self, stat: Stat) -> &str {
        let mut it = self.measurements_by_engine.iter();
        let mut best_engine = it.next().unwrap().0;
        for (engine, candidate) in self.measurements_by_engine.iter() {
            let best = &self.measurements_by_engine[best_engine];
            if candidate.duration(stat) < best.duration(stat) {
                best_engine = engine;
            }
        }
        best_engine
    }

    /// Return the engine name of the worst measurement in this group. The name
    /// returned is guaranteed to exist in this group.
    fn worst(&self, stat: Stat) -> &str {
        let mut it = self.measurements_by_engine.iter();
        let mut worst_engine = it.next().unwrap().0;
        for (engine, candidate) in self.measurements_by_engine.iter() {
            let worst = &self.measurements_by_engine[worst_engine];
            if candidate.duration(stat) > worst.duration(stat) {
                worst_engine = engine;
            }
        }
        worst_engine
    }

    /// Returns true if and only if at least one measurement in this group
    /// has throughputs available.
    fn any_throughput(&self) -> bool {
        self.measurements_by_engine
            .values()
            .any(|m| m.aggregate.tputs.is_some())
    }
}
