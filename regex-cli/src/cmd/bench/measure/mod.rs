use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use {anyhow::Context, once_cell::sync::Lazy, regex::Regex};

use crate::{
    app::{self, App, Args},
    cmd::bench::AggregateDuration,
    util::{Filter, ShortHumanDuration},
};

mod compile;
mod count;
mod count_captures;
mod grep;
mod new;
mod regexredux;

const ABOUT_SHORT: &'static str = "\
Run benchmarks and write measurements.
";

const ABOUT_LONG: &'static str = "\
Run benchmarks and write measurements.
";

pub fn define() -> App {
    let mut app =
        app::command("measure").about(ABOUT_SHORT).before_help(ABOUT_LONG);
    {
        const SHORT: &str =
            "The directory containing benchmarks and haystacks.";
        const LONG: &str = "\
The directory containing benchmarks and haystacks.

This flag specifies the directory that contains both the benchmark definitions
and the haystacks. The benchmark definitions must be in files with a '.toml'
extension. All haystacks should be in '{dir}/haystacks/' and have a '.txt'
extension. Both benchmark definitions and haystacks may be in sub-directories.

The default for this value is 'benchmarks'.
";
        app = app.arg(app::flag("dir").short("d").help(SHORT).long_help(LONG));
    }
    {
        const SHORT: &str = "Filter (using regex) which benchmarks to run.";
        const LONG: &str = "\
Filter (using regex) which benchmarks to run.

This flag may be given multiple times. The value can either be a whitelist
regex or a blacklist regex. To make it a blacklist regex, start it with a
'~'. If there is at least one whitelist regex, then a benchmark must match at
least one of them in order to run. If there are no whitelist regexes, then
a benchmark is only run when it does not match any blacklist regexes. The
last filter regex that matches (whether it be a whitelist or a blacklist) is
what takes precedence. So for example, a whitelist regex that matches after a
blacklist regex matches, that would result in that benchmark being run.

So for example, consider the benchmarks 'foo', 'bar', 'baz' and 'quux'.

* '-f foo' will run 'foo'.
* '-f ~foo' will run 'bar', 'baz' and 'quux'.
* '-f . -f ~ba -f bar' will run 'foo', 'bar' and 'quux'.

Filter regexes are matched on the full name of the benchmark, which takes the
form '{type}/{group}/{name}'.
";
        app = app
            .arg(app::mflag("filter").short("f").help(SHORT).long_help(LONG));
    }
    {
        const SHORT: &str = "Filter (using regex) which regex engines to run.";
        const LONG: &str = "\
Filter (using regex) which regex engines to run.

This is just like the -f/--filter flag (with the same whitelist/blacklist
rules), except it applies to which regex engines to benchmark. For example,
many benchmarks list a number of regex engines that it should run with, but
this filter permits specifying a smaller set of regex engines to benchmark.

To be clear, a benchmark can only run with regex engines it has been configured
with in a TOML file. This flag cannot add regex engines to existing benchmarks.
This flag can only select a subset of regex engines for each benchmark.

This filter is applied to every benchmark. It is useful, for example, if you
only want to run benchmarks for one of the regex engines instead of all of
them, which might take considerably longer.
";
        app = app
            .arg(app::mflag("engine").short("e").help(SHORT).long_help(LONG));
    }
    {
        const SHORT: &str = "The max number of benchmark iterations to run.";
        const LONG: &str = "\
The maximum number of iterations to run for each benchmark.

One of the difficulties of a benchmark harness is determining just how long to
run a benchmark for. We want to run it long enough that we get a decent sample,
but not too long that we are waiting forever for results. That is, there is a
point of diminishing returns.

This flag permits controlling the maximum number of iterations that a benchmark
will be executed for. In general, one should not need to change this, as it
would be better to tweak --bench-time instead. However, it is exposed in case
it's useful, and in particular, you might want to increase it in certain
circumstances for an usually fast routine.
";
        app = app.arg(app::flag("max-iters").help(SHORT).long_help(LONG));
    }
    {
        const SHORT: &str = "The max number of warmup iterations to execute.";
        const LONG: &str = "\
The max number of warmup iterations to execute.
";
        app =
            app.arg(app::flag("max-warmup-iters").help(SHORT).long_help(LONG));
    }
    {
        const SHORT: &str =
            "The approximate amount of time to run a benchmark.";
        const LONG: &str = "\
The approximate amount of time to run a benchmark.

This harness tries to balance \"benchmarks taking too long\" and \"benchmarks
need enough samples to be reliable\" by varying the number of times each
benchmark is executed. Slower search routines (for example) get executed
fewer times while faster routines get executed more. This is done by holding
invariant roughly how long one wants each benchmark to run for. This flag sets
that time.

In general, unless a benchmark is unusually fast, one should generally expect
each benchmark to take roughly this amount of time to complete.

The format for this flag is a duration specified in seconds, milliseconds,
microseconds or nanoseconds. Namely, '^[0-9]+(s|ms|us|ns)$'.
";
        app = app.arg(app::flag("bench-time").help(SHORT).long_help(LONG));
    }
    {
        const SHORT: &str =
            "The approximate amount of time to warmup a benchmark.";
        const LONG: &str = "\
The approximate amount of time to warmup a benchmark.

This is like --bench-time, but it controls the maximum amount of time to
spending \"warming\" up a benchmark. The idea of warming up a benchmark is
to execute the thing we're trying to measure for a period of time before
starting the process of collecting samples. The reason for doing this is
generally to fill up any internal caches being used to avoid extreme outliers,
and even to an extent, to give CPUs a chance to adjust their clock speeds
up. The idea here is that a \"warmed\" regex engine is more in line with real
world use cases.

As a general rule of thumb, warmup time should be one half the benchmark time.
Indeed, if this is not given, it automatically defaults to half the benchmark
time.
";
        app = app.arg(app::flag("warmup-time").help(SHORT).long_help(LONG));
    }
    {
        const SHORT: &str = "List benchmarks to run, but don't run them.";
        const LONG: &str = "\
List benchmarks to run, but don't run them.

This command does all of the work to collect benchmarks, haystacks, filter them
and validate them. But it does not actually run the benchmarks. Instead, it
prints every benchmark that will be executed. This is useful for seeing what
work will be done without actually doing it.
";
        app = app.arg(app::switch("list").help(SHORT).long_help(LONG));
    }
    {
        const SHORT: &str =
            "Verify that all selected benchmarks run successfully.";
        const LONG: &str = "\
Verify that all selected benchmarks run successfully.

This checks that all selected benchmarks can run through at least one iteration
without reporting an error or an incorrect answer. This can be useful for
quickly debugging a new benchmark or regex engine where the answers aren't
lining up.

This collects all errors reported and prints them. If no errors occurred, then
this prints nothing and exits successfully.
";
        app = app.arg(app::switch("verify").help(SHORT).long_help(LONG));
    }
    {
        const SHORT: &str = "Print extra information where possible.";
        const LONG: &str = "\
Print extra information where possible.

Where possible, this prints extra information. e.g., When using --verify, this
will print each benchmark that is being tested as it happens, as a way to see
progress.
";
        app = app.arg(app::switch("verbose").help(SHORT).long_help(LONG));
    }
    app
}

pub fn run(args: &Args) -> anyhow::Result<()> {
    // Parse everything and load what we need.
    let runner = MeasureArgs::new(args)?;
    let defs = runner
        .benchmark_definitions()?
        .filter_by_name(&runner.bench_filter)
        .filter_by_engine(&runner.engine_filter);
    let haystacks = runner.haystacks(&defs)?;

    // Collect all of the benchmarks we will run. Each benchmark definition can
    // spawn multiple benchmarks; one for each regex engine specified in the
    // definition.
    let mut benchmarks = vec![];
    for def in &defs.defs {
        for b in def.iter(&runner.bench_config, &haystacks)? {
            // While we did run the engine filter above, we run it again
            // because the filter above only excludes benchmark definitions
            // that have no matching engines at all. But we might still run
            // a subset. So why do we run it above? Well, this way, we avoid
            // loading haystacks into memory that will never be used.
            if !runner.engine_filter.include(&b.engine) {
                continue;
            }
            benchmarks.push(b);
        }
    }
    // If we just want to list which benchmarks we'll run, spit that out.
    if runner.list {
        let mut wtr = csv::Writer::from_writer(std::io::stdout());
        for b in benchmarks.iter() {
            wtr.write_record(&[b.def.full_name(), b.engine.clone()])?;
        }
        wtr.flush()?;
        return Ok(());
    }
    // Or if we just want to check that every benchmark runs correct, do that.
    // We spit out an error we find.
    if runner.verify {
        let mut errored = false;
        let mut wtr = csv::Writer::from_writer(std::io::stdout());
        for b in benchmarks.iter() {
            let agg = b.aggregate(b.verifier().collect());
            if let Some(err) = agg.err {
                errored = true;
                wtr.write_record(&[
                    b.def.full_name(),
                    b.engine.clone(),
                    err.to_string(),
                ])?;
            } else if runner.verbose {
                wtr.write_record(&[
                    b.def.full_name(),
                    b.engine.clone(),
                    "OK".to_string(),
                ])?;
            }
            wtr.flush()?;
        }
        anyhow::ensure!(!errored, "some benchmarks failed");
        return Ok(());
    }
    // Run our benchmarks and emit the results of each as a single CSV record.
    let mut wtr = csv::Writer::from_writer(std::io::stdout());
    for b in benchmarks.iter() {
        // Run the benchmark, collect the samples and turn the samples into a
        // collection of various aggregate statistics (mean+/-stddev, median,
        // min, max).
        let agg = b.aggregate(b.collect());
        // Our aggregate is initially captured in terms of how long it takes to
        // execute each iteration of the benchmark. But for searching, this is
        // not particularly intuitive. Instead, we convert strict timings into
        // throughputs, which give a much better idea of how fast something is
        // by relating it to how much can be searched in a single second.
        //
        // Literally every regex benchmark I've looked at reports measurements
        // as raw timings. Like, who the heck cares if a regex search completes
        // in 500ns? What does that mean? It's much clearer to say 500 MB/s.
        // I guess people consistently misunderstand that benchmarks are
        // fundamentally about communication first.
        //
        // Using throughputs doesn't quite make sense for the 'compile'
        // benchmarks, but '--units time' can be used with the benchmark
        // comparison commands to change units.
        wtr.serialize(agg.into_throughput())?;
        // Flush every record once we have it so that users can see that
        // progress is being made.
        wtr.flush()?;
    }
    Ok(())
}

/// The CLI arguments parsed from the 'measure' sub-command.
#[derive(Clone, Debug)]
struct MeasureArgs {
    /// The directory to find benchmark definitions and haystacks.
    dir: PathBuf,
    /// The filter to apply to benchmark "full names." That is, the name in
    /// the format of {benchmark_type}/{group}/{name}.
    bench_filter: Filter,
    /// The filter to apply to regex engine name.
    engine_filter: Filter,
    /// Various parameters to control how ever benchmark is executed.
    bench_config: BenchmarkConfig,
    /// Whether to just list the benchmarks that will be executed and
    /// then quit. This also tests that all of the benchmark data can be
    /// deserialized.
    list: bool,
    /// Whether to just verify all of the benchmarks without collecting any
    /// measurements.
    verify: bool,
    /// When enabled, print extra stuff where appropriate.
    verbose: bool,
}

impl MeasureArgs {
    /// Parse measurement args from the given CLI args.
    fn new(args: &Args) -> anyhow::Result<MeasureArgs> {
        let mut margs = MeasureArgs {
            dir: PathBuf::from("benchmarks"),
            // These unwraps are OK because an empty set of rules always works.
            bench_filter: Filter::new([].into_iter()).unwrap(),
            engine_filter: Filter::new([].into_iter()).unwrap(),
            bench_config: BenchmarkConfig::default(),
            list: args.is_present("list"),
            verify: args.is_present("verify"),
            verbose: args.is_present("verbose"),
        };
        if let Some(x) = args.value_of_os("dir") {
            margs.dir = PathBuf::from(x);
        }
        if let Some(rules) = args.values_of_os("filter") {
            margs.bench_filter = Filter::new(rules).context("-f/--filter")?;
        }
        if let Some(rules) = args.values_of_os("engine") {
            margs.engine_filter = Filter::new(rules).context("-e/--engine")?;
        }
        if let Some(x) = args.value_of_lossy("max-iters") {
            margs.bench_config.max_iters = x.parse().context("--max-iters")?;
        }
        if let Some(x) = args.value_of_lossy("max-warmup-iters") {
            margs.bench_config.max_warmup_iters =
                x.parse().context("--max-warmup-iters")?;
        }
        if let Some(x) = args.value_of_lossy("bench-time") {
            let hdur: ShortHumanDuration =
                x.parse().context("--bench-time")?;
            margs.bench_config.approx_max_benchmark_time =
                Duration::from(hdur);
        }
        if let Some(x) = args.value_of_lossy("warmup-time") {
            let hdur: ShortHumanDuration =
                x.parse().context("--warmup-time")?;
            margs.bench_config.approx_max_warmup_time = Duration::from(hdur);
        } else {
            let default = margs.bench_config.approx_max_benchmark_time / 2;
            margs.bench_config.approx_max_warmup_time = default;
        }
        Ok(margs)
    }

    /// Read and parse benchmark definitions from TOML files in the --dir
    /// directory.
    fn benchmark_definitions(&self) -> anyhow::Result<BenchmarkDefs> {
        let mut benches = BenchmarkDefs::new();
        benches.load_dir(&self.dir)?;
        Ok(benches)
    }

    /// Read all of the haystacks from disk that are referenced in the given
    /// benchmark definitions.
    fn haystacks(&self, benches: &BenchmarkDefs) -> anyhow::Result<Haystacks> {
        let dir = self.dir.join("haystacks");

        let mut haystacks = Haystacks::new();
        haystacks.load_dir(dir, &benches.haystack_paths())?;
        Ok(haystacks)
    }
}

/// The configuration for a benchmark. This is overridable via the CLI, and can
/// be useful on a case-by-case basis. In effect, it controls how benchmarks
/// are executed and generally permits explicitly configuring how long you
/// want to wait for benchmarks to run. Nobody wants to wait a long time, but
/// you kind of need to wait a little bit or else benchmark results tend to be
/// quite noisy.
#[derive(Clone, Debug)]
struct BenchmarkConfig {
    /// The maximum number of samples to collect.
    max_iters: u64,
    /// The maximum number of times to execute the benchmark before collecting
    /// samples.
    max_warmup_iters: u64,
    /// The approximate amount of time the benchmark should run. The idea here
    /// is to collect as many samples as possible, up to the max and only for
    /// as long as we are in our time budget.
    ///
    /// It'd be nice if we could just collect the same number of samples for
    /// every benchmark, but this is in practice basically impossible when your
    /// benchmarks include things that are blindingly fast like 'memmem' and
    /// things that are tortoise slow, like the Pike VM.
    approx_max_benchmark_time: Duration,
    /// Like max benchmark time, but for warmup time. As a general rule, it's
    /// usually good to have this be about half the benchmark time.
    approx_max_warmup_time: Duration,
}

impl Default for BenchmarkConfig {
    fn default() -> BenchmarkConfig {
        BenchmarkConfig {
            max_warmup_iters: 1_000_000,
            max_iters: 1_000_000,
            approx_max_benchmark_time: Duration::from_millis(3000),
            approx_max_warmup_time: Duration::from_millis(1500),
        }
    }
}

/// A collection of haystacks found on the filesystem.
///
/// Note that we only load haystacks that are pointed to by benchmarks that we
/// will run. So for example, if only a small subset of benchmarks are selected
/// to run, then only the haystacks referenced by those benchmarks (if any)
/// will be loaded into memory.
///
/// Note that haystacks loaded from files can be any sequence of bytes, even
/// invalid UTF-8. This is in contrast to haystacks defined inline in TOML
/// files, which must be valid UTF-8.
#[derive(Clone, Debug)]
struct Haystacks {
    /// A map from a haystack name (the file path relative to, e.g.,
    /// benches/haystacks) to the haystack itself. We use a Arc<[u8]> instead
    /// of a Vec<u8> so that we can reuse a single copy of the haystack
    /// everywhere without lifetimes. In the context of this tool, there is no
    /// real downside to Arc<[u8]> anyway.
    map: BTreeMap<String, Arc<[u8]>>,
}

impl Haystacks {
    /// Create a new empty set of haystacks.
    fn new() -> Haystacks {
        Haystacks { map: BTreeMap::new() }
    }

    /// Load the haystacks found in the given directory. But only load the
    /// haystacks that are in `which`. `which` should be a set of strings
    /// corresponding to file paths relative to the `{benchmark_dir}/haystacks`
    /// directory.
    fn load_dir<P: AsRef<Path>>(
        &mut self,
        dir: P,
        which: &BTreeSet<String>,
    ) -> anyhow::Result<()> {
        let dir = dir.as_ref();
        for name in which {
            self.load_file(name, dir.join(name))?;
        }
        Ok(())
    }

    /// Load a single file at the path given as a haystack. The haystack use
    /// get the name given as a key in the haystack map.
    fn load_file<P: AsRef<Path>>(
        &mut self,
        name: &str,
        path: P,
    ) -> anyhow::Result<()> {
        let path = path.as_ref();
        let data = std::fs::read(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        self.load_bytes(&name, data)
            .with_context(|| format!("error loading {}", path.display()))?;
        Ok(())
    }

    /// Load the raw bytes into the haystack map with the name given. If a
    /// haystack with the given name already exists, then an error is returned.
    fn load_bytes(&mut self, name: &str, data: Vec<u8>) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.map.contains_key(name),
            "found duplicate haystack '{}'",
            name
        );
        self.map.insert(name.to_owned(), Arc::from(data));
        Ok(())
    }

    /// Get the haystack witht he given name. If one does not exist, then an
    /// error is returned.
    fn get(&self, name: &str) -> anyhow::Result<Arc<[u8]>> {
        if let Some(ref haystack) = self.map.get(name) {
            return Ok(Arc::clone(haystack));
        }
        anyhow::bail!("could not find haystack '{}'", name)
    }
}

/// A sequence of benchmark definitions defined over possibly many TOML files.
/// Each benchmark definition typically defines one or more actual benchmarks,
/// with the number of actual benchmarks equal to the number of regex engines
/// included in the definition.
#[derive(Clone, Debug, serde::Deserialize)]
struct BenchmarkDefs {
    /// The definitions, in the order in which they're defined.
    #[serde(rename = "benches")]
    #[serde(default)] // allows empty TOML files
    defs: Vec<BenchmarkDef>,
    /// All of the benchmark definition names we've seen. We don't permit
    /// duplicates.
    #[serde(skip)]
    seen: BTreeSet<String>,
}

impl BenchmarkDefs {
    /// Create an empty set of benchmark definitions.
    fn new() -> BenchmarkDefs {
        BenchmarkDefs { defs: vec![], seen: BTreeSet::new() }
    }

    /// Load all benchmark definitions from the given directory. Any file
    /// with a 'toml' extension is read and deserialized. Also, the top-level
    /// 'haystacks' directory is skipped, as it is meant to only contain
    /// haystacks.
    fn load_dir<P: AsRef<Path>>(&mut self, dir: P) -> anyhow::Result<()> {
        let dir = dir.as_ref();
        let mut it = walkdir::WalkDir::new(dir).into_iter();
        while let Some(result) = it.next() {
            let dent = result?;
            if dent.file_name() == "haystacks" {
                it.skip_current_dir();
                continue;
            }
            if !dent.file_type().is_file() {
                continue;
            }
            let ext = match dent.path().extension() {
                None => continue,
                Some(ext) => ext,
            };
            if ext != "toml" {
                continue;
            }
            self.load_file(dent.path())?;
        }
        Ok(())
    }

    /// Load the benchmark definitions from the TOML file at the given path.
    fn load_file<P: AsRef<Path>>(&mut self, path: P) -> anyhow::Result<()> {
        let path = path.as_ref();
        let group = path
            .file_stem()
            .with_context(|| {
                format!("failed to get file name of {}", path.display())
            })?
            .to_str()
            .with_context(|| {
                format!("invalid UTF-8 found in {}", path.display())
            })?;
        let data = std::fs::read(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        self.load_slice(&group, &data)
            .with_context(|| format!("error loading {}", path.display()))?;
        Ok(())
    }

    /// Load the benchmark definitions from the TOML data. The group given is
    /// assigned to every benchmark definition. Typically the group name is the
    /// stem of the file name.
    fn load_slice(&mut self, group: &str, data: &[u8]) -> anyhow::Result<()> {
        let benches: BenchmarkDefs = toml::from_slice(&data)
            .with_context(|| format!("error decoding TOML for '{}'", group))?;
        for mut b in benches.defs {
            b.group = group.to_string();
            anyhow::ensure!(
                !self.seen.contains(&b.full_name()),
                "a benchmark with name '{}' has already been defined",
                b.full_name(),
            );
            self.seen.insert(b.full_name());
            b.validate()
                .with_context(|| format!("error loading test '{}'", b.name))?;
            self.defs.push(b);
        }
        Ok(())
    }

    /// Return a new set of benchmark definitions such that all definitions
    /// pass the given name filter.
    fn filter_by_name(&self, name_filter: &Filter) -> BenchmarkDefs {
        let mut new = BenchmarkDefs::new();
        for def in self.defs.iter() {
            if name_filter.include(&def.full_name()) {
                new.defs.push(def.clone());
            }
        }
        new
    }

    /// Return a new set of benchmark definitions such that all definitions
    /// have at least one regex engine that passes the given filter.
    fn filter_by_engine(&self, engine_filter: &Filter) -> BenchmarkDefs {
        let mut new = BenchmarkDefs::new();
        for def in self.defs.iter() {
            for engine in def.engines.iter() {
                if engine_filter.include(engine) {
                    new.defs.push(def.clone());
                    break;
                }
            }
        }
        new
    }

    /// Return all haystack paths from this set of benchmark definitions.
    ///
    /// This is useful for loading on the subset of haystacks that are
    /// actually needed to run benchmarks.
    fn haystack_paths(&self) -> BTreeSet<String> {
        let mut paths = BTreeSet::new();
        for def in &self.defs {
            if let Some(ref path) = def.haystack_path {
                paths.insert(path.clone());
            }
        }
        paths
    }
}

/// A definition of a benchmark.
///
/// Each definition usually corresponds to multiple benchmarks, one for each
/// regex engine specified.
///
/// In general, benchmark definitions specify the parameters of the benchmark.
/// Most importantly, that's the regex pattern itself, the haystack and the
/// task that needs to be performed. Some general purpose options, such as
/// Unicode mode and case insensitivity, may also be set. Benchmark definitions
/// do not encode regex engine specific options (like a JIT stack size).
/// Instead, if one wants to test multiple configurations of a particular
/// regex implementation, then each configuration should be used to produce a
/// different regex engine.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct BenchmarkDef {
    /// The type of the benchmark to run. This is effectively saying what kind
    /// of computation we want to measure. Whether it's just a simple count of
    /// all matches, or a grep-like benchmark or even just a benchmark of how
    /// long compilation of the regex itself takes.
    #[serde(rename = "type")]
    benchmark_type: BenchmarkType,
    /// This field is set internally and generally corresponds to the file
    /// name in which the benchmark is defined. It is a logical "grouping"
    /// in which the benchmark is a part of. The logical grouping is purely
    /// for organizational purposes to make the benchmark set a bit easier to
    /// digest.
    #[serde(skip)]
    group: String,
    /// The name of the benchmark. Multiple benchmarks might have the same
    /// name, but every benchmark must have a unique "full" name. The full
    /// name is constructed from the benchmark type, the group and the name
    /// given here. The name should be some short string that gives an idea
    /// of what the benchmark is measuring.
    name: String,
    /// The actual regex pattern to measure.
    regex: String,
    /// The haystack to search, inlined into the benchmark definition. One
    /// should only use this when the haystack is short. Either this or
    /// 'haystack-path' must be present.
    ///
    /// Note that this field cannot support invalid UTF-8, since TOML does
    /// not permit invalid UTF-8 anywhere in its file. To create a haystack
    /// with invalid UTF-8, put the data in a file and point to it with
    /// 'haystack-path'.
    haystack: Option<String>,
    /// A file path to a haystack. The path relative to the
    /// benchmarks/haystacks directory. Either this or 'haystack' must be
    /// present.
    haystack_path: Option<String>,
    /// Whether the regex should be matched case insensitively or not.
    case_insensitive: bool,
    /// Whether the regex engine's "Unicode mode" should be enabled.
    unicode: bool,
    /// A benchmark type specific option for 'count'. This indicates how many
    /// matches should be found in the benchmark's haystack.
    match_count: Option<u64>,
    /// A benchmark type specific option for 'count-captures'. This indicates
    /// how many capturing groups match. For every overall regex match,
    /// the sum should be incremented by the number of capturing groups
    /// that participated in the match (which may not be all of them, e.g.,
    /// '([0-9])|([a-z])').
    capture_count: Option<u64>,
    /// A benchmark type specific option for 'grep'. This indicates how many
    /// lines should match the regex in a haystack.
    line_count: Option<u64>,
    /// The regex engines that should be tested as part of this benchmark.
    ///
    /// Each regex engine is just a string that is recognized by this benchmark
    /// harness. Multiple strings may ultimately refer to the same regex
    /// engine, but with a different configuration. Namely, the benchmark
    /// data only permits configuring very general regex options in a first
    /// class way. Such options are expected to be found in pretty much
    /// every regex engine. The benchmark data does *not* permit configuring
    /// engine-specific options, for example, like JIT stack sizes or DFA cache
    /// sizes. Instead, those configurations should be done as different regex
    /// engines. For example, you might have 'pcre2/jit/default-stack-size' and
    /// 'pcre2/jit/large-stack-size'.
    engines: Vec<String>,
    /// An analysis that describes the benchmark, its motivation and, ideally,
    /// any engine-specific details that relate to the most recent results
    /// observed. The analysis should be a "living" document of the benchmark
    /// that gets updated as the results change. (An ambitious goal.)
    analysis: String,
}

impl BenchmarkDef {
    /// Return the full name of this benchmark.
    ///
    /// The full name consists of joining the benchmark's type, group and
    /// name together, separated by a '/'.
    fn full_name(&self) -> String {
        format!("{}/{}/{}", self.benchmark_type, self.group, self.name)
    }

    /// Return the haystack for this benchmark definition. This uses the given
    /// collection of haystacks to look up a haystack path if the haystack
    /// isn't inlined into the benchmark definition.
    ///
    /// If the haystack was not inlined and it could not be found in the given
    /// set of haystacks, then an error is returned.
    fn haystack(&self, haystacks: &Haystacks) -> anyhow::Result<Arc<[u8]>> {
        if let Some(ref haystack) = self.haystack {
            return Ok(Arc::from(haystack.as_bytes()));
        }
        // Unwrap is OK because validation guarantees that either 'haystack'
        // xor 'haystack_path' is present.
        haystacks.get(self.haystack_path.as_ref().unwrap())
    }

    /// Returns an iterator over all the different ways this benchmark should
    /// be run. For example, given a benchmark, it might have specified that
    /// it should be run with multiple regex engines, so there will be at
    /// least one run of the benchmark for each specified engine.
    ///
    /// If the haystack in the benchmark definition could not be found in the
    /// haystacks given, then this returns an error.
    fn iter(
        &self,
        config: &BenchmarkConfig,
        haystacks: &Haystacks,
    ) -> anyhow::Result<BenchmarkIter> {
        Ok(BenchmarkIter {
            it: self.engines.iter(),
            config: config.clone(),
            def: self,
            haystack: self.haystack(haystacks)?,
        })
    }

    /// Validate that our benchmark is consistent with our rules.
    ///
    /// We *mostly* rely on Serde and the type system to produce valid
    /// benchmarks by construction, but there are some bits that are too
    /// annoying to push into Serde/types.
    fn validate(&self) -> anyhow::Result<()> {
        static RE_GROUP: Lazy<Regex> =
            Lazy::new(|| Regex::new(r"^[a-z][-a-z0-9]+$").unwrap());
        static RE_NAME: Lazy<Regex> =
            Lazy::new(|| Regex::new(r"^[a-z][-a-z0-9]+$").unwrap());
        static RE_ENGINE: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"^[a-z][-a-z0-9]+(/[a-z][-a-z0-9]+)*$").unwrap()
        });

        // Benchmark type imposes constraints on which things can or can't
        // be set.
        match self.benchmark_type {
            BenchmarkType::Compile => {
                anyhow::ensure!(
                    self.match_count.is_some(),
                    "'compile' benchmarks must have 'match-count' set \
                     (to verify compiled regex has expected behavior)",
                );
                anyhow::ensure!(
                    self.capture_count.is_none(),
                    "'compile' benchmarks must not have 'capture-count' set",
                );
                anyhow::ensure!(
                    self.line_count.is_none(),
                    "'compile' benchmarks must not have 'line-count' set",
                );
            }
            BenchmarkType::Count => {
                anyhow::ensure!(
                    self.match_count.is_some(),
                    "'count' benchmarks must have 'match-count' set",
                );
                anyhow::ensure!(
                    self.capture_count.is_none(),
                    "'count' benchmarks must not have 'capture-count' set",
                );
                anyhow::ensure!(
                    self.line_count.is_none(),
                    "'count' benchmarks must not have 'line-count' set",
                );
            }
            BenchmarkType::CountCaptures => {
                anyhow::ensure!(
                    self.capture_count.is_some(),
                    "'count-captures' benchmarks must have 'capture-count' set",
                );
                anyhow::ensure!(
                    self.match_count.is_none(),
                    "'count-captures' benchmarks must not have 'match-count' set",
                );
                anyhow::ensure!(
                    self.line_count.is_none(),
                    "'count-captures' benchmarks must not have 'line-count' set",
                );
            }
            BenchmarkType::Grep => {
                anyhow::ensure!(
                    self.line_count.is_some(),
                    "'grep' benchmarks must have 'line-count' set",
                );
                anyhow::ensure!(
                    self.match_count.is_none(),
                    "'grep' benchmarks must not have 'match-count' set",
                );
                anyhow::ensure!(
                    self.capture_count.is_none(),
                    "'grep' benchmarks must not have 'capture-count' set",
                );
            }
            BenchmarkType::RegexRedux => {
                anyhow::ensure!(
                    self.regex.is_empty(),
                    "'regex-redux' benchmark must not set 'regex'",
                );
                anyhow::ensure!(
                    self.line_count.is_none(),
                    "'regex-redux' benchmarks must not have 'line-count' set",
                );
                anyhow::ensure!(
                    self.match_count.is_none(),
                    "'regex-redux' benchmarks must not have 'match-count' set",
                );
                anyhow::ensure!(
                    self.capture_count.is_none(),
                    "'regex-redux' benchmarks must not have 'capture-count' set",
                );
            }
        }
        // Group must be valid.
        anyhow::ensure!(
            RE_GROUP.is_match(&self.group),
            "group name '{}' does not match format '{}' \
             (group name is usually derived from TOML file name)",
            self.group,
            RE_GROUP.as_str(),
        );
        // Name must be valid.
        anyhow::ensure!(
            RE_NAME.is_match(&self.name),
            "benchmark name '{}' does not match format '{}'",
            self.name,
            RE_NAME.as_str(),
        );
        // We must have 'haystack' xor 'haystack-path'.
        anyhow::ensure!(
            !(self.haystack.is_some() && self.haystack_path.is_some()),
            "only one of 'haystack' and 'haystack-path' may be set",
        );
        anyhow::ensure!(
            self.haystack.is_some() || self.haystack_path.is_some(),
            "at least one of 'haystack' and 'haystack-path' must be set",
        );
        // Our engine names should conform as well.
        for engine in self.engines.iter() {
            anyhow::ensure!(
                RE_ENGINE.is_match(engine),
                "engine name '{}' does not match format '{}'",
                engine,
                RE_ENGINE.as_str(),
            );
        }
        // Analysis must be non-empty. Don't be lazy.
        anyhow::ensure!(
            !self.analysis.is_empty(),
            "'analysis' cannot be empty"
        );

        Ok(())
    }
}

/// The type of the benchmark. This basically controls what we're measuring and
/// how to execute the code under test.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
enum BenchmarkType {
    /// Benchmarks the compilation time of a regex.
    Compile,
    /// Benchmarks a simple regex search that returns a count of all
    /// leftmost-first non-overlapping matches in a haystack.
    Count,
    /// Like 'Count', except this counts the number of times every capturing
    /// group in the regex matches. This benchmark only includes regex engines
    /// that can return capture group spans.
    CountCaptures,
    /// Benchmarks a search over every line in a haystack, like grep. The
    /// execution model for this benchmark is to iterate over every line and
    /// run a regex search for each line. We specifically avoid trying to do
    /// anything more clever than that. The measurement is how long it takes
    /// to search every line in the haystack. The regex engine must correctly
    /// report how many lines match.
    Grep,
    /// This is the 'regex-redux' benchmark[1] from The Benchmark Game.
    ///
    /// Since this benchmark is more than just a simple match count, it
    /// requires its own bespoke implementation benchmark type. We use the same
    /// benchmark as The Game does, with these changes: 1) no multi-threading
    /// and 2) the haystack is smaller to make each iteration shorter.
    ///
    /// [1]:
    /// https://benchmarksgame-team.pages.debian.net/benchmarksgame/description/regexredux.html
    RegexRedux,
}

impl std::fmt::Display for BenchmarkType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use self::BenchmarkType::*;
        match *self {
            Compile => write!(f, "compile"),
            Count => write!(f, "count"),
            CountCaptures => write!(f, "count-captures"),
            Grep => write!(f, "grep"),
            RegexRedux => write!(f, "regex-redux"),
        }
    }
}

/// An iterator over all benchmarks from a benchmark definition.
///
/// The lifetime `'d` refers to the benchmark definition from which to generate
/// benchmarks.
#[derive(Debug)]
struct BenchmarkIter<'d> {
    config: BenchmarkConfig,
    def: &'d BenchmarkDef,
    haystack: Arc<[u8]>,
    it: std::slice::Iter<'d, String>,
}

impl<'b> Iterator for BenchmarkIter<'b> {
    type Item = Benchmark;

    fn next(&mut self) -> Option<Benchmark> {
        let engine = self.it.next()?;
        Some(Benchmark {
            config: self.config.clone(),
            def: self.def.clone(),
            haystack: Arc::clone(&self.haystack),
            engine: engine.to_string(),
        })
    }
}

/// A single benchmark that can be executed in order to collect timing samples.
/// Each sample corresponds to a single run of a single regex engine on a
/// particular haystack.
#[derive(Clone, Debug)]
struct Benchmark {
    /// The config, given from the command line.
    config: BenchmarkConfig,
    /// The definition, taken from TOML data.
    def: BenchmarkDef,
    /// The actual haystack data to search, taken from either the TOML data
    /// (via 'haystack') or from the file system (via 'haystack-path').
    haystack: Arc<[u8]>,
    /// The name of the regex engine to execute. This is guaranteed to match
    /// one of the values in 'def.engines'.
    engine: String,
}

impl Benchmark {
    /// Run and collect the results of this benchmark.
    ///
    /// This interrogates the benchmark type and runs the corresponding
    /// benchmark function to produce results.
    fn collect(&self) -> anyhow::Result<Results> {
        match self.def.benchmark_type {
            BenchmarkType::Compile => compile::run(self),
            BenchmarkType::Count => count::run(self),
            BenchmarkType::CountCaptures => count_captures::run(self),
            BenchmarkType::Grep => grep::run(self),
            BenchmarkType::RegexRedux => regexredux::run(self),
        }
    }

    /// Turn the given results collected from running this benchmark into
    /// a single set of aggregate statistics describing the samples in the
    /// results.
    fn aggregate(&self, result: anyhow::Result<Results>) -> AggregateDuration {
        match result {
            Ok(results) => results.to_aggregate(),
            Err(err) => self.aggregate_error(err.to_string()),
        }
    }

    /// Create a new "error" aggregate from this benchmark with the given
    /// error message. This is useful in cases where the benchmark couldn't
    /// run or there was some other discrepancy. Folding the error into the
    /// aggregate value itself avoids recording the error "out of band" and
    /// also avoids silently squashing it.
    fn aggregate_error(&self, err: String) -> AggregateDuration {
        AggregateDuration {
            full_name: self.def.full_name(),
            engine: self.engine.clone(),
            err: Some(err),
            ..AggregateDuration::default()
        }
    }

    /// This creates a new `Benchmark` that is suitable purely for
    /// verification. Namely, it modifies any config necessary to ensure that
    /// the benchmark will run only one iteration and report the result.
    fn verifier(&self) -> Benchmark {
        let config = BenchmarkConfig {
            max_iters: 1,
            max_warmup_iters: 0,
            approx_max_benchmark_time: Duration::ZERO,
            approx_max_warmup_time: Duration::ZERO,
        };
        Benchmark {
            config,
            def: self.def.clone(),
            haystack: Arc::clone(&self.haystack),
            engine: self.engine.clone(),
        }
    }

    /// Run the benchmark given a function to verify that the results are
    /// correct and a function to produce a result.
    ///
    /// This is intended to be used directly by the implementation of each
    /// regex engine. In particular, this might be called after some setup work
    /// (typically compiling the regex itself).
    fn run<T>(
        &self,
        mut verify: impl FnMut(&Benchmark, T) -> anyhow::Result<()>,
        mut test: impl FnMut() -> anyhow::Result<T>,
    ) -> anyhow::Result<Results> {
        let mut results = Results::new(self);
        let warmup_start = Instant::now();
        for _ in 0..self.config.max_warmup_iters {
            let result = test();
            verify(self, result?)?;
            if warmup_start.elapsed() >= self.config.approx_max_warmup_time {
                break;
            }
        }
        let bench_start = Instant::now();
        for _ in 0..self.config.max_iters {
            let start = Instant::now();
            let result = test();
            let elapsed = start.elapsed();
            verify(self, result?)?;
            results.samples.push(elapsed);
            if bench_start.elapsed() >= self.config.approx_max_benchmark_time {
                break;
            }
        }
        results.total = bench_start.elapsed();
        Ok(results)
    }
}

/// The raw results generated by running a benchmark.
#[derive(Clone, Debug)]
struct Results {
    /// The benchmark that was executed.
    benchmark: Benchmark,
    /// The total amount of time that the benchmark ran for.
    total: Duration,
    /// The individual timing samples collected from the benchmark. Each sample
    /// represents the time it takes for a single run of the thing being
    /// measured. This does not include warmup iterations.
    samples: Vec<Duration>,
}

impl Results {
    /// Create a new empty set of results for the given benchmark.
    fn new(b: &Benchmark) -> Results {
        Results {
            benchmark: b.clone(),
            total: Duration::default(),
            samples: vec![],
        }
    }

    /// Convert these results into aggregate statistical values. If there are
    /// no samples, then an "error" aggregate is returned.
    fn to_aggregate(&self) -> AggregateDuration {
        let mut samples = vec![];
        for &dur in self.samples.iter() {
            samples.push(dur.as_secs_f64());
        }
        // It's not quite clear how this could happen, but it's definitely
        // an error. This also makes some unwraps below OK, because we can
        // assume that 'timings' is non-empty.
        if samples.is_empty() {
            let err = "no samples or errors recorded".to_string();
            return self.benchmark.aggregate_error(err);
        }
        // We have no NaNs, so this is fine.
        samples.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap());
        AggregateDuration {
            full_name: self.benchmark.def.full_name(),
            engine: self.benchmark.engine.clone(),
            // We don't expect to have haystacks bigger than 2**64.
            haystack_len: u64::try_from(self.benchmark.haystack.len())
                .unwrap(),
            err: None,
            // We don't expect iterations to exceed 2**64.
            iters: u64::try_from(samples.len()).unwrap(),
            total: self.total,
            // OK because timings.len() > 0
            median: Duration::from_secs_f64(median(&samples).unwrap()),
            // OK because timings.len() > 0
            mean: Duration::from_secs_f64(mean(&samples).unwrap()),
            // OK because timings.len() > 0
            stddev: Duration::from_secs_f64(stddev(&samples).unwrap()),
            // OK because timings.len() > 0
            min: Duration::from_secs_f64(min(&samples).unwrap()),
            // OK because timings.len() > 0
            max: Duration::from_secs_f64(max(&samples).unwrap()),
        }
    }
}

fn mean(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        None
    } else {
        let sum: f64 = xs.iter().sum();
        Some(sum / (xs.len() as f64))
    }
}

fn stddev(xs: &[f64]) -> Option<f64> {
    let len = xs.len() as f64;
    let mean = mean(xs)?;
    let mut deviation_sum_squared = 0.0;
    for &x in xs.iter() {
        deviation_sum_squared += (x - mean).powi(2);
    }
    Some((deviation_sum_squared / len).sqrt())
}

fn median(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        None
    } else if xs.len() % 2 == 1 {
        // Works because integer division rounds down
        Some(xs[xs.len() / 2])
    } else {
        let second = xs.len() / 2;
        let first = second - 1;
        mean(&[xs[first], xs[second]])
    }
}

fn min(xs: &[f64]) -> Option<f64> {
    let mut it = xs.iter().copied();
    let mut min = it.next()?;
    for x in it {
        if x < min {
            min = x;
        }
    }
    Some(min)
}

fn max(xs: &[f64]) -> Option<f64> {
    let mut it = xs.iter().copied();
    let mut max = it.next()?;
    for x in it {
        if x > max {
            max = x;
        }
    }
    Some(max)
}
