use std::{
    collections::{BTreeMap, BTreeSet},
    convert::TryFrom,
    ffi::OsStr,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use {
    anyhow::Context,
    bstr::{BString, ByteSlice, ByteVec},
    regex::Regex,
};

use crate::{
    app::{self, App, Args},
    util::{self, ShortHumanDuration, Throughput},
};

mod count;

const ABOUT_SHORT: &'static str = "\
Run benchmarks.
";

const ABOUT_LONG: &'static str = "\
Run benchmarks.
";

pub fn define() -> App {
    let mut app =
        app::command("bench").about(ABOUT_SHORT).before_help(ABOUT_LONG);
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
        const SHORT: &str = "The max number of warmup iterations to execute.";
        const LONG: &str = "\
The max number of warmup iterations to execute.
";
        app =
            app.arg(app::flag("max-warmup-iters").help(SHORT).long_help(LONG));
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
    app
}

pub fn run(args: &Args) -> anyhow::Result<()> {
    use self::count;

    let runner = Runner::new(args)?;
    let benchdefs = runner.benchmarks()?;
    let haystacks = runner.haystacks(&benchdefs)?;

    let mut benches = vec![];
    for bdef in &benchdefs.benches {
        if !runner.bench_filter.include(&bdef.full_name()) {
            continue;
        }
        for b in bdef.iter(&runner.bench_config, &haystacks)? {
            if !runner.engine_filter.include(&b.engine) {
                continue;
            }
            benches.push(b);
        }
    }

    let mut wtr = csv::Writer::from_writer(std::io::stdout());
    if runner.list {
        for b in benches.iter() {
            wtr.write_record(&[
                b.def.full_name(),
                b.engine.clone(),
                b.def.haystack.clone(),
            ])?;
        }
        return Ok(());
    }
    for b in benches.iter() {
        let agg = b.aggregate(|| match b.def.benchmark_type {
            BenchmarkType::Compile => todo!(),
            BenchmarkType::Count => self::count::run(b),
            BenchmarkType::CountCaptures => todo!(),
            BenchmarkType::Grep => todo!(),
            BenchmarkType::RegexRedux => todo!(),
        });
        wtr.serialize(agg.into_throughput())?;
        wtr.flush()?;
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct Runner {
    dir: PathBuf,
    bench_filter: Filter,
    engine_filter: Filter,
    bench_config: BenchmarkConfig,
    list: bool,
}

impl Runner {
    fn new(args: &Args) -> anyhow::Result<Runner> {
        let mut runner = Runner {
            dir: PathBuf::from("benchmarks"),
            // These unwraps are OK because an empty set of rules always works.
            bench_filter: Filter::new([].into_iter()).unwrap(),
            engine_filter: Filter::new([].into_iter()).unwrap(),
            bench_config: BenchmarkConfig::default(),
            list: args.is_present("list"),
        };
        if let Some(x) = args.value_of_os("dir") {
            runner.dir = PathBuf::from(x);
        }
        if let Some(rules) = args.values_of_os("filter") {
            runner.bench_filter = Filter::new(rules).context("-f/--filter")?;
        }
        if let Some(rules) = args.values_of_os("engine") {
            runner.engine_filter =
                Filter::new(rules).context("-e/--engine")?;
        }
        if let Some(x) = args.value_of_lossy("max-iters") {
            runner.bench_config.max_iters =
                x.parse().context("--max-iters")?;
        }
        if let Some(x) = args.value_of_lossy("max-warmup-iters") {
            runner.bench_config.max_warmup_iters =
                x.parse().context("--max-warmup-iters")?;
        }
        if let Some(x) = args.value_of_lossy("bench-time") {
            let hdur: ShortHumanDuration =
                x.parse().context("--bench-time")?;
            runner.bench_config.approx_max_benchmark_time =
                Duration::from(hdur);
        }
        Ok(runner)
    }

    fn benchmarks(&self) -> anyhow::Result<BenchmarkDefs> {
        let mut benches = BenchmarkDefs::new();
        benches.load_dir(&self.dir)?;
        Ok(benches)
    }

    fn haystacks(&self, benches: &BenchmarkDefs) -> anyhow::Result<Haystacks> {
        let dir = self.dir.join("haystacks");

        let mut haystacks = Haystacks::new();
        haystacks.load_dir(dir, &benches.haystack_names())?;
        Ok(haystacks)
    }
}

#[derive(Clone, Debug)]
struct BenchmarkConfig {
    max_warmup_iters: u64,
    max_iters: u64,
    approx_max_benchmark_time: Duration,
}

impl Default for BenchmarkConfig {
    fn default() -> BenchmarkConfig {
        BenchmarkConfig {
            max_warmup_iters: 1_000,
            max_iters: 100_000,
            approx_max_benchmark_time: Duration::from_secs(3),
        }
    }
}

#[derive(Clone, Debug)]
struct Haystacks {
    /// A map from a haystack name (the file path relative to, e.g.,
    /// benches/haystacks) to the haystack itself. We use a Arc<u8> instead of
    /// a Vec<u8> so that we can reuse a single copy of the haystack everywhere
    /// without lifetimes.
    map: BTreeMap<String, Arc<[u8]>>,
}

impl Haystacks {
    fn new() -> Haystacks {
        Haystacks { map: BTreeMap::new() }
    }

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

    fn load_bytes(&mut self, name: &str, data: Vec<u8>) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.map.contains_key(name),
            "found duplicate haystack '{}'",
            name
        );
        self.map.insert(name.to_owned(), Arc::from(data));
        Ok(())
    }

    fn get(&self, name: &str) -> anyhow::Result<Arc<[u8]>> {
        if let Some(ref haystack) = self.map.get(name) {
            return Ok(Arc::clone(haystack));
        }
        anyhow::bail!("could not find haystack '{}'", name)
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
struct BenchmarkDefs {
    #[serde(default)] // allows empty TOML files
    benches: Vec<BenchmarkDef>,
}

impl BenchmarkDefs {
    fn new() -> BenchmarkDefs {
        BenchmarkDefs { benches: vec![] }
    }

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

    fn load_slice(&mut self, group: &str, data: &[u8]) -> anyhow::Result<()> {
        let mut benches: BenchmarkDefs = toml::from_slice(&data)
            .with_context(|| format!("error decoding TOML for '{}'", group))?;
        for b in &mut benches.benches {
            b.group = group.to_string();
            b.validate()
                .with_context(|| format!("error loading test '{}'", b.name))?;
        }
        self.benches.extend(benches.benches);
        Ok(())
    }

    fn haystack_names(&self) -> BTreeSet<String> {
        let mut names = BTreeSet::new();
        for b in &self.benches {
            names.insert(b.haystack.clone());
        }
        names
    }
}

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
    /// A pointer to a haystack. This is NOT the haystack itself. Instead,
    /// it is the name of a haystack that is resolved when the benchmark is
    /// executed. The pointer is a file path relative to the benches/haystacks
    /// directory.
    haystack: String,
    /// Whether the regex should be matched case insensitively or not.
    case_insensitive: bool,
    /// Whether the regex engine's "Unicode mode" should be enabled.
    unicode: bool,
    /// A benchmark type specific option for 'count'. This indicates how many
    /// matches should be found in the benchmark's haystack.
    match_count: Option<u64>,
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
}

impl BenchmarkDef {
    /// Return the full name of this benchmark.
    ///
    /// The full name consists of joining the benchmark's type, group and
    /// name together, separated by a '/'.
    fn full_name(&self) -> String {
        format!("{}/{}/{}", self.benchmark_type, self.group, self.name)
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
        let haystack = haystacks.get(&self.haystack)?;
        Ok(BenchmarkIter {
            it: self.engines.iter(),
            config: config.clone(),
            def: self,
            haystack,
        })
    }

    /// Validate that our benchmark is consistent with our rules.
    ///
    /// We *mostly* rely on Serde and the type system to produce valid
    /// benchmarks by construction, but there are some bits that are too
    /// annoying to push into Serde/types.
    fn validate(&self) -> anyhow::Result<()> {
        match self.benchmark_type {
            BenchmarkType::Compile => {}
            BenchmarkType::Count => {
                anyhow::ensure!(
                    self.match_count.is_some(),
                    "'count' benchmarks must have 'match_count' set",
                );
            }
            BenchmarkType::CountCaptures => {
                anyhow::ensure!(
                    self.match_count.is_some(),
                    "'count-capture' benchmarks must have 'capture_count' set",
                );
            }
            BenchmarkType::Grep => {
                anyhow::ensure!(
                    self.line_count.is_some(),
                    "'grep' benchmarks must have 'line_count' set",
                );
            }
            BenchmarkType::RegexRedux => {}
        }
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

#[derive(Debug)]
struct BenchmarkIter<'b> {
    config: BenchmarkConfig,
    def: &'b BenchmarkDef,
    haystack: Arc<[u8]>,
    it: std::slice::Iter<'b, String>,
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

#[derive(Clone, Debug)]
struct Benchmark {
    config: BenchmarkConfig,
    def: BenchmarkDef,
    haystack: Arc<[u8]>,
    engine: String,
}

impl Benchmark {
    fn aggregate(
        &self,
        mut results: impl FnMut() -> anyhow::Result<Results>,
    ) -> Aggregate {
        match results() {
            Ok(results) => results.to_aggregate(),
            Err(err) => Aggregate::errored(self, err.to_string()),
        }
    }

    fn run<T>(
        &self,
        mut verify: impl FnMut(&Benchmark, T) -> anyhow::Result<()>,
        mut test: impl FnMut() -> anyhow::Result<T>,
    ) -> anyhow::Result<Results> {
        let mut results = Results::new(self);
        let warmup_start = Instant::now();
        let max_warmup_time = self.config.approx_max_benchmark_time / 4;
        for _ in 0..self.config.max_warmup_iters {
            let mut start = Instant::now();
            let result = test();
            let elapsed = start.elapsed();
            verify(self, result?)?;
            if warmup_start.elapsed() >= max_warmup_time {
                break;
            }
        }
        let bench_start = Instant::now();
        for _ in 0..self.config.max_iters {
            let mut start = Instant::now();
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

#[derive(Clone, Debug)]
struct Results {
    benchmark: Benchmark,
    total: Duration,
    samples: Vec<Duration>,
}

impl Results {
    fn new(b: &Benchmark) -> Results {
        Results {
            benchmark: b.clone(),
            total: Duration::default(),
            samples: vec![],
        }
    }

    fn to_aggregate(&self) -> Aggregate {
        let mut samples = vec![];
        for &dur in self.samples.iter() {
            samples.push(dur.as_secs_f64());
        }
        // It's not quite clear how this could happen, but it's definitely
        // an error. This also makes some unwraps below OK, because we can
        // assume that 'timings' is non-empty.
        if samples.is_empty() {
            let err = "no samples or errors recorded".to_string();
            return Aggregate::errored(&self.benchmark, err);
        }
        // We have no NaNs, so this is fine.
        samples.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap());
        Aggregate {
            full_name: self.benchmark.def.full_name(),
            engine: self.benchmark.engine.clone(),
            haystack: self.benchmark.def.haystack.clone(),
            haystack_len: self.benchmark.haystack.len() as u64,
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

/// Aggregate statistics for a particular benchmark.
///
/// This could probably be simplified somewhat by attaching a `Benchmark`
/// to it, but it is an intentionally flattened structure so as to make
/// (de)serializing a bit more convenient.
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
struct Aggregate {
    full_name: String,
    engine: String,
    haystack: String,
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

impl Aggregate {
    /// Create a new aggregate value that represents an error.
    fn errored(b: &Benchmark, err: String) -> Aggregate {
        Aggregate {
            full_name: b.def.full_name(),
            engine: b.engine.clone(),
            haystack: b.def.haystack.clone(),
            err: Some(err),
            ..Aggregate::default()
        }
    }

    /// Convert this aggregate value from using duration to throughput.
    fn into_throughput(self) -> AggregateThroughput {
        // Getting stddev as a throughput is not quite as straight-forward. I
        // believe the correct thing to do here is to compute the ratio between
        // stddev and mean in terms of duration, then compute the throughput of
        // the mean and then use the ratio on the mean throughput to find the
        // stddev throughput.
        let ratio = self.stddev.as_secs_f64() / self.mean.as_secs_f64();
        let mean = Throughput::new(self.haystack_len, self.mean);
        let stddev =
            Throughput::from_bytes_per_second(mean.bytes_per_second() * ratio);
        AggregateThroughput {
            full_name: self.full_name,
            engine: self.engine,
            haystack: self.haystack,
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

/// Like 'Aggregate', but uses throughputs instead of durations. In my opinion,
/// throughput is easier to reason about for regex benchmarks. It gives you
/// the same information, but it also gives you some intuition for how long it
/// will take to search some data. Namely, throughput provides more bits of
/// information when compared to benchmark iteration duration.
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
struct AggregateThroughput {
    full_name: String,
    engine: String,
    haystack: String,
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
struct Filter {
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
    fn new<'a>(
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
            let (pattern, blacklist) = if rule.starts_with('~') {
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
    fn include(&self, subject: &str) -> bool {
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
