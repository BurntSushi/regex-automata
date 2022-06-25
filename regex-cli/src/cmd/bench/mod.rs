use std::{
    collections::{BTreeMap, BTreeSet},
    convert::TryFrom,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use {anyhow::Context, bstr::BString};

use crate::{
    app::{self, App, Args},
    util,
};

mod count;

const ABOUT: &'static str = "\
Run benchmarks.
";

pub fn define() -> App {
    let mut app =
        app::command("bench").about("Run benchmarks.").before_help(ABOUT);
    {
        const SHORT: &str =
            "The directory containing benchmarks and haystacks.";
        const LONG: &str = SHORT;
        app = app.arg(app::flag("dir").short("d").help(SHORT).long_help(LONG));
    }
    app
}

pub fn run(args: &Args) -> anyhow::Result<()> {
    use self::count;

    // BREADCRUMBS:
    //
    // We need filtering.
    //
    // Support other benchmark types.
    //
    // Our core benchmark runner needs to be smarter with respect to determining
    // how many iterations to run.
    //
    // Add warmup configuration.
    //
    // Consider if we need outlier detection, and if so, how to use it.

    let runner = Runner::new(args)?;
    let benchdefs = runner.benchmarks()?;
    let haystacks = runner.haystacks(&benchdefs)?;

    let mut benches = vec![];
    for bdef in &benchdefs.benches {
        for b in bdef.iter(&haystacks)? {
            benches.push(b);
        }
    }

    let mut aggs: Vec<Aggregate> = vec![];
    for b in benches.iter() {
        aggs.push(b.aggregate(|| match &*b.engine {
            "regex/api" => count::regex_api(b),
            "regex/automata/dfa/dense" => count::regex_automata_dfa_dense(b),
            "regex/automata/dfa/sparse" => count::regex_automata_dfa_sparse(b),
            "regex/automata/hybrid" => count::regex_automata_hybrid(b),
            "regex/automata/pikevm" => count::regex_automata_pikevm(b),
            "memchr/memmem" => count::memchr_memmem(b),
            name => anyhow::bail!("unknown regex engine '{}'", name),
        }));
    }
    dbg!(&aggs);
    Ok(())
}

#[derive(Clone, Debug)]
struct Runner {
    dir: PathBuf,
}

impl Runner {
    fn new(args: &Args) -> anyhow::Result<Runner> {
        let mut runner = Runner { dir: PathBuf::from("benches") };
        if let Some(x) = args.value_of_os("dir") {
            runner.dir = PathBuf::from(x);
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
    fn iter(&self, haystacks: &Haystacks) -> anyhow::Result<BenchmarkIter> {
        let haystack = haystacks.get(&self.haystack)?;
        Ok(BenchmarkIter { it: self.engines.iter(), def: self, haystack })
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
            BenchmarkType::Grep => {
                anyhow::ensure!(
                    self.line_count.is_some(),
                    "'grep' benchmarks must have 'line_count' set",
                );
            }
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
    /// Benchmarks a search over every line in a haystack, like grep. The
    /// execution model for this benchmark is to iterate over every line and
    /// run a regex search for each line. We specifically avoid trying to do
    /// anything more clever than that. The measurement is how long it takes
    /// to search every line in the haystack. The regex engine must correctly
    /// report how many lines match.
    Grep,
}

impl std::fmt::Display for BenchmarkType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use self::BenchmarkType::*;
        match *self {
            Compile => write!(f, "compile"),
            Count => write!(f, "count"),
            Grep => write!(f, "grep"),
        }
    }
}

#[derive(Debug)]
struct BenchmarkIter<'b> {
    def: &'b BenchmarkDef,
    haystack: Arc<[u8]>,
    it: std::slice::Iter<'b, String>,
}

impl<'b> Iterator for BenchmarkIter<'b> {
    type Item = Benchmark;

    fn next(&mut self) -> Option<Benchmark> {
        let engine = self.it.next()?;
        Some(Benchmark {
            def: self.def.clone(),
            haystack: Arc::clone(&self.haystack),
            engine: engine.to_string(),
        })
    }
}

#[derive(Clone, Debug)]
struct Benchmark {
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
            Ok(results) => results.to_aggregate(self),
            Err(err) => Aggregate::errored(self, err.to_string()),
        }
    }

    fn run<T>(
        &self,
        mut verify: impl FnMut(&Benchmark, T) -> anyhow::Result<()>,
        mut test: impl FnMut() -> anyhow::Result<T>,
    ) -> anyhow::Result<Results> {
        let mut results = Results::new();
        for i in 0..100 {
            let mut start = Instant::now();
            let result = test();
            let elapsed = start.elapsed();
            verify(self, result?)?;
        }
        let start = Instant::now();
        let iters = 1_000;
        for i in 0..iters {
            let mut start = Instant::now();
            let result = test();
            let elapsed = start.elapsed();
            verify(self, result?)?;
            results.results.push(elapsed);
        }
        results.total = start.elapsed();
        Ok(results)
    }
}

#[derive(Debug, Default)]
struct Results {
    total: Duration,
    results: Vec<Duration>,
}

impl Results {
    fn new() -> Results {
        Results::default()
    }

    fn to_aggregate(&self, b: &Benchmark) -> Aggregate {
        let mut timings = vec![];
        for &dur in self.results.iter() {
            timings.push(dur.as_secs_f64());
        }
        // It's not quite clear how this could happen, but it's definitely
        // an error. This also makes some unwraps below OK, because we can
        // assume that 'timings' is non-empty.
        if timings.is_empty() {
            let err = "no timings or errors recorded".to_string();
            return Aggregate::errored(b, err);
        }
        // We have no NaNs, so this is fine.
        timings.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap());
        Aggregate {
            full_name: b.def.full_name(),
            engine: b.engine.clone(),
            total: self.total,
            // OK because timings.len() > 0
            median: Duration::from_secs_f64(median(&timings).unwrap()),
            // OK because timings.len() > 0
            mean: Duration::from_secs_f64(mean(&timings).unwrap()),
            // OK because timings.len() > 0
            stddev: Duration::from_secs_f64(stddev(&timings).unwrap()),
            // OK because timings.len() > 0
            min: Duration::from_secs_f64(min(&timings).unwrap()),
            // OK because timings.len() > 0
            max: Duration::from_secs_f64(max(&timings).unwrap()),
            // We don't expect iterations to exceed 2**64.
            iters: u64::try_from(timings.len()).unwrap(),
            err: None,
        }
    }
}

#[derive(Clone, Debug, Default, serde::Serialize)]
struct Aggregate {
    full_name: String,
    engine: String,
    total: Duration,
    median: Duration,
    mean: Duration,
    stddev: Duration,
    min: Duration,
    max: Duration,
    iters: u64,
    err: Option<String>,
}

impl Aggregate {
    fn errored(b: &Benchmark, err: String) -> Aggregate {
        Aggregate {
            full_name: b.def.full_name(),
            engine: b.engine.clone(),
            err: Some(err),
            ..Aggregate::default()
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
