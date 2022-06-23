use std::{
    path::Path,
    time::{Duration, Instant},
};

use {anyhow::Context, bstr::BString};

use crate::{
    app::{self, App, Args},
    util,
};

const ABOUT: &'static str = "\
Run 'count' benchmarks.
";

pub fn define() -> App {
    app::command("count").about("Run 'count' benchmarks.").before_help(ABOUT)
}

pub fn run(args: &Args) -> anyhow::Result<()> {
    let mut outputs: Vec<anyhow::Result<BenchOutput>> = vec![];
    let mut benches = Benchmarks::new();
    benches.load("benches/count/subtitle.toml")?;
    for b in &benches.benches {
        for engine in &b.engines {
            let e: &str = &**engine;
            outputs.push(match e {
                "regex/api/regex" => run_regex_api_regex(b),
                "regex-automata/nfa/thompson/pikevm" => {
                    run_regex_automata_nfa_thompson_pikevm(b)
                }
                name => {
                    Err(anyhow::anyhow!("unrecognize regex engine '{}'", name))
                }
            });
        }
    }
    dbg!(&outputs);
    Ok(())
}

const SUBTITLE: &'static str = include_str!(
    "../../../../../benches/count/haystacks/opensubtitles/en-huge.txt"
);

fn benchmark(
    b: &Benchmark,
    mut count: impl FnMut(&[u8]) -> anyhow::Result<u64>,
) -> anyhow::Result<BenchOutput> {
    let iters = 1_000;
    let mut elapsed_sum = Duration::ZERO;
    for i in 0..iters {
        let mut start = Instant::now();
        let got = count(SUBTITLE.as_bytes())?;
        let elapsed = Instant::now().duration_since(start);
        elapsed_sum += elapsed;
    }
    Ok(BenchOutput {
        group: "".to_string(),
        name: b.name.clone(),
        median: 0.0,
        mean: (elapsed_sum.as_nanos() as f64) / (iters as f64),
        stddev: 0.0,
        min: 0,
        max: 0,
        iters,
    })
}

fn run_regex_api_regex(b: &Benchmark) -> anyhow::Result<BenchOutput> {
    use regex::bytes::Regex;
    let re = Regex::new(&b.regex)?;
    benchmark(b, |haystack| Ok(re.find_iter(haystack).count() as u64))
}

fn run_regex_automata_nfa_thompson_pikevm(
    b: &Benchmark,
) -> anyhow::Result<BenchOutput> {
    use automata::nfa::thompson::pikevm::PikeVM;
    let re = PikeVM::new(&b.regex)?;
    let mut cache = re.create_cache();
    benchmark(b, |haystack| {
        Ok(re.find_iter(&mut cache, haystack).count() as u64)
    })
}

#[derive(Clone, Debug, serde::Serialize)]
struct BenchOutput {
    group: String,
    name: String,
    median: f64,
    mean: f64,
    stddev: f64,
    min: u64,
    max: u64,
    iters: u64,
}

#[derive(Clone, Debug, serde::Deserialize)]
struct Benchmarks {
    benches: Vec<Benchmark>,
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct Benchmark {
    name: String,
    regex: String,
    haystack: String,
    match_count: usize,
    case_insensitive: bool,
    unicode: bool,
    utf8: bool,
    engines: Vec<String>,
}

impl Benchmarks {
    fn new() -> Benchmarks {
        Benchmarks { benches: vec![] }
    }

    fn load<P: AsRef<Path>>(&mut self, path: P) -> anyhow::Result<()> {
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
        let mut index = 1;
        let mut benches: Benchmarks = toml::from_slice(&data)
            .with_context(|| format!("error decoding TOML for '{}'", group))?;
        for b in &mut benches.benches {
            b.validate()
                .with_context(|| format!("error loading test '{}'", b.name))?;
        }
        self.benches.extend(benches.benches);
        Ok(())
    }
}

impl Benchmark {
    fn validate(&self) -> anyhow::Result<()> {
        Ok(())
    }
}
