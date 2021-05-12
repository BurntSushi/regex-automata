use std::time::Duration;

use criterion::{
    criterion_group, criterion_main, Bencher, Benchmark, Criterion, Throughput,
};
use regex_automata::dfa::{dense, RegexBuilder};
use regex_automata::nfa::thompson;

use crate::inputs::*;

mod inputs;

fn is_match(c: &mut Criterion) {
    let corpus = SHERLOCK_HUGE;
    define(c, "is-match", "sherlock-huge", corpus, move |b| {
        let re = RegexBuilder::new().build(r"\p{Greek}").unwrap();
        // let re = re.forward().to_sparse().unwrap();
        b.iter(|| {
            assert!(!re.is_match(corpus));
        });
    });

    // let corpus = OPEN_ZH_SMALL;
    let corpus = SHERLOCK_SMALL;
    define(c, "is-match", "sherlock-small", corpus, move |b| {
        let re = RegexBuilder::new().build(r"\p{Greek}").unwrap();
        // let re = re.forward().to_sparse().unwrap();
        b.iter(|| {
            assert!(!re.is_match(corpus));
        });
    });

    let corpus = SHERLOCK_TINY;
    define(c, "is-match", "sherlock-tiny", corpus, move |b| {
        let re = RegexBuilder::new().build(r"\p{Greek}").unwrap();
        b.iter(|| {
            assert!(!re.is_match(corpus));
        });
    });

    let corpus = EMPTY;
    define(c, "is-match", "empty", corpus, move |b| {
        let re = RegexBuilder::new().build(r"\p{Greek}").unwrap();
        b.iter(|| {
            assert!(!re.is_match(corpus));
        });
    });
}

// \w has 128,640 codepoints.
fn compile_unicode_word(c: &mut Criterion) {
    define_compile(c, "unicode-word", r"\w");
    define_compile_reverse(c, "unicode-word", r"\w");
}

// \p{Other_Math} has 1,362 codepoints
fn compile_unicode_other_math(c: &mut Criterion) {
    define_compile(c, "unicode-other-math", r"\p{Other_Math}");
}

// \p{Other_Uppercase} has 120 codepoints
fn compile_unicode_other_uppercase(c: &mut Criterion) {
    define_compile(
        c,
        "unicode-other-uppercase",
        r"\p{any}*?\p{Other_Uppercase}",
    );
}

fn compile_muammar(c: &mut Criterion) {
    define_compile(
        c,
        "muammar",
        r"\p{any}*?M[ou]'?am+[ae]r .*([AEae]l[- ])?[GKQ]h?[aeu]+([dtz][dhz]?)+af[iy]",
    );
}

fn define_compile(c: &mut Criterion, group_name: &str, pattern: &'static str) {
    let group = format!("fwd-compile/{}", group_name);
    define(c, &group, "default", &[], move |b| {
        b.iter(|| {
            let result = dense::Builder::new()
                .configure(dense::Config::new().anchored(true))
                .build(pattern);
            assert!(result.is_ok());
        });
    });
}

fn define_compile_reverse(
    c: &mut Criterion,
    group_name: &str,
    pattern: &'static str,
) {
    let group = format!("rev-compile/{}", group_name);
    define(c, &group, "default", &[], move |b| {
        b.iter(|| {
            let result = dense::Builder::new()
                .configure(dense::Config::new().anchored(true))
                .thompson(thompson::Config::new().reverse(true))
                .build(pattern);
            assert!(result.is_ok());
        });
    });
}

fn define(
    c: &mut Criterion,
    group_name: &str,
    bench_name: &str,
    corpus: &[u8],
    bench: impl FnMut(&mut Bencher) + 'static,
) {
    let tput = Throughput::Bytes(corpus.len() as u64);
    let benchmark = Benchmark::new(bench_name, bench)
        .throughput(tput)
        .sample_size(25)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3));
    c.bench(group_name, benchmark);
}

criterion_group!(g1, is_match);
criterion_group!(g2, compile_unicode_other_math);
criterion_group!(g3, compile_unicode_other_uppercase);
criterion_group!(g4, compile_muammar);
criterion_group!(g5, compile_unicode_word);
criterion_main!(g1, g2, g3, g4, g5);
