#![allow(dead_code, unused_imports, unused_variables)]

#[macro_use]
extern crate criterion;
extern crate regex_automata;

use criterion::{Bencher, Benchmark, Criterion, Throughput};
use regex_automata::{DFABuilder, RegexBuilder};

use inputs::{
    SHERLOCK_HUGE, SHERLOCK_TINY, EMPTY,
};

mod inputs;

fn is_match(c: &mut Criterion) {
    let corpus = SHERLOCK_HUGE.corpus;
    define(c, "is-match", "sherlock-huge", corpus, move |b| {
        let dfa = RegexBuilder::new()
            .anchored(false)
            .minimize(true)
            .premultiply(true)
            .byte_classes(false)
            .build(r"\p{Greek}")
            .unwrap();
        b.iter(|| {
            assert!(!dfa.is_match(corpus));
        });
    });

    let corpus = SHERLOCK_TINY.corpus;
    define(c, "is-match", "sherlock-tiny", corpus, move |b| {
        let dfa = RegexBuilder::new()
            .anchored(false)
            .minimize(true)
            .premultiply(true)
            .byte_classes(false)
            .build(r"\p{Greek}")
            .unwrap();
        b.iter(|| {
            assert!(!dfa.is_match(corpus));
        });
    });

    let corpus = EMPTY.corpus;
    define(c, "is-match", "empty", corpus, move |b| {
        let dfa = RegexBuilder::new()
            .anchored(false)
            .minimize(true)
            .premultiply(true)
            .byte_classes(false)
            .build(r"\p{Greek}")
            .unwrap();
        b.iter(|| {
            assert!(!dfa.is_match(corpus));
        });
    });
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

fn define_compile(
    c: &mut Criterion,
    group_name: &str,
    pattern: &'static str,
) {
    let group = format!("compile/{}", group_name);
    define(c, &group, "unminimized-noclasses", &[], move |b| {
        b.iter(|| {
            let result = DFABuilder::new()
                .anchored(true)
                .minimize(false)
                .premultiply(false)
                .byte_classes(false)
                .build(pattern);
            assert!(result.is_ok());
        });
    });
    define(c, &group, "unminimized-classes", &[], move |b| {
        b.iter(|| {
            let result = DFABuilder::new()
                .anchored(true)
                .minimize(false)
                .premultiply(false)
                .byte_classes(true)
                .build(pattern);
            assert!(result.is_ok());
        });
    });
    // TODO: Minimization is too slow to benchmark for now...
    /*
    define(c, &group, "minimized-noclasses", &[], move |b| {
        b.iter(|| {
            let result = DFABuilder::new()
                .anchored(true)
                .minimize(true)
                .premultiply(false)
                .byte_classes(false)
                .build(pattern);
            assert!(result.is_ok());
        });
    });
    define(c, &group, "minimized-classes", &[], move |b| {
        b.iter(|| {
            let result = DFABuilder::new()
                .anchored(true)
                .minimize(true)
                .premultiply(false)
                .byte_classes(true)
                .build(pattern);
            assert!(result.is_ok());
        });
    });
    */
}

fn define(
    c: &mut Criterion,
    group_name: &str,
    bench_name: &str,
    corpus: &[u8],
    bench: impl FnMut(&mut Bencher) + 'static,
) {
    let tput = Throughput::Bytes(corpus.len() as u32);
    let benchmark = Benchmark::new(bench_name, bench).throughput(tput);
    c.bench(group_name, benchmark);
}

criterion_group!(g1, is_match);
criterion_group!(g2, compile_unicode_other_math);
criterion_group!(g3, compile_unicode_other_uppercase);
criterion_group!(g4, compile_muammar);
criterion_main!(g1, g2, g3, g4);
