#![allow(dead_code, unused_imports, unused_variables)]

#[macro_use]
extern crate criterion;
extern crate regex_automata;

use criterion::{Bencher, Benchmark, Criterion, Throughput};
use regex_automata::DFABuilder;

use inputs::{
    SHERLOCK_HUGE, SHERLOCK_TINY, EMPTY,
};

mod inputs;

fn is_match(c: &mut Criterion) {
    let corpus = SHERLOCK_HUGE.corpus;
    define(c, "is_match", "sherlock-huge", corpus, move |b| {
        let dfa = DFABuilder::new()
            .anchored(false)
            .build(r"\p{Greek}")
            .unwrap();
        b.iter(|| {
            assert!(!dfa.is_match(corpus));
        });
    });

    let corpus = SHERLOCK_TINY.corpus;
    define(c, "is_match", "sherlock-tiny", corpus, move |b| {
        let dfa = DFABuilder::new()
            .anchored(false)
            .build(r"\p{Greek}")
            .unwrap();
        b.iter(|| {
            assert!(!dfa.is_match(corpus));
        });
    });

    let corpus = EMPTY.corpus;
    define(c, "is_match", "empty", corpus, move |b| {
        let dfa = DFABuilder::new()
            .anchored(false)
            .build(r"\p{Greek}")
            .unwrap();
        b.iter(|| {
            assert!(!dfa.is_match(corpus));
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
    let tput = Throughput::Bytes(corpus.len() as u32);
    let benchmark = Benchmark::new(bench_name, bench).throughput(tput);
    c.bench(group_name, benchmark);
}

criterion_group!(g1, is_match);
criterion_main!(g1);
