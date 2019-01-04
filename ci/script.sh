#!/bin/sh

set -ex

cargo doc --verbose
cargo build --verbose
cargo test --verbose --lib
cargo test --verbose --doc

cargo doc --verbose --no-default-features
cargo build --verbose --no-default-features

if [ "$TRAVIS_RUST_VERSION" = "nightly" ]; then
    # these tests take forever, so only do them on nightly
    cargo test --verbose --test default
    # compile benchmarks, but don't run them
    cargo bench --verbose --manifest-path bench/Cargo.toml ////
    # make sure the debug tool builds
    cargo build --verbose --manifest-path regex-automata-debug/Cargo.toml
fi
