#!/bin/sh

set -ex

cargo doc --verbose
cargo build --verbose
cargo test --verbose

cargo doc --verbose --no-default-features
cargo build --verbose --no-default-features

if [ "$TRAVIS_RUST_VERSION" = "nightly" ]; then
    # compile benchmarks, but don't run them
    cargo bench --manifest-path bench/Cargo.toml ////
fi
