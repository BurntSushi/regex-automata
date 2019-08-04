#!/bin/sh

set -ex

cargo doc --verbose
cargo build --verbose

# Our dev dependencies are increasing their MSRV more quickly then we want to,
# so only test the basic build on non-stable/beta/nightly builds.
if ! echo "$TRAVIS_RUST_VERSION" | grep -Eq '^[^0-9]+$'; then
    exit 0
fi

cargo test --verbose --lib
cargo test --verbose --doc

cargo doc --verbose --no-default-features
cargo build --verbose --no-default-features

if [ "$TRAVIS_RUST_VERSION" = "stable" ]; then
  rustup component add rustfmt
  cargo fmt -- --check
fi

# Validate no_std status if this version of rust supports embedded targets
# (starting with 1.31 stable)
if rustup target add thumbv7em-none-eabihf; then
    cargo build --verbose --no-default-features --target thumbv7em-none-eabihf
else
    echo "Skipping no_std test..."
fi

if [ "$TRAVIS_RUST_VERSION" = "nightly" ]; then
    # these tests take forever, so only do them on nightly
    cargo test --verbose --test default
    # compile benchmarks, but don't run them
    cargo bench --verbose --manifest-path bench/Cargo.toml ////
    # make sure the debug tool builds
    cargo build --verbose --manifest-path regex-automata-debug/Cargo.toml
fi
