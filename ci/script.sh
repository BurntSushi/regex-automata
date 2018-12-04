#!/bin/sh

set -ex

cargo doc --verbose
cargo build --verbose
cargo test --verbose
