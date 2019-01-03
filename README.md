regex-automata
==============
A low level regular expression library that uses deterministic finite automata.
It supports a rich syntax with Unicode support, has extensive options for
configuring the best space vs time trade off for your use case and provides
support for cheap deserialization of automata for use in `no_std` environments.

[![Linux build status](https://api.travis-ci.org/BurntSushi/regex-automata.svg)](https://travis-ci.org/BurntSushi/regex-automata)
[![Windows build status](https://ci.appveyor.com/api/projects/status/github/BurntSushi/regex-automata?svg=true)](https://ci.appveyor.com/project/BurntSushi/regex-automata)
[![](http://meritbadge.herokuapp.com/regex-automata)](https://crates.io/crates/regex-automata)

Dual-licensed under MIT or the [UNLICENSE](http://unlicense.org).


### Documentation

https://docs.rs/regex-automata


### Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
regex-automata = "0.1"
```

and this to your crate root (if you're using Rust 2015):

```rust
extern crate regex_automata;
```

