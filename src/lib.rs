#![allow(dead_code, unused_imports, unused_variables)]

extern crate regex_syntax;
extern crate utf8_ranges;

pub use builder::DFABuilder;
pub use dfa::DFA;
pub use error::{Error, ErrorKind};

mod builder;
mod determinize;
mod dfa;
mod error;
mod minimize;
mod nfa;
mod sparse;
