#![allow(dead_code, unused_imports, unused_variables)]

extern crate regex_syntax;
extern crate utf8_ranges;

pub use builder::DFABuilder;
pub use dfa::{DFA, DFAKind};
pub use dfa_ref::DFARef;
pub use error::{Error, ErrorKind};

mod builder;
mod determinize;
mod dfa;
mod dfa_ref;
mod error;
mod minimize;
mod nfa;
mod sparse;
