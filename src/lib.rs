/*!
TODO.
*/

// #![deny(missing_docs)]

extern crate byteorder;
extern crate regex_syntax;
extern crate utf8_ranges;

pub use builder::{DFABuilder, RegexBuilder};
pub use dfa::DFA;
pub use dfa_ref::DFARef;
pub use error::{Error, ErrorKind};
pub use regex::Regex;
pub use sparse::SparseDFA;
pub use state_id::StateID;

#[macro_use]
mod macros;
mod builder;
mod determinize;
mod dfa;
mod dfa_ref;
mod error;
mod regex;
mod minimize;
mod nfa;
mod sparse;
mod sparse_set;
mod state_id;
