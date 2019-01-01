/*!
TODO.
*/

// #![deny(missing_docs)]

extern crate byteorder;
extern crate regex_syntax;
extern crate utf8_ranges;

pub use builder::{DenseDFABuilder, RegexBuilder};
pub use dense::DenseDFA;
pub use dfa::DFA;
pub use error::{Error, ErrorKind};
pub use regex::Regex;
pub use state_id::StateID;

#[macro_use]
mod macros;
mod builder;
mod classes;
mod determinize;
mod dense;
// mod dense_ref;
mod dfa;
mod error;
mod regex;
mod minimize;
mod nfa;
pub mod sparse;
mod sparse_set;
mod state_id;
