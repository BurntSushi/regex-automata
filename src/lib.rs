/*!
TODO.
*/

#![deny(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate core;

extern crate byteorder;
#[cfg(feature = "std")]
extern crate regex_syntax;
#[cfg(feature = "std")]
extern crate utf8_ranges;

pub use dense::DenseDFA;
pub use dfa::DFA;
#[cfg(feature = "std")]
pub use error::{Error, ErrorKind};
pub use regex::Regex;
#[cfg(feature = "std")]
pub use regex::RegexBuilder;
pub use sparse::SparseDFA;
pub use state_id::StateID;

mod classes;
#[cfg(feature = "std")]
mod determinize;
#[path = "dense.rs"]
mod dense_imp;
mod dfa;
#[cfg(feature = "std")]
mod error;
mod regex;
#[cfg(feature = "std")]
mod minimize;
#[cfg(feature = "std")]
mod nfa;
#[path = "sparse.rs"]
mod sparse_imp;
#[cfg(feature = "std")]
mod sparse_set;
mod state_id;

/// Types and routines specific to dense DFAs.
///
/// This module is the home of [`DenseDFA`](enum.DenseDFA.html) and each of its
/// corresponding variant DFA types, such as [`Standard`](struct.Standard.html)
/// and [`ByteClass`](struct.ByteClass.html).
///
/// This module also contains a [builder](struct.Builder.html) for
/// configuring the construction of a dense DFA.
pub mod dense {
    pub use dense_imp::*;
}

/// Types and routines specific to sparse DFAs.
///
/// This module is the home of [`SparseDFA`](enum.SparseDFA.html) and each of
/// its corresponding variant DFA types, such as
/// [`Standard`](struct.Standard.html) and
/// [`ByteClass`](struct.ByteClass.html).
///
/// Unlike the [`dense`](../dense/index.html) module, this module does not
/// contain a builder specific for sparse DFAs. Instead, the intended way to
/// build a sparse DFA is either by using a default configuration with its
/// [constructor](enum.SparseDFA.html#method.new),
/// or by first
/// [configuring the construction of a dense DFA](../dense/struct.Builder.html)
/// and then calling
/// [`DenseDFA::to_sparse`](../enum.DenseDFA.html#method.to_sparse).
pub mod sparse {
    pub use sparse_imp::*;
}
