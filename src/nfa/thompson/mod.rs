#[cfg(feature = "nfa-pikevm")]
pub mod backtrack;
mod builder;
#[cfg(feature = "syntax")]
mod compiler;
mod error;
#[cfg(feature = "syntax")]
mod literal_trie;
mod map;
mod nfa;
#[cfg(feature = "nfa-pikevm")]
pub mod pikevm;
mod range_trie;

pub use self::{
    builder::Builder,
    error::Error,
    nfa::{PatternIter, SparseTransitions, State, Transition, NFA},
};
#[cfg(feature = "syntax")]
pub use compiler::{Compiler, Config};
