#[cfg(feature = "nfa-backtrack")]
pub mod backtrack;
mod builder;
#[cfg(feature = "syntax")]
mod compiler;
mod error;
#[cfg(feature = "syntax")]
mod literal_trie;
#[cfg(feature = "syntax")]
mod map;
mod nfa;
#[cfg(feature = "nfa-pikevm")]
pub mod pikevm;
#[cfg(feature = "syntax")]
mod range_trie;

pub use self::{
    builder::Builder,
    error::BuildError,
    nfa::{PatternIter, SparseTransitions, State, Transition, NFA},
};
#[cfg(feature = "syntax")]
pub use compiler::{Compiler, Config};
