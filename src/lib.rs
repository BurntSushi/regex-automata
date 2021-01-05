/*!
TODO
*/

// #![deny(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]
// #![cfg_attr(not(feature = "alloc"), allow(dead_code))]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
pub use crate::config::SyntaxConfig;
pub use crate::{
    bytes::{DeserializeError, SerializeError},
    matching::{
        pattern_limit, Match, MatchError, MatchKind, MultiMatch, PatternID,
    },
    state_id::StateID,
};

#[macro_use]
mod macros;

mod bytes;
mod classes;
pub mod dfa;
mod matching;
pub mod prefilter;
mod state_id;
mod util;
mod word;

#[cfg(feature = "alloc")]
mod config;
#[cfg(feature = "alloc")]
pub mod nfa;
#[cfg(feature = "alloc")]
mod sparse_set;
