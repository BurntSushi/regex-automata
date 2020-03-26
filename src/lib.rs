/*!
TODO
*/

#![allow(warnings)]
// #![deny(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]

pub use crate::bytes::{DeserializeError, SerializeError};
pub use crate::config::SyntaxConfig;
pub use crate::matching::{Match, MatchKind, MultiMatch, NoMatch, PatternID};
pub use crate::state_id::StateID;

#[macro_use]
mod macros;

pub mod dfa;
#[cfg(feature = "std")]
pub mod nfa;
pub mod prefilter;

mod bytes;
mod classes;
mod config;
mod matching;
#[cfg(feature = "std")]
mod sparse_set;
mod state_id;
mod util;
mod word;
