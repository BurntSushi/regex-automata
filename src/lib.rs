/*!
TODO
*/

// #![deny(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub use crate::{
    bytes::{DeserializeError, SerializeError},
    classes::{
        ByteClassElementRanges, ByteClassElements, ByteClassIter,
        ByteClassSet, ByteClasses, InputUnit,
    },
    matching::{
        pattern_limit, Match, MatchError, MatchKind, MultiMatch, PatternID,
    },
    state_id::StateID,
};
#[cfg(feature = "alloc")]
pub use crate::{classes::ByteClassRepresentatives, config::SyntaxConfig};

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
