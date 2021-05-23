/*!
TODO
*/

#![allow(warnings)]
// #![deny(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(any(
    target_pointer_width = "16",
    target_pointer_width = "32",
    target_pointer_width = "64"
)))]
compile_error!("regex-automata currently not supported on non-{16,32,64}");

#[cfg(feature = "alloc")]
extern crate alloc;

pub use crate::{
    bytes::{DeserializeError, SerializeError},
    classes::{
        ByteClassElementRanges, ByteClassElements, ByteClassIter,
        ByteClassSet, ByteClasses, InputUnit,
    },
    id::{PatternID, PatternIDError, StateID, StateIDError},
    matching::{Match, MatchError, MatchKind, MultiMatch},
};
#[cfg(feature = "alloc")]
pub use crate::{classes::ByteClassRepresentatives, config::SyntaxConfig};

#[macro_use]
mod macros;

mod bytes;
mod classes;
pub mod dfa;
mod id;
mod matching;
pub mod prefilter;
mod util;

#[cfg(feature = "alloc")]
mod config;
#[cfg(feature = "alloc")]
pub mod nfa;
#[cfg(feature = "alloc")]
mod sparse_set;
