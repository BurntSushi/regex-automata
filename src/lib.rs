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

#[cfg(feature = "alloc")]
pub use crate::util::syntax::SyntaxConfig;
pub use crate::util::{
    bytes::{DeserializeError, SerializeError},
    id::PatternID,
    matchtypes::{Match, MatchError, MatchKind, MultiMatch},
};

#[macro_use]
mod macros;

pub mod dfa;
#[cfg(feature = "alloc")]
pub mod hybrid;
#[cfg(feature = "alloc")]
pub mod nfa;
pub mod util;
