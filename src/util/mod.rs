/*!
TODO
*/

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

pub mod alphabet;
pub(crate) mod bytes;
#[cfg(feature = "alloc")]
pub(crate) mod determinize;
pub(crate) mod escape;
pub mod id;
pub mod iter;
#[cfg(feature = "alloc")]
pub(crate) mod lazy;
pub(crate) mod matchtypes;
pub(crate) mod nonmax;
pub mod prefilter;
#[cfg(feature = "alloc")]
pub(crate) mod sparse_set;
pub(crate) mod start;
#[cfg(feature = "alloc")]
pub(crate) mod syntax;
pub(crate) mod utf8;
