/*!
TODO
*/

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

pub mod alphabet;
pub mod id;
pub mod iter;
pub mod prefilter;

pub(crate) mod bytes;
#[cfg(feature = "alloc")]
pub(crate) mod determinize;
pub(crate) mod escape;
#[cfg(feature = "alloc")]
pub(crate) mod lazy;
pub(crate) mod matchtypes;
pub(crate) mod nonmax;
#[cfg(feature = "alloc")]
pub(crate) mod sparse_set;
pub(crate) mod start;
#[cfg(feature = "alloc")]
pub(crate) mod syntax;
pub(crate) mod utf8;
