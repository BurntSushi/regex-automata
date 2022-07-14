/*!
TODO
*/

pub mod alphabet;
#[cfg(feature = "alloc")]
pub mod captures;
pub mod iter;
pub mod prefilter;
pub mod primitives;

pub(crate) mod bytes;
#[cfg(feature = "alloc")]
pub(crate) mod determinize;
pub(crate) mod escape;
#[cfg(feature = "alloc")]
pub(crate) mod lazy;
pub(crate) mod search;
#[cfg(feature = "alloc")]
pub(crate) mod sparse_set;
pub(crate) mod start;
#[cfg(feature = "alloc")]
pub(crate) mod syntax;
pub(crate) mod utf8;
