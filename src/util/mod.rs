/*!
TODO
*/

pub mod alphabet;
#[cfg(feature = "alloc")]
pub mod captures;
pub mod iter;
pub mod look;
pub mod prefilter;
pub mod primitives;
pub mod search;
#[cfg(feature = "alloc")]
pub mod syntax;
pub mod wire;

#[cfg(feature = "alloc")]
pub(crate) mod determinize;
pub(crate) mod escape;
pub(crate) mod int;
#[cfg(feature = "alloc")]
pub(crate) mod lazy;
#[cfg(feature = "alloc")]
pub(crate) mod sparse_set;
pub(crate) mod start;
pub(crate) mod utf8;
