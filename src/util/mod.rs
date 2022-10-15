/*!
TODO
*/

pub mod alphabet;
#[cfg(feature = "alloc")]
pub mod captures;
pub mod iter;
pub mod lazy;
pub mod look;
#[cfg(feature = "alloc")]
pub mod pool;
pub mod prefilter;
pub mod primitives;
#[cfg(feature = "syntax")]
pub mod syntax;
pub mod wire;

#[cfg(any(feature = "dfa-build", feature = "hybrid"))]
pub(crate) mod determinize;
pub(crate) mod escape;
pub(crate) mod int;
pub(crate) mod memchr;
pub(crate) mod search;
#[cfg(feature = "alloc")]
pub(crate) mod sparse_set;
pub(crate) mod start;
pub(crate) mod unicode_data;
pub(crate) mod utf8;
