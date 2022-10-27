/*!
This crate provides an "expert" API for executing regular expressions using
finite automata.

**WARNING**: This `0.2` release of `regex-automata` was published
before it was ready to unblock work elsewhere that needed some
of the new APIs in this release. At the time of writing, it is
strongly preferred that you continue using the
[`regex-automata 0.1`](https://docs.rs/regex-automata/0.1/regex_automata/)
release. Since this release represents an unfinished state, please do not
create issues for this release unless it's for a critical bug.
*/

#![no_std]
// #![deny(missing_docs)]
// Some intra-doc links are broken when certain features are disabled, so we
// only bleat about it when most (all?) features are enabled.
#![cfg_attr(all(std, nfa, dfa, hybrid), deny(rustdoc::broken_intra_doc_links))]
#![cfg_attr(
    not(all(std, nfa, dfa, hybrid)),
    allow(rustdoc::broken_intra_doc_links)
)]
// Kinda similar, but eliminating all of the dead code and unused import
// warnings for every feature combo is a fool's errand. Instead, we just
// suppress those, but still let them through in a common configuration when we
// build most of everything.
#![cfg_attr(not(all(std, nfa, dfa, hybrid)), allow(dead_code, unused_imports))]
#![warn(missing_debug_implementations)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

#[cfg(not(any(
    target_pointer_width = "16",
    target_pointer_width = "32",
    target_pointer_width = "64"
)))]
compile_error!("not supported on non-{16,32,64}, please file an issue");

#[cfg(any(test, feature = "std"))]
extern crate std;

#[cfg(feature = "alloc")]
extern crate alloc;

#[doc(inline)]
pub use crate::util::primitives::PatternID;
pub use crate::util::search::*;

#[macro_use]
mod macros;

#[cfg(any(feature = "dfa-search", feature = "dfa-onepass"))]
pub mod dfa;
#[cfg(feature = "hybrid")]
pub mod hybrid;
#[cfg(feature = "meta")]
pub mod meta;
#[cfg(feature = "nfa-thompson")]
pub mod nfa;
pub mod util;
