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

#![allow(warnings)]
// #![deny(missing_docs)]
#![doc(test(attr(deny(warnings))))]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

#[cfg(not(any(
    target_pointer_width = "16",
    target_pointer_width = "32",
    target_pointer_width = "64"
)))]
compile_error!("not supported on non-{16,32,64}, please file an issue");

#[cfg(feature = "alloc")]
extern crate alloc;

#[doc(inline)]
pub use crate::util::primitives::PatternID;
#[cfg(feature = "alloc")]
pub use crate::util::syntax::SyntaxConfig;
pub use crate::util::{
    search::{
        HalfMatch, Input, Match, MatchError, MatchKind, PatternSet,
        PatternSetIter, Span,
    },
    wire::{DeserializeError, SerializeError},
};

#[macro_use]
mod macros;

pub mod dfa;
#[cfg(feature = "alloc")]
pub mod hybrid;
#[cfg(feature = "alloc")]
pub mod nfa;
pub mod util;
