use core::borrow::Borrow;

use alloc::sync::Arc;

use crate::{
    hybrid::error::{BuildError, CacheError},
    nfa::thompson,
    util::{alphabet::ByteSet, matchtypes::MatchKind},
};

pub use self::lazy::{Builder, Cache, Config, InertDFA, DFA};

mod error;
mod id;
mod lazy;
pub mod regex;
mod search;
