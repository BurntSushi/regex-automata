#![allow(warnings)]

#[cfg(feature = "dfa-onepass")]
use crate::dfa::onepass;
#[cfg(feature = "hybrid")]
use crate::hybrid;
#[cfg(feature = "nfa-backtrack")]
use crate::nfa::thompson::backtrack;

pub(crate) use self::strategy::Strategy;
pub use self::{
    error::BuildError,
    regex::{Builder, Cache, CapturesMatches, Config, FindMatches, Regex},
};

mod error;
mod regex;
mod strategy;
mod wrappers;
