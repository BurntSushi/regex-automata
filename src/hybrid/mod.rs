pub use self::{
    error::{BuildError, CacheError},
    id::{LazyStateID, OverlappingState},
    lazy::{Builder, Cache, Config, InertDFA, DFA},
};

mod error;
mod id;
mod lazy;
pub mod regex;
mod search;
