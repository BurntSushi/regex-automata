use std::fmt::Debug;
use std::hash::Hash;

use error::{Error, Result};

/// Return the unique identifier for a DFA's dead state in the chosen
/// representation indicated by `S`.
pub fn dead_id<S: StateID>() -> S {
    S::from_usize(0)
}

/// Check that the premultiplication of the given state identifier can fit into
/// the representation indicated by `S`. If it cannot, or if it overflows
/// `usize` itself, then an error is returned.
pub fn premultiply_overflow_error<S: StateID>(
    last_state: S,
    alphabet_len: usize,
) -> Result<()> {
    let requested_max = match last_state.to_usize().checked_mul(alphabet_len) {
        Some(requested_max) => requested_max,
        None => return Err(Error::premultiply_overflow(0, 0)),
    };
    if requested_max > S::max_id() {
        return Err(Error::premultiply_overflow(S::max_id(), requested_max));
    }
    Ok(())
}

/// Allocate the next sequential identifier for a fresh state given the
/// previously constructed state identified by `current`. If the next
/// sequential identifier would overflow `usize` or the chosen representation
/// indicated by `S`, then an error is returned.
pub fn next_state_id<S: StateID>(current: S) -> Result<S> {
    let next = match current.to_usize().checked_add(1) {
        Some(next) => next,
        None => return Err(Error::state_id_overflow(::std::usize::MAX)),
    };
    if next > S::max_id() {
        return Err(Error::state_id_overflow(S::max_id()));
    }
    Ok(S::from_usize(next))
}

pub trait StateID: Clone + Copy + Debug + Eq + Hash + PartialEq + PartialOrd + Ord {
    fn from_usize(n: usize) -> Self;
    fn to_usize(self) -> usize;
    fn max_id() -> usize;
}

impl StateID for usize {
    #[inline]
    fn from_usize(n: usize) -> usize { n }

    #[inline]
    fn to_usize(self) -> usize { self }

    #[inline]
    fn max_id() -> usize { ::std::usize::MAX }
}

impl StateID for u8 {
    #[inline]
    fn from_usize(n: usize) -> u8 { n as u8 }

    #[inline]
    fn to_usize(self) -> usize { self as usize }

    #[inline]
    fn max_id() -> usize { ::std::u8::MAX as usize }
}

impl StateID for u16 {
    #[inline]
    fn from_usize(n: usize) -> u16 { n as u16 }

    #[inline]
    fn to_usize(self) -> usize { self as usize }

    #[inline]
    fn max_id() -> usize { ::std::u16::MAX as usize }
}

#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
impl StateID for u32 {
    #[inline]
    fn from_usize(n: usize) -> u32 { n as u32 }

    #[inline]
    fn to_usize(self) -> usize { self as usize }

    #[inline]
    fn max_id() -> usize { ::std::u32::MAX as usize }
}

#[cfg(target_pointer_width = "64")]
impl StateID for u64 {
    #[inline]
    fn from_usize(n: usize) -> u64 { n as u64 }

    #[inline]
    fn to_usize(self) -> usize { self as usize }

    #[inline]
    fn max_id() -> usize { ::std::u64::MAX as usize }
}
