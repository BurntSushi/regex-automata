use core::fmt::Debug;
use core::hash::Hash;
use core::mem::size_of;

use crate::bytes::{self, Endian};

/// Return the unique identifier for a DFA's dead state in the chosen
/// representation indicated by `S`.
pub fn dead_id<S: StateID>() -> S {
    S::from_usize(0)
}

/// Ensure that callers cannot implement `StateID` by making an umplementable
/// trait its super trait.
///
/// While this isn't strictly necessary since `StateID` is not safe anyway,
/// it makes it a bit easier to reason about correctness with a small set of
/// known POD types. It's also the conservative choice and reduces the chance
/// for screw ups.
pub trait Sealed {}
impl Sealed for usize {}
impl Sealed for u8 {}
impl Sealed for u16 {}
#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
impl Sealed for u32 {}
#[cfg(target_pointer_width = "64")]
impl Sealed for u64 {}

/// A trait describing the representation of a DFA's state identifier.
///
/// The purpose of this trait is to safely express both the possible state
/// identifier representations that can be used in a DFA and to convert between
/// state identifier representations and types that can be used to efficiently
/// index memory (such as `usize`).
///
/// In general, one should not need to implement this trait explicitly. In
/// particular, this crate provides implementations for `u8`, `u16`, `u32`,
/// `u64` and `usize`. (`u32` and `u64` are only provided for targets that can
/// represent all corresponding values in a `usize`.)
///
/// # Safety
///
/// This trait is unsafe because the correctness of its implementations may be
/// relied upon by other unsafe code. For example, one possible way to
/// implement this trait incorrectly would be to return a maximum identifier
/// in `max_id` that is greater than the real maximum identifier. This will
/// likely result in wrap-on-overflow semantics in release mode, which can in
/// turn produce incorrect state identifiers. Those state identifiers may then
/// in turn access out-of-bounds memory in a DFA's search routine, where bounds
/// checks are explicitly elided for performance reasons.
pub unsafe trait StateID:
    Sealed + Clone + Copy + Debug + Eq + Hash + PartialEq + PartialOrd + Ord
{
    /// Convert from a `usize` to this implementation's representation.
    ///
    /// Implementors may assume that `n <= Self::max_id`. That is, implementors
    /// do not need to check whether `n` can fit inside this implementation's
    /// representation.
    fn from_usize(n: usize) -> Self;

    /// Convert this implementation's representation to a `usize`.
    ///
    /// Implementors must not return a `usize` value greater than
    /// `Self::max_id` and must not permit overflow when converting between the
    /// implementor's representation and `usize`. In general, the preferred
    /// way for implementors to achieve this is to simply not provide
    /// implementations of `StateID` that cannot fit into the target platform's
    /// `usize`.
    fn as_usize(self) -> usize;

    /// Return the maximum state identifier supported by this representation.
    ///
    /// Implementors must return a correct bound. Doing otherwise may result
    /// in memory unsafety.
    fn max_id() -> usize;

    /// Read a single state identifier from the given slice of bytes in native
    /// endian format.
    ///
    /// Implementors may assume that the given slice has length at least
    /// `size_of::<Self>()`.
    fn read_bytes(slice: &[u8]) -> Self;

    /// Write this state identifier to the given slice of bytes in native
    /// endian format.
    ///
    /// Implementors may assume that the given slice has length at least
    /// `size_of::<Self>()`.
    fn write_bytes(self, slice: &mut [u8]);
}

unsafe impl StateID for usize {
    #[inline]
    fn from_usize(n: usize) -> usize {
        n
    }

    #[inline]
    fn as_usize(self) -> usize {
        self
    }

    #[inline]
    fn max_id() -> usize {
        ::core::usize::MAX
    }

    #[inline]
    fn read_bytes(slice: &[u8]) -> Self {
        #[cfg(target_pointer_width = "16")]
        {
            bytes::read_u16(slice) as usize
        }
        #[cfg(target_pointer_width = "32")]
        {
            bytes::read_u32(slice) as usize
        }
        #[cfg(target_pointer_width = "64")]
        {
            bytes::read_u64(slice) as usize
        }
    }

    #[inline]
    fn write_bytes(self, slice: &mut [u8]) {
        #[cfg(target_pointer_width = "16")]
        {
            bytes::NE::write_u16(self as u16, slice)
        }
        #[cfg(target_pointer_width = "32")]
        {
            bytes::NE::write_u32(self as u32, slice)
        }
        #[cfg(target_pointer_width = "64")]
        {
            bytes::NE::write_u64(self as u64, slice)
        }
    }
}

unsafe impl StateID for u8 {
    #[inline]
    fn from_usize(n: usize) -> u8 {
        n as u8
    }

    #[inline]
    fn as_usize(self) -> usize {
        self as usize
    }

    #[inline]
    fn max_id() -> usize {
        ::core::u8::MAX as usize
    }

    #[inline]
    fn read_bytes(slice: &[u8]) -> Self {
        slice[0]
    }

    #[inline]
    fn write_bytes(self, slice: &mut [u8]) {
        slice[0] = self;
    }
}

unsafe impl StateID for u16 {
    #[inline]
    fn from_usize(n: usize) -> u16 {
        n as u16
    }

    #[inline]
    fn as_usize(self) -> usize {
        self as usize
    }

    #[inline]
    fn max_id() -> usize {
        ::core::u16::MAX as usize
    }

    #[inline]
    fn read_bytes(slice: &[u8]) -> Self {
        bytes::read_u16(slice)
    }

    #[inline]
    fn write_bytes(self, slice: &mut [u8]) {
        bytes::NE::write_u16(self, slice)
    }
}

#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
unsafe impl StateID for u32 {
    #[inline]
    fn from_usize(n: usize) -> u32 {
        n as u32
    }

    #[inline]
    fn as_usize(self) -> usize {
        self as usize
    }

    #[inline]
    fn max_id() -> usize {
        ::core::u32::MAX as usize
    }

    #[inline]
    fn read_bytes(slice: &[u8]) -> Self {
        bytes::read_u32(slice)
    }

    #[inline]
    fn write_bytes(self, slice: &mut [u8]) {
        bytes::NE::write_u32(self, slice)
    }
}

#[cfg(target_pointer_width = "64")]
unsafe impl StateID for u64 {
    #[inline]
    fn from_usize(n: usize) -> u64 {
        n as u64
    }

    #[inline]
    fn as_usize(self) -> usize {
        self as usize
    }

    #[inline]
    fn max_id() -> usize {
        ::core::u64::MAX as usize
    }

    #[inline]
    fn read_bytes(slice: &[u8]) -> Self {
        bytes::read_u64(slice)
    }

    #[inline]
    fn write_bytes(self, slice: &mut [u8]) {
        bytes::NE::write_u64(self, slice)
    }
}
