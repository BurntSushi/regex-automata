use core::convert::{Infallible, TryFrom};

/// An identifier for a regex pattern.
///
/// The identifier for a pattern corresponds to its relative position among
/// other patterns in a single finite state machine. Namely, when building
/// a multi-pattern regex engine, one must supply a sequence of patterns to
/// match. The position (starting at 0) of each pattern in that sequence
/// represents its identifier. This identifier is in turn used to identify and
/// report matches of that pattern in various APIs.
///
/// A pattern ID is guaranteed to be representable by a `usize`. Similarly,
/// the number of patterns in any regex engine in this crate is similarly
/// guaranteed to be representable by a `usize`. This applies to regex engines
/// that have been deserialized; a deserialization error will be returned if
/// it contains pattern IDs that violate these requirements in your current
/// environment.
///
/// # Representation
///
/// This type is always represented internally by a `u32` and is marked as
/// `repr(transparent)`. Thus, this type always has the same representation as
/// a `u32`.
///
/// # Indexing
///
/// For convenience, callers may use a `PatternID` to index slices.
///
/// # Safety
///
/// While a `PatternID` is meant to guarantee that its value fits into `usize`
/// (while using a possibly smaller representation than `usize` on some
/// targets), callers must not rely on this property for safety. Callers may
/// choose to rely on this property for correctness however.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct PatternID(u32);

impl PatternID {
    /// The maximum pattern ID value, represented as a `usize`.
    #[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
    pub const MAX: usize = core::u32::MAX as usize - 1;

    /// The maximum pattern ID value, represented as a `usize`.
    #[cfg(target_pointer_width = "16")]
    pub const MAX: usize = core::usize::MAX - 1;

    /// The number of bytes that a single `PatternID` uses in memory.
    pub const SIZE: usize = core::mem::size_of::<PatternID>();

    /// Create a new pattern ID.
    ///
    /// If the given identifier exceeds [`PatternID::MAX`], then this returns
    /// an error.
    pub fn new(id: usize) -> Result<PatternID, PatternIDError> {
        PatternID::try_from(id)
    }

    /// Create a new pattern ID without checking whether the given value
    /// exceeds [`PatternID::MAX`].
    ///
    /// While this is unchecked, providing an incorrect value must never
    /// sacrifice memory safety, as documented above.
    pub fn new_unchecked(id: usize) -> PatternID {
        PatternID(id as u32)
    }

    /// Return this pattern ID as a `usize`.
    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }

    /// Return the internal u32 of this pattern ID.
    pub fn as_u32(&self) -> u32 {
        self.0
    }

    /// Decode this pattern ID from the bytes given using the native endian
    /// byte order for the current target.
    ///
    /// If the decoded integer is not representable as a pattern ID for the
    /// current target, then this returns an error.
    pub fn from_ne_bytes(bytes: [u8; 4]) -> Result<PatternID, PatternIDError> {
        let id = u32::from_ne_bytes(bytes);
        if id > PatternID::MAX as u32 {
            return Err(PatternIDError { attempted: id as u64 });
        }
        Ok(PatternID::new_unchecked(id as usize))
    }

    /// Decode this pattern ID from the bytes given using the native endian
    /// byte order for the current target.
    ///
    /// This is analogous to [`PatternID::new_unchecked`] in that is does not
    /// check whether the decoded integer is representable as a pattern ID.
    pub fn from_ne_bytes_unchecked(bytes: [u8; 4]) -> PatternID {
        PatternID::new_unchecked(u32::from_ne_bytes(bytes) as usize)
    }

    /// Return the underlying pattern ID integer as raw bytes in native endian
    /// format.
    pub fn to_ne_bytes(&self) -> [u8; 4] {
        self.0.to_ne_bytes()
    }
}

impl<T> core::ops::Index<PatternID> for [T] {
    type Output = T;

    #[inline]
    fn index(&self, index: PatternID) -> &T {
        &self[index.as_usize()]
    }
}

impl<T> core::ops::IndexMut<PatternID> for [T] {
    #[inline]
    fn index_mut(&mut self, index: PatternID) -> &mut T {
        &mut self[index.as_usize()]
    }
}

impl TryFrom<usize> for PatternID {
    type Error = PatternIDError;

    fn try_from(id: usize) -> Result<PatternID, PatternIDError> {
        if id > PatternID::MAX {
            return Err(PatternIDError { attempted: id as u64 });
        }
        Ok(PatternID::new_unchecked(id))
    }
}

impl TryFrom<u8> for PatternID {
    type Error = Infallible;

    fn try_from(id: u8) -> Result<PatternID, Infallible> {
        Ok(PatternID::new_unchecked(id as usize))
    }
}

impl TryFrom<u16> for PatternID {
    type Error = PatternIDError;

    fn try_from(id: u16) -> Result<PatternID, PatternIDError> {
        if id as u32 > PatternID::MAX as u32 {
            return Err(PatternIDError { attempted: id as u64 });
        }
        Ok(PatternID::new_unchecked(id as usize))
    }
}

impl TryFrom<u32> for PatternID {
    type Error = PatternIDError;

    fn try_from(id: u32) -> Result<PatternID, PatternIDError> {
        if id > PatternID::MAX as u32 {
            return Err(PatternIDError { attempted: id as u64 });
        }
        Ok(PatternID::new_unchecked(id as usize))
    }
}

impl TryFrom<u64> for PatternID {
    type Error = PatternIDError;

    fn try_from(id: u64) -> Result<PatternID, PatternIDError> {
        if id > PatternID::MAX as u64 {
            return Err(PatternIDError { attempted: id });
        }
        Ok(PatternID::new_unchecked(id as usize))
    }
}

/// This error occurs when a pattern ID could not be constructed.
///
/// This occurs when given an integer exceeding the maximum pattern ID value.
///
/// When the `std` feature is enabled, this implements the `Error` trait.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PatternIDError {
    attempted: u64,
}

impl PatternIDError {
    /// Returns the value that failed to constructed a pattern ID.
    pub fn attempted(&self) -> u64 {
        self.attempted
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PatternIDError {}

impl core::fmt::Display for PatternIDError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(
            f,
            "failed to create PatternID from {}, which exceeds {}",
            self.attempted(),
            PatternID::MAX,
        )
    }
}

/// An identifier for a state in a regex engine.
///
/// A state ID is guaranteed to be representable by a `usize`. Similarly, the
/// number of states in any regex engine in this crate is similarly guaranteed
/// to be representable by a `usize`. This applies to regex engines that have
/// been deserialized; a deserialization error will be returned if it contains
/// state IDs that violate these requirements in your current environment.
///
/// # Representation
///
/// This type is always represented internally by a `u32` and is marked as
/// `repr(transparent)`. Thus, this type always has the same representation as
/// a `u32`.
///
/// # Indexing
///
/// For convenience, callers may use a `StateID` to index slices.
///
/// # Safety
///
/// While a `StateID` is meant to guarantee that its value fits into `usize`
/// (while using a possibly smaller representation than `usize` on some
/// targets), callers must not rely on this property for safety. Callers may
/// choose to rely on this property for correctness however.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct StateID(u32);

impl StateID {
    /// The maximum state ID value, represented as a `usize`.
    #[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
    pub const MAX: usize = core::u32::MAX as usize - 1;

    /// The maximum state ID value, represented as a `usize`.
    #[cfg(target_pointer_width = "16")]
    pub const MAX: usize = core::usize::MAX - 1;

    /// The number of bytes that a single `StateID` uses in memory.
    pub const SIZE: usize = core::mem::size_of::<StateID>();

    /// A unique state ID that always corresponds to the dead state for DFAs.
    pub(crate) const DEAD: StateID = StateID(0);

    /// Create a new state ID.
    ///
    /// If the given identifier exceeds [`StateID::MAX`], then this returns
    /// an error.
    pub fn new(id: usize) -> Result<StateID, StateIDError> {
        StateID::try_from(id)
    }

    /// Create a new pattern ID without checking whether the given value
    /// exceeds [`PatternID::MAX`].
    ///
    /// While this is unchecked, providing an incorrect value must never
    /// sacrifice memory safety, as documented above.
    pub fn new_unchecked(id: usize) -> StateID {
        StateID(id as u32)
    }

    /// Return this pattern ID as a `usize`.
    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }

    /// Return the internal u32 of this state ID.
    pub fn as_u32(&self) -> u32 {
        self.0
    }

    /// Decode this state ID from the bytes given using the native endian byte
    /// order for the current target.
    ///
    /// If the decoded integer is not representable as a state ID for the
    /// current target, then this returns an error.
    pub fn from_ne_bytes(bytes: [u8; 4]) -> Result<StateID, StateIDError> {
        let id = u32::from_ne_bytes(bytes);
        if id > StateID::MAX as u32 {
            return Err(StateIDError { attempted: id as u64 });
        }
        Ok(StateID::new_unchecked(id as usize))
    }

    /// Decode this state ID from the bytes given using the native endian
    /// byte order for the current target.
    ///
    /// This is analogous to [`StateID::new_unchecked`] in that is does not
    /// check whether the decoded integer is representable as a state ID.
    pub fn from_ne_bytes_unchecked(bytes: [u8; 4]) -> StateID {
        StateID::new_unchecked(u32::from_ne_bytes(bytes) as usize)
    }

    /// Return the underlying state ID integer as raw bytes in native endian
    /// format.
    pub fn to_ne_bytes(&self) -> [u8; 4] {
        self.0.to_ne_bytes()
    }
}

impl<T> core::ops::Index<StateID> for [T] {
    type Output = T;

    #[inline]
    fn index(&self, index: StateID) -> &T {
        &self[index.as_usize()]
    }
}

impl<T> core::ops::IndexMut<StateID> for [T] {
    #[inline]
    fn index_mut(&mut self, index: StateID) -> &mut T {
        &mut self[index.as_usize()]
    }
}

impl TryFrom<usize> for StateID {
    type Error = StateIDError;

    fn try_from(id: usize) -> Result<StateID, StateIDError> {
        if id > StateID::MAX {
            return Err(StateIDError { attempted: id as u64 });
        }
        Ok(StateID::new_unchecked(id))
    }
}

impl TryFrom<u8> for StateID {
    type Error = Infallible;

    fn try_from(id: u8) -> Result<StateID, Infallible> {
        Ok(StateID::new_unchecked(id as usize))
    }
}

impl TryFrom<u16> for StateID {
    type Error = StateIDError;

    fn try_from(id: u16) -> Result<StateID, StateIDError> {
        if id as u32 > StateID::MAX as u32 {
            return Err(StateIDError { attempted: id as u64 });
        }
        Ok(StateID::new_unchecked(id as usize))
    }
}

impl TryFrom<u32> for StateID {
    type Error = StateIDError;

    fn try_from(id: u32) -> Result<StateID, StateIDError> {
        if id > StateID::MAX as u32 {
            return Err(StateIDError { attempted: id as u64 });
        }
        Ok(StateID::new_unchecked(id as usize))
    }
}

impl TryFrom<u64> for StateID {
    type Error = StateIDError;

    fn try_from(id: u64) -> Result<StateID, StateIDError> {
        if id > StateID::MAX as u64 {
            return Err(StateIDError { attempted: id });
        }
        Ok(StateID::new_unchecked(id as usize))
    }
}

/// This error occurs when a state ID could not be constructed.
///
/// This occurs when given an integer exceeding the maximum state ID value.
///
/// When the `std` feature is enabled, this implements the `Error` trait.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateIDError {
    attempted: u64,
}

impl StateIDError {
    /// Returns the value that failed to constructed a state ID.
    pub fn attempted(&self) -> u64 {
        self.attempted
    }
}

#[cfg(feature = "std")]
impl std::error::Error for StateIDError {}

impl core::fmt::Display for StateIDError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(
            f,
            "failed to create StateID from {}, which exceeds {}",
            self.attempted(),
            StateID::MAX,
        )
    }
}
