use core::{convert::TryFrom, num::NonZeroUsize};

/// A `usize` that can never be `usize::MAX`.
///
/// This is similar to `core::num::NonZeroUsize`, but instead of not permitting
/// a zero value, this does not permit a max value.
///
/// This is useful in certain contexts where one wants to optimize the memory
/// usage of things that contain match offsets. Namely, since Rust slices
/// are guaranteed to never have a length exceeding `isize::MAX`, we can use
/// `usize::MAX` as a sentinel to indicate that no match was found. Indeed,
/// types like `Option<NonMaxUsize>` have exactly the same size in memory as a
/// `usize`.
///
/// This type is defined to be `repr(transparent)` for
/// `core::num::NonZeroUsize`, which is in turn defined to be
/// `repr(transparent)` for `usize`.
#[derive(Clone, Copy, Eq, Hash, PartialEq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct NonMaxUsize(NonZeroUsize);

impl NonMaxUsize {
    /// Create a new `NonMaxUsize` from the given value.
    ///
    /// This returns `None` only when the given value is equal to `usize::MAX`.
    #[inline]
    pub fn new(value: usize) -> Option<NonMaxUsize> {
        NonZeroUsize::new(value.wrapping_add(1)).map(NonMaxUsize)
    }

    /// Return the underlying `usize` value. The returned value is guaranteed
    /// to not equal `usize::MAX`.
    #[inline]
    pub fn get(self) -> usize {
        self.0.get().wrapping_sub(1)
    }
}

// We provide our own Debug impl because seeing the internal repr can be quite
// surprising if you aren't expecting it. e.g., 'NonMaxUsize(5)' vs just '5'.
impl core::fmt::Debug for NonMaxUsize {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "{:?}", self.get())
    }
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, PartialEq, PartialOrd, Ord,
)]
#[repr(transparent)]
pub struct SmallIndex(u32);

impl SmallIndex {
    /// The maximum index value.
    #[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
    pub const MAX: SmallIndex =
        SmallIndex::new_unchecked(core::i32::MAX as usize - 1);

    /// The maximum index value.
    #[cfg(target_pointer_width = "16")]
    pub const MAX: SmallIndex =
        SmallIndex::new_unchecked(core::isize::MAX - 1);

    /// The total number of values that can be represented as a small index.
    pub const LIMIT: usize = SmallIndex::MAX.as_usize() + 1;

    /// The zero index value.
    pub const ZERO: SmallIndex = SmallIndex::new_unchecked(0);

    /// The number of bytes that a single small index uses in memory.
    pub const SIZE: usize = core::mem::size_of::<SmallIndex>();

    /// Create a new small index.
    ///
    /// If the given index exceeds [`SmallIndex::MAX`], then this returns
    /// an error.
    #[inline]
    pub fn new(index: usize) -> Result<SmallIndex, SmallIndexError> {
        SmallIndex::try_from(index)
    }

    /// Create a new small index without checking whether the given value
    /// exceeds [`SmallIndex::MAX`].
    ///
    /// Using this routine with an invalid index value will result in
    /// unspecified behavior, but *not* undefined behavior. In particular, an
    /// invalid index value is likely to cause panics or possibly even silent
    /// logical errors.
    ///
    /// Callers must never rely on a `SmallIndex` to be within a certain range
    /// for memory safety.
    #[inline]
    pub const fn new_unchecked(index: usize) -> SmallIndex {
        SmallIndex(index as u32)
    }

    /// Like [`SmallIndex::new`], but panics if the given index is not valid.
    #[inline]
    pub fn must(index: usize) -> SmallIndex {
        SmallIndex::new(index).expect("invalid small index")
    }

    /// Return this small index as a `usize`. This is guaranteed to never
    /// overflow `usize`.
    #[inline]
    pub const fn as_usize(&self) -> usize {
        self.0 as usize
    }

    /// Return the internal `u32` of this small index. This is guaranteed to
    /// never overflow `u32`.
    #[inline]
    pub const fn as_u32(&self) -> u32 {
        self.0
    }

    /// Return the internal `u32` of this small index represented as an `i32`.
    /// This is guaranteed to never overflow an `i32`.
    #[inline]
    pub const fn as_i32(&self) -> i32 {
        self.0 as i32
    }

    /// Returns one more than this small index as a usize.
    ///
    /// Since a small index has constraints on its maximum value, adding `1` to
    /// it will always fit in a `usize`, `u32` and a `i32`.
    #[inline]
    pub fn one_more(&self) -> usize {
        self.0 as usize + 1
    }

    /// Decode this small index from the bytes given using the native endian
    /// byte order for the current target.
    ///
    /// If the decoded integer is not representable as a small index for the
    /// current target, then this returns an error.
    #[inline]
    pub fn from_ne_bytes(
        bytes: [u8; 4],
    ) -> Result<SmallIndex, SmallIndexError> {
        let id = u32::from_ne_bytes(bytes);
        if id > SmallIndex::MAX.as_u32() {
            return Err(SmallIndexError { attempted: id as u64 });
        }
        Ok(SmallIndex::new_unchecked(id as usize))
    }

    /// Decode this small index from the bytes given using the native endian
    /// byte order for the current target.
    ///
    /// This is analogous to [`SmallIndex::new_unchecked`] in that is does not
    /// check whether the decoded integer is representable as a small index.
    #[inline]
    pub fn from_ne_bytes_unchecked(bytes: [u8; 4]) -> SmallIndex {
        SmallIndex::new_unchecked(u32::from_ne_bytes(bytes) as usize)
    }

    /// Return the underlying small index integer as raw bytes in native endian
    /// format.
    #[inline]
    pub fn to_ne_bytes(&self) -> [u8; 4] {
        self.0.to_ne_bytes()
    }

    /// Returns an iterator over all small indices from 0 up to and not
    /// including the given length.
    ///
    /// If the given length exceeds [`SmallIndex::LIMIT`], then this panics.
    #[cfg(feature = "alloc")]
    pub(crate) fn iter(len: usize) -> SmallIndexIter {
        SmallIndexIter::new(len)
    }
}

impl<T> core::ops::Index<SmallIndex> for [T] {
    type Output = T;

    #[inline]
    fn index(&self, index: SmallIndex) -> &T {
        &self[index.as_usize()]
    }
}

impl<T> core::ops::IndexMut<SmallIndex> for [T] {
    #[inline]
    fn index_mut(&mut self, index: SmallIndex) -> &mut T {
        &mut self[index.as_usize()]
    }
}

#[cfg(feature = "alloc")]
impl<T> core::ops::Index<SmallIndex> for Vec<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: SmallIndex) -> &T {
        &self[index.as_usize()]
    }
}

#[cfg(feature = "alloc")]
impl<T> core::ops::IndexMut<SmallIndex> for Vec<T> {
    #[inline]
    fn index_mut(&mut self, index: SmallIndex) -> &mut T {
        &mut self[index.as_usize()]
    }
}

impl From<u8> for SmallIndex {
    fn from(index: u8) -> SmallIndex {
        SmallIndex::new_unchecked(index as usize)
    }
}

impl TryFrom<u16> for SmallIndex {
    type Error = SmallIndexError;

    fn try_from(index: u16) -> Result<SmallIndex, SmallIndexError> {
        if index as u32 > SmallIndex::MAX.as_u32() {
            return Err(SmallIndexError { attempted: index as u64 });
        }
        Ok(SmallIndex::new_unchecked(index as usize))
    }
}

impl TryFrom<u32> for SmallIndex {
    type Error = SmallIndexError;

    fn try_from(index: u32) -> Result<SmallIndex, SmallIndexError> {
        if index > SmallIndex::MAX.as_u32() {
            return Err(SmallIndexError { attempted: index as u64 });
        }
        Ok(SmallIndex::new_unchecked(index as usize))
    }
}

impl TryFrom<u64> for SmallIndex {
    type Error = SmallIndexError;

    fn try_from(index: u64) -> Result<SmallIndex, SmallIndexError> {
        if index > SmallIndex::MAX.as_u32() as u64 {
            return Err(SmallIndexError { attempted: index });
        }
        Ok(SmallIndex::new_unchecked(index as usize))
    }
}

impl TryFrom<usize> for SmallIndex {
    type Error = SmallIndexError;

    fn try_from(index: usize) -> Result<SmallIndex, SmallIndexError> {
        if index > SmallIndex::MAX.as_usize() {
            return Err(SmallIndexError { attempted: index as u64 });
        }
        Ok(SmallIndex::new_unchecked(index))
    }
}

#[cfg(test)]
impl quickcheck::Arbitrary for SmallIndex {
    fn arbitrary(gen: &mut quickcheck::Gen) -> SmallIndex {
        use core::cmp::max;

        let id = max(i32::MIN + 1, i32::arbitrary(gen)).abs();
        if id > SmallIndex::MAX.as_i32() {
            SmallIndex::MAX
        } else {
            SmallIndex::new(usize::try_from(id).unwrap()).unwrap()
        }
    }
}

/// This error occurs when a small index could not be constructed.
///
/// This occurs when given an integer exceeding the maximum small index value.
///
/// When the `std` feature is enabled, this implements the `Error` trait.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SmallIndexError {
    attempted: u64,
}

impl SmallIndexError {
    /// Returns the value that could not be converted to a small index.
    pub fn attempted(&self) -> u64 {
        self.attempted
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SmallIndexError {}

impl core::fmt::Display for SmallIndexError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(
            f,
            "failed to create smallindex from {:?}, which exceeds {:?}",
            self.attempted(),
            SmallIndex::MAX,
        )
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SmallIndexIter {
    rng: core::ops::Range<usize>,
}

impl SmallIndexIter {
    #[cfg(feature = "alloc")]
    fn new(len: usize) -> SmallIndexIter {
        assert!(
            len <= SmallIndex::LIMIT,
            "cannot create iterator with small indices when number of \
             elements exceed {:?}",
            SmallIndex::LIMIT,
        );
        SmallIndexIter { rng: 0..len }
    }
}

impl Iterator for SmallIndexIter {
    type Item = SmallIndex;

    fn next(&mut self) -> Option<SmallIndex> {
        if self.rng.start >= self.rng.end {
            return None;
        }
        let next_id = self.rng.start + 1;
        let id = core::mem::replace(&mut self.rng.start, next_id);
        // new_unchecked is OK since we asserted that the number of
        // elements in this iterator will fit in an ID at construction.
        Some(SmallIndex::new_unchecked(id))
    }
}

/// An iterator adapter that is like std::iter::Enumerate, but attaches "small
/// indices" instead. It requires `ExactSizeIterator`. At construction, it
/// ensures that the index of each element in the iterator is representable in
/// the corresponding "small index" type.
///
/// To use this type, import IteratorIndexExt and use `with_small_indices` on
/// any iterator. (`with_pattern_ids` and `with_state_ids` are also available.)
#[derive(Clone, Debug)]
pub(crate) struct WithSmallIndexIter<I> {
    it: I,
    ids: SmallIndexIter,
}

impl<I: Iterator + ExactSizeIterator> WithSmallIndexIter<I> {
    fn new(it: I) -> WithSmallIndexIter<I> {
        let ids = SmallIndex::iter(it.len());
        WithSmallIndexIter { it, ids }
    }
}

impl<I: Iterator + ExactSizeIterator> Iterator for WithSmallIndexIter<I> {
    type Item = (SmallIndex, I::Item);

    fn next(&mut self) -> Option<(SmallIndex, I::Item)> {
        let item = self.it.next()?;
        // Number of elements in this iterator must match, according
        // to contract of ExactSizeIterator.
        let id = self.ids.next().unwrap();
        Some((id, item))
    }
}

macro_rules! define_index_type {
    ($name:ident, $err:ident, $iter:ident, $withiter:ident) => {
        #[derive(
            Clone, Copy, Debug, Default, Eq, Hash, PartialEq, PartialOrd, Ord,
        )]
        #[repr(transparent)]
        pub struct $name(SmallIndex);

        impl $name {
            /// The maximum value.
            pub const MAX: $name = $name(SmallIndex::MAX);

            /// The total number of values that can be represented.
            pub const LIMIT: usize = SmallIndex::LIMIT;

            /// The zero value.
            pub const ZERO: $name = $name(SmallIndex::ZERO);

            /// The number of bytes that a single value uses in memory.
            pub const SIZE: usize = SmallIndex::SIZE;

            /// Create a new value that is represented by a "small index."
            ///
            /// If the given index exceeds the maximum allowed value, then this
            /// returns an error.
            #[inline]
            pub fn new(value: usize) -> Result<$name, $err> {
                SmallIndex::new(value).map($name).map_err($err)
            }

            /// Create a new value without checking whether the given argument
            /// exceeds the maximum.
            ///
            /// Using this routine with an invalid value will result in
            /// unspecified behavior, but *not* undefined behavior. In
            /// particular, an invalid ID value is likely to cause panics or
            /// possibly even silent logical errors.
            ///
            /// Callers must never rely on this type to be within a certain
            /// range for memory safety.
            #[inline]
            pub const fn new_unchecked(value: usize) -> $name {
                $name(SmallIndex::new_unchecked(value))
            }

            /// Like `new`, but panics if the given value is not valid.
            #[inline]
            pub fn must(value: usize) -> $name {
                // $name::new(value).expect("invalid value")
                $name::new(value).expect(concat!(
                    "invalid ",
                    stringify!($name),
                    " value"
                ))
            }

            /// Return the internal value as a `usize`. This is guaranteed to
            /// never overflow `usize`.
            #[inline]
            pub const fn as_usize(&self) -> usize {
                self.0.as_usize()
            }

            /// Return the internal value as a `u32`. This is guaranteed to
            /// never overflow `u32`.
            #[inline]
            pub const fn as_u32(&self) -> u32 {
                self.0.as_u32()
            }

            /// Return the internal value as a i32`. This is guaranteed to
            /// never overflow an `i32`.
            #[inline]
            pub const fn as_i32(&self) -> i32 {
                self.0.as_i32()
            }

            /// Returns one more than this value as a usize.
            ///
            /// Since values represented by a "small index" have constraints
            /// on their maximum value, adding `1` to it will always fit in a
            /// `usize`, `u32` and a `i32`.
            #[inline]
            pub fn one_more(&self) -> usize {
                self.0.one_more()
            }

            /// Decode this value from the bytes given using the native endian
            /// byte order for the current target.
            ///
            /// If the decoded integer is not representable as a small index
            /// for the current target, then this returns an error.
            #[inline]
            pub fn from_ne_bytes(bytes: [u8; 4]) -> Result<$name, $err> {
                SmallIndex::from_ne_bytes(bytes).map($name).map_err($err)
            }

            /// Decode this value from the bytes given using the native endian
            /// byte order for the current target.
            ///
            /// This is analogous to `new_unchecked` in that is does not check
            /// whether the decoded integer is representable as a small index.
            #[inline]
            pub fn from_ne_bytes_unchecked(bytes: [u8; 4]) -> $name {
                $name(SmallIndex::from_ne_bytes_unchecked(bytes))
            }

            /// Return the underlying integer as raw bytes in native endian
            /// format.
            #[inline]
            pub fn to_ne_bytes(&self) -> [u8; 4] {
                self.0.to_ne_bytes()
            }

            /// Returns an iterator over all values from 0 up to and not
            /// including the given length.
            ///
            /// If the given length exceeds this type's limit, then this
            /// panics.
            #[cfg(feature = "alloc")]
            pub(crate) fn iter(len: usize) -> $iter {
                $iter::new(len)
            }
        }

        impl<T> core::ops::Index<$name> for [T] {
            type Output = T;

            #[inline]
            fn index(&self, index: $name) -> &T {
                &self[index.as_usize()]
            }
        }

        impl<T> core::ops::IndexMut<$name> for [T] {
            #[inline]
            fn index_mut(&mut self, index: $name) -> &mut T {
                &mut self[index.as_usize()]
            }
        }

        #[cfg(feature = "alloc")]
        impl<T> core::ops::Index<$name> for Vec<T> {
            type Output = T;

            #[inline]
            fn index(&self, index: $name) -> &T {
                &self[index.as_usize()]
            }
        }

        #[cfg(feature = "alloc")]
        impl<T> core::ops::IndexMut<$name> for Vec<T> {
            #[inline]
            fn index_mut(&mut self, index: $name) -> &mut T {
                &mut self[index.as_usize()]
            }
        }

        impl From<u8> for $name {
            fn from(value: u8) -> $name {
                $name(SmallIndex::from(value))
            }
        }

        impl TryFrom<u16> for $name {
            type Error = $err;

            fn try_from(value: u16) -> Result<$name, $err> {
                SmallIndex::try_from(value).map($name).map_err($err)
            }
        }

        impl TryFrom<u32> for $name {
            type Error = $err;

            fn try_from(value: u32) -> Result<$name, $err> {
                SmallIndex::try_from(value).map($name).map_err($err)
            }
        }

        impl TryFrom<u64> for $name {
            type Error = $err;

            fn try_from(value: u64) -> Result<$name, $err> {
                SmallIndex::try_from(value).map($name).map_err($err)
            }
        }

        impl TryFrom<usize> for $name {
            type Error = $err;

            fn try_from(value: usize) -> Result<$name, $err> {
                SmallIndex::try_from(value).map($name).map_err($err)
            }
        }

        #[cfg(test)]
        impl quickcheck::Arbitrary for $name {
            fn arbitrary(gen: &mut quickcheck::Gen) -> $name {
                $name(SmallIndex::arbitrary(gen))
            }
        }

        /// This error occurs when a value could not be constructed.
        ///
        /// This occurs when given an integer exceeding the maximum allowed
        /// value.
        ///
        /// When the `std` feature is enabled, this implements the `Error`
        /// trait.
        #[derive(Clone, Debug, Eq, PartialEq)]
        pub struct $err(SmallIndexError);

        impl $err {
            /// Returns the value that could not be converted to an ID.
            pub fn attempted(&self) -> u64 {
                self.0.attempted()
            }
        }

        #[cfg(feature = "std")]
        impl std::error::Error for $err {}

        impl core::fmt::Display for $err {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(
                    f,
                    "failed to create {} from {:?}, which exceeds {:?}",
                    stringify!($name),
                    self.attempted(),
                    $name::MAX,
                )
            }
        }

        #[derive(Clone, Debug)]
        pub(crate) struct $iter(SmallIndexIter);

        impl $iter {
            #[cfg(feature = "alloc")]
            fn new(len: usize) -> $iter {
                assert!(
                    len <= $name::LIMIT,
                    "cannot create iterator for {} when number of \
                     elements exceed {:?}",
                    stringify!($name),
                    $name::LIMIT,
                );
                $iter(SmallIndexIter { rng: 0..len })
            }
        }

        impl Iterator for $iter {
            type Item = $name;

            fn next(&mut self) -> Option<$name> {
                self.0.next().map($name)
            }
        }

        /// An iterator adapter that is like std::iter::Enumerate, but attaches
        /// small index values instead. It requires `ExactSizeIterator`. At
        /// construction, it ensures that the index of each element in the
        /// iterator is representable in the corresponding small index type.
        #[derive(Clone, Debug)]
        pub(crate) struct $withiter<I> {
            it: I,
            ids: $iter,
        }

        impl<I: Iterator + ExactSizeIterator> $withiter<I> {
            fn new(it: I) -> $withiter<I> {
                let ids = $name::iter(it.len());
                $withiter { it, ids }
            }
        }

        impl<I: Iterator + ExactSizeIterator> Iterator for $withiter<I> {
            type Item = ($name, I::Item);

            fn next(&mut self) -> Option<($name, I::Item)> {
                let item = self.it.next()?;
                // Number of elements in this iterator must match, according
                // to contract of ExactSizeIterator.
                let id = self.ids.next().unwrap();
                Some((id, item))
            }
        }
    };
}

define_index_type!(
    PatternID,
    PatternIDError,
    PatternIDIter,
    WithPatternIDIter
);
define_index_type!(StateID, StateIDError, StateIDIter, WithStateIDIter);

/// A utility trait that defines a couple of adapters for making it convenient
/// to access indices as "small index" types. We require ExactSizeIterator so
/// that iterator construction can do a single check to make sure the index of
/// each element is representable by its small index type.
#[cfg(feature = "alloc")]
pub(crate) trait IteratorIndexExt: Iterator {
    fn with_small_indices(self) -> WithSmallIndexIter<Self>
    where
        Self: Sized + ExactSizeIterator,
    {
        WithSmallIndexIter::new(self)
    }

    fn with_pattern_ids(self) -> WithPatternIDIter<Self>
    where
        Self: Sized + ExactSizeIterator,
    {
        WithPatternIDIter::new(self)
    }

    fn with_state_ids(self) -> WithStateIDIter<Self>
    where
        Self: Sized + ExactSizeIterator,
    {
        WithStateIDIter::new(self)
    }
}

#[cfg(feature = "alloc")]
impl<I: Iterator> IteratorIndexExt for I {}
