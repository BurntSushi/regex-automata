use core::fmt;

use crate::{
    bytes::{self, DeserializeError, SerializeError},
    dfa::Error,
};

#[derive(Clone, Copy, Eq, PartialEq, PartialOrd, Ord)]
pub enum Byte {
    U8(u8),
    EOF(u16),
}

impl Byte {
    pub fn as_u8(self) -> Option<u8> {
        match self {
            Byte::U8(b) => Some(b),
            Byte::EOF(_) => None,
        }
    }

    pub fn as_eof(self) -> Option<usize> {
        match self {
            Byte::U8(_) => None,
            Byte::EOF(eof) => Some(eof as usize),
        }
    }

    pub fn as_usize(self) -> usize {
        match self {
            Byte::U8(b) => b as usize,
            Byte::EOF(eof) => eof as usize,
        }
    }

    pub fn is_eof(&self) -> bool {
        match *self {
            Byte::EOF(_) => true,
            _ => false,
        }
    }

    pub fn is_word_byte(&self) -> bool {
        self.as_u8().map_or(false, crate::word::is_word_byte)
    }
}

impl fmt::Debug for Byte {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Byte::U8(b) => crate::util::fmt_byte(f, b),
            Byte::EOF(_) => write!(f, "EOF"),
        }
    }
}

/// A representation of byte oriented equivalence classes.
///
/// This is used in a DFA to reduce the size of the transition table. This can
/// have a particularly large impact not only on the total size of a dense DFA,
/// but also on compile times.
#[derive(Clone, Copy)]
pub struct ByteClasses([u8; 256]);

impl ByteClasses {
    /// Creates a new set of equivalence classes where all bytes are mapped to
    /// the same class.
    pub fn empty() -> ByteClasses {
        ByteClasses([0; 256])
    }

    /// Creates a new set of equivalence classes where each byte belongs to
    /// its own equivalence class.
    pub fn singletons() -> ByteClasses {
        let mut classes = ByteClasses::empty();
        for i in 0..256 {
            classes.set(i as u8, i as u8);
        }
        classes
    }

    /// Deserializes a byte class map from the given slice. If the slice is of
    /// insufficient length or otherwise contains an impossible mapping, then
    /// an error is returned. Upon success, the number of bytes read along with
    /// the map are returned. The number of bytes read is always a multiple of
    /// 8.
    pub fn from_bytes(
        slice: &[u8],
    ) -> Result<(ByteClasses, usize), DeserializeError> {
        if slice.len() < 256 {
            return Err(DeserializeError::buffer_too_small("byte class map"));
        }
        let mut classes = ByteClasses::empty();
        for (b, &class) in slice[..256].iter().enumerate() {
            classes.set(b as u8, class);
        }
        for b in classes.iter() {
            if b.as_usize() >= classes.alphabet_len() {
                return Err(DeserializeError::generic(
                    "found equivalence class greater than alphabet len",
                ));
            }
        }
        Ok((classes, 256))
    }

    /// Writes this byte class map to the given byte buffer. if the given
    /// buffer is too small, then an error is returned. Upon success, the total
    /// number of bytes written is returned. The number of bytes written is
    /// guaranteed to be a multiple of 8.
    pub fn write_to(
        &self,
        mut dst: &mut [u8],
    ) -> Result<usize, SerializeError> {
        let nwrite = self.write_to_len();
        if dst.len() < nwrite {
            return Err(SerializeError::buffer_too_small("byte class map"));
        }
        for b in 0..=255 {
            dst[0] = self.get(b);
            dst = &mut dst[1..];
        }
        Ok(nwrite)
    }

    /// Returns the total number of bytes written by `write_to`.
    pub fn write_to_len(&self) -> usize {
        256
    }

    /// Set the equivalence class for the given byte.
    #[inline]
    pub fn set(&mut self, byte: u8, class: u8) {
        self.0[byte as usize] = class;
    }

    /// Get the equivalence class for the given byte.
    #[inline]
    pub fn get(&self, byte: u8) -> u8 {
        self.0[byte as usize]
    }

    /// Get the equivalence class for the given byte while forcefully
    /// eliding bounds checks.
    #[inline]
    pub unsafe fn get_unchecked(&self, byte: u8) -> u8 {
        *self.0.get_unchecked(byte as usize)
    }

    #[inline]
    pub fn eof(&self) -> Byte {
        Byte::EOF(self.alphabet_len() as u16 - 1)
    }

    #[inline]
    pub fn usize_to_byte(&self, b: usize) -> Byte {
        if b == self.alphabet_len() - 1 {
            self.eof()
        } else {
            assert!(b <= 255);
            Byte::U8(b as u8)
        }
    }

    /// Return the total number of elements in the alphabet represented by
    /// these equivalence classes. Equivalently, this returns the total number
    /// of equivalence classes.
    #[inline]
    pub fn alphabet_len(&self) -> usize {
        // Add one since the number of equivalence classes is one bigger than
        // the last one. But add another to account for the final EOF class
        // that isn't explicitly represented.
        self.0[255] as usize + 1 + 1
    }

    /// Returns the stride, as a base-2 exponent, required for these
    /// equivalence classes.
    ///
    /// The stride is always the smallest power of 2 that is greater than or
    /// equal to the alphabet length. This is done so that converting between
    /// state IDs and indices can be done with shifts alone, which is much
    /// faster than integer division.
    pub fn stride2(&self) -> usize {
        self.alphabet_len().next_power_of_two().trailing_zeros() as usize
    }

    /// Returns true if and only if every byte in this class maps to its own
    /// equivalence class. Equivalently, there are 257 equivalence classes
    /// and each class contains exactly one byte (plus the special EOF class).
    #[inline]
    pub fn is_singleton(&self) -> bool {
        self.alphabet_len() == 257
    }

    /// Returns an iterator over all equivalence classes in this set.
    pub fn iter(&self) -> ByteClassIter<'_> {
        ByteClassIter { classes: self, i: 0 }
    }

    /// Returns an iterator over a sequence of representative bytes from each
    /// equivalence class. Namely, this yields exactly N items, where N is
    /// equivalent to the number of equivalence classes. Each item is an
    /// arbitrary byte drawn from each equivalence class.
    ///
    /// This is useful when one is determinizing an NFA and the NFA's alphabet
    /// hasn't been converted to equivalence classes yet. Picking an arbitrary
    /// byte from each equivalence class then permits a full exploration of
    /// the NFA instead of using every possible byte value.
    pub fn representatives(&self) -> ByteClassRepresentatives<'_> {
        ByteClassRepresentatives { classes: self, byte: 0, last_class: None }
    }

    /// Returns an iterator of the bytes in the given equivalence class.
    pub fn elements(&self, class: Byte) -> ByteClassElements {
        ByteClassElements { classes: self, class, byte: 0 }
    }

    /// Returns an iterator of byte ranges in the given equivalence class.
    ///
    /// That is, a sequence of contiguous ranges are returned. Typically, every
    /// class maps to a single contiguous range.
    fn element_ranges(&self, class: Byte) -> ByteClassElementRanges {
        ByteClassElementRanges { elements: self.elements(class), range: None }
    }
}

impl fmt::Debug for ByteClasses {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_singleton() {
            write!(f, "ByteClasses({{singletons}})")
        } else {
            write!(f, "ByteClasses(")?;
            for (i, class) in self.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:?} => [", class.as_usize())?;
                for (start, end) in self.element_ranges(class) {
                    if start == end {
                        write!(f, "{:?}", start)?;
                    } else {
                        write!(f, "{:?}-{:?}", start, end)?;
                    }
                }
                write!(f, "]")?;
            }
            write!(f, ")")
        }
    }
}

/// An iterator over each equivalence class.
#[derive(Debug)]
pub struct ByteClassIter<'a> {
    classes: &'a ByteClasses,
    i: usize,
}

impl<'a> Iterator for ByteClassIter<'a> {
    type Item = Byte;

    fn next(&mut self) -> Option<Byte> {
        if self.i + 1 == self.classes.alphabet_len() {
            let class = self.i as u16;
            self.i += 1;
            Some(Byte::EOF(class))
        } else if self.i < self.classes.alphabet_len() {
            let class = self.i as u8;
            self.i += 1;
            Some(Byte::U8(class))
        } else {
            None
        }
    }
}

/// An iterator over representative bytes from each equivalence class.
#[derive(Debug)]
pub struct ByteClassRepresentatives<'a> {
    classes: &'a ByteClasses,
    byte: usize,
    last_class: Option<u8>,
}

impl<'a> Iterator for ByteClassRepresentatives<'a> {
    type Item = Byte;

    fn next(&mut self) -> Option<Byte> {
        while self.byte < 256 {
            let byte = self.byte as u8;
            let class = self.classes.get(byte);
            self.byte += 1;

            if self.last_class != Some(class) {
                self.last_class = Some(class);
                return Some(Byte::U8(byte));
            }
        }
        if self.byte == 256 {
            self.byte += 1;
            return Some(self.classes.eof());
        }
        None
    }
}

/// An iterator over all elements in an equivalence class.
#[derive(Debug)]
pub struct ByteClassElements<'a> {
    classes: &'a ByteClasses,
    class: Byte,
    byte: usize,
}

impl<'a> Iterator for ByteClassElements<'a> {
    type Item = Byte;

    fn next(&mut self) -> Option<Byte> {
        while self.byte < 256 {
            let byte = self.byte;
            self.byte += 1;
            if Byte::U8(self.classes.get(byte as u8)) == self.class {
                return Some(Byte::U8(byte as u8));
            }
        }
        if self.byte < 257 {
            self.byte += 1;
            if self.class.is_eof() {
                return Some(Byte::EOF(256));
            }
        }
        None
    }
}

/// An iterator over all elements in an equivalence class expressed as a
/// sequence of contiguous ranges.
#[derive(Debug)]
pub struct ByteClassElementRanges<'a> {
    elements: ByteClassElements<'a>,
    range: Option<(Byte, Byte)>,
}

impl<'a> Iterator for ByteClassElementRanges<'a> {
    type Item = (Byte, Byte);

    fn next(&mut self) -> Option<(Byte, Byte)> {
        loop {
            let element = match self.elements.next() {
                None => return self.range.take(),
                Some(element) => element,
            };
            match self.range.take() {
                None => {
                    self.range = Some((element, element));
                }
                Some((start, end)) => {
                    if end.as_usize() + 1 != element.as_usize()
                        || element.is_eof()
                    {
                        self.range = Some((element, element));
                        return Some((start, end));
                    }
                    self.range = Some((start, element));
                }
            }
        }
    }
}

/// A byte class set keeps track of an *approximation* of equivalence classes
/// of bytes during NFA construction. That is, every byte in an equivalence
/// class cannot discriminate between a match and a non-match.
///
/// For example, in the regex `[ab]+`, the bytes `a` and `b` would be in the
/// same equivalence class because it never matters whether an `a` or a `b` is
/// seen, and no combination of `a`s and `b`s in the text can discriminate
/// a match.
///
/// Note though that this does not compute the minimal set of equivalence
/// classes. For example, in the regex `[ac]+`, both `a` and `c` are in the
/// same equivalence class for the same reason that `a` and `b` are in the
/// same equivalence class in the aforementioned regex. However, in this
/// implementation, `a` and `c` are put into distinct equivalence classes.
/// The reason for this is implementation complexity. In the future, we should
/// endeavor to compute the minimal equivalence classes since they can have a
/// rather large impact on the size of the DFA.
///
/// The representation here is 256 booleans, all initially set to false. Each
/// boolean maps to its corresponding byte based on position. A `true` value
/// indicates the end of an equivalence class, where its corresponding byte
/// and all of the bytes corresponding to all previous contiguous `false`
/// values are in the same equivalence class.
///
/// This particular representation only permits contiguous ranges of bytes to
/// be in the same equivalence class, which means that we can never discover
/// the true minimal set of equivalence classes.
#[derive(Clone, Debug)]
pub struct ByteClassSet(ByteSet);

impl ByteClassSet {
    /// Create a new set of byte classes where all bytes are part of the same
    /// equivalence class.
    pub fn new() -> Self {
        ByteClassSet(ByteSet::empty())
    }

    /// Create a new set of byte classes where all bytes are part of the same
    /// equivalence class.
    pub fn empty() -> Self {
        ByteClassSet(ByteSet::empty())
    }

    /// Indicate the the range of byte given (inclusive) can discriminate a
    /// match between it and all other bytes outside of the range.
    pub fn set_range(&mut self, start: u8, end: u8) {
        debug_assert!(start <= end);
        if start > 0 {
            self.0.add(start - 1);
        }
        self.0.add(end);
    }

    /// Add the contiguous ranges in the set given to this byte class set.
    pub fn add_set(&mut self, set: &ByteSet) {
        for (start, end) in set.iter_ranges() {
            self.set_range(start, end);
        }
    }

    /// Convert this boolean set to a map that maps all byte values to their
    /// corresponding equivalence class. The last mapping indicates the largest
    /// equivalence class identifier (which is never bigger than 255).
    pub fn byte_classes(&self) -> ByteClasses {
        let mut classes = ByteClasses::empty();
        let mut class = 0u8;
        let mut b = 0u8;
        loop {
            classes.set(b, class);
            if b == 255 {
                break;
            }
            if self.0.contains(b) {
                class = class.checked_add(1).unwrap();
            }
            b = b.checked_add(1).unwrap();
        }
        classes
    }
}

/// A simple set of bytes that is reasonably cheap to copy and allocation free.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ByteSet {
    bits: BitSet,
}

/// The representation of a byte set. Split out so that we can define a
/// convenient Debug impl for it while keeping "ByteSet" in the output.
#[derive(Clone, Copy, Default, Eq, PartialEq)]
struct BitSet([u128; 2]);

impl ByteSet {
    /// Create an empty set of bytes.
    pub fn empty() -> ByteSet {
        ByteSet { bits: BitSet([0; 2]) }
    }

    /// Add a byte to this set.
    ///
    /// If the given byte already belongs to this set, then this is a no-op.
    pub fn add(&mut self, byte: u8) {
        let bucket = byte / 128;
        let bit = byte % 128;
        self.bits.0[bucket as usize] |= 1 << bit;
    }

    /// Add an inclusive range of bytes.
    pub fn add_all(&mut self, start: u8, end: u8) {
        for b in start..=end {
            self.add(b);
        }
    }

    /// Remove a byte from this set.
    ///
    /// If the given byte is not in this set, then this is a no-op.
    pub fn remove(&mut self, byte: u8) {
        let bucket = byte / 128;
        let bit = byte % 128;
        self.bits.0[bucket as usize] &= !(1 << bit);
    }

    /// Remove an inclusive range of bytes.
    pub fn remove_all(&mut self, start: u8, end: u8) {
        for b in start..=end {
            self.remove(b);
        }
    }

    /// Return true if and only if the given byte is in this set.
    pub fn contains(&self, byte: u8) -> bool {
        let bucket = byte / 128;
        let bit = byte % 128;
        self.bits.0[bucket as usize] & (1 << bit) > 0
    }

    /// Return true if and only if the given inclusive range of bytes is in
    /// this set.
    pub fn contains_range(&self, start: u8, end: u8) -> bool {
        (start..=end).all(|b| self.contains(b))
    }

    /// Returns an iterator over all bytes in this set.
    pub fn iter(&self) -> ByteSetIter {
        ByteSetIter { set: self, b: 0 }
    }

    /// Returns an iterator over all contiguous ranges of bytes in this set.
    pub fn iter_ranges(&self) -> ByteSetRangeIter {
        ByteSetRangeIter { set: self, b: 0 }
    }

    /// Return the number of bytes in this set.
    pub fn len(&self) -> usize {
        (self.bits.0[0].count_ones() + self.bits.0[1].count_ones()) as usize
    }

    /// Return true if and only if this set is empty.
    pub fn is_empty(&self) -> bool {
        self.bits.0 == [0, 0]
    }
}

impl core::fmt::Debug for BitSet {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        let mut fmtd = f.debug_set();
        for b in (0..256).map(|b| b as u8) {
            if (ByteSet { bits: *self }).contains(b) {
                fmtd.entry(&b);
            }
        }
        fmtd.finish()
    }
}

#[derive(Debug)]
pub struct ByteSetIter<'a> {
    set: &'a ByteSet,
    b: usize,
}

impl<'a> Iterator for ByteSetIter<'a> {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        while self.b <= 255 {
            let b = self.b as u8;
            self.b += 1;
            if self.set.contains(b) {
                return Some(b);
            }
        }
        None
    }
}

#[derive(Debug)]
pub struct ByteSetRangeIter<'a> {
    set: &'a ByteSet,
    b: usize,
}

impl<'a> Iterator for ByteSetRangeIter<'a> {
    type Item = (u8, u8);

    fn next(&mut self) -> Option<(u8, u8)> {
        while self.b <= 255 {
            let start = self.b as u8;
            self.b += 1;
            if !self.set.contains(start) {
                continue;
            }

            let mut end = start;
            while self.b <= 255 && self.set.contains(self.b as u8) {
                end = self.b as u8;
                self.b += 1;
            }
            return Some((start, end));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_classes() {
        let mut set = ByteClassSet::new();
        set.set_range(b'a', b'z');

        let classes = set.byte_classes();
        assert_eq!(classes.get(0), 0);
        assert_eq!(classes.get(1), 0);
        assert_eq!(classes.get(2), 0);
        assert_eq!(classes.get(b'a' - 1), 0);
        assert_eq!(classes.get(b'a'), 1);
        assert_eq!(classes.get(b'm'), 1);
        assert_eq!(classes.get(b'z'), 1);
        assert_eq!(classes.get(b'z' + 1), 2);
        assert_eq!(classes.get(254), 2);
        assert_eq!(classes.get(255), 2);

        let mut set = ByteClassSet::new();
        set.set_range(0, 2);
        set.set_range(4, 6);
        let classes = set.byte_classes();
        assert_eq!(classes.get(0), 0);
        assert_eq!(classes.get(1), 0);
        assert_eq!(classes.get(2), 0);
        assert_eq!(classes.get(3), 1);
        assert_eq!(classes.get(4), 2);
        assert_eq!(classes.get(5), 2);
        assert_eq!(classes.get(6), 2);
        assert_eq!(classes.get(7), 3);
        assert_eq!(classes.get(255), 3);
    }

    #[test]
    fn full_byte_classes() {
        let mut set = ByteClassSet::new();
        for i in 0..256u16 {
            set.set_range(i as u8, i as u8);
        }
        assert_eq!(set.byte_classes().alphabet_len(), 257);
    }

    #[test]
    fn elements_typical() {
        let mut set = ByteClassSet::new();
        set.set_range(b'b', b'd');
        set.set_range(b'g', b'm');
        set.set_range(b'z', b'z');
        let classes = set.byte_classes();
        // class 0: \x00-a
        // class 1: b-d
        // class 2: e-f
        // class 3: g-m
        // class 4: n-y
        // class 5: z-z
        // class 6: \x7B-\xFF
        // class 7: EOF
        assert_eq!(classes.alphabet_len(), 8);

        let elements = classes.elements(Byte::U8(0)).collect::<Vec<_>>();
        assert_eq!(elements.len(), 98);
        assert_eq!(elements[0], Byte::U8(b'\x00'));
        assert_eq!(elements[97], Byte::U8(b'a'));

        let elements = classes.elements(Byte::U8(1)).collect::<Vec<_>>();
        assert_eq!(
            elements,
            vec![Byte::U8(b'b'), Byte::U8(b'c'), Byte::U8(b'd')],
        );

        let elements = classes.elements(Byte::U8(2)).collect::<Vec<_>>();
        assert_eq!(elements, vec![Byte::U8(b'e'), Byte::U8(b'f')],);

        let elements = classes.elements(Byte::U8(3)).collect::<Vec<_>>();
        assert_eq!(
            elements,
            vec![
                Byte::U8(b'g'),
                Byte::U8(b'h'),
                Byte::U8(b'i'),
                Byte::U8(b'j'),
                Byte::U8(b'k'),
                Byte::U8(b'l'),
                Byte::U8(b'm'),
            ],
        );

        let elements = classes.elements(Byte::U8(4)).collect::<Vec<_>>();
        assert_eq!(elements.len(), 12);
        assert_eq!(elements[0], Byte::U8(b'n'));
        assert_eq!(elements[11], Byte::U8(b'y'));

        let elements = classes.elements(Byte::U8(5)).collect::<Vec<_>>();
        assert_eq!(elements, vec![Byte::U8(b'z')]);

        let elements = classes.elements(Byte::U8(6)).collect::<Vec<_>>();
        assert_eq!(elements.len(), 133);
        assert_eq!(elements[0], Byte::U8(b'\x7B'));
        assert_eq!(elements[132], Byte::U8(b'\xFF'));

        let elements = classes.elements(Byte::EOF(7)).collect::<Vec<_>>();
        assert_eq!(elements, vec![Byte::EOF(256)]);
    }

    #[test]
    fn elements_singletons() {
        let classes = ByteClasses::singletons();
        assert_eq!(classes.alphabet_len(), 257);

        let elements = classes.elements(Byte::U8(b'a')).collect::<Vec<_>>();
        assert_eq!(elements, vec![Byte::U8(b'a')]);

        let elements = classes.elements(Byte::EOF(5)).collect::<Vec<_>>();
        assert_eq!(elements, vec![Byte::EOF(256)]);
    }

    #[test]
    fn elements_empty() {
        let classes = ByteClasses::empty();
        assert_eq!(classes.alphabet_len(), 2);

        let elements = classes.elements(Byte::U8(0)).collect::<Vec<_>>();
        assert_eq!(elements.len(), 256);
        assert_eq!(elements[0], Byte::U8(b'\x00'));
        assert_eq!(elements[255], Byte::U8(b'\xFF'));

        let elements = classes.elements(Byte::EOF(1)).collect::<Vec<_>>();
        assert_eq!(elements, vec![Byte::EOF(256)]);
    }
}
