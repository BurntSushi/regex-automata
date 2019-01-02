use core::fmt;

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

    /// Copies the byte classes given. The given slice must have length 0 or
    /// length 256. Slices of length 0 are treated as singletons (every byte
    /// is its own class).
    pub fn from_slice(slice: &[u8]) -> ByteClasses {
        assert!(slice.is_empty() || slice.len() == 256);

        if slice.is_empty() {
            ByteClasses::singletons()
        } else {
            let mut classes = ByteClasses::empty();
            for (b, &class) in slice.iter().enumerate() {
                classes.set(b as u8, class);
            }
            classes
        }
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

    /// Return the total number of elements in the alphabet represented by
    /// these equivalence classes. Equivalently, this returns the total number
    /// of equivalence classes.
    #[inline]
    pub fn alphabet_len(&self) -> usize {
        self.0[255] as usize + 1
    }

    /// Returns true if and only if every byte in this class maps to its own
    /// equivalence class. Equivalently, there are 256 equivalence classes
    /// and each class contains exactly one byte.
    #[inline]
    pub fn is_singleton(&self) -> bool {
        self.alphabet_len() == 256
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
    #[cfg(feature = "std")]
    pub fn representatives(&self) -> ByteClassRepresentatives {
        ByteClassRepresentatives { classes: self, byte: 0, last_class: None }
    }

    /// Returns all of the bytes in the given equivalence class.
    ///
    /// The second element in the tuple indicates the number of elements in
    /// the array.
    fn elements(&self, equiv: u8) -> ([u8; 256], usize) {
        let (mut array, mut len) = ([0; 256], 0);
        for b in 0..256 {
            if self.get(b as u8) == equiv {
                array[len] = b as u8;
                len += 1;
            }
        }
        (array, len)
    }
}

impl fmt::Debug for ByteClasses {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_singleton() {
            write!(f, "ByteClasses({{singletons}})")
        } else {
            write!(f, "ByteClasses(")?;
            for equiv in 0..self.alphabet_len() {
                let (members, len) = self.elements(equiv as u8);
                write!(f, "{} => {:?}", equiv, &members[..len])?;
            }
            write!(f, ")")
        }
    }
}

/// An iterator over representative bytes from each equivalence class.
#[cfg(feature = "std")]
#[derive(Debug)]
pub struct ByteClassRepresentatives<'a> {
    classes: &'a ByteClasses,
    byte: usize,
    last_class: Option<u8>,
}

#[cfg(feature = "std")]
impl<'a> Iterator for ByteClassRepresentatives<'a> {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        while self.byte < 256 {
            let byte = self.byte as u8;
            let class = self.classes.get(byte);
            self.byte += 1;

            if self.last_class != Some(class) {
                self.last_class = Some(class);
                return Some(byte);
            }
        }
        None
    }
}
