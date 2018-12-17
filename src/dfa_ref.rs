use std::mem;
use std::slice;

use byteorder::{ByteOrder, NativeEndian};

use dfa::{ALPHABET_LEN, DFAKind};
use state_id::{StateID, dead_id};

/// A borrowed table-based deterministic finite automaton (DFA).
///
/// A `DFARef` is effectively a borrowed version of [`DFA`](struct.DFA.html).
/// In particular, the documentation for `DFA` applies equally well to this
/// type as well.
///
/// The key difference between `DFA` and `DFARef` is that the former requires
/// storing its transition table on the heap, where as the transition table
/// `DFARef` can be any region in memory, including, but not limited to,
/// heap memory, stack memory, read-only memory or a file-backed memory map.
///
/// This type is principally useful as a way of deserializing a DFA from
/// raw bytes in constant time without copying the transition table to the
/// heap. See [`DFARef::from_bytes`](struct.DFARef.html#method.from_bytes) for
/// an example.
#[derive(Clone, Copy, Debug)]
pub struct DFARef<'a, S = usize> {
    pub(crate) kind: DFAKind,
    pub(crate) start: S,
    pub(crate) state_count: usize,
    pub(crate) max_match: S,
    pub(crate) alphabet_len: usize,
    pub(crate) byte_classes: &'a [u8],
    pub(crate) trans: &'a [S],
}

impl<'a, S: StateID> DFARef<'a, S> {
    /// Returns true if and only if the given bytes match this DFA.
    ///
    /// This routine may short circuit if it knows that scanning future input
    /// will never lead to a different result. In particular, if a DFA enters
    /// a match state or a dead state, then this routine will return `true` or
    /// `false`, respectively, without inspecting any future input.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::DFA;
    ///
    /// # fn example() -> Result<(), regex_automata::Error> {
    /// let dfa = DFA::new("foo[0-9]+bar")?;
    /// assert_eq!(true, dfa.as_dfa_ref().is_match(b"foo12345bar"));
    /// assert_eq!(false, dfa.as_dfa_ref().is_match(b"foobar"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn is_match(&self, bytes: &[u8]) -> bool {
        self.is_match_inline(bytes)
    }

    /// Returns the first position at which a match is found.
    ///
    /// This routine stops scanning input in precisely the same circumstances
    /// as `is_match`. The key difference is that this routine returns the
    /// position at which it stopped scanning input if and only if a match
    /// was found. If no match is found, then `None` is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::DFA;
    ///
    /// # fn example() -> Result<(), regex_automata::Error> {
    /// let dfa = DFA::new("foo[0-9]+")?;
    /// assert_eq!(Some(4), dfa.as_dfa_ref().shortest_match(b"foo12345"));
    ///
    /// // Normally, the end of the leftmost first match here would be 3,
    /// // but the shortest match semantics detect a match earlier.
    /// let dfa = DFA::new("abc|a")?;
    /// assert_eq!(Some(1), dfa.as_dfa_ref().shortest_match(b"abc"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn shortest_match(&self, bytes: &[u8]) -> Option<usize> {
        self.shortest_match_inline(bytes)
    }

    /// Returns the end offset of the leftmost first match. If no match exists,
    /// then `None` is returned.
    ///
    /// The "leftmost first" match corresponds to the match with the smallest
    /// starting offset, but where the end offset is determined by preferring
    /// earlier branches in the original regular expression. For example,
    /// `Sam|Samwise` will match `Sam` in `Samwise`, but `Samwise|Sam` will
    /// match `Samwise` in `Samwise`.
    ///
    /// Generally speaking, the "leftmost first" match is how most backtracking
    /// regular expressions tend to work. This is in contrast to POSIX-style
    /// regular expressions that yield "leftmost longest" matches. Namely,
    /// both `Sam|Samwise` and `Samwise|Sam` match `Samwise` when using
    /// leftmost longest semantics.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::DFA;
    ///
    /// # fn example() -> Result<(), regex_automata::Error> {
    /// let dfa = DFA::new("foo[0-9]+")?;
    /// assert_eq!(Some(8), dfa.as_dfa_ref().find(b"foo12345"));
    ///
    /// // Even though a match is found after reading the first byte (`a`),
    /// // the leftmost first match semantics demand that we find the earliest
    /// // match that prefers earlier parts of the pattern over latter parts.
    /// let dfa = DFA::new("abc|a")?;
    /// assert_eq!(Some(3), dfa.as_dfa_ref().find(b"abc"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn find(&self, bytes: &[u8]) -> Option<usize> {
        self.find_inline(bytes)
    }

    /// Returns the start offset of the leftmost first match in reverse, by
    /// searching from the end of the input towards the start of the input. If
    /// no match exists, then `None` is returned.
    ///
    /// This routine is principally useful when used in conjunction with the
    /// [`DFABuilder::reverse`](struct.DFABuilder.html#method.reverse)
    /// configuration knob. In general, it's unlikely to be correct to use both
    /// `find` and `rfind` with the same DFA.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::DFABuilder;
    ///
    /// # fn example() -> Result<(), regex_automata::Error> {
    /// let dfa = DFABuilder::new().reverse(true).build("foo[0-9]+")?;
    /// assert_eq!(Some(0), dfa.as_dfa_ref().rfind(b"foo12345"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn rfind(&self, bytes: &[u8]) -> Option<usize> {
        self.rfind_inline(bytes)
    }

    /// Returns the memory usage, in bytes, of this DFA.
    ///
    /// The memory usage is computed based on the number of bytes used to
    /// represent this DFA's transition table. For an owned `DFA`, this
    /// corresponds to heap memory usage. For a `DFARef` built from static
    /// data, this corresponds to the amount of static data used.
    ///
    /// This does **not** include the stack size used up by this DFA. To
    /// compute that, used `std::mem::size_of::<DFARef>()`.
    pub fn memory_usage(&self) -> usize {
        self.byte_classes.len() + (self.trans.len() * mem::size_of::<S>())
    }

    /// Deserialize a DFA with a specific state identifier representation.
    ///
    /// Deserializing a DFA using this routine will **not** allocate any heap
    /// memory for the transition table. Specifically, deserialization is
    /// guaranteed to be a constant time operation.
    ///
    /// The bytes given should be generated by the serialization of a DFA with
    /// either the
    /// [`to_bytes_little_endian`](struct.DFA.html#method.to_bytes_little_endian)
    /// method or the
    /// [`to_bytes_big_endian`](struct.DFA.html#method.to_bytes_big_endian)
    /// endian, depending on the endianness of the machine you are
    /// deserializing this DFA from.
    ///
    /// If the state identifier representation is `usize`, then deserialization
    /// is dependent on the pointer size. For this reason, it is best to
    /// serialize DFAs using a fixed size representation for your state
    /// identifiers, such as `u8`, `u16`, `u32` or `u64`.
    ///
    /// # Panics
    ///
    /// The bytes given should be *trusted*. In particular, if the bytes are
    /// not a valid serialization of a DFA, or if the bytes are not aligned to
    /// an 8 byte boundary, or if the endianness of the serialized bytes is
    /// different than the endianness of the machine that is deserializing the
    /// DFA, then this routine will panic.
    ///
    /// # Safety
    ///
    /// This routine is unsafe because it permits callers to provide an
    /// arbitrary transition table with possibly incorrect transitions. While
    /// the various serialization routines on the `DFA` type will never return
    /// an incorrect transition table, there is no guarantee that the bytes
    /// provided here are correct. While deserialization does many checks (as
    /// documented above in the panic conditions), this routine does not check
    /// that the transition table is correct. Given an incorrect transition
    /// table, it is possible for the search routines to access out-of-bounds
    /// memory because of explicit bounds check elision.
    ///
    /// # Example
    ///
    /// This example shows how to serialize a DFA to raw bytes, deserialize it
    /// and then use it for searching. Note that we first convert the DFA to
    /// using `u16` for its state identifier representation before serializing
    /// it. While this isn't strictly necessary, it's good practice in order to
    /// decrease the size of the DFA and to avoid platform specific pitfalls
    /// such as differing pointer sizes.
    ///
    /// ```
    /// use regex_automata::{DFA, DFARef};
    ///
    /// # fn example() -> Result<(), regex_automata::Error> {
    /// let initial = DFA::new("foo[0-9]+")?;
    /// let bytes = initial.to_u16()?.to_bytes_native_endian()?;
    /// let dfa: DFARef<u16> = unsafe { DFARef::from_bytes(&bytes) };
    ///
    /// assert_eq!(Some(8), dfa.find(b"foo12345"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub unsafe fn from_bytes(mut buf: &'a [u8]) -> DFARef<'a, S> {
        // skip over label
        match buf.iter().position(|&b| b == b'\x00') {
            None => panic!("could not find label"),
            Some(i) => buf = &buf[i+1..],
        }

        // check that current endianness is same as endianness of DFA
        let endian_check = NativeEndian::read_u16(buf);
        buf = &buf[2..];
        if endian_check != 0xFEFF {
            panic!(
                "endianness mismatch, expected 0xFEFF but got 0x{:X}. \
                 are you trying to load a DFA serialized with a different \
                 endianness?",
                endian_check,
            );
        }

        // check that the version number is supported
        let version = NativeEndian::read_u16(buf);
        buf = &buf[2..];
        if version != 1 {
            panic!(
                "expected version 1, but found unsupported version {}",
                version,
            );
        }

        // read size of state
        let state_size = NativeEndian::read_u16(buf) as usize;
        if state_size != mem::size_of::<S>() {
            panic!(
                "state size of DFA ({}) does not match \
                 requested state size ({})",
                state_size, mem::size_of::<S>(),
            );
        }
        buf = &buf[2..];

        // read DFA kind
        let kind = DFAKind::from_byte(NativeEndian::read_u16(buf) as u8);
        buf = &buf[2..];

        // read start state
        let start = S::from_usize(NativeEndian::read_u64(buf) as usize);
        buf = &buf[8..];

        // read state count
        let state_count = NativeEndian::read_u64(buf) as usize;
        buf = &buf[8..];

        // read max match state
        let max_match = S::from_usize(NativeEndian::read_u64(buf) as usize);
        buf = &buf[8..];

        // read alphabet length
        let alphabet_len = NativeEndian::read_u64(buf) as usize;
        buf = &buf[8..];

        // read byte classes
        let byte_classes =
            if kind.is_byte_class() {
                &buf[..256]
            } else {
                &[]
            };
        buf = &buf[256..];

        assert_eq!(
            0,
            buf.as_ptr() as usize % mem::align_of::<S>(),
            "DFA transition table is not properly aligned"
        );
        let len = state_count * alphabet_len;
        assert!(
            buf.len() >= len,
            "insufficient transition table bytes, \
             expected at least {} but only have {}",
            len, buf.len()
        );

        let trans = slice::from_raw_parts(buf.as_ptr() as *const S, len);
        DFARef {
            kind, start, state_count, max_match,
            alphabet_len, byte_classes, trans,
        }
    }
}

impl<'a, S: StateID> DFARef<'a, S> {
    fn start(&self) -> S {
        self.start
    }

    fn is_match_state(&self, id: S) -> bool {
        self.is_possible_match_state(id) && !self.is_dead(id)
    }

    fn is_possible_match_state(&self, id: S) -> bool {
        id <= self.max_match
    }

    fn is_dead(&self, id: S) -> bool {
        id == dead_id()
    }

    #[allow(dead_code)]
    fn next_state(
        &self,
        current: S,
        input: u8,
    ) -> S {
        self.trans[current.to_usize() * ALPHABET_LEN + input as usize]
    }

    unsafe fn next_state_unchecked(
        &self,
        current: S,
        input: u8,
    ) -> S {
        *self.trans.get_unchecked(
            current.to_usize() * ALPHABET_LEN + input as usize,
        )
    }

    #[allow(dead_code)]
    fn next_state_premultiplied(
        &self,
        current: S,
        input: u8,
    ) -> S {
        self.trans[current.to_usize() + input as usize]
    }

    unsafe fn next_state_premultiplied_unchecked(
        &self,
        current: S,
        input: u8,
    ) -> S {
        *self.trans.get_unchecked(current.to_usize() + input as usize)
    }

    #[allow(dead_code)]
    fn next_state_byte_class(
        &self,
        current: S,
        input: u8,
    ) -> S {
        let input = self.byte_classes[input as usize];
        self.trans[current.to_usize() * self.alphabet_len + input as usize]
    }

    unsafe fn next_state_byte_class_unchecked(
        &self,
        current: S,
        input: u8,
    ) -> S {
        let input = *self.byte_classes.get_unchecked(input as usize);
        *self.trans.get_unchecked(
            current.to_usize() * self.alphabet_len + input as usize,
        )
    }

    #[allow(dead_code)]
    fn next_state_premultiplied_byte_class(
        &self,
        current: S,
        input: u8,
    ) -> S {
        let input = self.byte_classes[input as usize];
        self.trans[current.to_usize() + input as usize]
    }

    unsafe fn next_state_premultiplied_byte_class_unchecked(
        &self,
        current: S,
        input: u8,
    ) -> S {
        let input = *self.byte_classes.get_unchecked(input as usize);
        *self.trans.get_unchecked(current.to_usize() + input as usize)
    }
}

macro_rules! is_match {
    ($dfa:ident, $bytes:ident, $trans:ident) => {{
        let mut state = $dfa.start();
        if $dfa.is_possible_match_state(state) {
            return $dfa.is_match_state(state);
        }
        for &b in $bytes.iter() {
            state = unsafe { $dfa.$trans(state, b) };
            if $dfa.is_possible_match_state(state) {
                return $dfa.is_match_state(state);
            }
        }
        false
    }}
}

macro_rules! shortest_match {
    ($dfa:ident, $bytes:ident, $trans:ident) => {{
        let mut state = $dfa.start();
        if $dfa.is_possible_match_state(state) {
            return if $dfa.is_dead(state) { None } else { Some(0) };
        }
        for (i, &b) in $bytes.iter().enumerate() {
            state = unsafe { $dfa.$trans(state, b) };
            if $dfa.is_possible_match_state(state) {
                return if $dfa.is_dead(state) { None } else { Some(i + 1) };
            }
        }
        None
    }}
}

macro_rules! find {
    ($dfa:ident, $bytes:ident, $trans:ident) => {{
        let mut state = $dfa.start;
        let mut last_match =
            if $dfa.is_dead(state) {
                return None;
            } else if $dfa.is_match_state(state) {
                Some(0)
            } else {
                None
            };
        for (i, &b) in $bytes.iter().enumerate() {
            state = unsafe { $dfa.$trans(state, b) };
            if $dfa.is_possible_match_state(state) {
                if $dfa.is_dead(state) {
                    return last_match;
                }
                last_match = Some(i + 1);
            }
        }
        last_match
    }}
}

macro_rules! rfind {
    ($dfa:ident, $bytes:ident, $trans:ident) => {{
        let mut state = $dfa.start;
        let mut last_match =
            if $dfa.is_dead(state) {
                return None;
            } else if $dfa.is_match_state(state) {
                Some(0)
            } else {
                None
            };
        for (i, &b) in $bytes.iter().enumerate().rev() {
            state = unsafe { $dfa.$trans(state, b) };
            if $dfa.is_possible_match_state(state) {
                if $dfa.is_dead(state) {
                    return last_match;
                }
                last_match = Some(i);
            }
        }
        last_match
    }}
}

impl<'a, S: StateID> DFARef<'a, S> {
    #[inline(always)]
    pub(crate) fn is_match_inline(&self, bytes: &[u8]) -> bool {
        match self.kind {
            DFAKind::Basic => self.is_match_basic(bytes),
            DFAKind::Premultiplied => self.is_match_premultiplied(bytes),
            DFAKind::ByteClass => self.is_match_byte_class(bytes),
            DFAKind::PremultipliedByteClass => {
                self.is_match_premultiplied_byte_class(bytes)
            }
        }
    }

    fn is_match_basic(&self, bytes: &[u8]) -> bool {
        is_match!(self, bytes, next_state_unchecked)
    }

    fn is_match_premultiplied(&self, bytes: &[u8]) -> bool {
        is_match!(self, bytes, next_state_premultiplied_unchecked)
    }

    fn is_match_byte_class(&self, bytes: &[u8]) -> bool {
        is_match!(self, bytes, next_state_byte_class_unchecked)
    }

    fn is_match_premultiplied_byte_class(&self, bytes: &[u8]) -> bool {
        is_match!(self, bytes, next_state_premultiplied_byte_class_unchecked)
    }

    #[inline(always)]
    pub(crate) fn shortest_match_inline(&self, bytes: &[u8]) -> Option<usize> {
        match self.kind {
            DFAKind::Basic => self.shortest_match_basic(bytes),
            DFAKind::Premultiplied => self.shortest_match_premultiplied(bytes),
            DFAKind::ByteClass => self.shortest_match_byte_class(bytes),
            DFAKind::PremultipliedByteClass => {
                self.shortest_match_premultiplied_byte_class(bytes)
            }
        }
    }

    fn shortest_match_basic(&self, bytes: &[u8]) -> Option<usize> {
        shortest_match!(self, bytes, next_state_unchecked)
    }

    fn shortest_match_premultiplied(&self, bytes: &[u8]) -> Option<usize> {
        shortest_match!(self, bytes, next_state_premultiplied_unchecked)
    }

    fn shortest_match_byte_class(&self, bytes: &[u8]) -> Option<usize> {
        shortest_match!(self, bytes, next_state_byte_class_unchecked)
    }

    fn shortest_match_premultiplied_byte_class(
        &self,
        bytes: &[u8],
    ) -> Option<usize> {
        shortest_match!(
            self,
            bytes,
            next_state_premultiplied_byte_class_unchecked
        )
    }

    #[inline(always)]
    pub(crate) fn find_inline(&self, bytes: &[u8]) -> Option<usize> {
        match self.kind {
            DFAKind::Basic => self.find_basic(bytes),
            DFAKind::Premultiplied => self.find_premultiplied(bytes),
            DFAKind::ByteClass => self.find_byte_class(bytes),
            DFAKind::PremultipliedByteClass => {
                self.find_premultiplied_byte_class(bytes)
            }
        }
    }

    fn find_basic(&self, bytes: &[u8]) -> Option<usize> {
        find!(self, bytes, next_state_unchecked)
    }

    fn find_premultiplied(&self, bytes: &[u8]) -> Option<usize> {
        find!(self, bytes, next_state_premultiplied_unchecked)
    }

    fn find_byte_class(&self, bytes: &[u8]) -> Option<usize> {
        find!(self, bytes, next_state_byte_class_unchecked)
    }

    fn find_premultiplied_byte_class(&self, bytes: &[u8]) -> Option<usize> {
        find!(self, bytes, next_state_premultiplied_byte_class_unchecked)
    }

    #[inline(always)]
    pub(crate) fn rfind_inline(&self, bytes: &[u8]) -> Option<usize> {
        match self.kind {
            DFAKind::Basic => self.rfind_basic(bytes),
            DFAKind::Premultiplied => self.rfind_premultiplied(bytes),
            DFAKind::ByteClass => self.rfind_byte_class(bytes),
            DFAKind::PremultipliedByteClass => {
                self.rfind_premultiplied_byte_class(bytes)
            }
        }
    }

    fn rfind_basic(&self, bytes: &[u8]) -> Option<usize> {
        rfind!(self, bytes, next_state_unchecked)
    }

    fn rfind_premultiplied(&self, bytes: &[u8]) -> Option<usize> {
        rfind!(self, bytes, next_state_premultiplied_unchecked)
    }

    fn rfind_byte_class(&self, bytes: &[u8]) -> Option<usize> {
        rfind!(self, bytes, next_state_byte_class_unchecked)
    }

    fn rfind_premultiplied_byte_class(&self, bytes: &[u8]) -> Option<usize> {
        rfind!(self, bytes, next_state_premultiplied_byte_class_unchecked)
    }
}
