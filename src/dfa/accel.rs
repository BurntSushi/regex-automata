use core::convert::TryInto;

use crate::bytes::{self, DeserializeError, Endian, SerializeError};
use crate::dfa::Error;

const ACCEL_LEN: usize = 4;
const ACCEL_CAP: usize = 8;

pub fn find_fwd(needles: &[u8], haystack: &[u8], at: usize) -> Option<usize> {
    let bs = needles;
    let i = match needles.len() {
        1 => memchr::memchr(bs[0], &haystack[at..])?,
        2 => memchr::memchr2(bs[0], bs[1], &haystack[at..])?,
        3 => memchr::memchr3(bs[0], bs[1], bs[2], &haystack[at..])?,
        0 => panic!("cannot find with empty needles"),
        n => unreachable!("invalid needles length: {}", n),
    };
    Some(at + i)
}

pub fn find_rev(needles: &[u8], haystack: &[u8], at: usize) -> Option<usize> {
    let bs = needles;
    match needles.len() {
        1 => memchr::memrchr(bs[0], &haystack[..at]),
        2 => memchr::memrchr2(bs[0], bs[1], &haystack[..at]),
        3 => memchr::memrchr3(bs[0], bs[1], bs[2], &haystack[..at]),
        0 => panic!("cannot find with empty needles"),
        n => unreachable!("invalid needles length: {}", n),
    }
}

#[derive(Clone)]
pub struct Accels<A> {
    /// A length prefixed slice of contiguous accelerators.
    ///
    /// The first 8 bytes always correspond to the number of accelerators
    /// that follow.
    accels: A,
}

impl Accels<Vec<u8>> {
    pub fn empty() -> Accels<Vec<u8>> {
        Accels { accels: vec![0; 8] }
    }

    pub fn add(&mut self, accel: Accel) {
        self.accels.extend_from_slice(&accel.bytes);
        let len = self.len();
        self.set_len(len + 1);
    }

    fn set_len(&mut self, new_len: usize) {
        bytes::NE::write_u64(new_len as u64, &mut self.accels);
    }
}

impl<'a> Accels<&'a [u8]> {
    pub fn from_bytes(
        slice: &'a [u8],
    ) -> Result<(Accels<&'a [u8]>, usize), DeserializeError> {
        let count = bytes::try_read_u64_as_usize(slice, "accelerators count")?;
        let mut buf = &slice[8..];
        if buf.len() < count * ACCEL_CAP {
            return Err(DeserializeError::buffer_too_small("accelerators"));
        }
        buf = &buf[..count * ACCEL_CAP];
        // If every chunk is valid, then we declare the entire thing is valid.
        for chunk in buf.chunks(ACCEL_CAP) {
            let _ = Accel::from_slice(chunk)?;
        }
        let accels = &slice[..8 + count * ACCEL_CAP];
        Ok((Accels { accels }, accels.len()))
    }
}

impl<A: AsRef<[u8]>> Accels<A> {
    pub fn to_owned(&self) -> Accels<Vec<u8>> {
        Accels { accels: self.accels.as_ref().to_vec() }
    }

    pub fn as_ref(&self) -> Accels<&[u8]> {
        Accels { accels: self.accels.as_ref() }
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.accels.as_ref()
    }

    pub fn get(&self, i: usize) -> Option<Accel> {
        if i >= self.len() {
            return None;
        }
        let slice = self.accels.as_ref();
        let offset = 8 + i * ACCEL_CAP;
        let accel = Accel::from_slice(&slice[offset..])
            .expect("Accels must contain valid accelerators");
        Some(accel)
    }

    pub fn needles(&self, i: usize) -> &[u8] {
        let accels = self.accels.as_ref();
        let offset = 8 + i * ACCEL_CAP;
        let len = accels[offset] as usize;
        &self.accels.as_ref()[offset + 1..offset + 1 + len]
    }

    pub fn len(&self) -> usize {
        bytes::read_u64(self.as_bytes()) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn iter(&self) -> IterAccels<'_, A> {
        IterAccels { accels: self, i: 0 }
    }

    /// Writes these accelerators to the given byte buffer using the indicated
    /// endianness. If the given buffer is too small, then an error is
    /// returned. Upon success, the total number of bytes written is returned.
    /// The number of bytes written is guaranteed to be a multiple of 8.
    pub fn write_to<E: Endian>(
        &self,
        dst: &mut [u8],
    ) -> Result<usize, SerializeError> {
        let nwrite = self.write_to_len();
        assert_eq!(
            nwrite % 8,
            0,
            "expected accelerator bytes written to be a multiple of 8",
        );
        if dst.len() < nwrite {
            return Err(SerializeError::buffer_too_small("accelerators"));
        }

        E::write_u64(self.len() as u64, dst);
        dst[8..nwrite].copy_from_slice(&self.as_bytes()[8..nwrite]);
        Ok(nwrite)
    }

    /// Returns the total number of bytes written by `write_to`.
    pub fn write_to_len(&self) -> usize {
        self.as_bytes().len()
    }
}

impl<A: AsRef<[u8]>> core::fmt::Debug for Accels<A> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "Accels(")?;
        let mut list = f.debug_list();
        for a in self.iter() {
            list.entry(&a);
        }
        list.finish()?;
        write!(f, ")")
    }
}

#[derive(Debug)]
struct IterAccels<'a, A: AsRef<[u8]>> {
    accels: &'a Accels<A>,
    i: usize,
}

impl<'a, A: AsRef<[u8]>> Iterator for IterAccels<'a, A> {
    type Item = Accel;

    fn next(&mut self) -> Option<Accel> {
        let accel = self.accels.get(self.i)?;
        self.i += 1;
        Some(accel)
    }
}

/// Accel represents a structure for determining how to "accelerate" a DFA
/// state.
///
/// Namely, it contains zero or more bytes that must be seen in order for the
/// DFA to leave the state it is associated with. In practice, the actual range
/// is 1 to 3 bytes.
///
/// The purpose of acceleration is to identify states whose vast majority
/// of transitions are just loops back to the same state. For example,
/// in the regex `(?-u)^[^a]+b`, the corresponding DFA will have a state
/// (corresponding to `[^a]+`) where all transitions *except* for `b` loop back
/// to itself. Thus, this state can be "accelerated" by simply looking for the
/// next occurrence of `b` instead of explicitly following transitions.
#[derive(Clone)]
pub struct Accel {
    /// The first byte is the length. Subsequent bytes are the accelerated
    /// bytes.
    ///
    /// Note that we make every accelerator 8 bytes as a slightly wasteful
    /// way of making sure alignment is always correct for state ID sizes of
    /// 1, 2, 4 and 8.
    bytes: [u8; ACCEL_CAP],
}

impl Accel {
    /// Returns an empty accel, where no bytes are accelerated.
    pub fn new() -> Accel {
        Accel { bytes: [0; ACCEL_CAP] }
    }

    /// Returns a verified accelerator derived from the beginning of the given
    /// slice.
    ///
    /// If the slice is not long enough or contains invalid bytes for an
    /// accelerator, then this returns an error.
    pub fn from_slice(mut slice: &[u8]) -> Result<Accel, DeserializeError> {
        slice = &slice[..core::cmp::min(ACCEL_LEN, slice.len())];
        let bytes = slice
            .try_into()
            .map_err(|_| DeserializeError::buffer_too_small("accelerator"))?;
        Accel::from_bytes(bytes)
    }

    /// Returns a verified accelerator derived from raw bytes.
    ///
    /// If the given bytes are invalid, then this returns an error.
    pub fn from_bytes(bytes: [u8; 4]) -> Result<Accel, DeserializeError> {
        if bytes[0] as usize >= ACCEL_LEN {
            return Err(DeserializeError::generic(
                "accelerator bytes cannot have length more than 3",
            ));
        }
        Ok(Accel::from_bytes_unchecked(bytes))
    }

    /// Returns an accelerator derived from raw bytes.
    ///
    /// This does not check whether the given bytes are valid. Invalid bytes
    /// cannot sacrifice memory safety, but may result in panics or silent
    /// logic bugs.
    pub fn from_bytes_unchecked(bytes: [u8; 4]) -> Accel {
        Accel { bytes: [bytes[0], bytes[1], bytes[2], bytes[3], 0, 0, 0, 0] }
    }

    /// Attempts to add the given byte to this accelerator. If the accelerator
    /// is already full then this returns false. Otherwise, returns true.
    ///
    /// If the given byte is already in this accelerator, then it panics.
    pub fn add(&mut self, byte: u8) -> bool {
        if self.len() >= 3 {
            return false;
        }
        self.bytes[self.len() + 1] = byte;
        self.bytes[0] += 1;
        true
    }

    /// Return the number of bytes in this accelerator.
    pub fn len(&self) -> usize {
        self.bytes[0] as usize
    }

    /// Returns true if and only if there are no bytes in this accelerator.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the slice of bytes to accelerate.
    ///
    /// If this accelerator is empty, then this returns an empty slice.
    pub fn needles(&self) -> &[u8] {
        &self.bytes[1..1 + self.len()]
    }

    /// Returns the raw representation of this accelerator as a slice of bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns true if and only if this accelerator will accelerate the given
    /// byte.
    fn contains(&self, byte: u8) -> bool {
        self.needles().iter().position(|&b| b == byte).is_some()
    }
}

impl core::fmt::Debug for Accel {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "Accel(")?;
        let mut set = f.debug_set();
        for &b in self.needles() {
            set.entry(&crate::util::DebugByte(b));
        }
        set.finish()?;
        write!(f, ")")
    }
}
