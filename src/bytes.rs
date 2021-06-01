/*!
A collection of helper functions, types and traits for serializing automata.

This crate defines its own bespoke serialization mechanism for most structures
provided in the public API, namely, automata. A bespoke mechanism was developed
primarily because structures like automata demand a specific binary format.
Attempting to encode their rich structure in an existing serialization
format is just not feasible. Moreover, the format for each structure is
generally designed such that deserialization is cheap. More specifically, that
deserialization can be done in constant time.

In order to achieve this, most of the structures in this crate use an in-memory
representation that very closely corresponds to its binary serialized form.
This pervades and complicates everything, and in some cases, requires dealing
with alignment and reasoning about safety.

This technique does have major advantages. In particular, it permits doing
the potentially costly work of compiling a finite state machine in an offline
manner, and then loading it at runtime not only without having to re-compile
the regex, but even without the code required to do the compilation. This, for
example, permits one to use a pre-compiled DFA not only in environments without
Rust's standard library, but also in environments without a heap.

In the code below, whenever we insert some kind of padding, it's to enforce
an 8-byte alignment, unless otherwise noted. Namely, u64 is the largest
state ID type supported. This is mostly done for convenience as a lowest
common denominator (as opposed to using a minimal padding derived from
the state ID representation in use). Serialized objects with state ID
representations smaller than u64 only need to be aligned to the size of the
state ID representation, but all such alignments are compatible with an 8-byte
alignment. Moreover, a forced 8-byte alignment is only used for things that
don't take up much space. When it matters (e.g., the transition table), the
smallest possible alignment is used.

Also, serialization generally requires the caller to specify endianness,
where as deserialization always assumes native endianness (otherwise cheap
deserialization would be impossible). This implies that serializing a structure
generally requires serializing both its big-endian and little-endian variants,
and then loading the correct one based on the target's endianness.
*/

use core::{cmp, convert::TryInto};

use crate::StateID;

/// An error that occurs when serializing an object from this crate.
///
/// Serialization, as used in this crate, universally refers to the process
/// of transforming a structure (like a DFA) into a custom binary format
/// represented by `&[u8]`. To this end, serialization is generally infallible.
/// However, it can fail when caller provided buffer sizes are too small. When
/// that occurs, a serialization error is reported.
///
/// A `SerializeError` provides no introspection capabilities. Its only
/// supported operation is conversion to a human readable error message.
///
/// This error type implements the `std::error::Error` trait only when the
/// `std` feature is enabled. Otherwise, this type is defined in all
/// configurations.
#[derive(Debug)]
pub struct SerializeError {
    /// The name of the thing that a buffer is too small for.
    ///
    /// Currently, the only kind of serialization error is one that is
    /// committed by a caller: providing a destination buffer that is too
    /// small to fit the serialized object. This makes sense conceptually,
    /// since every valid inhabitant of a type should be serializable.
    ///
    /// This is somewhat exposed in the public API of this crate. For example,
    /// the `to_bytes_{big,little}_endian` APIs return a `Vec<u8>` and are
    /// guaranteed to never panic or error. This is only possible because the
    /// implementation guarantees that it will allocate a `Vec<u8>` that is
    /// big enough.
    ///
    /// In summary, if a new serialization error kind needs to be added, then
    /// it will need careful consideration.
    what: &'static str,
}

impl SerializeError {
    pub(crate) fn buffer_too_small(what: &'static str) -> SerializeError {
        SerializeError { what }
    }
}

impl core::fmt::Display for SerializeError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "destination buffer is too small to write {}", self.what)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SerializeError {}

/// An error that occurs when deserializing an object defined in this crate.
///
/// Serialization, as used in this crate, universally refers to the process
/// of transforming a structure (like a DFA) into a custom binary format
/// represented by `&[u8]`. Deserialization, then, refers to the process of
/// cheaply converting this binary format back to the object's in-memory
/// representation as defined in this crate. To the extent possible,
/// deserialization will report this error whenever this process fails.
///
/// A `DeserializeError` provides no introspection capabilities. Its only
/// supported operation is conversion to a human readable error message.
///
/// This error type implements the `std::error::Error` trait only when the
/// `std` feature is enabled. Otherwise, this type is defined in all
/// configurations.
#[derive(Debug)]
pub struct DeserializeError(DeserializeErrorKind);

#[derive(Debug)]
enum DeserializeErrorKind {
    Generic { msg: &'static str },
    BufferTooSmall { what: &'static str },
    InvalidUsize { what: &'static str },
    InvalidVarint { what: &'static str },
    VersionMismatch { expected: u64, found: u64 },
    EndianMismatch { expected: u64, found: u64 },
    StateSizeMismatch { expected: u64, found: u64 },
    AlignmentMismatch { alignment: u64, address: u64 },
    LabelMismatch { expected: &'static str },
}

impl DeserializeError {
    pub(crate) fn generic(msg: &'static str) -> DeserializeError {
        DeserializeError(DeserializeErrorKind::Generic { msg })
    }

    pub(crate) fn buffer_too_small(what: &'static str) -> DeserializeError {
        DeserializeError(DeserializeErrorKind::BufferTooSmall { what })
    }

    pub(crate) fn invalid_usize(what: &'static str) -> DeserializeError {
        DeserializeError(DeserializeErrorKind::InvalidUsize { what })
    }

    fn invalid_varint(what: &'static str) -> DeserializeError {
        DeserializeError(DeserializeErrorKind::InvalidVarint { what })
    }

    fn version_mismatch(expected: u64, found: u64) -> DeserializeError {
        DeserializeError(DeserializeErrorKind::VersionMismatch {
            expected,
            found,
        })
    }

    fn endian_mismatch(expected: u64, found: u64) -> DeserializeError {
        DeserializeError(DeserializeErrorKind::EndianMismatch {
            expected,
            found,
        })
    }

    fn state_size_mismatch(expected: u64, found: u64) -> DeserializeError {
        DeserializeError(DeserializeErrorKind::StateSizeMismatch {
            expected,
            found,
        })
    }

    fn alignment_mismatch(alignment: u64, address: u64) -> DeserializeError {
        DeserializeError(DeserializeErrorKind::AlignmentMismatch {
            alignment,
            address,
        })
    }

    fn label_mismatch(expected: &'static str) -> DeserializeError {
        DeserializeError(DeserializeErrorKind::LabelMismatch { expected })
    }
}

impl core::fmt::Display for DeserializeError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::DeserializeErrorKind::*;

        match self.0 {
            Generic { msg } => write!(f, "{}", msg),
            BufferTooSmall { what } => {
                write!(f, "buffer is too small to read {}", what)
            }
            InvalidUsize { what } => {
                write!(f, "{} is too big to fit in a usize", what)
            }
            InvalidVarint { what } => {
                write!(f, "could not decode valid varint for {}", what)
            }
            VersionMismatch { expected, found } => write!(
                f,
                "unsupported version: \
                 expected version {} but found version {}",
                expected, found,
            ),
            EndianMismatch { expected, found } => write!(
                f,
                "endianness mismatch: expected 0x{:X} but got 0x{:X}. \
                 (Are you trying to load an object serialized with a \
                 different endianness?)",
                expected, found,
            ),
            StateSizeMismatch { expected, found } => write!(
                f,
                "state size mismatch: caller requested a state size of {}, \
                 but serialized object has a state size of {}",
                expected, found,
            ),
            AlignmentMismatch { alignment, address } => write!(
                f,
                "alignment mismatch: serialize object starts at address \
                 0x{:X}, which is not aligned to a {} byte boundary",
                address, alignment,
            ),
            LabelMismatch { expected } => write!(
                f,
                "label mismatch: start of serialized object should \
                 contain a NUL terminated {:?} label, but a different \
                 label was found",
                expected,
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DeserializeError {}

/// Checks that the given slice has an alignment that matches `S`.
///
/// Since `S` is guaranteed to be one of {u8, u16, u32, u64, usize}, then it
/// follows that if the given slice has the same alignment as `S`, then it can
/// be safely cast to a `&[S]` (assuming a correct length).
pub fn check_alignment<S: StateID>(
    slice: &[u8],
) -> Result<(), DeserializeError> {
    let alignment = core::mem::align_of::<S>() as u64;
    let address = slice.as_ptr() as u64;
    if address % alignment == 0 {
        return Ok(());
    }
    Err(DeserializeError::alignment_mismatch(alignment, address))
}

/// Reads a possibly empty amount of padding, up to 7 bytes, from the beginning
/// of the given slice. All padding bytes must be NUL bytes.
///
/// This is useful because it can be theoretically necessary to pad the
/// beginning of a serialized object with NUL bytes to ensure that it starts
/// at a correctly aligned address. These padding bytes should come immediately
/// before the label.
///
/// This returns the number of bytes read from the given slice.
pub fn skip_initial_padding(slice: &[u8]) -> usize {
    let mut nread = 0;
    while nread < 7 && nread < slice.len() && slice[nread] == 0 {
        nread += 1;
    }
    nread
}

/// Allocate a byte buffer of the giving size, along with some initial padding
/// such that `buf[padding..]` has the same alignment as `S`. In particular,
/// callers should treat the first N bytes (second return value) as padding
/// bytes that must not be overwritten. In all cases, the following identity
/// holds:
///
/// ```ignore
/// let (buf, padding) = alloc_aligned_buffer(SIZE);
/// assert_eq!(SIZE, buf[padding..].len());
/// ```
///
/// In practice, padding is often zero.
pub fn alloc_aligned_buffer<S: StateID>(size: usize) -> (Vec<u8>, usize) {
    // TODO: This is a kludge because there's no easy way to allocate a Vec<u8>
    // with an alignment guaranteed to be greater than 1. We could create a
    // Vec<usize>, but this cannot be safely transmuted to a Vec<u8> without
    // concern, since reallocing or dropping the Vec<u8> is UB (different
    // alignment than the initial allocation).
    let mut buf = vec![0; size];
    let align = core::mem::align_of::<S>();
    let address = buf.as_ptr() as usize;
    if address % align == 0 {
        return (buf, 0);
    }
    // It's not quite clear how to robustly test this code, since the allocator
    // in my environment appears to always return addresses aligned to at
    // least 8 bytes, even when the alignment requirement is smaller. A feeble
    // attempt at ensuring correctness is provided with asserts.
    let padding = ((address & !0b111).checked_add(8).unwrap())
        .checked_sub(address)
        .unwrap();
    assert!(padding <= 7, "padding of {} is bigger than 7", padding);
    buf.extend(core::iter::repeat(0).take(padding));
    assert_eq!(size + padding, buf.len());
    assert_eq!(
        0,
        buf[padding..].as_ptr() as usize % align,
        "expected end of initial padding to be aligned to {}",
        align,
    );
    (buf, padding)
}

/// Reads a NUL terminated label starting at the beginning of the given slice.
///
/// If a NUL terminated label could not be found, then an error is returned.
/// Similary, if a label is found but doesn't match the expected label, then
/// an error is returned.
///
/// Upon success, the total number of bytes read (including padding bytes) is
/// returned.
pub fn read_label(
    slice: &[u8],
    expected_label: &'static str,
) -> Result<usize, DeserializeError> {
    let first_nul =
        slice[..cmp::min(slice.len(), 256)].iter().position(|&b| b == 0);
    let first_nul = match first_nul {
        Some(first_nul) => first_nul,
        None => {
            return Err(DeserializeError::generic(
                "could not find NUL terminated label \
                 at start of serialized object",
            ));
        }
    };
    let len = first_nul + padding_len(first_nul);
    if slice.len() < len {
        return Err(DeserializeError::generic(
            "could not find properly sized label at start of serialized object"
        ));
    }
    if expected_label.as_bytes() != &slice[..first_nul] {
        return Err(DeserializeError::label_mismatch(expected_label));
    }
    Ok(len)
}

/// Writes the given label to the buffer as a NUL terminated string. The label
/// given must not contain NUL, otherwise this will panic. Similarly, the label
/// must not be longer than 255 bytes, otherwise this will panic.
///
/// Additional NUL bytes are written as necessary to ensure that the number of
/// bytes written is always a multiple of 8.
///
/// Upon success, the total number of bytes written (including padding) is
/// returned.
pub fn write_label(
    label: &str,
    dst: &mut [u8],
) -> Result<usize, SerializeError> {
    let nwrite = write_label_len(label);
    if dst.len() < nwrite {
        return Err(SerializeError::buffer_too_small("label"));
    }
    dst[..label.len()].copy_from_slice(label.as_bytes());
    for i in 0..(nwrite - label.len()) {
        dst[label.len() + i] = 0;
    }
    Ok(nwrite)
}

/// Returns the total number of bytes (including padding) that would be written
/// for the given label. This panics if the given label contains a NUL byte or
/// is longer than 255 bytes. (The size restriction exists so that searching
/// for a label during deserialization can be done in small bounded space.)
pub fn write_label_len(label: &str) -> usize {
    if label.len() > 255 {
        panic!("label must not be longer than 255 bytes");
    }
    if label.as_bytes().iter().position(|&b| b == 0).is_some() {
        panic!("label must not contain NUL bytes");
    }
    let label_len = label.len() + 1; // +1 for the NUL terminator
    label_len + padding_len(label_len)
}

/// Reads the endianness check from the beginning of the given slice and
/// confirms that the endianness of the serialized object matches the expected
/// endianness. If the slice is too small or if the endianness check fails,
/// this returns an error.
///
/// Upon success, the total number of bytes read is returned.
pub fn read_endianness_check(slice: &[u8]) -> Result<usize, DeserializeError> {
    let n = try_read_u64(slice, "endianness check")?;
    if n != 0xFEFF {
        return Err(DeserializeError::endian_mismatch(0xFEFF, n));
    }
    Ok(write_endianness_check_len())
}

/// Writes 0xFEFF as an integer using the given endianness.
///
/// This is useful for writing into the header of a serialized object. It can
/// be read during deserialization as a sanity check to ensure the proper
/// endianness is used.
///
/// Upon success, the total number of bytes written is returned.
pub fn write_endianness_check<E: Endian>(
    dst: &mut [u8],
) -> Result<usize, SerializeError> {
    let nwrite = write_endianness_check_len();
    if dst.len() < nwrite {
        return Err(SerializeError::buffer_too_small("endianness check"));
    }
    E::write_u64(0xFEFF, dst);
    Ok(nwrite)
}

/// Returns the number of bytes written by the endianness check.
pub fn write_endianness_check_len() -> usize {
    8
}

/// Reads a version number from the beginning of the given slice and confirms
/// that is matches the expected version number given. If the slice is too
/// small or if the version numbers aren't equivalent, this returns an error.
///
/// Upon success, the total number of bytes read is returned.
///
/// N.B. Currently, we require that the version number is exactly equivalent.
/// In the future, if we bump the version number without a semver bump, then
/// we'll need to relax this a bit and support older versions.
pub fn read_version(
    slice: &[u8],
    expected_version: u64,
) -> Result<usize, DeserializeError> {
    let n = try_read_u64(slice, "version")?;
    if n != expected_version {
        return Err(DeserializeError::version_mismatch(expected_version, n));
    }
    Ok(write_version_len())
}

/// Writes the given version number to the beginning of the given slice.
///
/// This is useful for writing into the header of a serialized object. It can
/// be read during deserialization as a sanity check to ensure that the library
/// code supports the format of the serialized object.
///
/// Upon success, the total number of bytes written is returned.
pub fn write_version<E: Endian>(
    version: u64,
    dst: &mut [u8],
) -> Result<usize, SerializeError> {
    let nwrite = write_version_len();
    if dst.len() < nwrite {
        return Err(SerializeError::buffer_too_small("version number"));
    }
    E::write_u64(version, dst);
    Ok(nwrite)
}

/// Returns the number of bytes written by writing the version number.
pub fn write_version_len() -> usize {
    8
}

/// Reads a state size from the beginning of the given slice and confirms that
/// is matches the expected state size, as determined by the size of `S`. If
/// the slice is too small or if the state sizes aren't equivalent, then an
/// error is returned.
///
/// Upon success, the total number of bytes read is returned.
pub fn read_state_size<S: StateID>(
    slice: &[u8],
) -> Result<usize, DeserializeError> {
    let expected = core::mem::size_of::<S>() as u64;
    let n = try_read_u64(slice, "state size")?;
    if n != expected {
        return Err(DeserializeError::state_size_mismatch(expected, n));
    }
    Ok(write_state_size_len())
}

/// Writes the size of the state ID representation (as determined by `S`) to
/// the beginning of the given slice using the indicated endianness.
///
/// This is useful for writing into the header of a serialized object. It can
/// be read during deserialization as a sanity check to ensure that the caller
/// indicated state size matches the state size of the serialized object.
///
/// Upon success, the total number of bytes written is returned.
pub fn write_state_size<E: Endian, S: StateID>(
    dst: &mut [u8],
) -> Result<usize, SerializeError> {
    let size = core::mem::size_of::<S>() as u64;
    let nwrite = write_state_size_len();
    if dst.len() < nwrite {
        return Err(SerializeError::buffer_too_small("state size"));
    }
    E::write_u64(size, dst);
    Ok(nwrite)
}

/// Returns the number of bytes written by writing the state size.
pub fn write_state_size_len() -> usize {
    8
}

/// Write the given identifier to the beginning of the given slice of bytes
/// using the specified endianness. The given slice must have length at least
/// `size_of::<S>()`, or else this panics. Upon success, the total number of
/// bytes written is returned.
pub fn write_state_id<E: Endian, S: StateID>(id: S, dst: &mut [u8]) -> usize {
    let size = core::mem::size_of::<S>();
    // Guaranteed to pass because StateID is sealed.
    assert!(size == 1 || size == 2 || size == 4 || size == 8);

    match size {
        1 => dst[0] = id.as_usize() as u8,
        2 => E::write_u16(id.as_usize() as u16, dst),
        4 => E::write_u32(id.as_usize() as u32, dst),
        8 => E::write_u64(id.as_usize() as u64, dst),
        _ => unreachable!(),
    }
    size
}

/// Try to read a u64 as a usize from the beginning of the given slice in
/// native endian format. If the slice has fewer than 8 bytes or if the
/// deserialized number cannot be represented by usize, then this returns an
/// error. The error message will include the `what` description of what is
/// being deserialized, for better error messages. `what` should be a noun in
/// singular form.
pub fn try_read_u64_as_usize(
    slice: &[u8],
    what: &'static str,
) -> Result<usize, DeserializeError> {
    if slice.len() < 8 {
        return Err(DeserializeError::buffer_too_small(what));
    }
    read_u64(slice)
        .try_into()
        .map_err(|_| DeserializeError::invalid_usize(what))
}

/// Try to read a u16 from the beginning of the given slice in native endian
/// format. If the slice has fewer than 2 bytes, then this returns an error.
/// The error message will include the `what` description of what is being
/// deserialized, for better error messages. `what` should be a noun in
/// singular form.
pub fn try_read_u16(
    slice: &[u8],
    what: &'static str,
) -> Result<u16, DeserializeError> {
    if slice.len() < 2 {
        return Err(DeserializeError::buffer_too_small(what));
    }
    Ok(read_u16(slice))
}

/// Try to read a u32 from the beginning of the given slice in native endian
/// format. If the slice has fewer than 4 bytes, then this returns an error.
/// The error message will include the `what` description of what is being
/// deserialized, for better error messages. `what` should be a noun in
/// singular form.
pub fn try_read_u32(
    slice: &[u8],
    what: &'static str,
) -> Result<u32, DeserializeError> {
    if slice.len() < 4 {
        return Err(DeserializeError::buffer_too_small(what));
    }
    Ok(read_u32(slice))
}

/// Try to read a u64 from the beginning of the given slice in native endian
/// format. If the slice has fewer than 8 bytes, then this returns an error.
/// The error message will include the `what` description of what is being
/// deserialized, for better error messages. `what` should be a noun in
/// singular form.
pub fn try_read_u64(
    slice: &[u8],
    what: &'static str,
) -> Result<u64, DeserializeError> {
    if slice.len() < 8 {
        return Err(DeserializeError::buffer_too_small(what));
    }
    Ok(read_u64(slice))
}

/// Read a u16 from the beginning of the given slice in native endian format.
/// If the slice has fewer than 2 bytes, then this panics.
pub fn read_u16(slice: &[u8]) -> u16 {
    let bytes: [u8; 2] = slice[..2].try_into().unwrap();
    u16::from_ne_bytes(bytes)
}

/// Read a u32 from the beginning of the given slice in native endian format.
/// If the slice has fewer than 4 bytes, then this panics.
pub fn read_u32(slice: &[u8]) -> u32 {
    let bytes: [u8; 4] = slice[..4].try_into().unwrap();
    u32::from_ne_bytes(bytes)
}

/// Read a u64 from the beginning of the given slice in native endian format.
/// If the slice has fewer than 8 bytes, then this panics.
pub fn read_u64(slice: &[u8]) -> u64 {
    let bytes: [u8; 8] = slice[..8].try_into().unwrap();
    u64::from_ne_bytes(bytes)
}

/// Write a variable sized integer and return the total number of bytes
/// written. If the slice was not big enough to contain the bytes, then this
/// returns an error including the "what" description in it. This does no
/// padding.
///
/// See: https://developers.google.com/protocol-buffers/docs/encoding#varints
pub fn write_varu64(
    mut n: u64,
    what: &'static str,
    dst: &mut [u8],
) -> Result<usize, SerializeError> {
    let mut i = 0;
    while n >= 0b1000_0000 {
        if i >= dst.len() {
            return Err(SerializeError::buffer_too_small(what));
        }
        dst[i] = (n as u8) | 0b1000_0000;
        n >>= 7;
        i += 1;
    }
    if i >= dst.len() {
        return Err(SerializeError::buffer_too_small(what));
    }
    dst[i] = n as u8;
    Ok(i + 1)
}

/// Returns the total number of bytes that would be writen to encode n as a
/// variable sized integer.
///
/// See: https://developers.google.com/protocol-buffers/docs/encoding#varints
pub fn write_varu64_len(mut n: u64) -> usize {
    let mut i = 0;
    while n >= 0b1000_0000 {
        n >>= 7;
        i += 1;
    }
    i + 1
}

/// Like read_varu64, but attempts to cast the result to usize. If the integer
/// cannot fit into a usize, then an error is returned.
pub fn read_varu64_as_usize(
    slice: &[u8],
    what: &'static str,
) -> Result<(u64, usize), DeserializeError> {
    let (n, nread) = read_varu64(slice, what)?;
    let n = n.try_into().map_err(|_| DeserializeError::invalid_usize(what))?;
    Ok((n, nread))
}

/// Reads a variable sized integer from the beginning of slice, and returns the
/// integer along with the total number of bytes read. If a valid variable
/// sized integer could not be found, then an error is returned that includes
/// the "what" description in it.
///
/// https://developers.google.com/protocol-buffers/docs/encoding#varints
pub fn read_varu64(
    slice: &[u8],
    what: &'static str,
) -> Result<(u64, usize), DeserializeError> {
    let mut n: u64 = 0;
    let mut shift: u32 = 0;
    // The biggest possible value is u64::MAX, which needs all 64 bits which
    // requires 10 bytes (because 7 * 9 < 64). We use a limit to avoid reading
    // an unnecessary number of bytes.
    let limit = cmp::min(slice.len(), 10);
    for (i, &b) in slice[..limit].iter().enumerate() {
        if b < 0b1000_0000 {
            return match (b as u64).checked_shl(shift) {
                None => Err(DeserializeError::invalid_varint(what)),
                Some(b) => Ok((n | b, i + 1)),
            };
        }
        match ((b as u64) & 0b0111_1111).checked_shl(shift) {
            None => return Err(DeserializeError::invalid_varint(what)),
            Some(b) => n |= b,
        }
        shift += 7;
    }
    Err(DeserializeError::invalid_varint(what))
}

/// A simple trait for writing code generic over endianness.
///
/// This is similar to what byteorder provides, but we only need a very small
/// subset and can implement it safely due to our higher MSRV.
pub trait Endian {
    /// Writes a u16 to the given destination buffer in a particular
    /// endianness. If the destination buffer has a length smaller than 2, then
    /// this panics.
    fn write_u16(n: u16, dst: &mut [u8]);

    /// Writes a u32 to the given destination buffer in a particular
    /// endianness. If the destination buffer has a length smaller than 4, then
    /// this panics.
    fn write_u32(n: u32, dst: &mut [u8]);

    /// Writes a u64 to the given destination buffer in a particular
    /// endianness. If the destination buffer has a length smaller than 8, then
    /// this panics.
    fn write_u64(n: u64, dst: &mut [u8]);
}

/// Little endian writing.
pub enum LE {}
/// Big endian writing.
pub enum BE {}

#[cfg(target_endian = "little")]
pub type NE = LE;
#[cfg(target_endian = "big")]
pub type NE = BE;

impl Endian for LE {
    fn write_u16(n: u16, dst: &mut [u8]) {
        dst[..2].copy_from_slice(&n.to_le_bytes());
    }

    fn write_u32(n: u32, dst: &mut [u8]) {
        dst[..4].copy_from_slice(&n.to_le_bytes());
    }

    fn write_u64(n: u64, dst: &mut [u8]) {
        dst[..8].copy_from_slice(&n.to_le_bytes());
    }
}

impl Endian for BE {
    fn write_u16(n: u16, dst: &mut [u8]) {
        dst[..2].copy_from_slice(&n.to_be_bytes());
    }

    fn write_u32(n: u32, dst: &mut [u8]) {
        dst[..4].copy_from_slice(&n.to_be_bytes());
    }

    fn write_u64(n: u64, dst: &mut [u8]) {
        dst[..8].copy_from_slice(&n.to_be_bytes());
    }
}

/// Returns the number of additional bytes required to add to the given length
/// in order to make the total length a multiple of 8. The return value is
/// always less than 8.
pub fn padding_len(non_padding_len: usize) -> usize {
    (8 - (non_padding_len & 0b111)) & 0b111
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn labels() {
        let mut buf = [0; 1024];

        let nwrite = write_label("foo", &mut buf).unwrap();
        assert_eq!(nwrite, 8);
        assert_eq!(&buf[..nwrite], b"foo\x00\x00\x00\x00\x00");

        let nread = read_label(&buf, "foo").unwrap();
        assert_eq!(nread, 8);
    }

    #[test]
    #[should_panic]
    fn bad_label_interior_nul() {
        // interior NULs are not allowed
        write_label("foo\x00bar", &mut [0; 1024]);
    }

    #[test]
    fn bad_label_almost_too_long() {
        // ok
        write_label(&"z".repeat(255), &mut [0; 1024]).unwrap();
    }

    #[test]
    #[should_panic]
    fn bad_label_too_long() {
        // labels longer than 255 bytes are banned
        write_label(&"z".repeat(256), &mut [0; 1024]);
    }

    #[test]
    fn padding() {
        assert_eq!(0, padding_len(8));
        assert_eq!(7, padding_len(9));
        assert_eq!(6, padding_len(10));
        assert_eq!(5, padding_len(11));
        assert_eq!(4, padding_len(12));
        assert_eq!(3, padding_len(13));
        assert_eq!(2, padding_len(14));
        assert_eq!(1, padding_len(15));
        assert_eq!(0, padding_len(16));
    }
}
