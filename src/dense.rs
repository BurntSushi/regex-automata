#![allow(warnings)]

use std::fmt;
use std::iter;
use std::mem;
use std::slice;

use byteorder::{ByteOrder, BigEndian, LittleEndian, NativeEndian};

use builder::DenseDFABuilder;
use classes::ByteClasses;
use dfa::DFA;
use error::{Error, Result};
use minimize::Minimizer;
use sparse::SparseDFA;
use state_id::{StateID, dead_id, next_state_id, premultiply_overflow_error};

/// The size of the alphabet in a standard DFA.
///
/// Specifically, this length controls the number of transitions present in
/// each DFA state. However, when the byte class optimization is enabled,
/// then each DFA maps the space of all possible 256 byte values to at most
/// 256 distinct equivalence classes. In this case, the number of distinct
/// equivalence classes corresponds to the internal alphabet of the DFA, in the
/// sense that each DFA state has a number of transitions equal to the number
/// of equivalence classes despite supporting matching on all possible byte
/// values.
pub const ALPHABET_LEN: usize = 256;

/// A heap allocated table-based deterministic finite automaton (DFA).
///
/// A DFA represents the core matching primitive in this crate. That is,
/// logically, all DFAs have a single start state, one or more match states
/// and a transition table that maps the current state and the current byte of
/// input to the next state. A DFA can use this information to implement fast
/// searching. In particular, the use of a DFA generally makes the trade off
/// that match speed is the most valuable characteristic, even if building the
/// regex may take significant time *and* space. As such, the processing of
/// every byte of input is done with a small constant number of operations
/// that does not vary with the pattern, its size or the size of the alphabet.
/// If your needs don't line up with this trade off, then a DFA may not be an
/// adequate solution to your problem.
///
/// A DFA can be built using the default configuration via the
/// [`DenseDFA::new`](struct.DenseDFA.html#method.new) constructor. Otherwise, one can
/// configure various aspects via the [`DenseDFABuilder`](struct.DenseDFABuilder.html).
///
/// A single DFA fundamentally supports the following operations:
///
/// 1. Detection of a match.
/// 2. Location of the end of the first possible match.
/// 3. Location of the end of the leftmost-first match.
///
/// A notable absence from the above list of capabilities is the location of
/// the *start* of a match. In order to provide both the start and end of a
/// match, *two* DFAs are required. This functionality is provided by a
/// [`Regex`](struct.Regex.html), which can be built with its basic
/// constructor, [`Regex::new`](struct.Regex.html#method.new), or with
/// a [`RegexBuilder`](struct.RegexBuilder.html).
///
/// # State size
///
/// A `DenseDFA` has a single type parameter, `S`, which corresponds to the
/// representation used for the DFA's state identifiers as described by the
/// [`StateID`](trait.StateID.html) trait. This type parameter is, by default,
/// set to `usize`. Other valid choices provided by this crate include `u8`,
/// `u16`, `u32` and `u64`. The primary reason for choosing a different state
/// identifier representation than the default is to reduce the amount of
/// memory used by a DFA. Note though, that if the chosen representation cannot
/// accommodate the size of your DFA, then building the DFA will fail and
/// return an error.
///
/// While the reduction in heap memory used by a DFA is one reason for choosing
/// a smaller state identifier representation, another possible reason is for
/// decreasing the serialization size of a DFA, as returned by
/// [`to_bytes_little_endian`](struct.DenseDFA.html#method.to_bytes_little_endian),
/// [`to_bytes_big_endian`](struct.DenseDFA.html#method.to_bytes_big_endian)
/// or
/// [`to_bytes_native_endian`](struct.DenseDFA.html#method.to_bytes_native_endian).
#[derive(Clone, Debug)]
pub enum DenseDFA<T: AsRef<[S]>, S: StateID> {
    Standard(Standard<T, S>),
    ByteClass(ByteClass<T, S>),
    Premultiplied(Premultiplied<T, S>),
    PremultipliedByteClass(PremultipliedByteClass<T, S>),
    /// Hints that destructuring should not be exhaustive.
    ///
    /// This enum may grow additional variants, so this makes sure clients
    /// don't count on exhaustive matching. (Otherwise, adding a new variant
    /// could break existing code.)
    #[doc(hidden)]
    __Nonexhaustive,
}

impl<T: AsRef<[S]>, S: StateID> DenseDFA<T, S> {
    /// Return the internal DFA representation.
    ///
    /// All variants share the same internal representation.
    fn repr(&self) -> &Repr<T, S> {
        match *self {
            DenseDFA::Standard(ref r) => &r.0,
            DenseDFA::ByteClass(ref r) => &r.0,
            DenseDFA::Premultiplied(ref r) => &r.0,
            DenseDFA::PremultipliedByteClass(ref r) => &r.0,
            DenseDFA::__Nonexhaustive => unreachable!(),
        }
    }
}

impl DenseDFA<Vec<usize>, usize> {
    /// Parse the given regular expression using a default configuration and
    /// return the corresponding DFA.
    ///
    /// The default configuration uses `usize` for state IDs, premultiplies
    /// them and reduces the alphabet size by splitting bytes into equivalence
    /// classes. The DFA is *not* minimized.
    ///
    /// If you want a non-default configuration, then use the
    /// [`DenseDFABuilder`](struct.DenseDFABuilder.html)
    /// to set your own configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{DFA, DenseDFA};
    ///
    /// # fn example() -> Result<(), regex_automata::Error> {
    /// let dfa = DenseDFA::new("foo[0-9]+bar")?;
    /// assert_eq!(Some(11), dfa.find(b"foo12345bar"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn new(pattern: &str) -> Result<DenseDFA<Vec<usize>, usize>> {
        DenseDFABuilder::new().build(pattern)
    }
}

impl<S: StateID> DenseDFA<Vec<S>, S> {
    /// Parse the given regular expression using a default configuration and
    /// Create a new empty DFA that never matches any input.
    ///
    /// # Example
    ///
    /// In order to build an empty DFA, callers must provide a type hint
    /// indicating their choice of state identifier representation.
    ///
    /// ```
    /// use regex_automata::{DFA, DenseDFA};
    ///
    /// # fn example() -> Result<(), regex_automata::Error> {
    /// let dfa: DenseDFA<Vec<usize>, usize> = DenseDFA::empty();
    /// assert_eq!(None, dfa.find(b""));
    /// assert_eq!(None, dfa.find(b"foo"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn empty() -> DenseDFA<Vec<S>, S> {
        Repr::empty().into_dense_dfa()
    }
}

impl<T: AsRef<[S]>, S: StateID> DenseDFA<T, S> {
    /// Returns the memory usage, in bytes, of this DFA.
    ///
    /// The memory usage is computed based on the number of bytes used to
    /// represent this DFA's transition table. This corresponds to heap memory
    /// usage.
    ///
    /// This does **not** include the stack size used up by this DFA. To
    /// compute that, used `std::mem::size_of::<DenseDFA>()`.
    pub fn memory_usage(&self) -> usize {
        self.repr().memory_usage()
    }
}

impl<T: AsRef<[S]>, S: StateID> DenseDFA<T, S> {
    /// TODO...
    pub fn to_sparse(&self) -> Result<SparseDFA<Vec<u8>, S>> {
        self.to_sparse_sized()
    }

    /// TODO...
    pub fn to_sparse_sized<A: StateID>(
        &self,
    ) -> Result<SparseDFA<Vec<u8>, A>> {
        self.repr().to_sparse_sized()
    }

    /// Create a new DFA whose match semantics are equivalent to this DFA,
    /// but attempt to use `u8` for the representation of state identifiers.
    /// If `u8` is insufficient to represent all state identifiers in this
    /// DFA, then this returns an error.
    ///
    /// This is a convenience routine for `to_sized::<u8>()`.
    pub fn to_u8(&self) -> Result<DenseDFA<Vec<u8>, u8>> {
        self.to_sized()
    }

    /// Create a new DFA whose match semantics are equivalent to this DFA,
    /// but attempt to use `u16` for the representation of state identifiers.
    /// If `u16` is insufficient to represent all state identifiers in this
    /// DFA, then this returns an error.
    ///
    /// This is a convenience routine for `to_sized::<u16>()`.
    pub fn to_u16(&self) -> Result<DenseDFA<Vec<u16>, u16>> {
        self.to_sized()
    }

    /// Create a new DFA whose match semantics are equivalent to this DFA,
    /// but attempt to use `u32` for the representation of state identifiers.
    /// If `u32` is insufficient to represent all state identifiers in this
    /// DFA, then this returns an error.
    ///
    /// This is a convenience routine for `to_sized::<u32>()`.
    pub fn to_u32(&self) -> Result<DenseDFA<Vec<u32>, u32>> {
        self.to_sized()
    }

    /// Create a new DFA whose match semantics are equivalent to this DFA,
    /// but attempt to use `u64` for the representation of state identifiers.
    /// If `u64` is insufficient to represent all state identifiers in this
    /// DFA, then this returns an error.
    ///
    /// This is a convenience routine for `to_sized::<u64>()`.
    pub fn to_u64(&self) -> Result<DenseDFA<Vec<u64>, u64>> {
        self.to_sized()
    }

    /// Create a new DFA whose match semantics are equivalent to this DFA, but
    /// attempt to use `A` for the representation of state identifiers. If `A`
    /// is insufficient to represent all state identifiers in this DFA, then
    /// this returns an error.
    ///
    /// An alternative way to construct such a DFA is to use
    /// [`DenseDFABuilder::build_with_size`](struct.DenseDFABuilder.html#method.build_with_size).
    /// In general, using the builder is preferred since it will use the given
    /// state identifier representation throughout determinization (and
    /// minimization, if done), and thereby using less memory throughout the
    /// entire construction process. However, these routines are necessary
    /// in cases where, say, a minimized DFA could fit in a smaller state
    /// identifier representation, but the initial determinized DFA would not.
    pub fn to_sized<A: StateID>(&self) -> Result<DenseDFA<Vec<A>, A>> {
        self.repr().to_sized().map(|r| r.into_dense_dfa())
    }

    /// Serialize a DFA to raw bytes, aligned to an 8 byte boundary, in little
    /// endian format.
    ///
    /// If the state identifier representation of this DFA has a size different
    /// than 1, 2, 4 or 8 bytes, then this returns an error. All
    /// implementations of `StateID` provided by this crate satisfy this
    /// requirement.
    pub fn to_bytes_little_endian(&self) -> Result<Vec<u8>> {
        self.repr().to_bytes::<LittleEndian>()
    }

    /// Serialize a DFA to raw bytes, aligned to an 8 byte boundary, in big
    /// endian format.
    ///
    /// If the state identifier representation of this DFA has a size different
    /// than 1, 2, 4 or 8 bytes, then this returns an error. All
    /// implementations of `StateID` provided by this crate satisfy this
    /// requirement.
    pub fn to_bytes_big_endian(&self) -> Result<Vec<u8>> {
        self.repr().to_bytes::<BigEndian>()
    }

    /// Serialize a DFA to raw bytes, aligned to an 8 byte boundary, in native
    /// endian format. Generally, it is better to pick an explicit endianness
    /// using either `to_bytes_little_endian` or `to_bytes_big_endian`. This
    /// routine is useful in tests where the DFA is serialized and deserialized
    /// on the same platform.
    ///
    /// If the state identifier representation of this DFA has a size different
    /// than 1, 2, 4 or 8 bytes, then this returns an error. All
    /// implementations of `StateID` provided by this crate satisfy this
    /// requirement.
    pub fn to_bytes_native_endian(&self) -> Result<Vec<u8>> {
        self.repr().to_bytes::<NativeEndian>()
    }
}

impl<'a, S: StateID> DenseDFA<&'a [S], S> {
    /// Deserialize a DFA with a specific state identifier representation.
    ///
    /// Deserializing a DFA using this routine will allocate new heap memory
    /// for the transition table.
    ///
    /// The bytes given should be generated by the serialization of a DFA with
    /// either the
    /// [`to_bytes_little_endian`](struct.DenseDFA.html#method.to_bytes_little_endian)
    /// method or the
    /// [`to_bytes_big_endian`](struct.DenseDFA.html#method.to_bytes_big_endian)
    /// endian, depending on the endianness of the machine you are
    /// deserializing this DFA from.
    ///
    /// If the state identifier representation is `usize`, then deserialization
    /// is dependent on the pointer size. For this reason, it is best to
    /// serialize DFAs using a fixed size representation for your state
    /// identifiers, such as `u8`, `u16`, `u32` or `u64`.
    ///
    /// If you're loading a DFA from a memory mapped file or static memory,
    /// then you probably want to use
    /// [`DenseDFARef::from_bytes`](struct.DenseDFARef.html#method.from_bytes)
    /// instead. In particular, using `DenseDFARef` will not use any heap
    /// memory, is suitable for `no_std` environments and is a constant time
    /// operation.
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
    /// the various serialization routines will never return an incorrect
    /// transition table, there is no guarantee that the bytes provided here
    /// are correct. While deserialization does many checks (as documented
    /// above in the panic conditions), this routine does not check that the
    /// transition table is correct. Given an incorrect transition table, it is
    /// possible for the search routines to access out-of-bounds memory because
    /// of explicit bounds check elision.
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
    /// use regex_automata::{DFA, DenseDFA};
    ///
    /// # fn example() -> Result<(), regex_automata::Error> {
    /// let initial = DenseDFA::new("foo[0-9]+")?;
    /// let bytes = initial.to_u16()?.to_bytes_native_endian()?;
    /// let dfa: DenseDFA<&[u16], u16> = unsafe {
    ///     DenseDFA::from_bytes(&bytes)
    /// };
    ///
    /// assert_eq!(Some(8), dfa.find(b"foo12345"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub unsafe fn from_bytes(buf: &'a [u8]) -> DenseDFA<&'a [S], S> {
        Repr::from_bytes(buf).into_dense_dfa()
    }
}

impl<S: StateID> DenseDFA<Vec<S>, S> {
    /// Minimize this DFA in place.
    ///
    /// This is not part of the public API. It is only exposed to allow for
    /// more granular external benchmarking.
    #[doc(hidden)]
    pub fn minimize(&mut self) {
        self.repr_mut().minimize();
    }

    /// Return a mutable reference to the internal DFA representation.
    fn repr_mut(&mut self) -> &mut Repr<Vec<S>, S> {
        match *self {
            DenseDFA::Standard(ref mut r) => &mut r.0,
            DenseDFA::ByteClass(ref mut r) => &mut r.0,
            DenseDFA::Premultiplied(ref mut r) => &mut r.0,
            DenseDFA::PremultipliedByteClass(ref mut r) => &mut r.0,
            DenseDFA::__Nonexhaustive => unreachable!(),
        }
    }
}

impl<T: AsRef<[S]>, S: StateID> DFA for DenseDFA<T, S> {
    type ID = S;

    fn start_state(&self) -> S {
        self.repr().start_state()
    }

    fn is_match_state(&self, id: S) -> bool {
        self.repr().is_match_state(id)
    }

    fn is_possible_match_state(&self, id: S) -> bool {
        self.repr().is_possible_match_state(id)
    }

    fn is_dead_state(&self, id: S) -> bool {
        self.repr().is_dead_state(id)
    }

    fn next_state(&self, current: S, input: u8) -> S {
        match *self {
            DenseDFA::Standard(ref r) => r.next_state(current, input),
            DenseDFA::ByteClass(ref r) => r.next_state(current, input),
            DenseDFA::Premultiplied(ref r) => r.next_state(current, input),
            DenseDFA::PremultipliedByteClass(ref r) => {
                r.next_state(current, input)
            }
            DenseDFA::__Nonexhaustive => unreachable!(),
        }
    }

    unsafe fn next_state_unchecked(&self, current: S, input: u8) -> S {
        match *self {
            DenseDFA::Standard(ref r) => {
                r.next_state_unchecked(current, input)
            }
            DenseDFA::ByteClass(ref r) => {
                r.next_state_unchecked(current, input)
            }
            DenseDFA::Premultiplied(ref r) => {
                r.next_state_unchecked(current, input)
            }
            DenseDFA::PremultipliedByteClass(ref r) => {
                r.next_state_unchecked(current, input)
            }
            DenseDFA::__Nonexhaustive => unreachable!(),
        }
    }

    // We specialize the following methods because it lets us lift the
    // case analysis between the different types of sparse DFAs. Instead of
    // doing the case analysis for every transition, we do it once before
    // searching. For sparse DFAs, this doesn't seem to benefit performance as
    // much as it does for the dense DFAs, but it's easy to do so we might as
    // well do it.

    fn is_match(&self, bytes: &[u8]) -> bool {
        match *self {
            DenseDFA::Standard(ref r) => r.is_match(bytes),
            DenseDFA::ByteClass(ref r) => r.is_match(bytes),
            DenseDFA::Premultiplied(ref r) => r.is_match(bytes),
            DenseDFA::PremultipliedByteClass(ref r) => r.is_match(bytes),
            DenseDFA::__Nonexhaustive => unreachable!(),
        }
    }

    fn shortest_match(&self, bytes: &[u8]) -> Option<usize> {
        match *self {
            DenseDFA::Standard(ref r) => r.shortest_match(bytes),
            DenseDFA::ByteClass(ref r) => r.shortest_match(bytes),
            DenseDFA::Premultiplied(ref r) => r.shortest_match(bytes),
            DenseDFA::PremultipliedByteClass(ref r) => r.shortest_match(bytes),
            DenseDFA::__Nonexhaustive => unreachable!(),
        }
    }

    fn find(&self, bytes: &[u8]) -> Option<usize> {
        match *self {
            DenseDFA::Standard(ref r) => r.find(bytes),
            DenseDFA::ByteClass(ref r) => r.find(bytes),
            DenseDFA::Premultiplied(ref r) => r.find(bytes),
            DenseDFA::PremultipliedByteClass(ref r) => r.find(bytes),
            DenseDFA::__Nonexhaustive => unreachable!(),
        }
    }

    fn rfind(&self, bytes: &[u8]) -> Option<usize> {
        match *self {
            DenseDFA::Standard(ref r) => r.rfind(bytes),
            DenseDFA::ByteClass(ref r) => r.rfind(bytes),
            DenseDFA::Premultiplied(ref r) => r.rfind(bytes),
            DenseDFA::PremultipliedByteClass(ref r) => r.rfind(bytes),
            DenseDFA::__Nonexhaustive => unreachable!(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Standard<T: AsRef<[S]>, S: StateID>(Repr<T, S>);

impl<T: AsRef<[S]>, S: StateID> DFA for Standard<T, S> {
    type ID = S;

    fn start_state(&self) -> S {
        self.0.start_state()
    }

    fn is_match_state(&self, id: S) -> bool {
        self.0.is_match_state(id)
    }

    fn is_possible_match_state(&self, id: S) -> bool {
        self.0.is_possible_match_state(id)
    }

    fn is_dead_state(&self, id: S) -> bool {
        self.0.is_dead_state(id)
    }

    fn next_state(&self, current: S, input: u8) -> S {
        let o = current.to_usize() * ALPHABET_LEN + input as usize;
        self.0.trans()[o]
    }

    unsafe fn next_state_unchecked(&self, current: S, input: u8) -> S {
        let o = current.to_usize() * ALPHABET_LEN + input as usize;
        *self.0.trans().get_unchecked(o)
    }
}

#[derive(Clone, Debug)]
pub struct ByteClass<T: AsRef<[S]>, S: StateID>(Repr<T, S>);

impl<T: AsRef<[S]>, S: StateID> DFA for ByteClass<T, S> {
    type ID = S;

    fn start_state(&self) -> S {
        self.0.start_state()
    }

    fn is_match_state(&self, id: S) -> bool {
        self.0.is_match_state(id)
    }

    fn is_possible_match_state(&self, id: S) -> bool {
        self.0.is_possible_match_state(id)
    }

    fn is_dead_state(&self, id: S) -> bool {
        self.0.is_dead_state(id)
    }

    fn next_state(&self, current: S, input: u8) -> S {
        let input = self.0.byte_classes.get(input);
        let o = current.to_usize() * self.0.alphabet_len() + input as usize;
        self.0.trans()[o]
    }

    unsafe fn next_state_unchecked(&self, current: S, input: u8) -> S {
        let input = self.0.byte_classes.get_unchecked(input);
        let o = current.to_usize() * self.0.alphabet_len() + input as usize;
        *self.0.trans().get_unchecked(o)
    }
}

#[derive(Clone, Debug)]
pub struct Premultiplied<T: AsRef<[S]>, S: StateID>(Repr<T, S>);

impl<T: AsRef<[S]>, S: StateID> DFA for Premultiplied<T, S> {
    type ID = S;

    fn start_state(&self) -> S {
        self.0.start_state()
    }

    fn is_match_state(&self, id: S) -> bool {
        self.0.is_match_state(id)
    }

    fn is_possible_match_state(&self, id: S) -> bool {
        self.0.is_possible_match_state(id)
    }

    fn is_dead_state(&self, id: S) -> bool {
        self.0.is_dead_state(id)
    }

    fn next_state(&self, current: S, input: u8) -> S {
        let o = current.to_usize() + input as usize;
        self.0.trans()[o]
    }

    unsafe fn next_state_unchecked(&self, current: S, input: u8) -> S {
        let o = current.to_usize() + input as usize;
        *self.0.trans().get_unchecked(o)
    }
}

#[derive(Clone, Debug)]
pub struct PremultipliedByteClass<T: AsRef<[S]>, S: StateID>(Repr<T, S>);

impl<T: AsRef<[S]>, S: StateID> DFA for PremultipliedByteClass<T, S> {
    type ID = S;

    fn start_state(&self) -> S {
        self.0.start_state()
    }

    fn is_match_state(&self, id: S) -> bool {
        self.0.is_match_state(id)
    }

    fn is_possible_match_state(&self, id: S) -> bool {
        self.0.is_possible_match_state(id)
    }

    fn is_dead_state(&self, id: S) -> bool {
        self.0.is_dead_state(id)
    }

    fn next_state(&self, current: S, input: u8) -> S {
        let input = self.0.byte_classes.get(input);
        let o = current.to_usize() + input as usize;
        self.0.trans()[o]
    }

    unsafe fn next_state_unchecked(&self, current: S, input: u8) -> S {
        let input = self.0.byte_classes.get_unchecked(input);
        let o = current.to_usize() + input as usize;
        *self.0.trans().get_unchecked(o)
    }
}

/// The internal representation of a dense DFA.
///
/// This representation is shared by all DFA variants.
#[derive(Clone)]
pub(crate) struct Repr<T, S> {
    /// Whether the state identifiers in the transition table have been
    /// premultiplied or not.
    ///
    /// Premultiplied identifiers means that instead of your matching loop
    /// looking something like this:
    ///
    ///   state = dfa.start
    ///   for byte in haystack:
    ///       next = dfa.transitions[state * len(alphabet) + byte]
    ///       if dfa.is_match(next):
    ///           return true
    ///   return false
    ///
    /// it can instead look like this:
    ///
    ///   state = dfa.start
    ///   for byte in haystack:
    ///       next = dfa.transitions[state + byte]
    ///       if dfa.is_match(next):
    ///           return true
    ///   return false
    ///
    /// In other words, we save a multiplication instruction in the critical
    /// path. This turns out to be a decent performance win. The cost of using
    /// premultiplied state ids is that they can require a bigger state id
    /// representation.
    premultiplied: bool,
    /// The initial start state ID.
    start: S,
    /// The total number of states in this DFA. Note that a DFA always has at
    /// least one state---the dead state---even the empty DFA. In particular,
    /// the dead state always has ID 0 and is correspondingly always the first
    /// state. The dead state is never a match state.
    state_count: usize,
    /// States in a DFA have a *partial* ordering such that a match state
    /// always precedes any non-match state (except for the special dead
    /// state).
    ///
    /// `max_match` corresponds to the last state that is a match state. This
    /// encoding has two critical benefits. Firstly, we are not required to
    /// store any additional per-state information about whether it is a match
    /// state or not. Secondly, when searching with the DFA, we can do a single
    /// comparison with `max_match` for each byte instead of two comparisons
    /// for each byte (one testing whether it is a match and the other testing
    /// whether we've reached a dead state). Namely, to determine the status
    /// of the next state, we can do this:
    ///
    ///   next_state = transition[cur_state * alphabet_len + cur_byte]
    ///   if next_state <= max_match:
    ///       // next_state is either dead (no-match) or a match
    ///       return next_state != dead
    max_match: S,
    /// A set of equivalence classes, where a single equivalence class
    /// represents a set of bytes that never discriminate between a match
    /// and a non-match in the DFA. Each equivalence class corresponds to
    /// a single letter in this DFA's alphabet, where the maximum number of
    /// letters is 256 (each possible value of a byte). Consequently, the
    /// number of equivalence classes corresponds to the number of transitions
    /// for each DFA state.
    ///
    /// The only time the number of equivalence classes is fewer than 256 is
    /// if the DFA's kind uses byte classes. If the DFA doesn't use byte
    /// classes, then this vector is empty.
    byte_classes: ByteClasses,
    /// A contiguous region of memory representing the transition table in
    /// row-major order. The representation is dense. That is, every state has
    /// precisely the same number of transitions. The maximum number of
    /// transitions is 256. If a DFA has been instructed to use byte classes,
    /// then the number of transitions can be much less.
    ///
    /// In practice, T is either Vec<S> or &[S].
    trans: T,
}

impl<S: StateID> Repr<Vec<S>, S> {
    /// Create a new empty DFA with singleton byte classes (every byte is its
    /// own equivalence class).
    pub fn empty() -> Repr<Vec<S>, S> {
        Repr::empty_with_byte_classes(ByteClasses::singletons())
    }

    /// Create a new empty DFA with the given set of byte equivalence classes.
    /// An empty DFA never matches any input.
    pub fn empty_with_byte_classes(
        byte_classes: ByteClasses,
    ) -> Repr<Vec<S>, S> {
        let mut dfa = Repr {
            premultiplied: false,
            start: dead_id(),
            state_count: 0,
            max_match: S::from_usize(1),
            byte_classes: byte_classes,
            trans: vec![],
        };
        // Every state ID repr must be able to fit at least one state.
        dfa.add_empty_state().unwrap();
        dfa
    }
}

impl<T: AsRef<[S]>, S: StateID> Repr<T, S> {
    /// Convert this internal DFA representation to a DenseDFA based on its
    /// transition table access pattern.
    pub fn into_dense_dfa(self) -> DenseDFA<T, S> {
        match (self.premultiplied, self.byte_classes.is_singleton()) {
            // no premultiplication, no byte classes
            (false, true) => DenseDFA::Standard(Standard(self)),
            // no premultiplication, yes byte classes
            (false, false) => DenseDFA::ByteClass(ByteClass(self)),
            // yes premultiplication, no byte classes
            (true, true) => DenseDFA::Premultiplied(Premultiplied(self)),
            // yes premultiplication, yes byte classes
            (true, false) => {
                DenseDFA::PremultipliedByteClass(PremultipliedByteClass(self))
            }
        }
    }

    /// Return the starting state of this DFA.
    ///
    /// All searches using this DFA must begin at this state. There is exactly
    /// one starting state for every DFA. A starting state may be a dead state
    /// or a matching state or neither.
    pub fn start_state(&self) -> S {
        self.start
    }

    /// Returns true if and only if the given identifier corresponds to a dead
    /// state.
    pub fn is_dead_state(&self, id: S) -> bool {
        id == dead_id()
    }

    /// Returns true if and only if the given identifier corresponds to a match
    /// state.
    pub fn is_match_state(&self, id: S) -> bool {
        id <= self.max_match && id != dead_id()
    }

    /// Returns true if and only if the given identifier could correspond to
    /// either a match state or a dead state. If this returns false, then the
    /// given identifier does not correspond to either a match state or a dead
    /// state.
    pub fn is_possible_match_state(&self, id: S) -> bool {
        id <= self.max_match
    }

    /// Returns the maximum identifier for which a match state can exist.
    ///
    /// More specifically, the return identifier always corresponds to either
    /// a match state or a dead state. Namely, either
    /// `is_match_state(returned)` or `is_dead_state(returned)` is guaranteed
    /// to be true.
    pub fn max_match_state(&self) -> S {
        self.max_match
    }

    /// Return the byte classes used by this DFA.
    pub fn byte_classes(&self) -> &ByteClasses {
        &self.byte_classes
    }

    /// Returns an iterator over all states in this DFA.
    ///
    /// This iterator yields a tuple for each state. The first element of the
    /// tuple corresponds to a state's identifier, and the second element
    /// corresponds to the state itself (comprised of its transitions).
    ///
    /// If this DFA is premultiplied, then the state identifiers are in
    /// turn premultiplied as well, making them usable without additional
    /// modification.
    pub fn states(&self) -> StateIter<T, S> {
        let it = self.trans().chunks(self.alphabet_len());
        StateIter { dfa: self, it: it.enumerate() }
    }

    /// Return the total number of states in this DFA. Every DFA has at least
    /// 1 state, even the empty DFA.
    pub fn state_count(&self) -> usize {
        self.state_count
    }

    /// Return the number of elements in this DFA's alphabet.
    ///
    /// If this DFA doesn't use byte classes, then this is always equivalent
    /// to 256. Otherwise, it is guaranteed to be some value less than or equal
    /// to 256.
    pub fn alphabet_len(&self) -> usize {
        self.byte_classes.alphabet_len()
    }

    /// Returns the memory usage, in bytes, of this DFA.
    pub fn memory_usage(&self) -> usize {
        self.trans().len() * mem::size_of::<S>()
    }

    /// Convert the given state identifier to the state's index. The state's
    /// index corresponds to the position in which it appears in the transition
    /// table. When a DFA is NOT premultiplied, then a state's identifier is
    /// also its index. When a DFA is premultiplied, then a state's identifier
    /// is equal to `index * alphabet_len`. This routine reverses that.
    pub fn state_id_to_index(&self, id: S) -> usize {
        if self.premultiplied {
            id.to_usize() / self.alphabet_len()
        } else {
            id.to_usize()
        }
    }

    /// Return this DFA's transition table as a slice.
    fn trans(&self) -> &[S] {
        self.trans.as_ref()
    }

    /// Create a sparse DFA from the internal representation of a dense DFA.
    pub fn to_sparse_sized<A: StateID>(
        &self,
    ) -> Result<SparseDFA<Vec<u8>, A>> {
        SparseDFA::from_dense_sized(self)
    }

    /// Create a new DFA whose match semantics are equivalent to this DFA, but
    /// attempt to use `A` for the representation of state identifiers. If `A`
    /// is insufficient to represent all state identifiers in this DFA, then
    /// this returns an error.
    pub fn to_sized<A: StateID>(&self) -> Result<Repr<Vec<A>, A>> {
        // Check that this DFA can fit into A's representation.
        let mut last_state_id = self.state_count - 1;
        if self.premultiplied {
            last_state_id *= self.alphabet_len();
        }
        if last_state_id > A::max_id() {
            return Err(Error::state_id_overflow(A::max_id()));
        }

        // We're off to the races. The new DFA is the same as the old one,
        // but its transition table is truncated.
        let mut new = Repr {
            premultiplied: self.premultiplied,
            start: A::from_usize(self.start.to_usize()),
            state_count: self.state_count,
            max_match: A::from_usize(self.max_match.to_usize()),
            byte_classes: self.byte_classes.clone(),
            trans: vec![dead_id::<A>(); self.trans().len()],
        };
        for (i, id) in new.trans.iter_mut().enumerate() {
            *id = A::from_usize(self.trans()[i].to_usize());
        }
        Ok(new)
    }

    /// Serialize a DFA to raw bytes, aligned to an 8 byte boundary.
    ///
    /// If the state identifier representation of this DFA has a size different
    /// than 1, 2, 4 or 8 bytes, then this returns an error. All
    /// implementations of `StateID` provided by this crate satisfy this
    /// requirement.
    pub(crate) fn to_bytes<A: ByteOrder>(&self) -> Result<Vec<u8>> {
        let label = b"rust-regex-automata-dfa\x00";
        assert_eq!(24, label.len());

        let trans_size = mem::size_of::<S>() * self.trans().len();
        let size =
            // For human readable label.
            label.len()
            // endiannes check, must be equal to 0xFEFF for native endian
            + 2
            // For version number.
            + 2
            // Size of state ID representation, in bytes.
            // Must be 1, 2, 4 or 8.
            + 2
            // For DFA misc options.
            + 2
            // For start state.
            + 8
            // For state count.
            + 8
            // For max match state.
            + 8
            // For byte class map.
            + 256
            // For transition table.
            + trans_size;
        // sanity check, this can be updated if need be
        assert_eq!(312 + trans_size, size);
        // This must always pass. It checks that the transition table is at
        // a properly aligned address.
        assert_eq!(0, (size - trans_size) % 8);

        let mut buf = vec![0; size];
        let mut i = 0;

        // write label
        for &b in label {
            buf[i] = b;
            i += 1;
        }
        // endianness check
        A::write_u16(&mut buf[i..], 0xFEFF);
        i += 2;
        // version number
        A::write_u16(&mut buf[i..], 1);
        i += 2;
        // size of state ID
        let state_size = mem::size_of::<S>();
        if ![1, 2, 4, 8].contains(&state_size) {
            return Err(Error::serialize(&format!(
                "state size of {} not supported, must be 1, 2, 4 or 8",
                state_size
            )));
        }
        A::write_u16(&mut buf[i..], state_size as u16);
        i += 2;
        // DFA misc options
        let mut options = 0u16;
        if self.premultiplied {
            options |= 0b0000_0000_0000_0001;
        }
        A::write_u16(&mut buf[i..], options);
        i += 2;
        // start state
        A::write_u64(&mut buf[i..], self.start.to_usize() as u64);
        i += 8;
        // state count
        A::write_u64(&mut buf[i..], self.state_count as u64);
        i += 8;
        // max match state
        A::write_u64(
            &mut buf[i..],
            self.max_match.to_usize() as u64,
        );
        i += 8;
        // byte class map
        for b in (0..256).map(|b| b as u8) {
            buf[i] = self.byte_classes.get(b);
            i += 1;
        }
        // transition table
        for &id in self.trans() {
            if state_size == 1 {
                buf[i] = id.to_usize() as u8;
            } else if state_size == 2 {
                A::write_u16(&mut buf[i..], id.to_usize() as u16);
            } else if state_size == 4 {
                A::write_u32(&mut buf[i..], id.to_usize() as u32);
            } else {
                assert_eq!(8, state_size);
                A::write_u64(&mut buf[i..], id.to_usize() as u64);
            }
            i += state_size;
        }
        assert_eq!(size, i, "expected to consume entire buffer");

        Ok(buf)
    }
}

impl<'a, S: StateID> Repr<&'a [S], S> {
    /// The implementation for deserializing a DFA from raw bytes.
    pub unsafe fn from_bytes(mut buf: &'a [u8]) -> Repr<&'a [S], S> {
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
                 are you trying to load a DenseDFA serialized with a \
                 different endianness?",
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
                "state size of DenseDFA ({}) does not match \
                 requested state size ({})",
                state_size, mem::size_of::<S>(),
            );
        }
        buf = &buf[2..];

        // read miscellaneous options
        let kind = NativeEndian::read_u16(buf);
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

        // read byte classes
        let byte_classes = ByteClasses::from_slice(&buf[..256]);
        buf = &buf[256..];

        assert_eq!(
            0,
            buf.as_ptr() as usize % mem::align_of::<S>(),
            "DenseDFA transition table is not properly aligned"
        );
        let len = state_count * byte_classes.alphabet_len();
        assert!(
            buf.len() >= len,
            "insufficient transition table bytes, \
             expected at least {} but only have {}",
            len, buf.len()
        );

        // SAFETY: This is the only actual unsafe thing in this entire routine.
        // The key things we need to worry about here are alignment and size.
        // The two asserts above should cover both conditions.
        let trans = slice::from_raw_parts(buf.as_ptr() as *const S, len);
        Repr {
            premultiplied: kind & 0b0000_0000_0000_0001 > 0,
            start,
            state_count,
            max_match,
            byte_classes,
            trans,
        }
    }
}

/// The following methods implement mutable routines on the internal
/// representation of a DFA. As such, we must fix the first type parameter to
/// a `Vec<S>` since a generic `T: AsRef<[S]>` does not permit mutation. We
/// can get away with this because these methods are internal to the crate and
/// are exclusively used during construction of the DFA.
impl<S: StateID> Repr<Vec<S>, S> {
    pub fn premultiply(&mut self) -> Result<()> {
        if self.premultiplied || self.state_count <= 1 {
            return Ok(());
        }

        let alpha_len = self.alphabet_len();
        premultiply_overflow_error(
            S::from_usize(self.state_count - 1),
            alpha_len,
        )?;

        for id in (0..self.state_count).map(S::from_usize) {
            for (_, next) in self.get_state_mut(id).iter_mut() {
                *next = S::from_usize(next.to_usize() * alpha_len);
            }
        }
        self.premultiplied = true;
        self.start = S::from_usize(self.start.to_usize() * alpha_len);
        self.max_match = S::from_usize(self.max_match.to_usize() * alpha_len);
        Ok(())
    }

    /// Minimize this DFA using Hopcroft's algorithm.
    ///
    /// This cannot be called on a premultiplied DFA.
    pub fn minimize(&mut self) {
        assert!(!self.premultiplied, "can't minimize premultiplied DFA");

        Minimizer::new(self).run();
    }

    /// Set the start state of this DFA.
    ///
    /// Note that a start state cannot be set on a premultiplied DFA. Instead,
    /// DFAs should first be completely constructed and then premultiplied.
    pub fn set_start_state(&mut self, start: S) {
        assert!(!self.premultiplied, "can't set start on premultiplied DFA");
        assert!(start.to_usize() < self.state_count, "invalid start state");

        self.start = start;
    }

    /// Set the maximum state identifier that could possible correspond to a
    /// match state.
    ///
    /// Callers must uphold the invariant that any state identifier less than
    /// or equal to the identifier given is either a match state or the special
    /// dead state (which always has identifier 0 and whose transitions all
    /// lead back to itself).
    ///
    /// This cannot be called on a premultiplied DFA.
    pub fn set_max_match_state(&mut self, id: S) {
        assert!(!self.premultiplied, "can't set match on premultiplied DFA");
        assert!(id.to_usize() < self.state_count, "invalid max match state");

        self.max_match = id;
    }

    /// Add the given transition to this DFA. Both the `from` and `to` states
    /// must already exist.
    ///
    /// This cannot be called on a premultiplied DFA.
    pub fn add_transition(&mut self, from: S, byte: u8, to: S) {
        assert!(!self.premultiplied, "can't add trans to premultiplied DFA");
        assert!(from.to_usize() < self.state_count, "invalid from state");
        assert!(to.to_usize() < self.state_count, "invalid to state");

        let class = self.byte_classes.get(byte);
        let offset = from.to_usize() * self.alphabet_len() + class as usize;
        self.trans[offset] = to;
    }

    /// An an empty state (a state where all transitions lead to a dead state)
    /// and return its identifier. The identifier returned is guaranteed to
    /// not point to any other existing state.
    ///
    /// If adding a state would exhaust the state identifier space (given by
    /// `S`), then this returns an error. In practice, this means that the
    /// state identifier representation chosen is too small.
    ///
    /// This cannot be called on a premultiplied DFA.
    pub fn add_empty_state(&mut self) -> Result<S> {
        assert!(!self.premultiplied, "can't add state to premultiplied DFA");

        let id =
            if self.state_count == 0 {
                S::from_usize(0)
            } else {
                next_state_id(S::from_usize(self.state_count - 1))?
            };
        let alphabet_len = self.alphabet_len();
        self.trans.extend(iter::repeat(dead_id::<S>()).take(alphabet_len));
        // This should never panic, since state_count is a usize. The
        // transition table size would have run out of room long ago.
        self.state_count = self.state_count.checked_add(1).unwrap();
        Ok(id)
    }

    /// Return a mutable representation of the state corresponding to the given
    /// id. This is useful for implementing routines that manipulate DFA states
    /// (e.g., swapping states).
    ///
    /// This cannot be called on a premultiplied DFA.
    pub fn get_state_mut(&mut self, id: S) -> StateMut<S> {
        assert!(!self.premultiplied, "can't get state in premultiplied DFA");

        let alphabet_len = self.alphabet_len();
        let offset = id.to_usize() * alphabet_len;
        StateMut {
            transitions: &mut self.trans[offset..offset + alphabet_len],
        }
    }

    /// Swap the two states given in the transition table.
    ///
    /// This routine does not do anything to check the correctness of this
    /// swap. Callers must ensure that other states pointing to id1 and id2 are
    /// updated appropriately.
    ///
    /// This cannot be called on a premultiplied DFA.
    pub fn swap_states(&mut self, id1: S, id2: S) {
        assert!(!self.premultiplied, "can't swap states in premultiplied DFA");

        let o1 = id1.to_usize() * self.alphabet_len();
        let o2 = id2.to_usize() * self.alphabet_len();
        for b in 0..self.alphabet_len() {
            self.trans.swap(o1 + b, o2 + b);
        }
    }

    /// Truncate the states in this DFA to the given count.
    ///
    /// This routine does not do anything to check the correctness of this
    /// truncation. Callers must ensure that other states pointing to truncated
    /// states are updated appropriately.
    ///
    /// This cannot be called on a premultiplied DFA.
    pub fn truncate_states(&mut self, count: usize) {
        assert!(!self.premultiplied, "can't truncate in premultiplied DFA");

        let alphabet_len = self.alphabet_len();
        self.trans.truncate(count * alphabet_len);
        self.state_count = count;
    }

    /// This routine shuffles all match states in this DFA---according to the
    /// given map---to the beginning of the DFA such that every non-match state
    /// appears after every match state. (With one exception: the special dead
    /// state remains as the first state.) The given map should have length
    /// exactly equivalent to the number of states in this DFA.
    ///
    /// The purpose of doing this shuffling is to avoid the need to store
    /// additional state to determine whether a state is a match state or not.
    /// It also enables a single conditional in the core matching loop instead
    /// of two.
    ///
    /// This updates `self.max_match` to point to the last matching state as
    /// well as `self.start` if the starting state was moved.
    pub fn shuffle_match_states(&mut self, is_match: &[bool]) {
        assert!(
            !self.premultiplied,
            "cannot shuffle match states of premultiplied DFA"
        );
        assert_eq!(self.state_count, is_match.len());

        if self.state_count <= 2 {
            return;
        }

        let mut first_non_match = 1;
        while first_non_match < self.state_count && is_match[first_non_match] {
            first_non_match += 1;
        }

        let mut swaps: Vec<S> = vec![dead_id(); self.state_count];
        let mut cur = self.state_count - 1;
        while cur > first_non_match {
            if is_match[cur] {
                self.swap_states(
                    S::from_usize(cur),
                    S::from_usize(first_non_match),
                );
                swaps[cur] = S::from_usize(first_non_match);
                swaps[first_non_match] = S::from_usize(cur);

                first_non_match += 1;
                while first_non_match < cur && is_match[first_non_match] {
                    first_non_match += 1;
                }
            }
            cur -= 1;
        }
        for id in (0..self.state_count).map(S::from_usize) {
            for (_, next) in self.get_state_mut(id).iter_mut() {
                if swaps[next.to_usize()] != dead_id() {
                    *next = swaps[next.to_usize()];
                }
            }
        }
        if swaps[self.start.to_usize()] != dead_id() {
            self.start = swaps[self.start.to_usize()];
        }
        self.max_match = S::from_usize(first_non_match - 1);
    }
}

impl<T: AsRef<[S]>, S: StateID> fmt::Debug for Repr<T, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fn state_status<T: AsRef<[S]>, S: StateID>(
            dfa: &Repr<T, S>,
            id: S,
        ) -> String {
            let mut status = vec![b' ', b' '];
            if id == dead_id() {
                status[0] = b'D';
            } else if id == dfa.start {
                status[0] = b'>';
            }
            if dfa.is_match_state(id) {
                status[1] = b'*';
            }
            String::from_utf8(status).unwrap()
        }

        for (id, state) in self.states() {
            let status = state_status(self, id);
            writeln!(f, "{}{:04}: {:?}", status, id.to_usize(), state)?;
        }
        Ok(())
    }
}

/// An iterator over all states in a DFA.
///
/// This iterator yields a tuple for each state. The first element of the
/// tuple corresponds to a state's identifier, and the second element
/// corresponds to the state itself (comprised of its transitions).
///
/// If this DFA is premultiplied, then the state identifiers are in turn
/// premultiplied as well, making them usable without additional modification.
///
/// `'a` corresponding to the lifetime of original DFA, `T` corresponds to
/// the type of the transition table itself and `S` corresponds to the state
/// identifier representation.
pub(crate) struct StateIter<'a, T, S> {
    dfa: &'a Repr<T, S>,
    it: iter::Enumerate<slice::Chunks<'a, S>>,
}

impl<'a, T: AsRef<[S]>, S: StateID> Iterator for StateIter<'a, T, S> {
    type Item = (S, State<'a, S>);

    fn next(&mut self) -> Option<(S, State<'a, S>)> {
        self.it.next().map(|(id, chunk)| {
            let state = State { transitions: chunk };
            let id =
                if self.dfa.premultiplied {
                    id * self.dfa.alphabet_len()
                } else {
                    id
                };
            (S::from_usize(id), state)
        })
    }
}

/// An immutable representation of a single DFA state.
///
/// `'a` correspondings to the lifetime of a DFA's transition table and `S`
/// corresponds to the state identifier representation.
pub(crate) struct State<'a, S> {
    transitions: &'a [S],
}

impl<'a, S: StateID> State<'a, S> {
    /// Return an iterator over all transitions in this state. This yields
    /// a number of transitions equivalent to the alphabet length of the
    /// corresponding DFA.
    ///
    /// Each transition is represented by a tuple. The first element is
    /// the input byte for that transition and the second element is the
    /// transitions itself.
    pub fn transitions(&self) -> StateTransitionIter<S> {
        StateTransitionIter { it: self.transitions.iter().enumerate() }
    }

    /// Return an iterator over a sparse representation of the transitions in
    /// this state. Only non-dead transitions are returned.
    ///
    /// The "sparse" representation in this case corresponds to a sequence of
    /// triples. The first two elements of the triple comprise an inclusive
    /// byte range while the last element corresponds to the transition taken
    /// for all bytes in the range.
    ///
    /// This is somewhat more condensed than the classical sparse
    /// representation (where you have an element for every non-dead
    /// transition), but in practice, checking if a byte is in a range is very
    /// cheap and using ranges tends to conserve quite a bit more space.
    pub fn sparse_transitions(&self) -> Vec<(u8, u8, S)> {
        // TODO: Turn this into an iterator and skip over dead states.
        let mut ranges = vec![];
        let mut cur = None;
        for (i, &next_id) in self.transitions.iter().enumerate() {
            let b = i as u8;
            let (prev_start, prev_end, prev_next) = match cur {
                Some(range) => range,
                None => {
                    cur = Some((b, b, next_id));
                    continue;
                }
            };
            if prev_next == next_id {
                cur = Some((prev_start, b, prev_next));
            } else {
                ranges.push((prev_start, prev_end, prev_next));
                cur = Some((b, b, next_id));
            }
        }
        ranges.push(cur.unwrap());
        ranges
    }
}

/// An iterator over all transitions in a single DFA state. This yields
/// a number of transitions equivalent to the alphabet length of the
/// corresponding DFA.
///
/// Each transition is represented by a tuple. The first element is the input
/// byte for that transition and the second element is the transitions itself.
#[derive(Debug)]
pub(crate) struct StateTransitionIter<'a, S> {
    it: iter::Enumerate<slice::Iter<'a, S>>,
}

impl<'a, S: StateID> Iterator for StateTransitionIter<'a, S> {
    type Item = (u8, S);

    fn next(&mut self) -> Option<(u8, S)> {
        self.it.next().map(|(i, &id)| (i as u8, id))
    }
}

impl<'a, S: StateID> fmt::Debug for State<'a, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut transitions = vec![];
        for (start, end, next_id) in self.sparse_transitions() {
            if next_id == dead_id() {
                continue;
            }
            let line =
                if start == end {
                    format!("{} => {}", escape(start), next_id.to_usize())
                } else {
                    format!(
                        "{}-{} => {}",
                        escape(start), escape(end), next_id.to_usize(),
                    )
                };
            transitions.push(line);
        }
        write!(f, "{}", transitions.join(", "))?;
        Ok(())
    }
}

/// A mutable representation of a single DFA state.
///
/// `'a` correspondings to the lifetime of a DFA's transition table and `S`
/// corresponds to the state identifier representation.
pub(crate) struct StateMut<'a, S> {
    transitions: &'a mut [S],
}

impl<'a, S: StateID> StateMut<'a, S> {
    /// Return an iterator over all transitions in this state. This yields
    /// a number of transitions equivalent to the alphabet length of the
    /// corresponding DFA.
    ///
    /// Each transition is represented by a tuple. The first element is the
    /// input byte for that transition and the second element is a mutable
    /// reference to the transition itself.
    pub fn iter_mut(&mut self) -> StateTransitionIterMut<S> {
        StateTransitionIterMut { it: self.transitions.iter_mut().enumerate() }
    }
}

/// A mutable iterator over all transitions in a DFA state.
///
/// Each transition is represented by a tuple. The first element is the
/// input byte for that transition and the second element is a mutable
/// reference to the transition itself.
#[derive(Debug)]
pub(crate) struct StateTransitionIterMut<'a, S> {
    it: iter::Enumerate<slice::IterMut<'a, S>>,
}

impl<'a, S: StateID> Iterator for StateTransitionIterMut<'a, S> {
    type Item = (u8, &'a mut S);

    fn next(&mut self) -> Option<(u8, &'a mut S)> {
        self.it.next().map(|(i, id)| (i as u8, id))
    }
}

/// Return the given byte as its escaped string form.
fn escape(b: u8) -> String {
    use std::ascii;

    String::from_utf8(ascii::escape_default(b).collect::<Vec<_>>()).unwrap()
}

#[cfg(test)]
#[allow(dead_code)]
mod tests {
    use builder::DenseDFABuilder;
    use nfa::NFA;
    use super::*;

    #[test]
    fn errors_when_converting_to_smaller_dfa() {
        let pattern = r"\w";
        let dfa = DenseDFABuilder::new()
            .byte_classes(false)
            .anchored(true)
            .premultiply(false)
            .build_with_size::<u16>(pattern)
            .unwrap();
        assert!(dfa.to_u8().is_err());
    }

    #[test]
    fn errors_when_determinization_would_overflow() {
        let pattern = r"\w";

        let mut builder = DenseDFABuilder::new();
        builder.byte_classes(false).anchored(true).premultiply(false);
        // using u16 is fine
        assert!(builder.build_with_size::<u16>(pattern).is_ok());
        // // ... but u8 results in overflow (because there are >256 states)
        assert!(builder.build_with_size::<u8>(pattern).is_err());
    }

    #[test]
    fn errors_when_premultiply_would_overflow() {
        let pattern = r"[a-z]";

        let mut builder = DenseDFABuilder::new();
        builder.byte_classes(false).anchored(true).premultiply(false);
        // without premultiplication is OK
        assert!(builder.build_with_size::<u8>(pattern).is_ok());
        // ... but with premultiplication overflows u8
        builder.premultiply(true);
        assert!(builder.build_with_size::<u8>(pattern).is_err());
    }

    fn print_automata(pattern: &str) {
        println!("BUILDING AUTOMATA");
        let (nfa, dfa, mdfa) = build_automata(pattern);

        println!("{}", "#".repeat(100));
        println!("PATTERN: {:?}", pattern);
        println!("NFA:");
        println!("{:?}", nfa);

        println!("{}", "~".repeat(79));

        println!("DFA:");
        print!("{:?}", dfa);
        println!("{}", "~".repeat(79));

        println!("Minimal DFA:");
        print!("{:?}", mdfa);
        println!("{}", "~".repeat(79));

        println!("{}", "#".repeat(100));
    }

    // fn print_automata_counts(pattern: &str) {
        // let (nfa, dfa, mdfa) = build_automata(pattern);
        // println!("nfa # states: {:?}", nfa.len());
        // println!("dfa # states: {:?}", dfa.len());
        // println!("minimal dfa # states: {:?}", mdfa.len());
    // }

    fn build_automata(
        pattern: &str,
    ) -> (NFA, DenseDFA<Vec<usize>, usize>, DenseDFA<Vec<usize>, usize>) {
        let mut builder = DenseDFABuilder::new();
        builder.byte_classes(false).premultiply(false);
        builder.anchored(true);
        builder.allow_invalid_utf8(false);
        let nfa = builder.build_nfa(pattern).unwrap();
        let dfa = builder.build(pattern).unwrap();
        let min = builder.minimize(true).build(pattern).unwrap();

        (nfa, dfa, min)
    }

    #[test]
    fn scratch() {
        // let data = ::std::fs::read_to_string("/usr/share/dict/words").unwrap();
        // let mut words: Vec<&str> = data.lines().collect();
        // println!("{} words", words.len());
        // words.sort_by(|w1, w2| w1.len().cmp(&w2.len()).reverse());
        // let pattern = words.join("|");
        // print_automata_counts(&pattern);
        // print_automata(&pattern);

        // print_automata(r"[01]*1[01]{5}");
        // print_automata(r"X(.?){0,8}Y");
        // print_automata_counts(r"\p{alphabetic}");
        // print_automata(r"a*b+|cdefg");
        // print_automata(r"(..)*(...)*");

        // let pattern = r"\p{any}*?\p{Other_Uppercase}";
        // let pattern = r"\p{any}*?\w+";
        // print_automata_counts(pattern);
        // print_automata_counts(r"(?-u:\w)");

        // let pattern = r"\p{Greek}";
        let pattern = r"zZzZzZzZzZ";
        // let pattern = grapheme_pattern();
        // let pattern = r"\p{Ideographic}";
        // let pattern = r"\w";
        print_automata(pattern);
        let (_, _, dfa) = build_automata(pattern);
        let sparse = dfa.to_sparse_sized::<u16>().unwrap();
        println!("{:?}", sparse);

        // BREADCRUMBS: Look into sparse representation. Opporunities for
        // wins?
        //
        // Look at performance for computing next state. Naive find? Binary
        // search? memchr? Test on DFAs with smaller states. On DFAs with
        // larger states, naive search loses big time.
        //
        // When should we overhaul crate to use DFA trait? Maybe before we
        // dig into the above? Would be easier to manuever I guess.

        println!(
            "dense mem: {:?}, sparse mem: {:?}",
            dfa.to_u16().unwrap().memory_usage(),
            sparse.memory_usage(),
        );
    }

    fn grapheme_pattern() -> &'static str {
        r"(?x)
            (?:
                \p{gcb=CR}\p{gcb=LF}
                |
                [\p{gcb=Control}\p{gcb=CR}\p{gcb=LF}]
                |
                \p{gcb=Prepend}*
                (?:
                    (?:
                        (?:
                            \p{gcb=L}*
                            (?:\p{gcb=V}+|\p{gcb=LV}\p{gcb=V}*|\p{gcb=LVT})
                            \p{gcb=T}*
                        )
                        |
                        \p{gcb=L}+
                        |
                        \p{gcb=T}+
                    )
                    |
                    \p{gcb=RI}\p{gcb=RI}
                    |
                    \p{Extended_Pictographic}
                    (?:\p{gcb=Extend}*\p{gcb=ZWJ}\p{Extended_Pictographic})*
                    |
                    [^\p{gcb=Control}\p{gcb=CR}\p{gcb=LF}]
                )
                [\p{gcb=Extend}\p{gcb=ZWJ}\p{gcb=SpacingMark}]*
            )
    "
    }
}
