use core::convert::TryInto;
#[cfg(feature = "std")]
use core::fmt;
#[cfg(feature = "std")]
use core::iter;
use core::marker::PhantomData;
use core::mem::{align_of, size_of};
use core::slice;
#[cfg(feature = "std")]
use std::collections::{BTreeMap, BTreeSet};

use crate::bytes::{self, DeserializeError, Endian, SerializeError};
use crate::classes::{Byte, ByteClasses};
use crate::dfa::accel::Accels;
use crate::dfa::automaton::{
    fmt_state_indicator, Automaton, Start, MATCH_OFFSET,
};
use crate::dfa::dense;
#[cfg(feature = "std")]
use crate::dfa::error::Error;
use crate::dfa::special::Special;
use crate::matching::PatternID;
#[cfg(feature = "std")]
use crate::state_id::{dead_id, StateID};
use crate::util::DebugByte;
#[cfg(not(feature = "std"))]
use state_id::{dead_id, quit_id, StateID};

const LABEL: &str = "rust-regex-automata-dfa-sparse";
const VERSION: u64 = 2;

/// A sparse deterministic finite automaton (DFA) with variable sized states.
///
/// In contrast to a [dense DFA](../dense/struct.DFA.html), a sparse DFA uses
/// a more space efficient representation for its transitions. Consequently,
/// sparse DFAs may use much less memory than dense DFAs, but this comes at a
/// price. In particular, reading the more space efficient transitions takes
/// more work, and consequently, searching using a sparse DFA is typically
/// slower than a dense DFA.
///
/// A sparse DFA can be built using the default configuration via the
/// [`sparse::DFA::new`](struct.DFA.html#method.new) constructor.
/// Otherwise, one can configure various aspects of a dense DFA via
/// [`dense::Builder`](../dense/struct.Builder.html), and then convert a dense
/// DFA to a sparse DFA using
/// [`dense::DFA::to_sparse`](../dense/struct.DFA.html#method.to_sparse).
///
/// In general, a sparse DFA supports all the same operations as a dense DFA.
///
/// Making the choice between a dense and sparse DFA depends on your specific
/// work load. If you can sacrifice a bit of search time performance, then a
/// sparse DFA might be the best choice. In particular, while sparse DFAs are
/// probably always slower than dense DFAs, you may find that they are easily
/// fast enough for your purposes!
///
/// # Type parameters and state size
///
/// A `DFA` has two type parameters, `T` and `S`:
///
/// * `T` is the type of the DFA's transitions. `T` is typically `Vec<S>` or
///   `&[S]`.
/// * `S` is the representation used for the DFA's state identifiers as
///   described by the [`StateID`](../../trait.StateID.html) trait. `S` must
///   be one of `usize`, `u8`, `u16`, `u32` or `u64`. It defaults to
///   `usize`. The primary reason for choosing a different state identifier
///   representation than the default is to reduce the amount of memory used by
///   a DFA. Note though, that if the chosen representation cannot accommodate
///   the size of your DFA, then building the DFA will fail and return an
///   error.
///
/// While the reduction in heap memory used by a DFA is one reason for choosing
/// a smaller state identifier representation, another possible reason is for
/// decreasing the serialization size of a DFA, as returned by
/// [`to_bytes_little_endian`](struct.DFA.html#method.to_bytes_little_endian),
/// [`to_bytes_big_endian`](struct.DFA.html#method.to_bytes_big_endian)
/// or
/// [`to_bytes_native_endian`](struct.DFA.html#method.to_bytes_native_endian).
///
/// # The `Automaton` trait
///
/// This type implements the [`Automaton`](../trait.Automaton.html) trait,
/// which means it can be used for searching. For example:
///
/// ```
/// use regex_automata::dfa::{Automaton, HalfMatch, sparse::DFA};
///
/// let dfa = DFA::new("foo[0-9]+")?;
///
/// let expected = HalfMatch { pattern: 0, offset: 8 };
/// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo12345")?);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone)]
pub struct DFA<T, S = usize> {
    trans: Transitions<T, S>,
    starts: StartList<T, S>,
    special: Special<S>,
}

/// The transition table portion of a sparse DFA.
///
/// The transition table is the core part of the DFA in that it describes how
/// to move from one state to another based on the input sequence observed.
///
/// Unlike a typical dense table based DFA, states in a sparse transition
/// table have variable size. That is, states with more transitions use more
/// space than states with fewer transitions. This means that finding the next
/// transition takes more work than with a dense DFA, but also typically uses
/// much less space.
#[derive(Clone)]
struct Transitions<T, S> {
    /// The raw encoding of each state in this DFA.
    ///
    /// Each state has the following information:
    ///
    /// * A set of transitions to subsequent states. Transitions to the dead
    ///   state are omitted.
    /// * If the state can be accelerated, then any additional accelerator
    ///   information.
    /// * If the state is a match state, then the state contains all pattern
    ///   IDs that match when in that state.
    ///
    /// To decode a state, use Transitions::state.
    ///
    /// In practice, T is either Vec<u8> or &[u8].
    sparse: T,
    /// A set of equivalence classes, where a single equivalence class
    /// represents a set of bytes that never discriminate between a match
    /// and a non-match in the DFA. Each equivalence class corresponds to a
    /// single character in this DFA's alphabet, where the maximum number of
    /// characters is 257 (each possible value of a byte plus the special
    /// EOF transition). Consequently, the number of equivalence classes
    /// corresponds to the number of transitions for each DFA state. Note
    /// though that the *space* used by each DFA state in the transition table
    /// may be larger. The total space used by each DFA state is known as the
    /// stride and is documented above.
    ///
    /// The only time the number of equivalence classes is fewer than 257 is
    /// if the DFA's kind uses byte classes which is the default. Equivalence
    /// classes should generally only be disabled when debugging, so that
    /// the transitions themselves aren't obscured. Disabling them has no
    /// other benefit, since the equivalence class map is always used while
    /// searching. In the vast majority of cases, the number of equivalence
    /// classes is substantially smaller than 257, particularly when large
    /// Unicode classes aren't used.
    ///
    /// N.B. Equivalence classes aren't particularly useful in a sparse DFA
    /// in the current implementation, since equivalence classes generally tend
    /// to correspond to continuous ranges of bytes that map to the same
    /// transition. So in a sparse DFA, equivalence classes don't really lead
    /// to a space savings. In the future, it would be good to try and remove
    /// them from sparse DFAs entirely, but requires a bit of work since sparse
    /// DFAs are built from dense DFAs, which are in turn built on top of
    /// equivalence classes.
    classes: ByteClasses,
    /// The total number of states in this DFA. Note that a DFA always has at
    /// least one state---the dead state---even the empty DFA. In particular,
    /// the dead state always has ID 0 and is correspondingly always the first
    /// state. The dead state is never a match state.
    count: usize,
    /// The total number of unique patterns represented by these match states.
    patterns: usize,
    /// The state ID representation.
    _state_id: PhantomData<S>,
}

/// The set of all possible starting states in a DFA.
///
/// See the eponymous type in the `dense` module for more details. This type
/// is very similar to `dense::StartList`, except that its underlying
/// representation is `&[u8]` instead of `&[S]`. (The latter would require
/// sparse DFAs to be aligned, which is explicitly something we do not require
/// because we don't really need it.)
#[derive(Clone)]
struct StartList<T, S> {
    /// The initial start state IDs as a contiguous list of native endian
    /// encoded integers, represented by `S`.
    ///
    /// In practice, T is either Vec<u8> or &[u8].
    list: T,
    /// The state ID representation. This is what's actually stored in `list`.
    _state_id: PhantomData<S>,
}

#[cfg(feature = "std")]
impl DFA<Vec<u8>, usize> {
    /// Parse the given regular expression using a default configuration and
    /// return the corresponding sparse DFA.
    ///
    /// The default configuration uses `usize` for state IDs and reduces the
    /// alphabet size by splitting bytes into equivalence classes. The
    /// resulting DFA is *not* minimized.
    ///
    /// If you want a non-default configuration, then use the
    /// [`dense::Builder`](dense/struct.Builder.html)
    /// to set your own configuration, and then call
    /// [`dense::DFA::to_sparse`](struct.DFA.html#method.to_sparse)
    /// to create a sparse DFA.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, sparse};
    ///
    /// let dfa = sparse::DFA::new("foo[0-9]+bar")?;
    ///
    /// let expected = HalfMatch { pattern: 0, offset: 11 };
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo12345bar")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(pattern: &str) -> Result<DFA<Vec<u8>, usize>, Error> {
        dense::Builder::new()
            .build(pattern)
            .and_then(|dense| dense.to_sparse())
    }
}

#[cfg(feature = "std")]
impl<S: StateID> DFA<Vec<u8>, S> {
    /// Create a new DFA that matches every input.
    ///
    /// # Example
    ///
    /// In order to build a DFA that always matches, callers must provide a
    /// type hint indicating their choice of state identifier representation.
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, sparse};
    ///
    /// let dfa: sparse::DFA<Vec<u8>, usize> = sparse::DFA::always_match()?;
    ///
    /// let expected = HalfMatch { pattern: 0, offset: 0 };
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"")?);
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn always_match() -> Result<DFA<Vec<u8>, S>, Error> {
        dense::DFA::always_match()?.to_sparse()
    }

    /// Create a new sparse DFA that never matches any input.
    ///
    /// # Example
    ///
    /// In order to build a DFA that never matches, callers must provide a type
    /// hint indicating their choice of state identifier representation.
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, sparse};
    ///
    /// let dfa: sparse::DFA<Vec<u8>, usize> = sparse::DFA::never_match()?;
    /// assert_eq!(None, dfa.find_leftmost_fwd(b"")?);
    /// assert_eq!(None, dfa.find_leftmost_fwd(b"foo")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn never_match() -> Result<DFA<Vec<u8>, S>, Error> {
        dense::DFA::never_match()?.to_sparse()
    }

    /// The implementation for constructing a sparse DFA from a dense DFA.
    pub(crate) fn from_dense_sized<T, A, S2>(
        dfa: &dense::DFA<T, A, S>,
    ) -> Result<DFA<Vec<u8>, S2>, Error>
    where
        T: AsRef<[S]>,
        A: AsRef<[u8]>,
        S2: StateID,
    {
        // In order to build the transition table, we need to be able to write
        // state identifiers for each of the "next" transitions in each state.
        // Our state identifiers correspond to the byte offset in the
        // transition table at which the state is encoded. Therefore, we do not
        // actually know what the state identifiers are until we've allocated
        // exactly as much space as we need for each state. Thus, construction
        // of the transition table happens in two passes.
        //
        // In the first pass, we fill out the shell of each state, which
        // includes the transition count, the input byte ranges and zero-filled
        // space for the transitions and accelerators, if present. In this
        // first pass, we also build up a map from the state identifier index
        // of the dense DFA to the state identifier in this sparse DFA.
        //
        // In the second pass, we fill in the transitions based on the map
        // built in the first pass.

        let mut sparse =
            Vec::with_capacity(size_of::<S2>() * dfa.state_count());
        let mut remap: Vec<S2> = vec![dead_id(); dfa.state_count()];
        for state in dfa.states() {
            let pos = sparse.len();

            remap[dfa.to_index(state.id())] = usize_to_state_id(pos)?;
            // zero-filled space for the transition count
            sparse.push(0);
            sparse.push(0);

            let mut transition_count = 0;
            for (b1, b2, _) in state.sparse_transitions() {
                match (b1, b2) {
                    (Byte::U8(b1), Byte::U8(b2)) => {
                        transition_count += 1;
                        sparse.push(b1);
                        sparse.push(b2);
                    }
                    (Byte::EOF(_), Byte::EOF(_)) => {}
                    (Byte::U8(_), Byte::EOF(_))
                    | (Byte::EOF(_), Byte::U8(_)) => {
                        // can never occur because sparse_transitions never
                        // groups EOF with any other transition.
                        unreachable!()
                    }
                }
            }
            // Add dummy EOF transition. This is never actually read while
            // searching, but having space equivalent to the total number
            // of transitions is convenient. Otherwise, we'd need to track
            // a different number of transitions for the byte ranges as for
            // the 'next' states.
            transition_count += 1;
            sparse.push(0);
            sparse.push(0);

            // Fill in the transition count.
            // Since transition count is always <= 257, we use the most
            // significant bit to indicate whether this is a match state or
            // not.
            let ntrans = if dfa.is_match_state(state.id()) {
                transition_count | (1 << 15)
            } else {
                transition_count
            };
            bytes::NE::write_u16(ntrans, &mut sparse[pos..]);

            // zero-fill the actual transitions
            let zeros = transition_count as usize * size_of::<S2>();
            sparse.extend(iter::repeat(0).take(zeros));

            // If this is a match state, write the pattern IDs matched by this
            // state.
            if dfa.is_match_state(state.id()) {
                let plen = dfa.match_pattern_len(state.id());
                // Write the actual pattern IDs with a u32 length prefix.
                let mut pos = sparse.len();
                sparse.extend(iter::repeat(0).take(4 + 4 * plen));
                bytes::NE::write_u32(
                    // Will never fail since u32::MAX is invalid pattern ID.
                    // Thus, the number of pattern IDs representable by a u32.
                    plen.try_into().unwrap(),
                    &mut sparse[pos..],
                );
                pos += 4;
                for (i, pid) in dfa.match_pattern_ids(state.id()).enumerate() {
                    bytes::NE::write_u32(pid, &mut sparse[pos..]);
                    pos += 4;
                }
            }

            // And now add the accelerator, if one exists. An accelerator is
            // at most 4 bytes and at least 1 byte. The first byte is the
            // length, N. N bytes follow the length. The set of bytes that
            // follow correspond (exhaustively) to the bytes that must be seen
            // to leave this state.
            let accel = dfa.accelerator(state.id());
            sparse.push(accel.len().try_into().unwrap());
            sparse.extend_from_slice(accel);
        }

        let mut new = DFA {
            trans: Transitions {
                sparse,
                classes: dfa.byte_classes().clone(),
                count: dfa.state_count(),
                patterns: dfa.pattern_count(),
                _state_id: PhantomData,
            },
            starts: StartList::from_dense_dfa(dfa, &remap)?,
            special: dfa.special().remap(|id| remap[dfa.to_index(id)]),
        };
        for old_state in dfa.states() {
            let new_id = remap[dfa.to_index(old_state.id())];
            let mut new_state = new.trans.state_mut(new_id);
            let sparse = old_state.sparse_transitions();
            for (i, (_, _, next)) in sparse.enumerate() {
                let next = remap[dfa.to_index(next)];
                new_state.set_next_at(i, next);
            }
        }
        Ok(new)
    }
}

impl<T: AsRef<[u8]>, S: StateID> DFA<T, S> {
    /// Cheaply return a borrowed version of this sparse DFA. Specifically, the
    /// DFA returned always uses `&[u8]` for its transitions while keeping
    /// the same state identifier representation.
    pub fn as_ref<'a>(&'a self) -> DFA<&'a [u8], S> {
        DFA {
            trans: self.trans.as_ref(),
            starts: self.starts.as_ref(),
            special: self.special,
        }
    }

    /// Return an owned version of this sparse DFA. Specifically, the DFA
    /// returned always uses `Vec<u8>` for its transitions while keeping the
    /// same state identifier representation.
    ///
    /// Effectively, this returns a sparse DFA whose transitions live on the
    /// heap.
    #[cfg(feature = "std")]
    pub fn to_owned(&self) -> DFA<Vec<u8>, S> {
        DFA {
            trans: self.trans.to_owned(),
            starts: self.starts.to_owned(),
            special: self.special,
        }
    }

    /// Returns the memory usage, in bytes, of this DFA.
    ///
    /// The memory usage is computed based on the number of bytes used to
    /// represent this DFA's transition table. This typically corresponds to
    /// heap memory usage.
    ///
    /// This does **not** include the stack size used up by this DFA. To
    /// compute that, used `std::mem::size_of::<sparse::DFA>()`.
    pub fn memory_usage(&self) -> usize {
        self.trans.memory_usage() + self.starts.memory_usage()
    }
}

/// Routines for converting a sparse DFA to other representations, such as
/// smaller state identifiers or raw bytes suitable for persistent storage.
#[cfg(feature = "std")]
impl<T: AsRef<[u8]>, S: StateID> DFA<T, S> {
    /// Create a new DFA whose match semantics are equivalent to this DFA, but
    /// attempt to use `S2` for the representation of state identifiers. If
    /// `S2` is insufficient to represent all state identifiers in this DFA,
    /// then this returns an error.
    ///
    /// An alternative way to construct such a DFA is to use
    /// [`dense::DFA::to_sparse_sized`](../dense/struct.DFA.html#method.to_sparse_sized).
    /// In general, picking the appropriate size upon initial construction of
    /// a sparse DFA is preferred, since it will do the conversion in one
    /// step instead of two.
    ///
    /// # Example
    ///
    /// This example shows how to create a sparse DFA with `u16` as the state
    /// identifier representation.
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, sparse::DFA};
    ///
    /// let dfa = DFA::new("foo[0-9]+")?.to_sized::<u16>()?;
    ///
    /// let expected = HalfMatch { pattern: 0, offset: 8 };
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo12345")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_sized<S2: StateID>(&self) -> Result<DFA<Vec<u8>, S2>, Error> {
        // To build the new DFA, we proceed much like the initial construction
        // of the sparse DFA. Namely, since the state ID size is changing,
        // we don't actually know all of our state IDs until we've allocated
        // all necessary space. So we do one pass that allocates all of the
        // storage we need, and then another pass to fill in the transitions.
        let (trans, remap) = self.trans.to_sized()?;
        Ok(DFA {
            trans,
            starts: self.starts.to_sized(&remap),
            special: self.special.remap(|id| remap[&id]),
        })
    }

    /// Serialize this DFA as raw bytes to a `Vec<u8>` in little endian
    /// format.
    ///
    /// The written bytes are guaranteed to be deserialized correctly and
    /// without errors in a semver compatible release of this crate by a
    /// `DFA`'s deserialization APIs (assuming all other criteria for the
    /// deserialization APIs has been satisfied):
    ///
    /// * [`from_bytes`](struct.DFA.html#method.from_bytes)
    /// * [`from_bytes_unchecked`](struct.DFA.html#method.from_bytes_unchecked)
    ///
    /// Note that unlike a
    /// [`dense::DFA`'s](../dense/struct.DFA.html)
    /// serialization methods, this does not add any initial padding to the
    /// returned bytes. Padding isn't required for sparse DFAs since they have
    /// no alignment requirements.
    ///
    /// # Example
    ///
    /// This example shows how to serialize and deserialize a DFA:
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, sparse::DFA};
    ///
    /// // Compile our original DFA. We use 16-bit state identifiers to give
    /// // our state IDs a small fixed size.
    /// let original_dfa = DFA::new("foo[0-9]+")?.to_sized::<u16>()?;
    ///
    /// // N.B. We use native endianness here to make the example work, but
    /// // using to_bytes_little_endian would work on a little endian target.
    /// let buf = original_dfa.to_bytes_native_endian();
    /// // Even if buf has initial padding, DFA::from_bytes will automatically
    /// // ignore it.
    /// let dfa: DFA<&[u8], u16> = DFA::from_bytes(&buf)?.0;
    ///
    /// let expected = HalfMatch { pattern: 0, offset: 8 };
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo12345")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_bytes_little_endian(&self) -> Vec<u8> {
        self.to_bytes::<bytes::LE>()
    }

    /// Serialize this DFA as raw bytes to a `Vec<u8>` in big endian
    /// format.
    ///
    /// The written bytes are guaranteed to be deserialized correctly and
    /// without errors in a semver compatible release of this crate by a
    /// `DFA`'s deserialization APIs (assuming all other criteria for the
    /// deserialization APIs has been satisfied):
    ///
    /// * [`from_bytes`](struct.DFA.html#method.from_bytes)
    /// * [`from_bytes_unchecked`](struct.DFA.html#method.from_bytes_unchecked)
    ///
    /// Note that unlike a
    /// [`dense::DFA`'s](../dense/struct.DFA.html)
    /// serialization methods, this does not add any initial padding to the
    /// returned bytes. Padding isn't required for sparse DFAs since they have
    /// no alignment requirements.
    ///
    /// # Example
    ///
    /// This example shows how to serialize and deserialize a DFA:
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, sparse::DFA};
    ///
    /// // Compile our original DFA. We use 16-bit state identifiers to give
    /// // our state IDs a small fixed size.
    /// let original_dfa = DFA::new("foo[0-9]+")?.to_sized::<u16>()?;
    ///
    /// // N.B. We use native endianness here to make the example work, but
    /// // using to_bytes_big_endian would work on a big endian target.
    /// let buf = original_dfa.to_bytes_native_endian();
    /// // Even if buf has initial padding, DFA::from_bytes will automatically
    /// // ignore it.
    /// let dfa: DFA<&[u8], u16> = DFA::from_bytes(&buf)?.0;
    ///
    /// let expected = HalfMatch { pattern: 0, offset: 8 };
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo12345")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_bytes_big_endian(&self) -> Vec<u8> {
        self.to_bytes::<bytes::BE>()
    }

    /// Serialize this DFA as raw bytes to a `Vec<u8>` in native endian
    /// format.
    ///
    /// The written bytes are guaranteed to be deserialized correctly and
    /// without errors in a semver compatible release of this crate by a
    /// `DFA`'s deserialization APIs (assuming all other criteria for the
    /// deserialization APIs has been satisfied):
    ///
    /// * [`from_bytes`](struct.DFA.html#method.from_bytes)
    /// * [`from_bytes_unchecked`](struct.DFA.html#method.from_bytes_unchecked)
    ///
    /// Note that unlike a
    /// [`dense::DFA`'s](../dense/struct.DFA.html)
    /// serialization methods, this does not add any initial padding to the
    /// returned bytes. Padding isn't required for sparse DFAs since they have
    /// no alignment requirements.
    ///
    /// Generally speaking, native endian format should only be used when
    /// you know that the target you're compiling the DFA for matches the
    /// endianness of the target on which you're compiling DFA. For example,
    /// if serialization and deserialization happen in the same process or on
    /// the same machine. Otherwise, when serializing a DFA for use in a
    /// portable environment, you'll almost certainly want to serialize _both_
    /// a little endian and a big endian version and then load the correct one
    /// based on the target's configuration.
    ///
    /// # Example
    ///
    /// This example shows how to serialize and deserialize a DFA:
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, sparse::DFA};
    ///
    /// // Compile our original DFA. We use 16-bit state identifiers to give
    /// // our state IDs a small fixed size.
    /// let original_dfa = DFA::new("foo[0-9]+")?.to_sized::<u16>()?;
    ///
    /// let buf = original_dfa.to_bytes_native_endian();
    /// // Even if buf has initial padding, DFA::from_bytes will automatically
    /// // ignore it.
    /// let dfa: DFA<&[u8], u16> = DFA::from_bytes(&buf)?.0;
    ///
    /// let expected = HalfMatch { pattern: 0, offset: 8 };
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo12345")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_bytes_native_endian(&self) -> Vec<u8> {
        self.to_bytes::<bytes::NE>()
    }

    /// The implementation of the public `to_bytes` serialization methods,
    /// which is generic over endianness.
    fn to_bytes<E: Endian>(&self) -> Vec<u8> {
        let mut buf = vec![0; self.write_to_len()];
        // This should always succeed since the only possible serialization
        // error is providing a buffer that's too small, but we've ensured that
        // `buf` is big enough here.
        self.write_to::<E>(&mut buf).unwrap();
        buf
    }

    /// Serialize this DFA as raw bytes to the given slice, in little endian
    /// format. Upon success, the total number of bytes written to `dst` is
    /// returned.
    ///
    /// The written bytes are guaranteed to be deserialized correctly and
    /// without errors in a semver compatible release of this crate by a
    /// `DFA`'s deserialization APIs (assuming all other criteria for the
    /// deserialization APIs has been satisfied):
    ///
    /// * [`from_bytes`](struct.DFA.html#method.from_bytes)
    /// * [`from_bytes_unchecked`](struct.DFA.html#method.from_bytes_unchecked)
    ///
    /// # Errors
    ///
    /// This returns an error if the given destination slice is not big enough
    /// to contain the full serialized DFA. If an error occurs, then nothing
    /// is written to `dst`.
    ///
    /// # Example
    ///
    /// This example shows how to serialize and deserialize a DFA without
    /// dynamic memory allocation.
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, sparse::DFA};
    ///
    /// // Compile our original DFA. We use 16-bit state identifiers to give
    /// // our state IDs a small fixed size.
    /// let original_dfa = DFA::new("foo[0-9]+")?.to_sized::<u16>()?;
    ///
    /// // Create a 4KB buffer on the stack to store our serialized DFA.
    /// let mut buf = [0u8; 4 * (1<<10)];
    /// // N.B. We use native endianness here to make the example work, but
    /// // using write_to_little_endian would work on a little endian target.
    /// let written = original_dfa.write_to_native_endian(&mut buf)?;
    /// let dfa: DFA<&[u8], u16> = DFA::from_bytes(&buf[..written])?.0;
    ///
    /// let expected = HalfMatch { pattern: 0, offset: 8 };
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo12345")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn write_to_little_endian(
        &self,
        dst: &mut [u8],
    ) -> Result<usize, SerializeError> {
        self.write_to::<bytes::LE>(dst)
    }

    /// Serialize this DFA as raw bytes to the given slice, in big endian
    /// format. Upon success, the total number of bytes written to `dst` is
    /// returned.
    ///
    /// The written bytes are guaranteed to be deserialized correctly and
    /// without errors in a semver compatible release of this crate by a
    /// `DFA`'s deserialization APIs (assuming all other criteria for the
    /// deserialization APIs has been satisfied):
    ///
    /// * [`from_bytes`](struct.DFA.html#method.from_bytes)
    /// * [`from_bytes_unchecked`](struct.DFA.html#method.from_bytes_unchecked)
    ///
    /// # Errors
    ///
    /// This returns an error if the given destination slice is not big enough
    /// to contain the full serialized DFA. If an error occurs, then nothing
    /// is written to `dst`.
    ///
    /// # Example
    ///
    /// This example shows how to serialize and deserialize a DFA without
    /// dynamic memory allocation.
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, sparse::DFA};
    ///
    /// // Compile our original DFA. We use 16-bit state identifiers to give
    /// // our state IDs a small fixed size.
    /// let original_dfa = DFA::new("foo[0-9]+")?.to_sized::<u16>()?;
    ///
    /// // Create a 4KB buffer on the stack to store our serialized DFA.
    /// let mut buf = [0u8; 4 * (1<<10)];
    /// // N.B. We use native endianness here to make the example work, but
    /// // using write_to_big_endian would work on a big endian target.
    /// let written = original_dfa.write_to_native_endian(&mut buf)?;
    /// let dfa: DFA<&[u8], u16> = DFA::from_bytes(&buf[..written])?.0;
    ///
    /// let expected = HalfMatch { pattern: 0, offset: 8 };
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo12345")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn write_to_big_endian(
        &self,
        dst: &mut [u8],
    ) -> Result<usize, SerializeError> {
        self.write_to::<bytes::BE>(dst)
    }

    /// Serialize this DFA as raw bytes to the given slice, in native endian
    /// format. Upon success, the total number of bytes written to `dst` is
    /// returned.
    ///
    /// The written bytes are guaranteed to be deserialized correctly and
    /// without errors in a semver compatible release of this crate by a
    /// `DFA`'s deserialization APIs (assuming all other criteria for the
    /// deserialization APIs has been satisfied):
    ///
    /// * [`from_bytes`](struct.DFA.html#method.from_bytes)
    /// * [`from_bytes_unchecked`](struct.DFA.html#method.from_bytes_unchecked)
    ///
    /// Generally speaking, native endian format should only be used when
    /// you know that the target you're compiling the DFA for matches the
    /// endianness of the target on which you're compiling DFA. For example,
    /// if serialization and deserialization happen in the same process or on
    /// the same machine. Otherwise, when serializing a DFA for use in a
    /// portable environment, you'll almost certainly want to serialize _both_
    /// a little endian and a big endian version and then load the correct one
    /// based on the target's configuration.
    ///
    /// # Errors
    ///
    /// This returns an error if the given destination slice is not big enough
    /// to contain the full serialized DFA. If an error occurs, then nothing
    /// is written to `dst`.
    ///
    /// # Example
    ///
    /// This example shows how to serialize and deserialize a DFA without
    /// dynamic memory allocation.
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, sparse::DFA};
    ///
    /// // Compile our original DFA. We use 16-bit state identifiers to give
    /// // our state IDs a small fixed size.
    /// let original_dfa = DFA::new("foo[0-9]+")?.to_sized::<u16>()?;
    ///
    /// // Create a 4KB buffer on the stack to store our serialized DFA.
    /// let mut buf = [0u8; 4 * (1<<10)];
    /// let written = original_dfa.write_to_native_endian(&mut buf)?;
    /// let dfa: DFA<&[u8], u16> = DFA::from_bytes(&buf[..written])?.0;
    ///
    /// let expected = HalfMatch { pattern: 0, offset: 8 };
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo12345")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn write_to_native_endian(
        &self,
        dst: &mut [u8],
    ) -> Result<usize, SerializeError> {
        self.write_to::<bytes::NE>(dst)
    }

    /// The implementation of the public `write_to` serialization methods,
    /// which is generic over endianness.
    fn write_to<E: Endian>(
        &self,
        dst: &mut [u8],
    ) -> Result<usize, SerializeError> {
        let mut nw = 0;
        nw += bytes::write_label(LABEL, &mut dst[nw..])?;
        nw += bytes::write_endianness_check::<E>(&mut dst[nw..])?;
        nw += bytes::write_version::<E>(VERSION, &mut dst[nw..])?;
        nw += bytes::write_state_size::<E, S>(&mut dst[nw..])?;
        nw += {
            // Currently unused, intended for future flexibility
            E::write_u64(0, &mut dst[nw..]);
            8
        };
        nw += self.trans.write_to::<E>(&mut dst[nw..])?;
        nw += self.starts.write_to::<E>(&mut dst[nw..])?;
        nw += self.special.write_to::<E>(&mut dst[nw..])?;
        Ok(nw)
    }

    /// Return the total number of bytes required to serialize this DFA.
    ///
    /// This is useful for determining the size of the buffer required to pass
    /// to one of the serialization routines:
    ///
    /// * [`write_to_little_endian`](struct.DFA.html#method.write_to_little_endian)
    /// * [`write_to_big_endian`](struct.DFA.html#method.write_to_big_endian)
    /// * [`write_to_native_endian`](struct.DFA.html#method.write_to_native_endian)
    ///
    /// Passing a buffer smaller than the size returned by this method will
    /// result in a serialization error.
    ///
    /// # Example
    ///
    /// This example shows how to dynamically allocate enough room to serialize
    /// a sparse DFA.
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, sparse::DFA};
    ///
    /// // Compile our original DFA. We use 16-bit state identifiers to give
    /// // our state IDs a small fixed size.
    /// let original_dfa = DFA::new("foo[0-9]+")?.to_sized::<u16>()?;
    ///
    /// let mut buf = vec![0; original_dfa.write_to_len()];
    /// let written = original_dfa.write_to_native_endian(&mut buf)?;
    /// let dfa: DFA<&[u8], u16> = DFA::from_bytes(&buf[..written])?.0;
    ///
    /// let expected = HalfMatch { pattern: 0, offset: 8 };
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo12345")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn write_to_len(&self) -> usize {
        bytes::write_label_len(LABEL)
        + bytes::write_endianness_check_len()
        + bytes::write_version_len()
        + bytes::write_state_size_len()
        + 8 // unused, intended for future flexibility
        + self.trans.write_to_len()
        + self.starts.write_to_len()
        + self.special.write_to_len()
    }
}

impl<'a, S: StateID> DFA<&'a [u8], S> {
    /// Safely deserialize a sparse DFA with a specific state identifier
    /// representation. Upon success, this returns both the deserialized DFA
    /// and the number of bytes read from the given slice. Namely, the contents
    /// of the slice beyond the DFA are not read.
    ///
    /// Deserializing a DFA using this routine will never allocate heap memory.
    /// For safety purposes, the DFA's transitions will be verified such that
    /// every transition points to a valid state. If this verification is too
    /// costly, then a
    /// [`from_bytes_unchecked`](struct.DFA.html#method.from_bytes_unchecked)
    /// API is provided, which will always execute in constant time.
    ///
    /// The bytes given must be generated by one of the serialization APIs
    /// of a `DFA` using a semver compatible release of this crate. Those
    /// include:
    ///
    /// * [`to_bytes_little_endian`](struct.DFA.html#method.to_bytes_little_endian)
    /// * [`to_bytes_big_endian`](struct.DFA.html#method.to_bytes_big_endian)
    /// * [`to_bytes_native_endian`](struct.DFA.html#method.to_bytes_native_endian)
    /// * [`write_to_little_endian`](struct.DFA.html#method.write_to_little_endian)
    /// * [`write_to_big_endian`](struct.DFA.html#method.write_to_big_endian)
    /// * [`write_to_native_endian`](struct.DFA.html#method.write_to_native_endian)
    ///
    /// The `to_bytes` methods allocate and return a `Vec<u8>` for you. The
    /// `write_to` methods do not allocate and write to an existing slice
    /// (which may be on the stack). Since deserialization always uses the
    /// native endianness of the target platform, the serialization API you use
    /// should match the endianness of the target platform. (It's often a good
    /// idea to generate serialized DFAs for both forms of endianness and then
    /// load the correct one based on endianness.)
    ///
    /// If the state identifier representation is `usize`, then deserialization
    /// is dependent on the pointer size. For this reason, it is best to
    /// serialize DFAs using a fixed size representation for your state
    /// identifiers, such as `u8`, `u16`, `u32` or `u64`.
    ///
    /// # Errors
    ///
    /// Generally speaking, it's easier to state the conditions in which an
    /// error is _not_ returned. All of the following must be true:
    ///
    /// * The bytes given must be produced by one of the serialization APIs
    ///   on this DFA, as mentioned above.
    /// * The state ID representation chosen by type inference (that's the `S`
    ///   type parameter) must match the state ID representation in the given
    ///   serialized DFA.
    /// * The endianness of the target platform matches the endianness used to
    ///   serialized the provided DFA.
    ///
    /// If any of the above are not true, then an error will be returned.
    ///
    /// Note that unlike deserializing a
    /// [`dense::DFA`](../dense/struct.DFA.html),
    /// deserializing a sparse DFA has no alignment requirements. That is, an
    /// alignment of `1` is valid.
    ///
    /// # Panics
    ///
    /// This routine will never panic for any input.
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
    /// use regex_automata::dfa::{Automaton, HalfMatch, sparse::DFA};
    ///
    /// let initial = DFA::new("foo[0-9]+")?;
    /// let bytes = initial.to_sized::<u16>()?.to_bytes_native_endian();
    /// let dfa: DFA<&[u8], u16> = DFA::from_bytes(&bytes)?.0;
    ///
    /// let expected = HalfMatch { pattern: 0, offset: 8 };
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo12345")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: loading a DFA from static memory
    ///
    /// One use case this library supports is the ability to serialize a
    /// DFA to disk and then use `include_bytes!` to store it in a compiled
    /// Rust program. Those bytes can then be cheaply deserialized into a
    /// `DFA` structure at runtime and used for searching without having to
    /// re-compile the DFA (which can be quite costly).
    ///
    /// We can show this in two parts. The first part is serializing the DFA to
    /// a file:
    ///
    /// ```no_run
    /// use regex_automata::dfa::{Automaton, sparse::DFA};
    ///
    /// let dfa = DFA::new("foo[0-9]+")?;
    ///
    /// // Write a big endian serialized version of this DFA to a file.
    /// let bytes = dfa.to_sized::<u16>()?.to_bytes_big_endian();
    /// std::fs::write("foo.bigendian.dfa", &bytes)?;
    ///
    /// // Do it again, but this time for little endian.
    /// let bytes = dfa.to_sized::<u16>()?.to_bytes_little_endian();
    /// std::fs::write("foo.littleendian.dfa", &bytes)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// And now the second part is embedding the DFA into the compiled program
    /// and deserializing it at runtime on first use. We use conditional
    /// compilation to choose the correct endianness. As mentioned above, we
    /// do not need to employ any special tricks to ensure a proper alignment,
    /// since a sparse DFA has no alignment requirements.
    ///
    /// ```no_run
    /// use regex_automata::dfa::{Automaton, HalfMatch, sparse};
    ///
    /// type S = u16;
    /// type DFA = sparse::DFA<&'static [u8], S>;
    ///
    /// fn get_foo() -> &'static DFA {
    ///     use std::cell::Cell;
    ///     use std::mem::MaybeUninit;
    ///     use std::sync::Once;
    ///
    ///     # const _: &str = stringify! {
    ///     #[cfg(target_endian = "big")]
    ///     static BYTES: &[u8] = include_bytes!("foo.bigendian.dfa");
    ///     #[cfg(target_endian = "little")]
    ///     static BYTES: &[u8] = include_bytes!("foo.littleendian.dfa");
    ///     # };
    ///     # static BYTES: &[u8] = b"";
    ///
    ///     struct Lazy(Cell<MaybeUninit<DFA>>);
    ///     // SAFETY: This is safe because DFA impls Sync.
    ///     unsafe impl Sync for Lazy {}
    ///
    ///     static INIT: Once = Once::new();
    ///     static DFA: Lazy = Lazy(Cell::new(MaybeUninit::uninit()));
    ///
    ///     INIT.call_once(|| {
    ///         let (dfa, _) = DFA::from_bytes(BYTES)
    ///             .expect("serialized DFA should be valid");
    ///         // SAFETY: This is guaranteed to only execute once, and all
    ///         // we do with the pointer is write the DFA to it.
    ///         unsafe {
    ///             (*DFA.0.as_ptr()).as_mut_ptr().write(dfa);
    ///         }
    ///     });
    ///     // SAFETY: DFA is guaranteed to by initialized via INIT and is
    ///     // stored in static memory.
    ///     unsafe {
    ///         let dfa = (*DFA.0.as_ptr()).as_ptr();
    ///         std::mem::transmute::<*const DFA, &'static DFA>(dfa)
    ///     }
    /// }
    ///
    /// let dfa = get_foo();
    /// let expected = HalfMatch { pattern: 0, offset: 8 };
    /// assert_eq!(Ok(Some(expected)), dfa.find_leftmost_fwd(b"foo12345"));
    /// ```
    ///
    /// Alternatively, consider using
    /// [`lazy_static`](https://crates.io/crates/lazy_static)
    /// or
    /// [`once_cell`](https://crates.io/crates/once_cell),
    /// which will guarantee safety for you.
    pub fn from_bytes(
        mut slice: &'a [u8],
    ) -> Result<(DFA<&'a [u8], S>, usize), DeserializeError> {
        // SAFETY: This is safe because we validate both the sparse transitions
        // (by trying to decode every state) and start state ID list below. If
        // either validation fails, then we return an error.
        let (dfa, nread) = unsafe { DFA::from_bytes_unchecked(slice)? };
        dfa.trans.validate()?;
        dfa.starts.validate(&dfa.trans)?;
        Ok((dfa, nread))
    }

    /// Deserialize a DFA with a specific state identifier representation in
    /// constant time by omitting the verification of the validity of the
    /// sparse transitions.
    ///
    /// This is just like
    /// [`from_bytes`](struct.DFA.html#method.from_bytes),
    /// except it can potentially return a DFA that exhibits undefined behavior
    /// if its transitions contains invalid state identifiers.
    ///
    /// This routine is useful if you need to deserialize a DFA cheaply and
    /// cannot afford the transition validation performed by `from_bytes`.
    ///
    /// # Safety
    ///
    /// This routine is unsafe because it permits callers to provide
    /// arbitrary transitions with possibly incorrect state identifiers. While
    /// the various serialization routines will never return an incorrect
    /// DFA, there is no guarantee that the bytes provided here
    /// are correct. While `from_bytes_unchecked` will still do several forms
    /// of basic validation, this routine does not check that the transitions
    /// themselves are correct. Given an incorrect transition table, it is
    /// possible for the search routines to access out-of-bounds memory because
    /// of explicit bounds check elision.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, sparse::DFA};
    ///
    /// let initial = DFA::new("foo[0-9]+")?;
    /// let bytes = initial.to_sized::<u16>()?.to_bytes_native_endian();
    /// // SAFETY: This is guaranteed to be safe since the bytes given come
    /// // directly from a compatible serialization routine.
    /// let dfa: DFA<&[u8], u16> = unsafe {
    ///     DFA::from_bytes_unchecked(&bytes)?.0
    /// };
    ///
    /// let expected = HalfMatch { pattern: 0, offset: 8 };
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo12345")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub unsafe fn from_bytes_unchecked(
        mut slice: &'a [u8],
    ) -> Result<(DFA<&'a [u8], S>, usize), DeserializeError> {
        let mut nr = 0;

        nr += bytes::read_label(&slice[nr..], LABEL)?;
        nr += bytes::read_endianness_check(&slice[nr..])?;
        nr += bytes::read_version(&slice[nr..], VERSION)?;
        nr += bytes::read_state_size::<S>(&slice[nr..])?;

        let _unused = bytes::try_read_u64(&slice[nr..], "unused space")?;
        nr += 8;

        let (trans, nread) = Transitions::from_bytes_unchecked(&slice[nr..])?;
        nr += nread;

        let (starts, nread) = StartList::from_bytes_unchecked(&slice[nr..])?;
        nr += nread;

        let (special, nread): (Special<S>, usize) =
            Special::from_bytes(&slice[nr..])?;
        nr += nread;
        if special.max.as_usize() >= trans.sparse().len() {
            return Err(DeserializeError::generic(
                "max should not be greater than or equal to sparse bytes",
            ));
        }

        Ok((DFA { trans, starts, special }, nr))
    }
}

impl<T: AsRef<[u8]>, S: StateID> fmt::Debug for DFA<T, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "sparse::DFA(")?;
        for state in self.trans.states() {
            fmt_state_indicator(f, self, state.id())?;
            writeln!(f, "{:06}: {:?}", state.id().as_usize(), state)?;
        }
        writeln!(f, "")?;
        for (i, start_id) in self.starts.ids().enumerate() {
            let sty = Start::from_usize(i).expect("must have start type");
            writeln!(
                f,
                "START({}): {:?} => {:06}",
                i,
                sty,
                start_id.as_usize(),
            )?;
        }
        writeln!(f, "state count: {}", self.trans.count)?;
        writeln!(f, ")")?;
        Ok(())
    }
}

unsafe impl<T: AsRef<[u8]>, S: StateID> Automaton for DFA<T, S> {
    type ID = S;

    #[inline]
    fn is_special_state(&self, id: S) -> bool {
        self.special.is_special_state(id)
    }

    #[inline]
    fn is_dead_state(&self, id: S) -> bool {
        self.special.is_dead_state(id)
    }

    #[inline]
    fn is_quit_state(&self, id: S) -> bool {
        self.special.is_quit_state(id)
    }

    #[inline]
    fn is_match_state(&self, id: S) -> bool {
        self.special.is_match_state(id)
    }

    #[inline]
    fn is_start_state(&self, id: S) -> bool {
        self.special.is_start_state(id)
    }

    #[inline]
    fn is_accel_state(&self, id: S) -> bool {
        self.special.is_accel_state(id)
    }

    #[inline(always)]
    fn next_state(&self, current: S, input: u8) -> S {
        let input = self.trans.classes.get(input);
        self.trans.state(current).next(input)
    }

    #[inline]
    unsafe fn next_state_unchecked(&self, current: S, input: u8) -> S {
        self.next_state(current, input)
    }

    #[inline]
    fn next_eof_state(&self, current: S) -> S {
        self.trans.state(current).next_eof()
    }

    #[inline]
    fn patterns(&self) -> usize {
        self.trans.patterns
    }

    #[inline]
    fn match_offset(&self) -> usize {
        MATCH_OFFSET
    }

    #[inline]
    fn match_count(&self, id: Self::ID) -> usize {
        self.trans.state(id).pattern_count()
    }

    #[inline]
    fn match_pattern(&self, id: Self::ID, match_index: usize) -> PatternID {
        self.trans.state(id).pattern_id(match_index)
    }

    #[inline]
    fn start_state_forward(
        &self,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> S {
        let index = Start::from_position_fwd(bytes, start, end);
        self.starts.start(index)
    }

    #[inline]
    fn start_state_reverse(
        &self,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> S {
        let index = Start::from_position_rev(bytes, start, end);
        self.starts.start(index)
    }

    #[inline]
    fn accelerator(&self, id: Self::ID) -> &[u8] {
        self.trans.state(id).accelerator()
    }
}

impl<'a, S: StateID> Transitions<&'a [u8], S> {
    unsafe fn from_bytes_unchecked(
        mut slice: &'a [u8],
    ) -> Result<(Transitions<&'a [u8], S>, usize), DeserializeError> {
        let mut nread = 0;

        let count: usize =
            bytes::try_read_u64_as_usize(&slice[nread..], "state count")?;
        nread += 8;

        let patterns: usize =
            bytes::try_read_u64_as_usize(&slice[nread..], "pattern count")?;
        nread += 8;

        let (classes, n) = ByteClasses::from_bytes(&slice[nread..])?;
        nread += n;

        let len = bytes::try_read_u64_as_usize(
            &slice[nread..],
            "sparse transitions length",
        )?;
        nread += 8;

        if slice.len() < nread + len {
            return Err(DeserializeError::buffer_too_small(
                "sparse transitions",
            ));
        }
        let sparse = &slice[nread..nread + len];
        nread += len;

        let trans = Transitions {
            sparse,
            classes,
            count,
            patterns,
            _state_id: PhantomData,
        };
        Ok((trans, nread))
    }
}

impl<T: AsRef<[u8]>, S: StateID> Transitions<T, S> {
    /// Writes a serialized form of this transition table to the buffer given.
    /// If the buffer is too small, then an error is returned. To determine
    /// how big the buffer must be, use `write_to_len`.
    fn write_to<E: Endian>(
        &self,
        mut dst: &mut [u8],
    ) -> Result<usize, SerializeError> {
        let nwrite = self.write_to_len();
        if dst.len() < nwrite {
            return Err(SerializeError::buffer_too_small(
                "sparse transition table",
            ));
        }
        dst = &mut dst[..nwrite];

        // write state count
        E::write_u64(self.count as u64, dst);
        dst = &mut dst[8..];

        // write pattern count
        E::write_u64(self.patterns as u64, dst);
        dst = &mut dst[8..];

        // write byte class map
        let n = self.classes.write_to(dst)?;
        dst = &mut dst[n..];

        // write number of bytes in sparse transitions
        E::write_u64(self.sparse().len() as u64, dst);
        dst = &mut dst[8..];

        // write actual transitions
        dst.copy_from_slice(self.sparse());
        Ok(nwrite)
    }

    /// Returns the number of bytes the serialized form of this transition
    /// table will use.
    fn write_to_len(&self) -> usize {
        8   // state count
        + 8 // pattern count
        + self.classes.write_to_len()
        + 8 // sparse transitions length
        + self.sparse().len()
    }

    /// Validates that every state ID in this transition table is valid.
    ///
    /// That is, every state ID can be used to correctly index a state in this
    /// table.
    fn validate(&self) -> Result<(), DeserializeError> {
        // In order to validate everything, we not only need to make sure we
        // can decode every state, but that every transition in every state
        // points to a valid state. There are many duplicative transitions, so
        // we record state IDs that we've verified so that we don't redo the
        // decoding work.
        let mut verified = BTreeSet::new();
        // We need to make sure that we decode the correct number of states.
        // Otherwise, an empty set of transitions would validate even if the
        // recorded state count is non-empty.
        let mut count = 0;
        // We can't use the self.states() iterator because it assumes the state
        // encodings are valid. It could panic if they aren't.
        let mut id: S = dead_id();
        while id.as_usize() < self.sparse().len() {
            let state = self.try_state(id)?;
            verified.insert(id);
            // The next ID should be the offset immediately following `state`.
            id = S::from_usize(id.as_usize() + state.bytes_len());
            count += 1;

            // Now check that all transitions in this state are correct.
            for i in 0..state.ntrans {
                let to = state.next_at(i);
                if verified.contains(&to) {
                    continue;
                }
                let _ = self.try_state(to)?;
                verified.insert(id);
            }
            // And also that every pattern ID is valid.
            for i in 0..state.pattern_count() {
                let pid = state.pattern_id(i);
                if pid as usize >= self.patterns {
                    return Err(DeserializeError::generic(
                        "invalid pattern ID",
                    ));
                }
            }
        }
        if count != self.count {
            return Err(DeserializeError::generic(
                "mismatching sparse state count",
            ));
        }
        Ok(())
    }

    /// Converts these transitions to a borrowed value.
    fn as_ref(&self) -> Transitions<&'_ [u8], S> {
        Transitions {
            sparse: self.sparse(),
            classes: self.classes.clone(),
            count: self.count,
            patterns: self.patterns,
            _state_id: self._state_id,
        }
    }

    /// Converts these transitions to an owned value.
    fn to_owned(&self) -> Transitions<Vec<u8>, S> {
        Transitions {
            sparse: self.sparse().to_vec(),
            classes: self.classes.clone(),
            count: self.count,
            patterns: self.patterns,
            _state_id: self._state_id,
        }
    }

    /// Converts these transitions from using S as its state ID representation
    /// to using S2. If `size_of::<S2> >= size_of::<S>()`, then this always
    /// succeeds. If `size_of::<S2> < size_of::<S>()` and if S2 cannot
    /// represent every state ID in these transitions, then an error is
    /// returned.
    ///
    /// This also returns a mapping from old state IDs to new state IDs, which
    /// can be used to remap state IDs elsewhere (such as starting state IDs).
    fn to_sized<S2: StateID>(
        &self,
    ) -> Result<(Transitions<Vec<u8>, S2>, BTreeMap<S, S2>), Error> {
        let mut remap: BTreeMap<S, S2> = BTreeMap::new();
        let mut sparse = Vec::with_capacity(size_of::<S2>() * self.count);
        for state in self.states() {
            let pos = sparse.len();
            remap.insert(state.id(), usize_to_state_id(pos)?);

            let n = state.ntrans;
            let zeros = 2 + (n * 2) + (n * size_of::<S2>());
            sparse.extend(iter::repeat(0).take(zeros));

            let ntrans =
                if state.is_match { n as u16 | (1 << 15) } else { n as u16 };
            bytes::NE::write_u16(ntrans, &mut sparse[pos..]);
            let (s, e) = (pos + 2, pos + 2 + (n * 2));
            sparse[s..e].copy_from_slice(state.input_ranges);

            if state.is_match {
                let n = state.pattern_count();
                let zeros = 4 + (4 * n);
                let mut pos = sparse.len();
                sparse.extend(iter::repeat(0).take(zeros));
                bytes::NE::write_u32(n as u32, &mut sparse[pos..]);
                pos += 4;
                sparse[pos..pos + 4 * n].copy_from_slice(state.pattern_ids);
            }

            let accel = state.accelerator();
            sparse.push(accel.len().try_into().unwrap());
            sparse.extend_from_slice(accel);
        }
        let mut trans = Transitions {
            sparse,
            classes: self.classes.clone(),
            count: self.count,
            patterns: self.patterns,
            _state_id: PhantomData,
        };
        for (&old_id, &new_id) in remap.iter() {
            let old_state = self.state(old_id);
            let mut new_state = trans.state_mut(new_id);
            for i in 0..new_state.ntrans {
                let next = remap[&old_state.next_at(i)];
                new_state.set_next_at(i, next);
            }
        }
        Ok((trans, remap))
    }

    /// Return a convenient representation of the given state.
    ///
    /// This panics if the state is invalid.
    #[inline]
    fn state(&self, id: S) -> State<'_, S> {
        let mut state = &self.sparse()[id.as_usize()..];
        let mut ntrans = bytes::read_u16(&state) as usize;
        let is_match = (1 << 15) & ntrans != 0;
        ntrans &= !(1 << 15);
        state = &state[2..];

        let (input_ranges, state) = state.split_at(ntrans * 2);
        let (next, state) = state.split_at(ntrans * size_of::<S>());
        let (pattern_ids, state) = if is_match {
            let npats = bytes::read_u32(&state) as usize;
            state[4..].split_at(npats * 4)
        } else {
            (&[][..], state)
        };

        let accel_len = state[0] as usize;
        let accel = &state[1..accel_len + 1];
        State { id, is_match, ntrans, input_ranges, next, pattern_ids, accel }
    }

    /// Like `state`, but will return an error if the state encoding is
    /// invalid. This is useful for verifying states after deserialization,
    /// which is required for a safe deserialization API.
    ///
    /// Note that this only verifies that this state is decodable and that
    /// all of its data is consistent. It does not verify that its state ID
    /// transitions point to valid states themselves, nor does it verify that
    /// every pattern ID is valid.
    fn try_state(&self, id: S) -> Result<State<'_, S>, DeserializeError> {
        if id.as_usize() > self.sparse().len() {
            return Err(DeserializeError::generic("invalid sparse state ID"));
        }
        let mut state = &self.sparse()[id.as_usize()..];
        // Encoding format starts with a u16 that stores the total number of
        // transitions in this state.
        let mut ntrans =
            bytes::try_read_u16(state, "state transition count")? as usize;
        let is_match = (1 << 15) & ntrans != 0;
        ntrans &= !(1 << 15);
        state = &state[2..];
        if ntrans > 257 {
            return Err(DeserializeError::generic("invalid transition count"));
        }

        // Each transition has two pieces: an inclusive range of bytes on which
        // it is defined, and the state ID that those bytes transition to. The
        // pairs come first, followed by a corresponding sequence of state IDs.
        let input_ranges_len = ntrans * 2;
        if input_ranges_len > state.len() {
            return Err(DeserializeError::generic("no sparse byte pairs"));
        }
        let (input_ranges, state) = state.split_at(input_ranges_len);
        // Every range should be of the form A-B, where A<=B.
        for pair in input_ranges.chunks(2) {
            let (start, end) = (pair[0], pair[1]);
            if start > end {
                return Err(DeserializeError::generic("invalid input range"));
            }
        }

        // And now extract the corresponding sequence of state IDs. We leave
        // this sequence as a &[u8] instead of a &[S] because sparse DFAs do
        // not have any alignment requirements.
        let next_len = ntrans * self.id_len();
        if next_len > state.len() {
            return Err(DeserializeError::generic("no transition state IDs"));
        }
        let (next, state) = state.split_at(next_len);
        // We can at least verify that every state ID is in bounds.
        for idbytes in next.chunks(self.id_len()) {
            let id = S::read_bytes(idbytes);
            if id.as_usize() > self.sparse().len() {
                return Err(DeserializeError::generic(
                    "out of bounds next ID",
                ));
            }
        }

        // If this is a match state, then read the pattern IDs for this state.
        // Patterns IDs is a u32-length prefixed sequence of native endian
        // encoded 32-bit integers.
        let (pattern_ids, state) = if is_match {
            let npats: usize = bytes::try_read_u32(state, "pattern ID count")?
                .try_into()
                .map_err(|_| {
                    // If the number of patterns doesn't fit into usize, then
                    // we have a problem because the slice will be too big.
                    DeserializeError::invalid_usize("pattern ID count")
                })?;
            let state = &state[4..];

            let pattern_ids_len = npats * 4;
            if pattern_ids_len > state.len() {
                return Err(DeserializeError::generic(
                    "no sparse pattern IDs",
                ));
            }
            let (pattern_ids, state) = state.split_at(pattern_ids_len);
            // Every pattern ID should be strictly less than the one after it.
            let mut prev = None;
            for chunk in pattern_ids.chunks(4) {
                let pid = u32::from_ne_bytes(chunk.try_into().unwrap());
                if prev.map_or(false, |p| p >= pid) {
                    return Err(DeserializeError::generic(
                        "pattern IDs are not monotonically increasing",
                    ));
                }
                prev = Some(pid);
            }
            (pattern_ids, state)
        } else {
            (&[][..], state)
        };

        // Now read this state's accelerator info. The first byte is the length
        // of the accelerator, which is typically 0 (for no acceleration) but
        // is no bigger than 3. The length indicates the number of bytes that
        // follow, where each byte corresponds to a transition out of this
        // state.
        if state.is_empty() {
            return Err(DeserializeError::generic("no accelerator length"));
        }
        let (accel_len, state) = (state[0] as usize, &state[1..]);

        if accel_len > 3 || accel_len > state.len() {
            return Err(DeserializeError::generic(
                "invalid accelerator length",
            ));
        }
        let (accel, state) = (&state[..accel_len], &state[accel_len..]);

        Ok(State {
            id,
            is_match,
            ntrans,
            input_ranges,
            next,
            pattern_ids,
            accel,
        })
    }

    /// Return an iterator over all of the states in this DFA.
    ///
    /// The iterator returned yields tuples, where the first element is the
    /// state ID and the second element is the state itself.
    fn states(&self) -> StateIter<'_, T, S> {
        StateIter { trans: self, id: dead_id() }
    }

    /// Returns the sparse transitions as raw bytes.
    fn sparse(&self) -> &[u8] {
        self.sparse.as_ref()
    }

    /// Returns the number of bytes represented by a single state ID.
    fn id_len(&self) -> usize {
        core::mem::size_of::<S>()
    }

    /// Return the memory usage, in bytes, of these transitions.
    ///
    /// This does not include the size of a `Transitions` value itself.
    fn memory_usage(&self) -> usize {
        self.sparse().len()
    }
}

impl<T: AsMut<[u8]>, S: StateID> Transitions<T, S> {
    /// Return a convenient mutable representation of the given state.
    /// This panics if the state is invalid.
    fn state_mut(&mut self, id: S) -> StateMut<'_, S> {
        let mut state = &mut self.sparse_mut()[id.as_usize()..];
        let mut ntrans = bytes::read_u16(&state) as usize;
        let is_match = (1 << 15) & ntrans != 0;
        ntrans &= !(1 << 15);
        state = &mut state[2..];

        let (input_ranges, state) = state.split_at_mut(ntrans * 2);
        let (next, state) = state.split_at_mut(ntrans * size_of::<S>());
        let (pattern_ids, state) = if is_match {
            let npats = bytes::read_u32(&state) as usize;
            state[4..].split_at_mut(npats * 4)
        } else {
            (&mut [][..], state)
        };

        let accel_len = state[0] as usize;
        let accel = &mut state[1..accel_len + 1];
        StateMut {
            id,
            is_match,
            ntrans,
            input_ranges,
            next,
            pattern_ids,
            accel,
        }
    }

    /// Returns the sparse transitions as raw mutable bytes.
    fn sparse_mut(&mut self) -> &mut [u8] {
        self.sparse.as_mut()
    }
}

impl<S: StateID> StartList<Vec<u8>, S> {
    fn new(count: usize) -> StartList<Vec<u8>, S> {
        let stride = core::mem::size_of::<S>();
        StartList { list: vec![0; count * stride], _state_id: PhantomData }
    }

    fn from_dense_dfa<T: AsRef<[S]>, A: AsRef<[u8]>, S2: StateID>(
        dfa: &dense::DFA<T, A, S>,
        remap: &[S2],
    ) -> Result<StartList<Vec<u8>, S2>, Error> {
        // TODO: Shouldn't this possibly return an error?
        let mut sl = StartList::new(dfa.starts().len());
        for (i, &old_start_id) in dfa.starts().iter().enumerate() {
            let new_start_id = remap[dfa.to_index(old_start_id)];
            let start_index = Start::from_usize(i).unwrap();
            sl.set_start(start_index, new_start_id);
        }
        Ok(sl)
    }
}

impl<'a, S: StateID> StartList<&'a [u8], S> {
    unsafe fn from_bytes_unchecked(
        mut slice: &'a [u8],
    ) -> Result<(StartList<&'a [u8], S>, usize), DeserializeError> {
        let count: usize =
            bytes::try_read_u64_as_usize(slice, "sparse start ID count")?;
        slice = &slice[8..];

        let len = count * core::mem::size_of::<S>();
        if slice.len() < len {
            return Err(DeserializeError::buffer_too_small(
                "sparse start ID list",
            ));
        }
        let sl = StartList { list: &slice[..len], _state_id: PhantomData };
        let nread = 8 + len;
        Ok((sl, nread))
    }
}

impl<T: AsRef<[u8]>, S: StateID> StartList<T, S> {
    fn write_to<E: Endian>(
        &self,
        mut dst: &mut [u8],
    ) -> Result<usize, SerializeError> {
        let nwrite = self.write_to_len();
        if dst.len() < nwrite {
            return Err(SerializeError::buffer_too_small(
                "sparse starting list ids",
            ));
        }
        dst = &mut dst[..nwrite];

        // write state ID count
        E::write_u64(self.count() as u64, dst);
        dst = &mut dst[8..];

        // write start IDs
        dst.copy_from_slice(self.list());
        Ok(nwrite)
    }

    /// Returns the number of bytes the serialized form of this transition
    /// table will use.
    fn write_to_len(&self) -> usize {
        8 // state ID count
        + self.list().len()
    }

    /// Validates that every starting state ID in this list is valid.
    ///
    /// That is, every starting state ID can be used to correctly decode a
    /// state in the DFA's sparse transitions.
    fn validate(
        &self,
        trans: &Transitions<T, S>,
    ) -> Result<(), DeserializeError> {
        for id in self.ids() {
            let _ = trans.try_state(id)?;
        }
        Ok(())
    }

    /// Converts this start list to a borrowed value.
    fn as_ref(&self) -> StartList<&'_ [u8], S> {
        StartList { list: self.list(), _state_id: self._state_id }
    }

    /// Converts this start list to an owned value.
    fn to_owned(&self) -> StartList<Vec<u8>, S> {
        StartList { list: self.list().to_vec(), _state_id: self._state_id }
    }

    /// Converts this list of starting IDs from a list that uses S as its state
    /// ID representation to one that uses S2. If
    /// `size_of::<S2> >= size_of::<S>()`, then this always succeeds. If
    /// `size_of::<S2> < size_of::<S>()` and if S2 cannot represent every state
    /// ID in this list, then an error is returned.
    fn to_sized<S2: StateID>(
        &self,
        remap: &BTreeMap<S, S2>,
    ) -> StartList<Vec<u8>, S2> {
        let mut sl = StartList::new(self.count());
        for (i, old_start_id) in self.ids().enumerate() {
            let new_start_id = remap[&old_start_id];
            let start_index = Start::from_usize(i).unwrap();
            sl.set_start(start_index, new_start_id);
        }
        sl
    }

    /// Return the start state for the given index.
    fn start(&self, index: Start) -> S {
        let start = index.as_usize() * self.stride();
        let end = start + self.stride();
        S::read_bytes(&self.list()[start..end])
    }

    /// Returns the total number of start states in this list.
    fn count(&self) -> usize {
        assert!(self.list().len() % self.stride() == 0);
        self.list().len() / self.stride()
    }

    /// Return the total number of bytes that represents each state ID.
    fn stride(&self) -> usize {
        core::mem::size_of::<S>()
    }

    /// Return an iterator over all start IDs in this list.
    fn ids(&self) -> StartStateIDIter<'_, T, S> {
        StartStateIDIter { starts: self, i: 0 }
    }

    /// Returns the list as a raw slice of bytes.
    fn list(&self) -> &[u8] {
        self.list.as_ref()
    }

    /// Return the memory usage, in bytes, of this start list.
    ///
    /// This does not include the size of a `StartList` value itself.
    fn memory_usage(&self) -> usize {
        self.list().len()
    }
}

impl<T: AsRef<[u8]> + AsMut<[u8]>, S: StateID> StartList<T, S> {
    /// Set the start state for the given index.
    fn set_start(&mut self, index: Start, id: S) {
        let start = index.as_usize() * self.stride();
        let end = start + self.stride();
        id.write_bytes(&mut self.list_mut()[start..end]);
    }

    /// Returns the list of start IDs as a mutable slice of state IDs.
    fn list_mut(&mut self) -> &mut [u8] {
        self.list.as_mut()
    }
}

/// An iterator over all state state IDs in a sparse DFA.
struct StartStateIDIter<'a, T, S> {
    starts: &'a StartList<T, S>,
    i: usize,
}

impl<'a, T: AsRef<[u8]>, S: StateID> Iterator for StartStateIDIter<'a, T, S> {
    type Item = S;

    fn next(&mut self) -> Option<S> {
        let index = Start::from_usize(self.i)?;
        self.i += 1;
        Some(self.starts.start(index))
    }
}

impl<'a, T, S: StateID> fmt::Debug for StartStateIDIter<'a, T, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("StartStateIDIter").field("i", &self.i).finish()
    }
}

/// An iterator over all states in a sparse DFA.
///
/// This iterator yields tuples, where the first element is the state ID and
/// the second element is the state itself.
struct StateIter<'a, T, S> {
    trans: &'a Transitions<T, S>,
    id: S,
}

impl<'a, T: AsRef<[u8]>, S: StateID> Iterator for StateIter<'a, T, S> {
    type Item = State<'a, S>;

    fn next(&mut self) -> Option<State<'a, S>> {
        if self.id.as_usize() >= self.trans.sparse().len() {
            return None;
        }
        let state = self.trans.state(self.id);
        self.id = S::from_usize(self.id.as_usize() + state.bytes_len());
        Some(state)
    }
}

impl<'a, T, S: StateID> fmt::Debug for StateIter<'a, T, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("StateIter").field("id", &self.id).finish()
    }
}

/// A representation of a sparse DFA state that can be cheaply materialized
/// from a state identifier.
#[derive(Clone)]
struct State<'a, S> {
    /// The identifier of this state.
    id: S,
    /// Whether this is a match state or not.
    is_match: bool,
    /// The number of transitions in this state.
    ntrans: usize,
    /// Pairs of input ranges, where there is one pair for each transition.
    /// Each pair specifies an inclusive start and end byte range for the
    /// corresponding transition.
    input_ranges: &'a [u8],
    /// Transitions to the next state. This slice contains native endian
    /// encoded state identifiers, with `S` as the representation. Thus, there
    /// are `ntrans * size_of::<S>()` bytes in this slice.
    next: &'a [u8],
    /// If this is a match state, then this contains the pattern IDs that match
    /// when the DFA is in this state.
    ///
    /// This is a contiguous sequence of 32-bit native endian encoded integers.
    pattern_ids: &'a [u8],
    /// An accelerator for this state, if present. If this state has no
    /// accelerator, then this is an empty slice. When non-empty, this slice
    /// has length at most 3 and corresponds to the exhaustive set of bytes
    /// that must be seen in order to transition out of this state.
    accel: &'a [u8],
}

impl<'a, S: StateID> State<'a, S> {
    /// Searches for the next transition given an input byte. If no such
    /// transition could be found, then a dead state is returned.
    fn next(&self, input: u8) -> S {
        // This straight linear search was observed to be much better than
        // binary search on ASCII haystacks, likely because a binary search
        // visits the ASCII case last but a linear search sees it first. A
        // binary search does do a little better on non-ASCII haystacks, but
        // not by much. There might be a better trade off lurking here.
        for i in 0..(self.ntrans - 1) {
            let (start, end) = self.range(i);
            if start <= input && input <= end {
                return self.next_at(i);
            }
            // We could bail early with an extra branch: if input < b1, then
            // we know we'll never find a matching transition. Interestingly,
            // this extra branch seems to not help performance, or will even
            // hurt it. It's likely very dependent on the DFA itself and what
            // is being searched.
        }
        dead_id()
    }

    /// Returns the next state ID for the special EOF transition.
    fn next_eof(&self) -> S {
        self.next_at(self.ntrans - 1)
    }

    /// Returns the identifier for this state.
    fn id(&self) -> S {
        self.id
    }

    /// Returns the inclusive input byte range for the ith transition in this
    /// state.
    fn range(&self, i: usize) -> (u8, u8) {
        (self.input_ranges[i * 2], self.input_ranges[i * 2 + 1])
    }

    /// Returns the next state for the ith transition in this state.
    fn next_at(&self, i: usize) -> S {
        S::read_bytes(&self.next[i * size_of::<S>()..])
    }

    /// Returns the pattern ID for the given match index. If the match index
    /// is invalid, then this panics.
    fn pattern_id(&self, match_index: usize) -> PatternID {
        let i = match_index * 4;
        let bytes = &self.pattern_ids[i..i + 4];
        u32::from_ne_bytes(bytes.try_into().unwrap())
    }

    /// Returns the total number of pattern IDs for this state. This is always
    /// zero when `is_match` is false.
    fn pattern_count(&self) -> usize {
        assert_eq!(0, self.pattern_ids.len() % 4);
        self.pattern_ids.len() / 4
    }

    /// Return the total number of bytes that this state consumes in its
    /// encoded form.
    fn bytes_len(&self) -> usize {
        let mut len = 2
            + (self.ntrans * 2)
            + (self.ntrans * size_of::<S>())
            + (1 + self.accel.len());
        if self.is_match {
            len += 4 + self.pattern_ids.len();
        }
        len
    }

    /// Return an accelerator for this state.
    fn accelerator(&self) -> &'a [u8] {
        self.accel
    }
}

impl<'a, S: StateID> fmt::Debug for State<'a, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut printed = false;
        for i in 0..(self.ntrans - 1) {
            let next = self.next_at(i);
            if next == dead_id() {
                continue;
            }

            if printed {
                write!(f, ", ")?;
            }
            let (start, end) = self.range(i);
            if start == end {
                write!(f, "{:?} => {}", DebugByte(start), next.as_usize())?;
            } else {
                write!(
                    f,
                    "{:?}-{:?} => {}",
                    DebugByte(start),
                    DebugByte(end),
                    next.as_usize()
                )?;
            }
            printed = true;
        }
        let eof = self.next_at(self.ntrans - 1);
        if eof != dead_id() {
            write!(f, "EOF => {}", eof.as_usize())?;
        }
        Ok(())
    }
}

/// A representation of a mutable sparse DFA state that can be cheaply
/// materialized from a state identifier.
#[cfg(feature = "std")]
struct StateMut<'a, S> {
    /// The identifier of this state.
    id: S,
    /// Whether this is a match state or not.
    is_match: bool,
    /// The number of transitions in this state.
    ntrans: usize,
    /// Pairs of input ranges, where there is one pair for each transition.
    /// Each pair specifies an inclusive start and end byte range for the
    /// corresponding transition.
    input_ranges: &'a mut [u8],
    /// Transitions to the next state. This slice contains native endian
    /// encoded state identifiers, with `S` as the representation. Thus, there
    /// are `ntrans * size_of::<S>()` bytes in this slice.
    next: &'a mut [u8],
    /// If this is a match state, then this contains the pattern IDs that match
    /// when the DFA is in this state.
    ///
    /// This is a contiguous sequence of 32-bit native endian encoded integers.
    pattern_ids: &'a [u8],
    /// An accelerator for this state, if present. If this state has no
    /// accelerator, then this is an empty slice. When non-empty, this slice
    /// has length at most 3 and corresponds to the exhaustive set of bytes
    /// that must be seen in order to transition out of this state.
    accel: &'a mut [u8],
}

#[cfg(feature = "std")]
impl<'a, S: StateID> StateMut<'a, S> {
    /// Sets the ith transition to the given state.
    fn set_next_at(&mut self, i: usize, next: S) {
        next.write_bytes(&mut self.next[i * size_of::<S>()..]);
    }
}

#[cfg(feature = "std")]
impl<'a, S: StateID> fmt::Debug for StateMut<'a, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = State {
            id: self.id,
            is_match: self.is_match,
            ntrans: self.ntrans,
            input_ranges: self.input_ranges,
            next: self.next,
            pattern_ids: self.pattern_ids,
            accel: self.accel,
        };
        fmt::Debug::fmt(&state, f)
    }
}

/// Convert the given `usize` to the chosen state identifier
/// representation. If the given value cannot fit in the chosen
/// representation, then an error is returned.
#[cfg(feature = "std")]
fn usize_to_state_id<S: StateID>(value: usize) -> Result<S, Error> {
    if value > S::max_id() {
        Err(Error::state_id_overflow(S::max_id()))
    } else {
        Ok(S::from_usize(value))
    }
}

/// A binary search routine specialized specifically to a sparse DFA state's
/// transitions. Specifically, the transitions are defined as a set of pairs
/// of input bytes that delineate an inclusive range of bytes. If the input
/// byte is in the range, then the corresponding transition is a match.
///
/// This binary search accepts a slice of these pairs and returns the position
/// of the matching pair (the ith transition), or None if no matching pair
/// could be found.
///
/// Note that this routine is not currently used since it was observed to
/// either decrease performance when searching ASCII, or did not provide enough
/// of a boost on non-ASCII haystacks to be worth it. However, we leave it here
/// for posterity in case we can find a way to use it.
///
/// In theory, we could use the standard library's search routine if we could
/// cast a `&[u8]` to a `&[(u8, u8)]`, but I don't believe this is currently
/// guaranteed to be safe and is thus UB (since I don't think the in-memory
/// representation of `(u8, u8)` has been nailed down). One could define a
/// repr(C) type, but the casting doesn't seem justified.
#[allow(dead_code)]
#[inline(always)]
fn binary_search_ranges(ranges: &[u8], needle: u8) -> Option<usize> {
    debug_assert!(ranges.len() % 2 == 0, "ranges must have even length");
    debug_assert!(ranges.len() <= 512, "ranges should be short");

    let (mut left, mut right) = (0, ranges.len() / 2);
    while left < right {
        let mid = (left + right) / 2;
        let (b1, b2) = (ranges[mid * 2], ranges[mid * 2 + 1]);
        if needle < b1 {
            right = mid;
        } else if needle > b2 {
            left = mid + 1;
        } else {
            return Some(mid);
        }
    }
    None
}
