#![allow(warnings)]

// BREADCRUMBS: In the large Unicode classes, it looks like *a lot* of DFA
// states have several transitions that all map to the same state. We could
// potentially save a bit of space by simply storing only one state ID, and
// saying that any match of the input byte goes to that state ID.

use std::fmt;
use std::iter;
use std::marker::PhantomData;
use std::mem::size_of;

use byteorder::{ByteOrder, NativeEndian};

use classes::ByteClasses;
use dense::DenseDFA;
use dfa::DFA;
use error::Result;
use state_id::{StateID, dead_id, usize_to_state_id};

#[derive(Clone, Debug)]
pub enum SparseDFA<T: AsRef<[u8]>, S: StateID = usize> {
    Standard(SparseDFAStandard<T, S>),
    ByteClass(SparseDFAByteClass<T, S>),
    /// Hints that destructuring should not be exhaustive.
    ///
    /// This enum may grow additional variants, so this makes sure clients
    /// don't count on exhaustive matching. (Otherwise, adding a new variant
    /// could break existing code.)
    #[doc(hidden)]
    __Nonexhaustive,
}

impl<S: StateID> SparseDFA<Vec<u8>, S> {
    pub(crate) fn from_dfa_sized<T: AsRef<[S]>, A: StateID>(
        dfa: &DenseDFA<T, S>,
    ) -> Result<SparseDFA<Vec<u8>, A>> {
        SparseDFARepr::from_dfa_sized(dfa).map(|r| r.into_sparse_dfa())
    }
}

impl<T: AsRef<[u8]>, S: StateID> SparseDFA<T, S> {
    /// Cheaply return a borrowed version of this sparse DFA. Specifically, the
    /// DFA returned always uses `&[u8]` for its transition table while keeping
    /// the same state identifier representation.
    pub fn as_ref<'a>(&'a self) -> SparseDFA<&'a [u8], S> {
        match *self {
            SparseDFA::Standard(SparseDFAStandard(ref r)) => {
                SparseDFA::Standard(SparseDFAStandard(r.as_ref()))
            }
            SparseDFA::ByteClass(SparseDFAByteClass(ref r)) => {
                SparseDFA::ByteClass(SparseDFAByteClass(r.as_ref()))
            }
            SparseDFA::__Nonexhaustive => unreachable!(),
        }
    }

    /// Return an owned version of this sparse DFA. Specifically, the DFA
    /// returned always uses `Vec<u8>` for its transition table while keeping
    /// the same state identifier representation.
    ///
    /// Effectively, this returns a sparse DFA whose transition table lives
    /// on the heap.
    pub fn to_owned(&self) -> SparseDFA<Vec<u8>, S> {
        match *self {
            SparseDFA::Standard(SparseDFAStandard(ref r)) => {
                SparseDFA::Standard(SparseDFAStandard(r.to_owned()))
            }
            SparseDFA::ByteClass(SparseDFAByteClass(ref r)) => {
                SparseDFA::ByteClass(SparseDFAByteClass(r.to_owned()))
            }
            SparseDFA::__Nonexhaustive => unreachable!(),
        }
    }

    /// Returns the memory usage, in bytes, of this DFA.
    ///
    /// The memory usage is computed based on the number of bytes used to
    /// represent this DFA's transition table. This typically corresponds to
    /// heap memory usage.
    ///
    /// This does **not** include the stack size used up by this DFA. To
    /// compute that, used `std::mem::size_of::<SparseDFA>()`.
    pub fn memory_usage(&self) -> usize {
        self.repr().memory_usage()
    }

    fn repr(&self) -> &SparseDFARepr<T, S> {
        match *self {
            SparseDFA::Standard(ref r) => &r.0,
            SparseDFA::ByteClass(ref r) => &r.0,
            SparseDFA::__Nonexhaustive => unreachable!(),
        }
    }
}

impl<T: AsRef<[u8]>, S: StateID> DFA for SparseDFA<T, S> {
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
            SparseDFA::Standard(ref r) => r.next_state(current, input),
            SparseDFA::ByteClass(ref r) => r.next_state(current, input),
            SparseDFA::__Nonexhaustive => unreachable!(),
        }
    }

    unsafe fn next_state_unchecked(&self, current: S, input: u8) -> S {
        self.next_state(current, input)
    }

    // We specialize the following methods because it lets us lift the
    // case analysis between the different types of sparse DFAs. Instead of
    // doing the case analysis for every transition, we do it once before
    // searching. For sparse DFAs, this doesn't seem to benefit performance as
    // much as it does for the dense DFAs, but it's easy to do so we might as
    // well do it.

    fn is_match(&self, bytes: &[u8]) -> bool {
        match *self {
            SparseDFA::Standard(ref r) => r.is_match(bytes),
            SparseDFA::ByteClass(ref r) => r.is_match(bytes),
            SparseDFA::__Nonexhaustive => unreachable!(),
        }
    }

    fn shortest_match(&self, bytes: &[u8]) -> Option<usize> {
        match *self {
            SparseDFA::Standard(ref r) => r.shortest_match(bytes),
            SparseDFA::ByteClass(ref r) => r.shortest_match(bytes),
            SparseDFA::__Nonexhaustive => unreachable!(),
        }
    }

    fn find(&self, bytes: &[u8]) -> Option<usize> {
        match *self {
            SparseDFA::Standard(ref r) => r.find(bytes),
            SparseDFA::ByteClass(ref r) => r.find(bytes),
            SparseDFA::__Nonexhaustive => unreachable!(),
        }
    }

    fn rfind(&self, bytes: &[u8]) -> Option<usize> {
        match *self {
            SparseDFA::Standard(ref r) => r.rfind(bytes),
            SparseDFA::ByteClass(ref r) => r.rfind(bytes),
            SparseDFA::__Nonexhaustive => unreachable!(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SparseDFAStandard<T: AsRef<[u8]>, S: StateID = usize>(
    SparseDFARepr<T, S>,
);

impl<T: AsRef<[u8]>, S: StateID> DFA for SparseDFAStandard<T, S> {
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
        self.0.state(current).next(input)
    }

    unsafe fn next_state_unchecked(&self, current: S, input: u8) -> S {
        self.next_state(current, input)
    }
}

#[derive(Clone, Debug)]
pub struct SparseDFAByteClass<T: AsRef<[u8]>, S: StateID = usize>(
    SparseDFARepr<T, S>,
);

impl<T: AsRef<[u8]>, S: StateID> DFA for SparseDFAByteClass<T, S> {
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
        self.0.state(current).next(input)
    }

    unsafe fn next_state_unchecked(&self, current: S, input: u8) -> S {
        self.next_state(current, input)
    }
}

/// The underlying representation of a sparse DFA. This is shared by all of
/// the different variants of a sparse DFA.
#[derive(Clone)]
struct SparseDFARepr<T: AsRef<[u8]>, S: StateID = usize> {
    start: S,
    state_count: usize,
    max_match: S,
    byte_classes: ByteClasses,
    trans: T,
}

impl<T: AsRef<[u8]>, S: StateID> SparseDFARepr<T, S> {
    fn into_sparse_dfa(self) -> SparseDFA<T, S> {
        if self.byte_classes.is_singleton() {
            SparseDFA::Standard(SparseDFAStandard(self))
        } else {
            SparseDFA::ByteClass(SparseDFAByteClass(self))
        }
    }

    fn as_ref<'a>(&'a self) -> SparseDFARepr<&'a [u8], S> {
        SparseDFARepr {
            start: self.start,
            state_count: self.state_count,
            max_match: self.max_match,
            byte_classes: self.byte_classes.clone(),
            trans: self.trans(),
        }
    }

    fn to_owned(&self) -> SparseDFARepr<Vec<u8>, S> {
        SparseDFARepr {
            start: self.start,
            state_count: self.state_count,
            max_match: self.max_match,
            byte_classes: self.byte_classes.clone(),
            trans: self.trans().to_vec(),
        }
    }

    /// Return a convenient representation of the given state.
    ///
    /// This is marked as inline because it doesn't seem to get inlined
    /// otherwise, which leads to a fairly significant performance loss (~25%).
    #[inline]
    fn state<'a>(&'a self, id: S) -> State<'a, S> {
        let mut pos = id.to_usize();
        let ntrans = NativeEndian::read_u16(&self.trans()[pos..]) as usize;
        pos += 2;
        let input_ranges = &self.trans()[pos..pos + (ntrans * 2)];
        pos += 2 * ntrans;
        let next = &self.trans()[pos..pos + (ntrans * size_of::<S>())];
        State { _state_id_repr: PhantomData, ntrans, input_ranges, next }
    }

    /// Return an iterator over all of the states in this DFA.
    ///
    /// The iterator returned yields tuples, where the first element is the
    /// state ID and the second element is the state itself.
    fn states<'a>(&'a self) -> StateIter<'a, T, S> {
        StateIter { dfa: self, id: dead_id() }
    }

    fn memory_usage(&self) -> usize {
        self.trans().len()
    }

    fn start_state(&self) -> S {
        self.start
    }

    fn is_match_state(&self, id: S) -> bool {
        self.is_possible_match_state(id) && !self.is_dead_state(id)
    }

    fn is_possible_match_state(&self, id: S) -> bool {
        id <= self.max_match
    }

    fn is_dead_state(&self, id: S) -> bool {
        id == dead_id()
    }

    fn trans(&self) -> &[u8] {
        self.trans.as_ref()
    }
}

impl<S: StateID> SparseDFARepr<Vec<u8>, S> {
    fn from_dfa_sized<T: AsRef<[S]>, A: StateID>(
        dfa: &DenseDFA<T, S>,
    ) -> Result<SparseDFARepr<Vec<u8>, A>> {
        let state_count = dfa.len();

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
        // space for the transitions. In this first pass, we also build up a
        // map from the state identifier index of the dense DFA to the state
        // identifier in this sparse DFA.
        //
        // In the second pass, we fill in the transitions based on the map
        // built in the first pass.

        let mut trans = vec![];
        let mut remap: Vec<A> = vec![dead_id(); state_count];
        for (old_id, state) in dfa.iter() {
            let pos = trans.len();
            remap[dfa.state_id_to_index(old_id)] = usize_to_state_id(pos)?;
            // zero-filled space for the transition count
            trans.push(0);
            trans.push(0);

            let mut trans_count = 0;
            for (b1, b2, next) in state.sparse_transitions() {
                if next != dead_id() {
                    trans_count += 1;
                    trans.push(b1);
                    trans.push(b2);
                }
            }
            // fill in the transition count
            NativeEndian::write_u16(&mut trans[pos..], trans_count);

            // zero-fill the actual transitions
            let zeros = trans_count as usize * size_of::<A>();
            trans.extend(iter::repeat(0).take(zeros));
        }

        let mut pos = 0;
        for (_, state) in dfa.iter() {
            let trans_count = NativeEndian::read_u16(&trans[pos..]) as usize;
            pos += 2 + (2 * trans_count);
            for (b1, b2, next) in state.sparse_transitions() {
                if next != dead_id() {
                    let next = remap[dfa.state_id_to_index(next)];
                    next.write_bytes(&mut trans[pos..]);
                    pos += size_of::<A>();
                }
            }
        }

        let start = remap[dfa.state_id_to_index(dfa.start())];
        let max_match = remap[dfa.state_id_to_index(dfa.max_match_state())];
        let byte_classes = dfa.byte_classes().clone();
        Ok(SparseDFARepr {
            start, state_count, max_match, byte_classes, trans,
        })
    }
}

impl<T: AsRef<[u8]>, S: StateID> fmt::Debug for SparseDFARepr<T, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fn state_status<T: AsRef<[u8]>, S: StateID>(
            dfa: &SparseDFARepr<T, S>,
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

        write!(f, "\n")?;
        for (id, state) in self.states() {
            let status = state_status(self, id);
            writeln!(f, "{}{:04}: {:?}", status, id.to_usize(), state)?;
        }
        Ok(())
    }
}

/// An iterator over all states in a sparse DFA.
///
/// This iterator yields tuples, where the first element is the state ID and
/// the second element is the state itself.
#[derive(Debug)]
struct StateIter<'a, T: AsRef<[u8]>, S: StateID = usize> {
    dfa: &'a SparseDFARepr<T, S>,
    id: S,
}

impl<'a, T: AsRef<[u8]>, S: StateID> Iterator for StateIter<'a, T, S> {
    type Item = (S, State<'a, S>);

    fn next(&mut self) -> Option<(S, State<'a, S>)> {
        if self.id.to_usize() >= self.dfa.trans().len() {
            return None;
        }
        let id = self.id;
        let state = self.dfa.state(id);
        self.id = S::from_usize(self.id.to_usize() + state.bytes());
        Some((id, state))
    }
}

/// A representation of a sparse DFA state that can be cheaply materialized
/// from a state identifier.
#[derive(Clone)]
struct State<'a, S: StateID = usize> {
    /// The state identifier representation used by the DFA from which this
    /// state was extracted. Since our transition table is compacted in a
    /// &[u8], we don't actually use the state ID type parameter explicitly
    /// anywhere, so we fake it. This prevents callers from using an incorrect
    /// state ID representation to read from this state.
    _state_id_repr: PhantomData<S>,
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
        for i in 0..self.ntrans {
            let (start, end) = self.range(i);
            if start <= input && input <= end {
                return self.next_at(i)
            }
            // We could bail early with an extra branch: if input < b1, then
            // we know we'll never find a matching transition. Interestingly,
            // this extra branch seems to not help performance, or will even
            // hurt it. It's likely very dependent on the DFA itself and what
            // is being searched.
        }
        dead_id()
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

    /// Return the total number of bytes that this state consumes in its
    /// encoded form.
    fn bytes(&self) -> usize {
        2 + (self.ntrans * 2) + (self.ntrans * size_of::<S>())
    }
}

impl<'a, S: StateID> fmt::Debug for State<'a, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut transitions = vec![];
        for i in 0..self.ntrans {
            let next = self.next_at(i);
            if next == dead_id() {
                continue;
            }

            let (start, end) = self.range(i);
            if start == end {
                transitions.push(
                    format!("{} => {}", escape(start), next.to_usize()),
                );
            } else {
                transitions.push(
                    format!(
                        "{}-{} => {}",
                        escape(start),
                        escape(end),
                        next.to_usize(),
                    ),
                );
            }
        }
        write!(f, "{}", transitions.join(", "))
    }
}

/// Return the given byte as its escaped string form.
fn escape(b: u8) -> String {
    use std::ascii;

    String::from_utf8(ascii::escape_default(b).collect::<Vec<_>>()).unwrap()
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
/// cast a `&[u8]` to a `&[(u8, u8)]`, but I don't believe this currently
/// guaranteed to be safe and is thus UB (since I don't think the in-memory
/// representation of `(u8, u8)` has been nailed down).
#[inline(always)]
#[allow(dead_code)]
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
