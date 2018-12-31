#![allow(warnings)]

use std::fmt;
use std::iter;
use std::mem::size_of;

use byteorder::{ByteOrder, NativeEndian};

use dense::{ALPHABET_LEN, DenseDFA};
use error::Result;
use state_id::{StateID, dead_id, usize_to_state_id};

#[derive(Clone)]
pub struct SparseDFA<S: StateID = usize> {
    kind: SparseDFAKind,
    start: S,
    state_count: usize,
    max_match: S,
    byte_classes: Vec<u8>,
    trans: Vec<u8>,
}

impl<S: StateID> SparseDFA<S> {
    pub(crate) fn from_dfa_sized<T: StateID>(
        dfa: &DenseDFA<S>,
    ) -> Result<SparseDFA<T>> {
        let kind =
            if dfa.kind().is_byte_class() {
                SparseDFAKind::ByteClass
            } else {
                SparseDFAKind::Basic
            };
        let state_count = dfa.len();
        let byte_classes = dfa.byte_classes().to_vec();

        let mut trans = vec![];
        let mut remap: Vec<T> = vec![dead_id(); state_count];
        for (old_id, state) in dfa.iter() {
            let pos = trans.len();
            remap[dfa.state_id_to_index(old_id)] = usize_to_state_id(pos)?;
            trans.push(0);
            trans.push(0);

            let mut trans_count = 0;
            // for (b, next) in state.iter() {
            for (b1, b2, next) in state.sparse_transitions() {
                if next != dead_id() {
                    trans_count += 1;
                    trans.push(b1);
                    trans.push(b2);
                }
            }
            NativeEndian::write_u16(&mut trans[pos..], trans_count);

            let zeros = trans_count as usize * size_of::<T>();
            trans.extend(iter::repeat(0).take(zeros));
        }

        let mut pos = 0;
        for (_, state) in dfa.iter() {
            let trans_count = NativeEndian::read_u16(&trans[pos..]) as usize;
            pos += 2 + (2 * trans_count);
            // for (b, next) in state.iter() {
            for (b1, b2, next) in state.sparse_transitions() {
                if next != dead_id() {
                    let next = remap[dfa.state_id_to_index(next)];
                    next.write_bytes(&mut trans[pos..]);
                    pos += size_of::<T>();
                }
            }
        }
        let start = remap[dfa.state_id_to_index(dfa.start())];
        let max_match = remap[dfa.state_id_to_index(dfa.max_match_state())];
        Ok(SparseDFA {
            kind, start, state_count, max_match, byte_classes, trans,
        })
    }

    pub fn is_match(&self, bytes: &[u8]) -> bool {
        self.is_match_inline(bytes)
    }

    pub fn memory_usage(&self) -> usize {
        self.byte_classes.len() + self.trans.len()
    }
}

impl<S: StateID> SparseDFA<S> {
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

    fn next_state(
        &self,
        current: S,
        input: u8,
    ) -> S {
        let pos = current.to_usize();
        let ntrans = NativeEndian::read_u16(&self.trans[pos..]) as usize;
        let inputs = &self.trans[pos+2..pos+2+(2*ntrans)];
        let trans = &self.trans[pos+2+(2*ntrans)..pos+2+(2*ntrans)+(size_of::<S>() * ntrans)];

        // This straight linear search was observed to be much better than
        // binary search on ASCII haystacks, likely because a binary search
        // visits the ASCII case last but a linear search sees it first. A
        // binary search does do a little better on non-ASCII haystacks, but
        // not by much. There might be a better trade off lurking here.
        for i in 0..ntrans {
            let (b1, b2) = (inputs[i * 2], inputs[i * 2 + 1]);
            if b1 <= input && input <= b2 {
                return S::read_bytes(&trans[i * size_of::<S>()..]);
            }
            // We could bail early with an extra branch: if input < b1, then
            // we know we'll never find a matching transition. Interestingly,
            // this extra branch seems to not help performance, or will even
            // hurt it. It's likely very dependent on the DFA itself and what
            // is being searched.
        }
        dead_id()
    }

    fn next_state_byte_class(
        &self,
        current: S,
        input: u8,
    ) -> S {
        let input = self.byte_classes[input as usize];
        self.next_state(current, input)
    }
}

impl<S: StateID> SparseDFA<S> {
    #[inline(always)]
    pub(crate) fn is_match_inline(&self, bytes: &[u8]) -> bool {
        match self.kind {
            SparseDFAKind::Basic => self.is_match_basic(bytes),
            SparseDFAKind::ByteClass => self.is_match_byte_class(bytes),
        }
    }

    fn is_match_basic(&self, bytes: &[u8]) -> bool {
        is_match!(self, bytes, next_state)
    }

    fn is_match_byte_class(&self, bytes: &[u8]) -> bool {
        is_match!(self, bytes, next_state_byte_class)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum SparseDFAKind {
    Basic,
    ByteClass,
}

impl SparseDFAKind {
    pub fn is_byte_class(&self) -> bool {
        match *self {
            SparseDFAKind::Basic => false,
            SparseDFAKind::ByteClass => true,
        }
    }

    pub(crate) fn to_byte(&self) -> u8 {
        match *self {
            SparseDFAKind::Basic => 0,
            SparseDFAKind::ByteClass => 1,
        }
    }

    pub(crate) fn from_byte(b: u8) -> SparseDFAKind {
        match b {
            0 => SparseDFAKind::Basic,
            1 => SparseDFAKind::ByteClass,
            _ => panic!("invalid sparse DFA kind: 0x{:X}", b),
        }
    }
}

impl<S: StateID> fmt::Debug for SparseDFA<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fn state_status<S: StateID>(
            dfa: &SparseDFA<S>,
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

        let mut state_offset_map = vec![dead_id(); self.state_count];
        let mut index = 0;
        let mut pos = 0;
        while pos < self.trans.len() {
            state_offset_map[index] = S::from_usize(pos);

            let ntrans = NativeEndian::read_u16(&self.trans[pos..]) as usize;
            pos += 2 + (2 * ntrans);
            for i in 0..ntrans {
                let next = S::read_bytes(&self.trans[pos..]).to_usize();
                pos += size_of::<S>();
            }
            index += 1;
        }

        let mut index = 0;
        let mut pos = 0;
        while pos < self.trans.len() {
            let ntrans = NativeEndian::read_u16(&self.trans[pos..]) as usize;
            pos += 2;
            let inputs = &self.trans[pos..pos+(2 * ntrans)];
            pos += 2 * ntrans;

            let mut transitions = vec![];
            for i in 0..ntrans {
                let next = S::read_bytes(&self.trans[pos..]).to_usize();
                pos += size_of::<S>();
                if next == dead_id() {
                    continue;
                }

                let (b1, b2) = (inputs[i * 2], inputs[i * 2 + 1]);
                if b1 == b2 {
                    transitions.push(format!("{} => {}", escape(b1), next));
                } else {
                    transitions.push(
                        format!("{}-{} => {}", escape(b1), escape(b2), next),
                    );
                }
            }

            let id = state_offset_map[index];
            let status = state_status(self, id);
            let state = transitions.join(", ");
            writeln!(f, "{}{:04}: {}", status, id.to_usize(), state)?;

            index += 1;
        }
        Ok(())
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
