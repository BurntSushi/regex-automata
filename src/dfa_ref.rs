use std::mem::size_of;

use dfa::{ALPHABET_LEN, DFAKind, DFA};
use state_id::{StateID, dead_id};

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
    pub fn is_match(&self, bytes: &[u8]) -> bool {
        self.is_match_inline(bytes)
    }

    pub fn find(&self, bytes: &[u8]) -> Option<usize> {
        self.find_inline(bytes)
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
        self.byte_classes.len() + (self.trans.len() * size_of::<S>())
    }
}

impl<'a, S: StateID> DFARef<'a, S> {
    fn kind(&self) -> &DFAKind {
        &self.kind
    }

    fn len(&self) -> usize {
        self.state_count
    }

    fn alphabet_len(&self) -> usize {
        self.alphabet_len
    }

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
        let mut state = self.start;
        if self.is_possible_match_state(state) {
            return self.is_match_state(state);
        }
        for &b in bytes.iter() {
            state = unsafe { self.next_state_unchecked(state, b) };
            if self.is_possible_match_state(state) {
                return self.is_match_state(state);
            }
        }
        false
    }

    fn is_match_premultiplied(&self, bytes: &[u8]) -> bool {
        let mut state = self.start();
        if self.is_possible_match_state(state) {
            return self.is_match_state(state);
        }
        for &b in bytes.iter() {
            state = unsafe {
                self.next_state_premultiplied_unchecked(state, b)
            };
            if self.is_possible_match_state(state) {
                return self.is_match_state(state);
            }
        }
        false
    }

    fn is_match_byte_class(&self, bytes: &[u8]) -> bool {
        let mut state = self.start;
        if self.is_possible_match_state(state) {
            return self.is_match_state(state);
        }
        for &b in bytes.iter() {
            state = unsafe {
                self.next_state_byte_class_unchecked(state, b)
            };
            if self.is_possible_match_state(state) {
                return self.is_match_state(state);
            }
        }
        false
    }

    fn is_match_premultiplied_byte_class(&self, bytes: &[u8]) -> bool {
        let mut state = self.start;
        if self.is_possible_match_state(state) {
            return self.is_match_state(state);
        }
        for &b in bytes.iter() {
            state = unsafe {
                self.next_state_premultiplied_byte_class_unchecked(state, b)
            };
            if self.is_possible_match_state(state) {
                return self.is_match_state(state);
            }
        }
        false
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
        let mut state = self.start;
        let mut last_match =
            if state == dead_id() {
                return None;
            } else if state <= self.max_match {
                Some(0)
            } else {
                None
            };
        for (i, &b) in bytes.iter().enumerate() {
            state = self.trans[state.to_usize() * ALPHABET_LEN + b as usize];
            if state <= self.max_match {
                if state == dead_id() {
                    return last_match;
                }
                last_match = Some(i + 1);
            }
        }
        last_match
    }

    fn find_premultiplied(&self, bytes: &[u8]) -> Option<usize> {
        let mut state = self.start;
        let mut last_match =
            if state == dead_id() {
                return None;
            } else if state <= self.max_match {
                Some(0)
            } else {
                None
            };
        for (i, &b) in bytes.iter().enumerate() {
            state = self.trans[state.to_usize() + b as usize];
            if state <= self.max_match {
                if state == dead_id() {
                    return last_match;
                }
                last_match = Some(i + 1);
            }
        }
        last_match
    }

    fn find_byte_class(&self, bytes: &[u8]) -> Option<usize> {
        let mut state = self.start;
        let mut last_match =
            if state == dead_id() {
                return None;
            } else if state <= self.max_match {
                Some(0)
            } else {
                None
            };

        let alphabet_len = self.alphabet_len();
        for (i, &b) in bytes.iter().enumerate() {
            let b = self.byte_classes[b as usize];
            state = self.trans[state.to_usize() * alphabet_len + b as usize];
            if state <= self.max_match {
                if state == dead_id() {
                    return last_match;
                }
                last_match = Some(i + 1);
            }
        }
        last_match
    }

    fn find_premultiplied_byte_class(&self, bytes: &[u8]) -> Option<usize> {
        let mut state = self.start;
        let mut last_match =
            if state == dead_id() {
                return None;
            } else if state <= self.max_match {
                Some(0)
            } else {
                None
            };
        for (i, &b) in bytes.iter().enumerate() {
            let b = self.byte_classes[b as usize];
            state = self.trans[state.to_usize() + b as usize];
            if state <= self.max_match {
                if state == dead_id() {
                    return last_match;
                }
                last_match = Some(i + 1);
            }
        }
        last_match
    }
}
