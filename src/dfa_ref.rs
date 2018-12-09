use dfa::{ALPHABET_LEN, DEAD, DFAKind, StateID};

#[derive(Clone, Copy, Debug)]
pub struct DFARef<'a> {
    pub(crate) kind: DFAKind,
    pub(crate) start: StateID,
    pub(crate) state_count: usize,
    pub(crate) max_match: StateID,
    pub(crate) alphabet_len: usize,
    pub(crate) byte_classes: &'a [u8],
    pub(crate) trans: &'a [StateID],
}

impl<'a> DFARef<'a> {
    pub fn empty() -> DFARef<'static> {
        DFARef {
            kind: DFAKind::Basic,
            start: DEAD,
            state_count: 1,
            max_match: 1,
            alphabet_len: ALPHABET_LEN,
            byte_classes: &[],
            trans: &[DEAD; 256],
        }
    }

    pub fn is_match(&self, bytes: &[u8]) -> bool {
        self.is_match_inline(bytes)
    }

    pub fn find(&self, bytes: &[u8]) -> Option<usize> {
        self.find_inline(bytes)
    }
}

impl<'a> DFARef<'a> {
    pub fn kind(&self) -> &DFAKind {
        &self.kind
    }

    pub fn len(&self) -> usize {
        self.state_count
    }

    pub fn alphabet_len(&self) -> usize {
        self.alphabet_len
    }

    pub fn start(&self) -> StateID {
        self.start
    }

    pub fn is_match_state(&self, id: StateID) -> bool {
        self.is_possible_match_state(id) && !self.is_dead(id)
    }

    pub fn is_possible_match_state(&self, id: StateID) -> bool {
        id <= self.max_match
    }

    pub fn is_dead(&self, id: StateID) -> bool {
        id == DEAD
    }

    pub fn next_state(
        &self,
        current: StateID,
        input: u8,
    ) -> StateID {
        self.trans[current * ALPHABET_LEN + input as usize]
    }

    pub unsafe fn next_state_unchecked(
        &self,
        current: StateID,
        input: u8,
    ) -> StateID {
        *self.trans.get_unchecked(current * ALPHABET_LEN + input as usize)
    }

    pub fn next_state_premultiplied(
        &self,
        current: StateID,
        input: u8,
    ) -> StateID {
        self.trans[current + input as usize]
    }

    pub unsafe fn next_state_premultiplied_unchecked(
        &self,
        current: StateID,
        input: u8,
    ) -> StateID {
        *self.trans.get_unchecked(current + input as usize)
    }

    pub fn next_state_byte_class(
        &self,
        current: StateID,
        input: u8,
    ) -> StateID {
        let input = self.byte_classes[input as usize];
        self.trans[current * self.alphabet_len + input as usize]
    }

    pub unsafe fn next_state_byte_class_unchecked(
        &self,
        current: StateID,
        input: u8,
    ) -> StateID {
        let input = *self.byte_classes.get_unchecked(input as usize);
        *self.trans.get_unchecked(current * self.alphabet_len + input as usize)
    }

    pub fn next_state_premultiplied_byte_class(
        &self,
        current: StateID,
        input: u8,
    ) -> StateID {
        let input = self.byte_classes[input as usize];
        self.trans[current + input as usize]
    }

    pub unsafe fn next_state_premultiplied_byte_class_unchecked(
        &self,
        current: StateID,
        input: u8,
    ) -> StateID {
        let input = *self.byte_classes.get_unchecked(input as usize);
        *self.trans.get_unchecked(current + input as usize)
    }
}

impl<'a> DFARef<'a> {
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
            if state == DEAD {
                return None;
            } else if state <= self.max_match {
                Some(0)
            } else {
                None
            };
        for (i, &b) in bytes.iter().enumerate() {
            state = self.trans[state * ALPHABET_LEN + b as usize];
            if state <= self.max_match {
                if state == DEAD {
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
            if state == DEAD {
                return None;
            } else if state <= self.max_match {
                Some(0)
            } else {
                None
            };
        for (i, &b) in bytes.iter().enumerate() {
            state = self.trans[state + b as usize];
            if state <= self.max_match {
                if state == DEAD {
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
            if state == DEAD {
                return None;
            } else if state <= self.max_match {
                Some(0)
            } else {
                None
            };

        let alphabet_len = self.alphabet_len();
        for (i, &b) in bytes.iter().enumerate() {
            let b = self.byte_classes[b as usize];
            state = self.trans[state * alphabet_len + b as usize];
            if state <= self.max_match {
                if state == DEAD {
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
            if state == DEAD {
                return None;
            } else if state <= self.max_match {
                Some(0)
            } else {
                None
            };
        for (i, &b) in bytes.iter().enumerate() {
            let b = self.byte_classes[b as usize];
            state = self.trans[state + b as usize];
            if state <= self.max_match {
                if state == DEAD {
                    return last_match;
                }
                last_match = Some(i + 1);
            }
        }
        last_match
    }
}
