use std::fmt;
use std::iter;
use std::mem;
use std::slice;

use error::{Error, Result};
use determinize::Determinizer;
use dfa_ref::DFARef;
use minimize::Minimizer;
use nfa::NFA;
use state_id::{StateID, dead_id, next_state_id, premultiply_overflow_error};

pub const ALPHABET_LEN: usize = 256;

pub struct DFA<S = usize> {
    /// The type of DFA. This enum controls how the state transition table
    /// is interpreted. It is never correct to read the transition table
    /// without knowing the DFA's kind.
    kind: DFAKind,
    /// The initial start state ID.
    start: S,
    /// The total number of states in this DFA. Note that a DFA always has at
    /// least one state---the DEAD state---even the empty DFA. In particular,
    /// the DEAD state always has ID 0 and is correspondingly always the first
    /// state. The DEAD state is never a match state.
    state_count: usize,
    /// States in a DFA have a *partial* ordering such that a match state
    /// always precedes any non-match state (except for the special DEAD
    /// state).
    ///
    /// `max_match` corresponds to the last state that is a match state. This
    /// encoding has two critical benefits. Firstly, we are not required to
    /// store any additional per-state information about whether it is a match
    /// state or not. Secondly, when searching with the DFA, we can do a single
    /// comparison with `max_match` for each byte instead of two comparisons
    /// for each byte (one testing whether it is a match and the other testing
    /// whether we've reached a DEAD state). Namely, to determine the status
    /// of the next state, we can do this:
    ///
    ///   next_state = transition[cur_state * ALPHABET_LEN + cur_byte]
    ///   if next_state <= max_match:
    ///       // next_state is either DEAD (no-match) or a match
    ///       return next_state != DEAD
    max_match: S,
    /// The total number of bytes in this DFA's alphabet. This is always
    /// equivalent to 256, unless the DFA was built with byte classes, in which
    /// case, this is equal to the number of byte classes.
    alphabet_len: usize,
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
    byte_classes: Vec<u8>,
    /// A contiguous region of memory representing the transition table in
    /// row-major order. The representation is dense. That is, every state has
    /// precisely the same number of transitions. The maximum number of
    /// transitions is 256. If a DFA has been instructed to use byte classes,
    /// then the number of transitions can be much less.
    trans: Vec<S>,
}

impl<S: StateID> DFA<S> {
    pub fn empty() -> DFA<S> {
        DFA::empty_with_byte_classes(vec![])
    }

    pub(crate) fn empty_with_byte_classes(byte_classes: Vec<u8>) -> DFA<S> {
        assert!(byte_classes.is_empty() || byte_classes.len() == 256);

        let (kind, alphabet_len) =
            if byte_classes.is_empty() {
                (DFAKind::Basic, ALPHABET_LEN)
            } else {
                (DFAKind::ByteClass, byte_classes[255] as usize + 1)
            };
        let mut dfa = DFA {
            kind: kind,
            start: dead_id(),
            state_count: 0,
            max_match: S::from_usize(1),
            alphabet_len: alphabet_len,
            byte_classes: byte_classes,
            trans: vec![],
        };
        // Every state ID repr must be able to fit at least one state.
        dfa.add_empty_state().unwrap();
        dfa
    }

    pub fn is_match(&self, bytes: &[u8]) -> bool {
        self.as_dfa_ref().is_match_inline(bytes)
    }

    pub fn find(&self, bytes: &[u8]) -> Option<usize> {
        self.as_dfa_ref().find_inline(bytes)
    }

    pub fn as_dfa_ref(&self) -> DFARef<S> {
        DFARef {
            kind: self.kind,
            start: self.start,
            state_count: self.state_count,
            max_match: self.max_match,
            alphabet_len: self.alphabet_len,
            byte_classes: &self.byte_classes,
            trans: &self.trans,
        }
    }

    /// Returns the memory usage, in bytes, of this DFA.
    ///
    /// The memory usage is computed based on the number of bytes used to
    /// represent this DFA's transition table. This corresponds to heap memory
    /// usage.
    ///
    /// This does **not** include the stack size used up by this DFA. To
    /// compute that, used `std::mem::size_of::<DFA>()`.
    pub fn memory_usage(&self) -> usize {
        self.as_dfa_ref().memory_usage()
    }
}

impl<S: StateID> DFA<S> {
    pub fn to_u8(&self) -> Result<DFA<u8>> {
        self.to_sized()
    }

    pub fn to_u16(&self) -> Result<DFA<u16>> {
        self.to_sized()
    }

    pub fn to_u32(&self) -> Result<DFA<u32>> {
        self.to_sized()
    }

    pub fn to_u64(&self) -> Result<DFA<u64>> {
        self.to_sized()
    }

    pub fn to_sized<T: StateID>(&self) -> Result<DFA<T>> {
        // Check that this DFA can fit into T's representation.
        let mut last_state_id = self.state_count - 1;
        if self.kind.is_premultiplied() {
            last_state_id *= self.alphabet_len();
        }
        if last_state_id > T::max_id() {
            return Err(Error::state_id_overflow(T::max_id()));
        }

        // We're off to the races. The new DFA is the same as the old one,
        // but its transition table is truncated.
        let mut new = DFA {
            kind: self.kind,
            start: T::from_usize(self.start.to_usize()),
            state_count: self.state_count,
            max_match: T::from_usize(self.max_match.to_usize()),
            alphabet_len: self.alphabet_len,
            byte_classes: self.byte_classes.clone(),
            trans: vec![dead_id::<T>(); self.trans.len()],
        };
        for (i, id) in new.trans.iter_mut().enumerate() {
            *id = T::from_usize(self.trans[i].to_usize());
        }
        Ok(new)
    }
}

impl<S: StateID> DFA<S> {
    pub(crate) fn from_nfa(nfa: &NFA) -> Result<DFA<S>> {
        Determinizer::new(nfa).build()
    }

    pub(crate) fn from_nfa_with_byte_classes(nfa: &NFA) -> Result<DFA<S>> {
        Determinizer::new(nfa).with_byte_classes().build()
    }

    pub(crate) fn state_id_to_offset(&self, id: S) -> usize {
        if self.kind.is_premultiplied() {
            id.to_usize()
        } else {
            id.to_usize() * self.alphabet_len()
        }
    }

    pub(crate) fn byte_to_class(&self, b: u8) -> u8 {
        if self.kind.is_byte_class() {
            self.byte_classes[b as usize]
        } else {
            b
        }
    }

    pub(crate) fn equiv_bytes(&self) -> Vec<u8> {
        if !self.kind.is_byte_class() {
            return (0..ALPHABET_LEN).map(|b| b as u8).collect();
        }

        let mut equivs = vec![];
        let mut last_equiv = None;
        for b in 0usize..256 {
            let equiv = self.byte_classes[b];
            if last_equiv != Some(equiv) {
                equivs.push(b as u8);
                last_equiv = Some(equiv);
            }
        }
        equivs
    }

    pub(crate) fn len(&self) -> usize {
        self.state_count
    }

    pub(crate) fn alphabet_len(&self) -> usize {
        self.alphabet_len
    }

    pub(crate) fn start(&self) -> S {
        self.start
    }

    pub(crate) fn kind(&self) -> &DFAKind {
        &self.kind
    }

    pub(crate) fn is_match_state(&self, id: S) -> bool {
        id <= self.max_match && id != dead_id()
    }

    pub(crate) fn max_match_state(&self) -> S {
        self.max_match
    }

    pub(crate) fn set_start_state(&mut self, start: S) {
        assert!(start.to_usize() < self.len());
        self.start = start;
    }

    pub(crate) fn set_transition(
        &mut self,
        from: S,
        input: u8,
        to: S,
    ) {
        let input = self.byte_to_class(input);
        let i = self.state_id_to_offset(from) + input as usize;
        self.trans[i] = to;
    }

    pub(crate) fn add_empty_state(&mut self) -> Result<S> {
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

    pub(crate) fn get_state(&self, id: S) -> State<S> {
        let i = self.state_id_to_offset(id);
        State {
            transitions: &self.trans[i..i+self.alphabet_len()],
        }
    }

    pub(crate) fn get_state_mut(&mut self, id: S) -> StateMut<S> {
        let i = self.state_id_to_offset(id);
        let alphabet_len = self.alphabet_len();
        StateMut {
            transitions: &mut self.trans[i..i+alphabet_len],
        }
    }

    pub(crate) fn set_max_match_state(&mut self, id: S) {
        self.max_match = id;
    }

    pub(crate) fn iter(&self) -> StateIter<S> {
        let it = self.trans.chunks(self.alphabet_len());
        StateIter { dfa: self, it: it.enumerate() }
    }

    pub(crate) fn swap_states(&mut self, id1: S, id2: S) {
        let o1 = self.state_id_to_offset(id1);
        let o2 = self.state_id_to_offset(id2);
        for b in 0..self.alphabet_len() {
            self.trans.swap(o1 + b, o2 + b);
        }
    }

    pub(crate) fn truncate_states(&mut self, count: usize) {
        let alphabet_len = self.alphabet_len();
        self.trans.truncate(count * alphabet_len);
        self.state_count = count;
    }

    pub(crate) fn shuffle_match_states(&mut self, is_match: &[bool]) {
        assert!(
            !self.kind.is_premultiplied(),
            "cannot finish construction of premultiplied DFA"
        );

        if self.len() <= 2 {
            return;
        }

        let mut first_non_match = 1;
        while first_non_match < self.len() && is_match[first_non_match] {
            first_non_match += 1;
        }

        let mut swaps: Vec<S> = vec![dead_id(); self.len()];
        let mut cur = self.len() - 1;
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
        for id in (0..self.len()).map(S::from_usize) {
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

    pub(crate) fn minimize(&mut self) {
        assert!(!self.kind.is_premultiplied());
        Minimizer::new(self).run();
    }

    pub(crate) fn premultiply(&mut self) -> Result<()> {
        if self.kind.is_premultiplied() || self.len() == 0 {
            return Ok(());
        }

        let alpha_len = self.alphabet_len();
        premultiply_overflow_error(S::from_usize(self.len() - 1), alpha_len)?;

        for id in (0..self.len()).map(S::from_usize) {
            for (_, next) in self.get_state_mut(id).iter_mut() {
                *next = S::from_usize(next.to_usize() * alpha_len);
            }
        }
        self.kind = self.kind.premultiplied();
        self.start = S::from_usize(self.start.to_usize() * alpha_len);
        self.max_match = S::from_usize(self.max_match.to_usize() * alpha_len);
        Ok(())
    }
}

#[derive(Debug)]
pub struct StateIter<'a, S: StateID> {
    dfa: &'a DFA<S>,
    it: iter::Enumerate<slice::Chunks<'a, S>>,
}

impl<'a, S: StateID> Iterator for StateIter<'a, S> {
    type Item = (S, State<'a, S>);

    fn next(&mut self) -> Option<(S, State<'a, S>)> {
        self.it.next().map(|(id, chunk)| {
            let state = State { transitions: chunk };
            let id =
                if self.dfa.kind().is_premultiplied() {
                    id * self.dfa.alphabet_len()
                } else {
                    id
                };
            (S::from_usize(id), state)
        })
    }
}

pub struct State<'a, S> {
    transitions: &'a [S],
}

impl<'a, S: StateID> State<'a, S> {
    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    pub fn get(&self, b: u8) -> S {
        self.transitions[b as usize]
    }

    pub fn iter(&self) -> StateTransitionIter<S> {
        StateTransitionIter { it: self.transitions.iter().enumerate() }
    }

    fn sparse_transitions(&self) -> Vec<(u8, u8, S)> {
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

#[derive(Debug)]
pub struct StateTransitionIter<'a, S> {
    it: iter::Enumerate<slice::Iter<'a, S>>,
}

impl<'a, S: StateID> Iterator for StateTransitionIter<'a, S> {
    type Item = (u8, S);

    fn next(&mut self) -> Option<(u8, S)> {
        self.it.next().map(|(i, &id)| (i as u8, id))
    }
}

pub struct StateMut<'a, S> {
    transitions: &'a mut [S],
}

impl<'a, S: StateID> StateMut<'a, S> {
    pub fn iter_mut(&mut self) -> StateTransitionIterMut<S> {
        StateTransitionIterMut { it: self.transitions.iter_mut().enumerate() }
    }
}

#[derive(Debug)]
pub struct StateTransitionIterMut<'a, S> {
    it: iter::Enumerate<slice::IterMut<'a, S>>,
}

impl<'a, S: StateID> Iterator for StateTransitionIterMut<'a, S> {
    type Item = (u8, &'a mut S);

    fn next(&mut self) -> Option<(u8, &'a mut S)> {
        self.it.next().map(|(i, id)| (i as u8, id))
    }
}

#[derive(Clone, Copy, Debug)]
pub enum DFAKind {
    Basic,
    Premultiplied,
    ByteClass,
    PremultipliedByteClass,
}

impl DFAKind {
    pub fn is_byte_class(&self) -> bool {
        match *self {
            DFAKind::Basic | DFAKind::Premultiplied => false,
            DFAKind::ByteClass | DFAKind::PremultipliedByteClass => true,
        }
    }

    pub fn is_premultiplied(&self) -> bool {
        match *self {
            DFAKind::Basic | DFAKind::ByteClass => false,
            DFAKind::Premultiplied | DFAKind::PremultipliedByteClass => true,
        }
    }

    fn premultiplied(self) -> DFAKind {
        match self {
            DFAKind::Basic => DFAKind::Premultiplied,
            DFAKind::ByteClass => DFAKind::PremultipliedByteClass,
            DFAKind::Premultiplied | DFAKind::PremultipliedByteClass => {
                panic!("DFA already has pre-multiplied state IDs")
            }
        }
    }
}

impl<S: StateID> fmt::Debug for DFA<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fn state_status<S: StateID>(
            dfa: &DFA<S>,
            id: S,
            state: &State<S>,
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

        for (id, state) in self.iter() {
            let status = state_status(self, id, &state);
            writeln!(f, "{}{:04}: {:?}", status, id.to_usize(), state)?;
        }
        Ok(())
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

/// Return the given byte as its escaped string form.
fn escape(b: u8) -> String {
    use std::ascii;

    String::from_utf8(ascii::escape_default(b).collect::<Vec<_>>()).unwrap()
}

#[cfg(test)]
mod tests {
    use builder::DFABuilder;
    use super::*;

    #[test]
    fn errors_when_converting_to_smaller_dfa() {
        let pattern = r"\w";
        let dfa = DFABuilder::new()
            .byte_classes(false)
            .anchored(true)
            .premultiply(false)
            .build_with_size::<u16>(r"\w")
            .unwrap();
        assert!(dfa.to_u8().is_err());
    }

    #[test]
    fn errors_when_determinization_would_overflow() {
        let pattern = r"\w";

        let mut builder = DFABuilder::new();
        builder.byte_classes(false).anchored(true).premultiply(false);
        // using u16 is fine
        assert!(builder.build_with_size::<u16>(pattern).is_ok());
        // // ... but u8 results in overflow (because there are >256 states)
        assert!(builder.build_with_size::<u8>(pattern).is_err());
    }

    #[test]
    fn errors_when_premultiply_would_overflow() {
        let pattern = r"[a-z]";

        let mut builder = DFABuilder::new();
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
        // println!("PATTERN: {:?}", pattern);
        // println!("NFA:");
        // for (i, state) in nfa.states.borrow().iter().enumerate() {
            // println!("{:03X}: {:X?}", i, state);
        // }

        println!("{}", "~".repeat(79));

        println!("DFA:");
        print!("{:?}", dfa);
        println!("{}", "~".repeat(79));

        println!("Minimal DFA:");
        print!("{:?}", mdfa);
        println!("{}", "~".repeat(79));

        println!("{}", "#".repeat(100));
    }

    fn print_automata_counts(pattern: &str) {
        let (nfa, dfa, mdfa) = build_automata(pattern);
        println!("nfa # states: {:?}", nfa.len());
        println!("dfa # states: {:?}", dfa.len());
        println!("minimal dfa # states: {:?}", mdfa.len());
    }

    fn build_automata(pattern: &str) -> (NFA, DFA, DFA) {
        let mut builder = DFABuilder::new();
        builder.anchored(true).allow_invalid_utf8(true);
        let nfa = builder.build_nfa(pattern).unwrap();
        let dfa = builder.build(pattern).unwrap();
        let min = builder.minimize(true).build(pattern).unwrap();
        (nfa, dfa, min)
    }

    #[test]
    fn scratch() {
        // print_automata(grapheme_pattern());
        // let (nfa, mut dfa) = build_automata(grapheme_pattern());
        // let (nfa, dfa) = build_automata(r"a");
        // println!("# dfa states: {}", dfa.states.len());
        // println!("# dfa transitions: {}", 256 * dfa.states.len());
        // Minimizer::new(&mut dfa).run();
        // println!("# minimal dfa states: {}", dfa.states.len());
        // println!("# minimal dfa transitions: {}", 256 * dfa.states.len());
        // print_automata(r"\p{any}");
        // print_automata(r"[\u007F-\u0080]");

        // println!("building...");
        // let dfa = grapheme_dfa();
        // let dfa = build_automata_min(r"a|\p{gcb=RI}\p{gcb=RI}|\p{gcb=RI}");
        // println!("searching...");
        // let string = "\u{1f1e6}\u{1f1e6}";
        // let bytes = string.as_bytes();
        // println!("{:?}", dfa.find(bytes));

        // print_automata("a|zz|z");
        // let dfa = build_automata_min(r"a|zz|z");
        // println!("searching...");
        // let string = "zz";
        // let bytes = string.as_bytes();
        // println!("{:?}", dfa.find(bytes));

        // print_automata(r"[01]*1[01]{5}");
        // print_automata(r"X(.?){0,8}Y");
        // print_automata_counts(r"\p{alphabetic}");
        // print_automata(r"a*b+|cdefg");
        // print_automata(r"(..)*(...)*");
        print_automata(r"(?-u:\w)");
        print_automata_counts(r"(?-u:\w)");

        // let data = ::std::fs::read_to_string("/usr/share/dict/words").unwrap();
        // let mut words: Vec<&str> = data.lines().collect();
        // println!("{} words", words.len());
        // words.sort_by(|w1, w2| w1.len().cmp(&w2.len()).reverse());
        // let pattern = words.join("|");
        // print_automata_counts(&pattern);
        // print_automata(&pattern);
    }

    #[test]
    fn grapheme() {
        // let (nfa, dfa, mdfa) = build_automata(grapheme_pattern());
        let (nfa, dfa, mdfa) = build_automata(r"\w");
        let dfa = dfa.to_u16().unwrap();
        let mdfa = mdfa.to_u16().unwrap();
        println!("nfa states: {:?}", nfa.len());
        println!("dfa states: {:?} ({} bytes)", dfa.len(), dfa.memory_usage());
        println!("min dfa states: {:?} ({} bytes)", mdfa.len(), mdfa.memory_usage());
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
