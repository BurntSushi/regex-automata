use std::fmt;
use std::iter;
use std::mem;
use std::slice;

use determinize::Determinizer;
use minimize::Minimizer;
use nfa::NFA;

pub const DEAD: StateID = 0;
pub const ALPHABET_LEN: usize = 256;

pub type StateID = usize;

pub struct DFA {
    /// The type of DFA. This enum controls how the state transition table
    /// is interpreted. It is never correct to read the transition table
    /// without knowing the DFA's kind.
    kind: DFAKind,
    /// The initial start state ID.
    start: StateID,
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
    max_match: StateID,
    /// A set of equivalence classes, where a single equivalence class
    /// represents a set of bytes that never discriminate between a match
    /// and a non-match in the DFA. Each equivalence class corresponds to
    /// a single letter in this DFA's alphabet, where the maximum number of
    /// letters is 256 (each possible value of a byte). Consequently, the
    /// number of equivalence classes corresponds to the number of transitions
    /// for each DFA state.
    ///
    /// The only time the number of equivalence classes is fewer than 256 is
    /// if the DFA's kind uses byte classes.
    byte_classes: Vec<u8>,
    /// A contiguous region of memory representing the transition table in
    /// row-major order. The representation is dense. That is, every state has
    /// precisely the same number of transitions. The maximum number of
    /// transitions is 256. If a DFA has been instructed to use byte classes,
    /// then the number of transitions can be much less.
    trans: Vec<StateID>,
}

impl DFA {
    pub fn empty() -> DFA {
        DFA::empty_with_byte_classes(vec![])
    }

    pub(crate) fn empty_with_byte_classes(byte_classes: Vec<u8>) -> DFA {
        assert!(byte_classes.is_empty() || byte_classes.len() == 256);

        let kind =
            if byte_classes.is_empty() {
                DFAKind::Basic
            } else {
                DFAKind::ByteClass
            };
        let mut dfa = DFA {
            kind: kind,
            trans: vec![],
            state_count: 0,
            max_match: 1,
            byte_classes: byte_classes,
            start: DEAD,
        };
        dfa.add_empty_state();
        dfa
    }

    pub fn len(&self) -> usize {
        self.state_count
    }

    pub fn alphabet_len(&self) -> usize {
        if self.kind.is_byte_class() {
            self.byte_classes[255] as usize + 1
        } else {
            ALPHABET_LEN
        }
    }

    pub fn is_match_state(&self, id: StateID) -> bool {
        id != DEAD && id <= self.max_match
    }

    pub fn max_match_state(&self) -> StateID {
        self.max_match
    }

    pub fn start(&self) -> StateID {
        self.start
    }

    pub fn kind(&self) -> &DFAKind {
        &self.kind
    }

    pub fn is_match(&self, bytes: &[u8]) -> bool {
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
        if state <= self.max_match {
            return state != DEAD;
        }
        for &b in bytes.iter() {
            state = unsafe {
                *self.trans.get_unchecked(state * ALPHABET_LEN + b as usize)
            };
            if state <= self.max_match {
                return state != DEAD;
            }
        }
        false
    }

    fn is_match_premultiplied(&self, bytes: &[u8]) -> bool {
        let mut state = self.start;
        if state <= self.max_match {
            return state != DEAD;
        }
        for &b in bytes.iter() {
            state = unsafe {
                *self.trans.get_unchecked(state + b as usize)
            };
            if state <= self.max_match {
                return state != DEAD;
            }
        }
        false
    }

    fn is_match_byte_class(&self, bytes: &[u8]) -> bool {
        let mut state = self.start;
        if state <= self.max_match {
            return state != DEAD;
        }

        let alphabet_len = self.alphabet_len();
        for &b in bytes.iter() {
            state = unsafe {
                let b = *self.byte_classes.get_unchecked(b as usize);
                *self.trans.get_unchecked(state * alphabet_len + b as usize)
            };
            if state <= self.max_match {
                return state != DEAD;
            }
        }
        false
    }

    fn is_match_premultiplied_byte_class(&self, bytes: &[u8]) -> bool {
        let mut state = self.start;
        if state <= self.max_match {
            return state != DEAD;
        }
        for &b in bytes.iter() {
            state = unsafe {
                let b = *self.byte_classes.get_unchecked(b as usize);
                *self.trans.get_unchecked(state + b as usize)
            };
            if state <= self.max_match {
                return state != DEAD;
            }
        }
        false
    }

    pub fn find(&self, bytes: &[u8]) -> Option<usize> {
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

impl DFA {
    pub(crate) fn from_nfa(nfa: &NFA) -> DFA {
        Determinizer::new(nfa).build()
    }

    pub(crate) fn from_nfa_with_byte_classes(nfa: &NFA) -> DFA {
        Determinizer::new(nfa).with_byte_classes().build()
    }

    pub(crate) fn state_id_to_offset(&self, id: StateID) -> usize {
        if self.kind.is_premultiplied() {
            id
        } else {
            id * self.alphabet_len()
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

    pub(crate) fn set_start_state(&mut self, start: StateID) {
        assert!(start < self.len());
        self.start = start;
    }

    pub(crate) fn set_transition(
        &mut self,
        from: StateID,
        input: u8,
        to: StateID,
    ) {
        let input = self.byte_to_class(input);
        let i = self.state_id_to_offset(from) + input as usize;
        self.trans[i] = to;
    }

    pub(crate) fn add_empty_state(&mut self) -> StateID {
        let id = self.state_count;
        let alphabet_len = self.alphabet_len();
        self.trans.extend(iter::repeat(DEAD).take(alphabet_len));
        self.state_count += 1;
        id
    }

    pub(crate) fn get_state(&self, id: StateID) -> State {
        let i = self.state_id_to_offset(id);
        State {
            transitions: &self.trans[i..i+self.alphabet_len()],
        }
    }

    pub(crate) fn get_state_mut(&mut self, id: StateID) -> StateMut {
        let i = self.state_id_to_offset(id);
        let alphabet_len = self.alphabet_len();
        StateMut {
            transitions: &mut self.trans[i..i+alphabet_len],
        }
    }

    pub(crate) fn set_max_match_state(&mut self, id: StateID) {
        self.max_match = id;
    }

    pub(crate) fn iter(&self) -> StateIter {
        let it = self.trans.chunks(self.alphabet_len());
        StateIter { dfa: self, it: it.enumerate() }
    }

    pub(crate) fn swap_states(&mut self, id1: StateID, id2: StateID) {
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

        let mut swaps = vec![DEAD; self.len()];
        let mut cur = self.len() - 1;
        while cur > first_non_match {
            if is_match[cur] {
                self.swap_states(cur, first_non_match);
                swaps[cur] = first_non_match;
                swaps[first_non_match] = cur;

                first_non_match += 1;
                while first_non_match < cur && is_match[first_non_match] {
                    first_non_match += 1;
                }
            }
            cur -= 1;
        }
        for id in 0..self.len() {
            for (_, next) in self.get_state_mut(id).iter_mut() {
                if swaps[*next] != DEAD {
                    *next = swaps[*next];
                }
            }
        }
        if swaps[self.start] != DEAD {
            self.start = swaps[self.start];
        }
        self.max_match = first_non_match - 1;
    }

    pub(crate) fn minimize(&mut self) {
        assert!(!self.kind.is_premultiplied());
        Minimizer::new(self).run();
    }

    pub(crate) fn premultiply(&mut self) {
        if self.kind.is_premultiplied() {
            return;
        }

        let alphabet_len = self.alphabet_len();
        for id in 0..self.len() {
            for (_, next) in self.get_state_mut(id).iter_mut() {
                *next = *next * alphabet_len;
            }
        }
        self.kind = self.kind.premultiplied();
        self.start *= alphabet_len;
        self.max_match *= alphabet_len;
    }
}

#[derive(Debug)]
pub struct StateIter<'a> {
    dfa: &'a DFA,
    it: iter::Enumerate<slice::Chunks<'a, StateID>>,
}

impl<'a> Iterator for StateIter<'a> {
    type Item = (StateID, State<'a>);

    fn next(&mut self) -> Option<(StateID, State<'a>)> {
        self.it.next().map(|(id, chunk)| {
            let state = State { transitions: chunk };
            if self.dfa.kind().is_premultiplied() {
                (id * self.dfa.alphabet_len(), state)
            } else {
                (id, state)
            }
        })
    }
}

pub struct State<'a> {
    transitions: &'a [StateID],
}

impl<'a> State<'a> {
    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    pub fn get(&self, b: u8) -> StateID {
        self.transitions[b as usize]
    }

    pub fn iter(&self) -> StateTransitionIter {
        StateTransitionIter { it: self.transitions.iter().enumerate() }
    }

    fn sparse_transitions(&self) -> Vec<(u8, u8, StateID)> {
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
pub struct StateTransitionIter<'a> {
    it: iter::Enumerate<slice::Iter<'a, StateID>>,
}

impl<'a> Iterator for StateTransitionIter<'a> {
    type Item = (u8, StateID);

    fn next(&mut self) -> Option<(u8, StateID)> {
        self.it.next().map(|(i, &id)| (i as u8, id))
    }
}

pub struct StateMut<'a> {
    transitions: &'a mut [StateID],
}

impl<'a> StateMut<'a> {
    pub fn iter_mut(&mut self) -> StateTransitionIterMut {
        StateTransitionIterMut { it: self.transitions.iter_mut().enumerate() }
    }
}

#[derive(Debug)]
pub struct StateTransitionIterMut<'a> {
    it: iter::Enumerate<slice::IterMut<'a, StateID>>,
}

impl<'a> Iterator for StateTransitionIterMut<'a> {
    type Item = (u8, &'a mut StateID);

    fn next(&mut self) -> Option<(u8, &'a mut StateID)> {
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

impl fmt::Debug for DFA {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fn state_status(dfa: &DFA, id: StateID, state: &State) -> String {
            let mut status = vec![b' ', b' '];
            if id == DEAD {
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
            writeln!(f, "{}{:04}: {:?}", status, id, state)?;
        }
        Ok(())
    }
}

impl<'a> fmt::Debug for State<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut transitions = vec![];
        for (start, end, next_id) in self.sparse_transitions() {
            if next_id == DEAD {
                continue;
            }
            let line =
                if start == end {
                    format!("{} => {}", escape(start), next_id)
                } else {
                    format!(
                        "{}-{} => {}",
                        escape(start), escape(end), next_id,
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
        print_automata(r"[ab]+");

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
        let (nfa, dfa, mdfa) = build_automata(grapheme_pattern());
        println!("nfa states: {:?}", nfa.len());
        // println!("nfa classes: {:?}", nfa.byte_classes);

        let bytes = dfa.len() * dfa.alphabet_len() * 8;
        println!("dfa states: {:?} ({} bytes)", dfa.len(), bytes);
        let bytes = mdfa.len() * mdfa.alphabet_len() * 8;
        println!("min dfa states: {:?} ({} bytes)", mdfa.len(), bytes);
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
