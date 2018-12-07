use std::fmt;
use std::iter;
use std::mem;
use std::slice;

use determinize::Determinizer;
use minimize::Minimizer;
use nfa::NFA;

pub const DEAD: StateID = 0;
pub const ALPHABET_SIZE: usize = 256;

pub type StateID = usize;

pub struct DFA {
    trans: Vec<StateID>,
    is_match: Vec<bool>,
    max_match: StateID,
    /// The initial start state. This is either `0` for an empty DFA with a
    /// single dead state or `1` for the first DFA state built.
    start: StateID,
}

impl DFA {
    pub fn empty() -> DFA {
        let mut dfa = DFA {
            trans: vec![],
            is_match: vec![],
            max_match: 1,
            start: DEAD,
        };
        dfa.add_empty_state(false);
        dfa
    }

    pub fn len(&self) -> usize {
        self.is_match.len()
    }

    pub fn is_match(&self, bytes: &[u8]) -> bool {
        let mut state = self.start;
        if state <= self.max_match {
            return state != DEAD;
        }
        for &b in bytes.iter() {
            state = unsafe {
                *self.trans.get_unchecked(state * ALPHABET_SIZE + b as usize)
            };
            if state <= self.max_match {
                return state != DEAD;
            }
        }
        false
    }

    pub fn find(&self, bytes: &[u8]) -> Option<usize> {
        let mut state = self.start;
        let mut last_match =
            if state == DEAD {
                return None;
            } else if self.is_match[state] {
                Some(0)
            } else {
                None
            };
        for (i, &b) in bytes.iter().enumerate() {
            state = self.trans[state * ALPHABET_SIZE + b as usize];
            if state == DEAD {
                return last_match;
            } else if self.is_match[state] {
                last_match = Some(i + 1);
            }
        }
        last_match
    }

    pub fn minimize(&mut self) {
        Minimizer::new(self).run();
    }
}

impl DFA {
    pub(crate) fn from_nfa(nfa: &NFA) -> DFA {
        Determinizer::new(nfa).build()
    }

    pub(crate) fn start(&self) -> StateID {
        self.start
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
        let i = (from * ALPHABET_SIZE) + (input as usize);
        self.trans[i] = to;
    }

    pub(crate) fn add_empty_state(&mut self, is_match: bool) -> StateID {
        let id = self.is_match.len();
        self.is_match.push(is_match);
        self.trans.extend(0..ALPHABET_SIZE);
        id
    }

    pub(crate) fn get_state(&self, id: StateID) -> State {
        let i = id * ALPHABET_SIZE;
        State {
            is_match: self.is_match[id],
            transitions: &self.trans[i..i+ALPHABET_SIZE],
        }
    }

    pub(crate) fn get_state_mut(&mut self, id: StateID) -> StateMut {
        let i = id * ALPHABET_SIZE;
        StateMut {
            is_match: self.is_match[id],
            transitions: &mut self.trans[i..i+ALPHABET_SIZE],
        }
    }

    pub(crate) fn is_match_state(&self, id: StateID) -> bool {
        self.is_match[id]
    }

    pub(crate) fn max_match_state(&self) -> StateID {
        self.max_match
    }

    pub(crate) fn set_max_match_state(&mut self, id: StateID) {
        self.max_match = id;
    }

    pub(crate) fn iter(&self) -> StateIter {
        let it = self.is_match.iter().zip(self.trans.chunks(ALPHABET_SIZE));
        StateIter { it: it.enumerate() }
    }

    pub(crate) fn swap_states(&mut self, id1: StateID, id2: StateID) {
        for b in 0..ALPHABET_SIZE {
            self.trans.swap(id1 * ALPHABET_SIZE + b, id2 * ALPHABET_SIZE + b);
        }
        self.is_match.swap(id1, id2);
    }

    pub(crate) fn truncate_states(&mut self, count: usize) {
        self.trans.truncate(count * ALPHABET_SIZE);
        self.is_match.truncate(count);
    }

    pub(crate) fn shuffle_match_states(&mut self, is_match: &[bool]) {
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
}

#[derive(Debug)]
pub struct StateIter<'a> {
    it: iter::Enumerate<iter::Zip<slice::Iter<'a, bool>, slice::Chunks<'a, StateID>>>,
}

impl<'a> Iterator for StateIter<'a> {
    type Item = (StateID, State<'a>);

    fn next(&mut self) -> Option<(StateID, State<'a>)> {
        self.it.next().map(|(i, (&is_match, chunk))| {
            (i, State { is_match, transitions: chunk })
        })
    }
}

pub struct State<'a> {
    is_match: bool,
    transitions: &'a [StateID],
}

impl<'a> State<'a> {
    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    pub fn get(&self, b: u8) -> StateID {
        self.transitions[b as usize]
    }

    pub fn is_match(&self) -> bool {
        self.is_match
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
    is_match: bool,
    // transitions: Box<[StateID]>,
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

impl fmt::Debug for DFA {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fn state_status(id: StateID, state: &State) -> String {
            let mut status = vec![b' ', b' '];
            if id == 0 {
                status[0] = b'D';
            } else if id == 1 {
                status[0] = b'>';
            }
            if state.is_match {
                status[1] = b'*';
            }
            String::from_utf8(status).unwrap()
        }

        for (id, state) in self.iter() {
            writeln!(f, "{}{:04}: {:?}", state_status(id, &state), id, state)?;
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
        let (nfa, mut dfa) = build_automata(pattern);

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

        Minimizer::new(&mut dfa).run();

        println!("Minimal DFA:");
        print!("{:?}", dfa);
        println!("{}", "~".repeat(79));

        println!("{}", "#".repeat(100));
    }

    fn print_automata_counts(pattern: &str) {
        let (nfa, mut dfa) = build_automata(pattern);
        println!("nfa # states: {:?}", nfa.states.borrow().len());
        println!("dfa # states: {:?}", dfa.len());
        dfa.minimize();
        println!("minimal dfa # states: {:?}", dfa.len());
    }

    fn build_automata(pattern: &str) -> (NFA, DFA) {
        let mut builder = DFABuilder::new();
        builder.anchored(true).allow_invalid_utf8(true);
        let nfa = builder.build_nfa(pattern).unwrap();
        let dfa = builder.build(pattern).unwrap();
        (nfa, dfa)
    }

    fn build_automata_min(pattern: &str) -> DFA {
        let (_, mut dfa) = build_automata(pattern);
        Minimizer::new(&mut dfa).run();
        dfa
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
        print_automata(r"(a+|b)?");

        // let data = ::std::fs::read_to_string("/usr/share/dict/words").unwrap();
        // let mut words: Vec<&str> = data.lines().collect();
        // println!("{} words", words.len());
        // words.sort_by(|w1, w2| w1.len().cmp(&w2.len()).reverse());
        // let pattern = words.join("|");
        // print_automata_counts(&pattern);
        // print_automata(&pattern);
    }
}
