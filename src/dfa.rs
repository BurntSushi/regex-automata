use std::fmt;

use determinize::Determinizer;
use minimize::Minimizer;
use nfa::NFA;

pub const DEAD: StateID = 0;
pub const ALPHABET_SIZE: usize = 256;

pub type StateID = usize;

pub struct DFA {
    /// The set of DFA states and their transitions. Transitions point to
    /// indices in this list.
    pub(crate) states: Vec<State>,
    /// The initial start state. This is either `0` for an empty DFA with a
    /// single dead state or `1` for the first DFA state built.
    pub(crate) start: StateID,
}

pub struct State {
    pub(crate) is_match: bool,
    pub(crate) transitions: Box<[StateID]>,
}

impl DFA {
    pub fn empty() -> DFA {
        DFA {
            states: vec![State::empty()],
            start: DEAD,
        }
    }

    pub(crate) fn from_nfa(nfa: &NFA) -> DFA {
        Determinizer::new(nfa).build()
    }

    pub fn minimize(&mut self) {
        Minimizer::new(self).run();
    }

    pub(crate) fn set_transition(
        &mut self,
        from: StateID,
        input: u8,
        to: StateID,
    ) {
        self.states[from].transitions[input as usize] = to;
    }

    pub fn is_match(&self, bytes: &[u8]) -> bool {
        let mut state = self.start;
        if state == DEAD {
            return false;
        } else if self.states[state].is_match {
            return true;
        }
        for (i, &b) in bytes.iter().enumerate() {
            state = self.states[state].transitions[b as usize];
            if state == DEAD {
                return false;
            } else if self.states[state].is_match {
                return true;
            }
        }
        false
    }

    pub fn find(&self, bytes: &[u8]) -> Option<usize> {
        let mut state = self.start;
        let mut last_match =
            if state == DEAD {
                return None;
            } else if self.states[state].is_match {
                Some(0)
            } else {
                None
            };
        for (i, &b) in bytes.iter().enumerate() {
            state = self.states[state].transitions[b as usize];
            if state == DEAD {
                return last_match;
            } else if self.states[state].is_match {
                last_match = Some(i + 1);
            }
        }
        last_match
    }
}

impl State {
    pub fn empty() -> State {
        State {
            is_match: false,
            transitions: vec![DEAD; ALPHABET_SIZE].into_boxed_slice(),
        }
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

        for (id, state) in self.states.iter().enumerate() {
            writeln!(f, "{}{:04}: {:?}", state_status(id, state), id, state)?;
        }
        Ok(())
    }
}

impl fmt::Debug for State {
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
        println!("PATTERN: {:?}", pattern);
        println!("NFA:");
        for (i, state) in nfa.states.borrow().iter().enumerate() {
            println!("{:03X}: {:X?}", i, state);
        }

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

    fn build_automata(pattern: &str) -> (NFA, DFA) {
        let builder = DFABuilder::new();
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

        print_automata(r"[01]*1[01]{5}");
    }
}
