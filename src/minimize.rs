use std::cell::RefCell;
use std::rc::Rc;

use dfa::{self, ALPHABET_SIZE, DFA};

#[derive(Debug)]
pub struct Minimizer<'a> {
    dfa: &'a mut DFA,
    in_transitions: Vec<Vec<Vec<dfa::StateID>>>,
    partitions: Vec<StateSet>,
    waiting: Vec<StateSet>,
    // waiting_set: BTreeSet<StateSet>,
}

#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
struct StateSet(Rc<RefCell<Vec<dfa::StateID>>>);

impl<'a> Minimizer<'a> {
    pub fn new(dfa: &'a mut DFA) -> Minimizer<'a> {
        let in_transitions = Minimizer::incoming_transitions(dfa);
        let partitions = Minimizer::initial_partitions(dfa);
        let waiting = vec![partitions[0].clone()];
        // let mut waiting_set = BTreeSet::new();
        // waiting_set.insert(partitions[0].clone());

        // Minimizer { dfa, in_transitions, partitions, waiting, waiting_set }
        Minimizer { dfa, in_transitions, partitions, waiting }
    }

    pub fn run(mut self) {
        let mut incoming = StateSet::empty();

        while let Some(set) = self.waiting.pop() {
            for b in 0..=255 {
                self.find_incoming_to(b, &set, &mut incoming);

                let mut newparts = vec![];
                for p in 0..self.partitions.len() {
                    let x = self.partitions[p].intersection(&incoming);
                    if x.is_empty() {
                        newparts.push(self.partitions[p].clone());
                        continue;
                    }

                    let y = self.partitions[p].subtract(&incoming);
                    if y.is_empty() {
                        newparts.push(self.partitions[p].clone());
                        continue;
                    }

                    newparts.push(x.clone());
                    newparts.push(y.clone());
                    match self.waiting.iter().position(|s| s == &self.partitions[p]) {
                        Some(i) => {
                            self.waiting[i] = x;
                            self.waiting.push(y);
                        }
                        None => {
                            if x.len() <= y.len() {
                                self.waiting.push(x);
                            } else {
                                self.waiting.push(y);
                            }
                        }
                    }
                }
                self.partitions = newparts;
            }
        }

        let mut state_to_part = vec![dfa::DEAD; self.dfa.len()];
        for p in &self.partitions {
            p.iter(|id| state_to_part[id] = p.first());
        }

        let mut minimal_ids = vec![dfa::DEAD; self.dfa.len()];
        let mut new_id = 0;
        for (id, state) in self.dfa.iter() {
            if state_to_part[id] == id {
                minimal_ids[id] = new_id;
                new_id += 1;
            }
        }
        let minimal_count = new_id;

        for id in 0..self.dfa.len() {
            if state_to_part[id] != id {
                continue;
            }
            for (_, next) in self.dfa.get_state_mut(id).iter_mut() {
                *next = minimal_ids[state_to_part[*next]];
            }
            self.dfa.swap_states(id, minimal_ids[id]);
        }

        let old_start = self.dfa.start();
        self.dfa.set_start_state(minimal_ids[state_to_part[old_start]]);
        self.dfa.truncate_states(minimal_count);

        let old_max = self.dfa.max_match_state();
        for id in 1..self.dfa.len() {
            if state_to_part[id] > old_max {
                break;
            }
            self.dfa.set_max_match_state(id);
        }
    }

    fn find_incoming_to(
        &self,
        b: u8,
        set: &StateSet,
        incoming: &mut StateSet,
    ) {
        incoming.clear();
        set.iter(|id| {
            for &inid in &self.in_transitions[id][b as usize] {
                incoming.add(inid);
            }
        });
        incoming.canonicalize();
    }

    fn initial_partitions(dfa: &DFA) -> Vec<StateSet> {
        let mut is_match = StateSet::empty();
        let mut no_match = StateSet::empty();
        for (id, _) in dfa.iter() {
            if dfa.is_match_state(id) {
                is_match.add(id);
            } else {
                no_match.add(id);
            }
        }
        assert!(!is_match.is_empty(), "must have at least one matching state");

        let mut sets = vec![is_match];
        if !no_match.is_empty() {
            sets.push(no_match);
        }
        sets.sort_by_key(|s| s.len());
        sets
    }

    fn incoming_transitions(dfa: &DFA) -> Vec<Vec<Vec<dfa::StateID>>> {
        let mut incoming = vec![];
        for state in dfa.iter() {
            incoming.push(vec![vec![]; ALPHABET_SIZE]);
        }
        for (id, state) in dfa.iter() {
            for (b, next) in state.iter() {
                incoming[next][b as usize].push(id);
            }
        }
        incoming
    }
}

impl StateSet {
    fn empty() -> StateSet {
        StateSet(Rc::new(RefCell::new(vec![])))
    }

    fn add(&mut self, id: dfa::StateID) {
        self.0.borrow_mut().push(id);
    }

    fn first(&self) -> dfa::StateID {
        self.0.borrow()[0]
    }

    fn canonicalize(&mut self) {
        self.0.borrow_mut().sort();
        self.0.borrow_mut().dedup();
    }

    fn clear(&mut self) {
        self.0.borrow_mut().clear();
    }

    fn len(&self) -> usize {
        self.0.borrow().len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn deep_clone(&self) -> StateSet {
        let ids = self.0.borrow().iter().cloned().collect();
        StateSet(Rc::new(RefCell::new(ids)))
    }

    fn iter(&self, mut f: impl FnMut(dfa::StateID)) {
        for &id in self.0.borrow().iter() {
            f(id);
        }
    }

    fn intersection(&self, other: &StateSet) -> StateSet {
        if self.is_empty() || other.is_empty() {
            return StateSet::empty();
        }

        let mut result = StateSet::empty();
        let (seta, setb) = (self.0.borrow(), other.0.borrow());
        let (mut ita, mut itb) = (seta.iter().cloned(), setb.iter().cloned());
        let (mut a, mut b) = (ita.next().unwrap(), itb.next().unwrap());
        loop {
            if a == b {
                result.add(a);
                a = match ita.next() {
                    None => break,
                    Some(a) => a,
                };
                b = match itb.next() {
                    None => break,
                    Some(b) => b,
                };
            } else if a < b {
                a = match ita.next() {
                    None => break,
                    Some(a) => a,
                };
            } else {
                b = match itb.next() {
                    None => break,
                    Some(b) => b,
                };
            }
        }
        result
    }

    fn subtract(&self, other: &StateSet) -> StateSet {
        if self.is_empty() || other.is_empty() {
            return self.deep_clone();
        }

        let mut result = StateSet::empty();
        let (seta, setb) = (self.0.borrow(), other.0.borrow());
        let (mut ita, mut itb) = (seta.iter().cloned(), setb.iter().cloned());
        let (mut a, mut b) = (ita.next().unwrap(), itb.next().unwrap());
        loop {
            if a == b {
                a = match ita.next() {
                    None => break,
                    Some(a) => a,
                };
                b = match itb.next() {
                    None => { result.add(a); break; }
                    Some(b) => b,
                };
            } else if a < b {
                result.add(a);
                a = match ita.next() {
                    None => break,
                    Some(a) => a,
                };
            } else {
                b = match itb.next() {
                    None => { result.add(a); break; }
                    Some(b) => b,
                };
            }
        }
        for a in ita {
            result.add(a);
        }
        result
    }
}
