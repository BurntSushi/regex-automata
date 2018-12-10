use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use dfa::DFA;
use state_id::{StateID, dead_id};

pub struct Minimizer<'a, S> {
    dfa: &'a mut DFA<S>,
    in_transitions: Vec<Vec<Vec<S>>>,
    partitions: Vec<StateSet<S>>,
    waiting: Vec<StateSet<S>>,
    // waiting_set: BTreeSet<StateSet>,
}

impl<'a, S: StateID> fmt::Debug for Minimizer<'a, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Minimizer")
            .field("dfa", &self.dfa)
            .field("in_transitions", &self.in_transitions)
            .field("partitions", &self.partitions)
            .field("waiting", &self.waiting)
            .finish()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
struct StateSet<S>(Rc<RefCell<Vec<S>>>);

impl<'a, S: StateID> Minimizer<'a, S> {
    pub fn new(dfa: &'a mut DFA<S>) -> Minimizer<'a, S> {
        assert!(
            !dfa.kind().is_premultiplied(),
            "cannot minimize a premultiplied DFA"
        );

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
            for b in (0..self.dfa.alphabet_len()).map(|b| b as u8) {
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

        let mut state_to_part = vec![dead_id(); self.dfa.len()];
        for p in &self.partitions {
            p.iter(|id| state_to_part[id.to_usize()] = p.first());
        }

        let mut minimal_ids = vec![dead_id(); self.dfa.len()];
        let mut new_id = S::from_usize(0);
        for (id, state) in self.dfa.iter() {
            if state_to_part[id.to_usize()] == id {
                minimal_ids[id.to_usize()] = new_id;
                new_id = S::from_usize(new_id.to_usize() + 1);
            }
        }
        let minimal_count = new_id.to_usize();

        for id in (0..self.dfa.len()).map(S::from_usize) {
            if state_to_part[id.to_usize()] != id {
                continue;
            }
            for (_, next) in self.dfa.get_state_mut(id).iter_mut() {
                *next = minimal_ids[state_to_part[next.to_usize()].to_usize()];
            }
            self.dfa.swap_states(id, minimal_ids[id.to_usize()]);
        }

        let old_start = self.dfa.start();
        self.dfa.set_start_state(
            minimal_ids[state_to_part[old_start.to_usize()].to_usize()],
        );
        self.dfa.truncate_states(minimal_count);

        let old_max = self.dfa.max_match_state();
        for id in (1..self.dfa.len()).map(S::from_usize) {
            if state_to_part[id.to_usize()] > old_max {
                break;
            }
            self.dfa.set_max_match_state(id);
        }
    }

    fn find_incoming_to(
        &self,
        b: u8,
        set: &StateSet<S>,
        incoming: &mut StateSet<S>,
    ) {
        incoming.clear();
        set.iter(|id| {
            for &inid in &self.in_transitions[id.to_usize()][b as usize] {
                incoming.add(inid);
            }
        });
        incoming.canonicalize();
    }

    fn initial_partitions(dfa: &DFA<S>) -> Vec<StateSet<S>> {
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

    fn incoming_transitions(dfa: &DFA<S>) -> Vec<Vec<Vec<S>>> {
        let mut incoming = vec![];
        for state in dfa.iter() {
            incoming.push(vec![vec![]; dfa.alphabet_len()]);
        }
        for (id, state) in dfa.iter() {
            for (b, next) in state.iter() {
                incoming[next.to_usize()][b as usize].push(id);
            }
        }
        incoming
    }
}

impl<S: StateID> StateSet<S> {
    fn empty() -> StateSet<S> {
        StateSet(Rc::new(RefCell::new(vec![])))
    }

    fn add(&mut self, id: S) {
        self.0.borrow_mut().push(id);
    }

    fn first(&self) -> S {
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

    fn deep_clone(&self) -> StateSet<S> {
        let ids = self.0.borrow().iter().cloned().collect();
        StateSet(Rc::new(RefCell::new(ids)))
    }

    fn iter<F: FnMut(S)>(&self, mut f: F) {
        for &id in self.0.borrow().iter() {
            f(id);
        }
    }

    fn intersection(&self, other: &StateSet<S>) -> StateSet<S> {
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

    fn subtract(&self, other: &StateSet<S>) -> StateSet<S> {
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
