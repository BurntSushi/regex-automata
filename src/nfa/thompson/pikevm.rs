use alloc::sync::Arc;

use crate::{
    nfa::thompson::{self, Error, State, NFA},
    util::{id::StateID, matchtypes::MultiMatch, sparse_set::SparseSet},
};

#[derive(Clone, Copy, Debug, Default)]
pub struct Config {}

impl Config {
    /// Return a new default PikeVM configuration.
    pub fn new() -> Config {
        Config::default()
    }

    pub(crate) fn overwrite(self, o: Config) -> Config {
        Config {}
    }
}

/// A builder for a PikeVM.
#[derive(Clone, Debug)]
pub struct Builder {
    config: Config,
    thompson: thompson::Builder,
}

impl Builder {
    /// Create a new PikeVM builder with its default configuration.
    pub fn new() -> Builder {
        Builder {
            config: Config::default(),
            thompson: thompson::Builder::new(),
        }
    }

    pub fn build(&self, pattern: &str) -> Result<PikeVM, Error> {
        self.build_many(&[pattern])
    }

    pub fn build_many<P: AsRef<str>>(
        &self,
        patterns: &[P],
    ) -> Result<PikeVM, Error> {
        let nfa = self.thompson.build_many(patterns)?;
        self.build_from_nfa(Arc::new(nfa))
    }

    pub fn build_from_nfa(&self, nfa: Arc<NFA>) -> Result<PikeVM, Error> {
        Ok(PikeVM { nfa })
    }
}

#[derive(Clone, Debug)]
pub struct PikeVM {
    nfa: Arc<NFA>,
}

impl PikeVM {
    pub fn new(pattern: &str) -> Result<PikeVM, Error> {
        PikeVM::builder().build(pattern)
    }

    pub fn new_many<P: AsRef<str>>(patterns: &[P]) -> Result<PikeVM, Error> {
        PikeVM::builder().build_many(patterns)
    }

    pub fn config() -> Config {
        Config::new()
    }

    pub fn builder() -> Builder {
        Builder::new()
    }

    pub fn create_cache(&self) -> Cache {
        Cache::new(self.nfa())
    }

    pub fn create_captures(&self) -> Captures {
        Captures::new(self.nfa())
    }

    pub fn nfa(&self) -> &Arc<NFA> {
        &self.nfa
    }

    pub fn find_leftmost_at(
        &self,
        cache: &mut Cache,
        haystack: &[u8],
        start: usize,
        end: usize,
    ) -> Option<MultiMatch> {
        todo!()
    }

    fn step(
        &self,
        nlist: &mut Threads,
        slots: &mut [Slot],
        thread_caps: &mut [Slot],
        stack: &mut Vec<FollowEpsilon>,
        sid: StateID,
        haystack: &[u8],
        at: usize,
    ) -> bool {
        match *self.nfa.state(sid) {
            State::Fail
            | State::Look { .. }
            | State::Union { .. }
            | State::Capture { .. } => false,
            State::Range { ref range } => {
                if range.matches_byte(haystack[at]) {
                    self.epsilon_closure(
                        nlist,
                        thread_caps,
                        stack,
                        range.next,
                        haystack,
                        at + 1,
                    );
                }
                false
            }
            State::Sparse(ref sparse) => {
                if let Some(next) = sparse.matches_byte(haystack[at]) {
                    self.epsilon_closure(
                        nlist,
                        thread_caps,
                        stack,
                        next,
                        haystack,
                        at + 1,
                    );
                }
                false
            }
            State::Match { id } => {
                for (slot, val) in slots.iter_mut().zip(thread_caps.iter()) {
                    *slot = *val;
                }
                true
            }
        }
    }

    fn epsilon_closure(
        &self,
        nlist: &mut Threads,
        thread_caps: &mut [Slot],
        stack: &mut Vec<FollowEpsilon>,
        sid: StateID,
        haystack: &[u8],
        at: usize,
    ) {
        stack.push(FollowEpsilon::StateID(sid));
        while let Some(frame) = stack.pop() {
            match frame {
                FollowEpsilon::StateID(sid) => {
                    self.epsilon_closure_step(
                        nlist,
                        thread_caps,
                        stack,
                        sid,
                        haystack,
                        at,
                    );
                }
                FollowEpsilon::Capture { slot, pos } => {
                    thread_caps[slot] = pos;
                }
            }
        }
    }

    fn epsilon_closure_step(
        &self,
        nlist: &mut Threads,
        thread_caps: &mut [Slot],
        stack: &mut Vec<FollowEpsilon>,
        mut sid: StateID,
        haystack: &[u8],
        at: usize,
    ) {
        loop {
            if !nlist.set.insert(sid) {
                break;
            }
            match *self.nfa.state(sid) {
                State::Fail
                | State::Range { .. }
                | State::Sparse { .. }
                | State::Match { .. } => {
                    let t = &mut nlist.caps(sid);
                    for (slot, val) in t.iter_mut().zip(thread_caps.iter()) {
                        *slot = *val;
                    }
                    return;
                }
                State::Look { look, next } => {
                    if !look.matches(haystack, at) {
                        break;
                    }
                    sid = next;
                }
                State::Union { ref alternates } => {
                    sid = match alternates.get(0) {
                        None => break,
                        Some(&sid) => sid,
                    };
                    stack.extend(
                        alternates[1..]
                            .iter()
                            .copied()
                            .rev()
                            .map(FollowEpsilon::StateID),
                    );
                }
                State::Capture { next, slot } => {
                    if slot < thread_caps.len() {
                        stack.push(FollowEpsilon::Capture {
                            slot,
                            pos: thread_caps[slot],
                        });
                        thread_caps[slot] = Some(at);
                    }
                    sid = next;
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Captures {
    slots: Vec<Slot>,
}

impl Captures {
    pub fn new(nfa: &NFA) -> Captures {
        Captures { slots: vec![None; nfa.capture_slot_len()] }
    }
}

#[derive(Clone, Debug)]
pub struct Cache {
    stack: Vec<FollowEpsilon>,
    clist: Threads,
    nlist: Threads,
}

type Slot = Option<usize>;

#[derive(Clone, Debug)]
struct Threads {
    set: SparseSet,
    caps: Vec<Slot>,
    slots_per_thread: usize,
}

#[derive(Clone, Debug)]
enum FollowEpsilon {
    StateID(StateID),
    Capture { slot: usize, pos: Slot },
}

impl Cache {
    pub fn new(nfa: &NFA) -> Cache {
        Cache {
            stack: vec![],
            clist: Threads::new(nfa),
            nlist: Threads::new(nfa),
        }
    }
}

impl Threads {
    fn new(nfa: &NFA) -> Threads {
        let mut threads = Threads {
            set: SparseSet::new(0),
            caps: vec![],
            slots_per_thread: 0,
        };
        threads.resize(nfa);
        threads
    }

    fn resize(&mut self, nfa: &NFA) {
        if nfa.states().len() == self.set.capacity() {
            return;
        }
        self.slots_per_thread = nfa.capture_slot_len();
        self.set.resize(nfa.states().len());
        self.caps.resize(self.slots_per_thread * nfa.states().len(), None);
    }

    fn caps(&mut self, sid: StateID) -> &mut [Slot] {
        let i = sid.as_usize() * self.slots_per_thread;
        &mut self.caps[i..i + self.slots_per_thread]
    }
}
