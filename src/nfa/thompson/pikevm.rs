use alloc::sync::Arc;

use crate::{
    nfa::thompson::NFA,
    util::{id::StateID, sparse_set::SparseSet},
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
}

impl Builder {
    /// Create a new PikeVM builder with its default configuration.
    pub fn new() -> Builder {
        Builder { config: Config::default() }
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
    fn nfa(&self) -> &Arc<NFA> {
        &self.nfa
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
    IP(StateID),
    Capture { slot: usize, pos: Slot },
}

impl Cache {
    pub fn new(vm: &PikeVM) -> Cache {
        Cache {
            stack: vec![],
            clist: Threads::new(vm.nfa()),
            nlist: Threads::new(vm.nfa()),
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
        self.slots_per_thread = nfa.capture_len() * 2;
        self.set.resize(nfa.states().len());
        self.caps.resize(self.slots_per_thread * nfa.states().len(), None);
    }

    fn caps(&mut self, sid: StateID) -> &mut [Slot] {
        let i = sid.as_usize() * self.slots_per_thread;
        &mut self.caps[i..i + self.slots_per_thread]
    }
}

#[derive(Clone, Debug)]
pub struct Error(());
