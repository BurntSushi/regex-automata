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

impl PikeVM {}

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

impl Threads {
    fn new() -> Self {
        Threads { set: SparseSet::new(0), caps: vec![], slots_per_thread: 0 }
    }

    fn resize(&mut self, state_count: usize, ncaps: usize) {
        if state_count == self.set.capacity() {
            return;
        }
        self.slots_per_thread = ncaps * 2;
        self.set.resize(state_count);
        self.caps.resize(self.slots_per_thread * state_count, None);
    }

    fn caps(&mut self, sid: StateID) -> &mut [Slot] {
        let i = sid.as_usize() * self.slots_per_thread;
        &mut self.caps[i..i + self.slots_per_thread]
    }
}

#[derive(Clone, Debug)]
pub struct Error(());
