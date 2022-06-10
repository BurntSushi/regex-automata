use core::cell::RefCell;

use alloc::{
    sync::{Arc, Weak},
    vec,
    vec::Vec,
};

use crate::{
    nfa::thompson::{self, Captures, Error, State, NFA},
    util::{
        id::{PatternID, StateID},
        iter,
        nonmax::NonMaxUsize,
        prefilter::{self, Prefilter},
        search::{Match, MatchError, MatchKind, PatternSet, Search},
        sparse_set::SparseSet,
    },
};

/// A simple macro for conditionally executing instrumentation logic when
/// the 'trace' log level is enabled. This is a compile-time no-op when the
/// 'instrument-pikevm' feature isn't enabled. The intent here is that this
/// makes it easier to avoid doing extra work when instrumentation isn't
/// enabled.
///
/// This macro accepts a closure of type `|&mut Counters|`. The closure can
/// then increment counters (or whatever) in accordance with what one wants
/// to track.
macro_rules! instrument {
    ($fun:expr) => {
        #[cfg(feature = "instrument-pikevm")]
        {
            // let fun: impl FnMut(&mut Counters) = $fun;
            let mut fun: &mut dyn FnMut(&mut Counters) = &mut $fun;
            COUNTERS.with(|c: &RefCell<Counters>| fun(&mut *c.borrow_mut()));
        }
    };
}

/// Effectively global state used to keep track of instrumentation counters.
/// The "proper" way to do this is to thread it through the PikeVM, but it
/// makes the code quite icky. Since this is just a debugging feature, we're
/// content to relegate it to thread local state. When instrumentation is
/// enabled, the counters are reset at the beginning of every search and
/// printed (with the 'trace' log level) at the end of every search.
#[cfg(feature = "instrument-pikevm")]
thread_local! {
    static COUNTERS: RefCell<Counters> = RefCell::new(Counters::empty());
}

#[derive(Clone, Debug, Default)]
pub struct Config {
    anchored: Option<bool>,
    match_kind: Option<MatchKind>,
    utf8: Option<bool>,
}

impl Config {
    /// Return a new default PikeVM configuration.
    pub fn new() -> Config {
        Config::default()
    }

    pub fn anchored(mut self, yes: bool) -> Config {
        self.anchored = Some(yes);
        self
    }

    pub fn match_kind(mut self, kind: MatchKind) -> Config {
        self.match_kind = Some(kind);
        self
    }

    pub fn utf8(mut self, yes: bool) -> Config {
        self.utf8 = Some(yes);
        self
    }

    pub fn get_anchored(&self) -> bool {
        self.anchored.unwrap_or(false)
    }

    pub fn get_match_kind(&self) -> MatchKind {
        self.match_kind.unwrap_or(MatchKind::LeftmostFirst)
    }

    pub fn get_utf8(&self) -> bool {
        self.utf8.unwrap_or(true)
    }

    pub(crate) fn overwrite(&self, o: Config) -> Config {
        Config {
            anchored: o.anchored.or(self.anchored),
            match_kind: o.match_kind.or(self.match_kind),
            utf8: o.utf8.or(self.utf8),
        }
    }
}

/// A builder for a PikeVM.
#[derive(Clone, Debug)]
pub struct Builder {
    config: Config,
    thompson: thompson::Compiler,
}

impl Builder {
    /// Create a new PikeVM builder with its default configuration.
    pub fn new() -> Builder {
        Builder {
            config: Config::default(),
            thompson: thompson::Compiler::new(),
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
        self.build_from_nfa(nfa)
    }

    pub fn build_from_nfa(&self, nfa: NFA) -> Result<PikeVM, Error> {
        // TODO: Check that this is correct.
        // if !cfg!(all(
        // feature = "dfa",
        // feature = "syntax",
        // feature = "unicode-perl"
        // )) {
        // If the NFA has no captures, then the PikeVM doesn't work since it
        // relies on them in order to report match locations. However, in the
        // special case of an NFA with no patterns, it is allowed, since no
        // matches can ever be produced.
        if !nfa.has_capture() && nfa.pattern_len() > 0 {
            return Err(Error::missing_captures());
        }
        if !cfg!(feature = "syntax") {
            if nfa.has_word_boundary_unicode() {
                return Err(Error::unicode_word_unavailable());
            }
        }
        Ok(PikeVM { config: self.config.clone(), nfa })
    }

    pub fn configure(&mut self, config: Config) -> &mut Builder {
        self.config = self.config.overwrite(config);
        self
    }

    /// Set the syntax configuration for this builder using
    /// [`SyntaxConfig`](crate::SyntaxConfig).
    ///
    /// This permits setting things like case insensitivity, Unicode and multi
    /// line mode.
    ///
    /// These settings only apply when constructing a PikeVM directly from a
    /// pattern.
    pub fn syntax(
        &mut self,
        config: crate::util::syntax::SyntaxConfig,
    ) -> &mut Builder {
        self.thompson.syntax(config);
        self
    }

    /// Set the Thompson NFA configuration for this builder using
    /// [`nfa::thompson::Config`](crate::nfa::thompson::Config).
    ///
    /// This permits setting things like if additional time should be spent
    /// shrinking the size of the NFA.
    ///
    /// These settings only apply when constructing a PikeVM directly from a
    /// pattern.
    pub fn thompson(&mut self, config: thompson::Config) -> &mut Builder {
        self.thompson.configure(config);
        self
    }
}

#[derive(Clone, Debug)]
pub struct PikeVM {
    config: Config,
    nfa: NFA,
}

impl PikeVM {
    pub fn new(pattern: &str) -> Result<PikeVM, Error> {
        PikeVM::builder().build(pattern)
    }

    pub fn new_many<P: AsRef<str>>(patterns: &[P]) -> Result<PikeVM, Error> {
        PikeVM::builder().build_many(patterns)
    }

    pub fn new_from_nfa(nfa: NFA) -> Result<PikeVM, Error> {
        PikeVM::builder().build_from_nfa(nfa)
    }

    pub fn config() -> Config {
        Config::new()
    }

    pub fn builder() -> Builder {
        Builder::new()
    }

    pub fn create_cache(&self) -> Cache {
        Cache::new(self.get_nfa())
    }

    pub fn create_captures(&self) -> Captures {
        Captures::new(self.get_nfa().clone())
    }

    #[inline]
    pub fn get_config(&self) -> &Config {
        &self.config
    }

    #[inline]
    pub fn get_nfa(&self) -> &NFA {
        &self.nfa
    }
}

impl PikeVM {
    #[inline]
    pub fn is_match<H: AsRef<[u8]>>(
        &self,
        cache: &mut Cache,
        haystack: H,
    ) -> bool {
        let search = Search::new(haystack.as_ref())
            .earliest(true)
            .utf8(self.config.get_utf8());
        let mut caps = Captures::empty(self.nfa.clone());
        self.search(cache, None, &search, &mut caps);
        caps.is_match()
    }

    #[inline]
    pub fn find<H: AsRef<[u8]>>(
        &self,
        cache: &mut Cache,
        haystack: H,
        caps: &mut Captures,
    ) {
        let search =
            Search::new(haystack.as_ref()).utf8(self.config.get_utf8());
        self.search(cache, None, &search, caps)
    }

    #[inline]
    pub fn find_iter<'r: 'c, 'c, 'h, H: AsRef<[u8]> + ?Sized>(
        &'r self,
        cache: &'c mut Cache,
        haystack: &'h H,
    ) -> FindMatches<'h, 'c> {
        let search =
            Search::new(haystack.as_ref()).utf8(self.config.get_utf8());
        let mut caps = Captures::new_for_matches_only(self.get_nfa().clone());
        let it = iter::TryMatches::boxed(search, move |search| {
            self.search(cache, None, search, &mut caps);
            Ok(caps.get_match())
        })
        .infallible();
        FindMatches(it)
    }

    #[inline]
    pub fn captures_iter<'r: 'c, 'c, 'h, H: AsRef<[u8]> + ?Sized>(
        &'r self,
        cache: &'c mut Cache,
        haystack: &'h H,
    ) -> CapturesMatches<'h, 'c> {
        let search =
            Search::new(haystack.as_ref()).utf8(self.config.get_utf8());
        let caps = Arc::new(RefCell::new(self.create_captures()));
        let it = iter::TryMatches::boxed(search, {
            let caps = Arc::clone(&caps);
            move |search| {
                self.search(cache, None, search, &mut *caps.borrow_mut());
                Ok(caps.borrow().get_match())
            }
        })
        .infallible();
        CapturesMatches { caps, it }
    }

    #[inline]
    pub fn search(
        &self,
        cache: &mut Cache,
        mut pre: Option<&mut prefilter::Scanner>,
        search: &Search<'_>,
        caps: &mut Captures,
    ) {
        self.search_imp(cache, pre, search, caps);
        let m = match caps.get_match() {
            None => return,
            Some(m) => m,
        };
        if m.is_empty() {
            search
                .skip_empty_utf8_splits(m, |search| {
                    self.search_imp(cache, None, search, caps);
                    Ok(caps.get_match())
                })
                .unwrap();
        }
    }

    #[inline]
    pub fn which_overlapping_matches(
        &self,
        cache: &mut Cache,
        pre: Option<&mut prefilter::Scanner>,
        search: &Search<'_>,
        patset: &mut PatternSet,
    ) {
        self.which_overlapping_imp(cache, pre, search, patset)
    }
}

impl PikeVM {
    fn search_imp(
        &self,
        cache: &mut Cache,
        pre: Option<&mut prefilter::Scanner>,
        search: &Search<'_>,
        caps: &mut Captures,
    ) {
        // Why do we even care about this? Well, in our 'Captures'
        // representation, we use usize::MAX as a sentinel to indicate "no
        // match." This isn't problematic so long as our haystack doesn't have
        // a maximal length. Byte slices are guaranteed by Rust to have a
        // length that fits into isize, and so this assert should always pass.
        // But we put it here to make our assumption explicit.
        assert!(
            search.haystack().len() < core::usize::MAX,
            "byte slice lengths must be less than usize MAX",
        );
        instrument!(|c| c.reset(&self.nfa));
        caps.set_pattern(None);
        cache.clear(caps.slot_len());

        let allmatches =
            self.config.get_match_kind().continue_past_first_match();
        let anchored = self.config.get_anchored()
            || self.nfa.is_always_start_anchored()
            || search.get_pattern().is_some();
        let start_id = match search.get_pattern() {
            // We always use the anchored starting state here, even if doing an
            // unanchored search. The "unanchored" part of it is implemented
            // in the loop below, by computing the epsilon closure from the
            // anchored starting state whenever the current state set list is
            // empty.
            None => self.nfa.start_anchored(),
            Some(pid) => self.nfa.start_pattern(pid),
        };

        let Cache {
            ref mut stack,
            ref mut scratch_caps,
            ref mut clist,
            ref mut nlist,
        } = cache;
        let mut at = search.get_start();
        while at <= search.get_end() {
            if clist.set.is_empty() {
                if (caps.is_match() && !allmatches)
                    || (anchored && at > search.get_start())
                {
                    break;
                }
            }
            if clist.set.is_empty() || (!anchored && !caps.is_match()) {
                self.epsilon_closure(
                    stack,
                    clist,
                    scratch_caps,
                    start_id,
                    search.haystack(),
                    at,
                );
            }
            if self.steps(stack, clist, nlist, search.haystack(), at, caps) {
                if search.get_earliest() {
                    break;
                }
            }
            at += 1;
            core::mem::swap(clist, nlist);
            nlist.set.clear();
        }
        instrument!(|c| c.eprint(&self.nfa));
    }

    fn which_overlapping_imp(
        &self,
        cache: &mut Cache,
        pre: Option<&mut prefilter::Scanner>,
        search: &Search<'_>,
        patset: &mut PatternSet,
    ) {
        assert!(
            search.haystack().len() < core::usize::MAX,
            "byte slice lengths must be less than usize MAX",
        );
        instrument!(|c| c.reset(&self.nfa));
        cache.clear(0);

        let anchored = self.config.get_anchored()
            || self.nfa.is_always_start_anchored()
            || search.get_pattern().is_some();
        let start_id = match search.get_pattern() {
            None => self.nfa.start_anchored(),
            Some(pid) => self.nfa.start_pattern(pid),
        };

        let Cache {
            ref mut stack,
            ref mut scratch_caps,
            ref mut clist,
            ref mut nlist,
        } = cache;
        let mut at = search.get_start();
        while at <= search.get_end() {
            if anchored && clist.set.is_empty() && at > search.get_start() {
                break;
            }
            if clist.set.is_empty() || !anchored {
                self.epsilon_closure(
                    stack,
                    clist,
                    scratch_caps,
                    start_id,
                    search.haystack(),
                    at,
                );
            }
            self.steps_overlapping(
                stack,
                clist,
                nlist,
                search.haystack(),
                at,
                patset,
            );
            // If we found a match and filled our set, then there is no more
            // additional info that we can provide. Thus, we can quit.
            if patset.is_full() {
                break;
            }
            at += 1;
            core::mem::swap(clist, nlist);
            nlist.set.clear();
        }
        instrument!(|c| c.eprint(&self.nfa));
    }

    #[inline(always)]
    fn steps(
        &self,
        stack: &mut Vec<FollowEpsilon>,
        clist: &mut Threads,
        nlist: &mut Threads,
        haystack: &[u8],
        at: usize,
        caps: &mut Captures,
    ) -> bool {
        let mut matched = false;
        instrument!(|c| c.record_state_set(&clist.list));
        for sid in clist.list.drain(..) {
            let slots = &mut clist.caps[sid].slots[..caps.slot_len()];
            let pid = match self.step(stack, nlist, slots, sid, haystack, at) {
                None => continue,
                Some(pid) => pid,
            };
            matched = true;
            copy_to_captures(pid, slots, caps);
            if !self.config.get_match_kind().continue_past_first_match() {
                break;
            }
        }
        matched
    }

    #[inline(always)]
    fn steps_overlapping(
        &self,
        stack: &mut Vec<FollowEpsilon>,
        clist: &mut Threads,
        nlist: &mut Threads,
        haystack: &[u8],
        at: usize,
        patset: &mut PatternSet,
    ) {
        instrument!(|c| c.record_state_set(&clist.list));
        for sid in clist.list.drain(..) {
            let slots = &mut [];
            let pid = match self.step(stack, nlist, slots, sid, haystack, at) {
                None => continue,
                Some(pid) => pid,
            };
            patset.insert(pid);
        }
    }

    #[inline(always)]
    fn step(
        &self,
        stack: &mut Vec<FollowEpsilon>,
        nlist: &mut Threads,
        thread_caps: &mut [Slot],
        sid: StateID,
        haystack: &[u8],
        at: usize,
    ) -> Option<PatternID> {
        instrument!(|c| c.record_step(sid));
        match *self.nfa.state(sid) {
            State::Fail
            | State::Look { .. }
            | State::Union { .. }
            | State::BinaryUnion { .. }
            | State::Capture { .. } => None,
            State::ByteRange { ref trans } => {
                if trans.matches(haystack, at) {
                    // OK because 'at <= haystack.len() < usize::MAX', so
                    // adding 1 will never wrap.
                    let at = at.wrapping_add(1);
                    self.epsilon_closure(
                        stack,
                        nlist,
                        thread_caps,
                        trans.next,
                        haystack,
                        at,
                    );
                }
                None
            }
            State::Sparse(ref sparse) => {
                if let Some(next) = sparse.matches(haystack, at) {
                    // OK because 'at <= haystack.len() < usize::MAX', so
                    // adding 1 will never wrap.
                    let at = at.wrapping_add(1);
                    self.epsilon_closure(
                        stack,
                        nlist,
                        thread_caps,
                        next,
                        haystack,
                        at,
                    );
                }
                None
            }
            State::Match { pattern_id } => Some(pattern_id),
        }
    }

    #[inline(always)]
    fn epsilon_closure(
        &self,
        stack: &mut Vec<FollowEpsilon>,
        nlist: &mut Threads,
        thread_caps: &mut [Slot],
        sid: StateID,
        haystack: &[u8],
        at: usize,
    ) {
        instrument!(|c| {
            c.record_closure(sid);
            c.record_stack_push(sid);
        });
        stack.push(FollowEpsilon::StateID(sid));
        while let Some(frame) = stack.pop() {
            match frame {
                FollowEpsilon::StateID(sid) => {
                    self.epsilon_closure_step(
                        stack,
                        nlist,
                        thread_caps,
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

    #[inline(always)]
    fn epsilon_closure_step(
        &self,
        stack: &mut Vec<FollowEpsilon>,
        nlist: &mut Threads,
        thread_caps: &mut [Slot],
        mut sid: StateID,
        haystack: &[u8],
        at: usize,
    ) {
        // TODO: Also, get rid of the NFA utf-8 option? That would be nice...
        // Yes, I think we should. If we keep it, then we can't unconditionally
        // do our manual unanchored prefix.
        loop {
            instrument!(|c| c.record_set_insert(sid));
            if !nlist.set.insert(sid) {
                return;
            }
            match *self.nfa.state(sid) {
                State::Fail
                | State::Match { .. }
                | State::ByteRange { .. }
                | State::Sparse { .. } => {
                    nlist.caps(sid).copy_from_slice(thread_caps);
                    nlist.list.push(sid);
                    return;
                }
                State::Look { look, next } => {
                    if !look.matches(haystack, at) {
                        return;
                    }
                    sid = next;
                }
                State::Union { ref alternates } => {
                    sid = match alternates.get(0) {
                        None => return,
                        Some(&sid) => sid,
                    };
                    instrument!(|c| {
                        for &alt in &alternates[1..] {
                            c.record_stack_push(alt);
                        }
                    });
                    stack.extend(
                        alternates[1..]
                            .iter()
                            .copied()
                            .rev()
                            .map(FollowEpsilon::StateID),
                    );
                }
                State::BinaryUnion { alt1, alt2 } => {
                    sid = alt1;
                    instrument!(|c| c.record_stack_push(sid));
                    stack.push(FollowEpsilon::StateID(alt2));
                }
                State::Capture { next, slot } => {
                    if slot < thread_caps.len() {
                        instrument!(|c| c.record_stack_push(sid));
                        stack.push(FollowEpsilon::Capture {
                            slot,
                            pos: thread_caps[slot],
                        });
                        // OK because length of a slice must fit into an isize.
                        thread_caps[slot] =
                            Some(NonMaxUsize::new(at).unwrap());
                    }
                    sid = next;
                }
            }
        }
    }
}

type TryMatchesClosure<'h, 'c> =
    Box<dyn FnMut(&Search<'h>) -> Result<Option<Match>, MatchError> + 'c>;

/// An iterator over all non-overlapping matches for an infallible search.
///
/// The iterator yields a [`Match`] value until no more matches could be found.
/// If the underlying regex engine returns an error, then a panic occurs.
///
/// The lifetime parameters are as follows:
///
/// * `'h` represents the lifetime of the haystack being searched.
/// * `'c` represents the lifetime of the regex cache. The lifetime of the
/// regex object itself must outlive `'c`.
///
/// This iterator can be created with the [`PikeVM::find_iter`]
/// method.
#[derive(Debug)]
pub struct FindMatches<'h, 'c>(iter::Matches<'h, TryMatchesClosure<'h, 'c>>);

impl<'h, 'c> Iterator for FindMatches<'h, 'c> {
    type Item = Match;

    #[inline]
    fn next(&mut self) -> Option<Match> {
        self.0.next()
    }
}

/// An iterator over all non-overlapping leftmost matches, with their capturing
/// groups, for a particular search.
///
/// The iterator yields a [`Captures`] value until no more matches could be
/// found. If the underlying search returns an error, then this panics.
///
/// The lifetime parameters are as follows:
///
/// * `'h` represents the lifetime of the haystack being searched.
/// * `'c` represents the lifetime of the regex cache. The lifetime of the
/// regex object itself must outlive `'c`.
///
/// This iterator can be created with the [`Regex::captures_iter`]
/// method.
#[derive(Debug)]
pub struct CapturesMatches<'h, 'c> {
    it: iter::Matches<'h, TryMatchesClosure<'h, 'c>>,
    /// In order to avoid re-implementing our own iterator, we store the
    /// capturing groups here and inside the iterator's closure. Once the
    /// closure executes, we clone the capturing group and return it.
    ///
    /// If this sounds like it hurts perf, it probably does, but you aren't
    /// using this iterator if you care about perf because it allocates a
    /// fresh set of capturing groups on each iteration.
    caps: Arc<RefCell<Captures>>,
}

impl<'h, 'c> Iterator for CapturesMatches<'h, 'c> {
    type Item = Captures;

    #[inline]
    fn next(&mut self) -> Option<Captures> {
        self.it.next().map(|_| self.caps.borrow().clone())
    }
}

#[derive(Clone, Debug)]
pub struct Cache {
    stack: Vec<FollowEpsilon>,
    scratch_caps: Vec<Option<NonMaxUsize>>,
    clist: Box<Threads>,
    nlist: Box<Threads>,
}

#[derive(Clone, Debug)]
enum FollowEpsilon {
    StateID(StateID),
    Capture { slot: usize, pos: Slot },
}

type Slot = Option<NonMaxUsize>;

#[derive(Clone, Debug)]
struct Threads {
    set: SparseSet,
    list: Vec<StateID>,
    caps: Vec<Thread>,
    current_slot_len: usize,
}

#[derive(Clone, Debug)]
struct Thread {
    slots: Box<[Option<NonMaxUsize>]>,
}

impl Cache {
    pub fn new(nfa: &NFA) -> Cache {
        Cache {
            stack: vec![],
            scratch_caps: vec![None; nfa.capture_slot_len()],
            clist: Box::new(Threads::new(nfa)),
            nlist: Box::new(Threads::new(nfa)),
        }
    }

    fn clear(&mut self, slot_len: usize) {
        self.stack.clear();
        self.clist.set.clear();
        self.clist.list.clear();
        self.nlist.set.clear();
        self.nlist.list.clear();
        if slot_len != self.clist.current_slot_len {
            self.clist.current_slot_len = slot_len;
            self.nlist.current_slot_len = slot_len;
            self.scratch_caps.resize(slot_len, None);
        }
    }
}

impl Threads {
    fn new(nfa: &NFA) -> Threads {
        let mut threads = Threads {
            set: SparseSet::new(0),
            list: Vec::with_capacity(nfa.states().len()),
            caps: vec![],
            current_slot_len: 0,
        };
        threads.resize(nfa);
        threads
    }

    fn resize(&mut self, nfa: &NFA) {
        self.current_slot_len = nfa.capture_slot_len();
        self.set.resize(nfa.states().len());
        self.caps.resize(nfa.states().len(), Thread::new(nfa));
    }

    fn caps(&mut self, sid: StateID) -> &mut [Slot] {
        &mut self.caps[sid].slots[..self.current_slot_len]
    }
}

impl Thread {
    fn new(nfa: &NFA) -> Thread {
        let slots = vec![None; nfa.capture_slot_len()].into_boxed_slice();
        Thread { slots }
    }
}

fn copy_to_captures(pid: PatternID, slots: &[Slot], caps: &mut Captures) {
    caps.set_pattern(Some(pid));
    for (i, &slot) in slots.iter().enumerate() {
        caps.set_slot(i, slot.map(|s| s.get()));
    }
}

/// A set of counters that "instruments" a PikeVM search. To enable this, you
/// must enable the 'instrument-pikevm' feature. Then run your Rust program
/// with RUST_LOG=regex_automata::nfa::thompson::pikevm=trace set in the
/// environment. The metrics collected will be dumped automatically for every
/// search executed by the PikeVM.
///
/// NOTE: When 'instrument-pikevm' is enabled, it will likely cause an absolute
/// decrease in wall-clock performance, even if the 'trace' log level isn't
/// enabled. (Although, we do try to avoid extra costs when 'trace' isn't
/// enabled.) The main point of instrumentation is to get counts of various
/// events that occur during the PikeVM's execution.
///
/// This is a somewhat hacked together collection of metrics that are useful
/// to gather from a PikeVM search. In particular, it lets us scrutinize the
/// performance profile of a search beyond what general purpose profiling tools
/// give us. Namely, we orient the profiling data around the specific states of
/// the NFA.
///
/// In other words, this lets us see which parts of the NFA graph are most
/// frequently activated. This then provides direction for optimization
/// opportunities.
///
/// The really sad part about this is that it absolutely clutters up the PikeVM
/// implementation. :'( Another approach would be to just manually add this
/// code in whenever I want this kind of profiling data, but it's complicated
/// and tedious enough that I went with this approach... for now.
///
/// When instrumentation is enabled (which also turns on 'logging'), then a
/// `Counters` is initialized for every search and `trace`'d just before the
/// search returns to the caller.
///
/// Tip: When debugging performance problems with the PikeVM, it's best to try
/// to work with an NFA that is as small as possible. Otherwise the state graph
/// is likely to be too big to digest.
#[cfg(feature = "instrument-pikevm")]
#[derive(Clone, Debug)]
struct Counters {
    /// The number of times the NFA is in a particular permutation of states.
    state_sets: alloc::collections::BTreeMap<Vec<StateID>, u64>,
    /// The number of times 'step' is called for a particular state ID (which
    /// indexes this array).
    steps: Vec<u64>,
    /// The number of times an epsilon closure was computed for a state.
    closures: Vec<u64>,
    /// The number of times a particular state ID is pushed on to a stack while
    /// computing an epsilon closure.
    stack_pushes: Vec<u64>,
    /// The number of times a particular state ID is inserted into a sparse set
    /// while computing an epsilon closure.
    set_inserts: Vec<u64>,
}

#[cfg(feature = "instrument-pikevm")]
impl Counters {
    fn empty() -> Counters {
        Counters {
            state_sets: alloc::collections::BTreeMap::new(),
            steps: vec![],
            closures: vec![],
            stack_pushes: vec![],
            set_inserts: vec![],
        }
    }

    fn reset(&mut self, nfa: &NFA) {
        let len = nfa.states().len();

        self.state_sets.clear();

        self.steps.clear();
        self.steps.resize(len, 0);

        self.closures.clear();
        self.closures.resize(len, 0);

        self.stack_pushes.clear();
        self.stack_pushes.resize(len, 0);

        self.set_inserts.clear();
        self.set_inserts.resize(len, 0);
    }

    fn eprint(&self, nfa: &NFA) {
        trace!("===== START PikeVM Instrumentation Output =====");
        // We take the top-K most occurring state sets. Otherwise the output
        // is likely to be overwhelming. And we probably only care about the
        // most frequently occuring ones anyway.
        const LIMIT: usize = 20;
        let mut set_counts =
            self.state_sets.iter().collect::<Vec<(&Vec<StateID>, &u64)>>();
        set_counts.sort_by_key(|(_, &count)| core::cmp::Reverse(count));
        trace!("## PikeVM frequency of state sets (top {})", LIMIT);
        for (set, count) in set_counts.iter().take(LIMIT) {
            trace!("{:?}: {}", set, count);
        }
        if set_counts.len() > LIMIT {
            trace!(
                "... {} sets omitted (out of {} total)",
                set_counts.len() - LIMIT,
                set_counts.len(),
            );
        }

        trace!("");
        trace!("## PikeVM total frequency of events");
        trace!(
            "steps: {}, closures: {}, stack-pushes: {}, set-inserts: {}",
            self.steps.iter().copied().sum::<u64>(),
            self.closures.iter().copied().sum::<u64>(),
            self.stack_pushes.iter().copied().sum::<u64>(),
            self.set_inserts.iter().copied().sum::<u64>(),
        );

        trace!("");
        trace!("## PikeVM frequency of events broken down by state");
        for sid in 0..self.steps.len() {
            trace!(
                "{:06}: steps: {}, closures: {}, \
                 stack-pushes: {}, set-inserts: {}",
                sid,
                self.steps[sid],
                self.closures[sid],
                self.stack_pushes[sid],
                self.set_inserts[sid],
            );
        }

        trace!("");
        trace!("## NFA debug display");
        trace!("{:?}", nfa);
        trace!("===== END PikeVM Instrumentation Output =====");
    }

    fn record_state_set(&mut self, set: &[StateID]) {
        let set = set.iter().copied().collect::<Vec<StateID>>();
        *self.state_sets.entry(set).or_insert(0) += 1;
    }

    fn record_step(&mut self, sid: StateID) {
        self.steps[sid] += 1;
    }

    fn record_closure(&mut self, sid: StateID) {
        self.closures[sid] += 1;
    }

    fn record_stack_push(&mut self, sid: StateID) {
        self.stack_pushes[sid] += 1;
    }

    fn record_set_insert(&mut self, sid: StateID) {
        self.set_inserts[sid] += 1;
    }
}
