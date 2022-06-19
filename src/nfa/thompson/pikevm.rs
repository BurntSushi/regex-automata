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
        prefilter::Prefilter,
        search::{Input, Match, MatchError, MatchKind, PatternSet},
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
    pre: Option<Option<Arc<dyn Prefilter>>>,
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

    pub fn prefilter(mut self, pre: Option<Arc<dyn Prefilter>>) -> Config {
        self.pre = Some(pre);
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

    pub fn get_prefilter(&self) -> Option<&dyn Prefilter> {
        self.pre.as_ref().unwrap_or(&None).as_deref()
    }

    pub(crate) fn overwrite(&self, o: Config) -> Config {
        Config {
            anchored: o.anchored.or(self.anchored),
            match_kind: o.match_kind.or(self.match_kind),
            utf8: o.utf8.or(self.utf8),
            pre: o.pre.or_else(|| self.pre.clone()),
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
        // relies on them in order to report match locations. However, in
        // the special case of an NFA with no patterns, it is allowed, since
        // no matches can ever be produced. And importantly, an NFA with no
        // patterns has no capturing groups anyway, so this is necessary to
        // permit the PikeVM to work with regexes with zero patterns.
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

    pub fn create_input<'h, 'p, H: ?Sized + AsRef<[u8]>>(
        &'p self,
        haystack: &'h H,
    ) -> Input<'h, 'p> {
        let c = self.get_config();
        Input::new(haystack.as_ref())
            .prefilter(c.get_prefilter())
            .utf8(c.get_utf8())
    }

    pub fn create_cache(&self) -> Cache {
        Cache::new(self)
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
        let input = self.create_input(haystack.as_ref()).earliest(true);
        let mut caps = Captures::empty(self.nfa.clone());
        self.search(cache, &input, &mut caps);
        caps.is_match()
    }

    #[inline]
    pub fn find<H: AsRef<[u8]>>(
        &self,
        cache: &mut Cache,
        haystack: H,
        caps: &mut Captures,
    ) {
        let input = self.create_input(haystack.as_ref());
        self.search(cache, &input, caps)
    }

    #[inline]
    pub fn find_iter<'r, 'c, 'h, H: AsRef<[u8]> + ?Sized>(
        &'r self,
        cache: &'c mut Cache,
        haystack: &'h H,
    ) -> FindMatches<'r, 'c, 'h> {
        let input = self.create_input(haystack.as_ref());
        let caps = Captures::new_for_matches_only(self.get_nfa().clone());
        let it = iter::Searcher::new(input);
        FindMatches { re: self, cache, caps, it }
    }

    #[inline]
    pub fn captures_iter<'r, 'c, 'h, H: AsRef<[u8]> + ?Sized>(
        &'r self,
        cache: &'c mut Cache,
        haystack: &'h H,
    ) -> CapturesMatches<'r, 'c, 'h> {
        let input = self.create_input(haystack.as_ref());
        let caps = self.create_captures();
        let it = iter::Searcher::new(input);
        CapturesMatches { re: self, cache, caps, it }
    }

    #[inline]
    pub fn search(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        caps: &mut Captures,
    ) {
        self.search_imp(cache, input, caps);
        let m = match caps.get_match() {
            None => return,
            Some(m) => m,
        };
        if m.is_empty() {
            input
                .skip_empty_utf8_splits(m, |search| {
                    self.search_imp(cache, search, caps);
                    Ok(caps.get_match())
                })
                .unwrap();
        }
    }

    #[inline]
    pub fn which_overlapping_matches(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        patset: &mut PatternSet,
    ) {
        self.which_overlapping_imp(cache, input, patset)
    }
}

impl PikeVM {
    fn search_imp(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        caps: &mut Captures,
    ) {
        caps.set_pattern(None);
        cache.setup_search(caps.slot_len());
        // Why do we even care about this? Well, in our 'Captures'
        // representation, we use usize::MAX as a sentinel to indicate "no
        // match." This isn't problematic so long as our haystack doesn't have
        // a maximal length. Byte slices are guaranteed by Rust to have a
        // length that fits into isize, and so this assert should always pass.
        // But we put it here to make our assumption explicit.
        assert!(
            input.haystack().len() < core::usize::MAX,
            "byte slice lengths must be less than usize MAX",
        );
        instrument!(|c| c.reset(&self.nfa));

        let allmatches =
            self.config.get_match_kind().continue_past_first_match();
        let anchored = self.config.get_anchored()
            || self.nfa.is_always_start_anchored()
            || input.get_pattern().is_some();
        let start_id = match input.get_pattern() {
            // We always use the anchored starting state here, even if doing an
            // unanchored search. The "unanchored" part of it is implemented
            // in the loop below, by computing the epsilon closure from the
            // anchored starting state whenever the current state set list is
            // empty.
            None => self.nfa.start_anchored(),
            Some(pid) => self.nfa.start_pattern(pid),
        };

        let Cache { ref mut stack, ref mut curr, ref mut next } = cache;
        let mut at = input.start();
        while at <= input.end() {
            if curr.set.is_empty() {
                if (caps.is_match() && !allmatches)
                    || (anchored && at > input.start())
                {
                    break;
                }
            }
            if curr.set.is_empty() || (!anchored && !caps.is_match()) {
                let slots = next.slot_table.for_state(start_id);
                self.epsilon_closure(stack, slots, curr, start_id, input, at);
            }
            if self.steps(stack, curr, next, input, at, caps) {
                if input.get_earliest() {
                    break;
                }
            }
            at += 1;
            core::mem::swap(curr, next);
            next.set.clear();
        }
        instrument!(|c| c.eprint(&self.nfa));
    }

    fn which_overlapping_imp(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        patset: &mut PatternSet,
    ) {
        assert!(
            input.haystack().len() < core::usize::MAX,
            "byte slice lengths must be less than usize MAX",
        );
        instrument!(|c| c.reset(&self.nfa));
        cache.setup_search(0);

        let anchored = self.config.get_anchored()
            || self.nfa.is_always_start_anchored()
            || input.get_pattern().is_some();
        let start_id = match input.get_pattern() {
            None => self.nfa.start_anchored(),
            Some(pid) => self.nfa.start_pattern(pid),
        };

        let Cache { ref mut stack, ref mut curr, ref mut next } = cache;
        let mut at = input.start();
        while at <= input.end() {
            if anchored && curr.set.is_empty() && at > input.start() {
                break;
            }
            if curr.set.is_empty() || !anchored {
                let slots = &mut [];
                self.epsilon_closure(stack, slots, curr, start_id, input, at);
            }
            self.steps_overlapping(stack, curr, next, input, at, patset);
            // If we found a match and filled our set, then there is no more
            // additional info that we can provide. Thus, we can quit.
            if patset.is_full() {
                break;
            }
            at += 1;
            core::mem::swap(curr, next);
            next.set.clear();
        }
        instrument!(|c| c.eprint(&self.nfa));
    }

    #[inline(always)]
    fn steps(
        &self,
        stack: &mut Vec<FollowEpsilon>,
        curr: &mut ActiveStates,
        next: &mut ActiveStates,
        input: &Input<'_, '_>,
        at: usize,
        caps: &mut Captures,
    ) -> bool {
        let slot_len = caps.slot_len();
        let mut matched = false;
        instrument!(|c| c.record_state_set(&curr.set));
        let ActiveStates { ref set, ref mut slot_table } = *curr;
        for sid in set.iter() {
            let pid = match self.step(stack, slot_table, next, sid, input, at)
            {
                None => continue,
                Some(pid) => pid,
            };
            matched = true;
            let slots = slot_table.for_state(sid);
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
        curr: &mut ActiveStates,
        next: &mut ActiveStates,
        input: &Input<'_, '_>,
        at: usize,
        patset: &mut PatternSet,
    ) {
        instrument!(|c| c.record_state_set(&curr.set));
        let ActiveStates { ref set, ref mut slot_table } = *curr;
        for sid in set.iter() {
            let pid = match self.step(stack, slot_table, next, sid, input, at)
            {
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
        curr_slot_table: &mut SlotTable,
        next: &mut ActiveStates,
        sid: StateID,
        input: &Input<'_, '_>,
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
                if trans.matches(input.haystack(), at) {
                    let slots = curr_slot_table.for_state(sid);
                    // OK because 'at <= haystack.len() < usize::MAX', so
                    // adding 1 will never wrap.
                    let at = at.wrapping_add(1);
                    self.epsilon_closure(
                        stack, slots, next, trans.next, input, at,
                    );
                }
                None
            }
            State::Sparse(ref sparse) => {
                if let Some(next_sid) = sparse.matches(input.haystack(), at) {
                    let slots = curr_slot_table.for_state(sid);
                    // OK because 'at <= haystack.len() < usize::MAX', so
                    // adding 1 will never wrap.
                    let at = at.wrapping_add(1);
                    self.epsilon_closure(
                        stack, slots, next, next_sid, input, at,
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
        curr_slots: &mut [Option<NonMaxUsize>],
        next: &mut ActiveStates,
        sid: StateID,
        input: &Input<'_, '_>,
        at: usize,
    ) {
        instrument!(|c| {
            c.record_closure(sid);
            c.record_stack_push(sid);
        });
        stack.push(FollowEpsilon::Explore(sid));
        while let Some(frame) = stack.pop() {
            match frame {
                FollowEpsilon::RestoreCapture { slot, offset: pos } => {
                    curr_slots[slot] = pos;
                }
                FollowEpsilon::Explore(sid) => {
                    self.epsilon_closure_step(
                        stack, curr_slots, next, sid, input, at,
                    );
                }
            }
        }
    }

    #[inline(always)]
    fn epsilon_closure_step(
        &self,
        stack: &mut Vec<FollowEpsilon>,
        curr_slots: &mut [Option<NonMaxUsize>],
        next: &mut ActiveStates,
        mut sid: StateID,
        input: &Input<'_, '_>,
        at: usize,
    ) {
        loop {
            instrument!(|c| c.record_set_insert(sid));
            if !next.set.insert(sid) {
                return;
            }
            match *self.nfa.state(sid) {
                State::Fail
                | State::Match { .. }
                | State::ByteRange { .. }
                | State::Sparse { .. } => {
                    next.slot_table.for_state(sid).copy_from_slice(curr_slots);
                    return;
                }
                State::Look { look, next } => {
                    if !look.matches(input.haystack(), at) {
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
                            .map(FollowEpsilon::Explore),
                    );
                }
                State::BinaryUnion { alt1, alt2 } => {
                    sid = alt1;
                    instrument!(|c| c.record_stack_push(sid));
                    stack.push(FollowEpsilon::Explore(alt2));
                }
                State::Capture { next, slot } => {
                    if slot < curr_slots.len() {
                        instrument!(|c| c.record_stack_push(sid));
                        stack.push(FollowEpsilon::RestoreCapture {
                            slot,
                            offset: curr_slots[slot],
                        });
                        // OK because length of a slice must fit into an isize.
                        curr_slots[slot] = Some(NonMaxUsize::new(at).unwrap());
                    }
                    sid = next;
                }
            }
        }
    }
}

/// An iterator over all non-overlapping matches for a particular search.
///
/// The iterator yields a [`Match`] value until no more matches could be found.
/// If the underlying regex engine returns an error, then a panic occurs.
///
/// The lifetime parameters are as follows:
///
/// * `'r` represents the lifetime of the PikeVM.
/// * `'c` represents the lifetime of the PikeVM's cache.
/// * `'h` represents the lifetime of the haystack being searched.
///
/// This iterator can be created with the [`PikeVM::find_iter`] method.
#[derive(Debug)]
pub struct FindMatches<'r, 'c, 'h> {
    re: &'r PikeVM,
    cache: &'c mut Cache,
    caps: Captures,
    it: iter::Searcher<'h, 'r>,
}

impl<'r, 'c, 'h> Iterator for FindMatches<'r, 'c, 'h> {
    type Item = Match;

    #[inline]
    fn next(&mut self) -> Option<Match> {
        // Splitting 'self' apart seems necessary to appease borrowck.
        let FindMatches { re, ref mut cache, ref mut caps, ref mut it } =
            *self;
        it.advance(|input| {
            re.search(cache, input, caps);
            Ok(caps.get_match())
        })
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
/// * `'r` represents the lifetime of the PikeVM.
/// * `'c` represents the lifetime of the PikeVM's cache.
/// * `'h` represents the lifetime of the haystack being searched.
///
/// This iterator can be created with the [`Regex::captures_iter`] method.
#[derive(Debug)]
pub struct CapturesMatches<'r, 'c, 'h> {
    re: &'r PikeVM,
    cache: &'c mut Cache,
    caps: Captures,
    it: iter::Searcher<'h, 'r>,
}

impl<'r, 'c, 'h> Iterator for CapturesMatches<'r, 'c, 'h> {
    type Item = Captures;

    #[inline]
    fn next(&mut self) -> Option<Captures> {
        // Splitting 'self' apart seems necessary to appease borrowck.
        let CapturesMatches { re, ref mut cache, ref mut caps, ref mut it } =
            *self;
        it.advance(|input| {
            re.search(cache, input, caps);
            Ok(caps.get_match())
        });
        if caps.is_match() {
            Some(caps.clone())
        } else {
            None
        }
    }
}

/// A cache represents mutable state that a [`PikeVM`] requires during a
/// search.
///
/// For a given [`PikeVM`], its corresponding cache may be created either via
/// [`PikeVM::create_cache`], or via [`Cache::new`]. They are equivalent in
/// every way, except the former does not require explicitly importing `Cache`.
///
/// A particular `Cache` is coupled with the [`PikeVM`] from which it
/// was created. It may only be used with that `PikeVM`. A cache and its
/// allocations may be re-purposed via [`Cache::reset`], in which case, it can
/// only be used with the new `PikeVM` (and not the old one).
#[derive(Clone, Debug)]
pub struct Cache {
    /// Stack used while computing epsilon closure. This effectively lets us
    /// move what is more naturally expressed through recursion to a stack
    /// on the heap.
    stack: Vec<FollowEpsilon>,
    /// The current active states being explored for the current byte in the
    /// haystack.
    curr: ActiveStates,
    /// The next set of states we're building that will be explored for the
    /// next byte in the haystack.
    next: ActiveStates,
}

impl Cache {
    /// Create a new [`PikeVM`] cache.
    ///
    /// A potentially more convenient routine to create a cache is
    /// [`PikeVM::create_cache`], as it does not require also importing the
    /// `Cache` type.
    ///
    /// If you want to reuse the returned `Cache` with some other `PikeVM`,
    /// then you must call [`Cache::reset`] with the desired `PikeVM`.
    pub fn new(vm: &PikeVM) -> Cache {
        Cache {
            stack: vec![],
            curr: ActiveStates::new(vm),
            next: ActiveStates::new(vm),
        }
    }

    /// Reset this cache such that it can be used for searching with different
    /// [`PikeVM`].
    ///
    /// A cache reset permits reusing memory already allocated in this cache
    /// with a different `PikeVM`.
    ///
    /// # Example
    ///
    /// This shows how to re-purpose a cache for use with a different `PikeVM`.
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::pikevm::PikeVM, Match};
    ///
    /// let re1 = PikeVM::new(r"\w")?;
    /// let re2 = PikeVM::new(r"\W")?;
    ///
    /// let mut cache = re1.create_cache();
    /// assert_eq!(
    ///     Some(Match::must(0, 0..2)),
    ///     re1.find_iter(&mut cache, "Δ").next(),
    /// );
    ///
    /// // Using 'cache' with re2 is not allowed. It may result in panics or
    /// // incorrect results. In order to re-purpose the cache, we must reset
    /// // it with the PikeVM we'd like to use it with.
    /// //
    /// // Similarly, after this reset, using the cache with 're1' is also not
    /// // allowed.
    /// cache.reset(&re2);
    /// assert_eq!(
    ///     Some(Match::must(0, 0..3)),
    ///     re2.find_iter(&mut cache, "☃").next(),
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reset(&mut self, vm: &PikeVM) {
        self.curr.reset(vm);
        self.next.reset(vm);
    }

    /// Returns the heap memory usage, in bytes, of this cache.
    ///
    /// This does **not** include the stack size used up by this cache. To
    /// compute that, use `std::mem::size_of::<Cache>()`.
    pub fn memory_usage(&self) -> usize {
        use core::mem::size_of;
        (self.stack.len() * size_of::<FollowEpsilon>())
            + self.curr.memory_usage()
            + self.next.memory_usage()
    }

    /// Clears this cache. This should be called at the start of every search
    /// to ensure we start with a clean slate.
    ///
    /// This also sets the length of the capturing groups used in the current
    /// search. This permits an optimization where by 'SlotTable::for_state'
    /// only returns the number of slots equivalent to the number of slots
    /// given in the 'Captures' value. This may be less than the total number
    /// of possible slots, e.g., when one only wants to track overall match
    /// offsets. This in turn permits less copying of capturing group spans
    /// in the PikeVM.
    fn setup_search(&mut self, captures_slot_len: usize) {
        self.stack.clear();
        self.curr.setup_search(captures_slot_len);
        self.next.setup_search(captures_slot_len);
    }
}

/// A set of active states used to "simulate" the execution of an NFA via the
/// PikeVM.
///
/// There are two sets of these used during NFA simulation. One set corresponds
/// to the "current" set of states being traversed for the current position
/// in a haystack. The other set corresponds to the "next" set of states being
/// built, which will become the new "current" set for the next position in the
/// haystack. These two sets correspond to CLIST and NLIST in Thompson's
/// original paper regexes: https://dl.acm.org/doi/pdf/10.1145/363347.363387
///
/// In addition to representing a set of NFA states, this also maintains slot
/// values for each state. These slot values are what turn the NFA simulation
/// into the "Pike VM." Namely, they track capturing group values for each
/// state. During the computation of epsilon closure, we copy slot values from
/// states in the "current" set to the "next" set. Eventually, once a match
/// is found, the slot values for that match state are what we write to the
/// caller provided 'Captures' value.
#[derive(Clone, Debug)]
struct ActiveStates {
    /// The set of active NFA states. This set preserves insertion order, which
    /// is critical for simulating the match semantics of backtracking regex
    /// engines.
    set: SparseSet,
    /// The slots for every NFA state, where each slot stores a (possibly
    /// absent) offset. Every capturing group has two slots. One for a start
    /// offset and one for an end offset.
    slot_table: SlotTable,
}

impl ActiveStates {
    /// Create a new set of active states for the given PikeVM. The active
    /// states returned may only be used with the given PikeVM. (Use 'reset'
    /// to re-purpose the allocation for a different PikeVM.)
    fn new(vm: &PikeVM) -> ActiveStates {
        let mut active = ActiveStates {
            set: SparseSet::new(0),
            slot_table: SlotTable::new(),
        };
        active.reset(vm);
        active
    }

    /// Reset this set of active states such that it can be used with the given
    /// PikeVM (and only that PikeVM).
    fn reset(&mut self, vm: &PikeVM) {
        let nfa = vm.get_nfa();
        self.set.resize(vm.get_nfa().states().len());
        self.slot_table.reset(vm);
    }

    /// Return the heap memory usage, in bytes, used by this set of active
    /// states.
    ///
    /// This does not include the stack size of this value.
    fn memory_usage(&self) -> usize {
        self.set.memory_usage() + self.slot_table.memory_usage()
    }

    /// Setup this set of active states for a new search. The given slot
    /// length should be the number of slots in a caller provided 'Captures'
    /// (and may be zero).
    fn setup_search(&mut self, captures_slot_len: usize) {
        self.set.clear();
        self.slot_table.setup_search(captures_slot_len);
    }
}

/// A table of slots, where each row represent a state in an NFA. Thus, the
/// table has room for storing slots for every single state in an NFA.
///
/// This table is represented with a single contiguous allocation. In general,
/// the notion of "capturing group" doesn't really exist at this level of
/// abstraction, hence the name "slot" instead. (Indeed, every capturing group
/// maps to a pair of slots, one for the start offset and one for the end
/// offset.) Slots are indexed by the 'Captures' NFA state.
///
/// N.B. Not every state actually needs a row of slots. Namely, states that
/// only have epsilon transitions currently never have anything written to
/// their rows in this table. Thus, the table is somewhat wasteful in its heap
/// usage. However, it is important to maintain fast random access by state
/// ID, which means one giant table tends to work well. RE2 takes a different
/// approach here and allocates each row as its own reference counted thing.
/// I explored such a strategy at one point here, but couldn't get it to work
/// well using entirely safe code. (To the ambitious reader: I encourage you to
/// re-litigate that experiment.) I very much wanted to stick to safe code, but
/// could be convinced otherwise if there was a solid argument and the safety
/// was encapsulated well.
#[derive(Clone, Debug)]
struct SlotTable {
    /// The actual table of offsets.
    table: Vec<Option<NonMaxUsize>>,
    /// The number of slots per state, i.e., the table's stride or the length
    /// of each row.
    slots_per_state: usize,
    /// The number of slots in the caller-provided 'Captures' value for the
    /// current search. Setting this to 'slots_per_state' is always correct,
    /// but may be wasteful.
    slots_for_captures: usize,
}

impl SlotTable {
    /// Create a new slot table.
    ///
    /// One should call 'reset' with the corresponding PikeVM before use.
    fn new() -> SlotTable {
        SlotTable { table: vec![], slots_for_captures: 0, slots_per_state: 0 }
    }

    /// Reset this slot table such that it can be used with the given PikeVM
    /// (and only that PikeVM).
    fn reset(&mut self, vm: &PikeVM) {
        let nfa = vm.get_nfa();
        self.slots_per_state = nfa.capture_slot_len();
        // This is always correct, but may be reduced for a particular search
        // if a 'Captures' has fewer slots, e.g., none at all or only slots
        // for tracking the overall match instead of all slots for every
        // group.
        self.slots_for_captures = nfa.capture_slot_len();
        self.table.resize(nfa.states().len() * self.slots_per_state, None);
    }

    /// Return the heap memory usage, in bytes, used by this slot table.
    ///
    /// This does not include the stack size of this value.
    fn memory_usage(&self) -> usize {
        self.table.len() * core::mem::size_of::<Option<NonMaxUsize>>()
    }

    /// Perform any per-search setup for this slot table.
    ///
    /// In particular, this sets the length of the number of slots used in the
    /// 'Captures' given by the caller (if any at all). This number may be
    /// smaller than the total number of slots available, e.g., when the caller
    /// is only interested in tracking the overall match and not the spans of
    /// every matching capturing group. Only tracking the overall match can
    /// save a substantial amount of time copying capturing spans during a
    /// search.
    fn setup_search(&mut self, captures_slot_len: usize) {
        self.slots_for_captures = captures_slot_len;
    }

    /// Return a mutable slice of the slots for the given state.
    ///
    /// Note that the length of the slice returned may be less than the total
    /// number of slots available for this state. In particular, the length
    /// always matches the number of slots indicated via 'setup_search'.
    fn for_state(&mut self, sid: StateID) -> &mut [Option<NonMaxUsize>] {
        let i = sid.as_usize() * self.slots_per_state;
        &mut self.table[i..i + self.slots_for_captures]
    }
}

/// Represents a stack frame for use while computing an epsilon closure.
///
/// (An "epsilon closure" refers to the set of reachable NFA states from a
/// single state without consuming any input. That is, the set of all epsilon
/// transitions not only from that single state, but from every other state
/// reachable by an epsilon transition as well. This is why it's called a
/// "closure." Computing an epsilon closure is also done during DFA
/// determinization! Compare and contrast the epsilon closure here in this
/// PikeVM and the one used for determinization in crate::util::determinize.)
///
/// Computing the epsilon closure in a Thompson NFA proceeds via a depth
/// first traversal over all epsilon transitions from a particular state.
/// (A depth first traversal is important because it emulates the same priority
/// of matches that is typically found in backtracking regex engines.) This
/// depth first traversal is naturally expressed using recursion, but to avoid
/// a call stack size proportional to the size of a regex, we put our stack on
/// the heap instead.
///
/// This stack thus consists of call frames. The typical call frame is
/// `Explore`, which instructs epsilon closure to explore the epsilon
/// transitions from that state. (Subsequent epsilon transitions are then
/// pushed on to the stack as more `Explore` frames.) If the state ID being
/// explored has no epsilon transitions, then the capturing group slots are
/// copied from the original state that sparked the epsilon closure (from the
/// 'step' routine) to the state ID being explored. This way, capturing group
/// slots are forwarded from the previous state to the next.
///
/// The other stack frame, `RestoreCaptures`, instructs the epsilon closure to
/// set the position for a particular slot back to some particular offset. This
/// frame is pushed when `Explore` sees a `Capture` transition. `Explore` will
/// set the offset of the slot indicated in `Capture` to the current offset,
/// and then push the old offset on to the stack as a `RestoreCapture` frame.
/// Thus, the new offset is only used until the epsilon closure reverts back to
/// the `RestoreCapture` frame. In effect, this gives the `Capture` epsilon
/// transition its "scope" to only states that come "after" it during depth
/// first traversal.
#[derive(Clone, Debug)]
enum FollowEpsilon {
    /// Explore the epsilon transitions from a state ID.
    Explore(StateID),
    /// Reset the given `slot` to the given `offset` (which might be `None`).
    RestoreCapture { slot: usize, offset: Option<NonMaxUsize> },
}

/// Write the given pattern ID and slot values to the given `Captures`.
///
/// This is generally what you want to use once a match has been found. This
/// copies the internal slot data to the public `Captures` type.
fn copy_to_captures(
    pid: PatternID,
    slots: &[Option<NonMaxUsize>],
    caps: &mut Captures,
) {
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

    fn record_state_set(&mut self, set: &SparseSet) {
        let set = set.iter().collect::<Vec<StateID>>();
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
