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
        matchtypes::{Match, MatchKind},
        nonmax::NonMaxUsize,
        prefilter::{self, Prefilter},
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
            // Apparently I need to write an explicit type here otherwise
            // inference for the closure fails and using the macro becomes
            // tedious. There's no particular reason to use a dyn FnMut here,
            // but how else do we name the type? Anyway, this worked and this
            // is just for debugging/perf-profiling.
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

#[derive(Clone, Copy, Debug, Default)]
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

    pub(crate) fn overwrite(self, o: Config) -> Config {
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
        Ok(PikeVM { config: self.config, nfa })
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
        Cache::new(self.nfa())
    }

    pub fn create_captures(&self) -> Captures {
        Captures::new(self.nfa().clone())
    }

    pub fn nfa(&self) -> &NFA {
        &self.nfa
    }
}

impl PikeVM {
    pub fn is_match(&self, cache: &mut Cache, haystack: &[u8]) -> bool {
        let mut caps = Captures::empty(self.nfa.clone());
        self.find_leftmost(cache, haystack, &mut caps);
        caps.is_match()
    }

    pub fn find_earliest(
        &self,
        cache: &mut Cache,
        haystack: &[u8],
        caps: &mut Captures,
    ) {
        self.find_earliest_at(
            cache,
            None,
            None,
            haystack,
            0,
            haystack.len(),
            caps,
        )
    }

    pub fn find_leftmost(
        &self,
        cache: &mut Cache,
        haystack: &[u8],
        caps: &mut Captures,
    ) {
        self.find_leftmost_at(
            cache,
            None,
            None,
            haystack,
            0,
            haystack.len(),
            caps,
        )
    }

    pub fn find_overlapping(
        &self,
        cache: &mut Cache,
        haystack: &[u8],
        state: &mut OverlappingState,
        caps: &mut Captures,
    ) {
        self.find_overlapping_at(
            cache,
            None,
            None,
            haystack,
            0,
            haystack.len(),
            state,
            caps,
        )
    }

    pub fn find_earliest_iter<'r, 'c, 't>(
        &'r self,
        cache: &'c mut Cache,
        haystack: &'t [u8],
    ) -> FindEarliestMatches<'r, 'c, 't> {
        FindEarliestMatches::new(self, cache, haystack)
    }

    pub fn find_leftmost_iter<'r, 'c, 't>(
        &'r self,
        cache: &'c mut Cache,
        haystack: &'t [u8],
    ) -> FindLeftmostMatches<'r, 'c, 't> {
        FindLeftmostMatches::new(self, cache, haystack)
    }

    pub fn find_overlapping_iter<'r, 'c, 't>(
        &'r self,
        cache: &'c mut Cache,
        haystack: &'t [u8],
    ) -> FindOverlappingMatches<'r, 'c, 't> {
        FindOverlappingMatches::new(self, cache, haystack)
    }

    pub fn captures_earliest_iter<'r, 'c, 't>(
        &'r self,
        cache: &'c mut Cache,
        haystack: &'t [u8],
    ) -> CapturesEarliestMatches<'r, 'c, 't> {
        CapturesEarliestMatches::new(self, cache, haystack)
    }

    pub fn captures_leftmost_iter<'r, 'c, 't>(
        &'r self,
        cache: &'c mut Cache,
        haystack: &'t [u8],
    ) -> CapturesLeftmostMatches<'r, 'c, 't> {
        CapturesLeftmostMatches::new(self, cache, haystack)
    }

    pub fn captures_overlapping_iter<'r, 'c, 't>(
        &'r self,
        cache: &'c mut Cache,
        haystack: &'t [u8],
    ) -> CapturesOverlappingMatches<'r, 'c, 't> {
        CapturesOverlappingMatches::new(self, cache, haystack)
    }

    pub fn find_earliest_at(
        &self,
        cache: &mut Cache,
        pre: Option<&mut prefilter::Scanner>,
        pattern_id: Option<PatternID>,
        haystack: &[u8],
        start: usize,
        end: usize,
        caps: &mut Captures,
    ) {
        self.find_fwd(true, cache, pre, pattern_id, haystack, start, end, caps)
    }

    pub fn find_leftmost_at(
        &self,
        cache: &mut Cache,
        pre: Option<&mut prefilter::Scanner>,
        pattern_id: Option<PatternID>,
        haystack: &[u8],
        start: usize,
        end: usize,
        caps: &mut Captures,
    ) {
        self.find_fwd(
            false, cache, pre, pattern_id, haystack, start, end, caps,
        )
    }

    pub fn find_overlapping_at(
        &self,
        cache: &mut Cache,
        pre: Option<&mut prefilter::Scanner>,
        pattern_id: Option<PatternID>,
        haystack: &[u8],
        start: usize,
        end: usize,
        state: &mut OverlappingState,
        caps: &mut Captures,
    ) {
        self.find_overlapping_fwd(
            cache, pre, pattern_id, haystack, start, end, state, caps,
        )
    }
}

impl PikeVM {
    fn find_fwd(
        &self,
        earliest: bool,
        cache: &mut Cache,
        pre: Option<&mut prefilter::Scanner>,
        pattern_id: Option<PatternID>,
        haystack: &[u8],
        start: usize,
        end: usize,
        caps: &mut Captures,
    ) {
        // Why do we even care about this? Well, in our 'Captures'
        // representation, we use usize::MAX as a sentinel to indicate "no
        // match." This isn't problematic so long as our haystack doesn't have
        // a maximal length. Byte slices are guaranteed by Rust to have a
        // length that fits into isize, and so this assert should always pass.
        // But we put it here to make our assumption explicit.
        assert!(
            haystack.len() < core::usize::MAX,
            "byte slice lengths must be less than usize MAX",
        );
        instrument!(|c| c.reset(&self.nfa));
        caps.set_pattern(None);
        cache.clear(caps.slot_len());

        let match_kind = self.config.get_match_kind();
        let anchored = self.config.get_anchored()
            || self.nfa.is_always_start_anchored()
            || pattern_id.is_some();
        let start_id = match pattern_id {
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
        let mut at = start;
        while at <= end {
            if clist.set.is_empty() {
                if caps.is_match() || (anchored && at > start) {
                    break;
                }
            }
            if clist.set.is_empty() || (!anchored && !caps.is_match()) {
                self.epsilon_closure(
                    stack,
                    clist,
                    scratch_caps,
                    start_id,
                    haystack,
                    at,
                );
            }
            if self.steps(stack, clist, nlist, haystack, at, caps) && earliest
            {
                break;
            }
            at += 1;
            core::mem::swap(clist, nlist);
            nlist.set.clear();
        }
        instrument!(|c| c.eprint(&self.nfa));
    }

    fn find_overlapping_fwd(
        &self,
        cache: &mut Cache,
        pre: Option<&mut prefilter::Scanner>,
        pattern_id: Option<PatternID>,
        haystack: &[u8],
        start: usize,
        end: usize,
        state: &mut OverlappingState,
        caps: &mut Captures,
    ) {
        // NOTE: See 'find_fwd' for some commentary on this routine. We don't
        // duplicate the comments here to avoid them getting out of sync.

        assert!(
            haystack.len() < core::usize::MAX,
            "byte slice lengths must be less than usize MAX",
        );
        instrument!(|c| c.reset(&self.nfa));
        caps.set_pattern(None);
        if state.step_index.is_none() {
            cache.clear(caps.slot_len());
        }

        let match_kind = self.config.get_match_kind();
        let anchored = self.config.get_anchored()
            || self.nfa.is_always_start_anchored()
            || pattern_id.is_some();
        let start_id = match pattern_id {
            None => self.nfa.start_anchored(),
            Some(pid) => self.nfa.start_pattern(pid),
        };

        let Cache {
            ref mut stack,
            ref mut scratch_caps,
            ref mut clist,
            ref mut nlist,
        } = cache;
        let mut at = start;
        while at <= end {
            if anchored && clist.set.is_empty() && at > start {
                break;
            }
            if state.step_index.is_none()
                && !state.matched
                && (clist.set.is_empty() || !anchored)
            {
                self.epsilon_closure(
                    stack,
                    clist,
                    scratch_caps,
                    start_id,
                    haystack,
                    at,
                );
            }
            if self.steps_overlapping(
                stack, clist, nlist, haystack, at, state, caps,
            ) {
                break;
            }
            at += 1;
            core::mem::swap(clist, nlist);
            nlist.set.clear();
            nlist.list.clear();
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
        state: &mut OverlappingState,
        caps: &mut Captures,
    ) -> bool {
        instrument!(|c| c.record_state_set(&clist.list));
        let index = state.step_index.take().unwrap_or(0);
        for (i, &sid) in clist.list.iter().enumerate().skip(index) {
            let slots = &mut clist.caps[sid].slots[..caps.slot_len()];
            let pid = match self.step(stack, nlist, slots, sid, haystack, at) {
                None => continue,
                Some(pid) => pid,
            };
            copy_to_captures(pid, slots, caps);
            state.step_index = Some(i.checked_add(1).unwrap());
            state.matched =
                !self.config.get_match_kind().continue_past_first_match();
            return true;
        }
        false
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

/// An iterator over all non-overlapping leftmost matches for a particular
/// infallible search.
///
/// The iterator yields a [`Match`] value until no more matches could be
/// found. If the underlying search returns an error, then this panics.
///
/// The lifetime variables are as follows:
///
/// * `'r` is the lifetime of the regular expression itself.
/// * `'c` is the lifetime of the mutable cache used during search.
/// * `'t` is the lifetime of the text being searched.
#[derive(Debug)]
pub struct FindEarliestMatches<'r, 'c, 't> {
    vm: &'r PikeVM,
    cache: &'c mut Cache,
    captures: Captures,
    text: &'t [u8],
    last_end: usize,
    last_match: Option<usize>,
}

impl<'r, 'c, 't> FindEarliestMatches<'r, 'c, 't> {
    fn new(
        vm: &'r PikeVM,
        cache: &'c mut Cache,
        text: &'t [u8],
    ) -> FindEarliestMatches<'r, 'c, 't> {
        let captures = Captures::new_for_matches_only(vm.nfa().clone());
        FindEarliestMatches {
            vm,
            cache,
            captures,
            text,
            last_end: 0,
            last_match: None,
        }
    }
}

impl<'r, 'c, 't> Iterator for FindEarliestMatches<'r, 'c, 't> {
    type Item = Match;

    fn next(&mut self) -> Option<Match> {
        if self.last_end > self.text.len() {
            return None;
        }
        self.vm.find_earliest_at(
            self.cache,
            None,
            None,
            self.text,
            self.last_end,
            self.text.len(),
            &mut self.captures,
        );
        let m = self.captures.get_match()?;
        Some(handle_iter_match!(self, m, self.vm.config.get_utf8()))
    }
}

/// An iterator over all non-overlapping leftmost matches for a particular
/// infallible search.
///
/// The iterator yields a [`Match`] value until no more matches could be
/// found. If the underlying search returns an error, then this panics.
///
/// The lifetime variables are as follows:
///
/// * `'r` is the lifetime of the regular expression itself.
/// * `'c` is the lifetime of the mutable cache used during search.
/// * `'t` is the lifetime of the text being searched.
#[derive(Debug)]
pub struct FindLeftmostMatches<'r, 'c, 't> {
    vm: &'r PikeVM,
    cache: &'c mut Cache,
    captures: Captures,
    text: &'t [u8],
    last_end: usize,
    last_match: Option<usize>,
}

impl<'r, 'c, 't> FindLeftmostMatches<'r, 'c, 't> {
    fn new(
        vm: &'r PikeVM,
        cache: &'c mut Cache,
        text: &'t [u8],
    ) -> FindLeftmostMatches<'r, 'c, 't> {
        let captures = Captures::new_for_matches_only(vm.nfa().clone());
        FindLeftmostMatches {
            vm,
            cache,
            captures,
            text,
            last_end: 0,
            last_match: None,
        }
    }
}

impl<'r, 'c, 't> Iterator for FindLeftmostMatches<'r, 'c, 't> {
    type Item = Match;

    fn next(&mut self) -> Option<Match> {
        if self.last_end > self.text.len() {
            return None;
        }
        self.vm.find_leftmost_at(
            self.cache,
            None,
            None,
            self.text,
            self.last_end,
            self.text.len(),
            &mut self.captures,
        );
        let m = self.captures.get_match()?;
        Some(handle_iter_match!(self, m, self.vm.config.get_utf8()))
    }
}

#[derive(Debug)]
pub struct FindOverlappingMatches<'r, 'c, 't> {
    vm: &'r PikeVM,
    cache: &'c mut Cache,
    state: OverlappingState,
    captures: Captures,
    text: &'t [u8],
    last_end: usize,
}

impl<'r, 'c, 't> FindOverlappingMatches<'r, 'c, 't> {
    fn new(
        vm: &'r PikeVM,
        cache: &'c mut Cache,
        text: &'t [u8],
    ) -> FindOverlappingMatches<'r, 'c, 't> {
        let state = OverlappingState::start();
        let captures = Captures::new_for_matches_only(vm.nfa().clone());
        FindOverlappingMatches {
            vm,
            cache,
            state,
            captures,
            text,
            last_end: 0,
        }
    }
}

impl<'r, 'c, 't> Iterator for FindOverlappingMatches<'r, 'c, 't> {
    type Item = Match;

    fn next(&mut self) -> Option<Match> {
        if self.last_end > self.text.len() {
            return None;
        }
        self.vm.find_overlapping_at(
            self.cache,
            None,
            None,
            self.text,
            self.last_end,
            self.text.len(),
            &mut self.state,
            &mut self.captures,
        );
        let m = self.captures.get_match()?;
        Some(handle_iter_match_overlapping!(self, m))
    }
}

#[derive(Debug)]
pub struct CapturesEarliestMatches<'r, 'c, 't> {
    vm: &'r PikeVM,
    cache: &'c mut Cache,
    text: &'t [u8],
    last_end: usize,
    last_match: Option<usize>,
}

impl<'r, 'c, 't> CapturesEarliestMatches<'r, 'c, 't> {
    fn new(
        vm: &'r PikeVM,
        cache: &'c mut Cache,
        text: &'t [u8],
    ) -> CapturesEarliestMatches<'r, 'c, 't> {
        CapturesEarliestMatches {
            vm,
            cache,
            text,
            last_end: 0,
            last_match: None,
        }
    }
}

impl<'r, 'c, 't> Iterator for CapturesEarliestMatches<'r, 'c, 't> {
    type Item = Captures;

    fn next(&mut self) -> Option<Captures> {
        if self.last_end > self.text.len() {
            return None;
        }
        let mut caps = self.vm.create_captures();
        self.vm.find_earliest_at(
            self.cache,
            None,
            None,
            self.text,
            self.last_end,
            self.text.len(),
            &mut caps,
        );
        let m = caps.get_match()?;
        handle_iter_match!(self, m, self.vm.config.get_utf8());
        Some(caps)
    }
}

/// An iterator over all non-overlapping leftmost matches, with their capturing
/// groups, for a particular infallible search.
///
/// The iterator yields a [`Match`] value until no more matches could be
/// found. If the underlying search returns an error, then this panics.
///
/// The lifetime variables are as follows:
///
/// * `'r` is the lifetime of the regular expression itself.
/// * `'c` is the lifetime of the mutable cache used during search.
/// * `'t` is the lifetime of the text being searched.
#[derive(Debug)]
pub struct CapturesLeftmostMatches<'r, 'c, 't> {
    vm: &'r PikeVM,
    cache: &'c mut Cache,
    text: &'t [u8],
    last_end: usize,
    last_match: Option<usize>,
}

impl<'r, 'c, 't> CapturesLeftmostMatches<'r, 'c, 't> {
    fn new(
        vm: &'r PikeVM,
        cache: &'c mut Cache,
        text: &'t [u8],
    ) -> CapturesLeftmostMatches<'r, 'c, 't> {
        CapturesLeftmostMatches {
            vm,
            cache,
            text,
            last_end: 0,
            last_match: None,
        }
    }
}

impl<'r, 'c, 't> Iterator for CapturesLeftmostMatches<'r, 'c, 't> {
    type Item = Captures;

    fn next(&mut self) -> Option<Captures> {
        if self.last_end > self.text.len() {
            return None;
        }
        let mut caps = self.vm.create_captures();
        self.vm.find_leftmost_at(
            self.cache,
            None,
            None,
            self.text,
            self.last_end,
            self.text.len(),
            &mut caps,
        );
        let m = caps.get_match()?;
        handle_iter_match!(self, m, self.vm.config.get_utf8());
        Some(caps)
    }
}

#[derive(Debug)]
pub struct CapturesOverlappingMatches<'r, 'c, 't> {
    vm: &'r PikeVM,
    cache: &'c mut Cache,
    state: OverlappingState,
    text: &'t [u8],
    last_end: usize,
    last_match: Option<usize>,
}

impl<'r, 'c, 't> CapturesOverlappingMatches<'r, 'c, 't> {
    fn new(
        vm: &'r PikeVM,
        cache: &'c mut Cache,
        text: &'t [u8],
    ) -> CapturesOverlappingMatches<'r, 'c, 't> {
        let state = OverlappingState::start();
        CapturesOverlappingMatches {
            vm,
            cache,
            state,
            text,
            last_end: 0,
            last_match: None,
        }
    }
}

impl<'r, 'c, 't> Iterator for CapturesOverlappingMatches<'r, 'c, 't> {
    type Item = Captures;

    fn next(&mut self) -> Option<Captures> {
        if self.last_end > self.text.len() {
            return None;
        }
        let mut caps = self.vm.create_captures();
        self.vm.find_overlapping_at(
            self.cache,
            None,
            None,
            self.text,
            self.last_end,
            self.text.len(),
            &mut self.state,
            &mut caps,
        );
        let m = caps.get_match()?;
        handle_iter_match_overlapping!(self, m);
        Some(caps)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OverlappingState {
    step_index: Option<usize>,
    matched: bool,
}

impl OverlappingState {
    pub fn start() -> OverlappingState {
        OverlappingState { step_index: None, matched: false }
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
    current_slots: usize,
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

    fn clear(&mut self, slot_count: usize) {
        self.stack.clear();
        self.clist.set.clear();
        self.clist.list.clear();
        self.nlist.set.clear();
        self.nlist.list.clear();
        if slot_count != self.clist.current_slots {
            self.clist.current_slots = slot_count;
            self.nlist.current_slots = slot_count;
            self.scratch_caps.resize(slot_count, None);
        }
    }
}

impl Threads {
    fn new(nfa: &NFA) -> Threads {
        let mut threads = Threads {
            set: SparseSet::new(0),
            list: Vec::with_capacity(nfa.states().len()),
            caps: vec![],
            current_slots: 0,
        };
        threads.resize(nfa);
        threads
    }

    fn resize(&mut self, nfa: &NFA) {
        self.current_slots = nfa.capture_slot_len();
        self.set.resize(nfa.states().len());
        self.caps.resize(nfa.states().len(), Thread::new(nfa));
    }

    fn caps(&mut self, sid: StateID) -> &mut [Slot] {
        &mut self.caps[sid].slots[..self.current_slots]
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
