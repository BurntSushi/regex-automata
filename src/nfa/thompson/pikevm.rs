use alloc::{sync::Arc, vec, vec::Vec};

use crate::{
    nfa::thompson::{self, Captures, Error, State, NFA},
    util::{
        id::{PatternID, StateID},
        matchtypes::{MatchKind, MultiMatch},
        prefilter::{self, Prefilter},
        sparse_set::SparseSet,
    },
};

/// A simple macro for conditionally executing instrumentation logic when
/// the 'trace' log level is enabled. This is a compile-time no-op when the
/// 'instrument-pikevm' feature isn't enabled. The intent here is that this
/// makes it easier to avoid doing extra work when instrumentation isn't
/// enabled.
macro_rules! instrument {
    ($($tt:tt)*) => {
        #[cfg(feature = "instrument-pikevm")]
        {
            if log::log_enabled!(log::Level::Trace) {
                $($tt)*
            }
        }
    }
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
        Ok(PikeVM { config: self.config, nfa, pre: None })
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
    pre: Option<Arc<dyn Prefilter>>,
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

    pub fn is_match(&self, cache: &mut Cache, haystack: &[u8]) -> bool {
        // TODO: Fix this. We should need to allocate any captures.
        let mut caps = self.create_captures();
        self.find_leftmost(cache, haystack, &mut caps).is_some()
    }

    // BREADCRUMBS:
    //
    // 2) Consider the case of using a PikeVM with an NFA that has Capture
    // states, but where we don't want to track capturing groups (other than
    // group 0). This potentially saves a lot of copying around and what not. I
    // believe the current regex crate does this, for example. The interesting
    // bit here is how to handle the case of multiple patterns... It looks like
    // we might need a new state to differentiate "capture group" and "pattern
    // match offsets." Hmm..?
    //
    // 3) Permit the caller to specify a pattern ID to run an anchored-only
    // search on. This is done I guess?
    //
    // 4) How to do overlapping? The way multi-regex support works in the regex
    // crate currently is to run the PikeVM until either we reach the end of
    // the haystack or when we know all regexes have matched. The latter case
    // is probably quite rare, so the common case is likely that we're always
    // searching the entire input. The question is: can we emulate that with
    // our typical 'overlapping' APIs on DFAs? I believe we can. If so, then
    // all we need to do is provide an overlapping API on the PikeVM that
    // roughly matches the ones we provide on DFAs. For those APIs, the only
    // thing they need over non-overlapping APIs is "caller state." For DFAs,
    // the caller state is simple: it contains the last state visited and the
    // last match reported. For the PikeVM (and NFAs in general), the "last
    // state" is actually a *set* of NFA states. So I think what happens here
    // is that we can just force the `Cache` to subsume this role. We'll still
    // need some additional state to track the last match reported though.
    // Because when two or more patterns match at the same location, we need a
    // way to know to iterate over them. Although maybe it's not match index we
    // need, but the state index of the last NFA state processed in the cache.
    // Then we just pick up where we left off. There might be another match
    // state, in which case, we report it.

    pub fn find_leftmost(
        &self,
        cache: &mut Cache,
        haystack: &[u8],
        caps: &mut Captures,
    ) -> Option<MultiMatch> {
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

    pub fn captures_leftmost_iter<'r, 'c, 't>(
        &'r self,
        cache: &'c mut Cache,
        haystack: &'t [u8],
    ) -> CapturesLeftmostMatches<'r, 'c, 't> {
        CapturesLeftmostMatches::new(self, cache, haystack)
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
    ) -> Option<MultiMatch> {
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
    ) -> Option<MultiMatch> {
        self.find_fwd(
            false, cache, pre, pattern_id, haystack, start, end, caps,
        )
    }

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
    ) -> Option<MultiMatch> {
        let anchored = self.config.get_anchored()
            || self.nfa.is_always_start_anchored()
            || pattern_id.is_some();
        let start_id = match pattern_id {
            None => self.nfa.start_anchored(),
            Some(pid) => self.nfa.start_pattern(pid),
        };
        let mut at = start;
        let mut matched_pid = None;

        #[cfg(feature = "instrument-pikevm")]
        let mut counters = Counters::new(&self.nfa);

        cache.clear();
        'LOOP: while at <= end {
            if cache.clist.set.is_empty() {
                if matched_pid.is_some() || (anchored && at > start) {
                    break;
                }
            }
            if cache.clist.set.is_empty()
                || (!anchored && matched_pid.is_none())
            {
                self.epsilon_closure(
                    #[cfg(feature = "instrument-pikevm")]
                    &mut counters,
                    &mut cache.clist,
                    caps.slots_mut(),
                    &mut cache.stack,
                    start_id,
                    haystack,
                    at,
                );
            }
            instrument! {
                counters.record_state_set(&cache.clist.set);
            }

            for i in 0..cache.clist.set.len() {
                let sid = cache.clist.set.get(i);
                instrument! {
                    counters.record_step(sid);
                }

                let pid = match self.step(
                    #[cfg(feature = "instrument-pikevm")]
                    &mut counters,
                    &mut cache.nlist,
                    caps.slots_mut(),
                    cache.clist.caps(sid),
                    &mut cache.stack,
                    sid,
                    haystack,
                    at,
                ) {
                    None => continue,
                    Some(pid) => pid,
                };
                matched_pid = Some(pid);
                if earliest {
                    break 'LOOP;
                } else if self.config.get_match_kind() != MatchKind::All {
                    break;
                }
            }
            at += 1;
            cache.swap();
            cache.nlist.set.clear();
        }
        instrument! {
            counters.eprint(&self.nfa);
        }
        matched_pid.map(|pid| {
            // TODO: I think this should be set when the Match instruction
            // is seen?
            caps.set_pattern(pid);
            let m = caps.get(0).unwrap();
            MultiMatch::new(pid, m.start(), m.end())
        })
    }
}

impl PikeVM {
    /// Convenience function for returning this regex's prefilter as a trait
    /// object.
    ///
    /// If this regex doesn't have a prefilter, then `None` is returned.
    pub fn prefilter(&self) -> Option<&dyn Prefilter> {
        self.pre.as_ref().map(|x| &**x)
    }

    /// Attach the given prefilter to this regex.
    pub fn set_prefilter(&mut self, pre: Option<Arc<dyn Prefilter>>) {
        self.pre = pre;
    }

    /// Convenience function for returning a prefilter scanner.
    fn scanner(&self) -> Option<prefilter::Scanner> {
        self.prefilter().map(prefilter::Scanner::new)
    }
}

impl PikeVM {
    // BREADCRUMBS:
    //
    // Here's a good failing test to look at:
    //
    //     fowler/nullsubexpr/nullsubexpr69: expected to find
    //         [Captures([Some((0, (0, 1))), None, Some((0, (0, 1)))])]
    //     captures, but got
    //         [Captures([Some((0, (0, 1))), Some((0, (0, 0))), Some((0, (0, 1)))])]
    //     pattern:     ["(a*)*(x)"]
    //     input:       "x"
    //     test result: "captures_leftmost_iter"
    //
    // It looks like our PikeVM might not be handling empty capture group
    // matches correctly? The interesting bit here though is that at first
    // glance, it does look like '(a*)*' should match at the position before
    // 'x'. But it looks like that's now how most regex engines work.

    // #[inline(always)]
    // #[inline(never)]
    fn step(
        &self,
        #[cfg(feature = "instrument-pikevm")] counters: &mut Counters,
        nlist: &mut Threads,
        slots: &mut [Slot],
        thread_caps: &mut [Slot],
        stack: &mut Vec<FollowEpsilon>,
        sid: StateID,
        haystack: &[u8],
        at: usize,
    ) -> Option<PatternID> {
        match *self.nfa.state(sid) {
            State::Fail
            | State::Look { .. }
            | State::Union { .. }
            | State::BinaryUnion { .. }
            | State::Capture { .. } => None,
            State::ByteRange { ref trans } => {
                if trans.matches(haystack, at) {
                    self.epsilon_closure(
                        #[cfg(feature = "instrument-pikevm")]
                        counters,
                        nlist,
                        thread_caps,
                        stack,
                        trans.next,
                        haystack,
                        at + 1,
                    );
                }
                None
            }
            State::Sparse(ref sparse) => {
                if let Some(next) = sparse.matches(haystack, at) {
                    self.epsilon_closure(
                        #[cfg(feature = "instrument-pikevm")]
                        counters,
                        nlist,
                        thread_caps,
                        stack,
                        next,
                        haystack,
                        at + 1,
                    );
                }
                None
            }
            State::Match { pattern_id } => {
                // TODO: This is where we should call caps.set_pattern.
                slots.copy_from_slice(thread_caps);
                Some(pattern_id)
            }
        }
    }

    // #[inline(always)]
    #[inline(never)]
    fn epsilon_closure(
        &self,
        #[cfg(feature = "instrument-pikevm")] counters: &mut Counters,
        nlist: &mut Threads,
        thread_caps: &mut [Slot],
        stack: &mut Vec<FollowEpsilon>,
        sid: StateID,
        haystack: &[u8],
        at: usize,
    ) {
        instrument! {
            counters.record_closure(sid);
        }

        instrument! {
            counters.record_stack_push(sid);
        }
        stack.push(FollowEpsilon::StateID(sid));
        while let Some(frame) = stack.pop() {
            match frame {
                FollowEpsilon::StateID(sid) => {
                    self.epsilon_closure_step(
                        #[cfg(feature = "instrument-pikevm")]
                        counters,
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

    // #[inline(always)]
    // #[inline(never)]
    fn epsilon_closure_step(
        &self,
        #[cfg(feature = "instrument-pikevm")] counters: &mut Counters,
        nlist: &mut Threads,
        thread_caps: &mut [Slot],
        stack: &mut Vec<FollowEpsilon>,
        mut sid: StateID,
        haystack: &[u8],
        at: usize,
    ) {
        // BREADCRUMBS: OK so we've got two big wins: the return to manual
        // unanchored prefix (because we don't need to care about valid
        // for the unanchored prefix), and a rejiggering of the epsilon-closure
        // and step sequence: we move the 'byte' check into the epsilon closure
        // instead of in step. This keeps our states smaller.
        //
        // We also should rejigger how we deal with captures. RE2 has some
        // optimizations here. We really want to deal with allocation better
        // and perhaps avoid the 'caps' call. AND we want to be able to say
        // "we've seen this state" (to avoid epsilon closure loops) and
        // "don't step over this state" (because epsilon transitions do nothing
        // in 'step').
        //
        // So I think we need another kind of sparse set that permits storing
        // values with indices.
        //
        // Also, get rid of the NFA utf-8 option? That would be nice... Yes,
        // I think we should. If we keep it, then we can't unconditionally do
        // our manual unanchored prefix.
        loop {
            instrument! {
                counters.record_set_insert(sid);
            }
            if !nlist.set.insert(sid) {
                return;
            }
            match *self.nfa.state(sid) {
                State::Fail
                | State::ByteRange { .. }
                | State::Sparse { .. }
                | State::Match { .. } => {
                    let t = &mut nlist.caps(sid);
                    // TODO: What happens if 't' and 'thread_caps' aren't the
                    // same size? Depending on how we handle it, it may happen
                    // if we want to support running the PikeVM in a way that
                    // only tracks match start/end, and not all capturing
                    // groups.
                    t.copy_from_slice(thread_caps);
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
                    instrument! {
                        for &alt in &alternates[1..] {
                            counters.record_stack_push(alt);
                        }
                    }
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
                    instrument! {
                        counters.record_stack_push(sid);
                    }
                    stack.push(FollowEpsilon::StateID(alt2));
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

/// An iterator over all non-overlapping leftmost matches for a particular
/// infallible search.
///
/// The iterator yields a [`MultiMatch`] value until no more matches could be
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
    // scanner: Option<prefilter::Scanner<'r>>,
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
        FindEarliestMatches { vm, cache, text, last_end: 0, last_match: None }
    }
}

impl<'r, 'c, 't> Iterator for FindEarliestMatches<'r, 'c, 't> {
    type Item = MultiMatch;

    fn next(&mut self) -> Option<MultiMatch> {
        if self.last_end > self.text.len() {
            return None;
        }
        let mut caps = self.vm.create_captures();
        let m = self.vm.find_earliest_at(
            self.cache,
            None,
            None,
            self.text,
            self.last_end,
            self.text.len(),
            &mut caps,
        )?;
        Some(handle_iter_match!(self, m, self.vm.config.get_utf8()))
    }
}

/// An iterator over all non-overlapping leftmost matches for a particular
/// infallible search.
///
/// The iterator yields a [`MultiMatch`] value until no more matches could be
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
    // scanner: Option<prefilter::Scanner<'r>>,
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
        FindLeftmostMatches { vm, cache, text, last_end: 0, last_match: None }
    }
}

impl<'r, 'c, 't> Iterator for FindLeftmostMatches<'r, 'c, 't> {
    // type Item = Captures;
    type Item = MultiMatch;

    // fn next(&mut self) -> Option<Captures> {
    fn next(&mut self) -> Option<MultiMatch> {
        if self.last_end > self.text.len() {
            return None;
        }
        let mut caps = self.vm.create_captures();
        let m = self.vm.find_leftmost_at(
            self.cache,
            None,
            None,
            self.text,
            self.last_end,
            self.text.len(),
            &mut caps,
        )?;
        Some(handle_iter_match!(self, m, self.vm.config.get_utf8()))
    }
}

/// An iterator over all non-overlapping leftmost matches, with their capturing
/// groups, for a particular infallible search.
///
/// The iterator yields a [`MultiMatch`] value until no more matches could be
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
    // scanner: Option<prefilter::Scanner<'r>>,
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
        let m = self.vm.find_leftmost_at(
            self.cache,
            None,
            None,
            self.text,
            self.last_end,
            self.text.len(),
            &mut caps,
        )?;
        handle_iter_match!(self, m, self.vm.config.get_utf8());
        Some(caps)
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

    fn clear(&mut self) {
        self.stack.clear();
        self.clist.set.clear();
        self.nlist.set.clear();
    }

    fn swap(&mut self) {
        core::mem::swap(&mut self.clist, &mut self.nlist);
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
    fn new(nfa: &NFA) -> Counters {
        if log::log_enabled!(log::Level::Trace) {
            let len = nfa.states().len();
            Counters {
                state_sets: alloc::collections::BTreeMap::new(),
                steps: vec![0; len],
                closures: vec![0; len],
                stack_pushes: vec![0; len],
                set_inserts: vec![0; len],
            }
        } else {
            Counters {
                state_sets: alloc::collections::BTreeMap::new(),
                steps: vec![],
                closures: vec![],
                stack_pushes: vec![],
                set_inserts: vec![],
            }
        }
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
        let set = set.into_iter().collect::<Vec<StateID>>();
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
