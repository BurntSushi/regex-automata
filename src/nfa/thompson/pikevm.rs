use alloc::{sync::Arc, vec, vec::Vec};

use crate::{
    nfa::thompson::{self, Error, State, NFA},
    util::{
        id::{PatternID, StateID},
        matchtypes::MultiMatch,
        prefilter::{self, Prefilter},
        sparse_set::SparseSet,
    },
};

#[derive(Clone, Copy, Debug, Default)]
pub struct Config {
    anchored: Option<bool>,
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

    pub fn utf8(mut self, yes: bool) -> Config {
        self.utf8 = Some(yes);
        self
    }

    pub fn get_anchored(&self) -> bool {
        self.anchored.unwrap_or(false)
    }

    pub fn get_utf8(&self) -> bool {
        self.utf8.unwrap_or(true)
    }

    pub(crate) fn overwrite(self, o: Config) -> Config {
        Config {
            anchored: o.anchored.or(self.anchored),
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
        Captures::new(self.nfa())
    }

    pub fn nfa(&self) -> &NFA {
        &self.nfa
    }

    pub fn find_leftmost_iter<'r, 'c, 't>(
        &'r self,
        cache: &'c mut Cache,
        haystack: &'t [u8],
    ) -> FindLeftmostMatches<'r, 'c, 't> {
        FindLeftmostMatches::new(self, cache, haystack)
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
        let anchored = self.config.get_anchored()
            || self.nfa.is_always_start_anchored()
            || pattern_id.is_some();
        let start_id = match pattern_id {
            None if anchored => self.nfa.start_anchored(),
            None => self.nfa.start_unanchored(),
            // None => self.nfa.start_anchored(),
            Some(pid) => self.nfa.start_pattern(pid),
        };
        let mut at = start;
        let mut matched_pid = None;

        cache.clear();
        // FIXME: Putting the epsilon closure here is the most correct thing to
        // do, since it will correctly terminate a search when invalid UTF-8
        // is encountered if a only-valid-UTF-8 unanchored prefix is built
        // into the NFA. It also handles all other cases correctly. The main
        // problem is that this can lead to a fairly large slow-down, since the
        // list of states will always contain whatever this epsilon closure
        // populates them with (for unanchored searches). That's in contrast
        // to bypassing the unanchored prefix entirely by just re-adding the
        // *anchored* start state whenever the current list of states is empty.
        // (This would occur inside the loop below, but before the loop over
        // the current list of states.)
        //
        // But if we implement the unanchored prefix ourselves, then we also
        // need to provide a way for the search to terminate in the presence of
        // invalid UTF-8. This is... quite annoying to deal with. Here are some
        // options:
        //
        // 1) Accept the perf hit and keep the code simple.
        //
        // 2) Make the NFA expose what kind of unanchored prefix it has.
        // If it requires valid UTF-8, then... what? Write a specialized
        // UTF-8 decoder and only do an epsilon closure if we're at a valid
        // UTF-8 boundary?
        //
        // 3) Like 2, but expose something in the API that requires the
        // caller to say if the haystack is all valid UTF-8. If so, then the
        // unanchored prefix can never reject anything and all is well: the
        // fast version works. In cases where we have a valid UTF-8 unanchored
        // prefix but the haystack is not valid UTF-8, then we need to put the
        // epsilon closure at the start. (Which is also a huge bummer.)
        //
        // 4) Figure out how to make the simple code fast...
        //
        // How did the regex crate's old PikeVM implementation handle this?
        // Well, it rolled its own unanchored prefix (i.e., the fast way) but
        // its PikeVM was built on top of a input handling mechanism that tries
        // hard to deal only with Unicode codepoints. So the Pike VM there does
        // actually do its own UTF-8 decoding. Siiiiigh. We really do not want
        // to go back to that. Why? i) NFAs are now all byte based and ii) that
        // input handling mechanism greatly complicates the implementation and
        // was actually the source of some bugs. (ii) is likely possible to
        // mitigate by smarter code organization.
        //
        // Maybe we just go with (1) for now, accept the hit and try to
        // optimize this later? Siiiiiiiiiigh. Yup, that's what I'm doing.
        self.epsilon_closure(
            &mut cache.clist,
            &mut caps.slots,
            &mut cache.stack,
            start_id,
            haystack,
            at,
        );
        while at <= end && !cache.clist.set.is_empty() {
            // while at <= end {
            // if cache.clist.set.is_empty() {
            // if matched_pid.is_some() || (anchored && at > start) {
            // break;
            // }
            // }
            // if cache.clist.set.is_empty()
            // || (!anchored && matched_pid.is_none())
            // {
            // self.epsilon_closure(
            // &mut cache.clist,
            // &mut caps.slots,
            // &mut cache.stack,
            // start_id,
            // haystack,
            // at,
            // );
            // }
            for i in 0..cache.clist.set.len() {
                let sid = cache.clist.set.get(i);
                let pid = match self.step(
                    &mut cache.nlist,
                    &mut caps.slots,
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
                break;
            }
            at += 1;
            cache.swap();
            cache.nlist.set.clear();
        }
        matched_pid.map(|pid| {
            let (start, end) = self.nfa.slots(pid, 0);
            MultiMatch::new(
                pid,
                caps.slots[start].unwrap(),
                caps.slots[end].unwrap(),
            )
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
    // #[inline(always)]
    // #[inline(never)]
    fn step(
        &self,
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
            | State::Capture { .. } => None,
            State::ByteRange { ref trans } => {
                if trans.matches(haystack, at) {
                    self.epsilon_closure(
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
                slots.copy_from_slice(thread_caps);
                Some(pattern_id)
            }
        }
    }

    // #[inline(always)]
    #[inline(never)]
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

    // #[inline(always)]
    // #[inline(never)]
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
            // BREADCRUMBS: Time to look at the actual regex programs being
            // executed in the regex crate. There are some CRAZY perf
            // differences in the PikeVM. Methinks it has to do with the fact
            // that the regex crate has a "UnicodeRanges" state where as this
            // crate does not... Sigh. ---AG
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
        if m.is_empty() {
            // This is an empty match. To ensure we make progress, start
            // the next search at the smallest possible starting position
            // of the next match following this one.
            self.last_end = if self.vm.config.get_utf8() {
                crate::util::next_utf8(self.text, m.end())
            } else {
                m.end() + 1
            };
            // Don't accept empty matches immediately following a match.
            // Just move on to the next match.
            if Some(m.end()) == self.last_match {
                return self.next();
            }
        } else {
            self.last_end = m.end();
        }
        self.last_match = Some(m.end());
        Some(m)
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
