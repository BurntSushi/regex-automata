#![allow(warnings)]

use core::convert::TryFrom;

use alloc::vec;

use crate::{
    dfa::{error::Error, remapper::Remapper, DEAD},
    nfa::thompson::{self, NFA},
    util::{
        alphabet::{self, ByteClasses},
        captures::Captures,
        iter,
        look::Look,
        primitives::{NonMaxUsize, PatternID, SmallIndex, StateID},
        search::{Input, Match, MatchError, MatchKind},
        sparse_set::SparseSet,
    },
};

#[derive(Clone, Debug, Default)]
pub struct Config {
    match_kind: Option<MatchKind>,
    starts_for_each_pattern: Option<bool>,
    byte_classes: Option<bool>,
    size_limit: Option<Option<usize>>,
    utf8: Option<bool>,
}

impl Config {
    pub fn new() -> Config {
        Config::default()
    }

    pub fn match_kind(mut self, kind: MatchKind) -> Config {
        self.match_kind = Some(kind);
        self
    }

    pub fn starts_for_each_pattern(mut self, yes: bool) -> Config {
        self.starts_for_each_pattern = Some(yes);
        self
    }

    pub fn byte_classes(mut self, yes: bool) -> Config {
        self.byte_classes = Some(yes);
        self
    }

    pub fn size_limit(mut self, limit: Option<usize>) -> Config {
        self.size_limit = Some(limit);
        self
    }

    pub fn utf8(mut self, yes: bool) -> Config {
        self.utf8 = Some(yes);
        self
    }

    pub fn get_match_kind(&self) -> MatchKind {
        self.match_kind.unwrap_or(MatchKind::LeftmostFirst)
    }

    pub fn get_starts_for_each_pattern(&self) -> bool {
        self.starts_for_each_pattern.unwrap_or(false)
    }

    pub fn get_byte_classes(&self) -> bool {
        self.byte_classes.unwrap_or(true)
    }

    pub fn get_size_limit(&self) -> Option<usize> {
        self.size_limit.unwrap_or(None)
    }

    pub fn get_utf8(&self) -> bool {
        self.utf8.unwrap_or(true)
    }

    /// Overwrite the default configuration such that the options in `o` are
    /// always used. If an option in `o` is not set, then the corresponding
    /// option in `self` is used. If it's not set in `self` either, then it
    /// remains not set.
    pub(crate) fn overwrite(&self, o: Config) -> Config {
        Config {
            match_kind: o.match_kind.or(self.match_kind),
            starts_for_each_pattern: o
                .starts_for_each_pattern
                .or(self.starts_for_each_pattern),
            byte_classes: o.byte_classes.or(self.byte_classes),
            size_limit: o.size_limit.or(self.size_limit),
            utf8: o.utf8.or(self.utf8),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Builder {
    config: Config,
    thompson: thompson::Compiler,
}

impl Builder {
    pub fn new() -> Builder {
        Builder {
            config: Config::default(),
            thompson: thompson::Compiler::new(),
        }
    }

    pub fn build(&self, pattern: &str) -> Result<OnePass, Error> {
        self.build_many(&[pattern])
    }

    pub fn build_many<P: AsRef<str>>(
        &self,
        patterns: &[P],
    ) -> Result<OnePass, Error> {
        let nfa = self.thompson.build_many(patterns).map_err(Error::nfa)?;
        self.build_from_nfa(&nfa)
    }

    pub fn build_from_nfa(&self, nfa: &NFA) -> Result<OnePass, Error> {
        let implicit_slots = nfa.pattern_len() * 2;
        let explicit_slots =
            nfa.group_info().slot_len().saturating_sub(implicit_slots);
        if explicit_slots > Info::MAX_SLOTS {
            return Err(Error::one_pass_fail(
                "too many explicit capturing groups (max is 24)",
            ));
        }
        InternalBuilder::new(self.config.clone(), nfa).build()
    }

    pub fn configure(&mut self, config: Config) -> &mut Builder {
        self.config = self.config.overwrite(config);
        self
    }

    pub fn syntax(
        &mut self,
        config: crate::util::syntax::SyntaxConfig,
    ) -> &mut Builder {
        self.thompson.syntax(config);
        self
    }

    pub fn thompson(&mut self, config: thompson::Config) -> &mut Builder {
        self.thompson.configure(config);
        self
    }
}

#[derive(Debug)]
struct InternalBuilder<'a> {
    config: Config,
    nfa: &'a NFA,
    dfa: OnePass,
    classes: ByteClasses,
    nfa_to_dfa_id: Vec<StateID>,
    uncompiled_nfa_ids: Vec<StateID>,
    seen: SparseSet,
    stack: Vec<(StateID, Info)>,
    matched: bool,
}

impl<'a> InternalBuilder<'a> {
    fn new(config: Config, nfa: &'a NFA) -> InternalBuilder {
        let classes = if !config.get_byte_classes() {
            // A one-pass DFA will always use the equivalence class map, but
            // enabling this option is useful for debugging. Namely, this will
            // cause all transitions to be defined over their actual bytes
            // instead of an opaque equivalence class identifier. The former is
            // much easier to grok as a human.
            ByteClasses::singletons()
        } else {
            nfa.byte_class_set().byte_classes()
        };
        // Normally a DFA alphabet includes the EOI symbol, but we don't need
        // that in the one-pass DFA since we handle look-around explicitly
        // without encoding it into the DFA. Thus, we don't need to delay
        // matches by 1 byte. However, we reuse the space that *would* be used
        // by the EOI transition by putting match information there (like which
        // pattern matches and which look-around assertions need to hold). So
        // this means our real alphabet length is 1 fewer than what the byte
        // classes report, since we don't use EOI.
        let alphabet_len = classes.alphabet_len().checked_sub(1).unwrap();
        let stride2 = classes.stride2();
        let dfa = OnePass {
            config: config.clone(),
            nfa: nfa.clone(),
            table: vec![],
            starts: vec![],
            min_match_id: DEAD,
            classes: classes.clone(),
            alphabet_len,
            stride2,
            patinfo_offset: classes.alphabet_len() - 1,
        };
        InternalBuilder {
            config,
            nfa,
            dfa,
            classes,
            nfa_to_dfa_id: vec![DEAD; nfa.states().len()],
            uncompiled_nfa_ids: vec![],
            seen: SparseSet::new(nfa.states().len()),
            stack: vec![],
            matched: false,
        }
    }

    fn build(mut self) -> Result<OnePass, Error> {
        assert_eq!(DEAD, self.add_empty_state()?);

        let explicit_slot_start = self.nfa.pattern_len() * 2;
        self.add_start_state(self.nfa.start_anchored())?;
        if self.config.get_starts_for_each_pattern() {
            for pid in self.nfa.patterns() {
                self.add_start_state(self.nfa.start_pattern(pid))?;
            }
        }
        while let Some(nfa_id) = self.uncompiled_nfa_ids.pop() {
            let dfa_id = self.nfa_to_dfa_id[nfa_id];
            self.matched = false;
            self.seen.clear();
            self.stack_push(nfa_id, Info::empty())?;
            while let Some((id, info)) = self.stack.pop() {
                match *self.nfa.state(id) {
                    thompson::State::ByteRange { ref trans } => {
                        self.compile_transition(dfa_id, trans, info)?;
                    }
                    thompson::State::Sparse(ref sparse) => {
                        for trans in sparse.transitions.iter() {
                            self.compile_transition(dfa_id, trans, info)?;
                        }
                    }
                    thompson::State::Look { look, next } => {
                        self.stack_push(next, info.look_insert(look))?;
                    }
                    thompson::State::Union { ref alternates } => {
                        for &sid in alternates.iter().rev() {
                            self.stack_push(sid, info)?;
                        }
                    }
                    thompson::State::BinaryUnion { alt1, alt2 } => {
                        self.stack_push(alt2, info)?;
                        self.stack_push(alt1, info)?;
                    }
                    thompson::State::Capture { next, slot, .. } => {
                        let slot = slot.as_usize();
                        let info = if slot < explicit_slot_start {
                            info
                        } else {
                            info.slot_insert(slot - explicit_slot_start)
                        };
                        self.stack_push(next, info)?;
                    }
                    thompson::State::Fail => {
                        continue;
                    }
                    thompson::State::Match { pattern_id } => {
                        if self.matched {
                            return Err(Error::one_pass_fail(
                                "multiple epsilon transitions to match state",
                            ));
                        }
                        self.matched = true;
                        self.dfa.set_pattern_info(
                            dfa_id,
                            PatternInfo::empty()
                                .set_pattern_id(pattern_id)
                                .set_info(info),
                        );
                        // N.B. It is tempting to just bail out here when
                        // compiling a leftmost-first DFA, since we will never
                        // compile any more transitions in that case. But we
                        // actually need to keep going in order to verify that
                        // we actually have a one-pass regex. e.g., We might
                        // see more Match states (e.g., for other patterns)
                        // that imply that we don't have a one-pass regex.
                        // So instead, we mark that we've found a match and
                        // continue on. When we go to compile a new DFA state,
                        // we just skip that part. But otherwise check that the
                        // one-pass property is upheld.
                    }
                }
            }
        }
        self.dfa.min_match_id = StateID::new_unchecked(self.dfa.table.len());
        let mut match_ids = vec![];
        for i in 0..self.dfa.state_len() {
            let id = StateID::new_unchecked(i << self.dfa.stride2());
            let is_match = self.dfa.pattern_info(id).pattern_id().is_some();
            if is_match {
                match_ids.push(id);
            }
        }
        let mut remapper = Remapper::new(&self.dfa);
        let mut next_dest =
            StateID::new_unchecked(self.dfa.table.len() - self.dfa.stride());
        for id in match_ids.into_iter().rev() {
            remapper.swap(&mut self.dfa, next_dest, id);
            self.dfa.min_match_id = next_dest;
            next_dest = StateID::new_unchecked(
                next_dest.as_usize() - self.dfa.stride(),
            );
        }
        remapper.remap(&mut self.dfa);
        Ok(self.dfa)
    }

    fn stack_push(
        &mut self,
        nfa_id: StateID,
        info: Info,
    ) -> Result<(), Error> {
        if !self.seen.insert(nfa_id) {
            return Err(Error::one_pass_fail(
                "multiple epsilon transitions to same state",
            ));
        }
        self.stack.push((nfa_id, info));
        Ok(())
    }

    fn compile_transition(
        &mut self,
        dfa_id: StateID,
        trans: &thompson::Transition,
        info: Info,
    ) -> Result<(), Error> {
        // If we already have seen a match and we are compiling a leftmost
        // first DFA, then we shouldn't add any more transitions. This is
        // effectively how preference order and non-greediness is implemented.
        if !self.config.get_match_kind().continue_past_first_match()
            && self.matched
        {
            return Ok(());
        }
        let mut next_dfa_id = self.nfa_to_dfa_id[trans.next];
        if next_dfa_id == DEAD {
            next_dfa_id = self.add_dfa_state_for_nfa_state(trans.next)?;
        }
        for byte in self
            .classes
            .representatives(trans.start..=trans.end)
            .filter_map(|r| r.as_u8())
        {
            let oldtrans = self.dfa.transition(dfa_id, byte);
            let newtrans = Transition::new(next_dfa_id, info);
            if oldtrans.state_id() == DEAD {
                self.dfa.set_transition(dfa_id, byte, newtrans);
            } else if oldtrans != newtrans {
                return Err(Error::one_pass_fail("conflicting transition"));
            }
        }
        Ok(())
    }

    fn add_start_state(&mut self, nfa_id: StateID) -> Result<StateID, Error> {
        let dfa_id = self.add_dfa_state_for_nfa_state(nfa_id)?;
        self.dfa.starts.push(dfa_id);
        Ok(dfa_id)
    }

    fn add_dfa_state_for_nfa_state(
        &mut self,
        nfa_id: StateID,
    ) -> Result<StateID, Error> {
        // If we've already built a DFA state for the given NFA state, then
        // just return that. We definitely do not want to have more than one
        // DFA state in existence for the same NFA state, since all but one of
        // them will likely become unreachable. And at least some of them are
        // likely to wind up being incomplete.
        let existing_dfa_id = self.nfa_to_dfa_id[nfa_id];
        if existing_dfa_id != DEAD {
            return Ok(existing_dfa_id);
        }
        // If we don't have any DFA state yet, add it and then add the given
        // NFA state to the list of states to explore.
        let dfa_id = self.add_empty_state()?;
        self.nfa_to_dfa_id[nfa_id] = dfa_id;
        self.uncompiled_nfa_ids.push(nfa_id);
        Ok(dfa_id)
    }

    fn add_empty_state(&mut self) -> Result<StateID, Error> {
        let next = self.dfa.table.len();
        let id = StateID::new(next).map_err(|_| Error::too_many_states())?;
        self.dfa
            .table
            .extend(core::iter::repeat(Transition(0)).take(self.dfa.stride()));
        // The default empty value for 'PatternInfo' is sadly not all zeroes.
        // Instead, the special sentinel u32::MAX<<32 is used to indicate that
        // there is no pattern. So we need to explicitly set the pattern info
        // to the correct "empty" state.
        self.dfa.set_pattern_info(id, PatternInfo::empty());
        if let Some(size_limit) = self.config.get_size_limit() {
            if self.dfa.memory_usage() > size_limit {
                return Err(Error::one_pass_exceeded_size_limit(size_limit));
            }
        }
        Ok(id)
    }
}

#[derive(Debug)]
pub struct OnePass {
    config: Config,
    nfa: NFA,
    table: Vec<Transition>,
    starts: Vec<StateID>,
    /// Every state ID >= this value corresponds to a match state.
    min_match_id: StateID,
    classes: ByteClasses,
    alphabet_len: usize,
    stride2: usize,
    patinfo_offset: usize,
}

impl OnePass {
    pub fn new(pattern: &str) -> Result<OnePass, Error> {
        OnePass::builder().build(pattern)
    }

    pub fn new_many<P: AsRef<str>>(patterns: &[P]) -> Result<OnePass, Error> {
        OnePass::builder().build_many(patterns)
    }

    pub fn config() -> Config {
        Config::new()
    }

    pub fn builder() -> Builder {
        Builder::new()
    }

    pub fn create_cache(&self) -> Cache {
        Cache::new(self)
    }

    pub fn create_captures(&self) -> Captures {
        Captures::all(self.nfa.group_info().clone())
    }

    #[inline]
    pub fn create_input<'h, 'p, H: ?Sized + AsRef<[u8]>>(
        &'p self,
        haystack: &'h H,
    ) -> Input<'h, 'p> {
        let c = self.get_config();
        Input::new(haystack.as_ref()).utf8(c.get_utf8())
    }

    #[inline]
    pub fn get_config(&self) -> &Config {
        &self.config
    }

    #[inline]
    pub fn get_nfa(&self) -> &NFA {
        &self.nfa
    }

    #[inline]
    pub fn is_match<H: AsRef<[u8]>>(
        &self,
        cache: &mut Cache,
        haystack: H,
    ) -> bool {
        let input = self.create_input(haystack.as_ref()).earliest(true);
        self.search_slots(cache, &input, &mut []).is_some()
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
    pub fn search(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        caps: &mut Captures,
    ) {
        caps.set_pattern(None);
        let pid = self.search_slots(cache, input, caps.slots_mut());
        caps.set_pattern(pid);
    }

    #[inline]
    pub fn search_slots(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Option<PatternID> {
        let m = match self.search_imp(cache, input, slots) {
            None => return None,
            Some(pid) if !input.get_utf8() => return Some(pid),
            Some(pid) => {
                let slot_start = pid.as_usize() * 2;
                let slot_end = slot_start + 1;
                if slot_end >= slots.len() {
                    return Some(pid);
                }
                // These unwraps are OK because we know we have a match and
                // we know our caller provided slots are big enough.
                let start = slots[slot_start].unwrap().get();
                let end = slots[slot_end].unwrap().get();
                if start < end {
                    return Some(pid);
                }
                Match::new(pid, start..end)
            }
        };
        input
            .skip_empty_utf8_splits(m, |search| {
                let pid = match self.search_imp(cache, search, slots) {
                    None => return Ok(None),
                    Some(pid) => pid,
                };
                let slot_start = pid.as_usize() * 2;
                let slot_end = slot_start + 1;
                let start = slots[slot_start].unwrap().get();
                let end = slots[slot_end].unwrap().get();
                Ok(Some(Match::new(pid, start..end)))
            })
            .unwrap()
            .map(|m| m.pattern())
    }

    fn search_imp(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Option<PatternID> {
        let explicit_start = self.nfa.pattern_len() * 2;
        let explicit_slots = core::cmp::min(
            Info::MAX_SLOTS,
            slots.len().saturating_sub(explicit_start),
        );
        cache.setup_search(explicit_slots);
        for slot in slots.iter_mut() {
            *slot = None;
        }
        for slot in cache.explicit_slots() {
            *slot = None;
        }
        if input.is_done() {
            return None;
        }
        for pid in self.nfa.patterns() {
            let i = pid.as_usize() * 2;
            if i >= slots.len() {
                break;
            }
            slots[i] = NonMaxUsize::new(input.start());
        }
        let haystack = input.haystack();
        let mut pid = None;
        let mut sid = match input.get_pattern() {
            None => self.start(),
            Some(pid) => self.start_pattern(pid),
        };
        for at in input.start()..input.end() {
            if sid >= self.min_match_id {
                let patinfo = self.pattern_info(sid);
                let foundpid = patinfo.pattern_id_unchecked();
                if patinfo.info().look_matches(haystack, at) {
                    pid = Some(foundpid);
                    let i = foundpid.as_usize() * 2 + 1;
                    if i < slots.len() {
                        slots[i] = NonMaxUsize::new(at);
                    }
                    if explicit_start < slots.len() {
                        slots[explicit_start..]
                            .copy_from_slice(cache.explicit_slots());
                        patinfo
                            .info()
                            .slot_apply(at, &mut slots[explicit_start..]);
                    }
                    if input.get_earliest() {
                        return pid;
                    }
                }
            }

            let trans = self.transition(sid, haystack[at]);
            sid = trans.state_id();
            let info = trans.info();
            if sid == DEAD || !info.look_matches(haystack, at) {
                return pid;
            }
            info.slot_apply(at, cache.explicit_slots());
        }
        if sid >= self.min_match_id {
            let patinfo = self.pattern_info(sid);
            let foundpid = patinfo.pattern_id_unchecked();
            if patinfo.info().look_matches(haystack, input.end()) {
                pid = Some(foundpid);
                let i = foundpid.as_usize() * 2 + 1;
                if i < slots.len() {
                    slots[i] = NonMaxUsize::new(input.end());
                }
                if explicit_start < slots.len() {
                    slots[explicit_start..]
                        .copy_from_slice(cache.explicit_slots());
                    patinfo
                        .info()
                        .slot_apply(input.end(), &mut slots[explicit_start..]);
                }
            }
        }
        pid
    }

    fn start(&self) -> StateID {
        self.starts[0]
    }

    fn start_pattern(&self, pid: PatternID) -> StateID {
        self.starts[pid.one_more()]
    }

    pub fn alphabet_len(&self) -> usize {
        self.alphabet_len
    }

    pub fn stride(&self) -> usize {
        1 << self.stride2
    }

    pub fn pattern_len(&self) -> usize {
        self.nfa.pattern_len()
    }

    fn transition(&self, sid: StateID, byte: u8) -> Transition {
        let class = self.classes.get(byte);
        self.table[sid.as_usize() + class as usize]
    }

    fn set_transition(&mut self, sid: StateID, byte: u8, to: Transition) {
        let class = self.classes.get(byte);
        self.table[sid.as_usize() + class as usize] = to;
    }

    fn pattern_info(&self, sid: StateID) -> PatternInfo {
        PatternInfo(self.table[sid.as_usize() + self.patinfo_offset].0)
    }

    fn set_pattern_info(&mut self, sid: StateID, patinfo: PatternInfo) {
        self.table[sid.as_usize() + self.patinfo_offset] =
            Transition(patinfo.0);
    }

    pub(crate) fn state_len(&self) -> usize {
        self.table.len() >> self.stride2
    }

    pub fn stride2(&self) -> usize {
        self.stride2
    }

    pub(crate) fn swap_states(&mut self, id1: StateID, id2: StateID) {
        for b in 0..self.stride() {
            self.table.swap(id1.as_usize() + b, id2.as_usize() + b);
        }
    }

    pub(crate) fn remap(&mut self, map: impl Fn(StateID) -> StateID) {
        for i in 0..self.state_len() {
            let id = StateID::new_unchecked(i << self.stride2());
            for b in 0..self.alphabet_len() {
                let next = self.table[id.as_usize() + b].state_id();
                self.table[id.as_usize() + b].set_state_id(map(next));
            }
        }
        for i in 0..self.starts.len() {
            self.starts[i] = map(self.starts[i]);
        }
    }

    pub fn memory_usage(&self) -> usize {
        use core::mem::size_of;

        self.table.len() * size_of::<Transition>()
            + self.starts.len() * size_of::<StateID>()
    }
}

/// A cache represents mutable state that a [`OnePass`] DFA requires during a
/// search.
///
/// For a given [`OnePass`] DFA, its corresponding cache may be created either
/// via [`OnePass::create_cache`], or via [`Cache::new`]. They are equivalent
/// in every way, except the former does not require explicitly importing
/// `Cache`.
///
/// A particular `Cache` is coupled with the [`OnePass`] DFA from which it
/// was created. It may only be used with that `OnePass` DFA. A cache and its
/// allocations may be re-purposed via [`Cache::reset`], in which case, it can
/// only be used with the new `OnePass` DFA (and not the old one).
#[derive(Clone, Debug)]
pub struct Cache {
    /// Scratch space used to store slots during a search. Basically, we use
    /// the caller provided slots to store slots known when a match occurs.
    /// But after a match occurs, we might continue a search but ultimately
    /// fail to extend the match. When continuing the search, we need some
    /// place to store candidate capture offsets without overwriting the slot
    /// offsets recorded for the most recently seen match.
    explicit_slots: Vec<Option<NonMaxUsize>>,
    /// The number of slots in the caller-provided 'Captures' value for the
    /// current search. Setting this to the total number of slots for the
    /// underlying NFA is always correct, but may be wasteful.
    slots_for_captures: usize,
}

impl Cache {
    pub fn new(re: &OnePass) -> Cache {
        let mut cache =
            Cache { explicit_slots: vec![], slots_for_captures: 0 };
        cache.reset(re);
        cache
    }

    pub fn reset(&mut self, re: &OnePass) {
        let slot_len = re
            .nfa
            .group_info()
            .slot_len()
            .saturating_sub(2 * re.nfa.pattern_len());
        self.explicit_slots.resize(slot_len, None);
        self.slots_for_captures = slot_len;
    }

    pub fn memory_usage(&self) -> usize {
        self.explicit_slots.len() * core::mem::size_of::<Option<NonMaxUsize>>()
    }

    fn explicit_slots(&mut self) -> &mut [Option<NonMaxUsize>] {
        &mut self.explicit_slots[..self.slots_for_captures]
    }

    fn setup_search(&mut self, explicit_slot_len: usize) {
        self.slots_for_captures = explicit_slot_len;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Transition(u64);

impl Transition {
    fn new(sid: StateID, info: Info) -> Transition {
        Transition(((sid.as_u32() as u64) << 32) | (info.0 as u64))
    }

    fn state_id(&self) -> StateID {
        // OK because a Transition has a valid StateID in its upper 32 bits
        // by construction. The cast to usize is also correct, even on 16-bit
        // targets because, again, we know the upper 32 bits is a valid
        // StateID, which can never overflow usize on any supported target.
        StateID::new_unchecked((self.0 >> 32) as usize)
    }

    fn set_state_id(&mut self, sid: StateID) {
        *self = Transition::new(sid, self.info());
    }

    fn info(&self) -> Info {
        Info(self.0 as u32)
    }
}

#[derive(Clone, Copy, Debug)]
struct PatternInfo(u64);

impl PatternInfo {
    const MASK_INFO: u64 = u32::MAX as u64;
    const MASK_PATTERN_ID: u64 = (u32::MAX as u64) << 32;

    fn empty() -> PatternInfo {
        PatternInfo((u32::MAX as u64) << 32)
    }

    fn pattern_id(self) -> Option<PatternID> {
        let pid = (self.0 >> 32) as u32;
        if pid == u32::MAX {
            None
        } else {
            Some(PatternID::new_unchecked(pid as usize))
        }
    }

    /// Returns the pattern ID without checking whether it's valid. If this is
    /// called and there is no pattern ID in this `PatternInfo`, then this
    /// will likely produce an incorrect result or possibly even a panic or
    /// an overflow. But safety will not be violated.
    fn pattern_id_unchecked(self) -> PatternID {
        let pid = (self.0 >> 32) as u32;
        PatternID::new_unchecked(pid as usize)
    }

    fn set_pattern_id(self, pid: PatternID) -> PatternInfo {
        PatternInfo(
            (self.0 & PatternInfo::MASK_INFO) | ((pid.as_u32() as u64) << 32),
        )
    }

    fn info(self) -> Info {
        Info(self.0 as u32)
    }

    fn set_info(self, info: Info) -> PatternInfo {
        PatternInfo((info.0 as u64) | (self.0 & PatternInfo::MASK_PATTERN_ID))
    }
}

#[derive(Clone, Copy)]
struct Info(u32);

impl Info {
    // Our 'Info' is 32 bits. 8 bits are dedicated to assertions. The remaining
    // 24 are dedicated to slots.
    //
    // TODO: It seems like we should be able to support all slots for overall
    // matches for each pattern in the search loop itself. And the slots set
    // here need only refer to explicit capture groups. But I'm not sure, so
    // let's just do the simple thing for now. This optimization would also
    // imply that the actual slot would need to be offset by the number of
    // patterns.
    const MAX_SLOTS: usize = 24;

    fn empty() -> Info {
        Info(0)
    }

    fn slot_insert(self, slot: usize) -> Info {
        assert!(slot < Info::MAX_SLOTS);
        Info(self.0 | (1 << (8 + slot)))
    }

    fn slot_contains(self, slot: usize) -> bool {
        self.0 & (1 << (8 + slot)) != 0
    }

    fn slot_is_empty(self) -> bool {
        self.0 & !0b1111_1111 == 0
    }

    fn slot_len(self) -> usize {
        usize::try_from((self.0 & !0xFF).count_ones()).unwrap()
    }

    fn slot_apply(self, at: usize, slots: &mut [Option<NonMaxUsize>]) {
        if self.slot_is_empty() {
            return;
        }
        for (i, slot) in slots.iter_mut().enumerate() {
            if self.slot_contains(i) {
                *slot = NonMaxUsize::new(at);
            }
        }
    }

    fn look_is_empty(self) -> bool {
        self.0 & 0b1111_1111 == 0
    }

    fn look_insert(self, look: Look) -> Info {
        Info(self.0 | (look.as_repr() as u32))
    }

    fn look_contains(self, look: Look) -> bool {
        self.0 & (look.as_repr() as u32) != 0
    }

    fn look_matches(self, haystack: &[u8], at: usize) -> bool {
        if self.look_is_empty() {
            return true;
        }
        let mut looks = self.0 as u8;
        while looks != 0 {
            let look = Look::from_repr(1 << looks.trailing_zeros()).unwrap();
            looks &= !look.as_repr();
            if !look.matches(haystack, at) {
                return false;
            }
        }
        true
    }
}

impl core::fmt::Debug for Info {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fail_conflicting_transition() {
        let predicate = |err: &str| err.contains("conflicting transition");

        let err = OnePass::new(r"a*[ab]").unwrap_err().to_string();
        assert!(predicate(&err), "{}", err);
    }

    #[test]
    fn fail_multiple_epsilon() {
        let predicate = |err: &str| {
            err.contains("multiple epsilon transitions to same state")
        };

        let err = OnePass::new(r"(^|$)a").unwrap_err().to_string();
        assert!(predicate(&err), "{}", err);
    }

    #[test]
    fn fail_multiple_match() {
        let predicate = |err: &str| {
            err.contains("multiple epsilon transitions to match state")
        };

        let err = OnePass::new_many(&[r"^", r"$"]).unwrap_err().to_string();
        assert!(predicate(&err), "{}", err);
    }

    #[test]
    fn scratch() {
        // let re =
        // OnePass::new_many(&[r"(a)(b)(c+)", r"(x+)(y+)(z+)([\w--z])+"])
        // .unwrap();
        let re = OnePass::new_many(&[r"(abc)+?"]).unwrap();
        let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
        let input = Input::new("abcabc");
        re.search(&mut cache, &input, &mut caps);
        dbg!(&caps);
    }
}
