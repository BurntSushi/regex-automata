#![allow(warnings)]

use core::convert::TryFrom;

use alloc::vec;

use crate::{
    dfa::{error::Error, DEAD},
    nfa::thompson::{self, NFA},
    util::{
        alphabet::{self, ByteClasses},
        captures::Captures,
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
}

impl Config {
    pub fn new() -> Config {
        Config::default()
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
        InternalBuilder::new(self.config.clone(), nfa).build()
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
    matched: Patterns,
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
        let stride2 = classes.stride2();
        let dfa = OnePass {
            nfa: nfa.clone(),
            table: vec![],
            starts: vec![],
            classes: classes.clone(),
            stride2,
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
            matched: Patterns::empty(),
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
            self.matched = Patterns::empty();
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
                        if !self.matched.is_empty() {
                            return Err(Error::one_pass_fail(
                                "multiple epsilon transitions to match state",
                            ));
                        }
                        self.matched = self.matched.insert(pattern_id);
                        self.dfa.set_pattern_info(
                            dfa_id,
                            PatternInfo::empty()
                                .set_patterns(self.matched)
                                .set_info(info),
                        );
                    }
                }
            }
        }
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
        let mut next_dfa_id = self.nfa_to_dfa_id[trans.next];
        if next_dfa_id == DEAD {
            next_dfa_id = self.add_dfa_state_for_nfa_state(trans.next)?;
        }
        // I wonder if this is wrong? What if we're still
        // looking for a match for a different pattern? I think
        // that only applies to MatchKind::All, in which case,
        // we would mush on anyway.
        //
        // Also, I wonder if it would be better to do this when handling
        // the Match state. We would clear the stack and just stop.
        if !self.matched.is_empty() {
            return Ok(());
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
    nfa: NFA,
    table: Vec<Transition>,
    starts: Vec<StateID>,
    classes: ByteClasses,
    stride2: usize,
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
        cache.setup_search(slots.len().saturating_sub(explicit_start));
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
            let patinfo = self.pattern_info(sid);
            if !patinfo.patterns().is_empty()
                && patinfo.info().look_matches(haystack, at)
            {
                // TODO: fix this
                pid = Some(PatternID::ZERO);
                if explicit_start < slots.len() {
                    slots[explicit_start..]
                        .copy_from_slice(cache.explicit_slots());
                    patinfo
                        .info()
                        .slot_apply(at, &mut slots[explicit_start..]);
                }
                if slots.len() >= 2 {
                    slots[1] = NonMaxUsize::new(at);
                }
            }

            let trans = self.transition(sid, haystack[at]);
            sid = trans.state_id();
            let info = trans.info();
            if sid == DEAD || !info.look_matches(haystack, at) {
                dbg!("DEAD return");
                return pid;
            }
            info.slot_apply(at, cache.explicit_slots());
        }
        let patinfo = self.pattern_info(sid);
        if !patinfo.patterns().is_empty()
            && patinfo.info().look_matches(haystack, input.end())
        {
            // TODO: fix this
            pid = Some(PatternID::ZERO);
            if explicit_start < slots.len() {
                slots[explicit_start..]
                    .copy_from_slice(cache.explicit_slots());
                patinfo
                    .info()
                    .slot_apply(input.end(), &mut slots[explicit_start..]);
            }
            if slots.len() >= 2 {
                slots[1] = NonMaxUsize::new(input.end());
            }
        }
        dbg!("EOI return");
        pid
    }

    fn start(&self) -> StateID {
        self.starts[0]
    }

    fn start_pattern(&self, pid: PatternID) -> StateID {
        self.starts[pid.one_more()]
    }

    fn alphabet_len(&self) -> usize {
        self.classes.alphabet_len()
    }

    fn stride(&self) -> usize {
        1 << self.stride2
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
        let alphabet_len = self.alphabet_len();
        PatternInfo(self.table[sid.as_usize() + alphabet_len - 1].0)
    }

    fn set_pattern_info(&mut self, sid: StateID, patinfo: PatternInfo) {
        let alphabet_len = self.alphabet_len();
        self.table[sid.as_usize() + alphabet_len - 1] = Transition(patinfo.0);
    }

    fn memory_usage(&self) -> usize {
        use core::mem::size_of;

        self.table.len() * size_of::<Transition>()
            + self.starts.len() * size_of::<StateID>()
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
        // Guaranteed not to overflow because StateID will never overflow
        // u32 or usize.
        self.0 |= (sid.as_usize() as u64) << 32;
    }

    fn info(&self) -> Info {
        Info(self.0 as u32)
    }
}

#[derive(Clone, Copy, Debug)]
struct PatternInfo(u64);

impl PatternInfo {
    const MASK_INFO: u64 = u32::MAX as u64;
    const MASK_PATTERNS: u64 = (u32::MAX as u64) << 32;

    fn empty() -> PatternInfo {
        PatternInfo(0)
    }

    fn patterns(self) -> Patterns {
        Patterns((self.0 >> 32) as u32)
    }

    fn set_patterns(self, patterns: Patterns) -> PatternInfo {
        PatternInfo(
            (self.0 & PatternInfo::MASK_INFO) | ((patterns.0 as u64) << 32),
        )
    }

    fn info(self) -> Info {
        Info(self.0 as u32)
    }

    fn set_info(self, info: Info) -> PatternInfo {
        PatternInfo((info.0 as u64) | (self.0 & PatternInfo::MASK_PATTERNS))
    }
}

#[derive(Clone, Copy, Debug)]
struct Patterns(u32);

impl Patterns {
    const MAX: usize = 32;

    fn empty() -> Patterns {
        Patterns(0)
    }

    fn is_empty(self) -> bool {
        self.0 == 0
    }

    fn contains(self, pid: PatternID) -> bool {
        assert!(pid.as_usize() < Patterns::MAX);
        self.0 & (1 << pid.as_u32()) != 0
    }

    fn insert(self, pid: PatternID) -> Patterns {
        assert!(pid.as_usize() < Patterns::MAX);
        Patterns(self.0 | (1 << pid.as_u32()))
    }
}

#[derive(Clone, Copy, Debug)]
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

    fn look_insert(self, look: thompson::Look) -> Info {
        Info(self.0 | (look.as_repr() as u32))
    }

    fn look_contains(self, look: thompson::Look) -> bool {
        self.0 & (look.as_repr() as u32) != 0
    }

    fn look_matches(self, haystack: &[u8], at: usize) -> bool {
        let mut looks = self.0 as u8;
        while looks != 0 {
            let look = thompson::Look::from_repr(1 << looks.trailing_zeros())
                .unwrap();
            looks &= !look.as_repr();
            if !look.matches(haystack, at) {
                return false;
            }
        }
        true
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
            .saturating_sub(re.nfa.pattern_len());
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
        let re = OnePass::new(r"(a)(b)(c)").unwrap();
        let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
        let input = Input::new("abc");
        re.search(&mut cache, &input, &mut caps);
        dbg!(&caps);
    }
}
