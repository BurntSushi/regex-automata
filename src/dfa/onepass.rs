#![allow(warnings)]

use core::convert::TryFrom;

use alloc::vec;

use crate::{
    dfa::{error::Error, DEAD},
    nfa::thompson::{self, NFA},
    util::{
        alphabet::{self, ByteClasses},
        primitives::{PatternID, SmallIndex, StateID},
        search::MatchKind,
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
        todo!()
    }
}

#[derive(Debug)]
struct InternalBuilder<'a> {
    config: Config,
    nfa: &'a NFA,
    dfa: OnePass,
    nfa_to_dfa_id: Vec<StateID>,
    uncompiled_nfa_ids: Vec<StateID>,
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
        let stride2 = stride2(classes.alphabet_len());
        let dfa = OnePass { table: vec![], classes, stride2 };
        let nfa_to_dfa_id = vec![DEAD; nfa.states().len()];
        let uncompiled_nfa_ids = vec![];
        InternalBuilder { config, nfa, dfa, nfa_to_dfa_id, uncompiled_nfa_ids }
    }

    fn build(mut self) -> Result<OnePass, Error> {
        assert_eq!(DEAD, self.add_empty_state()?);
        let representatives: Vec<u8> = self
            .dfa
            .classes
            .representatives(..)
            .filter_map(|r| r.as_u8())
            .collect();
        let mut seen = SparseSet::new(self.nfa.states().len());
        let mut stack: Vec<(StateID, Info)> = vec![];

        // BREADCRUMBS: It kind of seems like the way we're handling multiple
        // patterns here is all wrong. In particular, I think we need to
        // maintain straight linear state when we do our depth first traversal,
        // instead of putting our state into the stack.
        //
        // Otherwise, I'm uneasy about losing the ordering of pattern IDs as
        // they're encountered. I believe that, at any given position, you
        // always want the one that appears first, which corresponds to the
        // natural sort order of pattern IDs. That is, if there are multiple
        // matches at position 'i', regardless of how long any of them are, the
        // pattern ID reported is always the smallest one.
        //
        // Maybe consider just doing single pattern first? Then go back and add
        // multi-pattern? Hmmm, nah...
        //
        // OK, I think I've fixed the multi-pattern handling, at least unless
        // my assumptions about ordering above are wrong.

        self.add_dfa_state_for_nfa_state(self.nfa.start_anchored())?;
        while let Some(nfa_id) = self.uncompiled_nfa_ids.pop() {
            let dfa_id = self.nfa_to_dfa_id[nfa_id];
            let mut matched = Patterns::empty();
            seen.clear();
            stack.push((nfa_id, Info::empty()));
            while let Some((id, info)) = stack.pop() {
                match *self.nfa.state(id) {
                    thompson::State::ByteRange { ref trans } => {
                        let mut next_dfa_id = self.nfa_to_dfa_id[trans.next];
                        if next_dfa_id == DEAD {
                            next_dfa_id =
                                self.add_dfa_state_for_nfa_state(trans.next)?;
                        }
                        // I wonder if this is wrong? What if we're still
                        // looking for a match for a different pattern? I think
                        // that only applies to MatchKind::All, in which case,
                        // we would mush on anyway.
                        if !matched.is_empty() {
                            continue;
                        }
                        let mut dfa_state = self.dfa.state_mut(dfa_id);
                        // FIXME: This isn't quite correct... We only want to
                        // look at equivalence classes within the range of
                        // bytes in our transition here.
                        // TODO: Consider factoring out this match arm into
                        // its own routine, so that we can call it for sparse
                        // transitions.
                        // TODO: Think about how this code might look if an
                        // NFA grew dense states. I guess dense states would
                        // probably use equivalence classes themselves, so it
                        // would just be a straight-forward iteration over
                        // them?
                        // TODO: Maybe write a method on ByteClasses that
                        // takes any range of bytes and returns an iterator
                        // over only representatives in that range. So, like
                        // the existing 'representatives()', but only for a
                        // specific range.
                        for &byte in representatives.iter() {
                            let oldtrans =
                                dfa_state.transitions()[byte as usize];
                            let newtrans = Transition::new(next_dfa_id, info);
                            if oldtrans.state_id() == DEAD {
                                dfa_state.transitions()[byte as usize] =
                                    newtrans;
                            } else if oldtrans != newtrans {
                                return Err(todo!());
                            }
                        }
                    }
                    thompson::State::Sparse(ref sparse) => {
                        todo!()
                    }
                    thompson::State::Look { look, next } => {
                        if !seen.insert(next) {
                            return Err(todo!());
                        }
                        stack.push((next, info.look_insert(look)));
                    }
                    thompson::State::Union { ref alternates } => {
                        for &sid in alternates.iter().rev() {
                            if !seen.insert(sid) {
                                return Err(todo!());
                            }
                            stack.push((sid, info));
                        }
                    }
                    thompson::State::BinaryUnion { alt1, alt2 } => {
                        if !seen.insert(alt1) || !seen.insert(alt2) {
                            return Err(todo!());
                        }
                        stack.push((alt2, info));
                        stack.push((alt1, info));
                    }
                    thompson::State::Capture { next, slot, .. } => {
                        if !seen.insert(next) {
                            return Err(todo!());
                        }
                        stack.push((next, info.slot_insert(slot)));
                    }
                    thompson::State::Fail => {
                        continue;
                    }
                    thompson::State::Match { pattern_id } => {
                        if matched.contains(pattern_id) {
                            return Err(todo!());
                        }
                        matched = matched.insert(pattern_id);
                        self.dfa.state_mut(dfa_id).set_pattern_info(
                            PatternInfo::empty()
                                .set_patterns(matched)
                                .set_info(info),
                        );
                    }
                }
            }
        }
        Ok(self.dfa)
    }

    fn add_dfa_state_for_nfa_state(
        &mut self,
        nfa_id: StateID,
    ) -> Result<StateID, Error> {
        assert!(!self.nfa.state(nfa_id).is_epsilon());
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
    table: Vec<Transition>,
    classes: ByteClasses,
    stride2: usize,
}

impl OnePass {
    fn alphabet_len(&self) -> usize {
        self.classes.alphabet_len()
    }

    fn stride(&self) -> usize {
        1 << self.stride2
    }

    fn state_mut(&mut self, id: StateID) -> StateMut<'_> {
        let offset = id.as_usize();
        let alphabet_len = self.alphabet_len();
        let raw = &mut self.table[offset..offset + alphabet_len + 1];
        StateMut { id, alphabet_len, raw }
    }

    fn memory_usage(&self) -> usize {
        self.table.len() * core::mem::size_of::<u64>()
    }
}

struct StateMut<'a> {
    id: StateID,
    alphabet_len: usize,
    raw: &'a mut [Transition],
}

impl<'a> StateMut<'a> {
    fn pattern_info(&mut self) -> PatternInfo {
        PatternInfo(self.raw[self.alphabet_len].0)
    }

    fn set_pattern_info(&mut self, pattern_info: PatternInfo) {
        self.raw[self.alphabet_len] = Transition(pattern_info.0);
    }

    fn transitions(&mut self) -> &mut [Transition] {
        &mut self.raw[..self.alphabet_len]
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

    fn slot_insert(self, slot: SmallIndex) -> Info {
        assert!(slot.as_usize() < Info::MAX_SLOTS);
        Info(self.0 | (1 << (8 + slot.as_usize())))
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
}

/// Computes the stride as a power of 2 for a one-pass DFA. The special sauce
/// here is that every state has 1+alphabet_len entries (each entry is a
/// u64), where the extra entry comes from match info. Like which look-around
/// assertions need to hold and which patterns have matched.
fn stride2(alphabet_len: usize) -> usize {
    let zeros = (1 + alphabet_len).next_power_of_two().trailing_zeros();
    usize::try_from(zeros).unwrap()
}
