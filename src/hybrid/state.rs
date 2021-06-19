use core::convert::TryFrom;

use alloc::sync::Arc;

use crate::{
    nfa::thompson::LookSet,
    util::{
        bytes::{self, Endian},
        id::{PatternID, StateID},
    },
};

#[derive(Clone, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub(crate) struct State(Arc<[u8]>);

/// This Borrow impl permits us to lookup any state in a map by its byte
/// representation. This is particularly convenient when one has a StateBuilder
/// and we want to see if a correspondingly equivalent state already exists. If
/// one does exist, then we can reuse the allocation required by StateBuilder
/// without having to convert it into a State first.
impl core::borrow::Borrow<[u8]> for State {
    fn borrow(&self) -> &[u8] {
        &*self.0
    }
}

impl State {
    pub(crate) fn is_match(&self) -> bool {
        self.repr().is_match()
    }

    pub(crate) fn is_from_word(&self) -> bool {
        self.repr().is_from_word()
    }

    pub(crate) fn look_have(&self) -> LookSet {
        self.repr().look_have()
    }

    pub(crate) fn look_need(&self) -> LookSet {
        self.repr().look_need()
    }

    pub(crate) fn iter_match_pattern_ids<F: FnMut(PatternID)>(
        &self,
        mut f: F,
    ) {
        self.repr().iter_match_pattern_ids(f)
    }

    pub(crate) fn iter_nfa_state_ids<F: FnMut(StateID)>(&self, mut f: F) {
        self.repr().iter_nfa_state_ids(f)
    }

    fn repr(&self) -> Repr<'_> {
        Repr(&*self.0)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct StateBuilderEmpty(Vec<u8>);

impl StateBuilderEmpty {
    pub(crate) fn new() -> StateBuilderEmpty {
        StateBuilderEmpty(vec![])
    }

    pub(crate) fn into_matches(mut self) -> StateBuilderMatches {
        self.0.extend_from_slice(&[0, 0, 0]);
        StateBuilderMatches(self.0)
    }

    pub(crate) fn clear(&mut self) {
        self.0.clear();
    }

    pub(crate) fn as_bytes(&self) -> &[u8] {
        self.0.as_slice()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct StateBuilderMatches(Vec<u8>);

impl StateBuilderMatches {
    pub(crate) fn into_nfa(mut self) -> StateBuilderNFA {
        self.repr_vec().close_match_pattern_ids();
        StateBuilderNFA(self.0)
    }

    pub(crate) fn clear(self) -> StateBuilderEmpty {
        let mut builder = StateBuilderEmpty(self.0);
        builder.clear();
        builder
    }

    pub(crate) fn is_match(&self) -> bool {
        self.repr().is_match()
    }

    pub(crate) fn set_is_match(&mut self) {
        self.repr_vec().set_is_match()
    }

    pub(crate) fn is_from_word(&self) -> bool {
        self.repr().is_from_word()
    }

    pub(crate) fn set_is_from_word(&mut self) {
        self.repr_vec().set_is_from_word()
    }

    pub(crate) fn look_have(&mut self) -> &mut LookSet {
        LookSet::from_repr_mut(&mut self.0[1])
    }

    pub(crate) fn look_need(&mut self) -> &mut LookSet {
        LookSet::from_repr_mut(&mut self.0[2])
    }

    pub(crate) fn add_match_pattern_id(&mut self, pid: PatternID) {
        self.repr_vec().add_match_pattern_id(pid)
    }

    fn repr(&self) -> Repr<'_> {
        Repr(&self.0)
    }

    fn repr_vec(&mut self) -> ReprVec<'_> {
        ReprVec(&mut self.0)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct StateBuilderNFA(Vec<u8>);

impl StateBuilderNFA {
    pub(crate) fn into_state(self) -> State {
        State(Arc::from(self.0))
    }

    pub(crate) fn clear(self) -> StateBuilderEmpty {
        let mut builder = StateBuilderEmpty(self.0);
        builder.clear();
        builder
    }

    pub(crate) fn is_match(&self) -> bool {
        self.repr().is_match()
    }

    pub(crate) fn set_is_match(&mut self) {
        self.repr_vec().set_is_match()
    }

    pub(crate) fn is_from_word(&self) -> bool {
        self.repr().is_from_word()
    }

    pub(crate) fn set_is_from_word(&mut self) {
        self.repr_vec().set_is_from_word()
    }

    pub(crate) fn look_have(&mut self) -> &mut LookSet {
        LookSet::from_repr_mut(&mut self.0[1])
    }

    pub(crate) fn look_need(&mut self) -> &mut LookSet {
        LookSet::from_repr_mut(&mut self.0[2])
    }

    pub(crate) fn add_nfa_state_id(&mut self, sid: StateID) {
        self.repr_vec().add_nfa_state_id(sid)
    }

    fn repr(&self) -> Repr<'_> {
        Repr(&self.0)
    }

    fn repr_vec(&mut self) -> ReprVec<'_> {
        ReprVec(&mut self.0)
    }
}

#[derive(Debug)]
struct Repr<'a>(&'a [u8]);

impl<'a> Repr<'a> {
    fn is_match(&self) -> bool {
        self.0[0] & (1 << 0) > 0
    }

    fn is_from_word(&self) -> bool {
        self.0[0] & (1 << 1) > 0
    }

    fn look_have(&self) -> LookSet {
        LookSet::from_repr(self.0[1])
    }

    fn look_need(&self) -> LookSet {
        LookSet::from_repr(self.0[2])
    }

    fn pattern_offset_end(&self) -> usize {
        if !self.is_match() {
            return 3;
        }
        let off64 = bytes::read_u64(&self.0[3..11]);
        // This is OK since we only ever serialize usize values, so
        // deserializing as a usize must always succeed.
        usize::try_from(off64).unwrap()
    }

    fn iter_match_pattern_ids<F: FnMut(PatternID)>(&self, mut f: F) {
        let mut pids = &self.0[3..self.pattern_offset_end()];
        while !pids.is_empty() {
            let (pid, nr) = read_varu32(pids);
            pids = &pids[nr..];
            // This is OK since we only ever serialize valid PatternIDs to
            // states. And since pattern IDs can never exceed a usize, the
            // unwrap is OK.
            f(PatternID::new_unchecked(usize::try_from(pid).unwrap()));
        }
    }

    fn iter_nfa_state_ids<F: FnMut(StateID)>(&self, mut f: F) {
        let mut sids = &self.0[self.pattern_offset_end()..];
        while !sids.is_empty() {
            let (sid, nr) = read_varu32(sids);
            sids = &sids[nr..];
            // This is OK since we only ever serialize valid StateIDs to
            // states. And since state IDs can never exceed a usize, the unwrap
            // is OK.
            f(StateID::new_unchecked(usize::try_from(sid).unwrap()));
        }
    }
}

#[derive(Debug)]
struct ReprVec<'a>(&'a mut Vec<u8>);

impl<'a> ReprVec<'a> {
    fn set_is_match(&mut self) {
        // If we never added space for recording the offset at which pattern
        // IDs in this state end, then add that space now.
        if self.0.len() <= 3 {
            self.0.extend(core::iter::repeat(0).take(8));
        }
        self.0[0] |= (1 << 0);
    }

    fn set_is_from_word(&mut self) {
        self.0[0] |= (1 << 1);
    }

    fn look_have_mut(&mut self) -> &mut LookSet {
        LookSet::from_repr_mut(&mut self.0[1])
    }

    fn look_need_mut(&mut self) -> &mut LookSet {
        LookSet::from_repr_mut(&mut self.0[2])
    }

    fn add_match_pattern_id(&mut self, pid: PatternID) {
        write_varu32(self.0, pid.as_u32());
    }

    fn close_match_pattern_ids(&mut self) {
        // If this isn't a match state, then there are no pattern IDs to
        // account for, so we don't need to write anything.
        if !self.repr().is_match() {
            return;
        }
        // This unwrap is OK since the number of patterns is guaranteed to be
        // representable by a u32. Conservatively, if we use 4 bytes for each
        // pattern and the maximum number of patterns were encoded, then
        // this offset would be 3 + (4 * PatternID::LIMIT) = ~(2^34 + 3), which
        // of course fits into a u64.
        let nfa_state_id_start = u64::try_from(self.0.len()).unwrap();
        bytes::NE::write_u64(nfa_state_id_start, &mut self.0[3..11]);
    }

    fn add_nfa_state_id(&mut self, sid: StateID) {
        write_varu32(self.0, sid.as_u32());
    }

    fn repr(&self) -> Repr<'_> {
        Repr(self.0.as_slice())
    }
}

/// https://developers.google.com/protocol-buffers/docs/encoding#varints
fn write_vari32(data: &mut Vec<u8>, n: i32) {
    let mut un = (n as u32) << 1;
    if n < 0 {
        un = !un;
    }
    write_varu32(data, un)
}

/// https://developers.google.com/protocol-buffers/docs/encoding#varints
fn read_vari32(data: &[u8]) -> (i32, usize) {
    let (un, i) = read_varu32(data);
    let mut n = (un >> 1) as i32;
    if un & 1 != 0 {
        n = !n;
    }
    (n, i)
}

/// https://developers.google.com/protocol-buffers/docs/encoding#varints
fn write_varu32(data: &mut Vec<u8>, mut n: u32) {
    while n >= 0b1000_0000 {
        data.push((n as u8) | 0b1000_0000);
        n >>= 7;
    }
    data.push(n as u8);
}

/// https://developers.google.com/protocol-buffers/docs/encoding#varints
fn read_varu32(data: &[u8]) -> (u32, usize) {
    let mut n: u32 = 0;
    let mut shift: u32 = 0;
    for (i, &b) in data.iter().enumerate() {
        if b < 0b1000_0000 {
            return (n | ((b as u32) << shift), i + 1);
        }
        n |= ((b as u32) & 0b0111_1111) << shift;
        shift += 7;
    }
    (0, 0)
}
