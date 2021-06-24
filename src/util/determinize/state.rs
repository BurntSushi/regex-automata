/*!
This module defines a DFA state representation and builders for constructing
DFA states.

This representation is specifically for use in implementations of NFA-to-DFA
conversion via powerset construction. (Also called "determinization" in this
crate.)

The term "DFA state" is somewhat overloaded in this crate. In some cases, it
refers to the set of transitions over an alphabet for a particular state. In
other cases, it refers to a set of NFA states. The former is really about the
final representation of a state in a DFA's transition table, where as the
latter---what this module is focusedon---is closer to an intermediate form that
is used to help eventually build the transition table.

This module exports four types. All four types represent the same idea: an
ordered set of NFA states. This ordered set represents the epsilon closure of a
particular NFA state, where the "epsilon closure" is the set of NFA states that
can be transitioned to without consuming any input. i.e., Follow all of theNFA
state's epsilon transitions. In addition, this implementation of DFA states
cares about two other things: the ordered set of pattern IDs corresponding
to the patterns that match if the state is a match state, and the set of
look-behind assertions that were true when the state was created.

The first, `State`, is a frozen representation of a state that cannot be
modified. It may be cheaply cloned without copying the state itself and can be
accessed safely from multiple threads simultaneously. This type is useful for
when one knows that the DFA state being constructed is distinct from any other
previously constructed states. Namely, powerset construction, in practice,
requires one to keep a cache of previously created DFA states. Otherwise,
the number of DFA states created in memory balloons to an impractically
large number. For this reason, equivalent states should endeavor to have an
equivalent byte-level representation. (In general, "equivalency" here means,
"equivalent assertions, pattern IDs and NFA state IDs." We do not require that
full DFA minimization be implemented here. This form of equivalency is only
surface deep and is more-or-less a practical necessity.)

The other three types represent different phases in the construction of a
DFA state. Internally, these three types (and `State`) all use the same
byte-oriented representation. That means one can use any of the builder types
to check whether the state it represents already exists or not. If it does,
then there is no need to freeze it into a `State` (which requires an alloc and
a copy). Here are the three types described succinctly:

* `StateBuilderEmpty` represents a state with no pattern IDs, no assertions
and no NFA states. Creating a `StateBuilderEmpty` performs no allocs. A
`StateBuilderEmpty` can only be used to query its underlying memory capacity,
or to convert into a builder for recording pattern IDs and/or assertions.
* `StateBuilderMatches` represents a state with zero or more pattern IDs, zero
or more satisfied assertions and zero NFA state IDs. A `StateBuilderMatches`
can only be used for adding pattern IDs and recording assertions.
* `StateBuilderNFA` represents a state with zero or more pattern IDs, zero or
more satisfied assertions and zero or more NFA state IDs. A `StateBuilderNFA`
can only be used for adding NFA state IDs and recording some assertions.

The expected flow here is to use the above builders to construct a candidate
DFA state to check if it already exists. If it does, then there's no need to
freeze it into a `State`. It it doesn't exist, then `StateBuilderNFA::to_state`
can be called to freeze the builder into an immutable `State`. In either
case, `clear` should be called on the builder to turn it back into a
`StateBuilderEmpty` that reuses the underyling memory.

The main purpose for splitting the builder into these distinct types is to
make it impossible to do things like adding a pattern ID after adding an NFA
state ID. Namely, this makes it simpler to use a space-and-time efficient
binary representation for the state. (The format is documented on the `Repr`
type below.) If we just used one type for everything, it would be possible for
callers to use an incorrect interleaving of calls and thus result in a corrupt
representation. I chose to use more type machinery to make this impossible to
do because 1) determinization is itself pretty complex and it wouldn't be too
hard to foul this up and 2) there isn't too much machinery involve and it's
well contained.

As an optimization, sometimes states won't have certain things set. For
example, if the underlying NFA has no word boundary assertions, then there is
no reason to set a state's look-behind assertion as to whether it was generated
from a word byte or not. Similarly, if a state has no NFA states corresponding
to look-around assertions, then there is no reason to set `look_have` to a
non-empty set. Finally, callers usually omit unconditional epsilon transitions
when adding NFA state IDs since they aren't discriminatory.

Finally, the binary representation used by these states is, thankfully, not
serialized anywhere. So any kind of change can be made with reckless abandon,
as long as everything in this module agrees.
*/

use core::convert::TryFrom;

use alloc::sync::Arc;

use crate::{
    nfa::thompson::LookSet,
    util::{
        bytes::{self, Endian},
        id::{PatternID, StateID},
    },
};

/// A DFA state that, at its core, is represented by an ordered set of NFA
/// states.
///
/// This type is intended to be used only in NFA-to-DFA conversion via powerset
/// construction.
///
/// It may be cheaply cloned and accessed safely from mulitple threads
/// simultaneously.
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

/// For docs on these routines, see the internal Repr and ReprVec types below.
impl State {
    pub(crate) fn dead() -> State {
        StateBuilderEmpty::new().into_matches().into_nfa().to_state()
    }

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

    pub(crate) fn match_pattern_ids(&self) -> Option<Vec<PatternID>> {
        self.repr().match_pattern_ids()
    }

    pub(crate) fn iter_match_pattern_ids<F: FnMut(PatternID)>(&self, f: F) {
        self.repr().iter_match_pattern_ids(f)
    }

    pub(crate) fn iter_nfa_state_ids<F: FnMut(StateID)>(&self, f: F) {
        self.repr().iter_nfa_state_ids(f)
    }

    pub(crate) fn memory_usage(&self) -> usize {
        self.0.len()
    }

    fn repr(&self) -> Repr<'_> {
        Repr(&*self.0)
    }
}

/// A state builder that represents an empty state.
///
/// This is a useful "initial condition" for state construction. It has no
/// NFA state IDs, no assertions set and no pattern IDs. No allocations are
/// made when new() is called. Its main use is for being converted into a
/// builder that can capture assertions and pattern IDs.
#[derive(Clone, Debug)]
pub(crate) struct StateBuilderEmpty(Vec<u8>);

/// For docs on these routines, see the internal Repr and ReprVec types below.
impl StateBuilderEmpty {
    pub(crate) fn new() -> StateBuilderEmpty {
        StateBuilderEmpty(vec![])
    }

    pub(crate) fn into_matches(mut self) -> StateBuilderMatches {
        self.0.extend_from_slice(&[0, 0, 0]);
        StateBuilderMatches(self.0)
    }

    fn clear(&mut self) {
        self.0.clear();
    }

    pub(crate) fn capacity(&self) -> usize {
        self.0.capacity()
    }
}

/// A state builder that collects assertions and pattern IDs.
///
/// When collecting pattern IDs is finished, this can be converted into a
/// builder that collects NFA state IDs.
#[derive(Clone, Debug)]
pub(crate) struct StateBuilderMatches(Vec<u8>);

/// For docs on these routines, see the internal Repr and ReprVec types below.
impl StateBuilderMatches {
    pub(crate) fn into_nfa(mut self) -> StateBuilderNFA {
        self.repr_vec().close_match_pattern_ids();
        StateBuilderNFA { repr: self.0, prev_nfa_state_id: StateID::ZERO }
    }

    pub(crate) fn clear(self) -> StateBuilderEmpty {
        let mut builder = StateBuilderEmpty(self.0);
        builder.clear();
        builder
    }

    pub(crate) fn is_match(&self) -> bool {
        self.repr().is_match()
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

/// A state builder that collects some assertions and NFA state IDs.
///
/// When collecting NFA state IDs is finished, this can be used to build a
/// `State` if necessary.
///
/// When dont with building a state (regardless of whether it got kept or not),
/// it's usually a good idea to call `clear` to get an empty builder back so
/// that it can be reused to build the next state.
#[derive(Clone, Debug)]
pub(crate) struct StateBuilderNFA {
    repr: Vec<u8>,
    prev_nfa_state_id: StateID,
}

/// For docs on these routines, see the internal Repr and ReprVec types below.
impl StateBuilderNFA {
    pub(crate) fn to_state(&self) -> State {
        State(Arc::from(&*self.repr))
    }

    pub(crate) fn clear(self) -> StateBuilderEmpty {
        let mut builder = StateBuilderEmpty(self.repr);
        builder.clear();
        builder
    }

    pub(crate) fn is_match(&self) -> bool {
        self.repr().is_match()
    }

    pub(crate) fn is_from_word(&self) -> bool {
        self.repr().is_from_word()
    }

    pub(crate) fn look_have(&mut self) -> &mut LookSet {
        LookSet::from_repr_mut(&mut self.repr[1])
    }

    pub(crate) fn look_need(&mut self) -> &mut LookSet {
        LookSet::from_repr_mut(&mut self.repr[2])
    }

    pub(crate) fn add_nfa_state_id(&mut self, sid: StateID) {
        ReprVec(&mut self.repr)
            .add_nfa_state_id(&mut self.prev_nfa_state_id, sid)
    }

    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.repr
    }

    fn repr(&self) -> Repr<'_> {
        Repr(&self.repr)
    }

    fn repr_vec(&mut self) -> ReprVec<'_> {
        ReprVec(&mut self.repr)
    }
}

/// Repr is a read-only view into the representation of a DFA state.
///
/// Primarily, a Repr is how we achieve DRY: we implement decoding the format
/// in one place, and then use a Repr to implement the various methods on the
/// public state types.
///
/// The format is as follows:
///
/// The first three bytes correspond to bitsets.
///
/// Byte 0 is a bitset corresponding to miscellaneous flags associated with the
/// state. Bit 0 is set to 1 if the state is a match state. Bit 1 is set to 1
/// if the state has pattern IDs explicitly written to it. (This is a flag that
/// is not meant to be set by determinization, but rather, is used as part of
/// an internal space-saving optimization.) Bit 2 is set to 1 if the state was
/// generated by a transition over a "word" byte. (Callers may not always set
/// this. For example, if the NFA has no word boundary assertion, then needing
/// to track whether a state came from a word byte or not is superfluous and
/// wasteful.)
///
/// Byte 1 corresponds to the look-behind assertions that were satisfied by
/// the transition that created this state. This generally only includes the
/// StartLine and StartText assertions. (Look-ahead assertions are not tracked
/// as part of states. Instead, these are applied by re-computing the epsilon
/// closure of a state when computing the transition function. See `next` in
/// the parent module.)
///
/// Byte 2 corresponds to the set of look-around assertions (including both
/// look-behind and look-ahead) that appear somewhere in this state's set of
/// NFA state IDs. This is used to determine whether this state's epsilon
/// closure should be re-computed when computing the transition function.
/// Namely, look-around assertions are "just" conditional epsilon transitions,
/// so if there are new assertions available when computing the transition
/// function, we should only re-compute the epsilon closure if those new
/// assertions are relevant to this particular state.
///
/// Bytes 3..11 correspond to a 64-bit native-endian encoded integer
/// corresponding to the byte offset at which the encoded NFA state IDs begin.
/// (Or, equivalently, the byte offset at which the encoded pattern IDs end.)
/// If the state is not a match state (byte 0 bit 0 is 0) or if it's only
/// pattern ID is PatternID::ZERO, then no integer is encoded at this position.
/// Instead, byte offset 3 is the position at which the first NFA state ID is
/// encoded.
///
/// For a match state with at least one non-ZERO pattern ID, the next bytes
/// correspond to a sequence of varuints[1] that represent each pattern ID, in
/// order, that this match state represents.
///
/// After the pattern IDs (if any), NFA state IDs are delta encoded as
/// varints.[1] The first NFA state ID is encoded as itself, and each
/// subsequent NFA state ID is encoded as the difference between itself and the
/// previous NFA state ID.
///
/// [1] - https://developers.google.com/protocol-buffers/docs/encoding#varints
#[derive(Debug)]
struct Repr<'a>(&'a [u8]);

impl<'a> Repr<'a> {
    /// Returns true if and only if this is a match state.
    ///
    /// If callers have added pattern IDs to this state, then callers MUST set
    /// this state as a match state explicitly. However, as a special case,
    /// states that are marked as match states but with no pattern IDs, then
    /// the state is treated as if it had a single pattern ID equivalent to
    /// PatternID::ZERO.
    fn is_match(&self) -> bool {
        self.0[0] & (1 << 0) > 0
    }

    /// Returns true if and only if this state has had at least one pattern
    /// ID added to it.
    ///
    /// This is an internal-only flag that permits the representation to save
    /// space in the common case of an NFA with one pattern in it. In that
    /// case, a match state can only ever have exactly one pattern ID:
    /// PatternID::ZERO. So there's no need to represent it.
    fn has_pattern_ids(&self) -> bool {
        self.0[0] & (1 << 1) > 0
    }

    /// Returns true if and only if this state is marked as having been created
    /// from a transition over a word byte. This is useful for checking whether
    /// a word boundary assertion is true or not, which requires look-behind
    /// (whether the current state came from a word byte or not) and look-ahead
    /// (whether the transition byte is a word byte or not).
    ///
    /// Since states with this set are distinct from states that don't have
    /// this set (even if they are otherwise equivalent), callers should not
    /// set this assertion unless the underlying NFA has at least one word
    /// boundary assertion somewhere. Otherwise, a superfluous number of states
    /// may be created.
    fn is_from_word(&self) -> bool {
        self.0[0] & (1 << 2) > 0
    }

    /// The set of look-behind assertions that were true in the transition that
    /// created this state.
    ///
    /// Generally, this should be empty if 'look_need' is empty, since there is
    /// no reason to track which look-behind assertions are true if the state
    /// has no conditional epsilon transitions.
    ///
    /// Satisfied look-ahead assertions are not tracked in states. Instead,
    /// these are re-computed on demand via epsilon closure when computing the
    /// transition function.
    fn look_have(&self) -> LookSet {
        LookSet::from_repr(self.0[1])
    }

    /// The set of look-around (both behind and ahead) assertions that appear
    /// at least once in this state's set of NFA states.
    ///
    /// This is used to determine whether the epsilon closure needs to be
    /// re-computed when computing the transition function. Namely, if the
    /// state has no conditional epsilon transitions, then there is no need
    /// to re-compute the epsilon closure.
    fn look_need(&self) -> LookSet {
        LookSet::from_repr(self.0[2])
    }

    /// Returns the offset into this state's representation where the pattern
    /// IDs end and the NFA state IDs begin.
    fn pattern_offset_end(&self) -> usize {
        if !self.has_pattern_ids() {
            return 3;
        }
        let off64 = bytes::read_u64(&self.0[3..11]);
        // This is OK since we only ever serialize usize values, so
        // deserializing as a usize must always succeed.
        usize::try_from(off64).unwrap()
    }

    /// Returns a copy of all match pattern IDs in this state. If this state
    /// is not a match state, then this returns None.
    fn match_pattern_ids(&self) -> Option<Vec<PatternID>> {
        if !self.is_match() {
            return None;
        }
        let mut pids = vec![];
        self.iter_match_pattern_ids(|pid| pids.push(pid));
        Some(pids)
    }

    /// Calls the given function on every pattern ID in this state.
    fn iter_match_pattern_ids<F: FnMut(PatternID)>(&self, mut f: F) {
        if !self.is_match() {
            return;
        }
        // As an optimization for a very common case, when this is a match
        // state for an NFA with only one pattern, we don't actually write the
        // pattern ID to the state representation. Instead, we know it must
        // be there since it is the only possible choice.
        if !self.has_pattern_ids() {
            f(PatternID::ZERO);
            return;
        }
        let mut pids = &self.0[11..self.pattern_offset_end()];
        while !pids.is_empty() {
            let (pid, nr) = read_varu32(pids);
            pids = &pids[nr..];
            // This is OK since we only ever serialize valid PatternIDs to
            // states. And since pattern IDs can never exceed a usize, the
            // unwrap is OK.
            f(PatternID::new_unchecked(usize::try_from(pid).unwrap()));
        }
    }

    /// Calls the given function on every NFA state ID in this state.
    fn iter_nfa_state_ids<F: FnMut(StateID)>(&self, mut f: F) {
        let mut sids = &self.0[self.pattern_offset_end()..];
        let mut prev = 0i32;
        while !sids.is_empty() {
            let (delta, nr) = read_vari32(sids);
            sids = &sids[nr..];
            let sid = prev + delta;
            prev = sid;
            // This is OK since we only ever serialize valid StateIDs to
            // states. And since state IDs can never exceed an isize, they must
            // always be able to fit into a usize, and thus cast is OK.
            f(StateID::new_unchecked(sid as usize))
        }
    }
}

/// ReprVec is a write-only view into the representation of a DFA state.
///
/// See Repr for more details on the purpose of this type and also the format.
///
/// Note that not all possible combinations of methods may be called. This is
/// precisely what the various StateBuilder types encapsulate: they only
/// permit valid combinations via Rust's linear typing.
#[derive(Debug)]
struct ReprVec<'a>(&'a mut Vec<u8>);

impl<'a> ReprVec<'a> {
    /// Set this state as a match state.
    ///
    /// This should not be exposed explicitly outside of this module. It is
    /// set automatically when a pattern ID is added.
    fn set_is_match(&mut self) {
        self.0[0] |= 1 << 0;
    }

    /// Set that this state has pattern IDs explicitly written to it.
    ///
    /// This should not be exposed explicitly outside of this module. This is
    /// used internally as a space saving optimization. Namely, if the state
    /// is a match state but does not have any pattern IDs written to it,
    /// then it is automatically inferred to have a pattern ID of ZERO.
    fn set_has_pattern_ids(&mut self) {
        self.0[0] |= 1 << 1;
    }

    /// Set this state as being built from a transition over a word byte.
    ///
    /// Setting this is only necessary when one needs to deal with word
    /// boundary assertions. Therefore, if the underlying NFA has no word
    /// boundary assertions, callers should not set this.
    fn set_is_from_word(&mut self) {
        self.0[0] |= 1 << 2;
    }

    /// Return a mutable reference to the 'look_have' assertion set.
    fn look_have_mut(&mut self) -> &mut LookSet {
        LookSet::from_repr_mut(&mut self.0[1])
    }

    /// Return a mutable reference to the 'look_need' assertion set.
    fn look_need_mut(&mut self) -> &mut LookSet {
        LookSet::from_repr_mut(&mut self.0[2])
    }

    /// Add a pattern ID to this state. All match states must have at least
    /// one pattern ID associated with it.
    ///
    /// The order in which patterns are added also corresponds to the order
    /// in which patterns are reported as matches.
    fn add_match_pattern_id(&mut self, pid: PatternID) {
        // As a (somewhat small) space saving optimization, in the case where
        // a matching state has exactly one pattern ID, PatternID::ZERO, we do
        // not write either the pattern ID or the offset indicating where
        // pattern IDs end. Instead, all we do is set the 'is_match' bit on
        // this state. Overall, this saves 9 bytes per match state for the
        // overwhelming majority of match states.
        //
        // In order to know whether pattern IDs need to be explicitly read or
        // not, we use another internal-only bit, 'has_pattern_ids', to
        // indicate whether they have been explicitly written or not.
        if !self.repr().has_pattern_ids() {
            if pid == PatternID::ZERO {
                self.set_is_match();
                return;
            }
            // Make room for 'close_match_pattern_ids' to write the offset
            // at which pattern IDs end (and NFA state IDs begin).
            self.0.extend(core::iter::repeat(0).take(8));
            self.set_has_pattern_ids();
            // If this was already a match state, then the only way that's
            // possible when the state doesn't have pattern IDs is if
            // PatternID::ZERO was added by the caller previously. In this
            // case, we are now adding a non-ZERO pattern ID after it, in
            // which case, we want to make sure to represent ZERO explicitly
            // now.
            if self.repr().is_match() {
                write_varu32(self.0, 0);
            } else {
                // Otherwise, just make sure the 'is_match' bit is set.
                self.set_is_match();
            }
        }
        write_varu32(self.0, pid.as_u32());
    }

    /// Indicate that no more pattern IDs will be added to this state.
    ///
    /// Once this is called, callers must not call it or 'add_match_pattern_id'
    /// again.
    ///
    /// This should not be exposed explicitly outside of this module. It
    /// should be called only when converting a StateBuilderMatches into a
    /// StateBuilderNFA.
    fn close_match_pattern_ids(&mut self) {
        // If we never wrote any pattern IDs, then there's nothing to do here.
        if !self.repr().has_pattern_ids() {
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

    /// Add an NFA state ID to this state. The order in which NFA states are
    /// added matters. It is the caller's responsibility to ensure that
    /// duplicate NFA state IDs are not added.
    fn add_nfa_state_id(&mut self, prev: &mut StateID, sid: StateID) {
        let delta = sid.as_i32() - prev.as_i32();
        write_vari32(self.0, delta);
        *prev = sid;
    }

    /// Return a read-only view of this state's representation.
    fn repr(&self) -> Repr<'_> {
        Repr(self.0.as_slice())
    }
}

/// Write a signed 32-bit integer using zig-zag encoding.
///
/// https://developers.google.com/protocol-buffers/docs/encoding#varints
fn write_vari32(data: &mut Vec<u8>, n: i32) {
    let mut un = (n as u32) << 1;
    if n < 0 {
        un = !un;
    }
    write_varu32(data, un)
}

/// Read a signed 32-bit integer using zig-zag encoding. Also, return the
/// number of bytes read.
///
/// https://developers.google.com/protocol-buffers/docs/encoding#varints
fn read_vari32(data: &[u8]) -> (i32, usize) {
    let (un, i) = read_varu32(data);
    let mut n = (un >> 1) as i32;
    if un & 1 != 0 {
        n = !n;
    }
    (n, i)
}

/// Write an unsigned 32-bit integer as a varint. In essence, `n` is written
/// as a sequence of bytes where all bytes except for the last one have the
/// most significant bit set. The least significant 7 bits correspond to the
/// actual bits of `n`. So in the worst case, a varint uses 5 bytes, but in
/// very common cases, it uses fewer than 4.
///
/// https://developers.google.com/protocol-buffers/docs/encoding#varints
fn write_varu32(data: &mut Vec<u8>, mut n: u32) {
    while n >= 0b1000_0000 {
        data.push((n as u8) | 0b1000_0000);
        n >>= 7;
    }
    data.push(n as u8);
}

/// Read an unsigned 32-bit varint. Also, return the number of bytes read.
///
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

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::quickcheck;

    #[test]
    fn scratch() {
        let mut b = StateBuilderEmpty::new().into_matches();
        b.add_match_pattern_id(PatternID::must(0));
        let s = b.into_nfa().to_state();
        assert_eq!(Some(vec![PatternID::must(0)]), s.match_pattern_ids());
    }

    quickcheck! {
        fn prop_state_read_write_nfa_state_ids(sids: Vec<StateID>) -> bool {
            let mut b = StateBuilderEmpty::new().into_matches().into_nfa();
            for &sid in &sids {
                b.add_nfa_state_id(sid);
            }
            let s = b.to_state();
            let mut got = vec![];
            s.iter_nfa_state_ids(|sid| got.push(sid));
            got == sids
        }

        fn prop_state_read_write_pattern_ids(pids: Vec<PatternID>) -> bool {
            let mut b = StateBuilderEmpty::new().into_matches();
            for &pid in &pids {
                b.add_match_pattern_id(pid);
            }
            let s = b.into_nfa().to_state();
            let mut got = vec![];
            s.iter_match_pattern_ids(|pid| got.push(pid));
            got == pids
        }

        fn prop_state_read_write_nfa_state_and_pattern_ids(
            sids: Vec<StateID>,
            pids: Vec<PatternID>
        ) -> bool {
            let mut b = StateBuilderEmpty::new().into_matches();
            for &pid in &pids {
                b.add_match_pattern_id(pid);
            }

            let mut b = b.into_nfa();
            for &sid in &sids {
                b.add_nfa_state_id(sid);
            }

            let s = b.to_state();
            let mut got_pids = vec![];
            s.iter_match_pattern_ids(|pid| got_pids.push(pid));
            let mut got_sids = vec![];
            s.iter_nfa_state_ids(|sid| got_sids.push(sid));
            got_pids == pids && got_sids == sids
        }

        fn prop_read_write_varu32(n: u32) -> bool {
            let mut buf = vec![];
            write_varu32(&mut buf, n);
            let (got, nread) = read_varu32(&buf);
            nread == buf.len() && got == n
        }

        fn prop_read_write_vari32(n: i32) -> bool {
            let mut buf = vec![];
            write_vari32(&mut buf, n);
            let (got, nread) = read_vari32(&buf);
            nread == buf.len() && got == n
        }
    }
}
