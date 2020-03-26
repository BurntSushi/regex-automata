use core::cmp;
use core::mem::size_of;

use crate::{
    bytes::{self, DeserializeError, Endian, SerializeError},
    dfa::Error,
    state_id::{dead_id, StateID},
};

macro_rules! err {
    ($msg:expr) => {
        return Err(DeserializeError::generic($msg));
    };
}

// Special represents the identifiers in a DFA that correspond to "special"
// states. If a state is one or more of the following, then it is considered
// special:
//
// * dead - A non-matching state where all outgoing transitions lead back to
//   itself. There is only one of these, regardless of whether minimization
//   has run. The dead state always has an ID of 0. i.e., It is always the
//   first state in a DFA.
// * quit - A state that is entered whenever a byte is seen that should cause
//   a DFA to give up and stop searching. This results in a NoMatch::Quit
//   error being returned. The default configuration for a DFA has no quit
//   bytes, which means this state is unreachable by default (and can be
//   removed during minimization). This state is only reachable when the
//   caller configures the DFA to quit on certain bytes. There is always at
//   most one of these states and it is always the second state. (Its actual
//   ID depends on the size of the alphabet in dense DFAs.)
// * match - An accepting state, i.e., indicative of a match. There may be
//   zero or more of these states.
// * accelerated - A state where all of its outgoing transitions, except a
//   few, loop back to itself. These states are candidates for acceleration
//   via memchr during search.
// * start - A non-matching state that indicates where the automaton should
//   start during a search. There is always at least one starting state and
//   all are guaranteed to be non-match states. (A start state cannot be a
//   match state because the DFAs in this crate delay all matches by one byte.
//   So every search that finds a match must move through one transition to
//   some other match state, even when searching an empty string.)
//
// These are not mutually exclusive categories. Namely, the following
// overlappings can occur:
//
// * {dead, start} - If a DFA can never lead to a match and it is minimized,
//   then it will typically compiled to something where all starting IDs
//   point to the DFA's dead state.
// * {dead, quit} - If a DFA was not configured to quit (the default), then
//   once the DFA is minimized, the quit sentinel state will be detected as
//   unreachable and removed. When this happens, the `quit_id` is equivalent
//   to `0` (which is always the ID of the dead state in every DFA). If a
//   quit state is present, then `quit_id` is guaranteed to be non-zero.
// * {match, accelerated} - It is possible for a match state to have the
//   majority of its transitions loop back to itself, which means it's
//   possible for a match state to be accelerated.
// * {start, accelerated} - Similarly, it is possible for a state state to be
//   accelerated. Note that it is possible for an accelerated state to be
//   neither a match or a start state. Also note that just because both match
//   and start states overlap with accelerated states does not mean that
//   match and start states overlap with each other. In fact, they are
//   guaranteed not to overlap.
//
// So the main problem we want to solve here is the *fast* detection of
// whether a state is special or not. And we also want to do this while
// storing as little extra data as possible.
//
// We achieve this by essentially shuffling all special states to the
// beginning of a DFA. That is, any special state appears before every
// other non-special state. By representing special states this way, we can
// determine whether a state is special or not by a single comparison, where
// special.max is the identifier of the last special state in the DFA:
//
//     if current_state <= special.max:
//         ... do something with special state
//
// The only thing left to do is to determine what kind of special state
// it is. Because what we do next depends on that. Since special states
// are typically rare, we can afford to do a bit more extra work, but we'd
// still like this to be as fast as possible. The trick we employ here is to
// continue shuffling states even within the special state range. Such that
// one contiguous region corresponds to match states, another for start states
// and then an overlapping range for accelerated states. At a high level, our
// special state detection might look like this (for leftmost searching, where
// we continue searching even after seeing a match):
//
//     byte = input[offset]
//     current_state = next_state(current_state, byte)
//     offset += 1
//     if current_state <= special.max:
//         if current_state == 0:
//             # We can never leave a dead state, so this always marks the
//             # end of our search.
//             return last_match
//         if current_state == special.quit_id:
//             # A quit state means we give up. If he DFA has no quit state,
//             # then special.quit_id == 0 == dead, which is handled by the
//             # conditional above.
//             return Err(NoMatch::Quit { byte, offset: offset - 1 })
//         if special.min_match <= current_state <= special.max_match:
//             last_match = Some(offset)
//             if special.min_accel <= current_state <= special.max_accel:
//                 offset = accelerate(input, offset)
//                 last_match = Some(offset)
//         elif special.min_start <= current_state <= special.max_start:
//             offset = prefilter.find(input, offset)
//             if special.min_accel <= current_state <= special.max_accel:
//                 offset = accelerate(input, offset)
//         elif special.min_accel <= current_state <= special.max_accel:
//             offset = accelerate(input, offset)
//
// There are some small details left out of the logic above. For example,
// in order to accelerate a state, we need to know which bytes to search for.
// This in turn implies some extra data we need to store in the DFA. To keep
// things compact, we would ideally only store
//
//     N = special.max_accel - special.min_accel + 1
//
// items. But state IDs are premultiplied, which means they are not contiguous.
// So in order to take a state ID and index an array of accelerated structures,
// we need to do:
//
//     i = (state_id - special.min_accel) / stride
//
// (N.B. 'stride' is always a power of 2, so the above can be implemented via
// 'state_id >> stride2', where 'stride2' is x in 2^x=stride.)
//
// Moreover, some of these specialty categories may be empty. For example,
// DFAs are not required to have any match states or any accelerated states.
// In that case, the lower and upper bounds are both set to 0 (the dead state
// ID) and the first `current_state == 0` check subsumes cases where the
// ranges are empty.
//
// Loop unrolling, if applicable, has also been left out of the logic above.
//
// Graphically, the ranges look like this, where asterisks indicate ranges
// that can be empty. Each 'x' is a state.
//
//      quit*
//  dead|
//     ||
//     xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
//     | |             |    | start |                       |
//     | |-------------|    |-------|                       |
//     |   match*   |          |    |                       |
//     |            |          |    |                       |
//     |            |----------|    |                       |
//     |                accel*      |                       |
//     |                            |                       |
//     |                            |                       |
//     |----------------------------|------------------------
//              special                   non-special*
//
// The type parameter `S` refers to the state ID representation used by the
// DFA. Typically, this is u8, u16, u32, u64 or usize.
#[derive(Clone, Copy, Debug)]
pub struct Special<S> {
    /// The identifier of the last special state in a DFA. A state is special
    /// if and only if its identifier is less than or equal to `max`.
    pub max: S,
    pub quit_id: S,
    pub min_match: S,
    pub max_match: S,
    pub min_accel: S,
    pub max_accel: S,
    pub min_start: S,
    pub max_start: S,
}

impl<S: StateID> Special<S> {
    /// Creates a new set of special ranges for a DFA. All ranges are
    /// initially empty (even ranges, like 'start', that cannot ultimately
    /// be empty).
    pub fn new() -> Special<S> {
        Special {
            max: dead_id(),
            quit_id: dead_id(),
            min_match: dead_id(),
            max_match: dead_id(),
            min_accel: dead_id(),
            max_accel: dead_id(),
            min_start: dead_id(),
            max_start: dead_id(),
        }
    }

    /// Convert the state IDs recorded here to a new representation. If the
    /// chosen representation is not big enough to fit the IDs, then an error
    /// is returned.
    pub fn to_sized<A: StateID>(&self) -> Result<Special<A>, Error> {
        if self.max.as_usize() > A::max_id() {
            return Err(Error::state_id_overflow(A::max_id()));
        }
        Ok(Special {
            max: A::from_usize(self.max.as_usize()),
            quit_id: A::from_usize(self.quit_id.as_usize()),
            min_match: A::from_usize(self.min_match.as_usize()),
            max_match: A::from_usize(self.max_match.as_usize()),
            min_accel: A::from_usize(self.min_accel.as_usize()),
            max_accel: A::from_usize(self.max_accel.as_usize()),
            min_start: A::from_usize(self.min_start.as_usize()),
            max_start: A::from_usize(self.max_start.as_usize()),
        })
    }

    /// Remaps all of the special state identifiers using the function given.
    pub fn remap<A: StateID>(&self, map: impl Fn(S) -> A) -> Special<A> {
        Special {
            max: map(self.max),
            quit_id: map(self.quit_id),
            min_match: map(self.min_match),
            max_match: map(self.max_match),
            min_accel: map(self.min_accel),
            max_accel: map(self.max_accel),
            min_start: map(self.min_start),
            max_start: map(self.max_start),
        }
    }

    /// Deserialize the given bytes into special state ranges. If the slice
    /// given is not big enough, then this returns an error. Similarly, if
    /// any of the expected invariants around special state ranges aren't
    /// upheld, an error is returned. Note that this does not guarantee that
    /// the information returned is correct.
    ///
    /// Upon success, this returns the number of bytes read in addition to the
    /// special state IDs themselves.
    pub fn from_bytes(
        mut slice: &[u8],
    ) -> Result<(Special<S>, usize), DeserializeError> {
        let size = size_of::<S>();
        if slice.len() < 8 * size {
            return Err(DeserializeError::buffer_too_small("special state"));
        }

        let max = S::read_bytes(slice);
        slice = &slice[size..];
        let quit_id = S::read_bytes(slice);
        slice = &slice[size..];
        let min_match = S::read_bytes(slice);
        slice = &slice[size..];
        let max_match = S::read_bytes(slice);
        slice = &slice[size..];
        let min_accel = S::read_bytes(slice);
        slice = &slice[size..];
        let max_accel = S::read_bytes(slice);
        slice = &slice[size..];
        let min_start = S::read_bytes(slice);
        slice = &slice[size..];
        let max_start = S::read_bytes(slice);
        slice = &slice[size..];

        let special = Special {
            max,
            quit_id,
            min_match,
            max_match,
            min_accel,
            max_accel,
            min_start,
            max_start,
        };
        special.validate()?;
        Ok((special, special.write_to_len()))
    }

    /// Validate that the information describing special states satisfies
    /// all known invariants.
    pub fn validate(&self) -> Result<(), DeserializeError> {
        // Check that both ends of the range are dead or neither are.
        if self.min_match == dead_id() && self.max_match != dead_id() {
            err!("min_match is dead, but max_match is not");
        }
        if self.min_match != dead_id() && self.max_match == dead_id() {
            err!("max_match is dead, but min_match is not");
        }
        if self.min_accel == dead_id() && self.max_accel != dead_id() {
            err!("min_accel is dead, but max_accel is not");
        }
        if self.min_accel != dead_id() && self.max_accel == dead_id() {
            err!("max_accel is dead, but min_accel is not");
        }
        if self.min_start == dead_id() && self.max_start != dead_id() {
            err!("min_start is dead, but max_start is not");
        }
        if self.min_start != dead_id() && self.max_start == dead_id() {
            err!("max_start is dead, but min_start is not");
        }

        // Check that ranges are well formed.
        if self.min_match > self.max_match {
            err!("min_match should not be greater than max_match");
        }
        if self.min_accel > self.max_accel {
            err!("min_accel should not be greater than max_accel");
        }
        if self.min_start > self.max_start {
            err!("min_start should not be greater than max_start");
        }

        // Check that ranges are ordered with respect to one another.
        if self.matches() && self.quit_id >= self.min_match {
            err!("quit_id should not be greater than min_match");
        }
        if self.accels() && self.quit_id >= self.min_accel {
            err!("quit_id should not be greater than min_accel");
        }
        if self.starts() && self.quit_id >= self.min_start {
            err!("quit_id should not be greater than min_start");
        }
        if self.matches() && self.accels() && self.min_accel < self.min_match {
            err!("min_match should not be greater than min_accel");
        }
        if self.matches() && self.starts() && self.min_start < self.min_match {
            err!("min_match should not be greater than min_start");
        }
        if self.accels() && self.starts() && self.min_start < self.min_accel {
            err!("min_accel should not be greater than min_start");
        }

        // Check that max is at least as big as everything else.
        if self.max < self.quit_id {
            err!("quit_id should not be greater than max");
        }
        if self.max < self.max_match {
            err!("max_match should not be greater than max");
        }
        if self.max < self.max_accel {
            err!("max_accel should not be greater than max");
        }
        if self.max < self.max_start {
            err!("max_start should not be greater than max");
        }

        Ok(())
    }

    /// Validate that the special state information is compatible with the
    /// given state count.
    pub fn validate_state_count(
        &self,
        count: usize,
        stride2: usize,
    ) -> Result<(), DeserializeError> {
        // We assume that 'validate' has already passed, so we know that 'max'
        // is truly the max. So that's all we need to check.
        if (self.max.as_usize() >> stride2) >= count {
            err!("max should not be greater than or equal to state count");
        }
        Ok(())
    }

    /// Write the IDs and ranges for special states to the given byte buffer.
    /// The buffer given must have enough room to store all data, otherwise
    /// this will return an error. The number of bytes written is returned
    /// on success. The number of bytes written is guaranteed to be a multiple
    /// of 8.
    pub fn write_to<E: Endian>(
        &self,
        mut dst: &mut [u8],
    ) -> Result<usize, SerializeError> {
        use crate::bytes::write_state_id as write;

        if dst.len() < self.write_to_len() {
            return Err(SerializeError::buffer_too_small("special state ids"));
        }

        let start = dst.as_ptr() as usize;
        let mut nwrite = 0;

        nwrite += write::<E, _>(self.max, &mut dst[nwrite..]);
        nwrite += write::<E, _>(self.quit_id, &mut dst[nwrite..]);
        nwrite += write::<E, _>(self.min_match, &mut dst[nwrite..]);
        nwrite += write::<E, _>(self.max_match, &mut dst[nwrite..]);
        nwrite += write::<E, _>(self.min_accel, &mut dst[nwrite..]);
        nwrite += write::<E, _>(self.max_accel, &mut dst[nwrite..]);
        nwrite += write::<E, _>(self.min_start, &mut dst[nwrite..]);
        nwrite += write::<E, _>(self.max_start, &mut dst[nwrite..]);

        assert_eq!(
            self.write_to_len(),
            nwrite,
            "expected to write certain number of bytes",
        );
        assert_eq!(
            nwrite % 8,
            0,
            "expected to write multiple of 8 bytes for special states",
        );
        Ok(nwrite)
    }

    /// Returns the total number of bytes written by `write_to`.
    pub fn write_to_len(&self) -> usize {
        size_of::<S>() * 8
    }

    /// Sets the maximum special state ID based on the current values. This
    /// should be used once all possible state IDs are set.
    pub fn set_max(&mut self) {
        self.max = cmp::max(
            self.quit_id,
            cmp::max(self.max_match, cmp::max(self.max_accel, self.max_start)),
        );
    }

    /// Returns true if and only if the given state ID is a special state.
    pub fn is_special_state(&self, id: S) -> bool {
        id <= self.max
    }

    /// Returns true if and only if the given state ID is a dead state.
    pub fn is_dead_state(&self, id: S) -> bool {
        id == dead_id()
    }

    /// Returns true if and only if the given state ID is a quit state.
    pub fn is_quit_state(&self, id: S) -> bool {
        !self.is_dead_state(id) && self.quit_id == id
    }

    /// Returns true if and only if the given state ID is a match state.
    pub fn is_match_state(&self, id: S) -> bool {
        !self.is_dead_state(id) && self.min_match <= id && id <= self.max_match
    }

    /// Returns true if and only if the given state ID is an accel state.
    pub fn is_accel_state(&self, id: S) -> bool {
        !self.is_dead_state(id) && self.min_accel <= id && id <= self.max_accel
    }

    /// Returns true if and only if the given state ID is a start state.
    pub fn is_start_state(&self, id: S) -> bool {
        !self.is_dead_state(id) && self.min_start <= id && id <= self.max_start
    }

    /// Returns the total number of match states for a dense table based DFA.
    pub fn match_len(&self, stride: usize) -> usize {
        if self.matches() {
            (self.max_match.as_usize() - self.min_match.as_usize() + stride)
                / stride
        } else {
            0
        }
    }

    /// Returns true if and only if there is at least one match state.
    pub fn matches(&self) -> bool {
        self.min_match != dead_id()
    }

    /// Returns the total number of accel states.
    pub fn accel_len(&self) -> usize {
        if self.accels() {
            self.max_accel.as_usize() - self.min_accel.as_usize() + 1
        } else {
            0
        }
    }

    /// Returns true if and only if there is at least one accel state.
    pub fn accels(&self) -> bool {
        self.min_accel != dead_id()
    }

    /// Returns the total number of start states.
    pub fn start_len(&self) -> usize {
        if self.starts() {
            self.max_start.as_usize() - self.min_start.as_usize() + 1
        } else {
            0
        }
    }

    /// Returns true if and only if there is at least one start state.
    pub fn starts(&self) -> bool {
        self.min_start != dead_id()
    }
}
