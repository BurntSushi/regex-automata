#![allow(warnings)]

use state_id::StateID;

// BREADCRUMBS:
//
// So it looks like we probably want DenseDFA to be this:
//
//   enum DenseDFA {
//     Standard(dense::Standard),
//     Premultiplied(dense::Premultiplied),
//     ByteClass(dense::ByteClass),
//     PremultipliedByteClass(dense::PremultipliedByteClass),
//   }
//
// where each of the dense::* types wrap DenseDFARepr, which is basically what
// DenseDFA is now. We can then override the is_match/... routines on DenseDFA
// to do case analysis once and dispatch to its inner DFA.
//
// Determinization and minimization then operate on DenseDFARepr.
//
// Do we still need DenseDFAKind? It looks like we do, so maybe determinization
// and minimization really should just use DenseDFA.
//
// Now, what about DenseDFARef? Does it really need to copy the structure of
// DenseDFA, and therefore need dense::StandardRef and so on?
//
// OK, it looks like we need to add a second type parameter to DenseDFA and
// SparseDFA. Specifically, DenseDFA should use T for its transition table,
// where T: AsRef<[StateID]>. That way, the same types can be used for both
// Vec<S> and &[S]. This has to be threaded all the way down to DenseDFARepr.
//
// The upshot here is that Regex stays the course with one type parameter:
// an implementation of the DFA trait. The ID associated type avoids needing
// to fuss with state IDs with the Regex type directly.

pub trait DFA {
    type ID: StateID;

    fn start_state(&self) -> Self::ID;

    fn is_match_state(&self, id: Self::ID) -> bool;

    fn is_possible_match_state(&self, id: Self::ID) -> bool;

    fn is_dead_state(&self, id: Self::ID) -> bool;

    fn next_state(&self, current: Self::ID, input: u8) -> Self::ID;

    unsafe fn next_state_unchecked(
        &self,
        current: Self::ID,
        input: u8,
    ) -> Self::ID;

    /// Returns true if and only if the given bytes match this DFA.
    ///
    /// This routine may short circuit if it knows that scanning future input
    /// will never lead to a different result. In particular, if a DFA enters
    /// a match state or a dead state, then this routine will return `true` or
    /// `false`, respectively, without inspecting any future input.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{DFA, DenseDFA};
    ///
    /// # fn example() -> Result<(), regex_automata::Error> {
    /// let dfa = DenseDFA::new("foo[0-9]+bar")?;
    /// assert_eq!(true, dfa.is_match(b"foo12345bar"));
    /// assert_eq!(false, dfa.is_match(b"foobar"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    fn is_match(&self, bytes: &[u8]) -> bool {
        let mut state = self.start_state();
        if self.is_possible_match_state(state) {
            return self.is_match_state(state);
        }
        for &b in bytes.iter() {
            state = unsafe { self.next_state_unchecked(state, b) };
            if self.is_possible_match_state(state) {
                return self.is_match_state(state);
            }
        }
        false
    }

    /// Returns the first position at which a match is found.
    ///
    /// This routine stops scanning input in precisely the same circumstances
    /// as `is_match`. The key difference is that this routine returns the
    /// position at which it stopped scanning input if and only if a match
    /// was found. If no match is found, then `None` is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{DFA, DenseDFA};
    ///
    /// # fn example() -> Result<(), regex_automata::Error> {
    /// let dfa = DenseDFA::new("foo[0-9]+")?;
    /// assert_eq!(Some(4), dfa.shortest_match(b"foo12345"));
    ///
    /// // Normally, the end of the leftmost first match here would be 3,
    /// // but the shortest match semantics detect a match earlier.
    /// let dfa = DenseDFA::new("abc|a")?;
    /// assert_eq!(Some(1), dfa.shortest_match(b"abc"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    fn shortest_match(&self, bytes: &[u8]) -> Option<usize> {
        let mut state = self.start_state();
        if self.is_possible_match_state(state) {
            return if self.is_dead_state(state) { None } else { Some(0) };
        }
        for (i, &b) in bytes.iter().enumerate() {
            state = unsafe { self.next_state_unchecked(state, b) };
            if self.is_possible_match_state(state) {
                return
                    if self.is_dead_state(state) {
                        None
                    } else {
                        Some(i + 1)
                    };
            }
        }
        None
    }

    /// Returns the end offset of the leftmost first match. If no match exists,
    /// then `None` is returned.
    ///
    /// The "leftmost first" match corresponds to the match with the smallest
    /// starting offset, but where the end offset is determined by preferring
    /// earlier branches in the original regular expression. For example,
    /// `Sam|Samwise` will match `Sam` in `Samwise`, but `Samwise|Sam` will
    /// match `Samwise` in `Samwise`.
    ///
    /// Generally speaking, the "leftmost first" match is how most backtracking
    /// regular expressions tend to work. This is in contrast to POSIX-style
    /// regular expressions that yield "leftmost longest" matches. Namely,
    /// both `Sam|Samwise` and `Samwise|Sam` match `Samwise` when using
    /// leftmost longest semantics.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{DFA, DenseDFA};
    ///
    /// # fn example() -> Result<(), regex_automata::Error> {
    /// let dfa = DenseDFA::new("foo[0-9]+")?;
    /// assert_eq!(Some(8), dfa.find(b"foo12345"));
    ///
    /// // Even though a match is found after reading the first byte (`a`),
    /// // the leftmost first match semantics demand that we find the earliest
    /// // match that prefers earlier parts of the pattern over latter parts.
    /// let dfa = DenseDFA::new("abc|a")?;
    /// assert_eq!(Some(3), dfa.find(b"abc"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    fn find(&self, bytes: &[u8]) -> Option<usize> {
        let mut state = self.start_state();
        let mut last_match =
            if self.is_dead_state(state) {
                return None;
            } else if self.is_match_state(state) {
                Some(0)
            } else {
                None
            };
        for (i, &b) in bytes.iter().enumerate() {
            state = unsafe { self.next_state_unchecked(state, b) };
            if self.is_possible_match_state(state) {
                if self.is_dead_state(state) {
                    return last_match;
                }
                last_match = Some(i + 1);
            }
        }
        last_match
    }

    /// Returns the start offset of the leftmost first match in reverse, by
    /// searching from the end of the input towards the start of the input. If
    /// no match exists, then `None` is returned.
    ///
    /// This routine is principally useful when used in conjunction with the
    /// [`DenseDFABuilder::reverse`](struct.DenseDFABuilder.html#method.reverse)
    /// configuration knob. In general, it's unlikely to be correct to use both
    /// `find` and `rfind` with the same DFA.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{DFA, DenseDFABuilder};
    ///
    /// # fn example() -> Result<(), regex_automata::Error> {
    /// let dfa = DenseDFABuilder::new().reverse(true).build("foo[0-9]+")?;
    /// assert_eq!(Some(0), dfa.rfind(b"foo12345"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    fn rfind(&self, bytes: &[u8]) -> Option<usize> {
        let mut state = self.start_state();
        let mut last_match =
            if self.is_dead_state(state) {
                return None;
            } else if self.is_match_state(state) {
                Some(0)
            } else {
                None
            };
        for (i, &b) in bytes.iter().enumerate().rev() {
            state = unsafe { self.next_state_unchecked(state, b) };
            if self.is_possible_match_state(state) {
                if self.is_dead_state(state) {
                    return last_match;
                }
                last_match = Some(i);
            }
        }
        last_match
    }
}

impl<'a, T: DFA> DFA for &'a T {
    type ID = T::ID;

    fn start_state(&self) -> Self::ID {
        (**self).start_state()
    }

    fn is_match_state(&self, id: Self::ID) -> bool {
        (**self).is_match_state(id)
    }

    fn is_possible_match_state(&self, id: Self::ID) -> bool {
        (**self).is_possible_match_state(id)
    }

    fn is_dead_state(&self, id: Self::ID) -> bool {
        (**self).is_dead_state(id)
    }

    fn next_state(&self, current: Self::ID, input: u8) -> Self::ID {
        (**self).next_state(current, input)
    }

    unsafe fn next_state_unchecked(
        &self,
        current: Self::ID,
        input: u8,
    ) -> Self::ID {
        (**self).next_state_unchecked(current, input)
    }
}
