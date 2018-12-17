use std::fmt;

use builder::MatcherBuilder;
use dfa::DFA;
use dfa_ref::DFARef;
use error::Result;
use state_id::StateID;

/// A regular expression that uses deterministic finite automata for fast
/// searching.
#[derive(Clone)]
pub struct Matcher<'a, S = usize> {
    forward: OwnOrBorrow<'a, S>,
    reverse: OwnOrBorrow<'a, S>,
}

impl Matcher<'static, usize> {
    /// Parse the given regular expression using a default configuration and
    /// return the corresponding matcher.
    ///
    /// The default configuration uses `usize` for state IDs, premultiplies
    /// them and reduces the alphabet size by splitting bytes into equivalence
    /// classes. The underlying DFAs are *not* minimized.
    ///
    /// If you want a non-default configuration, then use the
    /// [`MatcherBuilder`](struct.MatcherBuilder.html)
    /// to set your own configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::Matcher;
    ///
    /// # fn example() -> Result<(), regex_automata::Error> {
    /// let matcher = Matcher::new("foo[0-9]+bar")?;
    /// assert_eq!(Some((3, 14)), matcher.find(b"zzzfoo12345barzzz"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn new(pattern: &str) -> Result<Matcher<'static, usize>> {
        MatcherBuilder::new().build(pattern)
    }
}

impl<'a, S: StateID> Matcher<'a, S> {
    /// Returns true if and only if the given bytes match.
    ///
    /// This routine may short circuit if it knows that scanning future input
    /// will never lead to a different result. In particular, if the underlying
    /// DFA enters a match state or a dead state, then this routine will return
    /// `true` or `false`, respectively, without inspecting any future input.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::Matcher;
    ///
    /// # fn example() -> Result<(), regex_automata::Error> {
    /// let re = Matcher::new("foo[0-9]+bar")?;
    /// assert_eq!(true, re.is_match(b"foo12345bar"));
    /// assert_eq!(false, re.is_match(b"foobar"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn is_match(&self, input: &[u8]) -> bool {
        self.forward().is_match(input)
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
    /// use regex_automata::Matcher;
    ///
    /// # fn example() -> Result<(), regex_automata::Error> {
    /// let re = Matcher::new("foo[0-9]+")?;
    /// assert_eq!(Some(4), re.shortest_match(b"foo12345"));
    ///
    /// // Normally, the end of the leftmost first match here would be 3,
    /// // but the shortest match semantics detect a match earlier.
    /// let re = Matcher::new("abc|a")?;
    /// assert_eq!(Some(1), re.shortest_match(b"abc"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn shortest_match(&self, input: &[u8]) -> Option<usize> {
        self.forward().shortest_match(input)
    }

    /// Returns the start and end offset of the leftmost first match. If no
    /// match exists, then `None` is returned.
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
    /// use regex_automata::Matcher;
    ///
    /// # fn example() -> Result<(), regex_automata::Error> {
    /// let re = Matcher::new("foo[0-9]+")?;
    /// assert_eq!(Some((3, 11)), re.find(b"zzzfoo12345zzz"));
    ///
    /// // Even though a match is found after reading the first byte (`a`),
    /// // the leftmost first match semantics demand that we find the earliest
    /// // match that prefers earlier parts of the pattern over latter parts.
    /// let re = Matcher::new("abc|a")?;
    /// assert_eq!(Some((0, 3)), re.find(b"abc"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn find(&self, input: &[u8]) -> Option<(usize, usize)> {
        let end = match self.forward().find(input) {
            None => return None,
            Some(end) => end,
        };
        let start = self
            .reverse()
            .rfind(&input[..end])
            .expect("reverse search must match if forward search does");
        Some((start, end))
    }
}

impl<'a, S: StateID> Matcher<'a, S> {
    /// Build a new matcher from its constituent forward and reverse DFAs.
    ///
    /// It's not currently possible for a caller using this crate's public API
    /// to correctly use this method since the `DFABuilder` does not expose the
    /// necessary options to correctly build the reverse DFA.
    pub(crate) fn from_dfa(
        forward: DFA<S>,
        reverse: DFA<S>,
    ) -> Matcher<'static, S> {
        Matcher {
            forward: OwnOrBorrow::Owned(forward),
            reverse: OwnOrBorrow::Owned(reverse),
        }
    }

    /// Build a new matcher from its constituent forward and reverse borrowed
    /// DFAs.
    ///
    /// This is useful when deserializing a matcher from some arbitrary
    /// memory region. Note that currently, it is not possible to correctly
    /// build these DFAs directly using a `DFABuilder`. In particular, the
    /// forward and reverse DFAs given here *must* be DFAs corresponding to a
    /// previously built regex and retrieved using the
    /// [`Matcher::forward`](struct.Matcher.html#method.forward)
    /// and
    /// [`Matcher::reverse`](struct.Matcher.html#method.reverse)
    /// methods.
    ///
    /// # Example
    ///
    /// This example is a bit a contrived. The usual use of these methods
    /// would involve serializing `initial_re` somewhere and then deserializing
    /// it later to build a regex.
    ///
    /// ```
    /// use regex_automata::Matcher;
    ///
    /// # fn example() -> Result<(), regex_automata::Error> {
    /// let initial_re = Matcher::new("foo[0-9]+")?;
    /// assert_eq!(true, initial_re.is_match(b"foo123"));
    ///
    /// let (fwd, rev) = (initial_re.forward(), initial_re.reverse());
    /// let re = Matcher::from_dfa_refs(fwd, rev);
    /// assert_eq!(true, initial_re.is_match(b"foo123"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn from_dfa_refs(
        forward: DFARef<'a, S>,
        reverse: DFARef<'a, S>,
    ) -> Matcher<'a, S> {
        Matcher {
            forward: OwnOrBorrow::Borrowed(forward),
            reverse: OwnOrBorrow::Borrowed(reverse),
        }
    }

    /// Return the underlying DFA responsible for forward matching.
    pub fn forward<'b>(&'b self) -> DFARef<'b, S> {
        match self.forward {
            OwnOrBorrow::Owned(ref dfa) => dfa.as_dfa_ref(),
            OwnOrBorrow::Borrowed(dfa) => dfa,
        }
    }

    /// Return the underlying DFA responsible for reverse matching.
    pub fn reverse<'b>(&'b self) -> DFARef<'b, S> {
        match self.reverse {
            OwnOrBorrow::Owned(ref dfa) => dfa.as_dfa_ref(),
            OwnOrBorrow::Borrowed(dfa) => dfa,
        }
    }
}

#[derive(Clone)]
enum OwnOrBorrow<'a, S = usize> {
    Owned(DFA<S>),
    Borrowed(DFARef<'a, S>),
}

impl<'a, S: StateID> fmt::Debug for Matcher<'a, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Matcher")
            .field("forward", &self.forward)
            .field("reverse", &self.reverse)
            .finish()
    }
}

impl<'a, S: StateID> fmt::Debug for OwnOrBorrow<'a, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            OwnOrBorrow::Owned(ref dfa) => {
                f.debug_tuple("Owned").field(dfa).finish()
            }
            OwnOrBorrow::Borrowed(ref dfa) => {
                f.debug_tuple("Borrowed").field(dfa).finish()
            }
        }
    }
}
