use builder::RegexBuilder;
use dense::DenseDFA;
use dense_ref::DenseDFARef;
use error::Result;
use state_id::StateID;

/// A regular expression that uses deterministic finite automata for fast
/// searching.
#[derive(Clone, Debug)]
pub struct Regex<'a, S: StateID = usize> {
    forward: OwnOrBorrow<'a, S>,
    reverse: OwnOrBorrow<'a, S>,
}

impl Regex<'static, usize> {
    /// Parse the given regular expression using a default configuration and
    /// return the corresponding regex.
    ///
    /// The default configuration uses `usize` for state IDs, premultiplies
    /// them and reduces the alphabet size by splitting bytes into equivalence
    /// classes. The underlying DFAs are *not* minimized.
    ///
    /// If you want a non-default configuration, then use the
    /// [`RegexBuilder`](struct.RegexBuilder.html)
    /// to set your own configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::Regex;
    ///
    /// # fn example() -> Result<(), regex_automata::Error> {
    /// let re = Regex::new("foo[0-9]+bar")?;
    /// assert_eq!(Some((3, 14)), re.find(b"zzzfoo12345barzzz"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn new(pattern: &str) -> Result<Regex<'static, usize>> {
        RegexBuilder::new().build(pattern)
    }
}

impl<'a, S: StateID> Regex<'a, S> {
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
    /// use regex_automata::Regex;
    ///
    /// # fn example() -> Result<(), regex_automata::Error> {
    /// let re = Regex::new("foo[0-9]+bar")?;
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
    /// use regex_automata::Regex;
    ///
    /// # fn example() -> Result<(), regex_automata::Error> {
    /// let re = Regex::new("foo[0-9]+")?;
    /// assert_eq!(Some(4), re.shortest_match(b"foo12345"));
    ///
    /// // Normally, the end of the leftmost first match here would be 3,
    /// // but the shortest match semantics detect a match earlier.
    /// let re = Regex::new("abc|a")?;
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
    /// use regex_automata::Regex;
    ///
    /// # fn example() -> Result<(), regex_automata::Error> {
    /// let re = Regex::new("foo[0-9]+")?;
    /// assert_eq!(Some((3, 11)), re.find(b"zzzfoo12345zzz"));
    ///
    /// // Even though a match is found after reading the first byte (`a`),
    /// // the leftmost first match semantics demand that we find the earliest
    /// // match that prefers earlier parts of the pattern over latter parts.
    /// let re = Regex::new("abc|a")?;
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

    /// Returns an iterator over all non-overlapping leftmost first matches
    /// in the given bytes. If no match exists, then the iterator yields no
    /// elements.
    ///
    /// Note that if the regex can match the empty string, then it is
    /// possible for the iterator to yield a zero-width match at a location
    /// that is not a valid UTF-8 boundary (for example, between the code units
    /// of a UTF-8 encoded codepoint). This can happen regardless of whether
    /// [`allow_invalid_utf8`](struct.RegexBuilder.html#method.allow_invalid_utf8)
    /// was enabled or not.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::Regex;
    ///
    /// # fn example() -> Result<(), regex_automata::Error> {
    /// let re = Regex::new("foo[0-9]+")?;
    /// let text = b"foo1 foo12 foo123";
    /// let matches: Vec<(usize, usize)> = re.find_iter(text).collect();
    /// assert_eq!(matches, vec![(0, 4), (5, 10), (11, 17)]);
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn find_iter<'r, 't>(
        &'r self,
        input: &'t [u8],
    ) -> Matches<'a, 'r, 't, S> {
        Matches::new(self, input)
    }
}

impl<'a, S: StateID> Regex<'a, S> {
    /// Build a new regex from its constituent forward and reverse DFAs.
    ///
    /// It's not currently possible for a caller using this crate's public API
    /// to correctly use this method since the `DenseDFABuilder` does not expose the
    /// necessary options to correctly build the reverse DFA.
    pub(crate) fn from_dfas(
        forward: DenseDFA<S>,
        reverse: DenseDFA<S>,
    ) -> Regex<'static, S> {
        Regex {
            forward: OwnOrBorrow::Owned(forward),
            reverse: OwnOrBorrow::Owned(reverse),
        }
    }

    /// Build a new regex from its constituent forward and reverse borrowed
    /// DFAs.
    ///
    /// This is useful when deserializing a regex from some arbitrary
    /// memory region. Note that currently, it is not possible to correctly
    /// build these DFAs directly using a `DenseDFABuilder`. In particular, the
    /// forward and reverse DFAs given here *must* be DFAs corresponding to a
    /// previously built regex and retrieved using the
    /// [`Regex::forward`](struct.Regex.html#method.forward)
    /// and
    /// [`Regex::reverse`](struct.Regex.html#method.reverse)
    /// methods.
    ///
    /// # Example
    ///
    /// This example is a bit a contrived. The usual use of these methods
    /// would involve serializing `initial_re` somewhere and then deserializing
    /// it later to build a regex.
    ///
    /// ```
    /// use regex_automata::Regex;
    ///
    /// # fn example() -> Result<(), regex_automata::Error> {
    /// let initial_re = Regex::new("foo[0-9]+")?;
    /// assert_eq!(true, initial_re.is_match(b"foo123"));
    ///
    /// let (fwd, rev) = (initial_re.forward(), initial_re.reverse());
    /// let re = Regex::from_dfa_refs(fwd, rev);
    /// assert_eq!(true, initial_re.is_match(b"foo123"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn from_dfa_refs(
        forward: DenseDFARef<'a, S>,
        reverse: DenseDFARef<'a, S>,
    ) -> Regex<'a, S> {
        Regex {
            forward: OwnOrBorrow::Borrowed(forward),
            reverse: OwnOrBorrow::Borrowed(reverse),
        }
    }

    /// Return the underlying DFA responsible for forward matching.
    pub fn forward<'b>(&'b self) -> DenseDFARef<'b, S> {
        match self.forward {
            OwnOrBorrow::Owned(ref dfa) => dfa.as_dfa_ref(),
            OwnOrBorrow::Borrowed(dfa) => dfa,
        }
    }

    /// Return the underlying DFA responsible for reverse matching.
    pub fn reverse<'b>(&'b self) -> DenseDFARef<'b, S> {
        match self.reverse {
            OwnOrBorrow::Owned(ref dfa) => dfa.as_dfa_ref(),
            OwnOrBorrow::Borrowed(dfa) => dfa,
        }
    }
}

impl<'a, S: StateID> Regex<'a, S> {
    /// Create a new regex whose match semantics are equivalent to this regex,
    /// but attempt to use `u8` for the representation of state identifiers.
    /// If `u8` is insufficient to represent all state identifiers in this
    /// regex, then this returns an error.
    ///
    /// This is a convenience routine for `to_sized::<u8>()`.
    pub fn to_u8(&self) -> Result<Regex<'static, u8>> {
        self.to_sized()
    }

    /// Create a new regex whose match semantics are equivalent to this regex,
    /// but attempt to use `u16` for the representation of state identifiers.
    /// If `u16` is insufficient to represent all state identifiers in this
    /// regex, then this returns an error.
    ///
    /// This is a convenience routine for `to_sized::<u16>()`.
    pub fn to_u16(&self) -> Result<Regex<'static, u16>> {
        self.to_sized()
    }

    /// Create a new regex whose match semantics are equivalent to this regex,
    /// but attempt to use `u32` for the representation of state identifiers.
    /// If `u32` is insufficient to represent all state identifiers in this
    /// regex, then this returns an error.
    ///
    /// This is a convenience routine for `to_sized::<u32>()`.
    pub fn to_u32(&self) -> Result<Regex<'static, u32>> {
        self.to_sized()
    }

    /// Create a new regex whose match semantics are equivalent to this regex,
    /// but attempt to use `u64` for the representation of state identifiers.
    /// If `u64` is insufficient to represent all state identifiers in this
    /// regex, then this returns an error.
    ///
    /// This is a convenience routine for `to_sized::<u64>()`.
    pub fn to_u64(&self) -> Result<Regex<'static, u64>> {
        self.to_sized()
    }

    /// Create a new regex whose match semantics are equivalent to this regex,
    /// but attempt to use `T` for the representation of state identifiers. If
    /// `T` is insufficient to represent all state identifiers in this regex,
    /// then this returns an error.
    ///
    /// An alternative way to construct such a regex is to use
    /// [`RegexBuilder::build_with_size`](struct.RegexBuilder.html#method.build_with_size).
    /// In general, using the builder is preferred since it will use the
    /// given state identifier representation throughout determinization (and
    /// minimization, if done), and thereby using less memory throughout the
    /// entire construction process. However, these routines are necessary
    /// in cases where, say, a minimized regex could fit in a smaller state
    /// identifier representation, but the initial determinized regex would
    /// not.
    pub fn to_sized<T: StateID>(&self) -> Result<Regex<'static, T>> {
        use self::OwnOrBorrow::*;

        let forward = match self.forward {
            Owned(ref dfa) => dfa.to_sized::<T>()?,
            Borrowed(dfa) => DenseDFA::from_dfa_ref(dfa).to_sized::<T>()?,
        };
        let reverse = match self.reverse {
            Owned(ref dfa) => dfa.to_sized::<T>()?,
            Borrowed(dfa) => DenseDFA::from_dfa_ref(dfa).to_sized::<T>()?,
        };
        Ok(Regex::from_dfas(forward, reverse))
    }
}

#[derive(Clone, Debug)]
enum OwnOrBorrow<'a, S: StateID = usize> {
    Owned(DenseDFA<S>),
    Borrowed(DenseDFARef<'a, S>),
}

/// An iterator over all non-overlapping matches for a particular search.
///
/// The iterator yields a `(usize, usize)` value until no more matches could be
/// found. The first `usize` is the start of the match (inclusive) while the
/// second `usize` is the end of the match (exclusive).
///
/// `S` is the type used to represent state identifiers in the underlying
/// regex. The lifetime variables are as follows:
///
/// * `'d` is the lifetime of the underlying DFA transition table. If the regex
///   is built using the owned [`DenseDFA`](struct.DenseDFA.html) type, then this is
///   always equivalent to the `'static` lifetime.
/// * `'r` is the lifetime of the regular expression value itself.
/// * `'t` is the lifetime of the text being searched.
#[derive(Clone, Debug)]
pub struct Matches<'d, 'r, 't, S: StateID = usize> {
    re: &'r Regex<'d, S>,
    text: &'t [u8],
    last_end: usize,
    last_match: Option<usize>,
}

impl<'d, 'r, 't, S: StateID> Matches<'d, 'r, 't, S> {
    fn new(re: &'r Regex<'d, S>, text: &'t [u8]) -> Matches<'d, 'r, 't, S> {
        Matches {
            re: re,
            text: text,
            last_end: 0,
            last_match: None,
        }
    }
}

impl<'d, 'r, 't, S: StateID> Iterator for Matches<'d, 'r, 't, S> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        if self.last_end > self.text.len() {
            return None;
        }
        let (s, e) = match self.re.find(&self.text[self.last_end..]) {
            None => return None,
            Some((s, e)) => (self.last_end + s, self.last_end + e),
        };
        if s == e {
            // This is an empty match. To ensure we make progress, start
            // the next search at the smallest possible starting position
            // of the next match following this one.
            self.last_end = e + 1;
            // Don't accept empty matches immediately following a match.
            // Just move on to the next match.
            if Some(e) == self.last_match {
                return self.next();
            }
        } else {
            self.last_end = e;
        }
        self.last_match = Some(e);
        Some((s, e))
    }
}
