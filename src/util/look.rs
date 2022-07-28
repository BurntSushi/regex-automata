use core::convert::TryFrom;

use crate::util::{alphabet::ByteClassSet, int::U32, utf8};

/// A look-around assertion.
///
/// A simulation of the NFA can only move through conditional epsilon
/// transitions if the current position satisfies some look-around property.
/// Some assertions are look-behind (`StartLine`, `StartText`), some assertions
/// are look-ahead (`EndLine`, `EndText`) while other assertions are both
/// look-behind and look-ahead (`WordBoundary*`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Look {
    /// The current position is the beginning of the haystack (at position
    /// `0`).
    Start = 1 << 0,
    /// The current position is the end of the haystack (at position
    /// `haystack.len()`).
    End = 1 << 1,
    /// The previous position is either `\n` or the current position is the
    /// beginning of the haystack (at position `0`).
    StartLF = 1 << 2,
    /// The next position is either `\n` or the current position is the end of
    /// the haystack (at position `haystack.len()`).
    EndLF = 1 << 3,
    /// When tested at position `i`, where `p=haystack[i-1]` and
    /// `n=haystack[i]`, this assertion passes if and only if `is_word(p)
    /// != is_word(n)`. If `i=0`, then `is_word(p)=false` and if
    /// `i=haystack.len()`, then `is_word(n)=false`.
    WordAscii = 1 << 4,
    /// Same as for `WordBoundaryAscii`, but requires that
    /// `is_word(p) == is_word(n)`.
    ///
    /// Note that it is possible for this assertion to match at positions that
    /// split the UTF-8 encoding of a codepoint. For this reason, this may only
    /// be used when UTF-8 mode is disabled in the regex syntax.
    WordAsciiNegate = 1 << 5,
    /// When tested at position `i`, where `p=decode_utf8_rev(&haystack[..i])`
    /// and `n=decode_utf8(&haystack[i..])`, this assertion passes if and only
    /// if `is_word(p) != is_word(n)`. If `i=0`, then `is_word(p)=false` and if
    /// `i=haystack.len()`, then `is_word(n)=false`.
    WordUnicode = 1 << 6,
    /// Same as for `WordBoundaryUnicode`, but requires that
    /// `is_word(p) == is_word(n)`.
    WordUnicodeNegate = 1 << 7,
}

impl Look {
    const COUNT: usize = 8;

    /// Create a look-around assertion from its corresponding integer (as
    /// defined in `Look`). If the given integer does not correspond to any
    /// assertion, then `None` is returned.
    pub const fn from_repr(n: u8) -> Option<Look> {
        const CAPACITY: usize = 256;
        const fn mkmap() -> [Option<Look>; CAPACITY] {
            let mut map = [None; CAPACITY];
            let mut i = 0;
            while i < Look::COUNT {
                let look = Look::from_index_unchecked(i);
                // FIXME: Use as_usize() once const functions in traits are
                // stable.
                map[look.as_repr() as usize] = Some(look);
                i += 1;
            }
            map
        }
        const MAP: [Option<Look>; CAPACITY] = mkmap();
        // FIXME: Use as_usize() once const functions in traits are stable.
        MAP[n as usize]
    }

    pub const fn from_index(index: usize) -> Option<Look> {
        if index < Look::COUNT {
            Some(Look::from_index_unchecked(index))
        } else {
            None
        }
    }

    pub const fn from_index_unchecked(index: usize) -> Look {
        const BY_INDEX: [Look; Look::COUNT] = [
            Look::Start,
            Look::End,
            Look::StartLF,
            Look::EndLF,
            Look::WordAscii,
            Look::WordAsciiNegate,
            Look::WordUnicode,
            Look::WordUnicodeNegate,
        ];
        BY_INDEX[index]
    }

    /// Return the underlying representation of this look-around enumeration
    /// as an integer. Giving the return value to the [`Look::from_repr`]
    /// constructor is guaranteed to return the same look-around variant that
    /// one started with.
    pub const fn as_repr(self) -> u8 {
        // AFAIK, 'as' is the only way to zero-cost convert an int enum to an
        // actual int.
        self as u8
    }

    pub const fn as_index(self) -> usize {
        // OK since trailing zeroes will always be <= u8::MAX. (The only
        // way this would be false is if we defined more than 255 distinct
        // look-around assertions, which seems highly improbable.)
        //
        // FIXME: Use as_usize() once const functions in traits are stable.
        self.as_repr().trailing_zeros() as usize
    }

    pub const fn as_char(self) -> char {
        match self {
            Look::Start => 'A',
            Look::End => 'z',
            Look::StartLF => '^',
            Look::EndLF => '$',
            Look::WordAscii => 'b',
            Look::WordAsciiNegate => 'B',
            Look::WordUnicode => '𝛃',
            Look::WordUnicodeNegate => '𝚩',
        }
    }

    /// Flip the look-around assertion to its equivalent for reverse searches.
    /// For example, `StartLine` gets translated to `EndLine`.
    pub const fn reversed(self) -> Look {
        match self {
            Look::Start => Look::End,
            Look::End => Look::Start,
            Look::StartLF => Look::EndLF,
            Look::EndLF => Look::StartLF,
            Look::WordAscii => Look::WordAscii,
            Look::WordAsciiNegate => Look::WordAsciiNegate,
            Look::WordUnicode => Look::WordUnicode,
            Look::WordUnicodeNegate => Look::WordUnicodeNegate,
        }
    }

    /// Returns true when the position `at` in `haystack` satisfies this
    /// look-around assertion.
    ///
    /// This panics if `at > haystack.len()`.
    pub fn matches(self, haystack: &[u8], at: usize) -> bool {
        match self {
            Look::Start => is_start(haystack, at),
            Look::End => is_end(haystack, at),
            Look::StartLF => is_start_lf(haystack, at),
            Look::EndLF => is_end_lf(haystack, at),
            Look::WordAscii => is_word_ascii(haystack, at),
            Look::WordAsciiNegate => is_word_ascii_negate(haystack, at),
            Look::WordUnicode => is_word_unicode(haystack, at),
            Look::WordUnicodeNegate => is_word_unicode_negate(haystack, at),
        }
    }

    /// Split up the given byte classes into equivalence classes in a way that
    /// is consistent with this look-around assertion.
    pub(crate) fn add_to_byteset(self, set: &mut ByteClassSet) {
        match self {
            Look::Start | Look::End => {}
            Look::StartLF | Look::EndLF => {
                set.set_range(b'\n', b'\n');
            }
            Look::WordAscii
            | Look::WordAsciiNegate
            | Look::WordUnicode
            | Look::WordUnicodeNegate => {
                // We need to mark all ranges of bytes whose pairs result in
                // evaluating \b differently. This isn't technically correct
                // for Unicode word boundaries, but DFAs can't handle those
                // anyway, and thus, the byte classes don't need to either
                // since they are themselves only used in DFAs.
                //
                // FIXME: It seems like the calls to 'set_range' here are
                // completely invariant, which means we could just hard-code
                // them here without needing to write a loop. And we only need
                // to do this dance at most once per regex.
                //
                // FIXME: Is this correct for \B?
                let iswb = utf8::is_word_byte;
                // This unwrap is OK because we guard every use of 'asu8' with
                // a check that the input is <= 255.
                let asu8 = |b: u16| u8::try_from(b).unwrap();
                let mut b1: u16 = 0;
                let mut b2: u16;
                while b1 <= 255 {
                    b2 = b1 + 1;
                    while b2 <= 255 && iswb(asu8(b1)) == iswb(asu8(b2)) {
                        b2 += 1;
                    }
                    // The guards above guarantee that b2 can never get any
                    // bigger.
                    assert!(b2 <= 256);
                    // Subtracting 1 from b2 is always OK because it is always
                    // at least 1 greater than b1, and the assert above
                    // guarantees that the asu8 conversion will succeed.
                    set.set_range(asu8(b1), asu8(b2.checked_sub(1).unwrap()));
                    b1 = b2;
                }
            }
        }
    }
}

/// LookSet is a memory-efficient set of look-around assertions.
///
/// Callers may idempotently insert or remove any look-around assertion from a
/// set.
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub struct LookSet {
    bits: u8,
}

impl LookSet {
    pub const CAPACITY: usize = 8;

    pub const fn empty() -> LookSet {
        LookSet { bits: 0 }
    }

    pub const fn full() -> LookSet {
        LookSet { bits: !0 }
    }

    /// Return a LookSet from its representation.
    pub const fn from_repr(repr: u8) -> LookSet {
        LookSet { bits: repr }
    }

    /// Return the internal byte representation of this set.
    pub const fn to_repr(self) -> u8 {
        self.bits
    }

    /// Return true if and only if this set is empty.
    pub const fn is_empty(self) -> bool {
        self.bits == 0
    }

    /// Returns the number of elements in this set.
    pub const fn len(self) -> usize {
        // OK because max length is <= u8::MAX.
        //
        // FIXME: Use as_usize() once const functions in traits are stable.
        self.bits.count_ones() as usize
    }

    /// Insert the given look-around assertion into this set. If the assertion
    /// already exists, then this is a no-op.
    pub const fn insert(self, look: Look) -> LookSet {
        LookSet { bits: self.bits | look.as_repr() }
    }

    /// Remove the given look-around assertion from this set. If the assertion
    /// is not in this set, then this is a no-op.
    pub const fn remove(self, look: Look) -> LookSet {
        LookSet { bits: self.bits & !look.as_repr() }
    }

    /// Return true if and only if the given assertion is in this set.
    pub const fn contains(self, look: Look) -> bool {
        look.as_repr() & self.bits != 0
    }

    /// Subtract the given `other` set from the `self` set and return a new
    /// set.
    pub const fn subtract(self, other: LookSet) -> LookSet {
        LookSet { bits: self.bits & !other.bits }
    }

    /// Return the intersection of the given `other` set with the `self` set
    /// and return the resulting set.
    pub const fn intersect(self, other: LookSet) -> LookSet {
        LookSet { bits: self.bits & other.bits }
    }

    /// Returns an iterator over all of the look-around assertions in this set.
    pub const fn iter(self) -> LookSetIter {
        LookSetIter { set: self }
    }
}

impl core::fmt::Debug for LookSet {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        for look in self.iter() {
            write!(f, "{}", look.as_char())?;
        }
        Ok(())
    }
}

/// An iterator over all look-around assertions in a [`LookSet`].
///
/// This iterator is created by [`LookSet::iter`].
#[derive(Clone, Debug)]
pub struct LookSetIter {
    set: LookSet,
}

impl Iterator for LookSetIter {
    type Item = Look;

    fn next(&mut self) -> Option<Look> {
        // We'll never have more than u8::MAX distinct look-around assertions,
        // so 'index' will always fit into a usize.
        let index = self.set.bits.trailing_zeros().as_usize();
        let look = Look::from_index(index)?;
        self.set = self.set.remove(look);
        Some(look)
    }
}

#[inline]
pub fn is_start(haystack: &[u8], at: usize) -> bool {
    at == 0
}

#[inline]
pub fn is_end(haystack: &[u8], at: usize) -> bool {
    at == haystack.len()
}

#[inline]
pub fn is_start_lf(haystack: &[u8], at: usize) -> bool {
    at == 0 || haystack[at - 1] == b'\n'
}

#[inline]
pub fn is_end_lf(haystack: &[u8], at: usize) -> bool {
    at == haystack.len() || haystack[at] == b'\n'
}

#[inline]
pub fn is_word_ascii(haystack: &[u8], at: usize) -> bool {
    let word_before = at > 0 && utf8::is_word_byte(haystack[at - 1]);
    let word_after = at < haystack.len() && utf8::is_word_byte(haystack[at]);
    word_before != word_after
}

#[inline]
pub fn is_word_ascii_negate(haystack: &[u8], at: usize) -> bool {
    !is_word_ascii(haystack, at)
}

#[inline]
pub fn is_word_unicode(haystack: &[u8], at: usize) -> bool {
    let word_before = utf8::is_word_char_rev(haystack, at);
    let word_after = utf8::is_word_char_fwd(haystack, at);
    word_before != word_after
}

#[inline]
pub fn is_word_unicode_negate(haystack: &[u8], at: usize) -> bool {
    // This is pretty subtle. Why do we need to do UTF-8 decoding here? Well...
    // at time of writing, the is_word_char_{fwd,rev} routines will only return
    // true if there is a valid UTF-8 encoding of a "word" codepoint, and
    // false in every other case (including invalid UTF-8). This means that in
    // regions of invalid UTF-8 (which might be a subset of valid UTF-8!), it
    // would result in \B matching. While this would be questionable in the
    // context of truly invalid UTF-8, it is *certainly* wrong to report match
    // boundaries that split the encoding of a codepoint. So to work around
    // this, we ensure that we can decode a codepoint on either side of `at`.
    // If either direction fails, then we don't permit \B to match at all.
    //
    // Now, this isn't exactly optimal from a perf perspective. We could try
    // and detect this in is_word_char_{fwd,rev}, but it's not clear if it's
    // worth it. \B is, after all, rarely used.
    //
    // And in particular, we do *not* have to do this with \b, because \b
    // *requires* that at least one side of `at` be a "word" codepoint, which
    // in turn implies one side of `at` must be valid UTF-8. This in turn
    // implies that \b can never split a valid UTF-8 encoding of a codepoint.
    // In the case where one side of `at` is truly invalid UTF-8 and the other
    // side IS a word codepoint, then we want \b to match since it represents
    // a valid UTF-8 boundary. It also makes sense. For example, you'd want
    // \b\w+\b to match 'abc' in '\xFFabc\xFF'.
    //
    // Note also that this is not just '!is_word_unicode(..)' like it is for
    // the ASCII case. For example, neither \b nor \B is satisfied within
    // invalid UTF-8 sequences.
    let word_before = at > 0
        && match utf8::decode_last(&haystack[..at]) {
            None | Some(Err(_)) => return false,
            Some(Ok(_)) => utf8::is_word_char_rev(haystack, at),
        };
    let word_after = at < haystack.len()
        && match utf8::decode(&haystack[at..]) {
            None | Some(Err(_)) => return false,
            Some(Ok(_)) => utf8::is_word_char_fwd(haystack, at),
        };
    word_before == word_after
}

#[cfg(test)]
mod tests {
    use super::*;

    fn B<'a, T: 'a + ?Sized + AsRef<[u8]>>(string: &'a T) -> &'a [u8] {
        string.as_ref()
    }

    #[test]
    fn look_matches_start_line() {
        let look = Look::StartLF;

        assert!(look.matches(B(""), 0));
        assert!(look.matches(B("\n"), 0));
        assert!(look.matches(B("\n"), 1));
        assert!(look.matches(B("a"), 0));
        assert!(look.matches(B("\na"), 1));

        assert!(!look.matches(B("a"), 1));
        assert!(!look.matches(B("a\na"), 1));
    }

    #[test]
    fn look_matches_end_line() {
        let look = Look::EndLF;

        assert!(look.matches(B(""), 0));
        assert!(look.matches(B("\n"), 1));
        assert!(look.matches(B("\na"), 0));
        assert!(look.matches(B("\na"), 2));
        assert!(look.matches(B("a\na"), 1));

        assert!(!look.matches(B("a"), 0));
        assert!(!look.matches(B("\na"), 1));
        assert!(!look.matches(B("a\na"), 0));
        assert!(!look.matches(B("a\na"), 2));
    }

    #[test]
    fn look_matches_start_text() {
        let look = Look::Start;

        assert!(look.matches(B(""), 0));
        assert!(look.matches(B("\n"), 0));
        assert!(look.matches(B("a"), 0));

        assert!(!look.matches(B("\n"), 1));
        assert!(!look.matches(B("\na"), 1));
        assert!(!look.matches(B("a"), 1));
        assert!(!look.matches(B("a\na"), 1));
    }

    #[test]
    fn look_matches_end_text() {
        let look = Look::End;

        assert!(look.matches(B(""), 0));
        assert!(look.matches(B("\n"), 1));
        assert!(look.matches(B("\na"), 2));

        assert!(!look.matches(B("\na"), 0));
        assert!(!look.matches(B("a\na"), 1));
        assert!(!look.matches(B("a"), 0));
        assert!(!look.matches(B("\na"), 1));
        assert!(!look.matches(B("a\na"), 0));
        assert!(!look.matches(B("a\na"), 2));
    }

    #[test]
    #[cfg(not(miri))]
    fn look_matches_word_unicode() {
        let look = Look::WordUnicode;

        // \xF0\x9D\x9B\x83 = 𝛃 (in \w)
        // \xF0\x90\x86\x80 = 𐆀 (not in \w)

        // Simple ASCII word boundaries.
        assert!(look.matches(B("a"), 0));
        assert!(look.matches(B("a"), 1));
        assert!(look.matches(B("a "), 1));
        assert!(look.matches(B(" a "), 1));
        assert!(look.matches(B(" a "), 2));

        // Unicode word boundaries with a non-ASCII codepoint.
        assert!(look.matches(B("𝛃"), 0));
        assert!(look.matches(B("𝛃"), 4));
        assert!(look.matches(B("𝛃 "), 4));
        assert!(look.matches(B(" 𝛃 "), 1));
        assert!(look.matches(B(" 𝛃 "), 5));

        // Unicode word boundaries between non-ASCII codepoints.
        assert!(look.matches(B("𝛃𐆀"), 0));
        assert!(look.matches(B("𝛃𐆀"), 4));

        // Non word boundaries for ASCII.
        assert!(!look.matches(B(""), 0));
        assert!(!look.matches(B("ab"), 1));
        assert!(!look.matches(B("a "), 2));
        assert!(!look.matches(B(" a "), 0));
        assert!(!look.matches(B(" a "), 3));

        // Non word boundaries with a non-ASCII codepoint.
        assert!(!look.matches(B("𝛃b"), 4));
        assert!(!look.matches(B("𝛃 "), 5));
        assert!(!look.matches(B(" 𝛃 "), 0));
        assert!(!look.matches(B(" 𝛃 "), 6));
        assert!(!look.matches(B("𝛃"), 1));
        assert!(!look.matches(B("𝛃"), 2));
        assert!(!look.matches(B("𝛃"), 3));

        // Non word boundaries with non-ASCII codepoints.
        assert!(!look.matches(B("𝛃𐆀"), 1));
        assert!(!look.matches(B("𝛃𐆀"), 2));
        assert!(!look.matches(B("𝛃𐆀"), 3));
        assert!(!look.matches(B("𝛃𐆀"), 5));
        assert!(!look.matches(B("𝛃𐆀"), 6));
        assert!(!look.matches(B("𝛃𐆀"), 7));
        assert!(!look.matches(B("𝛃𐆀"), 8));
    }

    #[test]
    fn look_matches_word_ascii() {
        let look = Look::WordAscii;

        // \xF0\x9D\x9B\x83 = 𝛃 (in \w)
        // \xF0\x90\x86\x80 = 𐆀 (not in \w)

        // Simple ASCII word boundaries.
        assert!(look.matches(B("a"), 0));
        assert!(look.matches(B("a"), 1));
        assert!(look.matches(B("a "), 1));
        assert!(look.matches(B(" a "), 1));
        assert!(look.matches(B(" a "), 2));

        // Unicode word boundaries with a non-ASCII codepoint. Since this is
        // an ASCII word boundary, none of these match.
        assert!(!look.matches(B("𝛃"), 0));
        assert!(!look.matches(B("𝛃"), 4));
        assert!(!look.matches(B("𝛃 "), 4));
        assert!(!look.matches(B(" 𝛃 "), 1));
        assert!(!look.matches(B(" 𝛃 "), 5));

        // Unicode word boundaries between non-ASCII codepoints. Again, since
        // this is an ASCII word boundary, none of these match.
        assert!(!look.matches(B("𝛃𐆀"), 0));
        assert!(!look.matches(B("𝛃𐆀"), 4));

        // Non word boundaries for ASCII.
        assert!(!look.matches(B(""), 0));
        assert!(!look.matches(B("ab"), 1));
        assert!(!look.matches(B("a "), 2));
        assert!(!look.matches(B(" a "), 0));
        assert!(!look.matches(B(" a "), 3));

        // Non word boundaries with a non-ASCII codepoint.
        assert!(look.matches(B("𝛃b"), 4));
        assert!(!look.matches(B("𝛃 "), 5));
        assert!(!look.matches(B(" 𝛃 "), 0));
        assert!(!look.matches(B(" 𝛃 "), 6));
        assert!(!look.matches(B("𝛃"), 1));
        assert!(!look.matches(B("𝛃"), 2));
        assert!(!look.matches(B("𝛃"), 3));

        // Non word boundaries with non-ASCII codepoints.
        assert!(!look.matches(B("𝛃𐆀"), 1));
        assert!(!look.matches(B("𝛃𐆀"), 2));
        assert!(!look.matches(B("𝛃𐆀"), 3));
        assert!(!look.matches(B("𝛃𐆀"), 5));
        assert!(!look.matches(B("𝛃𐆀"), 6));
        assert!(!look.matches(B("𝛃𐆀"), 7));
        assert!(!look.matches(B("𝛃𐆀"), 8));
    }

    #[test]
    #[cfg(not(miri))]
    fn look_matches_word_unicode_negate() {
        let look = Look::WordUnicodeNegate;

        // \xF0\x9D\x9B\x83 = 𝛃 (in \w)
        // \xF0\x90\x86\x80 = 𐆀 (not in \w)

        // Simple ASCII word boundaries.
        assert!(!look.matches(B("a"), 0));
        assert!(!look.matches(B("a"), 1));
        assert!(!look.matches(B("a "), 1));
        assert!(!look.matches(B(" a "), 1));
        assert!(!look.matches(B(" a "), 2));

        // Unicode word boundaries with a non-ASCII codepoint.
        assert!(!look.matches(B("𝛃"), 0));
        assert!(!look.matches(B("𝛃"), 4));
        assert!(!look.matches(B("𝛃 "), 4));
        assert!(!look.matches(B(" 𝛃 "), 1));
        assert!(!look.matches(B(" 𝛃 "), 5));

        // Unicode word boundaries between non-ASCII codepoints.
        assert!(!look.matches(B("𝛃𐆀"), 0));
        assert!(!look.matches(B("𝛃𐆀"), 4));

        // Non word boundaries for ASCII.
        assert!(look.matches(B(""), 0));
        assert!(look.matches(B("ab"), 1));
        assert!(look.matches(B("a "), 2));
        assert!(look.matches(B(" a "), 0));
        assert!(look.matches(B(" a "), 3));

        // Non word boundaries with a non-ASCII codepoint.
        assert!(look.matches(B("𝛃b"), 4));
        assert!(look.matches(B("𝛃 "), 5));
        assert!(look.matches(B(" 𝛃 "), 0));
        assert!(look.matches(B(" 𝛃 "), 6));
        // These don't match because they could otherwise return an offset that
        // splits the UTF-8 encoding of a codepoint.
        assert!(!look.matches(B("𝛃"), 1));
        assert!(!look.matches(B("𝛃"), 2));
        assert!(!look.matches(B("𝛃"), 3));

        // Non word boundaries with non-ASCII codepoints. These also don't
        // match because they could otherwise return an offset that splits the
        // UTF-8 encoding of a codepoint.
        assert!(!look.matches(B("𝛃𐆀"), 1));
        assert!(!look.matches(B("𝛃𐆀"), 2));
        assert!(!look.matches(B("𝛃𐆀"), 3));
        assert!(!look.matches(B("𝛃𐆀"), 5));
        assert!(!look.matches(B("𝛃𐆀"), 6));
        assert!(!look.matches(B("𝛃𐆀"), 7));
        // But this one does, since 𐆀 isn't a word codepoint, and 8 is the end
        // of the haystack. So the "end" of the haystack isn't a word and 𐆀
        // isn't a word, thus, \B matches.
        assert!(look.matches(B("𝛃𐆀"), 8));
    }

    #[test]
    fn look_matches_word_ascii_negate() {
        let look = Look::WordAsciiNegate;

        // \xF0\x9D\x9B\x83 = 𝛃 (in \w)
        // \xF0\x90\x86\x80 = 𐆀 (not in \w)

        // Simple ASCII word boundaries.
        assert!(!look.matches(B("a"), 0));
        assert!(!look.matches(B("a"), 1));
        assert!(!look.matches(B("a "), 1));
        assert!(!look.matches(B(" a "), 1));
        assert!(!look.matches(B(" a "), 2));

        // Unicode word boundaries with a non-ASCII codepoint. Since this is
        // an ASCII word boundary, none of these match.
        assert!(look.matches(B("𝛃"), 0));
        assert!(look.matches(B("𝛃"), 4));
        assert!(look.matches(B("𝛃 "), 4));
        assert!(look.matches(B(" 𝛃 "), 1));
        assert!(look.matches(B(" 𝛃 "), 5));

        // Unicode word boundaries between non-ASCII codepoints. Again, since
        // this is an ASCII word boundary, none of these match.
        assert!(look.matches(B("𝛃𐆀"), 0));
        assert!(look.matches(B("𝛃𐆀"), 4));

        // Non word boundaries for ASCII.
        assert!(look.matches(B(""), 0));
        assert!(look.matches(B("ab"), 1));
        assert!(look.matches(B("a "), 2));
        assert!(look.matches(B(" a "), 0));
        assert!(look.matches(B(" a "), 3));

        // Non word boundaries with a non-ASCII codepoint.
        assert!(!look.matches(B("𝛃b"), 4));
        assert!(look.matches(B("𝛃 "), 5));
        assert!(look.matches(B(" 𝛃 "), 0));
        assert!(look.matches(B(" 𝛃 "), 6));
        assert!(look.matches(B("𝛃"), 1));
        assert!(look.matches(B("𝛃"), 2));
        assert!(look.matches(B("𝛃"), 3));

        // Non word boundaries with non-ASCII codepoints.
        assert!(look.matches(B("𝛃𐆀"), 1));
        assert!(look.matches(B("𝛃𐆀"), 2));
        assert!(look.matches(B("𝛃𐆀"), 3));
        assert!(look.matches(B("𝛃𐆀"), 5));
        assert!(look.matches(B("𝛃𐆀"), 6));
        assert!(look.matches(B("𝛃𐆀"), 7));
        assert!(look.matches(B("𝛃𐆀"), 8));
    }

    #[test]
    fn look_set() {
        let mut f = LookSet::default();
        assert!(!f.contains(Look::Start));
        assert!(!f.contains(Look::End));
        assert!(!f.contains(Look::StartLF));
        assert!(!f.contains(Look::EndLF));
        assert!(!f.contains(Look::WordUnicode));
        assert!(!f.contains(Look::WordUnicodeNegate));
        assert!(!f.contains(Look::WordAscii));
        assert!(!f.contains(Look::WordAsciiNegate));

        f = f.insert(Look::Start);
        assert!(f.contains(Look::Start));
        f = f.remove(Look::Start);
        assert!(!f.contains(Look::Start));

        f = f.insert(Look::End);
        assert!(f.contains(Look::End));
        f = f.remove(Look::End);
        assert!(!f.contains(Look::End));

        f = f.insert(Look::StartLF);
        assert!(f.contains(Look::StartLF));
        f = f.remove(Look::StartLF);
        assert!(!f.contains(Look::StartLF));

        f = f.insert(Look::EndLF);
        assert!(f.contains(Look::EndLF));
        f = f.remove(Look::EndLF);
        assert!(!f.contains(Look::EndLF));

        f = f.insert(Look::WordUnicode);
        assert!(f.contains(Look::WordUnicode));
        f = f.remove(Look::WordUnicode);
        assert!(!f.contains(Look::WordUnicode));

        f = f.insert(Look::WordUnicodeNegate);
        assert!(f.contains(Look::WordUnicodeNegate));
        f = f.remove(Look::WordUnicodeNegate);
        assert!(!f.contains(Look::WordUnicodeNegate));

        f = f.insert(Look::WordAscii);
        assert!(f.contains(Look::WordAscii));
        f = f.remove(Look::WordAscii);
        assert!(!f.contains(Look::WordAscii));

        f = f.insert(Look::WordAsciiNegate);
        assert!(f.contains(Look::WordAsciiNegate));
        f = f.remove(Look::WordAsciiNegate);
        assert!(!f.contains(Look::WordAsciiNegate));
    }

    #[test]
    fn look_set_iter() {
        let set = LookSet::empty();
        assert_eq!(0, set.iter().count());

        let set = LookSet::full();
        assert_eq!(8, set.iter().count());

        let set =
            LookSet::empty().insert(Look::StartLF).insert(Look::WordUnicode);
        assert_eq!(2, set.iter().count());

        let set = LookSet::empty().insert(Look::StartLF);
        assert_eq!(1, set.iter().count());

        let set = LookSet::empty().insert(Look::WordAsciiNegate);
        assert_eq!(1, set.iter().count());
    }
}
