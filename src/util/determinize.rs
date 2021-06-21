/// Start represents the four possible starting configurations of a DFA based
/// on the text being searched. Ultimately, this along with a pattern ID (if
/// specified) is what selects the start state to use in a DFA.
///
/// In a DFA that doesn't have starting states for each pattern, then it will
/// have a maximum of four starting states. If the DFA was compiled with start
/// states for each pattern, then it will have a maximum of four starting
/// states for searching for any pattern, and then another maximum of four
/// starting states for executing an anchored search for each pattern.
///
/// This ends up being represented as a table in the DFA where the stride of
/// that table is 4, and each entry is an index into the state transition
/// table. Note though that multiple entries in the table might point to the
/// same state if the states would otherwise be equivalent. (This is guaranteed
/// by minimization and may even be accomplished by normal determinization,
/// since it attempts to reuse equivalent states too.)
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Start {
    /// This occurs when the starting position is not any of the ones below.
    NonWordByte = 0,
    /// This occurs when the byte immediately preceding the start of the search
    /// is an ASCII word byte.
    WordByte = 1,
    /// This occurs when the starting position of the search corresponds to the
    /// beginning of the haystack.
    Text = 2,
    /// This occurs when the byte immediately preceding the start of the search
    /// is a line terminator. Specifically, `\n`.
    Line = 3,
}

impl Start {
    /// Return the starting state corresponding to the given integer. If no
    /// starting state exists for the given integer, then None is returned.
    pub(crate) fn from_usize(n: usize) -> Option<Start> {
        match n {
            0 => Some(Start::NonWordByte),
            1 => Some(Start::WordByte),
            2 => Some(Start::Text),
            3 => Some(Start::Line),
            _ => None,
        }
    }

    /// Returns the total number of starting state configurations.
    pub(crate) fn count() -> usize {
        4
    }

    /// Returns the starting state configuration for the given search
    /// parameters. If the given offset range is not valid, then this panics.
    #[inline(always)]
    pub(crate) fn from_position_fwd(
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Start {
        assert!(
            bytes.get(start..end).is_some(),
            "{}..{} is invalid",
            start,
            end
        );
        if start == 0 {
            Start::Text
        } else if bytes[start - 1] == b'\n' {
            Start::Line
        } else if crate::util::is_word_byte(bytes[start - 1]) {
            Start::WordByte
        } else {
            Start::NonWordByte
        }
    }

    /// Returns the starting state configuration for a reverse search with the
    /// given search parameters. If the given offset range is not valid, then
    /// this panics.
    #[inline(always)]
    pub(crate) fn from_position_rev(
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Start {
        assert!(
            bytes.get(start..end).is_some(),
            "{}..{} is invalid",
            start,
            end
        );
        if end == bytes.len() {
            Start::Text
        } else if bytes[end] == b'\n' {
            Start::Line
        } else if crate::util::is_word_byte(bytes[end]) {
            Start::WordByte
        } else {
            Start::NonWordByte
        }
    }

    /// Return this starting configuration as an integer. It is guaranteed to
    /// be less than `Start::count()`.
    #[inline(always)]
    pub(crate) fn as_usize(&self) -> usize {
        *self as usize
    }
}

#[cfg(test)]
mod tests {
    use super::Start;

    #[test]
    #[should_panic]
    fn start_fwd_bad_range() {
        Start::from_position_fwd(&[], 0, 1);
    }

    #[test]
    #[should_panic]
    fn start_rev_bad_range() {
        Start::from_position_rev(&[], 0, 1);
    }

    #[test]
    fn start_fwd() {
        let f = Start::from_position_fwd;

        assert_eq!(Start::Text, f(&[], 0, 0));
        assert_eq!(Start::Text, f(b"abc", 0, 3));
        assert_eq!(Start::Text, f(b"\nabc", 0, 3));

        assert_eq!(Start::Line, f(b"\nabc", 1, 3));

        assert_eq!(Start::WordByte, f(b"abc", 1, 3));

        assert_eq!(Start::NonWordByte, f(b" abc", 1, 3));
    }

    #[test]
    fn start_rev() {
        let f = Start::from_position_rev;

        assert_eq!(Start::Text, f(&[], 0, 0));
        assert_eq!(Start::Text, f(b"abc", 0, 3));
        assert_eq!(Start::Text, f(b"abc\n", 0, 4));

        assert_eq!(Start::Line, f(b"abc\nz", 0, 3));

        assert_eq!(Start::WordByte, f(b"abc", 0, 2));

        assert_eq!(Start::NonWordByte, f(b"abc ", 0, 3));
    }
}
