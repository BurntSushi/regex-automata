use crate::util::search::Input;

/// Represents the four possible starting configurations of a DFA search.
///
/// The starting configuration is determined by inspecting the the beginning of
/// the haystack (up to 1 byte). Ultimately, this along with a pattern ID (if
/// specified) is what selects the start state to use in a DFA.
///
/// In a DFA that doesn't have starting states for each pattern, then it will
/// have a maximum of four DFA start states. If the DFA was compiled with start
/// states for each pattern, then it will have a maximum of four DFA start
/// states for searching for any pattern, and then another maximum of four DFA
/// start states for executing an anchored search for each pattern.
///
/// This ends up being represented as a table in the DFA (whether lazy or fully
/// built) where the stride of that table is 4, and each entry is an index into
/// the state transition table. Note though that multiple entries in the table
/// might point to the same state if the states would otherwise be equivalent.
/// (This is guaranteed by DFA minimization and may even be accomplished by
/// normal determinization, since it attempts to reuse equivalent states too.)
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
    pub(crate) fn len() -> usize {
        4
    }

    /// Returns the starting state configuration for the given search
    /// parameters.
    #[inline(always)]
    pub(crate) fn from_position_fwd(input: &Input<'_, '_>) -> Start {
        match input
            .start()
            .checked_sub(1)
            .and_then(|i| input.haystack().get(i))
        {
            None => Start::Text,
            Some(&byte) => byte_to_start(byte),
        }
    }

    /// Returns the starting state configuration for a reverse search with the
    /// given search parameters. If the given offset range is not valid, then
    /// this panics.
    #[inline(always)]
    pub(crate) fn from_position_rev(input: &Input<'_, '_>) -> Start {
        match input.haystack().get(input.end()) {
            None => Start::Text,
            Some(&byte) => byte_to_start(byte),
        }
    }

    /// Return this starting configuration as an integer. It is guaranteed to
    /// be less than `Start::len()`.
    #[inline(always)]
    pub(crate) fn as_usize(&self) -> usize {
        // AFAIK, 'as' is the only way to zero-cost convert an int enum to an
        // actual int.
        *self as usize
    }
}

#[inline(always)]
fn byte_to_start(byte: u8) -> Start {
    const fn make_mapping() -> [Start; 256] {
        // FIXME: Use as_usize() once const functions in traits are stable.

        let mut map = [Start::NonWordByte; 256];
        map[b'\n' as usize] = Start::Line;
        map[b'_' as usize] = Start::WordByte;

        let mut byte = b'0';
        while byte <= b'9' {
            map[byte as usize] = Start::WordByte;
            byte += 1;
        }
        byte = b'A';
        while byte <= b'Z' {
            map[byte as usize] = Start::WordByte;
            byte += 1;
        }
        byte = b'a';
        while byte <= b'z' {
            map[byte as usize] = Start::WordByte;
            byte += 1;
        }
        map
    }
    const MAPPING: [Start; 256] = make_mapping();
    MAPPING[byte as usize]
}

#[cfg(test)]
mod tests {
    use super::Start;
    use crate::Input;

    #[test]
    fn start_fwd_bad_range() {
        assert_eq!(
            Start::Text,
            Start::from_position_fwd(&Input::new("").range(0..1))
        );
    }

    #[test]
    fn start_rev_bad_range() {
        assert_eq!(
            Start::Text,
            Start::from_position_rev(&Input::new("").range(0..1))
        );
    }

    #[test]
    fn start_fwd() {
        let f = |haystack, start, end| {
            let input = &Input::new(haystack).range(start..end);
            Start::from_position_fwd(input)
        };

        assert_eq!(Start::Text, f("", 0, 0));
        assert_eq!(Start::Text, f("abc", 0, 3));
        assert_eq!(Start::Text, f("\nabc", 0, 3));

        assert_eq!(Start::Line, f("\nabc", 1, 3));

        assert_eq!(Start::WordByte, f("abc", 1, 3));

        assert_eq!(Start::NonWordByte, f(" abc", 1, 3));
    }

    #[test]
    fn start_rev() {
        let f = |haystack, start, end| {
            let input = &Input::new(haystack).range(start..end);
            Start::from_position_rev(input)
        };

        assert_eq!(Start::Text, f("", 0, 0));
        assert_eq!(Start::Text, f("abc", 0, 3));
        assert_eq!(Start::Text, f("abc\n", 0, 4));

        assert_eq!(Start::Line, f("abc\nz", 0, 3));

        assert_eq!(Start::WordByte, f("abc", 0, 2));

        assert_eq!(Start::NonWordByte, f("abc ", 0, 3));
    }
}
