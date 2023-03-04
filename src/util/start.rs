use crate::util::{
    look::LookMatcher,
    search::Input,
    wire::{self, DeserializeError, SerializeError},
};

#[derive(Clone)]
pub(crate) struct StartByteMap {
    map: [Start; 256],
}

impl StartByteMap {
    pub(crate) fn new(lookm: &LookMatcher) -> StartByteMap {
        let mut map = [Start::NonWordByte; 256];
        map[usize::from(b'\n')] = Start::LineLF;
        map[usize::from(b'\r')] = Start::LineCR;
        map[usize::from(b'_')] = Start::WordByte;

        let mut byte = b'0';
        while byte <= b'9' {
            map[usize::from(byte)] = Start::WordByte;
            byte += 1;
        }
        byte = b'A';
        while byte <= b'Z' {
            map[usize::from(byte)] = Start::WordByte;
            byte += 1;
        }
        byte = b'a';
        while byte <= b'z' {
            map[usize::from(byte)] = Start::WordByte;
            byte += 1;
        }

        let lineterm = lookm.get_line_terminator();
        if lineterm != b'\r' && lineterm != b'\n' {
            map[usize::from(lineterm)] = Start::CustomLineTerminator;
        }
        StartByteMap { map }
    }

    pub(crate) fn fwd(&self, input: &Input) -> Start {
        match input
            .start()
            .checked_sub(1)
            .and_then(|i| input.haystack().get(i))
        {
            None => Start::Text,
            Some(&byte) => self.get(byte),
        }
    }

    pub(crate) fn rev(&self, input: &Input) -> Start {
        match input.haystack().get(input.end()) {
            None => Start::Text,
            Some(&byte) => self.get(byte),
        }
    }

    fn get(&self, byte: u8) -> Start {
        self.map[usize::from(byte)]
    }

    /// Deserializes a byte class map from the given slice. If the slice is of
    /// insufficient length or otherwise contains an impossible mapping, then
    /// an error is returned. Upon success, the number of bytes read along with
    /// the map are returned. The number of bytes read is always a multiple of
    /// 8.
    pub(crate) fn from_bytes(
        slice: &[u8],
    ) -> Result<(StartByteMap, usize), DeserializeError> {
        wire::check_slice_len(slice, 256, "start byte map")?;
        let mut map = [Start::NonWordByte; 256];
        for (i, &repr) in slice[..256].iter().enumerate() {
            map[i] = match Start::from_usize(usize::from(repr)) {
                Some(start) => start,
                None => {
                    return Err(DeserializeError::generic(
                        "found invalid starting configuration",
                    ))
                }
            };
        }
        Ok((StartByteMap { map }, 256))
    }

    /// Writes this byte class map to the given byte buffer. if the given
    /// buffer is too small, then an error is returned. Upon success, the total
    /// number of bytes written is returned. The number of bytes written is
    /// guaranteed to be a multiple of 8.
    pub fn write_to(&self, dst: &mut [u8]) -> Result<usize, SerializeError> {
        let nwrite = self.write_to_len();
        if dst.len() < nwrite {
            return Err(SerializeError::buffer_too_small("start byte map"));
        }
        for (i, &start) in self.map.iter().enumerate() {
            dst[i] = start.as_u8();
        }
        Ok(nwrite)
    }

    /// Returns the total number of bytes written by `write_to`.
    pub fn write_to_len(&self) -> usize {
        256
    }
}

impl core::fmt::Debug for StartByteMap {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use crate::util::escape::DebugByte;

        write!(f, "StartByteMap{{")?;
        for byte in 0..=255 {
            if byte > 0 {
                write!(f, ", ")?;
            }
            let start = self.map[usize::from(byte)];
            write!(f, "{:?} => {:?}", DebugByte(byte), start)?;
        }
        write!(f, "}}")?;
        Ok(())
    }
}

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
    LineLF = 3,
    /// This occurs when the byte immediately preceding the start of the search
    /// is a line terminator. Specifically, `\r`.
    LineCR = 4,
    /// This occurs when a custom line terminator has been set via a
    /// `LookMatcher`, and when that line terminator is neither a `\r` or a
    /// `\n`.
    ///
    /// If the custom line terminator is a word byte, then this start
    /// configuration is still selected. DFAs that implement word boundary
    /// assertions will likely need to check whether the customer line
    /// terminator is a word byte, in which case, it should behave as if the
    /// byte satisfies `\b` in addition to multi-line anchors.
    CustomLineTerminator = 5,
}

impl Start {
    /// Return the starting state corresponding to the given integer. If no
    /// starting state exists for the given integer, then None is returned.
    pub(crate) fn from_usize(n: usize) -> Option<Start> {
        match n {
            0 => Some(Start::NonWordByte),
            1 => Some(Start::WordByte),
            2 => Some(Start::Text),
            3 => Some(Start::LineLF),
            4 => Some(Start::LineCR),
            5 => Some(Start::CustomLineTerminator),
            _ => None,
        }
    }

    /// Returns the total number of starting state configurations.
    pub(crate) fn len() -> usize {
        6
    }

    /// Return this starting configuration as `u8` integer. It is guaranteed to
    /// be less than `Start::len()`.
    #[inline(always)]
    pub(crate) fn as_u8(&self) -> u8 {
        // AFAIK, 'as' is the only way to zero-cost convert an int enum to an
        // actual int.
        *self as u8
    }

    /// Return this starting configuration as a `usize` integer. It is
    /// guaranteed to be less than `Start::len()`.
    #[inline(always)]
    pub(crate) fn as_usize(&self) -> usize {
        usize::from(self.as_u8())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_fwd_done_range() {
        let smap = StartByteMap::new(&LookMatcher::default());
        assert_eq!(Start::Text, smap.fwd(&Input::new("").range(1..0)));
    }

    #[test]
    fn start_rev_done_range() {
        let smap = StartByteMap::new(&LookMatcher::default());
        assert_eq!(Start::Text, smap.rev(&Input::new("").range(1..0)));
    }

    #[test]
    fn start_fwd() {
        let f = |haystack, start, end| {
            let smap = StartByteMap::new(&LookMatcher::default());
            let input = &Input::new(haystack).range(start..end);
            smap.fwd(input)
        };

        assert_eq!(Start::Text, f("", 0, 0));
        assert_eq!(Start::Text, f("abc", 0, 3));
        assert_eq!(Start::Text, f("\nabc", 0, 3));

        assert_eq!(Start::LineLF, f("\nabc", 1, 3));

        assert_eq!(Start::LineCR, f("\rabc", 1, 3));

        assert_eq!(Start::WordByte, f("abc", 1, 3));

        assert_eq!(Start::NonWordByte, f(" abc", 1, 3));
    }

    #[test]
    fn start_rev() {
        let f = |haystack, start, end| {
            let smap = StartByteMap::new(&LookMatcher::default());
            let input = &Input::new(haystack).range(start..end);
            smap.rev(input)
        };

        assert_eq!(Start::Text, f("", 0, 0));
        assert_eq!(Start::Text, f("abc", 0, 3));
        assert_eq!(Start::Text, f("abc\n", 0, 4));

        assert_eq!(Start::LineLF, f("abc\nz", 0, 3));

        assert_eq!(Start::LineCR, f("abc\rz", 0, 3));

        assert_eq!(Start::WordByte, f("abc", 0, 2));

        assert_eq!(Start::NonWordByte, f("abc ", 0, 3));
    }
}
