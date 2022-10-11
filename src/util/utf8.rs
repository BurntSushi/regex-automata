#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Returns true if and only if the given byte is considered a word character.
/// This only applies to ASCII.
///
/// This was copied from regex-syntax so that we can use it to determine the
/// starting DFA state while searching without depending on regex-syntax. The
/// definition is never going to change, so there's no maintenance/bit-rot
/// hazard here.
#[inline(always)]
pub(crate) fn is_word_byte(b: u8) -> bool {
    const fn mkwordset() -> [bool; 256] {
        // FIXME: Use as_usize() once const functions in traits are stable.
        let mut set = [false; 256];
        set[b'_' as usize] = true;

        let mut byte = b'0';
        while byte <= b'9' {
            set[byte as usize] = true;
            byte += 1;
        }
        byte = b'A';
        while byte <= b'Z' {
            set[byte as usize] = true;
            byte += 1;
        }
        byte = b'a';
        while byte <= b'z' {
            set[byte as usize] = true;
            byte += 1;
        }
        set
    }
    const WORD: [bool; 256] = mkwordset();
    WORD[b as usize]
}

/// Decodes the next UTF-8 encoded codepoint from the given byte slice.
///
/// If no valid encoding of a codepoint exists at the beginning of the given
/// byte slice, then the first byte is returned instead.
///
/// This returns `None` if and only if `bytes` is empty.
#[inline(always)]
pub(crate) fn decode(bytes: &[u8]) -> Option<Result<char, u8>> {
    if bytes.is_empty() {
        return None;
    }
    let len = match len(bytes[0]) {
        None => return Some(Err(bytes[0])),
        Some(len) if len > bytes.len() => return Some(Err(bytes[0])),
        Some(1) => return Some(Ok(char::from(bytes[0]))),
        Some(len) => len,
    };
    match core::str::from_utf8(&bytes[..len]) {
        Ok(s) => Some(Ok(s.chars().next().unwrap())),
        Err(_) => Some(Err(bytes[0])),
    }
}

/// Decodes the last UTF-8 encoded codepoint from the given byte slice.
///
/// If no valid encoding of a codepoint exists at the end of the given byte
/// slice, then the last byte is returned instead.
///
/// This returns `None` if and only if `bytes` is empty.
#[inline(always)]
pub(crate) fn decode_last(bytes: &[u8]) -> Option<Result<char, u8>> {
    if bytes.is_empty() {
        return None;
    }
    let mut start = bytes.len() - 1;
    let limit = bytes.len().saturating_sub(4);
    while start > limit && !is_leading_or_invalid_byte(bytes[start]) {
        start -= 1;
    }
    match decode(&bytes[start..]) {
        None => None,
        Some(Ok(ch)) => Some(Ok(ch)),
        Some(Err(_)) => Some(Err(bytes[bytes.len() - 1])),
    }
}

/// Given a UTF-8 leading byte, this returns the total number of code units
/// in the following encoded codepoint.
///
/// If the given byte is not a valid UTF-8 leading byte, then this returns
/// `None`.
#[inline(always)]
fn len(byte: u8) -> Option<usize> {
    if byte <= 0x7F {
        return Some(1);
    } else if byte & 0b1100_0000 == 0b1000_0000 {
        return None;
    } else if byte <= 0b1101_1111 {
        Some(2)
    } else if byte <= 0b1110_1111 {
        Some(3)
    } else if byte <= 0b1111_0111 {
        Some(4)
    } else {
        None
    }
}

/// Returns true if and only if the given offset in the given bytes falls on a
/// valid UTF-8 encoded codepoint boundary.
///
/// If `bytes` is not valid UTF-8, then the behavior of this routine is
/// unspecified.
#[inline(always)]
pub(crate) fn is_boundary(bytes: &[u8], i: usize) -> bool {
    match bytes.get(i) {
        // The position at the end of the bytes always represents an empty
        // string, which is a valid boundary. But anything after that doesn't
        // make much sense to call valid a boundary.
        None => i == bytes.len(),
        // Other than ASCII (where the most significant bit is never set),
        // valid starting bytes always have their most significant two bits
        // set, where as continuation bytes never have their second most
        // significant bit set. Therefore, this only returns true when bytes[i]
        // corresponds to a byte that begins a valid UTF-8 encoding of a
        // Unicode scalar value.
        Some(&b) => b <= 0b0111_1111 || b >= 0b1100_0000,
    }
}

/// Returns true if and only if the given byte is either a valid leading UTF-8
/// byte, or is otherwise an invalid byte that can never appear anywhere in a
/// valid UTF-8 sequence.
#[inline(always)]
fn is_leading_or_invalid_byte(b: u8) -> bool {
    // In the ASCII case, the most significant bit is never set. The leading
    // byte of a 2/3/4-byte sequence always has the top two most significant
    // bits set. For bytes that can never appear anywhere in valid UTF-8, this
    // also returns true, since every such byte has its two most significant
    // bits set:
    //
    //     \xC0 :: 11000000
    //     \xC1 :: 11000001
    //     \xF5 :: 11110101
    //     \xF6 :: 11110110
    //     \xF7 :: 11110111
    //     \xF8 :: 11111000
    //     \xF9 :: 11111001
    //     \xFA :: 11111010
    //     \xFB :: 11111011
    //     \xFC :: 11111100
    //     \xFD :: 11111101
    //     \xFE :: 11111110
    //     \xFF :: 11111111
    (b & 0b1100_0000) != 0b1000_0000
}

#[cfg(feature = "alloc")]
#[inline(always)]
pub(crate) fn is_word_char_fwd(bytes: &[u8], mut at: usize) -> bool {
    use core::{ptr, sync::atomic::AtomicPtr};

    use crate::{
        dfa::{dense::DFA, Automaton, StartKind},
        util::{lazy, primitives::StateID},
        Anchored, Input,
    };

    static WORD: AtomicPtr<(DFA<Vec<u32>>, StateID)> =
        AtomicPtr::new(ptr::null_mut());

    let (dfa, start_id) = lazy::get_or_init(&WORD, || {
        // TODO: Should we use a lazy DFA here instead? It does complicate
        // things somewhat, since we then need a mutable cache, which probably
        // means a thread local.
        let dfa = DFA::builder()
            .configure(DFA::config().start_kind(StartKind::Anchored))
            .build(r"\w")
            .unwrap();
        // This is OK since '\w' contains no look-around.
        let input = Input::new("").anchored(Anchored::Yes);
        let start_id = dfa.start_state_forward(&input).expect("correct input");
        (dfa, start_id)
    });
    let mut sid = *start_id;
    while at < bytes.len() {
        let byte = bytes[at];
        sid = dfa.next_state(sid, byte);
        at += 1;
        if dfa.is_special_state(sid) {
            if dfa.is_match_state(sid) {
                return true;
            } else if dfa.is_dead_state(sid) {
                return false;
            }
        }
    }
    dfa.is_match_state(dfa.next_eoi_state(sid))
}

#[cfg(feature = "alloc")]
#[inline(always)]
pub(crate) fn is_word_char_rev(bytes: &[u8], mut at: usize) -> bool {
    use core::{ptr, sync::atomic::AtomicPtr};

    use crate::{
        dfa::{dense::DFA, Automaton, StartKind},
        nfa::thompson::NFA,
        util::{lazy, primitives::StateID},
        Anchored, Input,
    };

    static WORD: AtomicPtr<(DFA<Vec<u32>>, StateID)> =
        AtomicPtr::new(ptr::null_mut());

    let (dfa, start_id) = lazy::get_or_init(&WORD, || {
        let dfa = DFA::builder()
            .configure(DFA::config().start_kind(StartKind::Anchored))
            .thompson(NFA::config().reverse(true).shrink(true))
            .build(r"\w")
            .unwrap();
        // This is OK since '\w' contains no look-around.
        let input = Input::new("").anchored(Anchored::Yes);
        let start_id = dfa.start_state_reverse(&input).expect("correct input");
        (dfa, start_id)
    });
    let mut sid = *start_id;
    while at > 0 {
        at -= 1;
        let byte = bytes[at];
        sid = dfa.next_state(sid, byte);
        if dfa.is_special_state(sid) {
            if dfa.is_match_state(sid) {
                return true;
            } else if dfa.is_dead_state(sid) {
                return false;
            }
        }
    }
    dfa.is_match_state(dfa.next_eoi_state(sid))
}

/*
/// Returns the smallest possible index of the next valid UTF-8 sequence
/// starting after `i`.
///
/// For all inputs, including invalid UTF-8 and any value of `i`, the return
/// value is guaranteed to be greater than `i`. (If there is no value greater
/// than `i` that fits in `usize`, then this panics.)
///
/// Generally speaking, this should only be called on `text` when it is
/// permitted to assume that it is valid UTF-8 and where either `i >=
/// text.len()` or where `text[i]` is a leading byte of a UTF-8 sequence.
///
/// NOTE: This method was used in a previous conception of iterators where we
/// specifically tried to skip over empty matches that split a codepoint by
/// simply requiring that our next search begin at the beginning of codepoint.
/// But we ended up changing that technique to always advance by 1 byte and
/// then filter out matches that split a codepoint after-the-fact. Thus, we no
/// longer use this method. But I've kept it around in case we want to switch
/// back to this approach. Its guarantees are a little subtle, so I'd prefer
/// not to rebuild it from whole cloth.
pub(crate) fn next(text: &[u8], i: usize) -> usize {
    let b = match text.get(i) {
        None => return i.checked_add(1).unwrap(),
        Some(&b) => b,
    };
    // For cases where we see an invalid UTF-8 byte, there isn't much we can do
    // other than just start at the next byte.
    let inc = len(b).unwrap_or(1);
    i.checked_add(inc).unwrap()
}
*/
