use core::{ascii, fmt, str};

/// A type that wraps a single byte with a convenient fmt::Debug impl that
/// escapes the byte.
pub struct DebugByte(pub u8);

impl fmt::Debug for DebugByte {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // 10 bytes is enough to cover any output from ascii::escape_default.
        let mut bytes = [0u8; 10];
        let mut len = 0;
        for (i, mut b) in ascii::escape_default(self.0).enumerate() {
            // capitalize \xab to \xAB
            if i >= 2 && b'a' <= b && b <= b'f' {
                b -= 32;
            }
            bytes[len] = b;
            len += 1;
        }
        write!(f, "{}", str::from_utf8(&bytes[..len]).unwrap())
    }
}

/// Returns the smallest possible index of the next valid UTF-8 sequence
/// starting after `i`.
///
/// For all inputs, including invalid UTF-8 and any value of `i`, the return
/// value is guaranteed to be greater than `i`.
///
/// Generally speaking, this should only be called on `text` when it is
/// permitted to assume that it is valid UTF-8 and where either `i >=
/// text.len()` or where `text[i]` is a leading byte of a UTF-8 sequence.
pub fn next_utf8(text: &[u8], i: usize) -> usize {
    let b = match text.get(i) {
        None => return i.checked_add(1).unwrap(),
        Some(&b) => b,
    };
    let inc = if b <= 0x7F {
        1
    } else if b <= 0b110_11111 {
        2
    } else if b <= 0b1110_1111 {
        3
    } else if b <= 0b1111_0111 {
        4
    } else {
        // For cases where we see an invalid UTF-8 byte, there isn't much we
        // can do other than just start at the next byte.
        1
    };
    i.checked_add(inc).unwrap()
}

/// Returns true if and only if the given byte is considered a word character.
/// This only applies to ASCII.
///
/// This was copied from regex-syntax so that we can use it to determine the
/// starting DFA state while searching without depending on regex-syntax. The
/// definition is never going to change, so there's no maintenance/bit-rot
/// hazard here.
#[inline(always)]
pub fn is_word_byte(b: u8) -> bool {
    match b {
        b'_' | b'0'..=b'9' | b'a'..=b'z' | b'A'..=b'Z' => true,
        _ => false,
    }
}
