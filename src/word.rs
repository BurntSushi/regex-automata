/// Returns true if and only if the given byte is considered a word character.
/// This only applies to ASCII.
///
/// This was copied from regex-syntax so that we can use it to determine the
/// starting state while searching without depending on regex-syntax.
#[inline(always)]
pub fn is_word_byte(b: u8) -> bool {
    match b {
        b'_' | b'0'..=b'9' | b'a'..=b'z' | b'A'..=b'Z' => true,
        _ => false,
    }
}
