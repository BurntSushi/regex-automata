use core::{ascii, fmt, str};

/// A type that wraps a single byte with a convenient fmt::Debug impl that
/// escapes the byte.
pub struct DebugByte(pub u8);

impl fmt::Debug for DebugByte {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt_byte(f, self.0)
    }
}

/// Write the byte in its escaped form (using ascii::escape_default) to the
/// given formatter without allocating.
pub fn fmt_byte(f: &mut fmt::Formatter, b: u8) -> fmt::Result {
    // 10 bytes is enough to cover any output from ascii::escape_default.
    let mut bytes = [0u8; 10];
    let mut len = 0;
    for (i, mut b) in ascii::escape_default(b).enumerate() {
        // capitalize \xab to \xAB
        if i >= 2 && b'a' <= b && b <= b'f' {
            b -= 32;
        }
        bytes[len] = b;
        len += 1;
    }
    write!(f, r"{}", str::from_utf8(&bytes[..len]).unwrap())
}
