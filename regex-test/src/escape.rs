#![allow(dead_code)]

use std::ascii;
use std::str;

use bstr::{ByteSlice, ByteVec};

pub fn nice_raw_bytes(bytes: &[u8]) -> String {
    match str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        Err(_) => escape_bytes(bytes),
    }
}

pub fn escape_bytes(bytes: &[u8]) -> String {
    let escaped = bytes
        .iter()
        .flat_map(|&b| ascii::escape_default(b))
        .collect::<Vec<u8>>();
    String::from_utf8(escaped).unwrap()
}

pub fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| format!(r"\x{:02X}", b)).collect()
}

pub fn escape_default(s: &str) -> String {
    s.chars().flat_map(|c| c.escape_default()).collect()
}

pub fn escape(bytes: &[u8]) -> String {
    let mut escaped = String::new();
    for (s, e, ch) in bytes.char_indices() {
        if ch == '\u{FFFD}' {
            for b in bytes[s..e].bytes() {
                escape_byte(b, &mut escaped);
            }
        } else {
            escape_char(ch, &mut escaped);
        }
    }
    escaped
}

pub fn unescape<B: AsRef<[u8]>>(s: B) -> Vec<u8> {
    #[derive(Clone, Copy, Eq, PartialEq)]
    enum State {
        /// The state after seeing a `\`.
        Escape,
        /// The state after seeing a `\x`.
        HexFirst,
        /// The state after seeing a `\x[0-9A-Fa-f]`.
        HexSecond(char),
        /// Default state.
        Literal,
    }

    let mut bytes = vec![];
    let mut state = State::Literal;
    for c in s.as_ref().chars() {
        match state {
            State::Escape => match c {
                '\\' => {
                    bytes.push(b'\\');
                    state = State::Literal;
                }
                'n' => {
                    bytes.push(b'\n');
                    state = State::Literal;
                }
                'r' => {
                    bytes.push(b'\r');
                    state = State::Literal;
                }
                't' => {
                    bytes.push(b'\t');
                    state = State::Literal;
                }
                'x' => {
                    state = State::HexFirst;
                }
                c => {
                    bytes.push_char('\\');
                    bytes.push_char(c);
                    state = State::Literal;
                }
            },
            State::HexFirst => match c {
                '0'..='9' | 'A'..='F' | 'a'..='f' => {
                    state = State::HexSecond(c);
                }
                c => {
                    bytes.push_char('\\');
                    bytes.push_char('x');
                    bytes.push_char(c);
                    state = State::Literal;
                }
            },
            State::HexSecond(first) => match c {
                '0'..='9' | 'A'..='F' | 'a'..='f' => {
                    let ordinal = format!("{}{}", first, c);
                    let byte = u8::from_str_radix(&ordinal, 16).unwrap();
                    bytes.push_byte(byte);
                    state = State::Literal;
                }
                c => {
                    bytes.push_char('\\');
                    bytes.push_char('x');
                    bytes.push_char(first);
                    bytes.push_char(c);
                    state = State::Literal;
                }
            },
            State::Literal => match c {
                '\\' => {
                    state = State::Escape;
                }
                c => {
                    bytes.push_char(c);
                }
            },
        }
    }
    match state {
        State::Escape => bytes.push_char('\\'),
        State::HexFirst => bytes.push_str("\\x"),
        State::HexSecond(c) => {
            bytes.push_char('\\');
            bytes.push_char('x');
            bytes.push_char(c);
        }
        State::Literal => {}
    }
    bytes
}

/// Adds the given codepoint to the given string, escaping it if necessary.
fn escape_char(cp: char, into: &mut String) {
    if cp.is_ascii() {
        escape_byte(cp as u8, into);
    } else {
        into.push(cp);
    }
}

/// Adds the given byte to the given string, escaping it if necessary.
fn escape_byte(byte: u8, into: &mut String) {
    match byte {
        0x21..=0x5B | 0x5D..=0x7D => into.push(byte as char),
        b'\n' => into.push_str(r"\n"),
        b'\r' => into.push_str(r"\r"),
        b'\t' => into.push_str(r"\t"),
        b'\\' => into.push_str(r"\\"),
        _ => into.push_str(&format!(r"\x{:02X}", byte)),
    }
}

#[cfg(test)]
mod tests {
    use super::{escape, unescape};

    fn b(bytes: &'static [u8]) -> Vec<u8> {
        bytes.to_vec()
    }

    #[test]
    fn empty() {
        assert_eq!(b(b""), unescape(r""));
        assert_eq!(r"", escape(b""));
    }

    #[test]
    fn backslash() {
        assert_eq!(b(b"\\"), unescape(r"\\"));
        assert_eq!(r"\\", escape(b"\\"));
    }

    #[test]
    fn nul() {
        assert_eq!(b(b"\x00"), unescape(r"\x00"));
        assert_eq!(r"\x00", escape(b"\x00"));
    }

    #[test]
    fn nl() {
        assert_eq!(b(b"\n"), unescape(r"\n"));
        assert_eq!(r"\n", escape(b"\n"));
    }

    #[test]
    fn tab() {
        assert_eq!(b(b"\t"), unescape(r"\t"));
        assert_eq!(r"\t", escape(b"\t"));
    }

    #[test]
    fn carriage() {
        assert_eq!(b(b"\r"), unescape(r"\r"));
        assert_eq!(r"\r", escape(b"\r"));
    }

    #[test]
    fn nothing_simple() {
        assert_eq!(b(b"\\a"), unescape(r"\a"));
        assert_eq!(b(b"\\a"), unescape(r"\\a"));
        assert_eq!(r"\\a", escape(b"\\a"));
    }

    #[test]
    fn nothing_hex0() {
        assert_eq!(b(b"\\x"), unescape(r"\x"));
        assert_eq!(b(b"\\x"), unescape(r"\\x"));
        assert_eq!(r"\\x", escape(b"\\x"));
    }

    #[test]
    fn nothing_hex1() {
        assert_eq!(b(b"\\xz"), unescape(r"\xz"));
        assert_eq!(b(b"\\xz"), unescape(r"\\xz"));
        assert_eq!(r"\\xz", escape(b"\\xz"));
    }

    #[test]
    fn nothing_hex2() {
        assert_eq!(b(b"\\xzz"), unescape(r"\xzz"));
        assert_eq!(b(b"\\xzz"), unescape(r"\\xzz"));
        assert_eq!(r"\\xzz", escape(b"\\xzz"));
    }

    #[test]
    fn invalid_utf8() {
        assert_eq!(r"\xFF", escape(b"\xFF"));
        assert_eq!(r"a\xFFb", escape(b"a\xFFb"));
    }

    #[test]
    fn trailing_incomplete() {
        assert_eq!(b(b"\\xA"), unescape(r"\xA"));
    }
}
