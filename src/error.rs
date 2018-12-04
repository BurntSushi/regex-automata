use std::error;
use std::fmt;
use std::result;

pub type Result<T> = result::Result<T, Error>;

/// An error that occurred during the construction of a DFA.
#[derive(Clone, Debug)]
pub struct Error {
    kind: ErrorKind,
}

/// The kind of error that occurred.
#[derive(Clone, Debug)]
pub enum ErrorKind {
    /// An error that occurred while parsing a regular expression. Note that
    /// this error may be printed over multiple lines, and is generally
    /// intended to be end user readable on its own.
    Syntax(String),
    /// An error that occurred because an unsupported regex feature was used.
    /// The message string describes which unsupported feature was used.
    ///
    /// The primary regex features that are unsupported are those that require
    /// look-around, such as the `^` and `$` anchors and the word boundary
    /// assertion `\b`. These may be supported in the future.
    Unsupported(String),
}

impl Error {
    /// Return the kind of this error.
    pub fn kind(&self) -> &ErrorKind {
        &self.kind
    }

    pub(crate) fn syntax(err: regex_syntax::Error) -> Error {
        Error { kind: ErrorKind::Syntax(err.to_string()) }
    }

    pub(crate) fn unsupported_anchor() -> Error {
        let msg = r"anchors such as ^, $, \A and \z are not supported";
        Error { kind: ErrorKind::Unsupported(msg.to_string()) }
    }

    pub(crate) fn unsupported_word() -> Error {
        let msg = r"word boundary assertions (\b and \B) are not supported";
        Error { kind: ErrorKind::Unsupported(msg.to_string()) }
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        match self.kind {
            ErrorKind::Syntax(_) => "syntax error",
            ErrorKind::Unsupported(_) => "unsupported syntax",
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.kind {
            ErrorKind::Syntax(ref msg) => write!(f, "{}", msg),
            ErrorKind::Unsupported(ref msg) => write!(f, "{}", msg),
        }
    }
}
