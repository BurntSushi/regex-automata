use regex_syntax;

use crate::nfa;

/// An error that occurred during the construction of a DFA.
#[derive(Clone, Debug)]
pub struct Error {
    kind: ErrorKind,
}

/// The kind of error that occurred during the construction of a DFA.
#[derive(Clone, Debug)]
pub enum ErrorKind {
    /// An error that occurred while constructing an NFA as a precursor step
    /// before a DFA is compiled.
    NFA(nfa::Error),
    /// An error that occurred because an unsupported regex feature was used.
    /// The message string describes which unsupported feature was used.
    ///
    /// The primary regex feature that is unsupported by DFAs is the Unicode
    /// word boundary look-around assertion (`\b`). This can be worked around
    /// by either using an ASCII word boundary (`(?-u:\b)`) or by enabling the
    /// [`dense::Builder::allow_unicode_word_boundary`](dense/struct.Builder.html#method.allow_unicode_word_boundary)
    /// option when building a DFA.
    Unsupported(String),
    /// An error that occurred when attempting to serialize a DFA to bytes. The
    /// message string describes the problem.
    Serialize(&'static str),
    /// An error that occurred when attempting to deserialize a DFA from bytes.
    /// The message string describes the problem.
    Deserialize(&'static str),
    /// An error that occurs when constructing a DFA would require the use
    /// of a state ID that overflows the chosen state ID representation. For
    /// example, if one is using `u8` for state IDs and builds a DFA with too
    /// many states, then it's possible that `u8` will not be able to represent
    /// all state IDs. If this happens, then DFA construction will fail and
    /// this error kind will be returned.
    StateIDOverflow {
        /// The maximum possible state ID.
        max: usize,
    },
    /// Hints that destructuring should not be exhaustive.
    ///
    /// This enum may grow additional variants, so this makes sure clients
    /// don't count on exhaustive matching. (Otherwise, adding a new variant
    /// could break existing code.)
    #[doc(hidden)]
    __Nonexhaustive,
}

impl Error {
    /// Return the kind of this error.
    pub fn kind(&self) -> &ErrorKind {
        &self.kind
    }

    pub(crate) fn nfa(err: nfa::Error) -> Error {
        Error { kind: ErrorKind::NFA(err) }
    }

    pub(crate) fn unsupported_dfa_word_boundary_unicode() -> Error {
        let msg = "cannot build DFAs for regexes with Unicode word \
                   boundaries; switch to ASCII word boundaries, or \
                   heuristically enable Unicode word boundaries or use a \
                   different regex engine";
        Error { kind: ErrorKind::Unsupported(msg.to_string()) }
    }

    pub(crate) fn serialize(message: &'static str) -> Error {
        Error { kind: ErrorKind::Serialize(message) }
    }

    pub(crate) fn deserialize(message: &'static str) -> Error {
        Error { kind: ErrorKind::Deserialize(message) }
    }

    pub(crate) fn state_id_overflow(max: usize) -> Error {
        Error { kind: ErrorKind::StateIDOverflow { max } }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self.kind() {
            ErrorKind::NFA(ref err) => Some(err),
            ErrorKind::Unsupported(_) => None,
            ErrorKind::Serialize(_) => None,
            ErrorKind::Deserialize(_) => None,
            ErrorKind::StateIDOverflow { .. } => None,
            ErrorKind::__Nonexhaustive => unreachable!(),
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind() {
            ErrorKind::NFA(_) => write!(f, "error building NFA"),
            ErrorKind::Unsupported(ref msg) => {
                write!(f, "unsupported regex feature for DFAs: {}", msg)
            }
            ErrorKind::Serialize(ref msg) => {
                write!(f, "DFA serialization error: {}", msg)
            }
            ErrorKind::Deserialize(ref msg) => {
                write!(f, "DFA deserialization error, DFA is corrupt: {}", msg)
            }
            ErrorKind::StateIDOverflow { max } => write!(
                f,
                "building the DFA failed because it required building \
                 more states that can be identified, where the maximum \
                 ID for the chosen representation is {}",
                max,
            ),
            ErrorKind::__Nonexhaustive => unreachable!(),
        }
    }
}
