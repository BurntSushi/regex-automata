use crate::{id::StateID, nfa};

/// An error that occurred during the construction of a DFA.
#[derive(Clone, Debug)]
pub struct Error {
    kind: ErrorKind,
}

/// The kind of error that occurred during the construction of a DFA.
///
/// Note that this error is non-exhaustive. Adding new variants is not
/// considered a breaking change.
#[non_exhaustive]
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
    Unsupported(&'static str),
    /// An error that occurs if too states are produced while building an NFA.
    TooManyStates {
        /// The minimum number of states that are desired, which exceeds the
        /// limit.
        given: usize,
        /// The limit on the number of states.
        limit: usize,
    },
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
        Error { kind: ErrorKind::Unsupported(msg) }
    }

    pub(crate) fn too_many_states(given: usize) -> Error {
        let limit = StateID::LIMIT;
        Error { kind: ErrorKind::TooManyStates { given, limit } }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self.kind() {
            ErrorKind::NFA(ref err) => Some(err),
            ErrorKind::Unsupported(_) => None,
            ErrorKind::TooManyStates { .. } => None,
        }
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.kind() {
            ErrorKind::NFA(_) => write!(f, "error building NFA"),
            ErrorKind::Unsupported(ref msg) => {
                write!(f, "unsupported regex feature for DFAs: {}", msg)
            }
            ErrorKind::TooManyStates { given, limit } => write!(
                f,
                "attemped to compile {} DFA states, \
                 which exceeds the limit of {}",
                given, limit,
            ),
        }
    }
}
