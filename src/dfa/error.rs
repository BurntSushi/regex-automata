use crate::{
    dfa::automaton::Start,
    id::{PatternID, StateID},
    nfa,
};

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
    /// An error that occurs if too many states are produced while building a
    /// DFA.
    TooManyStates,
    /// An error that occurs if too many start states are needed while building
    /// a DFA.
    ///
    /// This is a kind of oddball error that occurs when building a DFA with
    /// start states enabled for each pattern and enough patterns to cause
    /// the table of start states to overflow `usize`.
    TooManyStartStates,
    TooManyMatchPatternIDs,
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

    pub(crate) fn too_many_states() -> Error {
        Error { kind: ErrorKind::TooManyStates }
    }

    pub(crate) fn too_many_start_states() -> Error {
        Error { kind: ErrorKind::TooManyStartStates }
    }

    pub(crate) fn too_many_match_pattern_ids() -> Error {
        Error { kind: ErrorKind::TooManyMatchPatternIDs }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self.kind() {
            ErrorKind::NFA(ref err) => Some(err),
            ErrorKind::Unsupported(_) => None,
            ErrorKind::TooManyStates => None,
            ErrorKind::TooManyStartStates => None,
            ErrorKind::TooManyMatchPatternIDs => None,
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
            ErrorKind::TooManyStates => write!(
                f,
                "number of DFA states exceeds limit of {}",
                StateID::LIMIT,
            ),
            ErrorKind::TooManyStartStates => {
                let stride = Start::count();
                // The start table has `stride` entries for starting states for
                // the entire DFA, and then `stride` entries for each pattern
                // if start states for each pattern are enabled (which is the
                // only way this error can occur). Thus, the total number of
                // patterns that can fit in the table is `stride` less than
                // what we can allocate.
                let limit = ((isize::MAX as usize) - stride) / stride;
                write!(
                    f,
                    "compiling DFA with start states exceeds pattern \
                     pattern limit of {}",
                    limit,
                )
            }
            ErrorKind::TooManyMatchPatternIDs => write!(
                f,
                "compiling DFA with total patterns in all match states \
                 exceeds limit of {}",
                PatternID::LIMIT,
            ),
        }
    }
}
