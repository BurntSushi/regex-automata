use crate::util::id::{PatternID, StateID};

/// An error that can occured during the construction of an NFA.
#[derive(Clone, Debug)]
pub struct Error {
    kind: ErrorKind,
}

/// The kind of error that occurred during the construction of an NFA.
#[derive(Clone, Debug)]
enum ErrorKind {
    /// An error that occurred while parsing a regular expression. Note that
    /// this error may be printed over multiple lines, and is generally
    /// intended to be end user readable on its own.
    Syntax(regex_syntax::Error),
    /// An error that occurs if too many patterns were given to the NFA
    /// compiler.
    TooManyPatterns {
        /// The number of patterns given, which exceeds the limit.
        given: usize,
        /// The limit on the number of patterns.
        limit: usize,
    },
    /// An error that occurs if too states are produced while building an NFA.
    TooManyStates {
        /// The minimum number of states that are desired, which exceeds the
        /// limit.
        given: usize,
        /// The limit on the number of states.
        limit: usize,
    },
    /// An error that occurs when NFA compilation exceeds a configured heap
    /// limit.
    ExceededSizeLimit {
        /// The configured limit, in bytes.
        limit: usize,
    },
}

impl Error {
    fn kind(&self) -> &ErrorKind {
        &self.kind
    }

    pub(crate) fn syntax(err: regex_syntax::Error) -> Error {
        Error { kind: ErrorKind::Syntax(err) }
    }

    pub(crate) fn too_many_patterns(given: usize) -> Error {
        let limit = PatternID::LIMIT;
        Error { kind: ErrorKind::TooManyPatterns { given, limit } }
    }

    pub(crate) fn too_many_states(given: usize) -> Error {
        let limit = StateID::LIMIT;
        Error { kind: ErrorKind::TooManyStates { given, limit } }
    }

    pub(crate) fn exceeded_size_limit(limit: usize) -> Error {
        Error { kind: ErrorKind::ExceededSizeLimit { limit } }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self.kind() {
            ErrorKind::Syntax(ref err) => Some(err),
            ErrorKind::TooManyPatterns { .. } => None,
            ErrorKind::TooManyStates { .. } => None,
            ErrorKind::ExceededSizeLimit { .. } => None,
        }
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.kind() {
            ErrorKind::Syntax(_) => write!(f, "error parsing regex"),
            ErrorKind::TooManyPatterns { given, limit } => write!(
                f,
                "attemped to compile {} patterns, \
                 which exceeds the limit of {}",
                given, limit,
            ),
            ErrorKind::TooManyStates { given, limit } => write!(
                f,
                "attemped to compile {} NFA states, \
                 which exceeds the limit of {}",
                given, limit,
            ),
            ErrorKind::ExceededSizeLimit { limit } => write!(
                f,
                "heap usage during NFA compilation exceeded limit of {:?}",
                limit,
            ),
        }
    }
}
