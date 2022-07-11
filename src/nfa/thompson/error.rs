use crate::util::primitives::{PatternID, StateID};

/// An error that can occured during the construction of a thompson NFA.
///
/// This error does not provide many introspection capabilities. There are
/// generally only two things you can do with it:
///
/// * Obtain a human readable message via its `std::fmt::Display` impl.
/// * Access an underlying [`regex_syntax::Error`] type from its `source`
/// method via the `std::error::Error` trait. This error only occurs when using
/// convenience routines for building an NFA directly from a pattern string.
///
/// Otherwise, errors typically occur when a limit has been breeched. For
/// example, if the total heap usage of the compiled NFA exceeds the limit
/// set by [`Config::nfa_size_limit`](crate::nfa::thompson::Config), then
/// building the NFA will fail.
#[derive(Clone, Debug)]
pub struct Error {
    kind: ErrorKind,
}

/// The kind of error that occurred during the construction of a thompson NFA.
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
    /// An error that occurs when an invalid capture group index is added to
    /// the NFA. An "invalid" index can be one that is too big (e.g., results
    /// in an integer overflow) or one that is discontinuous from previous
    /// capture group indices added.
    InvalidCapture {
        /// The invalid index that was given.
        index: usize,
    },
    /// An error that occurs when one tries to provide a name for the capture
    /// group at index 0. This capturing group must currently always be
    /// unnamed.
    FirstCaptureMustBeUnnamed,
    /// An error that occurs when an NFA contains a Unicode word boundary, but
    /// where the crate was compiled without the necessary data for dealing
    /// with Unicode word boundaries.
    UnicodeWordUnavailable,
    /// An error that occurs when one tries to build an NFA simulation (such as
    /// the PikeVM) without any capturing groups.
    MissingCaptures,
    /// An error that occurs when one tries to build a reverse NFA with
    /// captures enabled. Currently, this isn't supported, but we probably
    /// should support it at some point.
    UnsupportedCaptures,
    /// An error that occurs when duplicate capture group names for the same
    /// pattern are added to the NFA builder.
    ///
    /// NOTE: This error can never occur if you're using regex-syntax, since
    /// the parser itself will reject patterns with duplicate capture group
    /// names. This error can only occur when the builder is used to hand
    /// construct NFAs.
    DuplicateCaptureName {
        /// The pattern in which the duplicate capture group name was found.
        pattern: PatternID,
        /// The duplicate name.
        name: String,
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

    pub(crate) fn invalid_capture(index: usize) -> Error {
        Error { kind: ErrorKind::InvalidCapture { index } }
    }

    pub(crate) fn first_capture_must_be_unnamed() -> Error {
        Error { kind: ErrorKind::FirstCaptureMustBeUnnamed }
    }

    pub(crate) fn unicode_word_unavailable() -> Error {
        Error { kind: ErrorKind::UnicodeWordUnavailable }
    }

    pub(crate) fn missing_captures() -> Error {
        Error { kind: ErrorKind::MissingCaptures }
    }

    pub(crate) fn unsupported_captures() -> Error {
        Error { kind: ErrorKind::UnsupportedCaptures }
    }

    pub(crate) fn duplicate_capture_name(
        pattern: PatternID,
        name: &str,
    ) -> Error {
        Error {
            kind: ErrorKind::DuplicateCaptureName {
                pattern,
                name: name.to_string(),
            },
        }
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
            ErrorKind::InvalidCapture { .. } => None,
            ErrorKind::FirstCaptureMustBeUnnamed => None,
            ErrorKind::UnicodeWordUnavailable => None,
            ErrorKind::MissingCaptures => None,
            ErrorKind::UnsupportedCaptures => None,
            ErrorKind::DuplicateCaptureName { .. } => None,
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
                "heap usage during NFA compilation exceeded limit of {}",
                limit,
            ),
            ErrorKind::InvalidCapture { index } => write!(
                f,
                "capture group index {} is invalid (too big or discontinuous)",
                index,
            ),
            ErrorKind::FirstCaptureMustBeUnnamed => write!(
                f,
                "first capture group (at index 0) must always be unnamed",
            ),
            ErrorKind::UnicodeWordUnavailable => write!(
                f,
                "crate has been compiled without Unicode word boundary \
                 support, but the NFA contains Unicode word boundary \
                 assertions",
            ),
            ErrorKind::MissingCaptures => write!(
                f,
                "operation requires the NFA to have capturing groups, \
                 but the NFA given contains none",
            ),
            ErrorKind::UnsupportedCaptures => write!(
                f,
                "currently captures must be disabled when compiling \
                 a reverse NFA",
            ),
            ErrorKind::DuplicateCaptureName { pattern, ref name } => write!(
                f,
                "duplicate capture group name '{}' found for pattern {}",
                name,
                pattern.as_usize(),
            ),
        }
    }
}
