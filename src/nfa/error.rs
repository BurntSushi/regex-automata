/// An error that can occured during the construction of an NFA.
#[derive(Clone, Debug)]
pub struct Error {
    kind: ErrorKind,
}

/// The kind of error that occurred during the construction of an NFA.
#[derive(Clone, Debug)]
pub enum ErrorKind {
    /// An error that occurred while parsing a regular expression. Note that
    /// this error may be printed over multiple lines, and is generally
    /// intended to be end user readable on its own.
    Syntax(regex_syntax::Error),
    /// An error that occurs if too many patterns were given to the NFA
    /// compiler. The limit is `cmp::min(usize::MAX, u32::MAX) - 1`.
    TooManyPatterns {
        /// The number of patterns given, which exceeds the limit.
        given: usize,
        /// The limit on the number of patterns.
        limit: usize,
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

    pub(crate) fn syntax(err: regex_syntax::Error) -> Error {
        Error { kind: ErrorKind::Syntax(err) }
    }

    pub(crate) fn too_many_patterns(given: usize, limit: usize) -> Error {
        Error { kind: ErrorKind::TooManyPatterns { given, limit } }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self.kind() {
            ErrorKind::Syntax(ref err) => Some(err),
            ErrorKind::TooManyPatterns { .. } => None,
            ErrorKind::__Nonexhaustive => unreachable!(),
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind() {
            ErrorKind::Syntax(_) => write!(f, "error parsing regex"),
            ErrorKind::TooManyPatterns { given, limit } => write!(
                f,
                "attemped to compile {} patterns, \
                 which exceeds the limit of {}",
                given, limit,
            ),
            ErrorKind::__Nonexhaustive => unreachable!(),
        }
    }
}
