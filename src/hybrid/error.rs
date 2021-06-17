use crate::nfa;

#[derive(Clone, Debug)]
pub struct Error {
    kind: ErrorKind,
}

#[derive(Clone, Debug)]
enum ErrorKind {
    NFA(nfa::Error),
    Unsupported(&'static str),
}

impl Error {
    fn kind(&self) -> &ErrorKind {
        &self.kind
    }

    pub(crate) fn nfa(err: nfa::Error) -> Error {
        Error { kind: ErrorKind::NFA(err) }
    }

    pub(crate) fn unsupported_dfa_word_boundary_unicode() -> Error {
        let msg = "cannot build lazy DFAs for regexes with Unicode word \
                   boundaries; switch to ASCII word boundaries, or \
                   heuristically enable Unicode word boundaries or use a \
                   different regex engine";
        Error { kind: ErrorKind::Unsupported(msg) }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self.kind() {
            ErrorKind::NFA(ref err) => Some(err),
            ErrorKind::Unsupported(_) => None,
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
        }
    }
}
