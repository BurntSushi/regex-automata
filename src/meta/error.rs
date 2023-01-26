use regex_syntax::{ast, hir};

use crate::{hybrid, nfa, util::search::MatchError};

#[derive(Clone, Debug)]
pub struct BuildError {
    kind: BuildErrorKind,
}

#[derive(Clone, Debug)]
enum BuildErrorKind {
    AST(ast::Error),
    HIR(hir::Error),
    NFA(nfa::thompson::BuildError),
    Hybrid(hybrid::BuildError),
}

impl BuildError {
    pub(crate) fn ast(err: ast::Error) -> BuildError {
        BuildError { kind: BuildErrorKind::AST(err) }
    }

    pub(crate) fn hir(err: hir::Error) -> BuildError {
        BuildError { kind: BuildErrorKind::HIR(err) }
    }

    pub(crate) fn nfa(err: nfa::thompson::BuildError) -> BuildError {
        BuildError { kind: BuildErrorKind::NFA(err) }
    }

    pub(crate) fn hybrid(err: hybrid::BuildError) -> BuildError {
        BuildError { kind: BuildErrorKind::Hybrid(err) }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self.kind {
            BuildErrorKind::AST(ref err) => Some(err),
            BuildErrorKind::HIR(ref err) => Some(err),
            BuildErrorKind::NFA(ref err) => Some(err),
            BuildErrorKind::Hybrid(ref err) => Some(err),
        }
    }
}

impl core::fmt::Display for BuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.kind {
            BuildErrorKind::AST(_) => write!(f, "error parsing into AST"),
            BuildErrorKind::HIR(_) => write!(f, "error translating to HIR"),
            BuildErrorKind::NFA(_) => write!(f, "error building NFA"),
            BuildErrorKind::Hybrid(_) => {
                write!(f, "error building hybrid NFA/DFA")
            }
        }
    }
}

#[derive(Debug)]
pub(crate) enum RetryError {
    Quadratic(RetryQuadraticError),
    Fail(RetryFailError),
}

#[cfg(feature = "std")]
impl std::error::Error for RetryError {}

impl core::fmt::Display for RetryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            RetryError::Quadratic(ref err) => err.fmt(f),
            RetryError::Fail(ref err) => err.fmt(f),
        }
    }
}

impl From<MatchError> for RetryError {
    fn from(merr: MatchError) -> RetryError {
        RetryError::Fail(RetryFailError::from(merr))
    }
}

#[derive(Debug)]
pub(crate) struct RetryQuadraticError(());

impl RetryQuadraticError {
    pub(crate) fn new() -> RetryQuadraticError {
        RetryQuadraticError(())
    }
}

#[cfg(feature = "std")]
impl std::error::Error for RetryQuadraticError {}

impl core::fmt::Display for RetryQuadraticError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "regex engine gave up to avoid quadratic behavior")
    }
}

impl From<RetryQuadraticError> for RetryError {
    fn from(err: RetryQuadraticError) -> RetryError {
        RetryError::Quadratic(err)
    }
}

#[derive(Debug)]
pub(crate) struct RetryFailError {
    offset: usize,
    byte: Option<u8>,
}

impl RetryFailError {
    pub(crate) fn from_offset(offset: usize) -> RetryFailError {
        RetryFailError { offset, byte: None }
    }

    pub(crate) fn from_offset_byte(offset: usize, byte: u8) -> RetryFailError {
        RetryFailError { offset, byte: Some(byte) }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for RetryFailError {}

impl core::fmt::Display for RetryFailError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        use crate::util::escape::DebugByte;

        if let Some(byte) = self.byte {
            write!(
                f,
                "regex engine failed for byte {:?} at offset {:?}",
                DebugByte(byte),
                self.offset
            )
        } else {
            write!(f, "regex engine failed at offset {:?}", self.offset)
        }
    }
}

impl From<RetryFailError> for RetryError {
    fn from(err: RetryFailError) -> RetryError {
        RetryError::Fail(err)
    }
}

impl From<MatchError> for RetryFailError {
    fn from(merr: MatchError) -> RetryFailError {
        use crate::util::search::MatchErrorKind::*;

        match *merr.kind() {
            Quit { byte, offset } => {
                RetryFailError::from_offset_byte(offset, byte)
            }
            GaveUp { offset } => RetryFailError::from_offset(offset),
            HaystackTooLong { .. } | UnsupportedAnchored { .. } => {
                unreachable!("found impossible error in meta engine: {}", merr)
            }
        }
    }
}
