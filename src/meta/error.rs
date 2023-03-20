use regex_syntax::{ast, hir};

use crate::{nfa, util::search::MatchError, PatternID};

#[derive(Clone, Debug)]
pub struct BuildError {
    kind: BuildErrorKind,
}

#[derive(Clone, Debug)]
enum BuildErrorKind {
    Ast { pid: PatternID, err: ast::Error },
    Hir { pid: PatternID, err: hir::Error },
    NFA(nfa::thompson::BuildError),
}

impl BuildError {
    /// If it is known which pattern ID caused this build error to occur, then
    /// this method returns it.
    ///
    /// Some errors are not associated with a particular pattern. However, any
    /// errors that occur as part of parsing a pattern are guaranteed to be
    /// associated with a pattern ID.
    pub fn pattern(&self) -> Option<PatternID> {
        match self.kind {
            BuildErrorKind::Ast { pid, .. } => Some(pid),
            BuildErrorKind::Hir { pid, .. } => Some(pid),
            _ => None,
        }
    }

    pub(crate) fn ast(pid: PatternID, err: ast::Error) -> BuildError {
        BuildError { kind: BuildErrorKind::Ast { pid, err } }
    }

    pub(crate) fn hir(pid: PatternID, err: hir::Error) -> BuildError {
        BuildError { kind: BuildErrorKind::Hir { pid, err } }
    }

    pub(crate) fn nfa(err: nfa::thompson::BuildError) -> BuildError {
        BuildError { kind: BuildErrorKind::NFA(err) }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self.kind {
            BuildErrorKind::Ast { ref err, .. } => Some(err),
            BuildErrorKind::Hir { ref err, .. } => Some(err),
            BuildErrorKind::NFA(ref err) => Some(err),
        }
    }
}

impl core::fmt::Display for BuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.kind {
            BuildErrorKind::Ast { pid, .. } => {
                write!(f, "error parsing pattern {} into AST", pid.as_usize())
            }
            BuildErrorKind::Hir { pid, .. } => {
                write!(
                    f,
                    "error translating pattern {} to HIR",
                    pid.as_usize()
                )
            }
            BuildErrorKind::NFA(_) => write!(f, "error building NFA"),
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
}

impl RetryFailError {
    pub(crate) fn from_offset(offset: usize) -> RetryFailError {
        RetryFailError { offset }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for RetryFailError {}

impl core::fmt::Display for RetryFailError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "regex engine failed at offset {:?}", self.offset)
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
            Quit { offset, .. } => RetryFailError::from_offset(offset),
            GaveUp { offset } => RetryFailError::from_offset(offset),
            HaystackTooLong { .. } | UnsupportedAnchored { .. } => {
                unreachable!("found impossible error in meta engine: {}", merr)
            }
        }
    }
}
