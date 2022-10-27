use regex_syntax::{ast, hir};

use crate::nfa;

#[derive(Clone, Debug)]
pub struct BuildError {
    kind: BuildErrorKind,
}

#[derive(Clone, Debug)]
enum BuildErrorKind {
    AST(ast::Error),
    HIR(hir::Error),
    NFA(nfa::thompson::BuildError),
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
}

#[cfg(feature = "std")]
impl std::error::Error for BuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self.kind {
            BuildErrorKind::AST(ref err) => Some(err),
            BuildErrorKind::HIR(ref err) => Some(err),
            BuildErrorKind::NFA(ref err) => Some(err),
        }
    }
}

impl core::fmt::Display for BuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.kind {
            BuildErrorKind::AST(_) => write!(f, "error parsing into AST"),
            BuildErrorKind::HIR(_) => write!(f, "error translating to HIR"),
            BuildErrorKind::NFA(_) => write!(f, "error building NFA"),
        }
    }
}
