use regex_syntax::{ast, hir};

use crate::{hybrid, nfa, MatchError};

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

/// A special error that is internal to only the meta regex engine.
///
/// The main problem this type solves is a way to classify the different types
/// of errors that can occur while searching. This is important because the
/// *response* to a particular error depends on its type. For example, if one
/// gets a "quit" or "gave up" match error, then you must retry the search with
/// a regex engine that can't fail with that error. But if you get a "invalid
/// pattern ID" error, then that *must* have been the fault of the caller
/// directly, and that error should be propagated backup without any kind of
/// retrying.
///
/// This sort of case analysis goes even deeper. For example, both the reverse
/// suffix and reverse inner optimizations can quit if they detect quadratic
/// runtime behavior. We can't just use the normal `MatchError::GaveUp` for
/// this, because when that occurs, we assume it came from a full/lazy DFA and
/// thus jump to one of the slower NFA engines (including the onepass DFA).
/// Namely, if the reverse suffix/inner optimization fails, we actually want
/// to retry the search using the full/lazy DFA, since we don't yet have any
/// evidence that they will also fail. And since they are substantially quicker
/// than the NFA engines, it is worthwhile to get this right.
///
/// There are likely other designs that would ameliorate this. For example,
/// we might have a distinct error type for every regex engine. But I found
/// this idea to inhibit composition too much and makes inter-operation far too
/// annoying.
///
/// Note also that not all match errors can be represented by this type.
/// Namely, `MetaMatchError::from` will panic for any match error that is
/// deemed to be impossible to occur within the context of the meta regex
/// engine.
///
/// # Panics
///
/// A `MetaMatchError` always has to be converted back to a `MatchError`
/// before being propagated to the caller. This will automatically happen via
/// `?` because of the `From<MetaMatchError> for MatchError` impl. Crucially
/// though, this impl will panic if the `MetaMatchError` is anything but a
/// caller provoked error. Why? Because any other kind of match error must
/// never surface to the caller. The meta regex engine must contain those
/// errors and use different regex engines to service the request.
///
/// This "panic on error propagation" is perhaps a little footgunny... But
/// it's at least an internal-only mechanism, so we can get rid of it if it
/// turns out to be a persistent annoyance.
#[derive(Debug)]
pub(crate) enum MetaMatchError {
    /// This occurs when the match error corresponds to something wrong with
    /// the parameters given to the regex engine by the caller. These errors
    /// are always bubbled back up immediately to the caller of the meta regex
    /// engine when they occur. More to the point, errors in this category
    /// never occur as part of something that the meta regex engine itself does
    /// internally. (And if it does, there's a bug.)
    CallerProvoked(MatchError),
    /// This occurs when either the lazy DFA or a full DFA have heuristically
    /// stopped the search. This occurs only when seeing a non-ASCII byte when
    /// the regex contains a Unicode word boundary, or in the case of the lazy
    /// DFA, when it is believed that its cache usage is inefficient.
    ///
    /// Generally speaking, when this occurs, the current search should avoid
    /// re-running the DFA and use a regex engine that never fails (PikeVM,
    /// bounded backtracker or onepass DFA).
    FailedDFA(MatchError),
    /// This occurs when the reverse suffix/inner optimization in the meta
    /// regex engine is stopped in order to avoid quadratic behavior. When this
    /// happens, one generally wants to fall back to a DFA engine if possible.
    /// Indeed, that is the motivation for distinguishing this error from a
    /// generic "failed DFA" error. In the latter case, the fallback should
    /// be a slower NFA. But in this case, the fallback should be the normal
    /// forward DFA.
    FailedReverseOpt,
}

impl MetaMatchError {
    /// Creates a new meta match error representing a failed attempt at a
    /// reverse inner or reverse suffix optimization. This usually occurs when
    /// the optimization detects that quadratic runtime could occur.
    pub(crate) fn failed_reverse_opt() -> MetaMatchError {
        MetaMatchError::FailedReverseOpt
    }

    /// Returns true if and only if this error was provoked by the caller and
    /// should thus be returned immediately without trying any other fallback
    /// routines. (Because they would fail with the exact same error.)
    pub(crate) fn is_caller_provoked(&self) -> bool {
        matches!(*self, MetaMatchError::CallerProvoked(_))
    }

    /// Returns true if and only if this error corresponds to heuristic errors
    /// returned by either a full or lazy DFA. When this occurs, the meta
    /// regex engine is forced to retry the search with one of the slower but
    /// infallible NFA engines (including the onepass DFA).
    pub(crate) fn is_failed_dfa(&self) -> bool {
        matches!(*self, MetaMatchError::FailedDFA(_))
    }

    /// Returns true if and only if this error corresponds to a failed
    /// optimization, usually as a result of bailing out of undesirable
    /// behavior or performance. When this error occurs, the meta regex engine
    /// should fall back to standard full/lazy DFA if possible, and *not*
    /// immediately jump to one of the slower NFA engines.
    pub(crate) fn is_failed_opt(&self) -> bool {
        matches!(*self, MetaMatchError::FailedReverseOpt)
    }
}

impl From<MatchError> for MetaMatchError {
    fn from(err: MatchError) -> MetaMatchError {
        use crate::MatchErrorKind::*;

        match err.kind() {
			// These are only ever returned by the lazy or full DFAs when they
			// couldn't complete their search.
            Quit { .. } | GaveUp { .. } => MetaMatchError::FailedDFA(err),
			// These are only ever provoked by the caller when an invalid
			// pattern ID or an inappropriately sized pattern set are provided.
            InvalidInputPattern { .. } | InvalidSetCapacity { .. } => {
                MetaMatchError::CallerProvoked(err)
            }
			// Impossible because it can only be reported by the bounded
			// backtracker, and we never use the backtracker unless we know
			// the haystack is short enough.
            HaystackTooLong { .. }
			// Impossible because all regex engines in the meta regex engine
			// are configured to support anchored searches.
            | InvalidInputAnchored { .. }
            // Impossible because all regex engines except for the onepass DFA
            // in the meta regex engine are configured to support unanchored
            // searches. And we only use the onepass DFA when executing an
            // anchored search.
            | InvalidInputUnanchored { .. } => {
                unreachable!(
					"error impossible in meta regex engine: {:?}",
					err,
                )
            }
        }
    }
}

/*
impl From<MetaMatchError> for MatchError {
    fn from(merr: MetaMatchError) -> MatchError {
        match merr {
            MetaMatchError::CallerProvoked(err) => err,
            _ => unreachable!(
                "non-caller provoked meta match errors must be handled \
                 before propagation to a MatchError: {:?}",
                merr,
            ),
        }
    }
}
*/
