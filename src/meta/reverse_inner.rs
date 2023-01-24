use alloc::vec;

use regex_syntax::hir::{self, literal, Hir, HirKind};

use crate::{util::prefilter::Prefilter, MatchKind};

/// Attempts to extract an "inner" prefilter from the given HIR expressions. If
/// one was found, then a concatenation of the HIR expressions that precede it
/// is returned.
///
/// The idea here is that the prefilter returned can be used to find candidate
/// matches. And then the HIR returned can be used to build a reverse regex
/// matcher, which will find the start of the candidate match. Finally, the
/// match still has to be confirmed with a normal anchored forward scan to find
/// the end position of the match.
///
/// Note that this assumes leftmost-first match semantics, so callers must
/// not call this otherwise.
pub(crate) fn extract(hirs: &[&Hir]) -> Option<(Hir, Prefilter)> {
    if hirs.len() != 1 {
        debug!(
            "skipping reverse inner optimization since it only \
		 	 supports 1 pattern, {} were given",
            hirs.len(),
        );
        return None;
    }
    let concat = match top_concat(hirs[0]) {
        Some(concat) => concat,
        None => {
            debug!(
                "skipping reverse inner optimization because a top-level \
		 	     concatenation could not found",
            );
            return None;
        }
    };
    // We skip the first HIR because if it did have a prefix prefilter in it,
    // we probably wouldn't be here looking for an inner prefilter.
    for (i, hir) in concat.iter().enumerate().skip(1) {
        let pre = match prefilter(hir) {
            None => continue,
            Some(pre) => pre,
        };
        // Even if we got a prefilter, if it isn't consider "fast," then we
        // probably don't want to bother with it. Namely, since the reverse
        // inner optimization requires some overhead, it likely only makes
        // sense if the prefilter scan itself is (believed) to be much faster
        // than the regex engine.
        if !pre.is_fast() {
            debug!(
                "skipping extracted inner prefilter because \
				 it probably isn't fast"
            );
            continue;
        }
        let concat_prefix =
            Hir::concat(concat.iter().take(i).map(|h| h.clone()).collect());
        return Some((concat_prefix, pre));
    }
    debug!(
        "skipping reverse inner optimization because a top-level \
	     sub-expression with a fast prefilter could not be found"
    );
    None
}

/// Attempt to extract a prefilter from an HIR expression.
///
/// We do a little massaging here to do our best that the prefilter we get out
/// of this is *probably* fast. Basically, the false positive rate has a much
/// higher impact for things like the reverse inner optimization because more
/// work needs to potentially be done for each candidate match.
///
/// Note that this assumes leftmost-first match semantics, so callers must
/// not call this otherwise.
fn prefilter(hir: &Hir) -> Option<Prefilter> {
    let mut extractor = literal::Extractor::new();
    extractor.kind(literal::ExtractKind::Prefix);
    let mut prefixes = extractor.extract(hir);
    debug!(
        "inner prefixes (len={:?}) extracted before optimization: {:?}",
        prefixes.len(),
        prefixes
    );
    // Since these are inner literals, we know they cannot be exact. But the
    // extractor doesn't know this. We mark them as inexact because this might
    // impact literal optimization. Namely, optimization weights "all literals
    // are exact" as very high, because it presumes that any match results in
    // an overall match. But of course, that is not the case here.
    //
    // In practice, this avoids plucking out a ASCII-only \s as an alternation
    // of single-byte whitespace characters.
    prefixes.make_inexact();
    prefixes.optimize_for_prefix_by_preference();
    debug!(
        "inner prefixes (len={:?}) extracted after optimization: {:?}",
        prefixes.len(),
        prefixes
    );
    prefixes
        .literals()
        .and_then(|lits| Prefilter::new(MatchKind::LeftmostFirst, lits))
}

/// Looks for a "top level" HirKind::Concat item in the given HIR. This will
/// try to return one even if it's embedded in a capturing group, but is
/// otherwise pretty conservative in what is returned.
fn top_concat(mut hir: &Hir) -> Option<&[Hir]> {
    loop {
        hir = match hir.kind() {
            HirKind::Empty
            | HirKind::Literal(_)
            | HirKind::Class(_)
            | HirKind::Look(_)
            | HirKind::Repetition(_)
            | HirKind::Alternation(_) => return None,
            HirKind::Group(hir::Group { ref hir, .. }) => hir,
            HirKind::Concat(ref hirs) => return Some(hirs),
        };
    }
}
