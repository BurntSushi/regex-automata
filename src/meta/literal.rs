use alloc::{vec, vec::Vec};

use regex_syntax::hir::{literal, Hir};

use crate::{meta::regex::RegexInfo, util::search::MatchKind};

/// Extracts all of the prefix and suffix literals from the given HIR
/// expressions into a single `Seq` each. The literals in the sequence are
/// ordered with respect to the order of the given HIR expressions.
///
/// The sequences returned are each "optimized." That is, they may be
/// shrunk or even truncated according to heuristics with the intent of
/// making them more useful as a prefilter. (Which translates to both
/// using faster algorithms and minimizing the false positive rate.)
///
/// Note that this erases any connection between the literals and which
/// pattern (or patterns) they came from.
///
/// The match kind given must correspond to the match semantics of the
/// regex that is represented by the HIRs given. The match semantics may
/// change the literal sets returned.
pub(crate) fn prefixes<H>(kind: MatchKind, hirs: &[H]) -> literal::Seq
where
    H: core::borrow::Borrow<Hir>,
{
    let mut extractor = literal::Extractor::new();
    extractor.kind(literal::ExtractKind::Prefix);

    let mut prefixes = literal::Seq::empty();
    for hir in hirs {
        prefixes.union(&mut extractor.extract(hir.borrow()));
    }
    debug!(
        "prefixes (len={:?}) extracted before optimization: {:?}",
        prefixes.len(),
        prefixes
    );
    match kind {
        MatchKind::All => {
            prefixes.sort();
            prefixes.dedup();
        }
        MatchKind::LeftmostFirst => {
            prefixes.optimize_for_prefix_by_preference();
        }
    }
    debug!(
        "prefixes (len={:?}) extracted after optimization: {:?}",
        prefixes.len(),
        prefixes
    );
    prefixes
}

pub(crate) fn suffixes<H>(kind: MatchKind, hirs: &[H]) -> literal::Seq
where
    H: core::borrow::Borrow<Hir>,
{
    let mut extractor = literal::Extractor::new();
    extractor.kind(literal::ExtractKind::Suffix);

    let mut suffixes = literal::Seq::empty();
    for hir in hirs {
        suffixes.union(&mut extractor.extract(hir.borrow()));
    }
    debug!(
        "suffixes (len={:?}) extracted before optimization: {:?}",
        suffixes.len(),
        suffixes
    );
    match kind {
        MatchKind::All => {
            suffixes.sort();
            suffixes.dedup();
        }
        MatchKind::LeftmostFirst => {
            suffixes.optimize_for_suffix_by_preference();
        }
    }
    debug!(
        "suffixes (len={:?}) extracted after optimization: {:?}",
        suffixes.len(),
        suffixes
    );
    suffixes
}

/// Pull out an alternation of literals from the given sequence of HIR
/// expressions.
///
/// There are numerous ways for this to fail. Generally, this only applies
/// to regexes of the form 'foo|bar|baz|...|quux'. It can also fail if there
/// are "too few" alternates, in which case, the regex engine is likely faster.
///
/// And currently, this only returns something when 'hirs.len() == 1'.
pub(crate) fn alternation_literals(
    info: &RegexInfo,
    hirs: &[&Hir],
) -> Option<Vec<Vec<u8>>> {
    use regex_syntax::hir::{HirKind, Literal};

    // Might as well skip the work below if we know we can't build an
    // Aho-Corasick searcher.
    if !cfg!(feature = "perf-literal-multisubstring") {
        return None;
    }
    // This is pretty hacky, but basically, if `is_alternation_literal` is
    // true, then we can make several assumptions about the structure of our
    // HIR. This is what justifies the `unreachable!` statements below.
    //
    // This code should be refactored once we overhaul this crate's
    // optimization pipeline, because this is a terribly inflexible way to go
    // about things.
    if hirs.len() != 1
        || !info.props()[0].look_set().is_empty()
        || info.props()[0].explicit_captures_len() > 0
        || !info.props()[0].is_alternation_literal()
        || info.config().get_match_kind() != MatchKind::LeftmostFirst
    {
        return None;
    }
    let hir = &hirs[0];
    let alts = match *hir.kind() {
        HirKind::Alternation(ref alts) => alts,
        _ => return None, // one literal isn't worth it
    };

    let mut lits = vec![];
    for alt in alts {
        let mut lit = vec![];
        match *alt.kind() {
            HirKind::Literal(Literal(ref bytes)) => {
                lit.extend_from_slice(bytes)
            }
            HirKind::Concat(ref exprs) => {
                for e in exprs {
                    match *e.kind() {
                        HirKind::Literal(Literal(ref bytes)) => {
                            lit.extend_from_slice(bytes);
                        }
                        _ => unreachable!("expected literal, got {:?}", e),
                    }
                }
            }
            _ => unreachable!("expected literal or concat, got {:?}", alt),
        }
        lits.push(lit);
    }
    // Why do this? Well, when the number of literals is small, it's likely
    // that we'll use the lazy DFA which is in turn likely to be faster than
    // Aho-Corasick in such cases. Primarily because Aho-Corasick doesn't have
    // a "lazy DFA" but either a contiguous NFA or a full DFA. We rarely use
    // the latter because it is so hungry (in time and space), and the former
    // is decently fast, but not as fast as a well oiled lazy DFA.
    //
    // However, once the number starts getting large, the lazy DFA is likely
    // to start thrashing because of the modest default cache size. When
    // exactly does this happen? Dunno. But at whatever point that is (we make
    // a guess below based on ad hoc benchmarking), we'll want to cut over to
    // Aho-Corasick, where even the contiguous NFA is likely to do much better.
    if lits.len() < 3000 {
        debug!("skipping Aho-Corasick because there are too few literals");
        return None;
    }
    Some(lits)
}
