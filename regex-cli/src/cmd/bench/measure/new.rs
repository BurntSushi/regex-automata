/*!
This module defines convenience constructors for the various regex engines that
we benchmark. For most benchmarks, building a regex is a simple matter of taking
the parameters found in a 'Benchmark' and feeding them to an engine-specific
constructor.

One special case here is the regex-redux benchmark, which involves building
many regexes specific to that benchmark. In that case, the regex-redux
benchmark copies most of the constructors here, since there is no 'b.def.regex'
available to build.
*/

use super::Benchmark;

/// Constructor for the Rust regex API engine.
pub(super) fn regex_api(b: &Benchmark) -> anyhow::Result<regex::bytes::Regex> {
    let re = regex::bytes::RegexBuilder::new(&b.def.regex)
        .unicode(b.def.unicode)
        .case_insensitive(b.def.case_insensitive)
        .build()?;
    Ok(re)
}

/// Constructor for the fully compiled "dense" DFA.
pub(super) fn regex_automata_dfa_dense(
    b: &Benchmark,
) -> anyhow::Result<automata::dfa::regex::Regex> {
    use automata::dfa::regex::Regex;

    let re = Regex::builder()
        // Disabling UTF-8 here just means that iterators built by this regex
        // may report matches that split a UTF-8 encoding of a codepoint.
        .configure(Regex::config().utf8(false))
        .syntax(automata_syntax_config(b))
        .build(&b.def.regex)?;
    Ok(re)
}

/// Constructor for the fully compiled "sparse" DFA. A sparse DFA is different
/// from a dense DFA in that following a transition on a state requires a
/// non-constant time lookup to find the transition matching the current byte.
/// In exchange, a sparse DFA uses less heap memory.
pub(super) fn regex_automata_dfa_sparse(
    b: &Benchmark,
) -> anyhow::Result<
    automata::dfa::regex::Regex<automata::dfa::sparse::DFA<Vec<u8>>>,
> {
    use automata::dfa::regex::Regex;

    let re = Regex::builder()
        // Disabling UTF-8 here just means that iterators built by this regex
        // may report matches that split a UTF-8 encoding of a codepoint.
        .configure(Regex::config().utf8(false))
        .syntax(automata_syntax_config(b))
        .build_sparse(&b.def.regex)?;
    Ok(re)
}

/// Constructor for the hybrid NFA/DFA or "lazy DFA" regex engine. This builds
/// the underlying DFA at search time, but only up to a certain memory budget.
///
/// A lazy DFA, like fully compiled DFAs, cannot handle Unicode word
/// boundaries.
pub(super) fn regex_automata_hybrid(
    b: &Benchmark,
) -> anyhow::Result<automata::hybrid::regex::Regex> {
    use automata::hybrid::{dfa::DFA, regex::Regex};

    let re = Regex::builder()
        // Disabling UTF-8 here just means that iterators built by this regex
        // may report matches that split a UTF-8 encoding of a codepoint.
        .configure(Regex::config().utf8(false))
        // This makes it so the cache built by this regex will be at least bit
        // enough to make progress, no matter how big it needs to be. This is
        // useful in benchmarking to avoid cases where construction of hybrid
        // regexes fail because the default cache capacity is too small. We
        // could instead just set an obscenely large cache capacity, but it
        // is actually useful to both see how the default performs and what
        // happens when the cache is just barely big enough. (When barely big
        // enough, it's likely to get cleared very frequently and this will
        // overall reduce search speed.)
        .dfa(DFA::config().skip_cache_capacity_check(true))
        .syntax(automata_syntax_config(b))
        .build(&b.def.regex)?;
    Ok(re)
}

/// Constructor for the PikeVM, which can handle anything including Unicode
/// word boundaries and resolving capturing groups, but can be quite slow.
pub(super) fn regex_automata_pikevm(
    b: &Benchmark,
) -> anyhow::Result<automata::nfa::thompson::pikevm::PikeVM> {
    use automata::nfa::thompson::pikevm::PikeVM;

    let re = PikeVM::builder()
        // Disabling UTF-8 here just means that iterators built by this regex
        // may report matches that split a UTF-8 encoding of a codepoint.
        .configure(PikeVM::config().utf8(false))
        .syntax(automata_syntax_config(b))
        .build(&b.def.regex)?;
    Ok(re)
}

/// Constructor for the bounded backtracker. Like the PikeVM, it can handle
/// Unicode word boundaries and resolving capturing groups, but only works on
/// smaller inputs/regexes. The small size is required because it keeps track
/// of which byte/NFA-state pairs it has visited in order to avoid re-visiting
/// them. This avoids exponential worst case behavior.
///
/// The backtracker tends to be a bit quicker than the PikeVM.
pub(super) fn regex_automata_backtrack(
    b: &Benchmark,
) -> anyhow::Result<automata::nfa::thompson::backtrack::BoundedBacktracker> {
    use automata::nfa::thompson::backtrack::BoundedBacktracker;

    let re = BoundedBacktracker::builder()
        // Disabling UTF-8 here just means that iterators built by this regex
        // may report matches that split a UTF-8 encoding of a codepoint.
        .configure(BoundedBacktracker::config().utf8(false))
        .syntax(automata_syntax_config(b))
        .build(&b.def.regex)?;
    Ok(re)
}

/// A simple literal searcher. This obviously doesn't handle regexes, but it
/// gives us a nice baseline to compare regex searches to for the case of
/// simple literals.
pub(super) fn memchr_memmem(
    b: &Benchmark,
) -> anyhow::Result<memchr::memmem::Finder<'static>> {
    anyhow::ensure!(
        !b.def.case_insensitive,
        "memmem engine is incompatible with 'case-insensitive = true'"
    );
    Ok(memchr::memmem::Finder::new(&b.def.regex).into_owned())
}

/// The RE2 regex engine from Google.
///
/// See: https://github.com/google/re2
#[cfg(feature = "extre-re2")]
pub(super) fn re2_api(
    b: &Benchmark,
) -> anyhow::Result<crate::ffi::re2::Regex> {
    use crate::ffi::re2::Regex;

    let re = Regex::new(&b.def.regex, re2_options(b))?;
    Ok(re)
}

/// For regex-automata based regex engines, this builds a syntax configuration
/// from a benchmark definition.
pub(super) fn automata_syntax_config(b: &Benchmark) -> automata::SyntaxConfig {
    automata::SyntaxConfig::new()
        // Disabling UTF-8 just makes it possible to build regexes that won't
        // necessarily match UTF-8. Whether Unicode is actually usable or not
        // depends on the 'unicode' option below.
        .utf8(false)
        .unicode(b.def.unicode)
        .case_insensitive(b.def.case_insensitive)
}

/// For RE2 based regex engines, this creates an "options" value from the
/// given benchmark.
#[cfg(feature = "extre-re2")]
pub(super) fn re2_options(b: &Benchmark) -> crate::ffi::re2::Options {
    crate::ffi::re2::Options {
        utf8: b.def.unicode,
        case_sensitive: !b.def.case_insensitive,
    }
}
