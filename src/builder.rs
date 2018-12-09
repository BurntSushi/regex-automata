use regex_syntax::ParserBuilder;

use dfa::DFA;
use error::{Error, Result};
use nfa::{NFA, NFABuilder};

/// A builder for constructing a deterministic finite automaton from regular
/// expressions.
///
/// This builder permits configuring several aspects of the construction
/// process such as case insensitivity, Unicode support and various options
/// that impact the size of the generated DFA. In some cases, options (like
/// performing DFA minimization) can come with a substantial additional cost.
#[derive(Clone, Debug)]
pub struct DFABuilder {
    parser: ParserBuilder,
    nfa: NFABuilder,
    minimize: bool,
    premultiply: bool,
    byte_classes: bool,
}

impl DFABuilder {
    /// Create a new DFA builder with the default configuration.
    pub fn new() -> DFABuilder {
        DFABuilder {
            parser: ParserBuilder::new(),
            nfa: NFABuilder::new(),
            minimize: false,
            premultiply: true,
            byte_classes: true,
        }
    }

    /// Build a DFA from the given pattern.
    ///
    /// If there was a problem parsing or compiling the pattern, then an error
    /// is returned.
    pub fn build(&self, pattern: &str) -> Result<DFA> {
        let nfa = self.build_nfa(pattern)?;
        let mut dfa =
            if self.byte_classes {
                DFA::from_nfa_with_byte_classes(&nfa)
            } else {
                DFA::from_nfa(&nfa)
            };
        if self.minimize {
            dfa.minimize();
        }
        if self.premultiply {
            dfa.premultiply();
        }
        Ok(dfa)
    }

    /// Builds an NFA from the given pattern.
    pub(crate) fn build_nfa(&self, pattern: &str) -> Result<NFA> {
        let hir = self.parser.build().parse(pattern).map_err(Error::syntax)?;
        Ok(self.nfa.build(&hir)?)
    }

    /// Set whether matching must be anchored at the beginning of the input.
    ///
    /// When enabled, a match must begin at the start of the input. When
    /// disabled, the DFA will act as if the pattern started with a `.*?`,
    /// which enables a match to appear anywhere.
    ///
    /// By default this is disabled.
    pub fn anchored(&mut self, yes: bool) -> &mut DFABuilder {
        self.nfa.anchored(yes);
        self
    }

    /// Enable or disable the case insensitive flag by default.
    ///
    /// By default this is disabled. It may alternatively be selectively
    /// enabled in the regular expression itself via the `i` flag.
    pub fn case_insensitive(&mut self, yes: bool) -> &mut DFABuilder {
        self.parser.case_insensitive(yes);
        self
    }

    /// Enable verbose mode in the regular expression.
    ///
    /// When enabled, verbose mode permits insigificant whitespace in many
    /// places in the regular expression, as well as comments. Comments are
    /// started using `#` and continue until the end of the line.
    ///
    /// By default, this is disabled. It may be selectively enabled in the
    /// regular expression by using the `x` flag regardless of this setting.
    pub fn ignore_whitespace(&mut self, yes: bool) -> &mut DFABuilder {
        self.parser.ignore_whitespace(yes);
        self
    }

    /// Enable or disable the "dot matches any character" flag by default.
    ///
    /// By default this is disabled. It may alternatively be selectively
    /// enabled in the regular expression itself via the `s` flag.
    pub fn dot_matches_new_line(&mut self, yes: bool) -> &mut DFABuilder {
        self.parser.dot_matches_new_line(yes);
        self
    }

    /// Enable or disable the "swap greed" flag by default.
    ///
    /// By default this is disabled. It may alternatively be selectively
    /// enabled in the regular expression itself via the `U` flag.
    pub fn swap_greed(&mut self, yes: bool) -> &mut DFABuilder {
        self.parser.swap_greed(yes);
        self
    }

    /// Enable or disable the Unicode flag (`u`) by default.
    ///
    /// By default this is **enabled**. It may alternatively be selectively
    /// disabled in the regular expression itself via the `u` flag.
    ///
    /// Note that unless `allow_invalid_utf8` is enabled (it's disabled by
    /// default), a regular expression will fail to parse if Unicode mode is
    /// disabled and a sub-expression could possibly match invalid UTF-8.
    pub fn unicode(&mut self, yes: bool) -> &mut DFABuilder {
        self.parser.unicode(yes);
        self
    }

    /// When enabled, the builder will permit the construction of a regular
    /// expression that may match invalid UTF-8.
    ///
    /// When disabled (the default), the parser is guaranteed to produce
    /// an expression that will only ever match valid UTF-8 (otherwise, the
    /// builder will return an error).
    ///
    /// Perhaps surprisingly, when invalid UTF-8 isn't allowed, a negated ASCII
    /// word boundary (uttered as `(?-u:\B)` in the concrete syntax) will cause
    /// the parser to return an error. Namely, a negated ASCII word boundary
    /// can result in matching positions that aren't valid UTF-8 boundaries.
    pub fn allow_invalid_utf8(&mut self, yes: bool) -> &mut DFABuilder {
        self.parser.allow_invalid_utf8(yes);
        self.nfa.allow_invalid_utf8(yes);
        self
    }

    /// Set the nesting limit used for the regular expression parser.
    ///
    /// The nesting limit controls how deep the abstract syntax tree is allowed
    /// to be. If the AST exceeds the given limit (e.g., with too many nested
    /// groups), then an error is returned by the parser.
    ///
    /// The purpose of this limit is to act as a heuristic to prevent stack
    /// overflow when building a finite automaton from a regular expression's
    /// abstract syntax tree. In particular, construction currently uses
    /// recursion. In the future, the implementation may stop using recursion
    /// and this option will no longer be necessary.
    ///
    /// This limit is not checked until the entire AST is parsed. Therefore,
    /// if callers want to put a limit on the amount of heap space used, then
    /// they should impose a limit on the length, in bytes, of the concrete
    /// pattern string. In particular, this is viable since the parser will
    /// limit itself to heap space proportional to the lenth of the pattern
    /// string.
    ///
    /// Note that a nest limit of `0` will return a nest limit error for most
    /// patterns but not all. For example, a nest limit of `0` permits `a` but
    /// not `ab`, since `ab` requires a concatenation AST item, which results
    /// in a nest depth of `1`. In general, a nest limit is not something that
    /// manifests in an obvious way in the concrete syntax, therefore, it
    /// should not be used in a granular way.
    pub fn nest_limit(&mut self, limit: u32) -> &mut DFABuilder {
        self.parser.nest_limit(limit);
        self
    }

    /// Minimize the DFA.
    ///
    /// When enabled, the DFA built will be minimized such that it is as small
    /// as possible. Enabling this option is the same as building a DFA, and
    /// then calling `minimize` on it.
    ///
    /// Whether one enables minimization or not depends on the types of costs
    /// you're willing to pay and how much you care about its benefits. In
    /// particular, minimization has worst case `O(n*k*logn)` time and `O(k*n)`
    /// space, where `n` is the number of DFA states and `k` is the alphabet
    /// size. In practice, minimization can be quite costly in terms of both
    /// space and time, so it should only be done if you're willing to wait
    /// longer to produce a DFA. In general, you might want a minimal DFA in
    /// the following circumstances:
    ///
    /// 1. You would like to optimize for the size of the automaton. This can
    ///    manifest in one of two ways. Firstly, if you're converting the
    ///    DFA into Rust code (or a table embedded in the code), then a minimal
    ///    DFA will translate into a corresponding reduction in code  size, and
    ///    thus, also the final compiled binary size. Secondly, if you are
    ///    building many DFAs and putting them on the heap, you'll be able to
    ///    fit more if they are smaller. Note though that building a minimal
    ///    DFA itself requires additional space; you only realize the space
    ///    savings once the minimal DFA is constructed (at which point, the
    ///    space used for minimization is freed).
    /// 2. You've observed that a smaller DFA results in faster match
    ///    performance. Naively, this isn't guaranteed since there is no
    ///    inherent difference between matching with a bigger-than-minimal
    ///    DFA and a minimal DFA. However, a smaller DFA may make use of your
    ///    CPU's cache more efficiently.
    /// 3. You are trying to establish an equivalence between regular
    ///    languages. The standard method for this is to build a minimal DFA
    ///    for each language and then compare them. If the DFAs are equivalent
    ///    (up to state renaming), then the languages are equivalent.
    ///
    /// This option is disabled by default.
    pub fn minimize(&mut self, yes: bool) -> &mut DFABuilder {
        self.minimize = yes;
        self
    }

    /// Premultiply state identifiers in the DFA's transition table.
    ///
    /// When enabled, state identifiers are premultiplied to point to their
    /// corresponding row in the DFA's transition table. That is, given the
    /// `i`th state, its corresponding premultiplied identifier is `i * k`
    /// where `k` is the alphabet size of the DFA. (The alphabet size is at
    /// most 256, but is in practice smaller if byte classes is enabled.)
    ///
    /// When state identifiers are not premultiplied, then the identifier of
    /// the `i`th state is `i`.
    ///
    /// The advantage of premultiplying state identifiers is that is saves
    /// a multiplication instruction per byte when searching with the DFA.
    /// This has been observed to lead to a 20% performance benefit in
    /// micro-benchmarks.
    ///
    /// The primary disadvantage of premultiplying state identifiers is
    /// that they require a larger integer size to represent. For example,
    /// if your DFA has 200 states, then its premultiplied form requires
    /// 16 bits to represent every possible state identifier, where as its
    /// non-premultiplied form only requires 8 bits.
    ///
    /// This option is enabled by default.
    pub fn premultiply(&mut self, yes: bool) -> &mut DFABuilder {
        self.premultiply = yes;
        self
    }

    /// Shrink the size of the DFA's alphabet by mapping bytes to their
    /// equivalence classes.
    ///
    /// When enabled, each DFA will use a map from all possible bytes to their
    /// corresponding equivalence class. Each equivalence class represents a
    /// set of bytes that does not discriminate between a match and a non-match
    /// in the DFA. For example, the pattern `[ab]+` has at least two
    /// equivalence classes: a set containing `a` and `b` and a set containing
    /// every byte except for `a` and `b`. `a` and `b` are in the same
    /// equivalence classes because they never discriminate between a match
    /// and a non-match.
    ///
    /// The advantage of this map is that the size of the transition table can
    /// be reduced drastically from `#states * 256 * sizeof(id)` to
    /// `#states * k * sizeof(id)` where `k` is the number of equivalence
    /// classes.
    ///
    /// The disadvantage of this map is that every byte searched must be
    /// passed through this map before it can be used to determine the next
    /// transition. This has a small match time performance cost.
    ///
    /// This option is enabled by default.
    pub fn byte_classes(&mut self, yes: bool) -> &mut DFABuilder {
        self.byte_classes = yes;
        self
    }
}

impl Default for DFABuilder {
    fn default() -> DFABuilder {
        DFABuilder::new()
    }
}
