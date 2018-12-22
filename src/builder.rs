use regex_syntax::ParserBuilder;
use regex_syntax::hir::{self, Hir, HirKind};

use determinize::Determinizer;
use dfa::DFA;
use error::{Error, Result};
use regex::Regex;
use nfa::{NFA, NFABuilder};
use state_id::StateID;

/// A builder for a regex based on deterministic finite automatons.
///
/// This builder permits configuring several aspects of the construction
/// process such as case insensitivity, Unicode support and various options
/// that impact the size of the underlying DFAs. In some cases, options (like
/// performing DFA minimization) can come with a substantial additional cost.
///
/// This build generally constructs two DFAs, where one is responsible for
/// finding the end of a match and the other is responsible for finding the
/// start of a match. If you only need to detect whether something matched,
/// or only the end of a match, then you should use a
/// [`DFABuilder`](struct.DFABuilder.html)
/// to construct a single DFA, which is cheaper than building two DFAs.
#[derive(Clone, Debug)]
pub struct RegexBuilder {
    dfa: DFABuilder,
}

impl RegexBuilder {
    /// Create a new regex builder with the default configuration.
    pub fn new() -> RegexBuilder {
        RegexBuilder {
            dfa: DFABuilder::new(),
        }
    }

    /// Build a regex from the given pattern.
    ///
    /// If there was a problem parsing or compiling the pattern, then an error
    /// is returned.
    pub fn build(&self, pattern: &str) -> Result<Regex<'static>> {
        self.build_with_size::<usize>(pattern)
    }

    /// Build a regex from the given pattern using a specific representation
    /// for the underlying DFA state IDs.
    ///
    /// If there was a problem parsing or compiling the pattern, then an error
    /// is returned.
    ///
    /// The representation of state IDs is determined by the `S` type
    /// parameter. In general, `S` is usually one of `u8`, `u16`, `u32`, `u64`
    /// or `usize`, where `usize` is the default used for `build`. The purpose
    /// of specifying a representation for state IDs is to reduce the memory
    /// footprint of the underlying DFAs.
    ///
    /// When using this routine, the chosen state ID representation will be
    /// used throughout determinization and minimization, if minimization was
    /// requested. Even if the minimized DFAs can fit into the chosen state ID
    /// representation but the initial determinized DFA cannot, then this will
    /// still return an error. To get a minimized DFA with a smaller state ID
    /// representation, first build it with a bigger state ID representation,
    /// and then shrink the sizes of the DFAs using one of its conversion
    /// routines, such as [`DFA::to_u16`](struct.DFA.html#method.to_u16).
    /// Finally, reconstitute the regex via
    /// [`Regex::from_dfa`](struct.Regex.html#method.from_dfa).
    pub fn build_with_size<S: StateID>(
        &self,
        pattern: &str,
    ) -> Result<Regex<'static, S>> {
        let forward = self.dfa.build_with_size(pattern)?;
        let reverse = self.dfa
            .clone()
            .anchored(true)
            .reverse(true)
            .longest_match(true)
            .build_with_size(pattern)?;
        Ok(Regex::from_dfas(forward, reverse))
    }

    /// Set whether matching must be anchored at the beginning of the input.
    ///
    /// When enabled, a match must begin at the start of the input. When
    /// disabled, the regex will act as if the pattern started with a `.*?`,
    /// which enables a match to appear anywhere.
    ///
    /// By default this is disabled.
    pub fn anchored(&mut self, yes: bool) -> &mut RegexBuilder {
        self.dfa.anchored(yes);
        self
    }

    /// Enable or disable the case insensitive flag by default.
    ///
    /// By default this is disabled. It may alternatively be selectively
    /// enabled in the regular expression itself via the `i` flag.
    pub fn case_insensitive(&mut self, yes: bool) -> &mut RegexBuilder {
        self.dfa.case_insensitive(yes);
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
    pub fn ignore_whitespace(&mut self, yes: bool) -> &mut RegexBuilder {
        self.dfa.ignore_whitespace(yes);
        self
    }

    /// Enable or disable the "dot matches any character" flag by default.
    ///
    /// By default this is disabled. It may alternatively be selectively
    /// enabled in the regular expression itself via the `s` flag.
    pub fn dot_matches_new_line(&mut self, yes: bool) -> &mut RegexBuilder {
        self.dfa.dot_matches_new_line(yes);
        self
    }

    /// Enable or disable the "swap greed" flag by default.
    ///
    /// By default this is disabled. It may alternatively be selectively
    /// enabled in the regular expression itself via the `U` flag.
    pub fn swap_greed(&mut self, yes: bool) -> &mut RegexBuilder {
        self.dfa.swap_greed(yes);
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
    pub fn unicode(&mut self, yes: bool) -> &mut RegexBuilder {
        self.dfa.unicode(yes);
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
    pub fn allow_invalid_utf8(&mut self, yes: bool) -> &mut RegexBuilder {
        self.dfa.allow_invalid_utf8(yes);
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
    pub fn nest_limit(&mut self, limit: u32) -> &mut RegexBuilder {
        self.dfa.nest_limit(limit);
        self
    }

    /// Minimize the underlying DFAs.
    ///
    /// When enabled, the DFAs powering the resulting regex will be minimized
    /// such that it is as small as possible.
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
    pub fn minimize(&mut self, yes: bool) -> &mut RegexBuilder {
        self.dfa.minimize(yes);
        self
    }

    /// Premultiply state identifiers in the underlying DFA transition tables.
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
    pub fn premultiply(&mut self, yes: bool) -> &mut RegexBuilder {
        self.dfa.premultiply(yes);
        self
    }

    /// Shrink the size of the underlying DFA alphabet by mapping bytes to
    /// their equivalence classes.
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
    pub fn byte_classes(&mut self, yes: bool) -> &mut RegexBuilder {
        self.dfa.byte_classes(yes);
        self
    }
}

impl Default for RegexBuilder {
    fn default() -> RegexBuilder {
        RegexBuilder::new()
    }
}

/// A builder for constructing a deterministic finite automaton from regular
/// expressions.
///
/// This builder permits configuring several aspects of the construction
/// process such as case insensitivity, Unicode support and various options
/// that impact the size of the generated DFA. In some cases, options (like
/// performing DFA minimization) can come with a substantial additional cost.
///
/// This builder always constructs a *single* DFA. As such, this builder can
/// only be used to construct regexes that either detect the presence of a
/// match or find the end location of a match. A single DFA cannot produce both
/// the start and end of a match. For that information, use a
/// [`Regex`](struct.Regex.html), which can be similarly configured using
/// [`RegexBuilder`](struct.RegexBuilder.html).
#[derive(Clone, Debug)]
pub struct DFABuilder {
    parser: ParserBuilder,
    nfa: NFABuilder,
    minimize: bool,
    premultiply: bool,
    byte_classes: bool,
    reverse: bool,
    longest_match: bool,
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
            reverse: false,
            longest_match: false,
        }
    }

    /// Build a DFA from the given pattern.
    ///
    /// If there was a problem parsing or compiling the pattern, then an error
    /// is returned.
    pub fn build(&self, pattern: &str) -> Result<DFA> {
        self.build_with_size::<usize>(pattern)
    }

    /// Build a DFA from the given pattern using a specific representation for
    /// the DFA's state IDs.
    ///
    /// If there was a problem parsing or compiling the pattern, then an error
    /// is returned.
    ///
    /// The representation of state IDs is determined by the `S` type
    /// parameter. In general, `S` is usually one of `u8`, `u16`, `u32`, `u64`
    /// or `usize`, where `usize` is the default used for `build`. The purpose
    /// of specifying a representation for state IDs is to reduce the memory
    /// footprint of a DFA.
    ///
    /// When using this routine, the chosen state ID representation will be
    /// used throughout determinization and minimization, if minimization was
    /// requested. Even if the minimized DFA can fit into the chosen state ID
    /// representation but the initial determinized DFA cannot, then this will
    /// still return an error. To get a minimized DFA with a smaller state ID
    /// representation, first build it with a bigger state ID representation,
    /// and then shrink the size of the DFA using one of its conversion
    /// routines, such as [`DFA::to_u16`](struct.DFA.html#method.to_u16).
    pub fn build_with_size<S: StateID>(
        &self,
        pattern: &str,
    ) -> Result<DFA<S>> {
        let nfa = self.build_nfa(pattern)?;
        let mut dfa =
            if self.byte_classes {
                Determinizer::new(&nfa)
                    .with_byte_classes()
                    .longest_match(self.longest_match)
                    .build()
            } else {
                Determinizer::new(&nfa)
                    .longest_match(self.longest_match)
                    .build()
            }?;
        if self.minimize {
            dfa.minimize();
        }
        if self.premultiply {
            dfa.premultiply()?;
        }
        Ok(dfa)
    }

    /// Builds an NFA from the given pattern.
    pub(crate) fn build_nfa(&self, pattern: &str) -> Result<NFA> {
        let mut hir = self
            .parser
            .build()
            .parse(pattern)
            .map_err(Error::syntax)?;
        if self.reverse {
            hir = reverse_hir(hir);
        }
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
    /// as possible.
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

    /// Reverse the DFA.
    ///
    /// A DFA reversal is performed by reversing all of the concatenated
    /// sub-expressions in the original pattern, recursively. The resulting
    /// DFA can now be used to match the pattern starting from the end of a
    /// string instead of the beginning of a string.
    ///
    /// Generally speaking, a reverse DFA is most useful for finding the start
    /// of a match, since a single forward DFA is only capable of finding the
    /// end of a match. This start of match handling is done for you
    /// automatically if you build a [`Regex`](struct.Regex.html).
    pub fn reverse(&mut self, yes: bool) -> &mut DFABuilder {
        self.reverse = yes;
        self.nfa.reverse(yes);
        self
    }

    /// Find the longest possible match.
    ///
    /// This is distinct from the default leftmost-first match semantics in
    /// that it treats all NFA states as having equivalent priority. In other
    /// words, the longest possible match is always found and it is not
    /// possible to implement non-greedy match semantics when this is set. That
    /// is, `a+` and `a+?` are equivalent when this is enabled.
    ///
    /// In particular, a practical issue with this option at the moment is that
    /// it prevents unanchored searches from working correctly, since
    /// unanchored searches are implemented by prepending an non-greedy `.*?`
    /// to the beginning of the pattern. As stated above, non-greedy match
    /// semantics aren't supported. With that said, there is prior art in RE2
    /// that shows how this can be done. Instead of treating all NFA states as
    /// having equivalent priority, we instead group NFA states into sets, and
    /// treat members of each set as having equivalent priority, but having
    /// greater priority than all following members of different sets. Once
    /// this is implemented, we could consider exposing this option.
    ///
    /// Currently, this option is principally useful when building a reverse
    /// DFA for finding the start of a match. In this case, we must find the
    /// earliest possible matching position in order to satisfy leftmost-first
    /// semantics.
    fn longest_match(&mut self, yes: bool) -> &mut DFABuilder {
        self.longest_match = yes;
        self
    }
}

impl Default for DFABuilder {
    fn default() -> DFABuilder {
        DFABuilder::new()
    }
}

/// Reverse the given HIR expression.
fn reverse_hir(expr: Hir) -> Hir {
    match expr.into_kind() {
        HirKind::Empty => Hir::empty(),
        HirKind::Literal(hir::Literal::Byte(b)) => {
            Hir::literal(hir::Literal::Byte(b))
        }
        HirKind::Literal(hir::Literal::Unicode(c)) => {
            Hir::concat(
                c.encode_utf8(&mut [0; 4])
                .as_bytes()
                .iter()
                .cloned()
                .rev()
                .map(|b| {
                    if b <= 0x7F {
                        hir::Literal::Unicode(b as char)
                    } else {
                        hir::Literal::Byte(b)
                    }
                })
                .map(Hir::literal)
                .collect()
            )
        }
        HirKind::Class(cls) => Hir::class(cls),
        HirKind::Anchor(anchor) => Hir::anchor(anchor),
        HirKind::WordBoundary(anchor) => Hir::word_boundary(anchor),
        HirKind::Repetition(mut rep) => {
            rep.hir = Box::new(reverse_hir(*rep.hir));
            Hir::repetition(rep)
        }
        HirKind::Group(mut group) => {
            group.hir = Box::new(reverse_hir(*group.hir));
            Hir::group(group)
        }
        HirKind::Concat(exprs) => {
            let mut reversed = vec![];
            for e in exprs {
                reversed.push(reverse_hir(e));
            }
            reversed.reverse();
            Hir::concat(reversed)
        }
        HirKind::Alternation(exprs) => {
            let mut reversed = vec![];
            for e in exprs {
                reversed.push(reverse_hir(e));
            }
            Hir::alternation(reversed)
        }
    }
}
