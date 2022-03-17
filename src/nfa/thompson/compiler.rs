use core::{borrow::Borrow, cell::RefCell};

use alloc::sync::Arc;

use regex_syntax::{
    hir::{self, Anchor, Hir, WordBoundary},
    utf8::{Utf8Range, Utf8Sequences},
    ParserBuilder,
};

use crate::{
    nfa::thompson::{
        builder::Builder,
        error::Error,
        map::{Utf8BoundedMap, Utf8SuffixKey, Utf8SuffixMap},
        nfa::{Look, PatternIter, SparseTransitions, State, Transition, NFA},
        range_trie::RangeTrie,
    },
    util::id::{IteratorIDExt, PatternID, StateID},
};

/// The configuration used for a Thompson NFA compiler.
#[derive(Clone, Copy, Debug, Default)]
pub struct Config {
    reverse: Option<bool>,
    utf8: Option<bool>,
    nfa_size_limit: Option<Option<usize>>,
    shrink: Option<bool>,
    captures: Option<bool>,
    #[cfg(test)]
    unanchored_prefix: Option<bool>,
}

impl Config {
    /// Return a new default Thompson NFA compiler configuration.
    pub fn new() -> Config {
        Config::default()
    }

    /// Reverse the NFA.
    ///
    /// A NFA reversal is performed by reversing all of the concatenated
    /// sub-expressions in the original pattern, recursively. The resulting
    /// NFA can be used to match the pattern starting from the end of a string
    /// instead of the beginning of a string.
    ///
    /// Reversing the NFA is useful for building a reverse DFA, which is most
    /// useful for finding the start of a match after its ending position has
    /// been found.
    ///
    /// This is disabled by default.
    pub fn reverse(mut self, yes: bool) -> Config {
        self.reverse = Some(yes);
        self
    }

    /// Whether to enable UTF-8 mode or not.
    ///
    /// When UTF-8 mode is enabled (which is the default), unanchored searches
    /// will only match through valid UTF-8. If invalid UTF-8 is seen, then
    /// an unanchored search will stop at that point. This is equivalent to
    /// putting a `(?s:.)*?` at the start of the regex.
    ///
    /// When UTF-8 mode is disabled, then unanchored searches will match
    /// through any arbitrary byte. This is equivalent to putting a
    /// `(?s-u:.)*?` at the start of the regex.
    ///
    /// Generally speaking, UTF-8 mode should only be used when you know you
    /// are searching valid UTF-8, such as a Rust `&str`. If UTF-8 mode is used
    /// on input that is not valid UTF-8, then the regex is not likely to work
    /// as expected.
    ///
    /// This is enabled by default.
    pub fn utf8(mut self, yes: bool) -> Config {
        self.utf8 = Some(yes);
        self
    }

    /// Sets an approximate size limit on the total heap used by the NFA being
    /// compiled.
    ///
    /// This permits imposing constraints on the size of a compiled NFA. This
    /// may be useful in contexts where the regex pattern is untrusted and one
    /// wants to avoid using too much memory.
    ///
    /// This size limit does not apply to auxiliary heap used during
    /// compilation that is not part of the built NFA.
    ///
    /// Note that this size limit is applied during compilation in order for
    /// the limit to prevent too much heap from being used. However, the
    /// implementation may use an intermediate NFA representation that is
    /// otherwise slightly bigger than the final public form. Since the size
    /// limit may be applied to an intermediate representation, there is not
    /// necessarily a precise correspondence between the configured size limit
    /// and the heap usage of the final NFA.
    ///
    /// There is no size limit by default.
    ///
    /// # Example
    ///
    /// This example demonstrates how Unicode mode can greatly increase the
    /// size of the NFA.
    ///
    /// ```
    /// use regex_automata::nfa::thompson::NFA;
    ///
    /// // 300KB isn't enough!
    /// NFA::compiler()
    ///     .configure(NFA::config().nfa_size_limit(Some(300_000)))
    ///     .build(r"\w{20}")
    ///     .unwrap_err();
    ///
    /// // ... but 400KB probably is.
    /// let nfa = NFA::compiler()
    ///     .configure(NFA::config().nfa_size_limit(Some(400_000)))
    ///     .build(r"\w{20}")?;
    ///
    /// assert_eq!(nfa.pattern_len(), 1);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn nfa_size_limit(mut self, bytes: Option<usize>) -> Config {
        self.nfa_size_limit = Some(bytes);
        self
    }

    /// Apply best effort heuristics to shrink the NFA at the expense of more
    /// time/memory.
    ///
    /// This is enabled by default. Generally speaking, if one is using an NFA
    /// to compile a DFA, then the extra time used to shrink the NFA will be
    /// more than made up for during DFA construction (potentially by a lot).
    /// In other words, enabling this can substantially decrease the overall
    /// amount of time it takes to build a DFA.
    ///
    /// The only reason to disable this if you want to compile an NFA and start
    /// using it as quickly as possible without needing to build a DFA. e.g.,
    /// for an NFA simulation or for a lazy DFA.
    ///
    /// This is enabled by default.
    pub fn shrink(mut self, yes: bool) -> Config {
        self.shrink = Some(yes);
        self
    }

    /// Whether to include 'Capture' states in the NFA.
    ///
    /// This can only be enabled when compiling a forward NFA. This is
    /// always disabled---with no way to override it---when the `reverse`
    /// configuration is enabled.
    ///
    /// This is enabled by default.
    pub fn captures(mut self, yes: bool) -> Config {
        self.captures = Some(yes);
        self
    }

    /// Whether to compile an unanchored prefix into this NFA.
    ///
    /// This is enabled by default. It is made available for tests only to make
    /// it easier to unit test the output of the compiler.
    #[cfg(test)]
    fn unanchored_prefix(mut self, yes: bool) -> Config {
        self.unanchored_prefix = Some(yes);
        self
    }

    pub fn get_reverse(&self) -> bool {
        self.reverse.unwrap_or(false)
    }

    pub fn get_utf8(&self) -> bool {
        self.utf8.unwrap_or(true)
    }

    pub fn get_nfa_size_limit(&self) -> Option<usize> {
        self.nfa_size_limit.unwrap_or(None)
    }

    pub fn get_shrink(&self) -> bool {
        self.shrink.unwrap_or(true)
    }

    pub fn get_captures(&self) -> bool {
        !self.get_reverse() && self.captures.unwrap_or(true)
    }

    fn get_unanchored_prefix(&self) -> bool {
        #[cfg(test)]
        {
            self.unanchored_prefix.unwrap_or(true)
        }
        #[cfg(not(test))]
        {
            true
        }
    }

    pub(crate) fn overwrite(self, o: Config) -> Config {
        Config {
            reverse: o.reverse.or(self.reverse),
            utf8: o.utf8.or(self.utf8),
            nfa_size_limit: o.nfa_size_limit.or(self.nfa_size_limit),
            shrink: o.shrink.or(self.shrink),
            captures: o.captures.or(self.captures),
            #[cfg(test)]
            unanchored_prefix: o.unanchored_prefix.or(self.unanchored_prefix),
        }
    }
}

/// A builder for compiling an NFA.
#[derive(Clone, Debug)]
pub struct Compiler {
    parser: ParserBuilder,
    config: Config,
    /// The builder for actually constructing an NFA. This provides a
    /// convenient abstraction for writing a compiler.
    builder: RefCell<Builder>,
    /// State used for compiling character classes to UTF-8 byte automata.
    /// State is not retained between character class compilations. This just
    /// serves to amortize allocation to the extent possible.
    utf8_state: RefCell<Utf8State>,
    /// State used for arranging character classes in reverse into a trie.
    trie_state: RefCell<RangeTrie>,
    /// State used for caching common suffixes when compiling reverse UTF-8
    /// automata (for Unicode character classes).
    utf8_suffix: RefCell<Utf8SuffixMap>,
}

impl Compiler {
    /// Create a new NFA builder with its default configuration.
    pub fn new() -> Compiler {
        Compiler {
            parser: ParserBuilder::new(),
            config: Config::default(),
            builder: RefCell::new(Builder::new()),
            utf8_state: RefCell::new(Utf8State::new()),
            trie_state: RefCell::new(RangeTrie::new()),
            utf8_suffix: RefCell::new(Utf8SuffixMap::new(1000)),
        }
    }

    /// Compile the given regular expression into an NFA.
    ///
    /// If there was a problem parsing the regex, then that error is returned.
    ///
    /// Otherwise, if there was a problem building the NFA, then an error is
    /// returned. The only error that can occur is if the compiled regex would
    /// exceed the size limits configured on this builder.
    pub fn build(&self, pattern: &str) -> Result<NFA, Error> {
        self.build_many(&[pattern])
    }

    pub fn build_many<P: AsRef<str>>(
        &self,
        patterns: &[P],
    ) -> Result<NFA, Error> {
        let mut hirs = vec![];
        for p in patterns {
            hirs.push(
                self.parser
                    .build()
                    .parse(p.as_ref())
                    .map_err(Error::syntax)?,
            );
            log!(log::trace!("parsed: {:?}", p.as_ref()));
        }
        self.build_many_from_hir(&hirs)
    }

    /// Compile the given high level intermediate representation of a regular
    /// expression into an NFA.
    ///
    /// If there was a problem building the NFA, then an error is returned. The
    /// only error that can occur is if the compiled regex would exceed the
    /// size limits configured on this builder.
    pub fn build_from_hir(&self, expr: &Hir) -> Result<NFA, Error> {
        self.build_many_from_hir(&[expr])
    }

    /// Compile the given high level intermediate representation of a regular
    /// expression into an NFA.
    ///
    /// If there was a problem building the NFA, then an error is returned. The
    /// only error that can occur is if the compiled regex would exceed the
    /// size limits configured on this builder.
    pub fn build_many_from_hir<H: Borrow<Hir>>(
        &self,
        exprs: &[H],
    ) -> Result<NFA, Error> {
        self.compile(exprs)
    }

    /// Apply the given NFA configuration options to this builder.
    pub fn configure(&mut self, config: Config) -> &mut Compiler {
        self.config = self.config.overwrite(config);
        self
    }

    /// Set the syntax configuration for this builder using
    /// [`SyntaxConfig`](crate::SyntaxConfig).
    ///
    /// This permits setting things like case insensitivity, Unicode and multi
    /// line mode.
    ///
    /// This syntax configuration only applies when an NFA is built directly
    /// from a pattern string. If an NFA is built from an HIR, then all syntax
    /// settings are ignored.
    pub fn syntax(
        &mut self,
        config: crate::util::syntax::SyntaxConfig,
    ) -> &mut Compiler {
        config.apply(&mut self.parser);
        self
    }
}

impl Compiler {
    fn compile<H: Borrow<Hir>>(&self, exprs: &[H]) -> Result<NFA, Error> {
        if exprs.is_empty() {
            return Ok(NFA::never_match());
        }
        if exprs.len() > PatternID::LIMIT {
            return Err(Error::too_many_patterns(exprs.len()));
        }

        self.builder.borrow_mut().clear();
        self.builder
            .borrow_mut()
            .set_size_limit(self.config.get_nfa_size_limit());

        // We always add an unanchored prefix unless we were specifically told
        // not to (for tests only), or if we know that the regex is anchored
        // for all matches. When an unanchored prefix is not added, then the
        // NFA's anchored and unanchored start states are equivalent.
        let all_anchored =
            exprs.iter().all(|e| e.borrow().is_anchored_start());
        let anchored = !self.config.get_unanchored_prefix() || all_anchored;
        let unanchored_prefix = if anchored {
            self.c_empty()?
        } else {
            if self.config.get_utf8() {
                self.c_unanchored_prefix_valid_utf8()?
            } else {
                self.c_unanchored_prefix_invalid_utf8()?
            }
        };

        let compiled =
            self.c_alt(exprs.iter().with_pattern_ids().map(|(pid, e)| {
                let _ = self.start_pattern()?;
                let group_kind = hir::GroupKind::CaptureIndex(0);
                let one = self.c_group(&group_kind, e.borrow())?;
                let match_state_id = self.add_match()?;
                self.patch(one.end, match_state_id)?;
                let _ = self.finish_pattern(one.start)?;
                Ok(ThompsonRef { start: one.start, end: match_state_id })
            }))?;
        self.patch(unanchored_prefix.end, compiled.start)?;
        let nfa = self
            .builder
            .borrow_mut()
            .build(compiled.start, unanchored_prefix.start)?;

        trace!(
            "HIR-to-NFA compilation complete (reverse? {:?})",
            self.config.get_reverse(),
        );
        Ok(nfa)
    }

    fn c(&self, expr: &Hir) -> Result<ThompsonRef, Error> {
        use regex_syntax::hir::{Class, HirKind::*, Literal};

        match *expr.kind() {
            Empty => self.c_empty(),
            Literal(Literal::Unicode(ch)) => self.c_char(ch),
            Literal(Literal::Byte(b)) => self.c_range(b, b),
            Class(Class::Bytes(ref c)) => self.c_byte_class(c),
            Class(Class::Unicode(ref c)) => self.c_unicode_class(c),
            Anchor(ref anchor) => self.c_anchor(anchor),
            WordBoundary(ref wb) => self.c_word_boundary(wb),
            Repetition(ref rep) => self.c_repetition(rep),
            Group(ref group) => self.c_group(&group.kind, &group.hir),
            Concat(ref es) => self.c_concat(es.iter().map(|e| self.c(e))),
            Alternation(ref es) => self.c_alt(es.iter().map(|e| self.c(e))),
        }
    }

    fn c_concat<I>(&self, mut it: I) -> Result<ThompsonRef, Error>
    where
        I: DoubleEndedIterator<Item = Result<ThompsonRef, Error>>,
    {
        let first = if self.is_reverse() { it.next_back() } else { it.next() };
        let ThompsonRef { start, mut end } = match first {
            Some(result) => result?,
            None => return self.c_empty(),
        };
        loop {
            let next =
                if self.is_reverse() { it.next_back() } else { it.next() };
            let compiled = match next {
                Some(result) => result?,
                None => break,
            };
            self.patch(end, compiled.start)?;
            end = compiled.end;
        }
        Ok(ThompsonRef { start, end })
    }

    fn c_alt<I>(&self, mut it: I) -> Result<ThompsonRef, Error>
    where
        I: Iterator<Item = Result<ThompsonRef, Error>>,
    {
        let first = it.next().expect("alternations must be non-empty")?;
        let second = match it.next() {
            None => return Ok(first),
            Some(result) => result?,
        };

        let union = self.add_union()?;
        let end = self.add_empty()?;
        self.patch(union, first.start)?;
        self.patch(first.end, end)?;
        self.patch(union, second.start)?;
        self.patch(second.end, end)?;
        for result in it {
            let compiled = result?;
            self.patch(union, compiled.start)?;
            self.patch(compiled.end, end)?;
        }
        Ok(ThompsonRef { start: union, end })
    }

    fn c_group(
        &self,
        kind: &hir::GroupKind,
        expr: &Hir,
    ) -> Result<ThompsonRef, Error> {
        if !self.config.get_captures() {
            return self.c(expr);
        }
        let (capi, name) = match *kind {
            hir::GroupKind::NonCapturing => return self.c(expr),
            hir::GroupKind::CaptureIndex(index) => (index, None),
            hir::GroupKind::CaptureName { ref name, index } => {
                (index, Some(&**name))
            }
        };

        let start = self.add_capture_start(capi, name)?;
        let inner = self.c(expr)?;
        let end = self.add_capture_end(capi)?;

        self.patch(start, inner.start)?;
        self.patch(inner.end, end)?;
        Ok(ThompsonRef { start, end })
    }

    fn c_repetition(
        &self,
        rep: &hir::Repetition,
    ) -> Result<ThompsonRef, Error> {
        match rep.kind {
            hir::RepetitionKind::ZeroOrOne => {
                self.c_zero_or_one(&rep.hir, rep.greedy)
            }
            hir::RepetitionKind::ZeroOrMore => {
                self.c_at_least(&rep.hir, rep.greedy, 0)
            }
            hir::RepetitionKind::OneOrMore => {
                self.c_at_least(&rep.hir, rep.greedy, 1)
            }
            hir::RepetitionKind::Range(ref rng) => match *rng {
                hir::RepetitionRange::Exactly(count) => {
                    self.c_exactly(&rep.hir, count)
                }
                hir::RepetitionRange::AtLeast(m) => {
                    self.c_at_least(&rep.hir, rep.greedy, m)
                }
                hir::RepetitionRange::Bounded(min, max) => {
                    self.c_bounded(&rep.hir, rep.greedy, min, max)
                }
            },
        }
    }

    fn c_bounded(
        &self,
        expr: &Hir,
        greedy: bool,
        min: u32,
        max: u32,
    ) -> Result<ThompsonRef, Error> {
        let prefix = self.c_exactly(expr, min)?;
        if min == max {
            return Ok(prefix);
        }

        // It is tempting here to compile the rest here as a concatenation
        // of zero-or-one matches. i.e., for `a{2,5}`, compile it as if it
        // were `aaa?a?a?`. The problem here is that it leads to this program:
        //
        //     >000000: 61 => 01
        //      000001: 61 => 02
        //      000002: union(03, 04)
        //      000003: 61 => 04
        //      000004: union(05, 06)
        //      000005: 61 => 06
        //      000006: union(07, 08)
        //      000007: 61 => 08
        //      000008: MATCH
        //
        // And effectively, once you hit state 2, the epsilon closure will
        // include states 3, 5, 6, 7 and 8, which is quite a bit. It is better
        // to instead compile it like so:
        //
        //     >000000: 61 => 01
        //      000001: 61 => 02
        //      000002: union(03, 08)
        //      000003: 61 => 04
        //      000004: union(05, 08)
        //      000005: 61 => 06
        //      000006: union(07, 08)
        //      000007: 61 => 08
        //      000008: MATCH
        //
        // So that the epsilon closure of state 2 is now just 3 and 8.
        let empty = self.add_empty()?;
        let mut prev_end = prefix.end;
        for _ in min..max {
            let union = if greedy {
                self.add_union()
            } else {
                self.add_union_reverse()
            }?;
            let compiled = self.c(expr)?;
            self.patch(prev_end, union)?;
            self.patch(union, compiled.start)?;
            self.patch(union, empty)?;
            prev_end = compiled.end;
        }
        self.patch(prev_end, empty)?;
        Ok(ThompsonRef { start: prefix.start, end: empty })
    }

    fn c_at_least(
        &self,
        expr: &Hir,
        greedy: bool,
        n: u32,
    ) -> Result<ThompsonRef, Error> {
        if n == 0 {
            // When the expression cannot match the empty string, then we
            // can get away with something much simpler: just one 'alt'
            // instruction that optionally repeats itself. But if the expr
            // can match the empty string... see below.
            if !expr.is_match_empty() {
                let union = if greedy {
                    self.add_union()
                } else {
                    self.add_union_reverse()
                }?;
                let compiled = self.c(expr)?;
                self.patch(union, compiled.start)?;
                self.patch(compiled.end, union)?;
                return Ok(ThompsonRef { start: union, end: union });
            }

            // What's going on here? Shouldn't x* be simpler than this? It
            // turns out that when implementing leftmost-first (Perl-like)
            // match semantics, x* results in an incorrect preference order
            // when computing the transitive closure of states if and only if
            // 'x' can match the empty string. So instead, we compile x* as
            // (x+)?, which preserves the correct preference order.
            //
            // See: https://github.com/rust-lang/regex/issues/779
            let compiled = self.c(expr)?;
            let plus = if greedy {
                self.add_union()
            } else {
                self.add_union_reverse()
            }?;
            self.patch(compiled.end, plus)?;
            self.patch(plus, compiled.start)?;

            let question = if greedy {
                self.add_union()
            } else {
                self.add_union_reverse()
            }?;
            let empty = self.add_empty()?;
            self.patch(question, compiled.start)?;
            self.patch(question, empty)?;
            self.patch(plus, empty)?;
            Ok(ThompsonRef { start: question, end: empty })
        } else if n == 1 {
            let compiled = self.c(expr)?;
            let union = if greedy {
                self.add_union()
            } else {
                self.add_union_reverse()
            }?;
            self.patch(compiled.end, union)?;
            self.patch(union, compiled.start)?;
            Ok(ThompsonRef { start: compiled.start, end: union })
        } else {
            let prefix = self.c_exactly(expr, n - 1)?;
            let last = self.c(expr)?;
            let union = if greedy {
                self.add_union()
            } else {
                self.add_union_reverse()
            }?;
            self.patch(prefix.end, last.start)?;
            self.patch(last.end, union)?;
            self.patch(union, last.start)?;
            Ok(ThompsonRef { start: prefix.start, end: union })
        }
    }

    fn c_zero_or_one(
        &self,
        expr: &Hir,
        greedy: bool,
    ) -> Result<ThompsonRef, Error> {
        let union =
            if greedy { self.add_union() } else { self.add_union_reverse() }?;
        let compiled = self.c(expr)?;
        let empty = self.add_empty()?;
        self.patch(union, compiled.start)?;
        self.patch(union, empty)?;
        self.patch(compiled.end, empty)?;
        Ok(ThompsonRef { start: union, end: empty })
    }

    fn c_exactly(&self, expr: &Hir, n: u32) -> Result<ThompsonRef, Error> {
        let it = (0..n).map(|_| self.c(expr));
        self.c_concat(it)
    }

    fn c_byte_class(
        &self,
        cls: &hir::ClassBytes,
    ) -> Result<ThompsonRef, Error> {
        let end = self.add_empty()?;
        let mut trans = Vec::with_capacity(cls.ranges().len());
        for r in cls.iter() {
            trans.push(Transition {
                start: r.start(),
                end: r.end(),
                next: end,
            });
        }
        Ok(ThompsonRef { start: self.add_sparse(trans)?, end })
    }

    fn c_unicode_class(
        &self,
        cls: &hir::ClassUnicode,
    ) -> Result<ThompsonRef, Error> {
        // If all we have are ASCII ranges wrapped in a Unicode package, then
        // there is zero reason to bring out the big guns. We can fit all ASCII
        // ranges within a single sparse state.
        if cls.is_all_ascii() {
            let end = self.add_empty()?;
            let mut trans = Vec::with_capacity(cls.ranges().len());
            for r in cls.iter() {
                assert!(r.start() <= '\x7F');
                assert!(r.end() <= '\x7F');
                trans.push(Transition {
                    start: r.start() as u8,
                    end: r.end() as u8,
                    next: end,
                });
            }
            Ok(ThompsonRef { start: self.add_sparse(trans)?, end })
        } else if self.is_reverse() {
            if !self.config.get_shrink() {
                // When we don't want to spend the extra time shrinking, we
                // compile the UTF-8 automaton in reverse using something like
                // the "naive" approach, but will attempt to re-use common
                // suffixes.
                self.c_unicode_class_reverse_with_suffix(cls)
            } else {
                // When we want to shrink our NFA for reverse UTF-8 automata,
                // we cannot feed UTF-8 sequences directly to the UTF-8
                // compiler, since the UTF-8 compiler requires all sequences
                // to be lexicographically sorted. Instead, we organize our
                // sequences into a range trie, which can then output our
                // sequences in the correct order. Unfortunately, building the
                // range trie is fairly expensive (but not nearly as expensive
                // as building a DFA). Hence the reason why the 'shrink' option
                // exists, so that this path can be toggled off. For example,
                // we might want to turn this off if we know we won't be
                // compiling a DFA.
                let mut trie = self.trie_state.borrow_mut();
                trie.clear();

                for rng in cls.iter() {
                    for mut seq in Utf8Sequences::new(rng.start(), rng.end()) {
                        seq.reverse();
                        trie.insert(seq.as_slice());
                    }
                }
                let mut builder = self.builder.borrow_mut();
                let mut utf8_state = self.utf8_state.borrow_mut();
                let mut utf8c =
                    Utf8Compiler::new(&mut *builder, &mut *utf8_state)?;
                trie.iter(|seq| {
                    utf8c.add(&seq)?;
                    Ok(())
                })?;
                utf8c.finish()
            }
        } else {
            // In the forward direction, we always shrink our UTF-8 automata
            // because we can stream it right into the UTF-8 compiler. There
            // is almost no downside (in either memory or time) to using this
            // approach.
            let mut builder = self.builder.borrow_mut();
            let mut utf8_state = self.utf8_state.borrow_mut();
            let mut utf8c =
                Utf8Compiler::new(&mut *builder, &mut *utf8_state)?;
            for rng in cls.iter() {
                for seq in Utf8Sequences::new(rng.start(), rng.end()) {
                    utf8c.add(seq.as_slice())?;
                }
            }
            utf8c.finish()
        }

        // For reference, the code below is the "naive" version of compiling a
        // UTF-8 automaton. It is deliciously simple (and works for both the
        // forward and reverse cases), but will unfortunately produce very
        // large NFAs. When compiling a forward automaton, the size difference
        // can sometimes be an order of magnitude. For example, the '\w' regex
        // will generate about ~3000 NFA states using the naive approach below,
        // but only 283 states when using the approach above. This is because
        // the approach above actually compiles a *minimal* (or near minimal,
        // because of the bounded hashmap for reusing equivalent states) UTF-8
        // automaton.
        //
        // The code below is kept as a reference point in order to make it
        // easier to understand the higher level goal here. Although, it will
        // almost certainly bit-rot, so keep that in mind.
        /*
        let it = cls
            .iter()
            .flat_map(|rng| Utf8Sequences::new(rng.start(), rng.end()))
            .map(|seq| {
                let it = seq
                    .as_slice()
                    .iter()
                    .map(|rng| self.c_range(rng.start, rng.end));
                self.c_concat(it)
            });
        self.c_alt(it)
        */
    }

    fn c_unicode_class_reverse_with_suffix(
        &self,
        cls: &hir::ClassUnicode,
    ) -> Result<ThompsonRef, Error> {
        // N.B. It would likely be better to cache common *prefixes* in the
        // reverse direction, but it's not quite clear how to do that. The
        // advantage of caching suffixes is that it does give us a win, and
        // has a very small additional overhead.
        let mut cache = self.utf8_suffix.borrow_mut();
        cache.clear();

        let union = self.add_union()?;
        let alt_end = self.add_empty()?;
        for urng in cls.iter() {
            for seq in Utf8Sequences::new(urng.start(), urng.end()) {
                let mut end = alt_end;
                for brng in seq.as_slice() {
                    let key = Utf8SuffixKey {
                        from: end,
                        start: brng.start,
                        end: brng.end,
                    };
                    let hash = cache.hash(&key);
                    if let Some(id) = cache.get(&key, hash) {
                        end = id;
                        continue;
                    }

                    let compiled = self.c_range(brng.start, brng.end)?;
                    self.patch(compiled.end, end)?;
                    end = compiled.start;
                    cache.set(key, hash, end);
                }
                self.patch(union, end)?;
            }
        }
        Ok(ThompsonRef { start: union, end: alt_end })
    }

    fn c_anchor(&self, anchor: &Anchor) -> Result<ThompsonRef, Error> {
        let look = match *anchor {
            Anchor::StartLine => Look::StartLine,
            Anchor::EndLine => Look::EndLine,
            Anchor::StartText => Look::StartText,
            Anchor::EndText => Look::EndText,
        };
        let id = self.add_look(look)?;
        Ok(ThompsonRef { start: id, end: id })
    }

    fn c_word_boundary(
        &self,
        wb: &WordBoundary,
    ) -> Result<ThompsonRef, Error> {
        let look = match *wb {
            WordBoundary::Unicode => Look::WordBoundaryUnicode,
            WordBoundary::UnicodeNegate => Look::WordBoundaryUnicodeNegate,
            WordBoundary::Ascii => Look::WordBoundaryAscii,
            WordBoundary::AsciiNegate => Look::WordBoundaryAsciiNegate,
        };
        let id = self.add_look(look)?;
        Ok(ThompsonRef { start: id, end: id })
    }

    fn c_char(&self, ch: char) -> Result<ThompsonRef, Error> {
        let mut buf = [0; 4];
        let it = ch
            .encode_utf8(&mut buf)
            .as_bytes()
            .iter()
            .map(|&b| self.c_range(b, b));
        self.c_concat(it)
    }

    fn c_range(&self, start: u8, end: u8) -> Result<ThompsonRef, Error> {
        let id = self.add_range(start, end)?;
        Ok(ThompsonRef { start: id, end: id })
    }

    fn c_empty(&self) -> Result<ThompsonRef, Error> {
        let id = self.add_empty()?;
        Ok(ThompsonRef { start: id, end: id })
    }

    fn c_unanchored_prefix_valid_utf8(&self) -> Result<ThompsonRef, Error> {
        self.c_at_least(&Hir::any(false), false, 0)
    }

    fn c_unanchored_prefix_invalid_utf8(&self) -> Result<ThompsonRef, Error> {
        self.c_at_least(&Hir::any(true), false, 0)
    }

    // The below helpers are meant to be simple wrappers around the
    // corresponding Builder methods. For the most part, they let us write
    // 'self.add_foo()' instead of 'self.builder.borrow_mut().add_foo()', where
    // the latter is a mouthful. Some of the methods do inject a little bit
    // of extra logic. e.g., Flipping look-around operators when compiling in
    // reverse mode.

    fn patch(&self, from: StateID, to: StateID) -> Result<(), Error> {
        self.builder.borrow_mut().patch(from, to)
    }

    fn start_pattern(&self) -> Result<PatternID, Error> {
        self.builder.borrow_mut().start_pattern()
    }

    fn finish_pattern(&self, start_id: StateID) -> Result<PatternID, Error> {
        self.builder.borrow_mut().finish_pattern(start_id)
    }

    fn add_empty(&self) -> Result<StateID, Error> {
        self.builder.borrow_mut().add_empty()
    }

    fn add_range(&self, start: u8, end: u8) -> Result<StateID, Error> {
        self.builder.borrow_mut().add_range(Transition {
            start,
            end,
            next: StateID::ZERO,
        })
    }

    fn add_sparse(&self, ranges: Vec<Transition>) -> Result<StateID, Error> {
        self.builder.borrow_mut().add_sparse(ranges)
    }

    fn add_look(&self, mut look: Look) -> Result<StateID, Error> {
        if self.is_reverse() {
            look = look.reversed();
        }
        self.builder.borrow_mut().add_look(StateID::ZERO, look)
    }

    fn add_union(&self) -> Result<StateID, Error> {
        self.builder.borrow_mut().add_union(vec![])
    }

    fn add_union_reverse(&self) -> Result<StateID, Error> {
        self.builder.borrow_mut().add_union_reverse(vec![])
    }

    fn add_capture_start(
        &self,
        capture_index: u32,
        name: Option<&str>,
    ) -> Result<StateID, Error> {
        let name = name.map(|n| Arc::from(n));
        self.builder.borrow_mut().add_capture_start(capture_index, name)
    }

    fn add_capture_end(&self, capture_index: u32) -> Result<StateID, Error> {
        self.builder.borrow_mut().add_capture_end(capture_index)
    }

    fn add_fail(&self) -> Result<StateID, Error> {
        self.builder.borrow_mut().add_fail()
    }

    fn add_match(&self) -> Result<StateID, Error> {
        self.builder.borrow_mut().add_match()
    }

    fn is_reverse(&self) -> bool {
        self.config.get_reverse()
    }
}

/// A value that represents the result of compiling a sub-expression of a
/// regex's HIR. Specifically, this represents a sub-graph of the NFA that
/// has an initial state at `start` and a final state at `end`.
#[derive(Clone, Copy, Debug)]
struct ThompsonRef {
    start: StateID,
    end: StateID,
}

#[derive(Debug)]
struct Utf8Compiler<'a> {
    builder: &'a mut Builder,
    state: &'a mut Utf8State,
    target: StateID,
}

#[derive(Clone, Debug)]
struct Utf8State {
    compiled: Utf8BoundedMap,
    uncompiled: Vec<Utf8Node>,
}

#[derive(Clone, Debug)]
struct Utf8Node {
    trans: Vec<Transition>,
    last: Option<Utf8LastTransition>,
}

#[derive(Clone, Debug)]
struct Utf8LastTransition {
    start: u8,
    end: u8,
}

impl Utf8State {
    fn new() -> Utf8State {
        Utf8State { compiled: Utf8BoundedMap::new(10_000), uncompiled: vec![] }
    }

    fn clear(&mut self) {
        self.compiled.clear();
        self.uncompiled.clear();
    }
}

impl<'a> Utf8Compiler<'a> {
    fn new(
        builder: &'a mut Builder,
        state: &'a mut Utf8State,
    ) -> Result<Utf8Compiler<'a>, Error> {
        let target = builder.add_empty()?;
        state.clear();
        let mut utf8c = Utf8Compiler { builder, state, target };
        utf8c.add_empty();
        Ok(utf8c)
    }

    fn finish(&mut self) -> Result<ThompsonRef, Error> {
        self.compile_from(0)?;
        let node = self.pop_root();
        let start = self.compile(node)?;
        Ok(ThompsonRef { start, end: self.target })
    }

    fn add(&mut self, ranges: &[Utf8Range]) -> Result<(), Error> {
        let prefix_len = ranges
            .iter()
            .zip(&self.state.uncompiled)
            .take_while(|&(range, node)| {
                node.last.as_ref().map_or(false, |t| {
                    (t.start, t.end) == (range.start, range.end)
                })
            })
            .count();
        assert!(prefix_len < ranges.len());
        self.compile_from(prefix_len)?;
        self.add_suffix(&ranges[prefix_len..]);
        Ok(())
    }

    fn compile_from(&mut self, from: usize) -> Result<(), Error> {
        let mut next = self.target;
        while from + 1 < self.state.uncompiled.len() {
            let node = self.pop_freeze(next);
            next = self.compile(node)?;
        }
        self.top_last_freeze(next);
        Ok(())
    }

    fn compile(&mut self, node: Vec<Transition>) -> Result<StateID, Error> {
        let hash = self.state.compiled.hash(&node);
        if let Some(id) = self.state.compiled.get(&node, hash) {
            return Ok(id);
        }
        let id = self.builder.add_sparse(node.clone())?;
        self.state.compiled.set(node, hash, id);
        Ok(id)
    }

    fn add_suffix(&mut self, ranges: &[Utf8Range]) {
        assert!(!ranges.is_empty());
        let last = self
            .state
            .uncompiled
            .len()
            .checked_sub(1)
            .expect("non-empty nodes");
        assert!(self.state.uncompiled[last].last.is_none());
        self.state.uncompiled[last].last = Some(Utf8LastTransition {
            start: ranges[0].start,
            end: ranges[0].end,
        });
        for r in &ranges[1..] {
            self.state.uncompiled.push(Utf8Node {
                trans: vec![],
                last: Some(Utf8LastTransition { start: r.start, end: r.end }),
            });
        }
    }

    fn add_empty(&mut self) {
        self.state.uncompiled.push(Utf8Node { trans: vec![], last: None });
    }

    fn pop_freeze(&mut self, next: StateID) -> Vec<Transition> {
        let mut uncompiled = self.state.uncompiled.pop().unwrap();
        uncompiled.set_last_transition(next);
        uncompiled.trans
    }

    fn pop_root(&mut self) -> Vec<Transition> {
        assert_eq!(self.state.uncompiled.len(), 1);
        assert!(self.state.uncompiled[0].last.is_none());
        self.state.uncompiled.pop().expect("non-empty nodes").trans
    }

    fn top_last_freeze(&mut self, next: StateID) {
        let last = self
            .state
            .uncompiled
            .len()
            .checked_sub(1)
            .expect("non-empty nodes");
        self.state.uncompiled[last].set_last_transition(next);
    }
}

impl Utf8Node {
    fn set_last_transition(&mut self, next: StateID) {
        if let Some(last) = self.last.take() {
            self.trans.push(Transition {
                start: last.start,
                end: last.end,
                next,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use crate::{
        nfa::thompson::{SparseTransitions, State, Transition, NFA},
        util::id::{PatternID, StateID},
    };

    use super::{Compiler, Config};

    fn build(pattern: &str) -> NFA {
        Compiler::new()
            .configure(Config::new().captures(false).unanchored_prefix(false))
            .build(pattern)
            .unwrap()
    }

    fn pid(id: usize) -> PatternID {
        PatternID::new(id).unwrap()
    }

    fn sid(id: usize) -> StateID {
        StateID::new(id).unwrap()
    }

    fn s_byte(byte: u8, next: usize) -> State {
        let next = sid(next);
        let trans = Transition { start: byte, end: byte, next };
        State::Range { range: trans }
    }

    fn s_range(start: u8, end: u8, next: usize) -> State {
        let next = sid(next);
        let trans = Transition { start, end, next };
        State::Range { range: trans }
    }

    fn s_sparse(ranges: &[(u8, u8, usize)]) -> State {
        let ranges = ranges
            .iter()
            .map(|&(start, end, next)| Transition {
                start,
                end,
                next: sid(next),
            })
            .collect();
        State::Sparse(SparseTransitions { ranges })
    }

    fn s_union(alts: &[usize]) -> State {
        State::Union {
            alternates: alts
                .iter()
                .map(|&id| sid(id))
                .collect::<Vec<StateID>>()
                .into_boxed_slice(),
        }
    }

    fn s_match(id: usize) -> State {
        State::Match { pattern_id: pid(id) }
    }

    // Test that building an unanchored NFA has an appropriate `(?s:.)*?`
    // prefix.
    #[test]
    fn compile_unanchored_prefix() {
        // When the machine can only match valid UTF-8.
        let nfa = Compiler::new()
            .configure(Config::new().captures(false))
            .build(r"a")
            .unwrap();
        // There should be many states since the `.` in `(?s:.)*?` matches any
        // Unicode scalar value.
        assert_eq!(11, nfa.states().len());
        assert_eq!(nfa.states()[10], s_match(0));
        assert_eq!(nfa.states()[9], s_byte(b'a', 10));

        // When the machine can match through invalid UTF-8.
        let nfa = Compiler::new()
            .configure(Config::new().captures(false).utf8(false))
            .build(r"a")
            .unwrap();
        assert_eq!(
            nfa.states(),
            &[
                s_union(&[2, 1]),
                s_range(0, 255, 0),
                s_byte(b'a', 3),
                s_match(0),
            ]
        );
    }

    #[test]
    fn compile_empty() {
        assert_eq!(build("").states(), &[s_match(0),]);
    }

    #[test]
    fn compile_literal() {
        assert_eq!(build("a").states(), &[s_byte(b'a', 1), s_match(0),]);
        assert_eq!(
            build("ab").states(),
            &[s_byte(b'a', 1), s_byte(b'b', 2), s_match(0),]
        );
        assert_eq!(
            build("☃").states(),
            &[s_byte(0xE2, 1), s_byte(0x98, 2), s_byte(0x83, 3), s_match(0)]
        );

        // Check that non-UTF-8 literals work.
        let nfa = Compiler::new()
            .configure(
                Config::new()
                    .captures(false)
                    .utf8(false)
                    .unanchored_prefix(false),
            )
            .syntax(crate::SyntaxConfig::new().utf8(false))
            .build(r"(?-u)\xFF")
            .unwrap();
        assert_eq!(nfa.states(), &[s_byte(b'\xFF', 1), s_match(0),]);
    }

    #[test]
    fn compile_class() {
        assert_eq!(
            build(r"[a-z]").states(),
            &[s_range(b'a', b'z', 1), s_match(0),]
        );
        assert_eq!(
            build(r"[x-za-c]").states(),
            &[s_sparse(&[(b'a', b'c', 1), (b'x', b'z', 1)]), s_match(0)]
        );
        assert_eq!(
            build(r"[\u03B1-\u03B4]").states(),
            &[s_range(0xB1, 0xB4, 2), s_byte(0xCE, 0), s_match(0)]
        );
        assert_eq!(
            build(r"[\u03B1-\u03B4\u{1F919}-\u{1F91E}]").states(),
            &[
                s_range(0xB1, 0xB4, 5),
                s_range(0x99, 0x9E, 5),
                s_byte(0xA4, 1),
                s_byte(0x9F, 2),
                s_sparse(&[(0xCE, 0xCE, 0), (0xF0, 0xF0, 3)]),
                s_match(0),
            ]
        );
        assert_eq!(
            build(r"[a-z☃]").states(),
            &[
                s_byte(0x83, 3),
                s_byte(0x98, 0),
                s_sparse(&[(b'a', b'z', 3), (0xE2, 0xE2, 1)]),
                s_match(0),
            ]
        );
    }

    #[test]
    fn compile_repetition() {
        assert_eq!(
            build(r"a?").states(),
            &[s_union(&[1, 2]), s_byte(b'a', 2), s_match(0),]
        );
        assert_eq!(
            build(r"a??").states(),
            &[s_union(&[2, 1]), s_byte(b'a', 2), s_match(0),]
        );
    }

    #[test]
    fn compile_group() {
        assert_eq!(
            build(r"ab+").states(),
            &[s_byte(b'a', 1), s_byte(b'b', 2), s_union(&[1, 3]), s_match(0)]
        );
        assert_eq!(
            build(r"(ab)").states(),
            &[s_byte(b'a', 1), s_byte(b'b', 2), s_match(0)]
        );
        assert_eq!(
            build(r"(ab)+").states(),
            &[s_byte(b'a', 1), s_byte(b'b', 2), s_union(&[0, 3]), s_match(0)]
        );
    }

    #[test]
    fn compile_alternation() {
        assert_eq!(
            build(r"a|b").states(),
            &[s_byte(b'a', 3), s_byte(b'b', 3), s_union(&[0, 1]), s_match(0)]
        );
        assert_eq!(
            build(r"|b").states(),
            &[s_byte(b'b', 2), s_union(&[2, 0]), s_match(0)]
        );
        assert_eq!(
            build(r"a|").states(),
            &[s_byte(b'a', 2), s_union(&[0, 2]), s_match(0)]
        );
    }

    #[test]
    fn many_start_pattern() {
        let nfa = Compiler::new()
            .configure(Config::new().captures(false).unanchored_prefix(false))
            .build_many(&["a", "b"])
            .unwrap();
        assert_eq!(
            nfa.states(),
            &[
                s_byte(b'a', 1),
                s_match(0),
                s_byte(b'b', 3),
                s_match(1),
                s_union(&[0, 2]),
            ]
        );
        assert_eq!(nfa.start_anchored().as_usize(), 4);
        assert_eq!(nfa.start_unanchored().as_usize(), 4);
        // Test that the start states for each individual pattern are correct.
        assert_eq!(nfa.start_pattern(pid(0)), sid(0));
        assert_eq!(nfa.start_pattern(pid(1)), sid(2));
    }
}
