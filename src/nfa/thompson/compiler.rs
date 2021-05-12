// This module provides an NFA compiler using Thompson's construction
// algorithm. The compiler takes a regex-syntax::Hir as input and emits an NFA
// graph as output. The NFA graph is structured in a way that permits it to be
// executed by a virtual machine and also used to efficiently build a DFA.
//
// The compiler deals with a slightly expanded set of NFA states that notably
// includes an empty node that has exactly one epsilon transition to the next
// state. In other words, it's a "goto" instruction if one views Thompson's NFA
// as a set of bytecode instructions. These goto instructions are removed in
// a subsequent phase before returning the NFA to the caller. The purpose of
// these empty nodes is that they make the construction algorithm substantially
// simpler to implement. We remove them before returning to the caller because
// they can represent substantial overhead when traversing the NFA graph
// (either while searching using the NFA directly or while building a DFA).
//
// In the future, it would be nice to provide a Glushkov compiler as well,
// as it would work well as a bit-parallel NFA for smaller regexes. But
// the Thompson construction is one I'm more familiar with and seems more
// straight-forward to deal with when it comes to large Unicode character
// classes.
//
// Internally, the compiler uses interior mutability to improve composition
// in the face of the borrow checker. In particular, we'd really like to be
// able to write things like this:
//
//     self.c_concat(exprs.iter().map(|e| self.c(e)))
//
// Which elegantly uses iterators to build up a sequence of compiled regex
// sub-expressions and then hands it off to the concatenating compiler
// routine. Without interior mutability, the borrow checker won't let us
// borrow `self` mutably both inside and outside the closure at the same
// time.

use std::borrow::Borrow;
use std::cell::RefCell;
use std::mem;

use regex_syntax::hir::{
    self, Anchor, Class, Hir, HirKind, Literal, WordBoundary,
};
use regex_syntax::utf8::{Utf8Range, Utf8Sequences};
use regex_syntax::ParserBuilder;

use crate::classes::ByteClassSet;
use crate::{pattern_limit, PatternID};

use super::super::error::Error;
use super::map::{Utf8BoundedMap, Utf8SuffixKey, Utf8SuffixMap};
use super::range_trie::RangeTrie;
use super::{Look, State, StateID, Transition, NFA};

/// The configuration used for compiling a Thompson NFA from a regex pattern.
#[derive(Clone, Copy, Debug, Default)]
pub struct Config {
    reverse: Option<bool>,
    utf8: Option<bool>,
    shrink: Option<bool>,
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
    pub fn reverse(mut self, yes: bool) -> Config {
        self.reverse = Some(yes);
        self
    }

    /// Whether to enable UTF-8 mode or not.
    ///
    /// When UTF-8 mode is enabled (which is the default), unanchored searches
    /// will only match through valid UTF-8. If invalid UTF-8 is seen, then
    /// an unanchored search will stop at that point. This is equivalent to
    /// putting a `(?s:.)*` at the start of the regex.
    ///
    /// When UTF-8 mode is disabled, then unanchored searches will match
    /// through any arbitrary byte. This is equivalent to putting a `(?s-u:.)*`
    /// at the start of the regex.
    ///
    /// Generally speaking, UTF-8 mode should only be used when you know you
    /// are searching valid UTF-8. Typically, this should only be disabled in
    /// precisely the cases where the regex itself is permitted to match
    /// invalid UTF-8.
    pub fn utf8(mut self, yes: bool) -> Config {
        self.utf8 = Some(yes);
        self
    }

    /// Apply best effort heuristics to shrink the NFA at the expense of more
    /// time/memory.
    ///
    /// This is enabled by default. Generally speaking, if one is using an NFA
    /// to compile DFA, then the extra time used to shrink the NFA will be
    /// more than made up for during DFA construction (potentially by a lot).
    /// In other words, enabling this can substantially decrease the overall
    /// amount of time it takes to build a DFA.
    ///
    /// The only reason to disable this if you want to compile an NFA and start
    /// using it as quickly as possible without needing to build a DFA.
    pub fn shrink(mut self, yes: bool) -> Config {
        self.shrink = Some(yes);
        self
    }

    /// Whether to compile an unanchored prefix into this NFA.
    ///
    /// This is enabled by default. It is made available for tests only to make
    /// it easier to unit test the output of the compiler.
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

    pub fn get_shrink(&self) -> bool {
        self.shrink.unwrap_or(true)
    }

    fn get_unanchored_prefix(&self) -> bool {
        self.unanchored_prefix.unwrap_or(true)
    }

    pub(crate) fn overwrite(self, o: Config) -> Config {
        Config {
            reverse: o.reverse.or(self.reverse),
            utf8: o.utf8.or(self.utf8),
            shrink: o.shrink.or(self.shrink),
            unanchored_prefix: o.unanchored_prefix.or(self.unanchored_prefix),
        }
    }
}

/// A builder for compiling an NFA.
#[derive(Clone, Debug)]
pub struct Builder {
    config: Config,
    parser: ParserBuilder,
}

impl Builder {
    /// Create a new NFA builder with its default configuration.
    pub fn new() -> Builder {
        Builder { config: Config::default(), parser: ParserBuilder::new() }
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

    pub fn build_many_from_hir<H: Borrow<Hir>>(
        &self,
        exprs: &[H],
    ) -> Result<NFA, Error> {
        let mut nfa = NFA::always_match();
        self.build_many_from_hir_with(&mut Compiler::new(), &mut nfa, exprs)?;
        Ok(nfa)
    }

    /// Compile the given high level intermediate representation of a regular
    /// expression into the NFA given using the given compiler. Callers may
    /// prefer this over `build` if they would like to reuse allocations while
    /// compiling many regular expressions.
    ///
    /// On success, the given NFA is completely overwritten with the NFA
    /// produced by the compiler.
    ///
    /// If there was a problem building the NFA, then an error is returned.
    /// The only error that can occur is if the compiled regex would exceed
    /// the size limits configured on this builder. When an error is returned,
    /// the contents of `nfa` are unspecified and should not be relied upon.
    /// However, it can still be reused in subsequent calls to this method.
    fn build_from_hir_with(
        &self,
        compiler: &mut Compiler,
        nfa: &mut NFA,
        expr: &Hir,
    ) -> Result<(), Error> {
        self.build_many_from_hir_with(compiler, nfa, &[expr])
    }

    fn build_many_from_hir_with<H: Borrow<Hir>>(
        &self,
        compiler: &mut Compiler,
        nfa: &mut NFA,
        exprs: &[H],
    ) -> Result<(), Error> {
        compiler.configure(self.config);
        compiler.compile(nfa, exprs)
    }

    /// Apply the given NFA configuration options to this builder.
    pub fn configure(&mut self, config: Config) -> &mut Builder {
        self.config = self.config.overwrite(config);
        self
    }

    /// Set the syntax configuration for this builder using
    /// [`SyntaxConfig`](../../struct.SyntaxConfig.html).
    ///
    /// This permits setting things like case insensitivity, Unicode and multi
    /// line mode.
    ///
    /// This syntax configuration generally only applies when an NFA is built
    /// directly from a pattern string. If an NFA is built from an HIR, then
    /// all syntax settings are ignored.
    pub fn syntax(&mut self, config: crate::SyntaxConfig) -> &mut Builder {
        config.apply(&mut self.parser);
        self
    }
}

/// A compiler that converts a regex abstract syntax to an NFA via Thompson's
/// construction. Namely, this compiler permits epsilon transitions between
/// states.
#[derive(Clone, Debug)]
pub struct Compiler {
    /// The set of compiled NFA states. Once a state is compiled, it is
    /// assigned a state ID equivalent to its index in this list. Subsequent
    /// compilation can modify previous states by adding new transitions.
    states: RefCell<Vec<CState>>,
    /// The configuration from the builder.
    config: Config,
    /// State used for compiling character classes to UTF-8 byte automata.
    /// State is not retained between character class compilations. This just
    /// serves to amortize allocation to the extent possible.
    utf8_state: RefCell<Utf8State>,
    /// State used for arranging character classes in reverse into a trie.
    trie_state: RefCell<RangeTrie>,
    /// State used for caching common suffixes when compiling reverse UTF-8
    /// automata (for Unicode character classes).
    utf8_suffix: RefCell<Utf8SuffixMap>,
    /// A map used to re-map state IDs when translating the compiler's internal
    /// NFA state representation to the external NFA representation.
    remap: RefCell<Vec<StateID>>,
    /// A set of compiler internal state IDs that correspond to states that are
    /// exclusively epsilon transitions, i.e., goto instructions, combined with
    /// the state that they point to. This is used to record said states while
    /// transforming the compiler's internal NFA representation to the external
    /// form.
    empties: RefCell<Vec<(StateID, StateID)>>,
}

/// A compiler intermediate state representation for an NFA that is only used
/// during compilation. Once compilation is done, `CState`s are converted
/// to `State`s (defined in the parent module), which have a much simpler
/// representation.
#[derive(Clone, Debug, Eq, PartialEq)]
enum CState {
    /// An empty state whose only purpose is to forward the automaton to
    /// another state via en epsilon transition. These are useful during
    /// compilation but are otherwise removed at the end.
    Empty { next: StateID },
    /// A state that only transitions to `next` if the current input byte is
    /// in the range `[start, end]` (inclusive on both ends).
    Range { range: Transition },
    /// A state with possibly many transitions, represented in a sparse
    /// fashion. Transitions are ordered lexicographically by input range.
    /// As such, this may only be used when every transition has equal
    /// priority. (In practice, this is only used for encoding large UTF-8
    /// automata.) In contrast, a `Union` state has each alternate in order
    /// of priority. Priority is used to implement greedy matching and also
    /// alternations themselves, e.g., `abc|a` where `abc` has priority over
    /// `a`.
    ///
    /// To clarify, it is possible to remove `Sparse` and represent all things
    /// that `Sparse` is used for via `Union`. But this creates a more bloated
    /// NFA with more epsilon transitions than is necessary in the special case
    /// of character classes.
    Sparse { ranges: Vec<Transition> },
    /// A conditional epsilon transition satisfied via some sort of
    /// look-around.
    Look { look: Look, next: StateID },
    /// An alternation such that there exists an epsilon transition to all
    /// states in `alternates`, where matches found via earlier transitions
    /// are preferred over later transitions.
    Union { alternates: Vec<StateID> },
    /// An alternation such that there exists an epsilon transition to all
    /// states in `alternates`, where matches found via later transitions are
    /// preferred over earlier transitions.
    ///
    /// This "reverse" state exists for convenience during compilation that
    /// permits easy construction of non-greedy combinations of NFA states. At
    /// the end of compilation, Union and UnionReverse states are merged into
    /// one Union type of state, where the latter has its epsilon transitions
    /// reversed to reflect the priority inversion.
    ///
    /// The "convenience" here arises from the fact that as new states are
    /// added to the list of `alternates`, we would like that add operation
    /// to be amortized constant time. But if we used a `Union`, we'd need to
    /// prepend the state, which takes O(n) time. There are other approaches we
    /// could use to solve this, but this seems simple enough.
    UnionReverse { alternates: Vec<StateID> },
    /// A match state. There is exactly one such occurrence of this state in
    /// an NFA for each pattern compiled into the NFA.
    Match(PatternID),
}

/// A value that represents the result of compiling a sub-expression of a
/// regex's HIR. Specifically, this represents a sub-graph of the NFA that
/// has an initial state at `start` and a final state at `end`.
#[derive(Clone, Copy, Debug)]
pub struct ThompsonRef {
    start: StateID,
    end: StateID,
}

impl Compiler {
    /// Create a new compiler.
    pub fn new() -> Compiler {
        Compiler {
            states: RefCell::new(vec![]),
            config: Config::default(),
            utf8_state: RefCell::new(Utf8State::new()),
            trie_state: RefCell::new(RangeTrie::new()),
            utf8_suffix: RefCell::new(Utf8SuffixMap::new(1000)),
            remap: RefCell::new(vec![]),
            empties: RefCell::new(vec![]),
        }
    }

    /// Configure and prepare this compiler from the builder's knobs.
    ///
    /// The compiler is must always reconfigured by the builder before using it
    /// to build an NFA. Namely, this will also clear any latent state in the
    /// compiler used during previous compilations.
    fn configure(&mut self, config: Config) {
        self.config = config;
        self.states.borrow_mut().clear();
        // We don't need to clear anything else since they are cleared on
        // their own and only when they are used.
    }

    /// Convert the current intermediate NFA to its final compiled form.
    fn compile<H: Borrow<Hir>>(
        &self,
        nfa: &mut NFA,
        exprs: &[H],
    ) -> Result<(), Error> {
        if exprs.is_empty() {
            *nfa = NFA::never_match();
            return Ok(());
        }
        if exprs.len() > pattern_limit() {
            return Err(Error::too_many_patterns(
                exprs.len(),
                pattern_limit(),
            ));
        }

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

        let mut start_pattern = vec![0; exprs.len()];
        let compiled =
            self.c_alternation(exprs.iter().enumerate().map(|(i, e)| {
                let one = self.c(e.borrow())?;
                let match_state_id = self.add_match(i as PatternID);
                self.patch(one.end, match_state_id);
                start_pattern[i] = one.start;
                Ok(ThompsonRef { start: one.start, end: match_state_id })
            }))?;
        self.patch(unanchored_prefix.end, compiled.start);
        self.finish(
            nfa,
            compiled.start,
            unanchored_prefix.start,
            &start_pattern,
        );
        Ok(())
    }

    /// Finishes the compilation process and populates the provide NFA with
    /// the final graph.
    fn finish(
        &self,
        nfa: &mut NFA,
        start_anchored: StateID,
        start_unanchored: StateID,
        start_pattern: &[StateID],
    ) {
        let mut bstates = self.states.borrow_mut();
        let mut remap = self.remap.borrow_mut();
        remap.resize(bstates.len(), 0);
        let mut empties = self.empties.borrow_mut();
        empties.clear();

        nfa.clear();
        let mut byteset = ByteClassSet::new();

        // The idea here is to convert our intermediate states to their final
        // form. The only real complexity here is the process of converting
        // transitions, which are expressed in terms of state IDs. The new
        // set of states will be smaller because of partial epsilon removal,
        // so the state IDs will not be the same.
        for (id, bstate) in bstates.iter_mut().enumerate() {
            match *bstate {
                CState::Empty { next } => {
                    // Since we're removing empty states, we need to handle
                    // them later since we don't yet know which new state this
                    // empty state will be mapped to.
                    empties.push((id, next));
                }
                CState::Range { range } => {
                    byteset.set_range(range.start, range.end);
                    remap[id] = nfa.add(State::Range { range });
                }
                CState::Sparse { ref mut ranges } => {
                    let ranges = mem::replace(ranges, vec![]);
                    for r in &ranges {
                        byteset.set_range(r.start, r.end);
                    }
                    remap[id] = nfa.add(State::Sparse {
                        ranges: ranges.into_boxed_slice(),
                    });
                }
                CState::Look { look, next } => {
                    look.add_to_byteset(&mut byteset);
                    remap[id] = nfa.add(State::Look { look, next });
                }
                CState::Union { ref mut alternates } => {
                    let alternates = mem::replace(alternates, vec![]);
                    remap[id] = nfa.add(State::Union {
                        alternates: alternates.into_boxed_slice(),
                    });
                }
                CState::UnionReverse { ref mut alternates } => {
                    let mut alternates = mem::replace(alternates, vec![]);
                    alternates.reverse();
                    remap[id] = nfa.add(State::Union {
                        alternates: alternates.into_boxed_slice(),
                    });
                }
                CState::Match(mid) => {
                    remap[id] = nfa.add(State::Match(mid));
                }
            }
        }
        for &(empty_id, mut empty_next) in empties.iter() {
            // empty states can point to other empty states, forming a chain.
            // So we must follow the chain until the end, which must end at
            // a non-empty state, and therefore, a state that is correctly
            // remapped. We are guaranteed to terminate because our compiler
            // never builds a loop among only empty states.
            while let CState::Empty { next } = bstates[empty_next] {
                empty_next = next;
            }
            remap[empty_id] = remap[empty_next];
        }
        for state in nfa.states_mut() {
            state.remap(&remap);
        }
        // The compiler always begins the NFA at the first state.
        nfa.set_byte_class_set(byteset.clone());
        nfa.set_byte_classes(byteset.byte_classes());
        nfa.set_start_anchored(remap[start_anchored]);
        nfa.set_start_unanchored(remap[start_unanchored]);
        for (i, &old_id) in start_pattern.iter().enumerate() {
            nfa.set_start_pattern(i as PatternID, remap[old_id]);
        }
    }

    fn c(&self, expr: &Hir) -> Result<ThompsonRef, Error> {
        match *expr.kind() {
            HirKind::Empty => self.c_empty(),
            HirKind::Literal(Literal::Unicode(ch)) => self.c_char(ch),
            HirKind::Literal(Literal::Byte(b)) => self.c_range(b, b),
            HirKind::Class(Class::Bytes(ref c)) => self.c_byte_class(c),
            HirKind::Class(Class::Unicode(ref c)) => self.c_unicode_class(c),
            HirKind::Anchor(ref anchor) => self.c_anchor(anchor),
            HirKind::WordBoundary(ref wb) => self.c_word_boundary(wb),
            HirKind::Repetition(ref rep) => self.c_repetition(rep),
            HirKind::Group(ref group) => self.c(&*group.hir),
            HirKind::Concat(ref es) => {
                self.c_concat(es.iter().map(|e| self.c(e)))
            }
            HirKind::Alternation(ref es) => {
                self.c_alternation(es.iter().map(|e| self.c(e)))
            }
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
            self.patch(end, compiled.start);
            end = compiled.end;
        }
        Ok(ThompsonRef { start, end })
    }

    fn c_alternation<I>(&self, mut it: I) -> Result<ThompsonRef, Error>
    where
        I: Iterator<Item = Result<ThompsonRef, Error>>,
    {
        let first = it.next().expect("alternations must be non-empty")?;
        let second = match it.next() {
            None => return Ok(first),
            Some(result) => result?,
        };

        let union = self.add_union();
        let end = self.add_empty();
        self.patch(union, first.start);
        self.patch(first.end, end);
        self.patch(union, second.start);
        self.patch(second.end, end);
        for result in it {
            let compiled = result?;
            self.patch(union, compiled.start);
            self.patch(compiled.end, end);
        }
        Ok(ThompsonRef { start: union, end })
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
        //     000001: 61 => 02
        //     000002: union(03, 04)
        //     000003: 61 => 04
        //     000004: union(05, 06)
        //     000005: 61 => 06
        //     000006: union(07, 08)
        //     000007: 61 => 08
        //     000008: MATCH
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
        let empty = self.add_empty();
        let mut prev_end = prefix.end;
        for _ in min..max {
            let union = if greedy {
                self.add_union()
            } else {
                self.add_reverse_union()
            };
            let compiled = self.c(expr)?;
            self.patch(prev_end, union);
            self.patch(union, compiled.start);
            self.patch(union, empty);
            prev_end = compiled.end;
        }
        self.patch(prev_end, empty);
        Ok(ThompsonRef { start: prefix.start, end: empty })
    }

    fn c_at_least(
        &self,
        expr: &Hir,
        greedy: bool,
        n: u32,
    ) -> Result<ThompsonRef, Error> {
        if n == 0 {
            let union = if greedy {
                self.add_union()
            } else {
                self.add_reverse_union()
            };
            let compiled = self.c(expr)?;
            self.patch(union, compiled.start);
            self.patch(compiled.end, union);
            Ok(ThompsonRef { start: union, end: union })
        } else if n == 1 {
            let compiled = self.c(expr)?;
            let union = if greedy {
                self.add_union()
            } else {
                self.add_reverse_union()
            };
            self.patch(compiled.end, union);
            self.patch(union, compiled.start);
            Ok(ThompsonRef { start: compiled.start, end: union })
        } else {
            let prefix = self.c_exactly(expr, n - 1)?;
            let last = self.c(expr)?;
            let union = if greedy {
                self.add_union()
            } else {
                self.add_reverse_union()
            };
            self.patch(prefix.end, last.start);
            self.patch(last.end, union);
            self.patch(union, last.start);
            Ok(ThompsonRef { start: prefix.start, end: union })
        }
    }

    fn c_zero_or_one(
        &self,
        expr: &Hir,
        greedy: bool,
    ) -> Result<ThompsonRef, Error> {
        let union =
            if greedy { self.add_union() } else { self.add_reverse_union() };
        let compiled = self.c(expr)?;
        let empty = self.add_empty();
        self.patch(union, compiled.start);
        self.patch(union, empty);
        self.patch(compiled.end, empty);
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
        let end = self.add_empty();
        let mut trans = Vec::with_capacity(cls.ranges().len());
        for r in cls.iter() {
            trans.push(Transition {
                start: r.start(),
                end: r.end(),
                next: end,
            });
        }
        Ok(ThompsonRef { start: self.add_sparse(trans), end })
    }

    fn c_unicode_class(
        &self,
        cls: &hir::ClassUnicode,
    ) -> Result<ThompsonRef, Error> {
        // If all we have are ASCII ranges wrapped in a Unicode package, then
        // there is zero reason to bring out the big guns. We can fit all ASCII
        // ranges within a single sparse transition.
        if cls.is_all_ascii() {
            let end = self.add_empty();
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
            Ok(ThompsonRef { start: self.add_sparse(trans), end })
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
                // exists, so that this path can be toggled off.
                let mut trie = self.trie_state.borrow_mut();
                trie.clear();

                for rng in cls.iter() {
                    for mut seq in Utf8Sequences::new(rng.start(), rng.end()) {
                        seq.reverse();
                        trie.insert(seq.as_slice());
                    }
                }
                let mut utf8_state = self.utf8_state.borrow_mut();
                let mut utf8c = Utf8Compiler::new(self, &mut *utf8_state);
                trie.iter(|seq| {
                    utf8c.add(&seq);
                });
                Ok(utf8c.finish())
            }
        } else {
            // In the forward direction, we always shrink our UTF-8 automata
            // because we can stream it right into the UTF-8 compiler. There
            // is almost no downside (in either memory or time) to using this
            // approach.
            let mut utf8_state = self.utf8_state.borrow_mut();
            let mut utf8c = Utf8Compiler::new(self, &mut *utf8_state);
            for rng in cls.iter() {
                for seq in Utf8Sequences::new(rng.start(), rng.end()) {
                    utf8c.add(seq.as_slice());
                }
            }
            Ok(utf8c.finish())
        }

        // For reference, the code below is the "naive" version of compiling a
        // UTF-8 automaton. It is deliciously simple (and works for both the
        // forward and reverse cases), but will unfortunately produce very
        // large NFAs. When compiling a forward automaton, the size difference
        // can sometimes be an order of magnitude. For example, the '\w' regex
        // will generate about ~3000 NFA states using the naive approach below,
        // but only 283 states when using the approach above. This is because
        // the approach above actually compiles a *minimal* (or near minimal,
        // because of the bounded hashmap) UTF-8 automaton.
        //
        // The code below is kept as a reference point in order to make it
        // easier to understand the higher level goal here.
        /*
        let it = cls
            .iter()
            .flat_map(|rng| Utf8Sequences::new(rng.start(), rng.end()))
            .map(|seq| {
                let it = seq
                    .as_slice()
                    .iter()
                    .map(|rng| Ok(self.c_range(rng.start, rng.end)));
                self.c_concat(it)
            });
        self.c_alternation(it);
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

        let union = self.add_union();
        let alt_end = self.add_empty();
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
                    self.patch(compiled.end, end);
                    end = compiled.start;
                    cache.set(key, hash, end);
                }
                self.patch(union, end);
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
        let id = self.add_look(look);
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
        let id = self.add_look(look);
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
        let id = self.add_range(start, end);
        Ok(ThompsonRef { start: id, end: id })
    }

    fn c_empty(&self) -> Result<ThompsonRef, Error> {
        let id = self.add_empty();
        Ok(ThompsonRef { start: id, end: id })
    }

    fn c_unanchored_prefix_valid_utf8(&self) -> Result<ThompsonRef, Error> {
        self.c(&Hir::repetition(hir::Repetition {
            kind: hir::RepetitionKind::ZeroOrMore,
            greedy: false,
            hir: Box::new(Hir::any(false)),
        }))
    }

    fn c_unanchored_prefix_invalid_utf8(&self) -> Result<ThompsonRef, Error> {
        self.c(&Hir::repetition(hir::Repetition {
            kind: hir::RepetitionKind::ZeroOrMore,
            greedy: false,
            hir: Box::new(Hir::any(true)),
        }))
    }

    fn patch(&self, from: StateID, to: StateID) {
        match self.states.borrow_mut()[from] {
            CState::Empty { ref mut next } => {
                *next = to;
            }
            CState::Range { ref mut range } => {
                range.next = to;
            }
            CState::Sparse { .. } => {
                panic!("cannot patch from a sparse NFA state")
            }
            CState::Look { ref mut next, .. } => {
                *next = to;
            }
            CState::Union { ref mut alternates } => {
                alternates.push(to);
            }
            CState::UnionReverse { ref mut alternates } => {
                alternates.push(to);
            }
            CState::Match(_) => {}
        }
    }

    fn add_empty(&self) -> StateID {
        let id = self.states.borrow().len();
        self.states.borrow_mut().push(CState::Empty { next: 0 });
        id
    }

    fn add_range(&self, start: u8, end: u8) -> StateID {
        let id = self.states.borrow().len();
        let trans = Transition { start, end, next: 0 };
        let state = CState::Range { range: trans };
        self.states.borrow_mut().push(state);
        id
    }

    fn add_sparse(&self, ranges: Vec<Transition>) -> StateID {
        if ranges.len() == 1 {
            let id = self.states.borrow().len();
            let state = CState::Range { range: ranges[0] };
            self.states.borrow_mut().push(state);
            return id;
        }
        let id = self.states.borrow().len();
        let state = CState::Sparse { ranges };
        self.states.borrow_mut().push(state);
        id
    }

    fn add_look(&self, mut look: Look) -> StateID {
        if self.is_reverse() {
            look = look.reversed();
        }
        let id = self.states.borrow().len();
        self.states.borrow_mut().push(CState::Look { look, next: 0 });
        id
    }

    fn add_union(&self) -> StateID {
        let id = self.states.borrow().len();
        let state = CState::Union { alternates: vec![] };
        self.states.borrow_mut().push(state);
        id
    }

    fn add_reverse_union(&self) -> StateID {
        let id = self.states.borrow().len();
        let state = CState::UnionReverse { alternates: vec![] };
        self.states.borrow_mut().push(state);
        id
    }

    fn add_match(&self, pid: PatternID) -> StateID {
        let id = self.states.borrow().len();
        self.states.borrow_mut().push(CState::Match(pid));
        id
    }

    fn is_reverse(&self) -> bool {
        self.config.get_reverse()
    }
}

#[derive(Debug)]
struct Utf8Compiler<'a> {
    nfac: &'a Compiler,
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
        Utf8State { compiled: Utf8BoundedMap::new(5000), uncompiled: vec![] }
    }

    fn clear(&mut self) {
        self.compiled.clear();
        self.uncompiled.clear();
    }
}

impl<'a> Utf8Compiler<'a> {
    fn new(nfac: &'a Compiler, state: &'a mut Utf8State) -> Utf8Compiler<'a> {
        let target = nfac.add_empty();
        state.clear();
        let mut utf8c = Utf8Compiler { nfac, state, target };
        utf8c.add_empty();
        utf8c
    }

    fn finish(&mut self) -> ThompsonRef {
        self.compile_from(0);
        let node = self.pop_root();
        let start = self.compile(node);
        ThompsonRef { start, end: self.target }
    }

    fn add(&mut self, ranges: &[Utf8Range]) {
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
        self.compile_from(prefix_len);
        self.add_suffix(&ranges[prefix_len..]);
    }

    fn compile_from(&mut self, from: usize) {
        let mut next = self.target;
        while from + 1 < self.state.uncompiled.len() {
            let node = self.pop_freeze(next);
            next = self.compile(node);
        }
        self.top_last_freeze(next);
    }

    fn compile(&mut self, node: Vec<Transition>) -> StateID {
        let hash = self.state.compiled.hash(&node);
        if let Some(id) = self.state.compiled.get(&node, hash) {
            return id;
        }
        let id = self.nfac.add_sparse(node.clone());
        self.state.compiled.set(node, hash, id);
        id
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
    use regex_syntax::hir::Hir;
    use regex_syntax::ParserBuilder;

    use super::{Builder, Config, PatternID, State, StateID, Transition, NFA};

    fn build(pattern: &str) -> NFA {
        Builder::new()
            .configure(Config::new().unanchored_prefix(false))
            .build(pattern)
            .unwrap()
    }

    fn s_byte(byte: u8, next: StateID) -> State {
        let trans = Transition { start: byte, end: byte, next };
        State::Range { range: trans }
    }

    fn s_range(start: u8, end: u8, next: StateID) -> State {
        let trans = Transition { start, end, next };
        State::Range { range: trans }
    }

    fn s_sparse(ranges: &[(u8, u8, StateID)]) -> State {
        let ranges = ranges
            .iter()
            .map(|&(start, end, next)| Transition { start, end, next })
            .collect();
        State::Sparse { ranges }
    }

    fn s_union(alts: &[StateID]) -> State {
        State::Union { alternates: alts.to_vec().into_boxed_slice() }
    }

    fn s_match(id: PatternID) -> State {
        State::Match(id)
    }

    // Test that building an unanchored NFA has an appropriate `.*?` prefix.
    #[test]
    fn compile_unanchored_prefix() {
        // When the machine can only match valid UTF-8.
        let nfa = Builder::new().build(r"a").unwrap();
        // There should be many states since the `.` in `.*?` matches any
        // Unicode scalar value.
        assert_eq!(11, nfa.len());
        assert_eq!(nfa.states[10], s_match(0));
        assert_eq!(nfa.states[9], s_byte(b'a', 10));

        // When the machine can match through invalid UTF-8.
        let nfa = Builder::new()
            .configure(Config::new().utf8(false))
            .build(r"a")
            .unwrap();
        assert_eq!(
            nfa.states,
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
        assert_eq!(build("").states, &[s_match(0),]);
    }

    #[test]
    fn compile_literal() {
        assert_eq!(build("a").states, &[s_byte(b'a', 1), s_match(0),]);
        assert_eq!(
            build("ab").states,
            &[s_byte(b'a', 1), s_byte(b'b', 2), s_match(0),]
        );
        assert_eq!(
            build("☃").states,
            &[s_byte(0xE2, 1), s_byte(0x98, 2), s_byte(0x83, 3), s_match(0)]
        );

        // Check that non-UTF-8 literals work.
        let nfa = Builder::new()
            .configure(Config::new().utf8(false).unanchored_prefix(false))
            .syntax(crate::SyntaxConfig::new().allow_invalid_utf8(true))
            .build(r"(?-u)\xFF")
            .unwrap();
        assert_eq!(nfa.states, &[s_byte(b'\xFF', 1), s_match(0),]);
    }

    #[test]
    fn compile_class() {
        assert_eq!(
            build(r"[a-z]").states,
            &[s_range(b'a', b'z', 1), s_match(0),]
        );
        assert_eq!(
            build(r"[x-za-c]").states,
            &[s_sparse(&[(b'a', b'c', 1), (b'x', b'z', 1)]), s_match(0)]
        );
        assert_eq!(
            build(r"[\u03B1-\u03B4]").states,
            &[s_range(0xB1, 0xB4, 2), s_byte(0xCE, 0), s_match(0)]
        );
        assert_eq!(
            build(r"[\u03B1-\u03B4\u{1F919}-\u{1F91E}]").states,
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
            build(r"[a-z☃]").states,
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
            build(r"a?").states,
            &[s_union(&[1, 2]), s_byte(b'a', 2), s_match(0),]
        );
        assert_eq!(
            build(r"a??").states,
            &[s_union(&[2, 1]), s_byte(b'a', 2), s_match(0),]
        );
    }

    #[test]
    fn compile_group() {
        assert_eq!(
            build(r"ab+").states,
            &[s_byte(b'a', 1), s_byte(b'b', 2), s_union(&[1, 3]), s_match(0)]
        );
        assert_eq!(
            build(r"(ab)").states,
            &[s_byte(b'a', 1), s_byte(b'b', 2), s_match(0)]
        );
        assert_eq!(
            build(r"(ab)+").states,
            &[s_byte(b'a', 1), s_byte(b'b', 2), s_union(&[0, 3]), s_match(0)]
        );
    }

    #[test]
    fn compile_alternation() {
        assert_eq!(
            build(r"a|b").states,
            &[s_byte(b'a', 3), s_byte(b'b', 3), s_union(&[0, 1]), s_match(0)]
        );
        assert_eq!(
            build(r"|b").states,
            &[s_byte(b'b', 2), s_union(&[2, 0]), s_match(0)]
        );
        assert_eq!(
            build(r"a|").states,
            &[s_byte(b'a', 2), s_union(&[0, 2]), s_match(0)]
        );
    }

    #[test]
    fn many_start_pattern() {
        let nfa = Builder::new()
            .configure(Config::new().unanchored_prefix(false))
            .build_many(&["a", "b"])
            .unwrap();
        assert_eq!(
            nfa.states,
            &[
                s_byte(b'a', 1),
                s_match(0),
                s_byte(b'b', 3),
                s_match(1),
                s_union(&[0, 2]),
            ]
        );
        assert_eq!(nfa.start_anchored(), 4);
        assert_eq!(nfa.start_unanchored(), 4);
        // Test that the start states for each individual pattern are correct.
        assert_eq!(nfa.start_pattern(0), 0);
        assert_eq!(nfa.start_pattern(1), 2);
    }
}
