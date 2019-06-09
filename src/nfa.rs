use std::cell::RefCell;
use std::fmt;
use std::iter;

use regex_syntax::hir::{self, Hir, HirKind};

use classes::ByteClasses;
use error::{Error, Result};

/// The representation for an NFA state identifier.
pub type StateID = usize;

/// A final compiled NFA.
///
/// The states of the NFA are indexed by state IDs, which are how transitions
/// are expressed.
#[derive(Clone)]
pub struct NFA {
    /// Whether this NFA can only match at the beginning of input or not.
    ///
    /// When true, a match should only be reported if it begins at the 0th
    /// index of the haystack.
    anchored: bool,
    /// The starting state of this NFA.
    start: StateID,
    /// The state list. This list is guaranteed to be indexable by the starting
    /// state ID, and it is also guaranteed to contain exactly one `Match`
    /// state.
    states: Vec<State>,
    /// A mapping from any byte value to its corresponding equivalence class
    /// identifier. Two bytes in the same equivalence class cannot discriminate
    /// between a match or a non-match. This map can be used to shrink the
    /// total size of a DFA's transition table with a small match-time cost.
    ///
    /// Note that the NFA's transitions are *not* defined in terms of these
    /// equivalence classes. The NFA's transitions are defined on the original
    /// byte values.
    byte_classes: ByteClasses,
}

/// A state in a final compiled NFA.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum State {
    /// A state that transitions to `next` if and only if the current input
    /// byte is in the range `[start, end]` (inclusive).
    Range { start: u8, end: u8, next: StateID },
    /// An alternation such that there exists an epsilon transition to all
    /// states in `alternates`, where matches found via earlier transitions
    /// are preferred over later transitions.
    Union { alternates: Vec<StateID> },
    /// A match state. There is exactly one such occurrence of this state in
    /// an NFA.
    Match,
}

impl NFA {
    /// Returns true if and only if this NFA is anchored.
    pub fn is_anchored(&self) -> bool {
        self.anchored
    }

    /// Return the number of states in this NFA.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Return the ID of the initial state of this NFA.
    pub fn start(&self) -> StateID {
        self.start
    }

    /// Return the NFA state corresponding to the given ID.
    pub fn state(&self, id: StateID) -> &State {
        &self.states[id]
    }

    /// Return the set of equivalence classes for this NFA. The slice returned
    /// always has length 256 and maps each possible byte value to its
    /// corresponding equivalence class ID (which is never more than 255).
    pub fn byte_classes(&self) -> &ByteClasses {
        &self.byte_classes
    }
}

impl State {
    /// Returns true if and only if this state contains one or more epsilon
    /// transitions.
    pub fn is_epsilon(&self) -> bool {
        match *self {
            State::Range { .. } | State::Match => false,
            State::Union { .. } => true,
        }
    }

    /// Remap the transitions in this state using the given map. Namely, the
    /// given map should be indexed according to the transitions currently
    /// in this state.
    ///
    /// This is used during the final phase of the NFA compiler, which turns
    /// its intermediate NFA into the final NFA.
    fn remap(&mut self, remap: &[StateID]) {
        match *self {
            State::Range { ref mut next, .. } => *next = remap[*next],
            State::Union { ref mut alternates } => {
                for alt in alternates {
                    *alt = remap[*alt];
                }
            }
            State::Match => {}
        }
    }
}

/// A builder for compiling an NFA.
#[derive(Clone, Debug)]
pub struct NFABuilder {
    anchored: bool,
    allow_invalid_utf8: bool,
    reverse: bool,
}

impl NFABuilder {
    /// Create a new NFA builder with its default configuration.
    pub fn new() -> NFABuilder {
        NFABuilder {
            anchored: false,
            allow_invalid_utf8: false,
            reverse: false,
        }
    }

    /// Compile the given high level intermediate representation of a regular
    /// expression into an NFA.
    ///
    /// If there was a problem building the NFA, then an error is returned.
    /// For example, if the regex uses unsupported features (such as zero-width
    /// assertions), then an error is returned.
    pub fn build(&self, mut expr: Hir) -> Result<NFA> {
        if self.reverse {
            expr = reverse_hir(expr);
        }
        let compiler = NFACompiler {
            states: RefCell::new(vec![]),
            reverse: self.reverse,
        };

        let mut start = compiler.add_empty();
        if !self.anchored {
            let compiled =
                if self.allow_invalid_utf8 {
                    compiler.compile_unanchored_prefix_invalid_utf8()
                } else {
                    compiler.compile_unanchored_prefix_valid_utf8()
                }?;
            compiler.patch(start, compiled.start);
            start = compiled.end;
        }
        let compiled = compiler.compile(&expr)?;
        let match_id = compiler.add_match();
        compiler.patch(start, compiled.start);
        compiler.patch(compiled.end, match_id);
        Ok(NFA { anchored: self.anchored, ..compiler.to_nfa() })
    }

    /// Set whether matching must be anchored at the beginning of the input.
    ///
    /// When enabled, a match must begin at the start of the input. When
    /// disabled, the NFA will act as if the pattern started with a `.*?`,
    /// which enables a match to appear anywhere.
    ///
    /// By default this is disabled.
    pub fn anchored(&mut self, yes: bool) -> &mut NFABuilder {
        self.anchored = yes;
        self
    }

    /// When enabled, the builder will permit the construction of an NFA that
    /// may match invalid UTF-8.
    ///
    /// When disabled (the default), the builder is guaranteed to produce a
    /// regex that will only ever match valid UTF-8 (otherwise, the builder
    /// will return an error).
    pub fn allow_invalid_utf8(&mut self, yes: bool) -> &mut NFABuilder {
        self.allow_invalid_utf8 = yes;
        self
    }

    /// Reverse the NFA.
    ///
    /// A NFA reversal is performed by reversing all of the concatenated
    /// sub-expressions in the original pattern, recursively. The resulting
    /// NFA can be used to match the pattern starting from the end of a string
    /// instead of the beginning of a string.
    ///
    /// Reversing the NFA is useful for building a reverse DFA, which is most
    /// useful for finding the start of a match.
    pub fn reverse(&mut self, yes: bool) -> &mut NFABuilder {
        self.reverse = yes;
        self
    }
}

/// A compiler that converts a regex AST (well, a high-level IR) to an NFA via
/// Thompson's construction. Namely, we permit epsilon transitions.
///
/// The compiler deals with a slightly expanded set of NFA states that notably
/// includes an empty node that has exactly one epsilon transition to the
/// next state. In other words, it's a "goto" instruction if one views
/// Thompson's NFA as a set of bytecode instructions. These goto instructions
/// are removed in a subsequent phase before returning the NFA to the caller.
/// The purpose of these empty nodes is that they make the construction
/// algorithm substantially simpler to implement.
#[derive(Debug)]
struct NFACompiler {
    /// The set of compiled NFA states. Once a state is compiled, it is
    /// assigned a state ID equivalent to its index in this list. Subsequent
    /// compilation can modify previous states by adding new transitions.
    ///
    /// We use a RefCell here because the borrow checker otherwise makes
    /// logical decomposition into methods much harder otherwise.
    states: RefCell<Vec<BState>>,
    /// When true, we are compiling an HIR in reverse. Note that we actually
    /// reverse the HIR before handing it to this compiler, but the compiler
    /// does need to know to reverse UTF-8 automata since the HIR is expressed
    /// in terms of Unicode codepoints.
    reverse: bool,
}

/// A "builder" intermediate state representation for an NFA that is only used
/// during compilation. Once compilation is done, `BState`s are converted to
/// `State`s, which have a much simpler representation.
#[derive(Clone, Debug, Eq, PartialEq)]
enum BState {
    /// An empty state whose only purpose is to forward the automaton to
    /// another state via en epsilon transition. These are useful during
    /// compilation but are otherwise removed at the end.
    Empty { next: StateID },
    /// A state that only transitions to `next` if the current input byte is
    /// in the range `[start, end]` (inclusive on both ends).
    Range { start: u8, end: u8, next: StateID },
    /// An alternation such that there exists an epsilon transition to all
    /// states in `alternates`, where matches found via earlier transitions
    /// are preferred over later transitions.
    Union { alternates: Vec<StateID> },
    /// An alternation such that there exists an epsilon transition to all
    /// states in `alternates`, where matches found via later transitions
    /// are preferred over earlier transitions.
    ///
    /// This "reverse" state exists for convenience during compilation that
    /// permits easy construction of non-greedy combinations of NFA states.
    /// At the end of compilation, Union and UnionReverse states are merged
    /// into one Union type of state, where the latter has its epsilon
    /// transitions reversed to reflect the priority inversion.
    UnionReverse { alternates: Vec<StateID> },
    /// A match state. There is exactly one such occurrence of this state in
    /// an NFA.
    Match,
}

/// A value that represents the result of compiling a sub-expression of a
/// regex's HIR. Specifically, this represents a sub-graph of the NFA that
/// has an initial state at `start` and a final state at `end`.
#[derive(Clone, Copy, Debug)]
struct ThompsonRef {
    start: StateID,
    end: StateID,
}

impl NFACompiler {
    /// Convert the current intermediate NFA to its final compiled form.
    fn to_nfa(&self) -> NFA {
        let bstates = self.states.borrow();
        let mut states = vec![];
        let mut remap = vec![0; bstates.len()];
        let mut empties = vec![];
        let mut byteset = ByteClassSet::new();

        // The idea here is to convert our intermediate states to their final
        // form. The only real complexity here is the process of converting
        // transitions, which are expressed in terms of state IDs. The new
        // set of states will be smaller because of partial epsilon removal,
        // so the state IDs will not be the same.
        for (id, bstate) in bstates.iter().enumerate() {
            match *bstate {
                BState::Empty { next } => {
                    // Since we're removing empty states, we need to handle
                    // them later since we don't yet know which new state this
                    // empty state will be mapped to.
                    empties.push((id, next));
                }
                BState::Range { start, end, next } => {
                    remap[id] = states.len();
                    states.push(State::Range { start, end, next });
                    byteset.set_range(start, end);
                }
                BState::Union { ref alternates } => {
                    remap[id] = states.len();

                    let alternates = alternates.clone();
                    states.push(State::Union { alternates });
                }
                BState::UnionReverse { ref alternates } => {
                    remap[id] = states.len();

                    let mut alternates = alternates.clone();
                    alternates.reverse();
                    states.push(State::Union { alternates });
                }
                BState::Match => {
                    remap[id] = states.len();
                    states.push(State::Match);
                }
            }
        }
        for (empty_id, mut empty_next) in empties {
            // empty states can point to other empty states, forming a chain.
            // So we must follow the chain until the end, which must point to
            // a non-empty state, and therefore, a state that is correctly
            // remapped.
            while let BState::Empty { next } = bstates[empty_next] {
                empty_next = next;
            }
            remap[empty_id] = remap[empty_next];
        }
        for state in &mut states {
            state.remap(&remap);
        }
        // The compiler always begins the NFA at the first state.
        let byte_classes = byteset.byte_classes();
        NFA { anchored: false, start: remap[0], states, byte_classes }
    }

    fn compile(&self, expr: &Hir) -> Result<ThompsonRef> {
        match *expr.kind() {
            HirKind::Empty => {
                let id = self.add_empty();
                Ok(ThompsonRef { start: id, end: id })
            }
            HirKind::Literal(hir::Literal::Unicode(ch)) => {
                let mut buf = [0; 4];
                let it = ch
                    .encode_utf8(&mut buf)
                    .as_bytes()
                    .iter()
                    .map(|&b| Ok(self.compile_range(b, b)));
                self.compile_concat(it)
            }
            HirKind::Literal(hir::Literal::Byte(b)) => {
                Ok(self.compile_range(b, b))
            }
            HirKind::Class(hir::Class::Bytes(ref cls)) => {
                let it = cls
                    .iter()
                    .map(|rng| Ok(self.compile_range(rng.start(), rng.end())));
                self.compile_alternation(it)
            }
            HirKind::Class(hir::Class::Unicode(ref cls)) => {
                self.compile_unicode_class(cls)
            }
            HirKind::Repetition(ref rep) => {
                self.compile_repetition(rep)
            }
            HirKind::Group(ref group) => {
                self.compile(&*group.hir)
            }
            HirKind::Concat(ref exprs) => {
                self.compile_concat(exprs.iter().map(|e| self.compile(e)))
            }
            HirKind::Alternation(ref exprs) => {
                self.compile_alternation(exprs.iter().map(|e| self.compile(e)))
            }
            HirKind::Anchor(_) => {
                Err(Error::unsupported_anchor())
            }
            HirKind::WordBoundary(_) => {
                Err(Error::unsupported_word())
            }
        }
    }

    fn compile_concat<I>(
        &self,
        mut it: I,
    ) -> Result<ThompsonRef>
    where I: Iterator<Item=Result<ThompsonRef>>
    {
        let ThompsonRef { start, mut end } = match it.next() {
            Some(result) => result?,
            None => return Ok(self.compile_empty()),
        };
        for result in it {
            let compiled = result?;
            self.patch(end, compiled.start);
            end = compiled.end;
        }
        Ok(ThompsonRef { start, end })
    }

    fn compile_alternation<I>(
        &self,
        it: I,
    ) -> Result<ThompsonRef>
    where I: Iterator<Item=Result<ThompsonRef>>
    {
        let alternates = it.collect::<Result<Vec<ThompsonRef>>>()?;
        assert!(!alternates.is_empty(), "alternations must be non-empty");

        if alternates.len() == 1 {
            return Ok(alternates[0]);
        }

        let union = self.add_union();
        let empty = self.add_empty();
        for compiled in alternates {
            self.patch(union, compiled.start);
            self.patch(compiled.end, empty);
        }
        Ok(ThompsonRef { start: union, end: empty })
    }

    fn compile_repetition(
        &self,
        rep: &hir::Repetition,
    ) -> Result<ThompsonRef> {
        match rep.kind {
            hir::RepetitionKind::ZeroOrOne => {
                self.compile_zero_or_one(&rep.hir, rep.greedy)
            }
            hir::RepetitionKind::ZeroOrMore => {
                self.compile_at_least(&rep.hir, rep.greedy, 0)
            }
            hir::RepetitionKind::OneOrMore => {
                self.compile_at_least(&rep.hir, rep.greedy, 1)
            }
            hir::RepetitionKind::Range(ref rng) => {
                match *rng {
                    hir::RepetitionRange::Exactly(count) => {
                        self.compile_exactly(&rep.hir, count)
                    }
                    hir::RepetitionRange::AtLeast(m) => {
                        self.compile_at_least(&rep.hir, rep.greedy, m)
                    }
                    hir::RepetitionRange::Bounded(min, max) => {
                        self.compile_bounded(&rep.hir, rep.greedy, min, max)
                    }
                }
            }
        }
    }

    fn compile_bounded(
        &self,
        expr: &Hir,
        greedy: bool,
        min: u32,
        max: u32,
    ) -> Result<ThompsonRef> {
        let prefix = self.compile_exactly(expr, min)?;
        if min == max {
            return Ok(prefix);
        }

        let suffix = self.compile_concat(
            (min..max).map(|_| self.compile_zero_or_one(expr, greedy))
        )?;
        self.patch(prefix.end, suffix.start);
        Ok(ThompsonRef {
            start: prefix.start,
            end: suffix.end,
        })
    }

    fn compile_at_least(
        &self,
        expr: &Hir,
        greedy: bool,
        n: u32,
    ) -> Result<ThompsonRef> {
        if n == 0 {
            let union =
                if greedy {
                    self.add_union()
                } else {
                    self.add_reverse_union()
                };
            let compiled = self.compile(expr)?;
            self.patch(union, compiled.start);
            self.patch(compiled.end, union);
            Ok(ThompsonRef { start: union, end: union })
        } else if n == 1 {
            let compiled = self.compile(expr)?;
            let union =
                if greedy {
                    self.add_union()
                } else {
                    self.add_reverse_union()
                };
            self.patch(compiled.end, union);
            self.patch(union, compiled.start);
            Ok(ThompsonRef { start: compiled.start, end: union })
        } else {
            let prefix = self.compile_exactly(expr, n - 1)?;
            let last = self.compile(expr)?;
            let union =
                if greedy {
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

    fn compile_zero_or_one(
        &self,
        expr: &Hir,
        greedy: bool,
    ) -> Result<ThompsonRef> {
        let union =
            if greedy {
                self.add_union()
            } else {
                self.add_reverse_union()
            };
        let compiled = self.compile(expr)?;
        let empty = self.add_empty();
        self.patch(union, compiled.start);
        self.patch(union, empty);
        self.patch(compiled.end, empty);
        Ok(ThompsonRef { start: union, end: empty })
    }

    fn compile_exactly(&self, expr: &Hir, n: u32) -> Result<ThompsonRef> {
        let it = iter::repeat(())
            .take(n as usize)
            .map(|_| self.compile(expr));
        self.compile_concat(it)
    }

    fn compile_unicode_class(
        &self,
        cls: &hir::ClassUnicode,
    ) -> Result<ThompsonRef> {
        use utf8_ranges::Utf8Sequences;

        let it = cls
            .iter()
            .flat_map(|rng| Utf8Sequences::new(rng.start(), rng.end()))
            .map(|seq| {
                if self.reverse {
                    self.compile_concat(
                        seq.as_slice()
                            .iter()
                            .rev()
                            .map(|rng| {
                                Ok(self.compile_range(rng.start, rng.end))
                            })
                    )
                } else {
                    self.compile_concat(
                        seq.as_slice()
                            .iter()
                            .map(|rng| {
                                Ok(self.compile_range(rng.start, rng.end))
                            })
                    )
                }
            });
        self.compile_alternation(it)
    }

    fn compile_range(&self, start: u8, end: u8) -> ThompsonRef {
        let id = self.add_range(start, end);
        ThompsonRef { start: id, end: id }
    }

    fn compile_empty(&self) -> ThompsonRef {
        let id = self.add_empty();
        ThompsonRef { start: id, end: id }
    }

    fn compile_unanchored_prefix_valid_utf8(&self) -> Result<ThompsonRef> {
        self.compile(&Hir::repetition(hir::Repetition {
            kind: hir::RepetitionKind::ZeroOrMore,
            greedy: false,
            hir: Box::new(Hir::any(false)),
        }))
    }

    fn compile_unanchored_prefix_invalid_utf8(&self) -> Result<ThompsonRef> {
        self.compile(&Hir::repetition(hir::Repetition {
            kind: hir::RepetitionKind::ZeroOrMore,
            greedy: false,
            hir: Box::new(Hir::any(true)),
        }))
    }

    fn patch(&self, from: StateID, to: StateID) {
        match self.states.borrow_mut()[from] {
            BState::Empty { ref mut next } => {
                *next = to;
            }
            BState::Range { ref mut next, .. } => {
                *next = to;
            }
            BState::Union { ref mut alternates } => {
                alternates.push(to);
            }
            BState::UnionReverse { ref mut alternates } => {
                alternates.push(to);
            }
            BState::Match => {}
        }
    }

    fn add_empty(&self) -> StateID {
        let id = self.states.borrow().len();
        self.states.borrow_mut().push(BState::Empty { next: 0 });
        id
    }

    fn add_range(&self, start: u8, end: u8) -> StateID {
        let id = self.states.borrow().len();
        let state = BState::Range { start, end, next: 0 };
        self.states.borrow_mut().push(state);
        id
    }

    fn add_union(&self) -> StateID {
        let id = self.states.borrow().len();
        let state = BState::Union { alternates: vec![] };
        self.states.borrow_mut().push(state);
        id
    }

    fn add_reverse_union(&self) -> StateID {
        let id = self.states.borrow().len();
        let state = BState::UnionReverse { alternates: vec![] };
        self.states.borrow_mut().push(state);
        id
    }

    fn add_match(&self) -> StateID {
        let id = self.states.borrow().len();
        self.states.borrow_mut().push(BState::Match);
        id
    }
}

/// A byte class set keeps track of an *approximation* of equivalence classes
/// of bytes during NFA construction. That is, every byte in an equivalence
/// class cannot discriminate between a match and a non-match.
///
/// For example, in the regex `[ab]+`, the bytes `a` and `b` would be in the
/// same equivalence class because it never matters whether an `a` or a `b` is
/// seen, and no combination of `a`s and `b`s in the text can discriminate
/// a match.
///
/// Note though that this does not compute the minimal set of equivalence
/// classes. For example, in the regex `[ac]+`, both `a` and `c` are in the
/// same equivalence class for the same reason that `a` and `b` are in the
/// same equivalence class in the aforementioned regex. However, in this
/// implementation, `a` and `c` are put into distinct equivalence classes.
/// The reason for this is implementation complexity. In the future, we should
/// endeavor to compute the minimal equivalence classes since they can have a
/// rather large impact on the size of the DFA.
///
/// The representation here is 256 booleans, all initially set to false. Each
/// boolean maps to its corresponding byte based on position. A `true` value
/// indicates the end of an equivalence class, where its corresponding byte
/// and all of the bytes corresponding to all previous contiguous `false`
/// values are in the same equivalence class.
///
/// This particular representation only permits contiguous ranges of bytes to
/// be in the same equivalence class, which means that we can never discover
/// the true minimal set of equivalence classes.
#[derive(Debug)]
struct ByteClassSet(Vec<bool>);

impl ByteClassSet {
    /// Create a new set of byte classes where all bytes are part of the same
    /// equivalence class.
    fn new() -> Self {
        ByteClassSet(vec![false; 256])
    }

    /// Indicate the the range of byte given (inclusive) can discriminate a
    /// match between it and all other bytes outside of the range.
    fn set_range(&mut self, start: u8, end: u8) {
        debug_assert!(start <= end);
        if start > 0 {
            self.0[start as usize - 1] = true;
        }
        self.0[end as usize] = true;
    }

    /// Convert this boolean set to a map that maps all byte values to their
    /// corresponding equivalence class. The last mapping indicates the largest
    /// equivalence class identifier (which is never bigger than 255).
    fn byte_classes(&self) -> ByteClasses {
        let mut classes = ByteClasses::empty();
        let mut class = 0u8;
        let mut i = 0;
        loop {
            classes.set(i as u8, class as u8);
            if i >= 255 {
                break;
            }
            if self.0[i] {
                class = class.checked_add(1).unwrap();
            }
            i += 1;
        }
        classes
    }
}

impl fmt::Debug for NFA {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, state) in self.states.iter().enumerate() {
            let status = if i == self.start { '>' } else { ' ' };
            writeln!(f, "{}{:06X}: {:?}", status, i, state)?;
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use regex_syntax::ParserBuilder;
    use regex_syntax::hir::Hir;

    use super::{ByteClassSet, NFA, NFABuilder, State, StateID};

    fn parse(pattern: &str) -> Hir {
        ParserBuilder::new().build().parse(pattern).unwrap()
    }

    fn build(pattern: &str) -> NFA {
        NFABuilder::new().anchored(true).build(parse(pattern)).unwrap()
    }

    fn s_byte(byte: u8, next: StateID) -> State {
        State::Range { start: byte, end: byte, next }
    }

    fn s_range(start: u8, end: u8, next: StateID) -> State {
        State::Range { start, end, next }
    }

    fn s_union(alts: &[StateID]) -> State {
        State::Union { alternates: alts.to_vec() }
    }

    fn s_match() -> State {
        State::Match
    }

    #[test]
    fn errors() {
        // unsupported anchors
        assert!(NFABuilder::new().build(parse(r"^")).is_err());
        assert!(NFABuilder::new().build(parse(r"$")).is_err());
        assert!(NFABuilder::new().build(parse(r"\A")).is_err());
        assert!(NFABuilder::new().build(parse(r"\z")).is_err());

        // unsupported word boundaries
        assert!(NFABuilder::new().build(parse(r"\b")).is_err());
        assert!(NFABuilder::new().build(parse(r"\B")).is_err());
        assert!(NFABuilder::new().build(parse(r"(?-u)\b")).is_err());
    }

    // Test that building an unanchored NFA has an appropriate `.*?` prefix.
    #[test]
    fn compile_unanchored_prefix() {
        // When the machine can only match valid UTF-8.
        let nfa = NFABuilder::new()
            .anchored(false)
            .build(parse(r"a"))
            .unwrap();
        // There should be many states since the `.` in `.*?` matches any
        // Unicode scalar value.
        assert_eq!(31, nfa.len());
        assert_eq!(nfa.states[30], s_match());
        assert_eq!(nfa.states[29], s_byte(b'a', 30));

        // When the machine can match invalid UTF-8.
        let nfa = NFABuilder::new()
            .anchored(false)
            .allow_invalid_utf8(true)
            .build(parse(r"a"))
            .unwrap();
        assert_eq!(nfa.states, &[
            s_union(&[2, 1]),
            s_range(0, 255, 0),
            s_byte(b'a', 3),
            s_match(),
        ]);
    }

    #[test]
    fn compile_empty() {
        assert_eq!(build("").states, &[
            s_match(),
        ]);
    }

    #[test]
    fn compile_literal() {
        assert_eq!(build("a").states, &[
            s_byte(b'a', 1),
            s_match(),
        ]);
        assert_eq!(build("ab").states, &[
            s_byte(b'a', 1),
            s_byte(b'b', 2),
            s_match(),
        ]);
        assert_eq!(build("☃").states, &[
            s_byte(0xE2, 1),
            s_byte(0x98, 2),
            s_byte(0x83, 3),
            s_match(),
        ]);

        // Check that non-UTF-8 literals work.
        let hir = ParserBuilder::new()
            .allow_invalid_utf8(true)
            .build()
            .parse(r"(?-u)\xFF")
            .unwrap();
        let nfa = NFABuilder::new()
            .anchored(true)
            .allow_invalid_utf8(true)
            .build(hir)
            .unwrap();
        assert_eq!(nfa.states, &[
            s_byte(b'\xFF', 1),
            s_match(),
        ]);
    }

    #[test]
    fn compile_class() {
        assert_eq!(build(r"[a-z]").states, &[
            s_range(b'a', b'z', 1),
            s_match(),
        ]);
        assert_eq!(build(r"[x-za-c]").states, &[
            s_range(b'a', b'c', 3),
            s_range(b'x', b'z', 3),
            s_union(&[0, 1]),
            s_match(),
        ]);
        assert_eq!(build(r"[\u03B1-\u03B4]").states, &[
            s_byte(0xCE, 1),
            s_range(0xB1, 0xB4, 2),
            s_match(),
        ]);
        assert_eq!(build(r"[\u03B1-\u03B4\u{1F919}-\u{1F91E}]").states, &[
            s_byte(0xCE, 1),
            s_range(0xB1, 0xB4, 7),

            s_byte(0xF0, 3),
            s_byte(0x9F, 4),
            s_byte(0xA4, 5),
            s_range(0x99, 0x9E, 7),

            s_union(&[0, 2]),
            s_match(),
        ]);
    }

    #[test]
    fn compile_repetition() {
        assert_eq!(build(r"a?").states, &[
            s_union(&[1, 2]),
            s_byte(b'a', 2),
            s_match(),
        ]);
        assert_eq!(build(r"a??").states, &[
            s_union(&[2, 1]),
            s_byte(b'a', 2),
            s_match(),
        ]);
    }

    #[test]
    fn compile_group() {
        assert_eq!(build(r"ab+").states, &[
            s_byte(b'a', 1),
            s_byte(b'b', 2),
            s_union(&[1, 3]),
            s_match(),
        ]);
        assert_eq!(build(r"(ab)").states, &[
            s_byte(b'a', 1),
            s_byte(b'b', 2),
            s_match(),
        ]);
        assert_eq!(build(r"(ab)+").states, &[
            s_byte(b'a', 1),
            s_byte(b'b', 2),
            s_union(&[0, 3]),
            s_match(),
        ]);
    }

    #[test]
    fn compile_alternation() {
        assert_eq!(build(r"a|b").states, &[
            s_byte(b'a', 3),
            s_byte(b'b', 3),
            s_union(&[0, 1]),
            s_match(),
        ]);
    }

    #[test]
    fn byte_classes() {
        let mut set = ByteClassSet::new();
        set.set_range(b'a', b'z');

        let classes = set.byte_classes();
        assert_eq!(classes.get(0), 0);
        assert_eq!(classes.get(1), 0);
        assert_eq!(classes.get(2), 0);
        assert_eq!(classes.get(b'a' - 1), 0);
        assert_eq!(classes.get(b'a'), 1);
        assert_eq!(classes.get(b'm'), 1);
        assert_eq!(classes.get(b'z'), 1);
        assert_eq!(classes.get(b'z' + 1), 2);
        assert_eq!(classes.get(254), 2);
        assert_eq!(classes.get(255), 2);

        let mut set = ByteClassSet::new();
        set.set_range(0, 2);
        set.set_range(4, 6);
        let classes = set.byte_classes();
        assert_eq!(classes.get(0), 0);
        assert_eq!(classes.get(1), 0);
        assert_eq!(classes.get(2), 0);
        assert_eq!(classes.get(3), 1);
        assert_eq!(classes.get(4), 2);
        assert_eq!(classes.get(5), 2);
        assert_eq!(classes.get(6), 2);
        assert_eq!(classes.get(7), 3);
        assert_eq!(classes.get(255), 3);
    }

    #[test]
    fn full_byte_classes() {
        let mut set = ByteClassSet::new();
        for i in 0..256u16 {
            set.set_range(i as u8, i as u8);
        }
        assert_eq!(set.byte_classes().alphabet_len(), 256);
    }
}
