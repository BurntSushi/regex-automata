use std::cell::RefCell;
use std::fmt;
use std::iter;

use regex_syntax::ParserBuilder;
use regex_syntax::hir::{self, Hir, HirKind};

use error::{Error, Result};

pub type StateID = usize;

#[derive(Clone)]
pub struct NFA {
    start: StateID,
    states: Vec<State>,
    byte_classes: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum State {
    Range { start: u8, end: u8, next: StateID },
    Union { alternates: Vec<StateID> },
    Match,
}

impl NFA {
    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn start(&self) -> StateID {
        self.start
    }

    pub fn state(&self, id: StateID) -> &State {
        &self.states[id]
    }

    pub fn byte_classes(&self) -> &[u8] {
        &self.byte_classes
    }
}

impl State {
    pub fn is_epsilon(&self) -> bool {
        match *self {
            State::Range { .. } | State::Match => false,
            State::Union { .. } => true,
        }
    }

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

#[derive(Clone, Debug)]
pub struct NFABuilder {
    anchored: bool,
    allow_invalid_utf8: bool,
}

impl NFABuilder {
    pub fn new() -> NFABuilder {
        NFABuilder {
            anchored: false,
            allow_invalid_utf8: false,
        }
    }

    pub fn build(&self, expr: &Hir) -> Result<NFA> {
        let compiler = NFACompiler::new();
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
        let compiled = compiler.compile(expr)?;
        let match_id = compiler.add_match();
        compiler.patch(start, compiled.start);
        compiler.patch(compiled.end, match_id);
        Ok(compiler.to_nfa())
    }

    pub fn anchored(&mut self, yes: bool) -> &mut NFABuilder {
        self.anchored = yes;
        self
    }

    pub fn allow_invalid_utf8(&mut self, yes: bool) -> &mut NFABuilder {
        self.allow_invalid_utf8 = yes;
        self
    }
}

#[derive(Debug)]
struct NFACompiler {
    states: RefCell<Vec<BState>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum BState {
    Empty { next: StateID },
    Range { start: u8, end: u8, next: StateID },
    Union { alternates: Vec<StateID> },
    UnionReverse { alternates: Vec<StateID> },
    Match,
}

#[derive(Clone, Copy, Debug)]
struct ThompsonRef {
    start: StateID,
    end: StateID,
}

impl NFACompiler {
    fn new() -> NFACompiler {
        NFACompiler { states: RefCell::new(vec![]) }
    }

    fn to_nfa(&self) -> NFA {
        let bstates = self.states.borrow();
        let mut states = vec![];
        let mut remap = vec![0; bstates.len()];
        let mut empties = vec![];
        let mut byteset = ByteClassSet::new();
        for (id, bstate) in bstates.iter().enumerate() {
            match *bstate {
                BState::Empty { mut next } => {
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
        NFA { start: remap[0], states, byte_classes }
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
                let it = seq.as_slice()
                    .iter()
                    .map(|rng| Ok(self.compile_range(rng.start, rng.end)));
                self.compile_concat(it)
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

impl BState {
    fn is_empty(&self) -> bool {
        match *self {
            BState::Empty { .. } => true,
            _ => false,
        }
    }
}

#[derive(Debug)]
struct ByteClassSet(Vec<bool>);

impl ByteClassSet {
    fn new() -> Self {
        ByteClassSet(vec![false; 256])
    }

    fn set_range(&mut self, start: u8, end: u8) {
        debug_assert!(start <= end);
        if start > 0 {
            self.0[start as usize - 1] = true;
        }
        self.0[end as usize] = true;
    }

    fn byte_classes(&self) -> Vec<u8> {
        let mut byte_classes = vec![0; 256];
        let mut class = 0u8;
        let mut i = 0;
        loop {
            byte_classes[i] = class as u8;
            if i >= 255 {
                break;
            }
            if self.0[i] {
                class = class.checked_add(1).unwrap();
            }
            i += 1;
        }
        byte_classes
    }
}

impl fmt::Debug for NFA {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, state) in self.states.iter().enumerate() {
            let status = if i == self.start { '>' } else { ' ' };
            writeln!(f, "{}{:06X}: {:X?}", status, i, state)?;
        }
        Ok(())
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
        NFABuilder::new().anchored(true).build(&parse(pattern)).unwrap()
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
        assert!(NFABuilder::new().build(&parse(r"^")).is_err());
        assert!(NFABuilder::new().build(&parse(r"$")).is_err());
        assert!(NFABuilder::new().build(&parse(r"\A")).is_err());
        assert!(NFABuilder::new().build(&parse(r"\z")).is_err());

        // unsupported word boundaries
        assert!(NFABuilder::new().build(&parse(r"\b")).is_err());
        assert!(NFABuilder::new().build(&parse(r"\B")).is_err());
        assert!(NFABuilder::new().build(&parse(r"(?-u)\b")).is_err());
    }

    // Test that building an unanchored NFA has an appropriate `.*?` prefix.
    #[test]
    fn compile_unanchored_prefix() {
        // When the machine can only match valid UTF-8.
        let nfa = NFABuilder::new()
            .anchored(false)
            .build(&parse(r"a"))
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
            .build(&parse(r"a"))
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
            .build(&hir)
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
    fn byte_classes() {
        let mut set = ByteClassSet::new();
        set.set_range(b'a', b'z');

        let classes = set.byte_classes();
        assert_eq!(classes[0], 0);
        assert_eq!(classes[1], 0);
        assert_eq!(classes[2], 0);
        assert_eq!(classes[b'a' as usize - 1], 0);
        assert_eq!(classes[b'a' as usize], 1);
        assert_eq!(classes[b'm' as usize], 1);
        assert_eq!(classes[b'z' as usize], 1);
        assert_eq!(classes[b'z' as usize + 1], 2);
        assert_eq!(classes[254], 2);
        assert_eq!(classes[255], 2);

        let mut set = ByteClassSet::new();
        set.set_range(0, 2);
        set.set_range(4, 6);
        let classes = set.byte_classes();
        assert_eq!(classes[0], 0);
        assert_eq!(classes[1], 0);
        assert_eq!(classes[2], 0);
        assert_eq!(classes[3], 1);
        assert_eq!(classes[4], 2);
        assert_eq!(classes[5], 2);
        assert_eq!(classes[6], 2);
        assert_eq!(classes[7], 3);
        assert_eq!(classes[255], 3);
    }

    #[test]
    fn full_byte_classes() {
        let mut set = ByteClassSet::new();
        for i in 0..256u16 {
            set.set_range(i as u8, i as u8);
        }
        assert_eq!(set.byte_classes().len(), 256);
    }
}
