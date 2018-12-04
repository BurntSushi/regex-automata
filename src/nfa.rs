use std::cell::RefCell;
use std::iter;

use regex_syntax::Parser;
use regex_syntax::hir::{self, Hir, HirKind};

use error::{Error, Result};

#[derive(Debug)]
pub struct NFA {
    pub states: RefCell<Vec<State>>,
}

pub type StateID = usize;

#[derive(Debug)]
pub enum State {
    Empty { next: StateID },
    Range { start: u8, end: u8, next: StateID },
    Union { alternates: Vec<StateID>, reverse: bool },
    Match,
}

#[derive(Debug)]
struct ThompsonRef {
    start: StateID,
    end: StateID,
}

impl NFA {
    pub fn empty() -> NFA {
        NFA { states: RefCell::new(vec![]) }
    }

    pub fn from_pattern(pattern: &str) -> Result<NFA> {
        NFA::from_hir(&Parser::new().parse(pattern).map_err(Error::syntax)?)
    }

    pub fn from_hir(expr: &Hir) -> Result<NFA> {
        let nfa = NFA::empty();
        let start = nfa.add_empty();
        let compiled = nfa.compile(expr)?;
        let match_id = nfa.add_match();
        nfa.patch(start, compiled.start);
        nfa.patch(compiled.end, match_id);
        Ok(nfa)
    }

    fn compile(&self, expr: &Hir) -> Result<ThompsonRef> {
        match expr.kind() {
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
                Ok(self.compile_range(*b, *b))
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
        let union = self.add_union();

        let mut alternate_ends = vec![];
        for result in it {
            let compiled = result?;
            self.patch(union, compiled.start);
            alternate_ends.push(compiled.end);
        }
        assert!(!alternate_ends.is_empty(), "alternations must be non-empty");

        let empty = self.add_empty();
        for id in alternate_ends {
            self.patch(id, empty);
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

    fn patch(&self, from: StateID, to: StateID) {
        match self.states.borrow_mut()[from] {
            State::Empty { ref mut next } => {
                *next = to;
            }
            State::Range { ref mut next, .. } => {
                *next = to;
            }
            State::Union { ref mut alternates, reverse: false } => {
                alternates.push(to);
            }
            State::Union { ref mut alternates, reverse: true } => {
                alternates.insert(0, to);
            }
            State::Match => {}
        }
    }

    fn add_empty(&self) -> StateID {
        let id = self.states.borrow().len();
        self.states.borrow_mut().push(State::Empty { next: 0 });
        id
    }

    fn add_range(&self, start: u8, end: u8) -> StateID {
        let id = self.states.borrow().len();
        let state = State::Range { start, end, next: 0 };
        self.states.borrow_mut().push(state);
        id
    }

    fn add_union(&self) -> StateID {
        let id = self.states.borrow().len();
        let state = State::Union { alternates: vec![], reverse: false };
        self.states.borrow_mut().push(state);
        id
    }

    fn add_reverse_union(&self) -> StateID {
        let id = self.states.borrow().len();
        let state = State::Union { alternates: vec![], reverse: true };
        self.states.borrow_mut().push(state);
        id
    }

    fn add_match(&self) -> StateID {
        let id = self.states.borrow().len();
        self.states.borrow_mut().push(State::Match);
        id
    }
}

impl State {
    pub fn is_epsilon(&self) -> bool {
        match *self {
            State::Range { .. } | State::Match => false,
            State::Empty { .. } | State::Union { .. } => true,
        }
    }
}
