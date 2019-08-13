use std::cell::RefCell;
use std::cmp;
use std::fmt;
use std::mem;
use std::ops::RangeInclusive;
use std::u32;

use regex_syntax::utf8::Utf8Range;

type StateID = u32;

const FINAL: StateID = 0;
const ROOT: StateID = 1;

#[derive(Clone)]
pub struct RangeTrie {
    states: Vec<State>,
    free: Vec<State>,
    iter_stack: RefCell<Vec<NextIter>>,
    iter_ranges: RefCell<Vec<Utf8Range>>,
    dupe_stack: Vec<NextDupe>,
    insert_stack: Vec<NextInsert>,
}

#[derive(Clone)]
struct State {
    transitions: Vec<Transition>,
}

#[derive(Clone)]
struct Transition {
    range: Utf8Range,
    next_id: StateID,
}

#[derive(Clone, Debug)]
struct NextInsert {
    state_id: StateID,
    ranges: [Utf8Range; 4],
    len: u8,
}

impl NextInsert {
    fn new(state_id: StateID, ranges: &[Utf8Range]) -> NextInsert {
        let len = ranges.len();
        assert!(len <= 4);

        let mut tmp = [Utf8Range { start: 0, end: 0 }; 4];
        tmp[..len].copy_from_slice(ranges);
        NextInsert { state_id, ranges: tmp, len: len as u8 }
    }

    fn state_id(&self) -> StateID {
        self.state_id
    }

    fn ranges(&self) -> &[Utf8Range] {
        &self.ranges[..self.len as usize]
    }
}

#[derive(Clone, Debug)]
struct NextDupe {
    old_id: StateID,
    parent_id: StateID,
}

#[derive(Clone, Debug)]
struct NextIter {
    state_id: StateID,
    tidx: usize,
}

impl RangeTrie {
    pub fn new() -> RangeTrie {
        let mut trie = RangeTrie {
            states: vec![],
            free: vec![],
            iter_stack: RefCell::new(vec![]),
            iter_ranges: RefCell::new(vec![]),
            dupe_stack: vec![],
            insert_stack: vec![],
        };
        trie.clear();
        trie
    }

    pub fn clear(&mut self) {
        self.free.extend(self.states.drain(..));
        self.add_empty(); // final
        self.add_empty(); // root
    }

    pub fn iter<F: FnMut(&[Utf8Range])>(&self, mut f: F) {
        let mut stack = self.iter_stack.borrow_mut();
        stack.clear();
        let mut ranges = self.iter_ranges.borrow_mut();
        ranges.clear();

        stack.push(NextIter { state_id: ROOT, tidx: 0 });
        while let Some(NextIter { mut state_id, mut tidx }) = stack.pop() {
            loop {
                let state = &self.states[state_id as usize];
                if tidx >= state.transitions.len() {
                    ranges.pop();
                    break;
                }
                let t = &state.transitions[tidx];
                ranges.push(t.range);
                if t.next_id == FINAL {
                    f(&ranges);
                    ranges.pop();
                    tidx += 1;
                } else {
                    stack.push(NextIter { state_id, tidx: tidx + 1 });
                    state_id = t.next_id;
                    tidx = 0;
                }
            }
        }
    }

    pub fn insert(&mut self, ranges: &[Utf8Range]) {
        assert!(!ranges.is_empty());

        let mut stack = mem::replace(&mut self.insert_stack, vec![]);
        stack.clear();

        stack.push(NextInsert::new(ROOT, ranges));
        while let Some(next) = stack.pop() {
            let (state_id, ranges) = (next.state_id(), next.ranges());
            assert!(!ranges.is_empty());

            let mut new = ranges[0];
            let rest = &ranges[1..];

            let mut i = self.states[state_id as usize].find(new);
            if i == self.states[state_id as usize].transitions.len() {
                let next_id = if rest.is_empty() {
                    FINAL
                } else {
                    let next_id = self.add_empty();
                    stack.push(NextInsert::new(next_id, rest));
                    next_id
                };
                self.states[state_id as usize]
                    .transitions
                    .push(Transition { range: new, next_id });
                continue;
            }

            'OUTER: loop {
                let old =
                    self.states[state_id as usize].transitions[i].clone();
                let split = match Split::new(old.range, new) {
                    Some(split) => split,
                    None => {
                        let next_id = if rest.is_empty() {
                            FINAL
                        } else {
                            let next_id = self.add_empty();
                            stack.push(NextInsert::new(next_id, rest));
                            next_id
                        };
                        self.states[state_id as usize]
                            .transitions
                            .insert(i, Transition { range: new, next_id });
                        continue;
                    }
                };
                let splits = split.as_slice();
                let mut first = true;
                for (j, &srange) in splits.iter().enumerate() {
                    match srange {
                        SplitRange::Old(r) => {
                            let dup_id = self.duplicate(old.next_id);
                            if first {
                                self.states[state_id as usize].transitions
                                    [i] =
                                    Transition { range: r, next_id: dup_id };
                            } else {
                                self.states[state_id as usize]
                                    .transitions
                                    .insert(
                                        i,
                                        Transition {
                                            range: r,
                                            next_id: dup_id,
                                        },
                                    );
                            }
                        }
                        SplitRange::New(r) => {
                            {
                                let trans = &self.states[state_id as usize]
                                    .transitions;
                                if j + 1 == splits.len()
                                    && i < trans.len()
                                    && intersects(r, trans[i].range)
                                {
                                    new = r;
                                    continue 'OUTER;
                                }
                            }
                            let next_id = if rest.is_empty() {
                                FINAL
                            } else {
                                let next_id = self.add_empty();
                                stack.push(NextInsert::new(next_id, rest));
                                next_id
                            };
                            if first {
                                self.states[state_id as usize].transitions
                                    [i] = Transition { range: r, next_id };
                            } else {
                                self.states[state_id as usize]
                                    .transitions
                                    .insert(
                                        i,
                                        Transition { range: r, next_id },
                                    );
                            }
                        }
                        SplitRange::Both(r) => {
                            if !rest.is_empty() {
                                stack.push(NextInsert::new(old.next_id, rest));
                            }
                            if first {
                                self.states[state_id as usize].transitions
                                    [i] = Transition {
                                    range: r,
                                    next_id: old.next_id,
                                };
                            } else {
                                self.states[state_id as usize]
                                    .transitions
                                    .insert(
                                        i,
                                        Transition {
                                            range: r,
                                            next_id: old.next_id,
                                        },
                                    );
                            }
                        }
                    }
                    i += 1;
                    first = false;
                }
                break;
            }
        }
        self.insert_stack = stack;
    }

    pub fn add_empty(&mut self) -> StateID {
        if self.states.len() as u64 > u32::MAX as u64 {
            // This generally should not happen since a range trie is only
            // ever used to compile a single sequence of Unicode scalar values.
            // If we ever got to this point, we would, at *minimum*, be using
            // 96GB in just the range trie alone.
            panic!("too many sequences added to range trie");
        }
        let id = self.states.len() as StateID;
        if let Some(mut state) = self.free.pop() {
            state.clear();
            self.states.push(state);
        } else {
            self.states.push(State { transitions: vec![] });
        }
        id
    }

    fn duplicate(&mut self, state_id: StateID) -> StateID {
        if state_id == FINAL {
            return FINAL;
        }
        let new_id = self.add_empty();

        let mut stack = mem::replace(&mut self.dupe_stack, vec![]);
        stack.clear();

        stack.push(NextDupe { old_id: state_id, parent_id: new_id });
        while let Some(NextDupe { old_id, parent_id }) = stack.pop() {
            for i in 0..self.states[old_id as usize].transitions.len() {
                let t = self.states[old_id as usize].transitions[i].clone();
                if t.next_id == FINAL {
                    self.states[parent_id as usize]
                        .transitions
                        .push(Transition { range: t.range, next_id: FINAL });
                    continue;
                }
                let new_id = self.add_empty();
                self.states[parent_id as usize]
                    .transitions
                    .push(Transition { range: t.range, next_id: new_id });
                stack.push(NextDupe { old_id: t.next_id, parent_id: new_id });
            }
        }
        self.dupe_stack = stack;
        new_id
    }
}

impl State {
    fn find(&self, range: Utf8Range) -> usize {
        // Benchmarks suggest that binary search is just a bit faster than
        // straight linear search. Specifically when using the debug tool:
        //
        //   hyperfine "regex-automata-debug debug -acqr '\w{40} ecurB'"
        binary_search(&self.transitions, |t| {
            range.end < t.range.start || !(t.range.end < range.start)
        })
    }

    fn clear(&mut self) {
        self.transitions.clear();
    }
}

fn binary_search<T, F>(xs: &[T], mut pred: F) -> usize
where
    F: FnMut(&T) -> bool,
{
    let (mut left, mut right) = (0, xs.len());
    while left < right {
        let mid = (left + right) / 2;
        if pred(&xs[mid]) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    left
}

// given [a,b] and [x,y], where a <= b, x <= y, b < 256 and y < 256, we define
// the follow distinct relationships where at least one must apply. The order
// of these matters, since multiple can match. The first to match applies.
//
//   1. b < x <=> [a,b] < [x,y]
//   2. y < a <=> [x,y] < [a,b]
//
// In the case of (1) and (2), these are the only cases where there is no
// overlap. Or otherwise, the intersection of [a,b] and [x,y] is empty. In
// order to compute the intersection, one can do [max(a,x), min(b,y)]. The
// intersection in all of the following cases is non-empty.
//
//    3. a = x && b = y <=> [a,b] == [x,y]
//    4. a = x && b < y <=> [x,y] right-extends [a,b]
//    5. b = y && a > x <=> [x,y] left-extends [a,b]
//    6. x = a && y < b <=> [a,b] right-extends [x,y]
//    7. y = b && x > a <=> [a,b] left-extends [x,y]
//    8. a > x && b < y <=> [x,y] covers [a,b]
//    9. x > a && y < b <=> [a,b] covers [x,y]
//   10. b = x && a < y <=> [a,b] is left-adjacent to [x,y]
//   11. y = a && x < b <=> [x,y] is left-adjacent to [a,b]
//   12. b > x && b < y <=> [a,b] left-overlaps [x,y]
//   13. y > a && y < b <=> [x,y] left-overlaps [a,b]
//
// In cases 3-13, we can form rules that partition the ranges into a
// non-overlapping ordered sequence of ranges:
//
//    3. [a,b]
//    4. [a,b], [b+1,y]
//    5. [x,a-1], [a,b]
//    6. [x,y], [y+1,b]
//    7. [a,x-1], [x,y]
//    8. [x,a-1], [a,b], [b+1,y]
//    9. [a,x-1], [x,y], [y+1,b]
//   10. [a,b-1], [b,b], [b+1,y]
//   11. [x,y-1], [y,y], [y+1,b]
//   12. [a,x-1], [x,b], [b+1,y]
//   13. [x,a-1], [a,y], [y+1,b]
//
// BREADCRUMBS:
//
// Use the above rules to define a "split" operation, e.g., given [a,b] how
// does [x,y] "split" [a,b]?
//
// If there's no intersection then the split operation is not defined.

#[derive(Clone, Debug, Eq, PartialEq)]
struct Split {
    partitions: [SplitRange; 3],
    len: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SplitRange {
    Old(Utf8Range),
    New(Utf8Range),
    Both(Utf8Range),
}

impl Split {
    fn new(o: Utf8Range, n: Utf8Range) -> Option<Split> {
        let range = |r: RangeInclusive<u8>| Utf8Range {
            start: *r.start(),
            end: *r.end(),
        };
        let old = |r| SplitRange::Old(range(r));
        let new = |r| SplitRange::New(range(r));
        let both = |r| SplitRange::Both(range(r));

        // Use same names as the comment above to make it easier to compare.
        let (a, b, x, y) = (o.start, o.end, n.start, n.end);

        if b < x || y < a {
            // case 1, case 2
            None
        } else if a == x && b == y {
            // case 3
            Some(Split::parts1(both(a..=b)))
        } else if a == x && b < y {
            // case 4
            Some(Split::parts2(both(a..=b), new(b + 1..=y)))
        } else if b == y && a > x {
            // case 5
            Some(Split::parts2(new(x..=a - 1), both(a..=b)))
        } else if x == a && y < b {
            // case 6
            Some(Split::parts2(both(x..=y), old(y + 1..=b)))
        } else if y == b && x > a {
            // case 7
            Some(Split::parts2(old(a..=x - 1), both(x..=y)))
        } else if a > x && b < y {
            // case 8
            Some(Split::parts3(new(x..=a - 1), both(a..=b), new(b + 1..=y)))
        } else if x > a && y < b {
            // case 9
            Some(Split::parts3(old(a..=x - 1), both(x..=y), old(y + 1..=b)))
        } else if b == x && a < y {
            // case 10
            Some(Split::parts3(old(a..=b - 1), both(b..=b), new(b + 1..=y)))
        } else if y == a && x < b {
            // case 11
            Some(Split::parts3(new(x..=y - 1), both(y..=y), old(y + 1..=b)))
        } else if b > x && b < y {
            // case 12
            Some(Split::parts3(old(a..=x - 1), both(x..=b), new(b + 1..=y)))
        } else if y > a && y < b {
            // case 13
            Some(Split::parts3(new(x..=a - 1), both(a..=y), old(y + 1..=b)))
        } else {
            unreachable!()
        }
    }

    fn parts1(r1: SplitRange) -> Split {
        // This value doesn't matter since it is never accessed.
        let nada = SplitRange::Old(Utf8Range { start: 0, end: 0 });
        Split { partitions: [r1, nada, nada], len: 1 }
    }

    fn parts2(r1: SplitRange, r2: SplitRange) -> Split {
        // This value doesn't matter since it is never accessed.
        let nada = SplitRange::Old(Utf8Range { start: 0, end: 0 });
        Split { partitions: [r1, r2, nada], len: 2 }
    }

    fn parts3(r1: SplitRange, r2: SplitRange, r3: SplitRange) -> Split {
        Split { partitions: [r1, r2, r3], len: 3 }
    }

    fn as_slice(&self) -> &[SplitRange] {
        &self.partitions[..self.len]
    }
}

impl fmt::Debug for RangeTrie {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "")?;
        for (i, state) in self.states.iter().enumerate() {
            let status = if i == FINAL as usize { '*' } else { ' ' };
            writeln!(f, "{}{:06}: {:?}", status, i, state)?;
        }
        Ok(())
    }
}

impl fmt::Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let rs = self
            .transitions
            .iter()
            .map(|t| format!("{:?}", t))
            .collect::<Vec<String>>()
            .join(", ");
        write!(f, "{}", rs)
    }
}

impl fmt::Debug for Transition {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.range.start == self.range.end {
            write!(f, "{:02X} => {:02X}", self.range.start, self.next_id)
        } else {
            write!(
                f,
                "{:02X}-{:02X} => {:02X}",
                self.range.start, self.range.end, self.next_id
            )
        }
    }
}

fn intersect(r1: Utf8Range, r2: Utf8Range) -> Option<Utf8Range> {
    let start = cmp::max(r1.start, r2.start);
    let end = cmp::min(r1.end, r2.end);
    if start <= end {
        Some(Utf8Range { start, end })
    } else {
        None
    }
}

fn intersects(r1: Utf8Range, r2: Utf8Range) -> bool {
    intersect(r1, r2).is_some()
}

#[cfg(test)]
mod tests {
    use std::ops::RangeInclusive;

    use regex_syntax::utf8::Utf8Range;

    use super::*;

    fn r(range: RangeInclusive<u8>) -> Utf8Range {
        Utf8Range { start: *range.start(), end: *range.end() }
    }

    fn split_maybe(
        old: RangeInclusive<u8>,
        new: RangeInclusive<u8>,
    ) -> Option<Split> {
        Split::new(r(old), r(new))
    }

    fn split(
        old: RangeInclusive<u8>,
        new: RangeInclusive<u8>,
    ) -> Vec<SplitRange> {
        split_maybe(old, new).unwrap().as_slice().to_vec()
    }

    #[test]
    fn intersection() {
        assert_eq!(Some(r(0..=0)), intersect(r(0..=0), r(0..=0)));
        assert_eq!(Some(r(1..=1)), intersect(r(1..=1), r(1..=1)));
        assert_eq!(Some(r(5..=10)), intersect(r(5..=10), r(5..=10)));

        assert_eq!(Some(r(0..=0)), intersect(r(0..=0), r(0..=1)));
        assert_eq!(Some(r(0..=0)), intersect(r(0..=0), r(0..=2)));

        assert_eq!(None, intersect(r(0..=0), r(1..=1)));
        assert_eq!(None, intersect(r(1..=1), r(0..=0)));
    }

    #[test]
    fn no_splits() {
        // case 1
        assert_eq!(None, split_maybe(0..=1, 2..=3));
        // case 2
        assert_eq!(None, split_maybe(2..=3, 0..=1));
    }

    #[test]
    fn splits() {
        let range = |r: RangeInclusive<u8>| Utf8Range {
            start: *r.start(),
            end: *r.end(),
        };
        let old = |r| SplitRange::Old(range(r));
        let new = |r| SplitRange::New(range(r));
        let both = |r| SplitRange::Both(range(r));

        // case 3
        assert_eq!(split(0..=0, 0..=0), vec![both(0..=0)]);
        assert_eq!(split(9..=9, 9..=9), vec![both(9..=9)]);

        // case 4
        assert_eq!(split(0..=5, 0..=6), vec![both(0..=5), new(6..=6)]);
        assert_eq!(split(0..=5, 0..=8), vec![both(0..=5), new(6..=8)]);
        assert_eq!(split(5..=5, 5..=8), vec![both(5..=5), new(6..=8)]);

        // case 5
        assert_eq!(split(1..=5, 0..=5), vec![new(0..=0), both(1..=5)]);
        assert_eq!(split(3..=5, 0..=5), vec![new(0..=2), both(3..=5)]);
        assert_eq!(split(5..=5, 0..=5), vec![new(0..=4), both(5..=5)]);

        // case 6
        assert_eq!(split(0..=6, 0..=5), vec![both(0..=5), old(6..=6)]);
        assert_eq!(split(0..=8, 0..=5), vec![both(0..=5), old(6..=8)]);
        assert_eq!(split(5..=8, 5..=5), vec![both(5..=5), old(6..=8)]);

        // case 7
        assert_eq!(split(0..=5, 1..=5), vec![old(0..=0), both(1..=5)]);
        assert_eq!(split(0..=5, 3..=5), vec![old(0..=2), both(3..=5)]);
        assert_eq!(split(0..=5, 5..=5), vec![old(0..=4), both(5..=5)]);

        // case 8
        assert_eq!(
            split(3..=6, 2..=7),
            vec![new(2..=2), both(3..=6), new(7..=7)],
        );
        assert_eq!(
            split(3..=6, 1..=8),
            vec![new(1..=2), both(3..=6), new(7..=8)],
        );

        // case 9
        assert_eq!(
            split(2..=7, 3..=6),
            vec![old(2..=2), both(3..=6), old(7..=7)],
        );
        assert_eq!(
            split(1..=8, 3..=6),
            vec![old(1..=2), both(3..=6), old(7..=8)],
        );

        // case 10
        assert_eq!(
            split(3..=6, 6..=7),
            vec![old(3..=5), both(6..=6), new(7..=7)],
        );
        assert_eq!(
            split(3..=6, 6..=8),
            vec![old(3..=5), both(6..=6), new(7..=8)],
        );
        assert_eq!(
            split(5..=6, 6..=7),
            vec![old(5..=5), both(6..=6), new(7..=7)],
        );

        // case 11
        assert_eq!(
            split(6..=7, 3..=6),
            vec![new(3..=5), both(6..=6), old(7..=7)],
        );
        assert_eq!(
            split(6..=8, 3..=6),
            vec![new(3..=5), both(6..=6), old(7..=8)],
        );
        assert_eq!(
            split(6..=7, 5..=6),
            vec![new(5..=5), both(6..=6), old(7..=7)],
        );

        // case 12
        assert_eq!(
            split(3..=7, 5..=9),
            vec![old(3..=4), both(5..=7), new(8..=9)],
        );
        assert_eq!(
            split(3..=5, 4..=6),
            vec![old(3..=3), both(4..=5), new(6..=6)],
        );

        // case 13
        assert_eq!(
            split(5..=9, 3..=7),
            vec![new(3..=4), both(5..=7), old(8..=9)],
        );
        assert_eq!(
            split(4..=6, 3..=5),
            vec![new(3..=3), both(4..=5), old(6..=6)],
        );
    }

    #[test]
    fn scratch_explicit() {
        use regex_syntax::utf8::Utf8Sequences;

        // let (s, e) = ('\u{0800}', '\u{1000}');
        // let (s, e) = ('\u{0}', '\u{FFFF}');
        let (s, e) = ('\u{0}', '\u{10FFFF}');

        let mut trie = RangeTrie::new();
        for seq in Utf8Sequences::new(s, e) {
            let mut seq = seq.as_slice().to_vec();
            seq.reverse();
            trie.insert(&seq);
        }
        trie.iter(|seq| {
            eprintln!("{:?}", seq);
        });
        dbg!(trie);
    }

    #[test]
    fn scratch_class() {
        use regex_syntax::utf8::Utf8Sequences;
        use regex_syntax::{hir, Parser};

        let pattern = r"\p{any}";

        let hir = Parser::new().parse(&format!("[{}]", pattern)).unwrap();
        let cls = match hir.into_kind() {
            hir::HirKind::Class(hir::Class::Unicode(cls)) => cls,
            _ => unreachable!(),
        };

        let mut trie = RangeTrie::new();
        for range in cls.iter() {
            for seq in Utf8Sequences::new(range.start(), range.end()) {
                let mut seq = seq.as_slice().to_vec();
                seq.reverse();
                trie.insert(&seq);
            }
        }
        trie.iter(|seq| {
            eprintln!("{:?}", seq);
        });
        // dbg!(trie);
    }

    // BREADCRUMBS: Quickcheck. Then go through examples.
}
