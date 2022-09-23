use crate::{
    nfa::thompson::{self, compiler::ThompsonRef, Builder, Error},
    util::primitives::{IteratorIndexExt, StateID},
};

#[derive(Clone)]
pub(crate) struct LiteralTrie {
    states: Vec<State>,
    reverse: bool,
}

impl LiteralTrie {
    pub(crate) fn forward() -> LiteralTrie {
        let mut trie = LiteralTrie { states: vec![], reverse: false };
        // OK because we always have space for at least one state.
        assert_eq!(0, trie.add_state().unwrap().as_usize());
        trie
    }

    pub(crate) fn reverse() -> LiteralTrie {
        let mut trie = LiteralTrie { states: vec![], reverse: true };
        // OK because we always have space for at least one state.
        assert_eq!(0, trie.add_state().unwrap().as_usize());
        trie
    }

    pub(crate) fn add(&mut self, bytes: &[u8]) -> Result<(), Error> {
        // DERP: Add bytes in reverse for compiling reverse NFA.
        let mut prev = StateID::ZERO;
        let mut it = bytes.iter().copied();
        loop {
            let byte =
                match if self.reverse { it.next_back() } else { it.next() } {
                    None => break,
                    Some(byte) => byte,
                };
            let next = match self.states[prev].next_state(byte) {
                None => {
                    let next = self.add_state()?;
                    self.states[prev].set_next_state(byte, next);
                    next
                }
                Some(sid) => sid,
            };
            prev = next;
        }
        self.states[prev].add_match();
        Ok(())
    }

    fn add_state(&mut self) -> Result<StateID, Error> {
        let id = StateID::new(self.states.len())
            .map_err(|_| Error::too_many_states(self.states.len()))?;
        self.states.push(State::default());
        Ok(id)
    }

    pub(crate) fn compile(
        &self,
        nfac: &mut Builder,
    ) -> Result<ThompsonRef, Error> {
        let out = nfac.add_empty()?;
        let start = self.compile_state(nfac, out, StateID::ZERO)?;
        Ok(ThompsonRef { start, end: out })
    }

    fn compile_state(
        &self,
        nfac: &mut Builder,
        out: StateID,
        sid: StateID,
    ) -> Result<StateID, Error> {
        let union = nfac.add_union(vec![])?;
        for (i, chunk) in self.states[sid].chunks().enumerate() {
            if i > 0 {
                nfac.patch(union, out)?;
            }
            if chunk.is_empty() {
                continue;
            }
            let mut sparse = vec![];
            for t in chunk.iter() {
                let next = self.compile_state(nfac, out, t.next)?;
                sparse.push(thompson::Transition {
                    start: t.byte,
                    end: t.byte,
                    next,
                });
            }
            let chunk_id = nfac.add_sparse(sparse)?;
            nfac.patch(union, chunk_id)?;
        }
        Ok(union)
    }

    fn to_hir(&self) -> regex_syntax::hir::Hir {
        self.to_hir_state(StateID::ZERO)
    }

    fn to_hir_state(&self, sid: StateID) -> regex_syntax::hir::Hir {
        use regex_syntax::hir::Hir;

        let mut alt = vec![];
        for (i, chunk) in self.states[sid].chunks().enumerate() {
            if i > 0 {
                alt.push(Hir::empty());
            }
            if chunk.is_empty() {
                continue;
            }
            let mut chunk_alt = vec![];
            for t in chunk.iter() {
                chunk_alt.push(Hir::concat(vec![
                    Hir::literal(vec![t.byte]),
                    self.to_hir_state(t.next),
                ]));
            }
            alt.push(Hir::alternation(chunk_alt));
        }
        Hir::alternation(alt)
    }

    /*
    fn to_syntax(&self) -> String {
        self.to_syntax_state(StateID::ZERO)
    }

    fn to_syntax_state(&self, sid: StateID) -> String {
        let mut alt = vec![];
        for (i, chunk) in self.states[sid].chunks().enumerate() {
            if i > 0 {
                alt.push("(?:)".to_string());
            }
            if chunk.is_empty() {
                continue;
            }
            let mut chunk_alt = vec![];
            for t in chunk.iter() {
                let mut branch =
                    format!("(?:{})", self.to_syntax_state(t.next));
                branch.insert(0, char::from(t.byte));
                chunk_alt.push(branch);
            }
            alt.push(chunk_alt.join("|"));
        }
        alt.join("|")
    }
    */
}

impl core::fmt::Debug for LiteralTrie {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        writeln!(f, "LiteralTrie(")?;
        for (sid, state) in self.states.iter().with_state_ids() {
            writeln!(f, "{:06?}: {:?}", sid.as_usize(), state)?;
        }
        writeln!(f, ")")?;
        Ok(())
    }
}

#[derive(Clone, Default)]
struct State {
    transitions: Vec<Transition>,
    chunks: Vec<(usize, usize)>,
}

impl State {
    fn next_state(&self, byte: u8) -> Option<StateID> {
        for &t in self.active_chunk().iter() {
            if t.byte == byte {
                return Some(t.next);
            }
        }
        None
    }

    fn set_next_state(&mut self, byte: u8, next: StateID) {
        let chunk_start = self.active_chunk_start();
        let trans = &self.transitions[chunk_start..];
        let t = Transition { byte, next };
        match trans.binary_search_by_key(&byte, |&t| t.byte) {
            Ok(i) => self.transitions[chunk_start + i] = t,
            Err(i) => self.transitions.insert(chunk_start + i, t),
        }
    }

    fn add_match(&mut self) {
        let chunk_start = self.active_chunk_start();
        let chunk_end = self.transitions.len();
        self.chunks.push((chunk_start, chunk_end));
    }

    fn chunk_len(&self) -> usize {
        // +1 is for the active chunk.
        // Number of matches always equals self.chunk_len()-1.
        self.chunks.len() + 1
    }

    fn chunks(&self) -> impl Iterator<Item = &[Transition]> {
        let last = core::iter::once(self.active_chunk());
        self.chunks.iter().map(|&(s, e)| &self.transitions[s..e]).chain(last)
    }

    fn active_chunk(&self) -> &[Transition] {
        &self.transitions[self.active_chunk_start()..]
    }

    fn active_chunk_start(&self) -> usize {
        match self.chunks.last() {
            None => 0,
            Some(&(_, end)) => end,
        }
    }
}

impl core::fmt::Debug for State {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        let mut spacing = " ";
        for (i, chunk) in self.chunks().enumerate() {
            if i > 0 {
                write!(f, "{}MATCH", spacing)?;
            }
            spacing = "";
            for (j, t) in chunk.iter().enumerate() {
                spacing = " ";
                if j == 0 && i > 0 {
                    write!(f, " ")?;
                } else if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:?}", t)?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Transition {
    byte: u8,
    next: StateID,
}

impl core::fmt::Debug for Transition {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(
            f,
            "{:?} => {}",
            crate::util::escape::DebugByte(self.byte),
            self.next.as_usize()
        )
    }
}

// BREADCRUMBS: This whole fucking idea is completely bunk. My idea was
// to build a trie, and when leftmost-first was used, I would simply omit
// adding literals that had a prefix already in the trie. This works fine
// if the alternation of literals is the entire pattern. But for example,
// '\b(sam|samwise)\b' matches 'samwise' in 'samwise' and not 'sam'. So
// removing 'samwise' would be incorrect.
//
// I went down this path because I perceived it to be easier and because it
// would work in most cases... But it's wrong wrong wrong.
//
// Instead we're going to have to solve the more general problem if we want
// this type of optimization to work. So 'sam|samwise' would get rewritten as
// 'sam(?:|wise)', which is equivalent and achieves our goal of not necessarily
// having one giant 'union' NFA state.
//
// So how to do it? I think the fundamental insight is that so long as we're
// only dealing with literals, the main thing you can't do is reorder match
// states. That is, you cannot move a literal that came after a prefix of
// itself to before that prefix.
//
// I still think we should explore using a trie for this. It *feels* like the
// right data structure. Otherwise, I think this problem breaks down into
// recursion, which I'd like to avoid. The main problem with the trie approach
// is that ordering gets lost. So we need to figure out some way to prevent
// re-orderings while utilizing a trie to write our alternation of literals
// more efficiently.
//
// OK, so a trie can't really work on its own... But maybe we can augment it a
// bit and use "marks" to indicate boundaries that can't be crossed?
//
// So instead of having a traditional finite state machine, we just keep a
// sequence of transitions and these transitions are in the same order as the
// literals seen. Here's a good example:
//
//   zapper|z|zap
//
// This should get rewritten to the equivalent:
//
//   z(?:apper||ap)
//
// Depending on how you do the trie, you could get any of these incorrect
// results:
//
//   z(?:|apper|ap) or z(?:apper|ap|) or even z(?:ap|apper|)
//
// So I think the transitions for each state need to look like
//
//   enum Transition { Next { byte: u8, state_id: StateID }, Match, }
//
// And the transition lookup routine is to search *backwards* from the end of
// the transitions in the current state until you either find a macthing byte
// or a Match. If you see a matching byte, then you follow it kind of like a
// normal trie. But if you see a match, you can't move past it, so you add a
// new transition for your current byte even if such a transition exists before
// the Match.
//
// This lets us (optimally, I believe) exploit redundancy in an alternation
// of literals without changing semantics. (If leftmost-first match semantics
// aren't used, then a totally normal trie can be used because the order no
// longer matters.)
//
// Once the trie is constructed, I believe it should then be straight-forward
// to build an HIR value from the trie, and then compile that. ... Although, I
// really wanted to utilize sparse NFA states here. That is, in a normal trie,
// every state can be trivially converted to a sparse NFA state since there
// is at most one transition out for each byte. But... that's not true in the
// modified model above. Although, for a given state, we should still be able
// to chunk it up into one or more sparse states, where they are themselves
// in an alternation with epsilon transitions to a "match" (or "out") state
// between them. In the case of non-leftmost-first semantics, there will only
// ever be one chunk and thus we get our wish...
//
// The main bummer of this approach is that finding a transition requires a
// linear scan of a state's transitions because we aren't storing them in
// order...
//
// Wait, isn't it true that any arbitrary re-ordering can occur *within*
// each "chunk" above? Yes, I believe so, because they are by definition
// non-overlapping so the preference order is never actually applicable?
// If that's true, then perhaps our transitions don't need to be one flat
// sequence, but rather, a `Vec<Vec<(u8, StateID)>>`, although that is a bit
// unfortunate. We could use a flat sequence with an index to the start of the
// currently active chunk.
//
// OK also, above, I was wrong about converting this trie to HIR. We want to
// convert it to a Thompson NFA directly so we can write our sparse states.
// If we convert to an HIR, then we'd need to add more sophistication to the
// Thompson compiler elsewhere in order to make that HIR use sparse states.
// (Which we should probably do anyway...)
//
// I think we just want a depth first traversal and it should be pretty
// straight-forward from there? No, it is not so straight-forward I think, but
// it feels doable. I can almost see the light at the end of the tunnel.

/*
/// Returns true if this trie contains a literal that is a prefix of the
/// bytes given.
///
/// When leftmost-first match semantics are enabled, this returning true
/// generally means that the given bytes shouldn't be added to the trie
/// because it's not possible for it to ever match.
fn contains_prefix(&self, bytes: &[u8]) -> bool {
    let mut sid = StateID::ZERO;
    for &byte in bytes.iter() {
        sid = match self.states[sid].next_state(byte) {
            None => break,
            Some(sid) => sid,
        };
        if self.states[sid].is_match {
            return true;
        }
    }
    self.states[sid].is_match
}
*/

#[cfg(test)]
mod tests {
    use regex_syntax::hir::Hir;

    use super::*;

    #[test]
    fn zap() {
        let mut trie = LiteralTrie::forward();
        trie.add(b"zapper").unwrap();
        trie.add(b"z").unwrap();
        trie.add(b"zap").unwrap();

        let got = trie.to_hir();
        let expected = Hir::concat(vec![
            Hir::literal("z".as_bytes()),
            Hir::alternation(vec![
                Hir::literal("apper".as_bytes()),
                Hir::empty(),
                Hir::literal("ap".as_bytes()),
            ]),
        ]);
        assert_eq!(expected, got);
    }

    #[test]
    fn scratch() {
        let mut trie = LiteralTrie::forward();
        trie.add(b"zapper").unwrap();
        trie.add(b"z").unwrap();
        trie.add(b"zap").unwrap();
        dbg!(&trie);
        println!("#### HIR");
        let hir = trie.to_hir();
        dbg!(&hir);
        println!("{}", hir);
    }
}
