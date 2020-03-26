use crate::dfa::{automaton::Automaton, dense, sparse};
use crate::StateID;

impl<T: AsRef<[S]>, A: AsRef<[u8]>, S: StateID> fst::Automaton
    for dense::DFA<T, A, S>
{
    type State = S;

    #[inline]
    fn start(&self) -> S {
        self.start_state_forward(&[], 0, 0)
    }

    #[inline]
    fn is_match(&self, state: &S) -> bool {
        self.is_match_state(*state)
    }

    #[inline]
    fn accept(&self, state: &S, byte: u8) -> S {
        if fst::Automaton::is_match(self, state) {
            return *state;
        }
        self.next_state(*state, byte)
    }

    #[inline]
    fn accept_eof(&self, state: &S) -> Option<S> {
        if fst::Automaton::is_match(self, state) {
            return Some(*state);
        }
        Some(self.next_eof_state(*state))
    }

    #[inline]
    fn can_match(&self, state: &S) -> bool {
        !self.is_dead_state(*state)
    }
}

impl<T: AsRef<[u8]>, S: StateID> fst::Automaton for sparse::DFA<T, S> {
    type State = S;

    #[inline]
    fn start(&self) -> S {
        self.start_state_forward(&[], 0, 0)
    }

    #[inline]
    fn is_match(&self, state: &S) -> bool {
        self.is_match_state(*state)
    }

    #[inline]
    fn accept(&self, state: &S, byte: u8) -> S {
        if fst::Automaton::is_match(self, state) {
            return *state;
        }
        self.next_state(*state, byte)
    }

    #[inline]
    fn accept_eof(&self, state: &S) -> Option<S> {
        if fst::Automaton::is_match(self, state) {
            return Some(*state);
        }
        Some(self.next_eof_state(*state))
    }

    #[inline]
    fn can_match(&self, state: &S) -> bool {
        !self.is_dead_state(*state)
    }
}

#[cfg(test)]
mod tests {
    use bstr::BString;
    use fst::{Automaton, IntoStreamer, Set, Streamer};

    use crate::dfa::{dense, sparse};

    fn search<A: Automaton, D: AsRef<[u8]>>(
        set: &Set<D>,
        aut: A,
    ) -> Vec<BString> {
        let mut stream = set.search(aut).into_stream();

        let mut results = vec![];
        while let Some(key) = stream.next() {
            results.push(BString::from(key));
        }
        results
    }

    #[test]
    fn dense_anywhere() {
        let set =
            Set::from_iter(&["a", "bar", "baz", "wat", "xba", "xbax", "z"])
                .unwrap();
        let dfa = dense::DFA::new("ba.*").unwrap();
        let got = search(&set, &dfa);
        assert_eq!(got, vec!["bar", "baz", "xba", "xbax"]);
    }

    #[test]
    fn dense_anchored() {
        let set =
            Set::from_iter(&["a", "bar", "baz", "wat", "xba", "xbax", "z"])
                .unwrap();
        let dfa = dense::Builder::new()
            .configure(dense::Config::new().anchored(true))
            .build("ba.*")
            .unwrap();
        let got = search(&set, &dfa);
        assert_eq!(got, vec!["bar", "baz"]);
    }

    #[test]
    fn dense_assertions_start() {
        let set =
            Set::from_iter(&["a", "bar", "baz", "wat", "xba", "xbax", "z"])
                .unwrap();
        let dfa = dense::Builder::new().build("^ba.*").unwrap();
        let got = search(&set, &dfa);
        assert_eq!(got, vec!["bar", "baz"]);
    }

    #[test]
    fn dense_assertions_end() {
        let set =
            Set::from_iter(&["a", "bar", "bax", "wat", "xba", "xbax", "z"])
                .unwrap();
        let dfa = dense::Builder::new().build(".*x$").unwrap();
        let got = search(&set, &dfa);
        assert_eq!(got, vec!["bax", "xbax"]);
    }

    #[test]
    fn dense_assertions_word() {
        let set =
            Set::from_iter(&["foo", "foox", "xfoo", "zzz foo zzz"]).unwrap();
        let dfa = dense::Builder::new().build(r"(?-u)\bfoo\b").unwrap();
        let got = search(&set, &dfa);
        assert_eq!(got, vec!["foo", "zzz foo zzz"]);
    }

    #[test]
    fn sparse_anywhere() {
        let set =
            Set::from_iter(&["a", "bar", "baz", "wat", "xba", "xbax", "z"])
                .unwrap();
        let dfa = sparse::DFA::new("ba.*").unwrap();
        let got = search(&set, &dfa);
        assert_eq!(got, vec!["bar", "baz", "xba", "xbax"]);
    }

    #[test]
    fn sparse_anchored() {
        let set =
            Set::from_iter(&["a", "bar", "baz", "wat", "xba", "xbax", "z"])
                .unwrap();
        let dfa = dense::Builder::new()
            .configure(dense::Config::new().anchored(true))
            .build("ba.*")
            .unwrap()
            .to_sparse()
            .unwrap();
        let got = search(&set, &dfa);
        assert_eq!(got, vec!["bar", "baz"]);
    }

    #[test]
    fn sparse_assertions_start() {
        let set =
            Set::from_iter(&["a", "bar", "baz", "wat", "xba", "xbax", "z"])
                .unwrap();
        let dfa =
            dense::Builder::new().build("^ba.*").unwrap().to_sparse().unwrap();
        let got = search(&set, &dfa);
        assert_eq!(got, vec!["bar", "baz"]);
    }

    #[test]
    fn sparse_assertions_end() {
        let set =
            Set::from_iter(&["a", "bar", "bax", "wat", "xba", "xbax", "z"])
                .unwrap();
        let dfa =
            dense::Builder::new().build(".*x$").unwrap().to_sparse().unwrap();
        let got = search(&set, &dfa);
        assert_eq!(got, vec!["bax", "xbax"]);
    }

    #[test]
    fn sparse_assertions_word() {
        let set =
            Set::from_iter(&["foo", "foox", "xfoo", "zzz foo zzz"]).unwrap();
        let dfa = dense::Builder::new()
            .build(r"(?-u)\bfoo\b")
            .unwrap()
            .to_sparse()
            .unwrap();
        let got = search(&set, &dfa);
        assert_eq!(got, vec!["foo", "zzz foo zzz"]);
    }
}
