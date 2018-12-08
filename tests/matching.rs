use regex_automata::{DFABuilder, Error, ErrorKind};

use fowler;

type GeneratedTest = (
    &'static str,
    &'static str,
    &'static [u8],
    &'static [Option<(usize, usize)>],
);

struct SuiteTest {
    name: &'static str,
    pattern: &'static str,
    input: &'static [u8],
    mat: Option<(usize, usize)>,
}

impl SuiteTest {
    fn collection(tests: &[GeneratedTest]) -> Vec<SuiteTest> {
        tests.iter().cloned().map(SuiteTest::new).collect()
    }

    fn new(gentest: GeneratedTest) -> SuiteTest {
        SuiteTest {
            name: gentest.0,
            pattern: gentest.1,
            input: gentest.2,
            mat: gentest.3[0],
        }
    }

    fn run_is_match<F: FnMut(&[u8]) -> bool>(&self, mut is_match: F) {
        assert_eq!(
            self.mat.is_some(),
            is_match(self.input),
            "is_match disagreement: test: {}, pattern: {}, input: {}",
            self.name, self.pattern, String::from_utf8_lossy(self.input),
        );
    }

    fn run_find_end<F: FnMut(&[u8]) -> Option<usize>>(&self, mut find: F) {
        assert_eq!(
            self.mat.map(|(_, end)| end),
            find(self.input),
            "match end location disagreement: \
             test: {}, pattern: {}, input: {}",
            self.name, self.pattern, String::from_utf8_lossy(self.input),
        );
    }
}

#[test]
fn suite_dfa() {
    let builder = DFABuilder::new();

    let tests = SuiteTest::collection(fowler::TESTS);
    for test in &tests {
        let dfa = match ignore_unsupported(builder.build(test.pattern)) {
            None => continue,
            Some(dfa) => dfa,
        };
        test.run_is_match(|x| dfa.is_match(x));
        test.run_find_end(|x| dfa.find(x));
    }
}

#[test]
fn suite_dfa_minimal() {
    let mut builder = DFABuilder::new();
    builder.minimize(true);

    let tests = SuiteTest::collection(fowler::TESTS);
    for test in &tests {
        // TODO: These tests take too long with minimization. Make
        // minimization faster.
        if test.name.starts_with("repetition_10") {
            continue;
        }
        if test.name.starts_with("repetition_11") {
            continue;
        }

        let dfa = match ignore_unsupported(builder.build(test.pattern)) {
            None => continue,
            Some(dfa) => dfa,
        };
        test.run_is_match(|x| dfa.is_match(x));
        test.run_find_end(|x| dfa.find(x));
    }
}

fn ignore_unsupported<T>(res: Result<T, Error>) -> Option<T> {
    let err = match res {
        Ok(t) => return Some(t),
        Err(err) => err,
    };
    if let ErrorKind::Unsupported(_) = *err.kind() {
        return None;
    }
    panic!("{}", err);
}
