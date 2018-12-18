use std::collections::BTreeMap;
use std::fmt::{self, Write};

use serde_json;

lazy_static! {
    pub static ref SUITE: RegexTestCollection = {
        let mut col = RegexTestCollection::new();
        col.extend(RegexTestGroups::fowler_basic());
        col.extend(RegexTestGroups::fowler_nullsubexpr());
        col.extend(RegexTestGroups::fowler_repetition());
        col.extend(RegexTestGroups::unicode());
        col
    };
}

#[derive(Clone, Debug)]
pub struct RegexTestCollection {
    pub groups: BTreeMap<String, RegexTestGroup>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RegexTestGroups {
    pub groups: Vec<RegexTestGroup>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RegexTestGroup {
    pub name: String,
    pub tests: Vec<RegexTest>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RegexTest {
    #[serde(default)]
    pub fowler_line_number: Option<u64>,
    #[serde(default)]
    pub options: Vec<RegexTestOption>,
    pub pattern: String,
    #[serde(with = "serde_bytes")]
    pub input: Vec<u8>,
    pub full_match: Option<Match>,
    #[serde(default)]
    pub captures: Vec<Option<Match>>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum RegexTestOption {
    CaseInsensitive,
}

#[derive(Clone, Copy, Deserialize, Eq, PartialEq)]
pub struct Match {
    pub start: usize,
    pub end: usize,
}

impl RegexTestCollection {
    fn new() -> RegexTestCollection {
        RegexTestCollection { groups: BTreeMap::new() }
    }

    fn extend(&mut self, groups: RegexTestGroups) {
        for group in groups.groups {
            if !self.groups.contains_key(&group.name) {
                self.groups.insert(group.name.clone(), group);
            } else {
                self.groups
                    .get_mut(&group.name)
                    .unwrap()
                    .tests.extend(group.tests);
            }
        }
    }

    pub fn tester(&self) -> RegexTester {
        RegexTester { collection: self }
    }
}

impl RegexTestGroups {
    fn load(slice: &[u8]) -> RegexTestGroups {
        serde_json::from_slice(slice).unwrap()
    }

    fn fowler_basic() -> RegexTestGroups {
        let raw = include_bytes!("../data/tests/fowler/basic.json");
        RegexTestGroups::load(raw)
    }

    fn fowler_nullsubexpr() -> RegexTestGroups {
        let raw = include_bytes!("../data/tests/fowler/nullsubexpr.json");
        RegexTestGroups::load(raw)
    }

    fn fowler_repetition() -> RegexTestGroups {
        let raw = include_bytes!("../data/tests/fowler/repetition.json");
        RegexTestGroups::load(raw)
    }

    fn unicode() -> RegexTestGroups {
        let raw = include_bytes!("../data/tests/unicode.json");
        RegexTestGroups::load(raw)
    }
}

#[derive(Debug)]
pub struct RegexTester<'a> {
    collection: &'a RegexTestCollection,
}

impl<'a> RegexTester<'a> {
    pub fn is_match<F>(
        &self,
        mut is_match: F,
    ) -> RegexTestResults
    where F: FnMut(&RegexTest) -> Option<bool>
    {
        let mut results = RegexTestResults::new();
        for group in self.collection.groups.values() {
            for test in group.tests.iter() {
                let got = match is_match(test) {
                    None => continue,
                    Some(got) => got,
                };
                let expected = test.full_match.is_some();
                if got == expected {
                    results.succeeded += 1;
                    continue;
                }
                results.failed.push(RegexTestFailure {
                    group: group.name.clone(),
                    test: test.clone(),
                    kind: RegexTestFailureKind::IsMatch,
                });
            }
        }
        results
    }

    pub fn find<F>(
        &self,
        mut find: F,
    ) -> RegexTestResults
    where F: FnMut(&RegexTest) -> Option<Option<(usize, usize)>>
    {
        let mut results = RegexTestResults::new();
        for group in self.collection.groups.values() {
            for test in group.tests.iter() {
                let got = match find(test) {
                    None => continue,
                    Some(None) => None,
                    Some(Some((start, end))) => Some(Match { start, end }),
                };
                if got == test.full_match {
                    results.succeeded += 1;
                    continue;
                }
                results.failed.push(RegexTestFailure {
                    group: group.name.clone(),
                    test: test.clone(),
                    kind: RegexTestFailureKind::Find { got },
                });
            }
        }
        results
    }
}

#[derive(Clone, Debug)]
pub struct RegexTestResults {
    /// The number of successful tests.
    pub succeeded: usize,
    /// Failed tests, indexed by group name.
    pub failed: Vec<RegexTestFailure>,
}

#[derive(Clone, Debug)]
pub struct RegexTestFailure {
    group: String,
    test: RegexTest,
    kind: RegexTestFailureKind,
}

#[derive(Clone, Debug)]
pub enum RegexTestFailureKind {
    IsMatch,
    Find { got: Option<Match> },
}

impl RegexTestResults {
    fn new() -> RegexTestResults {
        RegexTestResults { succeeded: 0, failed: vec![] }
    }

    pub fn assert(&self) {
        if self.failed.is_empty() {
            return;
        }
        let failures = self
            .failed
            .iter()
            .map(|f| f.to_string())
            .collect::<Vec<String>>()
            .join("\n");
        panic!(
            "found {} failures:\n{}\n{}\n{}",
            self.failed.len(),
            "~".repeat(79),
            failures.trim(),
            "~".repeat(79)
        )
    }
}

impl fmt::Display for RegexTestFailure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}: {}\n    \
            options: {:?}\n    \
            pattern: {}\n    \
            pattern (escape): {}\n    \
            input: {}\n    \
            input (escape): {}",
            self.group,
            self.kind.fmt(&self.test)?,
            self.test.options,
            self.test.pattern,
            escape_default(&self.test.pattern),
            nice_raw_bytes(&self.test.input),
            escape_bytes(&self.test.input)
        )
    }
}

impl RegexTestFailureKind {
    fn fmt(&self, test: &RegexTest) -> Result<String, fmt::Error> {
        let mut buf = String::new();
        match *self {
            RegexTestFailureKind::IsMatch => {
                if let Some(m) = test.full_match {
                    write!(buf, "expected match (at {}), but none found", m)?
                } else {
                    write!(buf, "expected no match, but found a match")?
                }
            }
            RegexTestFailureKind::Find { got } => {
                write!(
                    buf,
                    "expected {:?}, but found {:?}",
                    test.full_match,
                    got
                )?
            }
        }
        Ok(buf)
    }
}

impl fmt::Display for Match {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.start, self.end)
    }
}

impl fmt::Debug for Match {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.start, self.end)
    }
}

fn nice_raw_bytes(bytes: &[u8]) -> String {
    use std::str;

    match str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        Err(_) => escape_bytes(bytes),
    }
}

fn escape_bytes(bytes: &[u8]) -> String {
    use std::ascii;

    let escaped = bytes
        .iter()
        .flat_map(|&b| ascii::escape_default(b))
        .collect::<Vec<u8>>();
    String::from_utf8(escaped).unwrap()
}

fn escape_default(s: &str) -> String {
    s.chars().flat_map(|c| c.escape_default()).collect()
}
