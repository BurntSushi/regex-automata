use std::collections::BTreeMap;
use std::fmt::{self, Write};

use regex_automata::{ErrorKind, Regex, RegexBuilder, StateID};
use serde_json;

lazy_static! {
    pub static ref SUITE: RegexTestCollection = {
        let mut col = RegexTestCollection::new();
        col.extend(RegexTestGroups::fowler_basic());
        col.extend(RegexTestGroups::fowler_nullsubexpr());
        col.extend(RegexTestGroups::fowler_repetition());
        col.extend(RegexTestGroups::fowler_repetition_long());
        col.extend(RegexTestGroups::unicode());
        col.extend(RegexTestGroups::invalid_utf8());
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
    pub group_name: String,
    #[serde(default)]
    pub options: Vec<RegexTestOption>,
    pub pattern: String,
    #[serde(with = "serde_bytes")]
    pub input: Vec<u8>,
    pub full_match: Option<Match>,
    #[serde(default)]
    pub captures: Vec<Option<Match>>,
    #[serde(default)]
    pub fowler_line_number: Option<u64>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum RegexTestOption {
    CaseInsensitive,
    NoUnicode,
    Escaped,
    #[serde(rename = "invalid-utf8")]
    InvalidUTF8,
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

    pub fn tests(&self) -> impl Iterator<Item=&RegexTest> {
        self.groups.values().flat_map(|g| g.tests.iter())
    }
}

impl RegexTestGroups {
    fn load(slice: &[u8]) -> RegexTestGroups {
        let mut data: RegexTestGroups = serde_json::from_slice(slice).unwrap();
        for group in &mut data.groups {
            for test in &mut group.tests {
                test.group_name = group.name.clone();
                if test.options.contains(&RegexTestOption::Escaped) {
                    test.input = unescape_bytes(&test.input);
                }
            }
        }
        data
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

    fn fowler_repetition_long() -> RegexTestGroups {
        let raw = include_bytes!("../data/tests/fowler/repetition-long.json");
        RegexTestGroups::load(raw)
    }

    fn unicode() -> RegexTestGroups {
        let raw = include_bytes!("../data/tests/unicode.json");
        RegexTestGroups::load(raw)
    }

    fn invalid_utf8() -> RegexTestGroups {
        let raw = include_bytes!("../data/tests/invalid-utf8.json");
        RegexTestGroups::load(raw)
    }
}

#[derive(Debug)]
pub struct RegexTester {
    results: RegexTestResults,
    skip_expensive: bool,
}

impl RegexTester {
    pub fn new() -> RegexTester {
        RegexTester {
            results: RegexTestResults::default(),
            skip_expensive: false,
        }
    }

    pub fn skip_expensive(mut self) -> RegexTester {
        self.skip_expensive = true;
        self
    }

    pub fn assert(&self) {
        self.results.assert();
    }

    pub fn test_all<'a, I: Iterator<Item=&'a RegexTest>>(
        &mut self,
        builder: RegexBuilder,
        tests: I,
    ) {
        for test in tests {
            let builder = builder.clone();
            let re: Regex<usize> = match self.build_regex(builder, test) {
                None => continue,
                Some(re) => re,
            };
            self.test_is_match(test, &re);
            self.test_find(test, &re);
        }
    }

    pub fn build_regex<S: StateID>(
        &self,
        mut builder: RegexBuilder,
        test: &RegexTest,
    ) -> Option<Regex<'static, S>> {
        if self.skip(test) {
            return None;
        }
        self.apply_options(test, &mut builder);
        match builder.build_with_size::<S>(&test.pattern) {
            Ok(re) => Some(re),
            Err(err) => {
                if let ErrorKind::Unsupported(_) = *err.kind() {
                    None
                } else {
                    panic!("failed to build '{:?}': {}", test.pattern, err);
                }
            }
        }
    }

    pub fn test_is_match<'a, S: StateID>(
        &mut self,
        test: &RegexTest,
        re: &Regex<'a, S>,
    ) {
        let got = re.is_match(&test.input);
        let expected = test.full_match.is_some();
        if got == expected {
            self.results.succeeded += 1;
            return;
        }
        self.results.failed.push(RegexTestFailure {
            test: test.clone(),
            kind: RegexTestFailureKind::IsMatch,
        });
    }

    pub fn test_find<'a, S: StateID>(
        &mut self,
        test: &RegexTest,
        re: &Regex<'a, S>,
    ) {
        let got = re
            .find(&test.input)
            .map(|(start, end)| Match { start, end });
        if got == test.full_match {
            self.results.succeeded += 1;
            return;
        }
        self.results.failed.push(RegexTestFailure {
            test: test.clone(),
            kind: RegexTestFailureKind::Find { got },
        });
    }

    fn skip(&self, test: &RegexTest) -> bool {
        if self.skip_expensive {
            if test.group_name == "repetition-long" {
                return true;
            }
        }
        false
    }

    fn apply_options(&self, test: &RegexTest, builder: &mut RegexBuilder) {
        for opt in &test.options {
            match *opt {
                RegexTestOption::CaseInsensitive => {
                    builder.case_insensitive(true);
                }
                RegexTestOption::NoUnicode => {
                    builder.unicode(false);
                }
                RegexTestOption::Escaped => {}
                RegexTestOption::InvalidUTF8 => {
                    builder.allow_invalid_utf8(true);
                }
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RegexTestResults {
    /// The number of successful tests.
    pub succeeded: usize,
    /// Failed tests, indexed by group name.
    pub failed: Vec<RegexTestFailure>,
}

#[derive(Clone, Debug)]
pub struct RegexTestFailure {
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
            input (escape): {}\n    \
            input (hex): {}",
            self.test.group_name,
            self.kind.fmt(&self.test)?,
            self.test.options,
            self.test.pattern,
            escape_default(&self.test.pattern),
            nice_raw_bytes(&self.test.input),
            escape_bytes(&self.test.input),
            hex_bytes(&self.test.input)
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

fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| format!(r"\x{:02X}", b)).collect()
}

fn escape_default(s: &str) -> String {
    s.chars().flat_map(|c| c.escape_default()).collect()
}

fn unescape_bytes(bytes: &[u8]) -> Vec<u8> {
    use std::str;
    use unescape::unescape;

    unescape(&str::from_utf8(bytes).expect("all input must be valid UTF-8"))
}
