use {anyhow::Context, regex::Regex};

use crate::args::Usage;

/// This defines a flag for controlling the use of color in the output.
#[derive(Debug)]
pub enum Color {
    /// Color is only enabled when the output is a tty.
    Auto,
    /// Color is always enabled.
    Always,
    /// Color is disabled.
    Never,
}

impl Color {
    pub const USAGE: Usage = Usage::new(
        "-c, --color <mode>",
        "One of: auto, always, never.",
        r#"
Whether to use color (default: auto).

When enabled, a modest amount of color is used to help make the output more
digestible, typically be enabling quick eye scanning.
"#,
    );

    /// Return a possibly colorized stdout.
    #[allow(dead_code)]
    pub fn stdout(&self) -> Box<dyn termcolor::WriteColor> {
        use termcolor::{Ansi, NoColor};

        if self.should_color() {
            Box::new(Ansi::new(std::io::stdout()))
        } else {
            Box::new(NoColor::new(std::io::stdout()))
        }
    }

    /// Return a possibly colorized stdout, just like 'stdout', except the
    /// output supports elastic tabstops.
    pub fn elastic_stdout(&self) -> Box<dyn termcolor::WriteColor> {
        use {
            tabwriter::TabWriter,
            termcolor::{Ansi, NoColor},
        };

        if self.should_color() {
            Box::new(Ansi::new(TabWriter::new(std::io::stdout())))
        } else {
            Box::new(NoColor::new(TabWriter::new(std::io::stdout())))
        }
    }

    /// Return true if colors should be used. When the color choice is 'auto',
    /// this only returns true if stdout is a tty.
    pub fn should_color(&self) -> bool {
        match *self {
            Color::Auto => atty::is(atty::Stream::Stdout),
            Color::Always => true,
            Color::Never => false,
        }
    }
}

impl Default for Color {
    fn default() -> Color {
        Color::Auto
    }
}

impl std::str::FromStr for Color {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Color> {
        let color = match s {
            "auto" => Color::Auto,
            "always" => Color::Always,
            "never" => Color::Never,
            unknown => {
                anyhow::bail!(
                    "unrecognized color config '{}', must be \
                     one of auto, always or never.",
                    unknown,
                )
            }
        };
        Ok(color)
    }
}

/// Filter is the implementation of whitelist/blacklist rules. If there are no
/// rules, everything matches. If there's at least one whitelist rule, then you
/// need at least one whitelist rule to match to get through the filter. If
/// there are no whitelist regexes, then you can't match any of the blacklist
/// regexes.
///
/// This filter also has precedence built into that. That means that the order
/// of rules matters. So for example, if you have a whitelist regex that
/// matches AFTER a blacklist regex matches, then the input is considered to
/// have matched the filter.
#[derive(Clone, Debug, Default)]
pub struct Filter {
    rules: Vec<FilterRule>,
}

impl Filter {
    /// Create a new filter from one whitelist regex pattern.
    ///
    /// More rules may be added, but this is a convenience routine for a simple
    /// filter.
    pub fn from_pattern(pat: &str) -> anyhow::Result<Filter> {
        let mut filter = Filter::default();
        filter.add(pat.parse()?);
        Ok(filter)
    }

    /// Add the given rule to this filter.
    pub fn add(&mut self, rule: FilterRule) {
        self.rules.push(rule);
    }

    /// Return true if and only if the given subject passes this filter.
    pub fn include(&self, subject: &str) -> bool {
        // If we have no rules, then everything matches.
        if self.rules.is_empty() {
            return true;
        }
        // If we have any whitelist rules, then 'include' starts off as false,
        // as we need at least one whitelist rule in that case to match. If all
        // we have are blacklists though, then we start off with include=true,
        // and we only get excluded if one of those blacklists is matched.
        let mut include = self.rules.iter().all(|r| r.blacklist);
        for rule in &self.rules {
            if rule.re.is_match(subject) {
                include = !rule.blacklist;
            }
        }
        include
    }
}

/// A single rule in a filter, which is a combination of a regex and whether
/// it's a blacklist rule or not.
#[derive(Clone, Debug)]
pub struct FilterRule {
    re: Regex,
    blacklist: bool,
}

impl std::str::FromStr for FilterRule {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<FilterRule> {
        let (pattern, blacklist) =
            if s.starts_with('!') { (&s[1..], true) } else { (&*s, false) };
        let re = Regex::new(pattern).context("filter regex is not valid")?;
        Ok(FilterRule { re, blacklist })
    }
}

/// This defines a implementation for a flag that wants a single byte. This is
/// useful because there are some APIs that require a single byte. For example,
/// setting a line terminator.
///
/// This in particular supports the ability to write the byte via an escape
/// sequence. For example, `--flag '\xFF'` will parse to the single byte 0xFF.
///
/// If the flag value is empty or if it unescapes into something with more than
/// one byte, then it is considered an error.
#[derive(Debug, Default)]
pub struct OneByte(pub u8);

impl std::str::FromStr for OneByte {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<OneByte> {
        let bytes = crate::escape::unescape(s);
        anyhow::ensure!(
            bytes.len() == 1,
            "expected exactly one byte, but got {} bytes",
            bytes.len(),
        );
        Ok(OneByte(bytes[0]))
    }
}

/// This defines a implementation for a flag that wants a possibly empty set
/// of bytes. This is useful because there are some APIs that require multiple
/// individual bytes. For example, setting quit bytes for a DFA.
///
/// This in particular supports the ability to write the byte set via a
/// sequence of escape sequences. For example, `--flag 'a\xFF\t'` will parse to
/// the sequence 0x61 0xFF 0x09.
///
/// By default, the set is empty. If the flag value has a duplicate byte, then
/// an error is returned. An empty value corresponds to the empty set.
#[derive(Debug, Default)]
pub struct ByteSet(pub Vec<u8>);

impl std::str::FromStr for ByteSet {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<ByteSet> {
        let mut set = vec![];
        let mut seen = [false; 256];
        for &byte in crate::escape::unescape(s).iter() {
            anyhow::ensure!(
                !seen[usize::from(byte)],
                "saw duplicate byte 0x{:2X} in '{}'",
                byte,
                s,
            );
            seen[usize::from(byte)] = true;
            set.push(byte);
        }
        set.sort();
        Ok(ByteSet(set))
    }
}

/// Provides an implementation of the --start-kind flag, for use with DFA
/// configuration.
#[derive(Debug)]
pub struct StartKind {
    pub kind: regex_automata::dfa::StartKind,
}

impl StartKind {
    pub const USAGE: Usage = Usage::new(
        "--start-kind <kind>",
        "One of: both, unanchored, anchored.",
        r#"
Sets the start states supported by a DFA. The default is 'both', but it can
be set to either 'unanchored' or 'anchored'. The benefit of only supporting
unanchored or anchored start states is that it usually leads to a smaller
overall automaton.
"#,
    );
}

impl Default for StartKind {
    fn default() -> StartKind {
        StartKind { kind: regex_automata::dfa::StartKind::Both }
    }
}

impl std::str::FromStr for StartKind {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<StartKind> {
        let kind = match s {
            "both" => regex_automata::dfa::StartKind::Both,
            "unanchored" => regex_automata::dfa::StartKind::Unanchored,
            "anchored" => regex_automata::dfa::StartKind::Anchored,
            unk => anyhow::bail!("unrecognized start kind '{}'", unk),
        };
        Ok(StartKind { kind })
    }
}

/// Provides an implementation of the --match-kind flag, for use with most
/// regex matchers.
#[derive(Debug)]
pub struct MatchKind {
    pub kind: regex_automata::MatchKind,
}

impl MatchKind {
    pub const USAGE: Usage = Usage::new(
        "-k, --match-kind <kind>",
        "One of: leftmost-first, all.",
        r#"
Selects the match semantics for the regex engine. The choices are
'leftmost-first' (the default) or 'all'.

'leftmost-first' semantics look for the leftmost match, and when there are
multiple leftmost matches, match priority disambiguates them. For example,
in the haystack 'samwise', the regex 'samwise|sam' will match 'samwise' when
using leftmost-first semantics. Similarly, the regex 'sam|samwise' will match
'sam'.

'all' semantics results in including all possible match states in the
underlying automaton. When performing an unanchored leftmost search, this has
the effect of finding the last match, which is usually not what you want.
When performing an anchored leftmost search, it has the effect of finding the
longest possible match, which might be what you want. (So there is no support
for greedy vs non-greedy searching. Everything is greedy.) 'all' is also useful
for overlapping searches, since all matches are reportable in this scheme.
"#,
    );
}

impl Default for MatchKind {
    fn default() -> MatchKind {
        MatchKind { kind: regex_automata::MatchKind::LeftmostFirst }
    }
}

impl std::str::FromStr for MatchKind {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<MatchKind> {
        let kind = match s {
            "leftmost-first" => regex_automata::MatchKind::LeftmostFirst,
            "all" => regex_automata::MatchKind::All,
            unk => anyhow::bail!("unrecognized match kind '{}'", unk),
        };
        Ok(MatchKind { kind })
    }
}
