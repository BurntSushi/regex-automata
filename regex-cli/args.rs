use std::{
    fmt::{Debug, Display, Write},
    str::FromStr,
};

use {
    anyhow::Context,
    lexopt::{Arg, Parser, ValueExt},
    regex::Regex,
};

/// Parses the argument from the given parser as a command name, and returns
/// it. If the next arg isn't a simple value then this returns an error.
///
/// This also handles the case where -h/--help is given, in which case, the
/// given usage information is converted into an error and printed.
pub fn next_as_command(usage: &str, p: &mut Parser) -> anyhow::Result<String> {
    let usage = usage.trim();
    let arg = match p.next()? {
        Some(arg) => arg,
        None => anyhow::bail!("{}", usage),
    };
    let cmd = match arg {
        Arg::Value(cmd) => cmd.string()?,
        Arg::Short('h') | Arg::Long("help") => anyhow::bail!("{}", usage),
        arg => return Err(arg.unexpected().into()),
    };
    Ok(cmd)
}

/// Parses the next 'p.value()' into 'T'. Any error messages will include the
/// given flag name in them.
pub fn parse<T>(p: &mut Parser, flag_name: &'static str) -> anyhow::Result<T>
where
    T: FromStr,
    <T as FromStr>::Err: Display + Debug + Send + Sync + 'static,
{
    // This is written somewhat awkwardly and the type signature is also pretty
    // funky primarily because of the following two things: 1) the 'FromStr'
    // impls in this crate just use 'anyhow::Error' for their error type and 2)
    // 'anyhow::Error' does not impl 'std::error::Error'.
    let osv = p.value().context(flag_name)?;
    let strv = match osv.to_str() {
        Some(strv) => strv,
        None => {
            let err = lexopt::Error::NonUnicodeValue(osv.into());
            return Err(anyhow::Error::from(err).context(flag_name));
        }
    };
    let parsed = match strv.parse() {
        Err(err) => return Err(anyhow::Error::msg(err).context(flag_name)),
        Ok(parsed) => parsed,
    };
    Ok(parsed)
}

/// Like `parse`, but permits the string value "none" to indicate absent. This
/// is useful for parsing things like limits, where "no limit" is a legal
/// value. But it can be used for anything.
pub fn parse_maybe<T>(
    p: &mut Parser,
    flag_name: &'static str,
) -> anyhow::Result<Option<T>>
where
    T: FromStr,
    <T as FromStr>::Err: Display + Debug + Send + Sync + 'static,
{
    // This is written somewhat awkwardly and the type signature is also pretty
    // funky primarily because of the following two things: 1) the 'FromStr'
    // impls in this crate just use 'anyhow::Error' for their error type and 2)
    // 'anyhow::Error' does not impl 'std::error::Error'.
    let osv = p.value().context(flag_name)?;
    let strv = match osv.to_str() {
        Some(strv) => strv,
        None => {
            let err = lexopt::Error::NonUnicodeValue(osv.into());
            return Err(anyhow::Error::from(err).context(flag_name));
        }
    };
    if strv == "none" {
        return Ok(None);
    }
    let parsed = match strv.parse() {
        Err(err) => return Err(anyhow::Error::msg(err).context(flag_name)),
        Ok(parsed) => parsed,
    };
    Ok(Some(parsed))
}

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
digestible, typically be enabling quick eye scanning. For example, when enabled
for the various benchmark comparison commands, the "best" timings are
colorized. The choices are: auto, always, never.
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

/// A type for expressing the documentation of a flag.
///
/// The `Usage::short` and `Usage::long` functions take a slice of usages and
/// format them into a human readable display. It does simple word wrapping and
/// column alignment for you.
#[derive(Clone, Copy, Debug)]
pub struct Usage {
    /// The format of the flag, for example, '-k, --match-kind <kind>'.
    pub format: &'static str,
    /// A very short description of the flag. Should fit on one line along with
    /// the format.
    pub short: &'static str,
    /// A longer form description of the flag. May be multiple paragraphs long
    /// (but doesn't have to be).
    pub long: &'static str,
}

impl Usage {
    /// Create a new usage from the given components.
    pub const fn new(
        format: &'static str,
        short: &'static str,
        long: &'static str,
    ) -> Usage {
        Usage { format, short, long }
    }

    /// Format a two column table from the given usages, where the first
    /// column is the format and the second column is the short description.
    pub fn short(usages: &[Usage]) -> String {
        const MIN_SPACE: usize = 2;

        let mut result = String::new();
        let max_len = match usages.iter().map(|u| u.format.len()).max() {
            None => return result,
            Some(len) => len,
        };
        for usage in usages.iter() {
            let padlen = MIN_SPACE + (max_len - usage.format.len());
            let padding = " ".repeat(padlen);
            writeln!(result, "    {}{}{}", usage.format, padding, usage.short)
                .unwrap();
        }
        result
    }

    /// Print the format of each usage and its long description below the
    /// format. This also does appropriate indentation with the assumption that
    /// it is in an OPTIONS section of a bigger usage message.
    pub fn long(usages: &[Usage]) -> String {
        let wrap_opts = textwrap::Options::new(79)
            .initial_indent("        ")
            .subsequent_indent("        ");
        let mut result = String::new();
        for (i, usage) in usages.iter().enumerate() {
            if i > 0 {
                writeln!(result, "").unwrap();
            }
            writeln!(result, "    {}", usage.format).unwrap();
            for (i, paragraph) in usage.long.trim().split("\n\n").enumerate() {
                if i > 0 {
                    result.push('\n');
                }
                let flattened = paragraph.replace("\n", " ");
                for line in textwrap::wrap(&flattened, &wrap_opts) {
                    result.push_str(&line);
                    result.push('\n');
                }
            }
        }
        result
    }
}
