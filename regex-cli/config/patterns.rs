use std::path::PathBuf;

use {
    anyhow::Context,
    lexopt::{Arg, Parser, ValueExt},
};

use crate::{
    args::{self, Usage},
    config::Configurable,
};

#[derive(Debug, Default)]
pub struct Config {
    patterns: Vec<String>,
    fixed_strings: bool,
    combine: bool,
    mode: Mode,
}

impl Config {
    /// Creates a new configuration that will greedily treat every positional
    /// argument as a pattern. This also supports all other ways of providing
    /// patterns, i.e., the `-p/--pattern` and `-f/--file` flags.
    ///
    /// This is useful for commands that don't accept any other kind of
    /// positional arguments.
    pub fn positional() -> Config {
        Config { mode: Mode::Positional, ..Config::default() }
    }

    /// Creates a new configuration that will never treat a positional argument
    /// as a pattern. Instead, it only reads patterns from the `-p/--pattern`
    /// and `-f/--file` flags.
    ///
    /// This is useful for commands that accept other kinds of positional
    /// arguments. Forcing the use of a flag helps avoid resolving more
    /// complicated ambiguities regarding how to treat each positional
    /// argument.
    ///
    /// This is equivalent to `Config::default()`.
    pub fn only_flags() -> Config {
        Config::default()
    }

    /// Returns all of the pattern strings from this configuration, escaping
    /// and joining them if requested. When joining is requested, then at most
    /// one pattern is returned.
    ///
    /// Note that it is legal for this to return zero patterns!
    pub fn get(&self) -> anyhow::Result<Vec<String>> {
        let mut pats = self.patterns.clone();
        if self.fixed_strings {
            pats = pats.iter().map(|p| regex_syntax::escape(p)).collect();
        }
        if self.combine {
            // FIXME: This is... not technically correct, since someone could
            // provide a pattern `ab(cd` and then `ef)gh`. Neither are valid
            // patterns, but by joining them with a |, we get `ab(cd|ef)gh`
            // which is valid. The solution to this is I think to try and
            // parse the regex to make sure it's valid, but we should be
            // careful to only use the AST parser. The problem here is that
            // we don't technically have the configuration of the parser at
            // this point. We could *ask* for it. We could also just assume a
            // default configuration since the AST parser doesn't have many
            // configuration knobs. But probably we should just ask for the
            // parser configuration here.
            pats = vec![pats.join("|")];
        }
        Ok(pats)
    }
}

impl Configurable for Config {
    fn configure(
        &mut self,
        p: &mut Parser,
        arg: &mut Arg,
    ) -> anyhow::Result<bool> {
        match *arg {
            Arg::Short('p') | Arg::Long("pattern") => {
                let pat = p.value().context("-p/--pattern have a value")?;
                let pat = pat
                    .string()
                    .context("-p/--pattern must be valid UTF-8")?;
                self.patterns.push(pat);
            }
            Arg::Short('F') | Arg::Long("fixed-strings") => {
                self.fixed_strings = true;
            }
            Arg::Short('f') | Arg::Long("pattern-file") => {
                let path =
                    PathBuf::from(p.value().context("-f/--pattern-file")?);
                let contents =
                    std::fs::read_to_string(&path).with_context(|| {
                        anyhow::anyhow!("failed to read {}", path.display())
                    })?;
                self.patterns.extend(contents.lines().map(|x| x.to_string()));
            }
            Arg::Long("combine-patterns") => {
                self.combine = true;
            }
            Arg::Value(ref mut v) => {
                if !matches!(self.mode, Mode::Positional) {
                    return Ok(false);
                }
                let v = std::mem::take(v);
                self.patterns
                    .push(v.string().context("patterns must be valid UTF-8")?);
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn usage(&self) -> &[Usage] {
        const USAGES: &'static [Usage] = &[
            Usage::new(
                "--start <bound>",
                "Set the start of the search.",
                r#"
This sets the start bound of a search. It must be a valid offset for the
haystack, up to and including the length of the haystack.

When not set, the start bound is 0.
"#,
            ),
            Usage::new(
                "--end <bound>",
                "Set the end of the search.",
                r#"
This sets the end bound of a search. It must be a valid offset for the
haystack, up to and including the length of the haystack.

When not set, the end bound is the length of the haystack.
"#,
            ),
            Usage::new(
                "--anchored",
                "Enable anchored mode for the search.",
                r#"
Enabled anchored mode for the search. When enabled and if a match is found, the
start of the match is guaranteed to be equivalent to the start bound of the
search.
"#,
            ),
            Usage::new(
                "--pattern-id <pid>",
                "Set pattern to search for.",
                r#"
Set the pattern to search for. This automatically enables anchored mode for the
search since regex engines for this crate only support anchored searches for
specific patterns.

When set and if a match is found, the start of the match is guaranteed to be
equivalent to the start bound of the search and the pattern ID is guaranteed
to be equivalent to the one set by this flag.

When not set, a search may match any of the patterns given.
"#,
            ),
            Usage::new(
                "--earliest",
                "Returns a match as soon as it is known.",
                r#"
This enables "earliest" mode, which asks the regex engine to stop searching as
soon as a match is found. The specific offset returned may vary depending on
the regex engine since not all regex engines detect matches in the same way.
"#,
            ),
        ];
        USAGES
    }
}

/// The parsing behavior of a pattern configuration. That is, either treat
/// positional arguments as patterns or not.
///
/// The default is to only parse patterns from flags.
#[derive(Debug)]
enum Mode {
    Positional,
    OnlyFlags,
}

impl Default for Mode {
    fn default() -> Mode {
        Mode::OnlyFlags
    }
}
