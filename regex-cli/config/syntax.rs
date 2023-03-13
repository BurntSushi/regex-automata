use std::borrow::Borrow;

use {
    anyhow::Context,
    lexopt::{Arg, Parser},
    regex_automata::util::syntax,
    regex_syntax::{ast::Ast, hir::Hir},
};

use crate::{
    args::{self, Usage},
    config::Configurable,
};

/// This exposes all of the configuration knobs on a regex_automata::Input via
/// CLI flags. The only aspect of regex_automata::Input that this does not
/// cover is the haystack, which should be provided by other means (usually
/// with `Haystack`).
#[derive(Debug, Default)]
pub struct Config {
    syntax: syntax::Config,
}

impl Config {
    /// Return a `Syntax` object from this configuration.
    pub fn syntax(&self) -> anyhow::Result<syntax::Config> {
        Ok(self.syntax.clone())
    }

    /// Parses the given pattern into an `Ast`.
    fn ast(&self, pattern: &str) -> anyhow::Result<Ast> {
        regex_syntax::ast::parse::ParserBuilder::new()
            .nest_limit(self.syntax.get_nest_limit())
            .octal(self.syntax.get_octal())
            .ignore_whitespace(self.syntax.get_ignore_whitespace())
            .build()
            .parse(pattern)
            .context("failed to parse pattern")
    }

    /// Parses the given patterns into a corresponding sequence of `Ast`s. If
    /// any of the patterns fail to parse, then an error is returned.
    pub fn asts<P: AsRef<str>>(
        &self,
        patterns: &[P],
    ) -> anyhow::Result<Vec<Ast>> {
        patterns
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let p = p.as_ref();
                self.ast(p).with_context(|| {
                    format!("failed to parse pattern {} to AST: '{}'", i, p,)
                })
            })
            .collect()
    }

    /// Translates the given pattern and `Ast` into an `Hir`.
    pub fn hir(&self, pattern: &str, ast: &Ast) -> anyhow::Result<Hir> {
        regex_syntax::hir::translate::TranslatorBuilder::new()
            .utf8(self.syntax.get_utf8())
            .case_insensitive(self.syntax.get_case_insensitive())
            .multi_line(self.syntax.get_multi_line())
            .dot_matches_new_line(self.syntax.get_dot_matches_new_line())
            .swap_greed(self.syntax.get_swap_greed())
            .unicode(self.syntax.get_unicode())
            .build()
            .translate(pattern, ast)
            .context("failed to translate pattern")
    }

    /// Translates the given patterns and corresponding `Ast`s into a
    /// corresponding sequence of `Hir`s. If any of the patterns fail to
    /// translate, then an error is returned.
    pub fn hirs<P: AsRef<str>, A: Borrow<Ast>>(
        &self,
        patterns: &[P],
        asts: &[A],
    ) -> anyhow::Result<Vec<Hir>> {
        patterns
            .iter()
            .zip(asts.iter())
            .enumerate()
            .map(|(i, (pat, ast))| {
                let (pat, ast) = (pat.as_ref(), ast.borrow());
                self.hir(pat, ast).with_context(|| {
                    format!(
                        "failed to translate pattern {} to HIR: '{}'",
                        i, pat,
                    )
                })
            })
            .collect()
    }
}

impl Configurable for Config {
    fn configure(
        &mut self,
        p: &mut Parser,
        arg: &mut Arg,
    ) -> anyhow::Result<bool> {
        match *arg {
            Arg::Short('i') | Arg::Long("case-insensitive") => {
                self.syntax = self.syntax.case_insensitive(true);
            }
            Arg::Long("multi-line") => {
                self.syntax = self.syntax.multi_line(true);
            }
            Arg::Long("dot-matches-new-line") => {
                self.syntax = self.syntax.dot_matches_new_line(true);
            }
            Arg::Long("crlf") => {
                self.syntax = self.syntax.crlf(true);
            }
            Arg::Long("swap-greed") => {
                self.syntax = self.syntax.swap_greed(true);
            }
            Arg::Long("ignore-whitespace") => {
                self.syntax = self.syntax.ignore_whitespace(true);
            }
            Arg::Short('U') | Arg::Long("no-unicode") => {
                self.syntax = self.syntax.unicode(false);
            }
            Arg::Short('b') | Arg::Long("no-utf8-syntax") => {
                self.syntax = self.syntax.utf8(false);
            }
            Arg::Long("nest-limit") => {
                let limit = args::parse(p, "--nest-limit")?;
                self.syntax = self.syntax.nest_limit(limit);
            }
            Arg::Long("octal") => {
                self.syntax = self.syntax.octal(true);
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
