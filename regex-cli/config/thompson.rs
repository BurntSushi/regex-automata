use std::borrow::Borrow;

use {
    anyhow::Context,
    lexopt::{Arg, Parser},
    regex_automata::nfa::thompson,
    regex_syntax::hir::Hir,
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
    thompson: thompson::Config,
}

impl Config {
    /// Return a ``thompson::Config` object from this configuration.
    pub fn thompson(&self) -> anyhow::Result<thompson::Config> {
        Ok(self.thompson.clone())
    }

    /// Compiles the given `Hir` expressions into an NFA. If compilation fails,
    /// then an error is returned. (And there is generally no way to know which
    /// pattern caused a failure.)
    pub fn from_hirs<H: Borrow<Hir>>(
        &self,
        hirs: &[H],
    ) -> anyhow::Result<thompson::NFA> {
        thompson::Compiler::new()
            .configure(self.thompson.clone())
            .build_many_from_hir(hirs)
            .context("failed to compile Thompson NFA")
    }
}

impl Configurable for Config {
    fn configure(
        &mut self,
        p: &mut Parser,
        arg: &mut Arg,
    ) -> anyhow::Result<bool> {
        match *arg {
            Arg::Short('B') | Arg::Long("no-utf8-nfa") => {
                self.thompson = self.thompson.clone().utf8(false);
            }
            Arg::Short('r') | Arg::Long("reverse") => {
                self.thompson = self.thompson.clone().reverse(true);
            }
            Arg::Long("nfa-size-limit") => {
                let limit = args::parse_maybe(p, "--nfa-size-limit")?;
                self.thompson = self.thompson.clone().nfa_size_limit(limit);
            }
            Arg::Long("shrink") => {
                self.thompson = self.thompson.clone().shrink(true);
            }
            Arg::Long("no-captures") => {
                self.thompson = self.thompson.clone().captures(false);
            }
            // TODO: Add LookMatcher support. That's going to have to be
            // a separate dependency sadly I think? Maybe not.
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
