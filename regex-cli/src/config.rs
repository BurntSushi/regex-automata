use std::{borrow::Borrow, convert::TryFrom, fs, path::PathBuf};

use anyhow::Context;
use automata::{
    dfa::{self, dense, onepass, sparse},
    hybrid, meta,
    nfa::thompson::{self, backtrack, pikevm},
    Anchored, MatchKind, PatternID,
};
use bstr::{BStr, BString, ByteSlice};

use crate::{
    app::{self, flag, switch, App, Args},
    escape,
    util::{self, Table},
};

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
    /// Define a --color flag on the given app.
    pub fn define(app: App) -> App {
        const SHORT: &str = "Whether to use color (default: auto).";
        const LONG: &str = "\
Whether to use color (default: auto).

When enabled, a modest amount of color is used to help make the output more
digestible, typically be enabling quick eye scanning. For example, when enabled
for the various benchmark comparison commands, the \"best\" timings are
colorized. The choices are: auto, always, never.
";
        app.arg(app::flag("color").help(SHORT).long_help(LONG))
    }

    /// Read and parse any --color options given in the CLI args. If none
    /// exist, a default is returned. If one does exist but is invalid, then an
    /// error is returned.
    pub fn get(args: &Args) -> anyhow::Result<Color> {
        if let Some(choice) = args.value_of_lossy("color") {
            choice.parse()
        } else {
            Ok(Color::Auto)
        }
    }

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

#[derive(Debug)]
pub struct Patterns(Vec<String>);

impl Patterns {
    /// Defines both a positional 'pattern' argument (which can be provided
    /// zero or more times) and a 'pattern-file' flag (which can also be
    /// provided zero or more times).
    pub fn define(mut app: App) -> App {
        {
            const SHORT: &str = "A regex pattern (must be valid UTF-8).";
            app = app.arg(app::arg("pattern").multiple(true).help(SHORT));
        }
        {
            const SHORT: &str = "Read patterns from a file.";
            app = app.arg(app::mflag("pattern-file").short("f").help(SHORT));
        }
        {
            const SHORT: &str = "Treat all patterns as literal strings.";
            app = app.arg(app::switch("fixed-strings").short("F").help(SHORT));
        }
        {
            const SHORT: &str =
                "Join all patterns into one pattern via an alternation.";
            app = app.arg(app::switch("combine-patterns").help(SHORT));
        }
        app
    }

    /// Reads at least one pattern from either positional arguments (preferred)
    /// or from pattern files. If no patterns could be found, then an error
    /// is returned.
    pub fn get(args: &Args) -> anyhow::Result<Patterns> {
        let mut pats = if let Some(pfile) = args.value_of_os("pattern-file") {
            let path = std::path::Path::new(pfile);
            let contents =
                std::fs::read_to_string(path).with_context(|| {
                    anyhow::anyhow!("failed to read {}", path.display())
                })?;
            contents.lines().map(|x| x.to_string()).collect()
        } else {
            if args.value_of_os("pattern-file").is_some() {
                anyhow::bail!(
                    "cannot provide both positional patterns and \
                     --pattern-file"
                );
            }
            let mut patterns = vec![];
            if let Some(os_patterns) = args.values_of_os("pattern") {
                for (i, p) in os_patterns.enumerate() {
                    let p = match p.to_str() {
                        Some(p) => p,
                        None => {
                            anyhow::bail!("pattern {} is not valid UTF-8", i)
                        }
                    };
                    patterns.push(p.to_string());
                }
            }
            patterns
        };
        let escape = args.is_present("fixed-strings");
        let combine = args.is_present("combine-patterns");
        if escape {
            pats = pats.iter().map(|s| syntax::escape(s)).collect();
        }
        if combine {
            pats = vec![pats.join("|")];
        }
        Ok(Patterns(pats))
    }

    /// Returns a slice of the patterns read.
    pub fn as_strings(&self) -> &[String] {
        &self.0
    }
}

impl IntoIterator for Patterns {
    type IntoIter = std::vec::IntoIter<String>;
    type Item = String;

    fn into_iter(self) -> std::vec::IntoIter<String> {
        self.0.into_iter()
    }
}

/*
#[derive(Debug)]
pub struct File(PathBuf);

impl File {
    /// Defines a single required positional parameter that accepts a file
    /// path.
    pub fn define(app: App) -> App {
        const SHORT: &str = "A file path.";
        app.arg(app::arg("file").help(SHORT).required(true))
    }

    /// Reads the file path given on the command line from the given arguments.
    pub fn get(args: &Args) -> anyhow::Result<File> {
        let f = args
            .value_of_os("file")
            .expect("expected non-None value for required 'file' argument");
        Ok(File(PathBuf::from(f)))
    }

    /// Create a file-backed read-only memory map from this file path.
    ///
    /// This is unsafe because creating memory maps is unsafe. In general,
    /// callers must assume that the underlying file is not mutated.
    pub unsafe fn mmap(&self) -> anyhow::Result<memmap2::Mmap> {
        let file = fs::File::open(&self.0)
            .with_context(|| format!("failed to open {}", self.0.display()))?;
        memmap2::Mmap::map(&file)
            .with_context(|| format!("failed to mmap {}", self.0.display()))
    }
}
*/

/// A convenient way to read haystacks from CLI parameters.
///
/// This principally provides a way for users to inline the haystack as a CLI
/// argument, or to use '@path/to/file' to give a haystack as the contents of
/// a file.
#[derive(Debug)]
pub struct Haystack {
    kind: HaystackKind,
}

/// The different ways that a haystack can be provided.
#[derive(Debug)]
enum HaystackKind {
    /// The haystack is a string given directly as a CLI argument.
    Literal(BString),
    /// The haystack is the contents of a file from the path given.
    Path(PathBuf),
}

impl Haystack {
    /// Defines a single required positional parameter that accepts input from
    /// an escaped string or a file. An escaped string is the
    /// default, where hex escape sequences like '\x7F' are recognized as their
    /// corresponding byte value.
    ///
    /// If the parameter starts with a '@', then the rest of the value is
    /// interpreted as a file path. The leading '@' cannot be escaped. To match
    /// a literal '@' in the leading position, use '\x40' or '[@]'.
    pub fn define(mut app: App) -> App {
        {
            const SHORT: &str = "An inline string or a @-prefixed file path.";
            app = app.arg(app::arg("haystack").help(SHORT).required(true));
        }
        app
    }

    /// Reads the input given on the command line from the given arguments.
    pub fn get(args: &Args) -> anyhow::Result<Haystack> {
        let haystack_arg = args
            .value_of_os("haystack")
            .expect("expected non-None value for required 'haystack' argument")
            // Converting this to a string technically makes it impossible
            // to provide a file path that contains invalid UTF-8, but
            // supporting that is a pain because of the lack of string-like
            // APIs on OsStr.
            //
            // FIXME: Maybe just require a flag to specify a file path, like
            // how patterns works?
            //
            // Keep in mind: what kind of reuse can we expect with this
            // config for a hypothetical 'regex-cli grep' tool? The grep
            // tool really just wants file paths to search... Hmm. One
            // possibility is interpreting '/re/' as a regex and everything
            // else as a file path.
            //
            // Or maybe we should just split up how a haystack is defined and
            // how input options are specified... Yeah...
            .to_string_lossy();
        let kind = if haystack_arg.as_bytes().get(0) == Some(&b'@') {
            HaystackKind::Path(PathBuf::from(&haystack_arg[1..]))
        } else {
            HaystackKind::Literal(BString::from(escape::unescape(
                &haystack_arg,
            )))
        };
        Ok(Haystack { kind })
    }

    /// If the input is a file, then memory map and pass the contents of the
    /// file to the given closure. Otherwise, if it's an inline literal, then
    /// pass it to the closure.
    pub fn with_bytes<T>(
        &self,
        mut f: impl FnMut(&BStr) -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        match self.kind {
            HaystackKind::Literal(ref lit) => f(lit.as_bstr()),
            HaystackKind::Path(ref p) => {
                let file = fs::File::open(p).with_context(|| {
                    format!("failed to open {}", p.display())
                })?;
                // SAFETY: We assume this is OK to do since we assume that our
                // search input is immutable. We specifically never try to
                // mutate the bytes from the file or treat them as anything
                // other than a slice of bytes.
                let mmap = unsafe {
                    memmap2::Mmap::map(&file).with_context(|| {
                        format!("failed to mmap {}", p.display())
                    })?
                };
                f(<&BStr>::from(&*mmap))
            }
        }
    }
}

/// This exposes all of the configuration knobs on a regex_automata::Input via
/// CLI flags. The only aspect of regex_automata::Input that this does not
/// cover is the haystack, which should be provided by other means (usually
/// with `Haystack`).
///
/// This also doesn't permit setting a prefilter at present. It's not quite
/// clear if it should go here or not, because a prefilter really needs to be
/// computed from the patterns given.
#[derive(Debug)]
pub struct Input {
    start: Option<usize>,
    end: Option<usize>,
    anchored: bool,
    pattern_id: Option<PatternID>,
    earliest: bool,
}

impl Input {
    /// Defines CLI flags on the given app that are relevant to the
    /// configuration of a regex_automata::Input, *except* for the haystack.
    /// The haystack should be supplied in another way, such as with
    /// `Haystack`.
    pub fn define(mut app: App) -> App {
        {
            const SHORT: &str = "Set the start offset of the search.";
            app = app.arg(flag("start").help(SHORT))
        }
        {
            const SHORT: &str = "Set the end offset of the search.";
            app = app.arg(flag("end").help(SHORT))
        }
        {
            const SHORT: &str = "Whether to run an anchored search.";
            app = app.arg(switch("anchored").short("a").help(SHORT))
        }
        {
            const SHORT: &str =
                "Run an anchored search for just the given pattern.";
            app = app.arg(flag("pattern-id").help(SHORT))
        }
        {
            const SHORT: &str =
                "Whether to report matches as early as possible.";
            app = app.arg(switch("earliest").help(SHORT))
        }
        app
    }

    /// Reads the input given on the command line from the given arguments.
    pub fn get(args: &Args) -> anyhow::Result<Input> {
        let mut input = Input {
            start: None,
            end: None,
            anchored: args.is_present("anchored"),
            pattern_id: None,
            earliest: args.is_present("earliest"),
        };
        if let Some(n) = args.value_of_lossy("start") {
            input.start = Some(n.parse().context("failed to parse --start")?);
        }
        if let Some(n) = args.value_of_lossy("end") {
            input.end = Some(n.parse().context("failed to parse --end")?);
        }
        if let Some(n) = args.value_of_lossy("pattern-id") {
            let pid = n.parse().context("failed to parse --pattern-id")?;
            input.pattern_id = Some(PatternID::new(pid)?);
        }
        Ok(input)
    }

    /// Return an input configuration given the haystack to search. The input
    /// configuration (other than the haystack) is drawn from the CLI flags
    /// passed for `Input`.
    pub fn input<'h>(&self, haystack: &'h [u8]) -> automata::Input<'h, '_> {
        let mut input = automata::Input::new(haystack).earliest(self.earliest);
        if let Some(start) = self.start {
            input.set_start(start);
        }
        if let Some(end) = self.end {
            input.set_end(end);
        }
        if let Some(pid) = self.pattern_id {
            input.set_anchored(Anchored::Pattern(pid));
        } else if self.anchored {
            input.set_anchored(Anchored::Yes)
        } else {
            // The default, but we set it explicitly anyway.
            input.set_anchored(Anchored::No)
        }
        input
    }

    /// Pass the regex_automata::Input configuration (derived from CLI
    /// parameters) to the closure given. Any error returned by the closure is
    /// returned by this routine.
    ///
    /// This uses `Haystack::with_bytes` to get the haystack and produce a
    /// `regex_automata::Input` configuration.
    pub fn with_input<T>(
        &self,
        haystack: &Haystack,
        mut f: impl FnMut(&automata::Input<'_, '_>) -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        haystack.with_bytes(|bytes| f(&self.input(bytes)))
    }
}

/// Flags specific to overlapping searches.
///
/// This should only be used for regex engines that support overlapping
/// searches.
#[derive(Debug)]
pub struct Overlapping {
    /// Whether to enable overlapping search or not.
    enabled: bool,
}

impl Overlapping {
    /// Defines all overlapping related flags on the given app.
    pub fn define(mut app: App) -> App {
        {
            const SHORT: &str = "Whether to run an overlapping search.";
            app = app.arg(switch("overlapping").help(SHORT));
        }
        app
    }

    /// Reads all overlapping related flags and returns its configuration.
    pub fn get(args: &Args) -> anyhow::Result<Overlapping> {
        let enabled = args.is_present("overlapping");
        Ok(Overlapping { enabled })
    }

    /// Whether to do an overlapping search or not.
    pub fn enabled(&self) -> bool {
        self.enabled
    }
}

/// Flags specific to searching for entire matches.
#[derive(Debug)]
pub struct Find {
    matches: bool,
}

impl Find {
    pub fn define(mut app: App) -> App {
        {
            const SHORT: &str = "Show the spans of each match found.";
            const LONG: &str = "\
Show the offsets of each match found.

Each match is printed on its own line. Every match contains three pieces of
information: the regex that matched, the start byte offset and the end byte
offset.
";
            app = app.arg(switch("matches").help(SHORT).long_help(LONG));
        }
        app
    }

    pub fn get(args: &Args) -> anyhow::Result<Find> {
        let matches = args.is_present("matches");
        Ok(Find { matches })
    }

    pub fn matches(&self) -> bool {
        self.matches
    }
}

/// Flags specific to searching for capturing groups.
#[derive(Debug)]
pub struct Captures {
    matches: bool,
}

impl Captures {
    pub fn define(mut app: App) -> App {
        {
            const SHORT: &str = "Show the spans of each match found.";
            const LONG: &str = "\
Show the spans of each match found.

Each match is printed on its own line. Every match includes all the capturing
groups for the corresponding regex that matched along with the spans of each
group (if they exist for the match).
";
            app = app.arg(switch("matches").help(SHORT).long_help(LONG));
        }
        app
    }

    pub fn get(args: &Args) -> anyhow::Result<Captures> {
        let matches = args.is_present("matches");
        Ok(Captures { matches })
    }

    pub fn matches(&self) -> bool {
        self.matches
    }
}

/// Handles the CLI flag for specifying the start kind of a DFA.
///
/// The default is to support both unanchored and anchored searches. Setting
/// this to anchored, for example, is useful for building smaller DFAs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StartKind {
    kind: automata::dfa::StartKind,
}

impl StartKind {
    pub fn define(app: App) -> App {
        const SHORT: &str = "Set the starting state configuration for DFAs.";
        app.arg(flag("start-kind").help(SHORT))
    }

    pub fn get(args: &Args) -> anyhow::Result<automata::dfa::StartKind> {
        use automata::dfa::StartKind;

        Ok(match args.value_of_lossy("start-kind") {
            None => StartKind::Both,
            Some(value) => match &*value {
                "both" => StartKind::Both,
                "unanchored" => StartKind::Unanchored,
                "anchored" => StartKind::Anchored,
                unk => anyhow::bail!("unrecognized start kind: {:?}", unk),
            },
        })
    }
}

#[derive(Debug)]
pub struct Syntax(automata::util::syntax::Config);

impl Syntax {
    pub fn define(mut app: App) -> App {
        {
            const SHORT: &str = "Enable case insensitive mode.";
            const LONG: &str = "\
Enable case insensitive mode.

When enabled, the regex pattern will be compiled in case insensitive mode. This
results in, for example, 'a' matching either 'a' or 'A'.

Case insensitive mode is impacted by whether Unicode mode is enabled or not.
For example, when Unicode mode is enabled, 's' will match any of 's', 'S', or
'ſ'. But when Unicode mode is disabled, 's' will only match either 's' or 'S'.

This mode can be toggled inside the regex with the 'i' flag.
";
            app = app.arg(
                switch("case-insensitive")
                    .short("i")
                    .help(SHORT)
                    .long_help(LONG),
            );
        }

        {
            const SHORT: &str = "Enable multi-line mode.";
            const LONG: &str = "\
Enable multi-line mode.

When enabled, '^' and '$' will match immediately after and immediately before
a newline, respectively, in addition to matching at the beginning and end of
the haystack.

This mode can be toggled inside the regex with the 'm' flag.
";
            app = app.arg(switch("multi-line").help(SHORT).long_help(LONG));
        }

        {
            const SHORT: &str = "Enable dot-matches-new-line mode.";
            const LONG: &str = "\
Enable dot-matches-new-line mode.

When enabled, '.' will match newlines. By default, '.' will match any Unicode
scalar value (or any byte, when Unicode mode is disabled) except for '\\n'.

This mode can be toggled inside the regex with the 's' flag.
";
            app = app.arg(
                switch("dot-matches-new-line").help(SHORT).long_help(LONG),
            );
        }

        {
            const SHORT: &str = "Enable swap-greed mode.";
            const LONG: &str = "\
Enable swap-greed mode.

When enabled, repetition operators will be ungreedy by default. Repetition
operators written to be ungreedy will in turn be greedy. That is, 'a*' becomes
'a*?' and 'a*?' becomes 'a*'.

This mode can be toggled inside the regex with the 'U' flag.
";
            app = app.arg(switch("swap-greed").help(SHORT).long_help(LONG));
        }

        {
            const SHORT: &str = "Enable whitespace insensitive mode.";
            const LONG: &str = "\
Enable whitespace insensitive mode.

When enabled, all whitespace in the regex will be considered insignificant and
ignored. Moreover, everything at and after a '#' character will be ignored and
treated as a comment. This mode is useful for writing regexes that are easier
to read.

This mode can be toggled inside the regex with the 'x' flag.
";
            app = app
                .arg(switch("ignore-whitespace").help(SHORT).long_help(LONG));
        }

        {
            const SHORT: &str = "Disable Unicode mode.";
            const LONG: &str = "\
Disable Unicode mode.

When Unicode mode is disabled, certain syntactic elements in the regex will no
longer be \"Unicode aware\".

This mode can be toggled inside the regex with the 'u' flag.
";
            app = app.arg(
                switch("no-unicode").short("U").help(SHORT).long_help(LONG),
            );
        }

        {
            const SHORT: &str =
                "Allow matching invalid UTF-8 (arbitrary bytes).";
            const LONG: &str = "\
Allow matching invalid UTF-8, or equivalently, allow matching arbitrary bytes.

When UTF-8 mode is disabled, the regex is permitted to match arbitrary
bytes. Otherwise, when UTF-8 mode is enabled, all regexes are guaranteed to
match valid UTF-8 or will otherwise fail to compile. Disabling UTF-8 mode is
sometimes necessary when Unicode mode is disabled. For example, a '.' when
Unicode mode is disabled with match any byte except for '\\n', which means it
can match invalid UTF-8. Therefore, the only way to compile '.' when Unicode
mode is disabled is to also disable UTF-8 mode.

This mode cannot be toggled inside the regex.
";
            app = app.arg(
                switch("no-utf8-syntax")
                    .short("b")
                    .help(SHORT)
                    .long_help(LONG),
            );
        }

        {
            const SHORT: &str = "Set a nest limit.";
            const LONG: &str = "\
Set a nesting limit on the regex pattern.

Patterns with a nesting level greater than this limit will fail to compile.
";
            app = app.arg(flag("nest-limit").help(SHORT).long_help(LONG));
        }

        {
            const SHORT: &str = "Enable octal escape sequences.";
            const LONG: &str = "\
Enable octal escape sequences.

When enabled, the syntax '\\123' can be used to match a Unicode scalar value
matching a number written in octal notation.

This is disabled by default since it is rarely used, and when disabled, permits
emitting better error messages warning users that backreferences are not
supported.

This mode cannot be toggled inside the regex.
";
            app = app.arg(switch("octal").help(SHORT).long_help(LONG));
        }

        app
    }

    pub fn get(args: &Args) -> anyhow::Result<Syntax> {
        let mut c = automata::util::syntax::Config::new()
            .case_insensitive(args.is_present("case-insensitive"))
            .multi_line(args.is_present("multi-line"))
            .dot_matches_new_line(args.is_present("dot-matches-new-line"))
            .swap_greed(args.is_present("swap-greed"))
            .ignore_whitespace(args.is_present("ignore-whitespace"))
            .unicode(!args.is_present("no-unicode"))
            .utf8(!args.is_present("no-utf8-syntax"))
            .octal(args.is_present("octal"));
        if let Some(n) = args.value_of_lossy("nest-limit") {
            let limit = n.parse().context("failed to parse --nest-limit")?;
            c = c.nest_limit(limit);
        }
        Ok(Syntax(c))
    }

    pub fn ast(&self, pattern: &str) -> anyhow::Result<syntax::ast::Ast> {
        syntax::ast::parse::ParserBuilder::new()
            .nest_limit(self.0.get_nest_limit())
            .octal(self.0.get_octal())
            .ignore_whitespace(self.0.get_ignore_whitespace())
            .build()
            .parse(pattern)
            .context("failed to parse pattern")
    }

    pub fn asts<P: AsRef<str>>(
        &self,
        patterns: &[P],
    ) -> anyhow::Result<Vec<syntax::ast::Ast>> {
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

    pub fn hir(
        &self,
        pattern: &str,
        ast: &syntax::ast::Ast,
    ) -> anyhow::Result<syntax::hir::Hir> {
        syntax::hir::translate::TranslatorBuilder::new()
            .utf8(self.0.get_utf8())
            .case_insensitive(self.0.get_case_insensitive())
            .multi_line(self.0.get_multi_line())
            .dot_matches_new_line(self.0.get_dot_matches_new_line())
            .swap_greed(self.0.get_swap_greed())
            .unicode(self.0.get_unicode())
            .build()
            .translate(pattern, ast)
            .context("failed to translate pattern")
    }

    pub fn hirs<P: AsRef<str>, A: Borrow<syntax::ast::Ast>>(
        &self,
        patterns: &[P],
        asts: &[A],
    ) -> anyhow::Result<Vec<syntax::hir::Hir>> {
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

#[derive(Debug)]
pub struct Thompson(thompson::Config);

impl Thompson {
    pub fn define(mut app: App) -> App {
        {
            const SHORT: &str = "Disable UTF-8 mode for the NFA.";
            const LONG: &str = "\
Disables UTF-8 mode for the NFA.

This mode cannot be toggled inside the regex.
";
            app = app.arg(
                switch("no-utf8-nfa").short("B").help(SHORT).long_help(LONG),
            );
        }
        {
            const SHORT: &str = "Compile a reverse NFA.";
            const LONG: &str = "\
Compile a reverse NFA.

The NFA compiled will match the regex provided in reverse. That is, it matches
as if starting from the end of the input instead of the beginning.

Typically, a reverse NFA is never used for matching directly, since a forward
NFA executed via the Pike VM can find the start and end location of a match in
a single pass. Instead, a reverse NFA is used to build a DFA or a lazy DFA to
perform a reverse search that is used to find the starting location of a match.
";
            app = app
                .arg(switch("reverse").short("r").help(SHORT).long_help(LONG));
        }
        {
            const SHORT: &str =
                "Set a size limit, in bytes, on the NFA compiled.";
            const LONG: &str = "\
Set a size limit, in bytes, on the NFA compiled.

This permits imposing constraints on the size of a compiled NFA. This may be
useful in contexts where the regex pattern is untrusted and one wants to avoid
using too much memory.

This size limit does not apply to auxiliary heap used during compilation that
is not part of the built NFA.

Note that this size limit is applied during compilation in order for the limit
to prevent too much heap from being used. However, the implementation may use
an intermediate NFA representation that is otherwise slightly bigger than the
final public form. Since the size limit may be applied to an intermediate
representation, there is not necessarily a precise correspondence between the
configured size limit and the heap usage of the final NFA.

The default for this flag is 'none', which sets no size limit.
";
            app = app.arg(flag("nfa-size-limit").help(SHORT).long_help(LONG));
        }
        {
            const SHORT: &str = "Enable NFA shrinking.";
            const LONG: &str = "\
Enable NFA shrinking.

By default, when compiling an NFA, faster compilation is preferred over
smaller NFAs. When NFA shrinking is enabled, the trade off is reversed.
Namely, some extra effort is expended to reduce the size of the NFA. While
implementation details may change, currently this only occurs when compiling
large Unicode character classes in reverse. This extra work can make NFA
compilation slower. However, the reduction in size can be big enough to cause a
dramatic performance improvement when the NFA is determinized into a DFA.
";
            app = app.arg(switch("shrink").help(SHORT).long_help(LONG));
        }
        {
            const SHORT: &str = "Do not include capturing groups.";
            const LONG: &str = "\
Do not include capturing groups in the NFA.

By default, when compiling an NFA, capturing groups will be included. They are
represented as unconditional epsilon transitions in the NFA graph, and permit
an NFA simulation to record information (such as the current position) when
a search passes through them. Each capturing group gets translated into two
NFA states representing distinct \"slots.\" These slots are represented by
indices. An NFA simulation can use these indices to record the aforementioned
information.

Note that even if the pattern does not have any explicit capturing groups in
them, at least one such capturing group always implicitly exists: the capturing
group corresponding to the entire match.

These are useful to disable primarily in two cases. First is that getting rid
of them may make the NFA easier for a human to read and analyze. Second is that
they are useless when building a DFA, since a DFA doesn't (and cannot) support
capturing groups. So in the context of building a DFA, capturing group NFA
states are precisely equivalent to unconditional epsilon transitions.

This tool will error if you try to use something that does require capturing
groups (such as a search with the PikeVM).
";
            app = app.arg(switch("no-captures").help(SHORT).long_help(LONG));
        }

        app
    }

    pub fn get(args: &Args) -> anyhow::Result<Thompson> {
        let mut c = thompson::Config::new()
            .utf8(!args.is_present("no-utf8-nfa"))
            .reverse(args.is_present("reverse"))
            .shrink(args.is_present("shrink"))
            .captures(!args.is_present("no-captures"));
        if let Some(x) = args.value_of_lossy("nfa-size-limit") {
            if x.to_lowercase() == "none" {
                c = c.nfa_size_limit(None);
            } else {
                let limit =
                    x.parse().context("failed to parse --nfa-size-limit")?;
                c = c.nfa_size_limit(Some(limit));
            }
        }
        Ok(Thompson(c))
    }

    pub fn from_hirs<H: Borrow<syntax::hir::Hir>>(
        &self,
        exprs: &[H],
    ) -> anyhow::Result<thompson::NFA> {
        thompson::Compiler::new()
            .configure(self.0.clone())
            .build_many_from_hir(exprs)
            .context("failed to compile Thompson NFA")
    }
}

#[derive(Debug)]
pub struct PikeVM {
    config: pikevm::Config,
}

impl PikeVM {
    pub fn define(mut app: App) -> App {
        {
            const SHORT: &str = "Choose the match kind.";
            const LONG: &str = "\
Choose the match kind.

This permits setting the match kind to either 'leftmost-first' (the default)
or 'all'. The former will attempt to find the longest match starting at the
leftmost position, but prioritizing alternations in the regex that appear
first. For example, with leftmost-first enabled, 'Sam|Samwise' will match 'Sam'
in 'Samwise' while 'Samwise|Sam' would match 'Samwise'.

'all' match semantics will include all possible matches, including the longest
possible match. 'all' is most commonly used when compiling a reverse DFA to
determine the starting position of a match. Note that when 'all' is used, there
is no distinction between greedy and non-greedy regexes. Everything is greedy
all the time.
";
            app = app.arg(
                flag("match-kind").short("k").help(SHORT).long_help(LONG),
            );
        }
        app
    }

    pub fn get(args: &Args) -> anyhow::Result<PikeVM> {
        let kind = match args.value_of_lossy("match-kind") {
            None => MatchKind::LeftmostFirst,
            Some(value) => match &*value {
                "all" => MatchKind::All,
                "leftmost-first" => MatchKind::LeftmostFirst,
                unk => anyhow::bail!("unrecognized match kind: {:?}", unk),
            },
        };
        let config = pikevm::Config::new().match_kind(kind);
        Ok(PikeVM { config })
    }

    pub fn builder(
        &self,
        syntax: &Syntax,
        thompson: &Thompson,
    ) -> pikevm::Builder {
        let mut builder = pikevm::PikeVM::builder();
        builder
            .configure(self.config.clone())
            .syntax(syntax.0)
            .thompson(thompson.0.clone());
        builder
    }

    pub fn from_patterns(
        &self,
        table: &mut Table,
        syntax: &Syntax,
        thompson: &Thompson,
        patterns: &Patterns,
    ) -> anyhow::Result<pikevm::PikeVM> {
        let patterns = patterns.as_strings();
        let b = self.builder(syntax, thompson);
        let (vm, time) = util::timeitr(|| b.build_many(patterns))?;
        table.add("build pike vm time", time);
        Ok(vm)
    }
}

#[derive(Debug)]
pub struct Backtrack {
    config: backtrack::Config,
}

impl Backtrack {
    pub fn define(mut app: App) -> App {
        {
            const SHORT: &str =
                "Set the visited set capacity used to bound backtracking.";
            const LONG: &str = "\
Set the visited set capacity used to bound backtracking.

The visited capacity represents the amount of heap memory (in bytes) to
allocate toward tracking which parts of the backtracking search have been done
before. The heap memory needed for any particular search is proportional to
'haystack.len() * nfa.states().len()', whichc an be quite large. Therefore, the
bounded backtracker is typically only able to run on shorter haystacks.

For a given regex, increasing the visited capacity means that the
maximum haystack length that can be searched is increased.

The default capacity is a reasonable but empirically chosen size.
";
            app =
                app.arg(flag("visited-capacity").help(SHORT).long_help(LONG));
        }
        app
    }

    pub fn get(args: &Args) -> anyhow::Result<Backtrack> {
        let mut config = backtrack::Config::new();
        if let Some(x) = args.value_of_lossy("visited-capacity") {
            let limit =
                x.parse().context("failed to parse --visited-capacity")?;
            config = config.visited_capacity(limit);
        }
        Ok(Backtrack { config })
    }

    pub fn builder(
        &self,
        syntax: &Syntax,
        thompson: &Thompson,
    ) -> backtrack::Builder {
        let mut builder = backtrack::BoundedBacktracker::builder();
        builder
            .configure(self.config.clone())
            .syntax(syntax.0)
            .thompson(thompson.0.clone());
        builder
    }

    pub fn from_patterns(
        &self,
        table: &mut Table,
        syntax: &Syntax,
        thompson: &Thompson,
        patterns: &Patterns,
    ) -> anyhow::Result<backtrack::BoundedBacktracker> {
        let patterns = patterns.as_strings();
        let b = self.builder(syntax, thompson);
        let (vm, time) = util::timeitr(|| b.build_many(patterns))?;
        table.add("build bounded backtracker time", time);
        Ok(vm)
    }
}

#[derive(Debug)]
pub struct OnePass {
    config: onepass::Config,
}

impl OnePass {
    pub fn define(mut app: App) -> App {
        {
            const SHORT: &str = "Disable the use of equivalence classes.";
            const LONG: &str = "\
Disable the use of equivalence classes.

When disabled, every state in the one-pass DFA will always have 256 transitions
When enabled (the default), transitions are grouped into equivalence classes
where every byte in the same class cannot possibly differentiate between a
match and a non-match.

Enabling byte classes is always a good idea, since it both decreases the
amount of space required and also the amount of time it takes to build the DFA
(since there are fewer transitions to create). The only reason to disable byte
classes is for debugging the representation of a DFA, since equivalence class
identifiers will be used for the transitions instead of the actual bytes.
";
            app = app.arg(
                switch("no-byte-classes")
                    .short("C")
                    .help(SHORT)
                    .long_help(LONG),
            );
        }
        {
            const SHORT: &str = "Choose the match kind.";
            const LONG: &str = "\
Choose the match kind.

This permits setting the match kind to either 'leftmost-first' (the default)
or 'all'. The former will attempt to find the longest match starting at the
leftmost position, but prioritizing alternations in the regex that appear
first. For example, with leftmost-first enabled, 'Sam|Samwise' will match 'Sam'
in 'Samwise' while 'Samwise|Sam' would match 'Samwise'.

'all' match semantics will include all possible matches, including the longest
possible match. Note that when 'all' is used, there is no distinction between
greedy and non-greedy regexes. Everything is greedy all the time.
";
            app = app.arg(
                flag("match-kind").short("k").help(SHORT).long_help(LONG),
            );
        }
        {
            const SHORT: &str = "Add start states for each pattern.";
            const LONG: &str = "\
Whether to compile a separate start state for each pattern in the automaton.

When enabled, a separate anchored start state is added for each pattern in the
one-pass DFA. When this start state is used, then the DFA will only search for
matches for the pattern, even if there are other patterns in the DFA.

The main downside of this option is that it can potentially increase the size
of the DFA and/or increase the time it takes to build the DFA.

The main reason one might want to enable this (it's disabled by
default) for one-pass DFAs specifically is to enable searching for specific
patterns instead of any pattern. Otherwise, you would need to compile a new
DFA for each pattern.

By default this is disabled.
";
            app = app.arg(
                switch("starts-for-each-pattern").help(SHORT).long_help(LONG),
            );
        }
        {
            const SHORT: &str =
                "Set a size limit, in bytes, on the compiled one-pass DFA.";
            const LONG: &str = "\
Set a size limit, in bytes, on the compiled one-pass DFA.

This size limit is expressed in bytes and is applied when attempting to convert
an NFA into a one-pass DFA. If the DFA's heap usage, and only the DFA, exceeds
this configured limit, then construction is stopped and an error is returned.

This limit does not apply to auxiliary storage used during construction that
isn't part of the generated DFA.

The default for this flag is 'none', which sets no size limit.
";
            app = app.arg(flag("dfa-size-limit").help(SHORT).long_help(LONG));
        }
        app
    }

    pub fn get(args: &Args) -> anyhow::Result<OnePass> {
        let kind = match args.value_of_lossy("match-kind") {
            None => MatchKind::LeftmostFirst,
            Some(value) => match &*value {
                "all" => MatchKind::All,
                "leftmost-first" => MatchKind::LeftmostFirst,
                unk => anyhow::bail!("unrecognized match kind: {:?}", unk),
            },
        };
        let mut c = onepass::Config::new()
            .byte_classes(!args.is_present("no-byte-classes"))
            .match_kind(kind)
            .starts_for_each_pattern(
                args.is_present("starts-for-each-pattern"),
            );
        if let Some(x) = args.value_of_lossy("dfa-size-limit") {
            if x.to_lowercase() == "none" {
                c = c.size_limit(None);
            } else {
                let limit =
                    x.parse().context("failed to parse --dfa-size-limit")?;
                c = c.size_limit(Some(limit));
            }
        }
        Ok(OnePass { config: c })
    }

    pub fn from_nfa(
        &self,
        nfa: thompson::NFA,
    ) -> anyhow::Result<onepass::DFA> {
        onepass::Builder::new()
            .configure(self.config.clone())
            .build_from_nfa(nfa)
            .context("failed to compile one-pass DFA")
    }

    pub fn from_patterns(
        &self,
        table: &mut Table,
        syntax: &Syntax,
        thompson: &Thompson,
        patterns: &Patterns,
    ) -> anyhow::Result<onepass::DFA> {
        let patterns = patterns.as_strings();

        let (asts, time) = util::timeitr(|| syntax.asts(patterns))?;
        table.add("parse time", time);
        let (hirs, time) = util::timeitr(|| syntax.hirs(patterns, &asts))?;
        table.add("translate time", time);
        let (nfa, time) = util::timeitr(|| thompson.from_hirs(&hirs))?;
        table.add("compile nfa time", time);
        table.add("nfa states", nfa.states().len());
        table.add("nfa memory", nfa.memory_usage());
        let (dfa, time) = util::timeitr(|| self.from_nfa(nfa))?;
        table.add("compile one-pass dfa time", time);
        table.add("one-pass states", dfa.state_len());
        table.add("one-pass memory", dfa.memory_usage());
        table.add("one-pass cache memory", dfa.create_cache().memory_usage());
        table.add("one-pass alphabet length", dfa.alphabet_len());
        table.add("one-pass stride", 1 << dfa.stride2());

        Ok(dfa)
    }
}

#[derive(Debug)]
pub struct Dense {
    config: dense::Config,
}

impl Dense {
    pub fn define(mut app: App) -> App {
        app = StartKind::define(app);
        {
            const SHORT: &str = "Disable DFA state acceleration.";
            const LONG: &str = "\
Disable DFA state acceleration.

When enabled (the default), the DFA compilation process will attempt to
identify states that are eligible to be accelerated. A state can be accelerated
when there are a very small set of bytes that must be seen in order for the DFA
to leave that state. (Where every other byte would represent a transition back
to the same state.)

When disabled, no acceleration is performed. Generally speaking, acceleration
is more common when Unicode mode is disabled, as states tend to be simpler.
";
            app = app.arg(
                switch("no-accelerate").short("A").help(SHORT).long_help(LONG),
            );
        }
        {
            const SHORT: &str = "Minimize the DFA.";
            const LONG: &str = "\
Minimize the DFA.

When enabled, the DFA will be minimized such that it will be as small as
possible. This is useful when generating DFAs to embed in a Rust program, since
this can dramatically decrease the space used by a DFA. The disadvantage of
minimization is that it is typically very costly to do in both space and time.

Minimization is disabled by default.
";
            app = app.arg(
                switch("minimize").short("m").help(SHORT).long_help(LONG),
            );
        }
        {
            const SHORT: &str = "Disable the use of equivalence classes.";
            const LONG: &str = "\
Disable the use of equivalence classes.

When disabled, every state in the DFA will always have 257 transitions (256 for
each possible byte and 1 more for the special end-of-input transition). When
enabled (the default), transitions are grouped into equivalence classes where
every byte in the same class cannot possible differentiate between a match and
a non-match.

Enabling byte classes is always a good idea, since it both decreases the
amount of space required and also the amount of time it takes to build the DFA
(since there are fewer transitions to create). The only reason to disable byte
classes is for debugging the representation of a DFA, since equivalence class
identifiers will be used for the transitions instead of the actual bytes.
";
            app = app.arg(
                switch("no-byte-classes")
                    .short("C")
                    .help(SHORT)
                    .long_help(LONG),
            );
        }
        {
            const SHORT: &str = "Choose the match kind.";
            const LONG: &str = "\
Choose the match kind.

This permits setting the match kind to either 'leftmost-first' (the default)
or 'all'. The former will attempt to find the longest match starting at the
leftmost position, but prioritizing alternations in the regex that appear
first. For example, with leftmost-first enabled, 'Sam|Samwise' will match 'Sam'
in 'Samwise' while 'Samwise|Sam' would match 'Samwise'.

'all' match semantics will include all possible matches, including the longest
possible match. 'all' is most commonly used when compiling a reverse DFA to
determine the starting position of a match. Note that when 'all' is used, there
is no distinction between greedy and non-greedy regexes. Everything is greedy
all the time.
";
            app = app.arg(
                flag("match-kind").short("k").help(SHORT).long_help(LONG),
            );
        }
        {
            const SHORT: &str = "Add start states for each pattern.";
            const LONG: &str = "\
Whether to compile a separate start state for each pattern in the automaton.

When enabled, a separate anchored start state is added for each pattern in the
DFA. When this start state is used, then the DFA will only search for matches
for the pattern, even if there are other patterns in the DFA.

The main downside of this option is that it can potentially increase the size
of the DFA and/or increase the time it takes to build the DFA.

There are a few reasons one might want to enable this (it's disabled by
default):

1. When looking for the start of an overlapping match (using a reverse DFA),
doing it correctly requires starting the reverse search using the starting
state of the pattern that matched in the forward direction.

2. When you want to use a DFA with multiple patterns to both search for matches
of any pattern or to search for matches of one particular pattern while
using the same DFA. (Otherwise, you would need to compile a new DFA for each
pattern.)

3. Since the start states added for each pattern are anchored, if you compile
an unanchored DFA with one pattern while also enabling this option, then you
can use the same DFA to perform anchored or unanchored searches.

By default this is disabled.
";
            app = app.arg(
                switch("starts-for-each-pattern").help(SHORT).long_help(LONG),
            );
        }
        {
            const SHORT: &str =
                "Heuristically enable Unicode word boundaries.";
            const LONG: &str = "\
Heuristically enable Unicode word boundaries.

When enabled, the DFA will attempt to match Unicode word boundaries by assuming
they are ASCII word boundaries. To ensure that incorrect or missing matches
are avoid, the DFA will be configured to automatically quit whenever it sees
a non-ASCII byte. (In which case, the user must return an error or try a
different regex engine that supports Unicode word boundaries.)

Since enabling this may cause the DFA to give up on a search without reporting
either a match or a non-match, callers using this must be prepared to handle
an error at search time.

This is disabled by default.
";
            app = app.arg(
                switch("unicode-word-boundary")
                    .short("w")
                    .help(SHORT)
                    .long_help(LONG),
            );
        }
        {
            const SHORT: &str = "Set the quit bytes for this DFA.";
            const LONG: &str = "\
Set the quit bytes for this DFA.

This enables one to explicitly set the bytes which should trigger a DFA to quit
during searching without reporting either a match or a non-match. This is the
same mechanism by which the --unicode-word-boundaries flag works, but provides
a way for callers to explicitly control which bytes cause a DFA to quit for
their own application. For example, this can be useful to set if one wants to
report matches on a line-by-line basis without first splitting the haystack
into lines.

Currently, all bytes specified must be in ASCII, but this restriction may be
lifted in the future. Bytes can be specified using a single value. e.g.,

    --quit abc

Will cause the DFA to quit whenever it sees one of 'a', 'b' or 'c'.
";
            app = app.arg(switch("quit").help(SHORT).long_help(LONG));
        }
        {
            const SHORT: &str = "Enable start state specialization.";
            const LONG: &str = "\
Enable start state specialization.

When start states are specialized, an implementor of a search routine using a
DFA can tell when the search has entered a starting state. When start states
aren't specialized, then it is impossible to know whether the search has
entered a start state.

Ideally, this option wouldn't need to exist and we could always specialize
start states. The problem is that start states can be quite active. This in
turn means that an efficient search routine is likely to ping-pong between a
heavily optimized hot loop that handles most states and to a less optimized
specialized handling of start states. This causes branches to get heavily
mispredicted and overall can materially decrease throughput. Therefore,
specializing start states should only be enabled when it is needed.

Knowing whether a search is in a start state is typically useful when a
prefilter is active for the search. A prefilter is typically only run when in
a start state and a prefilter can greatly accelerate a search. Therefore, the
possible cost of specializing start states is worth it in this case. Otherwise,
if you have no prefilter, there is likely no reason to specialize start states.

This is disabled by default.
";
            app = app.arg(
                switch("specialize-start-states").help(SHORT).long_help(LONG),
            );
        }
        {
            const SHORT: &str =
                "Set a size limit, in bytes, on the compiled DFA.";
            const LONG: &str = "\
Set a size limit, in bytes, on the compiled DFA.

This size limit is expressed in bytes and is applied during determinization
of an NFA into a DFA. If the DFA's heap usage, and only the DFA, exceeds this
configured limit, then determinization is stopped and an error is returned.

This limit does not apply to auxiliary storage used during determinization that
isn't part of the generated DFA.

This limit is only applied during determinization. Currently, there is no way
to post-pone this check to after minimization if minimization was enabled.

The total limit on heap used during determinization is the sum of the DFA and
determinization size limits.

The default for this flag is 'none', which sets no size limit.
";
            app = app.arg(flag("dfa-size-limit").help(SHORT).long_help(LONG));
        }
        {
            const SHORT: &str =
                "Set a size limit, in bytes, to be used by determinization.";
            const LONG: &str = "\
Set a size limit, in bytes, to be used by determinization.

This size limit is expressed in bytes and is applied during determinization
of an NFA into a DFA. If the heap used for auxiliary storage during
determinization (memory that is not in the DFA but necessary for building the
DFA) exceeds this configured limit, then determinization is stopped and an
error is returned.

This limit does not apply to heap used by the DFA itself.

The total limit on heap used during determinization is the sum of the DFA and
determinization size limits.

The default for this flag is 'none', which sets no size limit.
";
            app = app.arg(
                flag("determinize-size-limit").help(SHORT).long_help(LONG),
            );
        }
        app
    }

    pub fn get(args: &Args) -> anyhow::Result<Dense> {
        let start_kind = StartKind::get(args)?;
        let match_kind = match args.value_of_lossy("match-kind") {
            None => MatchKind::LeftmostFirst,
            Some(value) => match &*value {
                "all" => MatchKind::All,
                "leftmost-first" => MatchKind::LeftmostFirst,
                unk => anyhow::bail!("unrecognized match kind: {:?}", unk),
            },
        };
        let mut c = dense::Config::new()
            .accelerate(!args.is_present("no-accelerate"))
            .minimize(args.is_present("minimize"))
            .byte_classes(!args.is_present("no-byte-classes"))
            .match_kind(match_kind)
            .start_kind(start_kind)
            .starts_for_each_pattern(
                args.is_present("starts-for-each-pattern"),
            )
            .unicode_word_boundary(args.is_present("unicode-word-boundary"))
            .specialize_start_states(
                args.is_present("specialize-start-states"),
            );
        if let Some(quits) = args.value_of_lossy("quit") {
            for ch in quits.chars() {
                if !ch.is_ascii() {
                    anyhow::bail!("quit bytes must be ASCII");
                }
                // MSRV(1.59): use the 'TryFrom<char> for u8' impl.
                c = c.quit(u8::try_from(u32::from(ch)).unwrap(), true);
            }
        }
        if let Some(x) = args.value_of_lossy("dfa-size-limit") {
            if x.to_lowercase() == "none" {
                c = c.dfa_size_limit(None);
            } else {
                let limit =
                    x.parse().context("failed to parse --dfa-size-limit")?;
                c = c.dfa_size_limit(Some(limit));
            }
        }
        if let Some(x) = args.value_of_lossy("determinize-size-limit") {
            if x.to_lowercase() == "none" {
                c = c.determinize_size_limit(None);
            } else {
                let limit = x
                    .parse()
                    .context("failed to parse --determinize-size-limit")?;
                c = c.determinize_size_limit(Some(limit));
            }
        }
        Ok(Dense { config: c })
    }

    pub fn from_nfa(
        &self,
        nfa: &thompson::NFA,
    ) -> anyhow::Result<dense::DFA<Vec<u32>>> {
        dense::Builder::new()
            .configure(self.config.clone())
            .build_from_nfa(nfa)
            .context("failed to compile dense DFA")
    }

    pub fn from_patterns_dense(
        &self,
        table: &mut Table,
        syntax: &Syntax,
        thompson: &Thompson,
        dense: &Dense,
        patterns: &Patterns,
    ) -> anyhow::Result<dense::DFA<Vec<u32>>> {
        let patterns = patterns.as_strings();

        let (asts, time) = util::timeitr(|| syntax.asts(patterns))?;
        table.add("parse time", time);
        let (hirs, time) = util::timeitr(|| syntax.hirs(patterns, &asts))?;
        table.add("translate time", time);
        let (nfa, time) = util::timeitr(|| thompson.from_hirs(&hirs))?;
        table.add("compile nfa time", time);
        table.add("nfa memory", nfa.memory_usage());
        let (dfa, time) = util::timeitr(|| dense.from_nfa(&nfa))?;
        table.add("compile dense dfa time", time);
        table.add("dense dfa memory", dfa.memory_usage());
        table.add("dense alphabet length", dfa.alphabet_len());
        table.add("dense stride", 1 << dfa.stride2());

        Ok(dfa)
    }

    pub fn from_patterns_sparse(
        &self,
        table: &mut Table,
        syntax: &Syntax,
        thompson: &Thompson,
        dense: &Dense,
        patterns: &Patterns,
    ) -> anyhow::Result<sparse::DFA<Vec<u8>>> {
        let dfa = self
            .from_patterns_dense(table, syntax, thompson, dense, patterns)?;
        let (sdfa, time) = util::timeitr(|| dfa.to_sparse())?;
        table.add("compile sparse dfa time", time);
        table.add("sparse dfa memory", sdfa.memory_usage());

        Ok(sdfa)
    }
}

#[derive(Debug)]
pub struct RegexDFA {
    /// DFA regexes have no config currently.
    _nothing: (),
}

impl RegexDFA {
    pub fn define(app: App) -> App {
        app
    }

    pub fn get(_args: &Args) -> anyhow::Result<RegexDFA> {
        Ok(RegexDFA { _nothing: () })
    }

    pub fn builder(
        &self,
        syntax: &Syntax,
        thompson: &Thompson,
        dense: &Dense,
    ) -> dfa::regex::Builder {
        let mut builder = dfa::regex::Builder::new();
        builder
            .syntax(syntax.0)
            .thompson(thompson.0.clone())
            .dense(dense.config.clone());
        builder
    }

    pub fn from_patterns_dense(
        &self,
        table: &mut Table,
        syntax: &Syntax,
        thompson: &Thompson,
        dense: &Dense,
        patterns: &Patterns,
    ) -> anyhow::Result<dfa::regex::Regex<dense::DFA<Vec<u32>>>> {
        let patterns = patterns.as_strings();
        let b = self.builder(syntax, thompson, dense);
        let (re, time) = util::timeitr(|| b.build_many(patterns))?;
        let mem_fwd = re.forward().memory_usage();
        let mem_rev = re.forward().memory_usage();
        table.add("compile dense regex time", time);
        table.add("dense regex (forward DFA) memory", mem_fwd);
        table.add("dense regex (reverse DFA) memory", mem_rev);
        table.add("dense regex memory", mem_fwd + mem_rev);
        Ok(re)
    }

    pub fn from_patterns_sparse(
        &self,
        table: &mut Table,
        syntax: &Syntax,
        thompson: &Thompson,
        dense: &Dense,
        patterns: &Patterns,
    ) -> anyhow::Result<dfa::regex::Regex<sparse::DFA<Vec<u8>>>> {
        let re = self
            .from_patterns_dense(table, syntax, thompson, dense, patterns)?;
        let (sre, time) = util::timeitr(|| {
            let (fwd, rev) = (re.forward(), re.reverse());
            fwd.to_sparse().and_then(|f| {
                rev.to_sparse().map(|r| {
                    let b = self.builder(syntax, thompson, dense);
                    b.build_from_dfas(f, r)
                })
            })
        })?;
        let mem_fwd = sre.forward().memory_usage();
        let mem_rev = sre.forward().memory_usage();
        table.add("compile sparse regex time", time);
        table.add("sparse regex (forward DFA) memory", mem_fwd);
        table.add("sparse regex (reverse DFA) memory", mem_rev);
        table.add("sparse regex memory", mem_fwd + mem_rev);
        Ok(sre)
    }
}

#[derive(Debug)]
pub struct Hybrid {
    config: hybrid::dfa::Config,
}

impl Hybrid {
    pub fn define(mut app: App) -> App {
        {
            const SHORT: &str = "Disable the use of equivalence classes.";
            const LONG: &str = "\
Disable the use of equivalence classes.

When disabled, every state in the lazy DFA will always have 257 transitions
(256 for each possible byte and 1 more for the special end-of-input
transition). When enabled (the default), transitions are grouped into
equivalence classes where every byte in the same class cannot possible
differentiate between a match and a non-match.

Enabling byte classes is always a good idea, since it both decreases the
amount of space required and also the amount of time it takes to build the DFA
(since there are fewer transitions to create). The only reason to disable byte
classes is for debugging the representation of a DFA, since equivalence class
identifiers will be used for the transitions instead of the actual bytes.
";
            app = app.arg(
                switch("no-byte-classes")
                    .short("C")
                    .help(SHORT)
                    .long_help(LONG),
            );
        }
        {
            const SHORT: &str = "Choose the match kind.";
            const LONG: &str = "\
Choose the match kind.

This permits setting the match kind to either 'leftmost-first' (the default)
or 'all'. The former will attempt to find the longest match starting at the
leftmost position, but prioritizing alternations in the regex that appear
first. For example, with leftmost-first enabled, 'Sam|Samwise' will match 'Sam'
in 'Samwise' while 'Samwise|Sam' would match 'Samwise'.

'all' match semantics will include all possible matches, including the longest
possible match. 'all' is most commonly used when compiling a reverse lazy DFA
to determine the starting position of a match. Note that when 'all' is used,
there is no distinction between greedy and non-greedy regexes. Everything is
greedy all the time.
";
            app = app.arg(
                flag("match-kind").short("k").help(SHORT).long_help(LONG),
            );
        }
        {
            const SHORT: &str = "Add start states for each pattern.";
            const LONG: &str = "\
Whether to compile a separate start state for each pattern in the automaton.

When enabled, a separate anchored start state is added for each pattern in the
lazy DFA. When this start state is used, then the lazy DFA will only search for
matches for the pattern, even if there are other patterns in the lazy DFA.

The main downside of this option is that it can potentially increase the size
of the lazy DFA.

There are a few reasons one might want to enable this (it's disabled by
default):

1. When looking for the start of an overlapping match (using a reverse lazy
DFA), doing it correctly requires starting the reverse search using the
starting state of the pattern that matched in the forward direction.

2. When you want to use a lazy DFA with multiple patterns to both search for
matches of any pattern or to search for matches of one particular pattern while
using the same lazy DFA. (Otherwise, you would need to build a new lazy DFA
for each pattern.)

3. Since the start states added for each pattern are anchored, if you build an
unanchored lazy DFA with one pattern while also enabling this option, then you
can use the same lazy DFA to perform anchored or unanchored searches.

By default this is disabled.
";
            app = app.arg(
                switch("starts-for-each-pattern").help(SHORT).long_help(LONG),
            );
        }
        {
            const SHORT: &str =
                "Heuristically enable Unicode word boundaries.";
            const LONG: &str = "\
Heuristically enable Unicode word boundaries.

When enabled, the lazy DFA will attempt to match Unicode word boundaries by
assuming they are ASCII word boundaries. To ensure that incorrect or missing
matches are avoid, the lazy DFA will be configured to automatically quit
whenever it sees a non-ASCII byte. (In which case, the user must return an
error or try a different regex engine that supports Unicode word boundaries.)

Since enabling this may cause the lazy DFA to give up on a search without
reporting either a match or a non-match, callers using this must be prepared to
handle an error at search time.

This is disabled by default.
";
            app = app.arg(
                switch("unicode-word-boundary")
                    .short("w")
                    .help(SHORT)
                    .long_help(LONG),
            );
        }
        {
            const SHORT: &str = "Set the quit bytes for this lazy DFA.";
            const LONG: &str = "\
Set the quit bytes for this lazy DFA.

This enables one to explicitly set the bytes which should trigger a lazy DFA
to quit during searching without reporting either a match or a non-match. This
is the same mechanism by which the --unicode-word-boundaries flag works, but
provides a way for callers to explicitly control which bytes cause a lazy DFA
to quit for their own application. For example, this can be useful to set if
one wants to report matches on a line-by-line basis without first splitting the
haystack into lines.

Currently, all bytes specified must be in ASCII, but this restriction may be
lifted in the future. Bytes can be specified using a single value. e.g.,

    --quit abc

Will cause the lazy DFA to quit whenever it sees one of 'a', 'b' or 'c'.
";
            app = app.arg(switch("quit").help(SHORT).long_help(LONG));
        }
        {
            const SHORT: &str = "Enable start state specialization.";
            const LONG: &str = "\
Enable start state specialization.

When start states are specialized, an implementor of a search routine using
a lazy DFA can tell when the search has entered a starting state. When start
states aren't specialized, then it is impossible to know whether the search has
entered a start state.

Ideally, this option wouldn't need to exist and we could always specialize
start states. The problem is that start states can be quite active. This in
turn means that an efficient search routine is likely to ping-pong between a
heavily optimized hot loop that handles most states and to a less optimized
specialized handling of start states. This causes branches to get heavily
mispredicted and overall can materially decrease throughput. Therefore,
specializing start states should only be enabled when it is needed.

Knowing whether a search is in a start state is typically useful when a
prefilter is active for the search. A prefilter is typically only run when in
a start state and a prefilter can greatly accelerate a search. Therefore, the
possible cost of specializing start states is worth it in this case. Otherwise,
if you have no prefilter, there is likely no reason to specialize start states.

This is disabled by default.
";
            app = app.arg(
                switch("specialize-start-states").help(SHORT).long_help(LONG),
            );
        }
        {
            const SHORT: &str = "Set the DFA cache capacity, in bytes.";
            const LONG: &str = "\
Set the DFA cache capacity, in bytes.

A hybrid NFA/DFA uses a fixed size cache that contains, among other things,
the DFA transition table. This flag controls how big that cache is. If the
cache is filled during search, then it is cleared. Any previously generated
transitions (for example) will need to be re-computed if they are visited
again. Depending on other settings (minimum cache clear count and minimum bytes
seen per generated state), the search may eventually quit.
";
            app = app.arg(flag("cache-capacity").help(SHORT).long_help(LONG));
        }
        {
            const SHORT: &str = "Skip DFA cache capacity check.";
            const LONG: &str = "\
Skip DFA cache capacity check.

When enabled, creating a lazy DFA will not quite a minimum cache capacity.
Instead, if the provided cache capacity is insufficient, it will automatically
set the cache capacity to the minimum allowed.

This is typically only useful for debugging. For example, this is useful in
conjunction with setting the cache capacity to zero, which means that the cache
capacity will always be set to its minimum amount. This permits testing extreme
cases where the cache is frequently reset.

It is generally not a good idea to set this in general, since setting the cache
capacity to its minimum will likely result in lots of cache clearing and thus,
ineffective use of the DFA.
";
            app = app.arg(
                switch("skip-cache-capacity-check")
                    .help(SHORT)
                    .long_help(LONG),
            );
        }
        {
            const SHORT: &str = "Set the minimum cache clear count.";
            const LONG: &str = "\
Set the minimum cache clear count before giving up on the search.

A hybrid NFA/DFA uses a fixed size cache that contains, among other things,
the DFA transition table. When the cache fills up, it is cleared, such that
more transitions may be generated at the cost of potentially re-generating
transitions that were previously in the cache. If the cache is cleared more
than the number of times set by this flag, then the search will stop with an
error.

The default for this flag is 'none', which sets no minimum. This implies that
the search will never give up and will instead continually clear the cache.

The main reason for stopping the search is that refilling the cache repeatedly
can wind up being quite costly. Costly enough where an alternative search
technique would likely be superior.
";
            app = app.arg(
                flag("min-cache-clear-count").help(SHORT).long_help(LONG),
            );
        }
        app
    }

    pub fn get(args: &Args) -> anyhow::Result<Hybrid> {
        let kind = match args.value_of_lossy("match-kind") {
            None => MatchKind::LeftmostFirst,
            Some(value) => match &*value {
                "all" => MatchKind::All,
                "leftmost-first" => MatchKind::LeftmostFirst,
                unk => anyhow::bail!("unrecognized match kind: {:?}", unk),
            },
        };
        let mut c = hybrid::dfa::Config::new()
            .byte_classes(!args.is_present("no-byte-classes"))
            .match_kind(kind)
            .starts_for_each_pattern(
                args.is_present("starts-for-each-pattern"),
            )
            .unicode_word_boundary(args.is_present("unicode-word-boundary"))
            .specialize_start_states(
                args.is_present("specialize-start-states"),
            )
            .skip_cache_capacity_check(
                args.is_present("skip-cache-capacity-check"),
            );
        if let Some(quits) = args.value_of_lossy("quit") {
            for ch in quits.chars() {
                if !ch.is_ascii() {
                    anyhow::bail!("quit bytes must be ASCII");
                }
                // MSRV(1.59): use the 'TryFrom<char> for u8' impl.
                c = c.quit(u8::try_from(u32::from(ch)).unwrap(), true);
            }
        }
        if let Some(n) = args.value_of_lossy("cache-capacity") {
            let limit =
                n.parse().context("failed to parse --cache-capacity")?;
            c = c.cache_capacity(limit);
        }
        if let Some(n) = args.value_of_lossy("min-cache-clear-count") {
            if n.to_lowercase() == "none" {
                c = c.minimum_cache_clear_count(None);
            } else {
                let limit = n
                    .parse()
                    .context("failed to parse --min-cache-clear-count")?;
                c = c.minimum_cache_clear_count(Some(limit));
            }
        }
        Ok(Hybrid { config: c })
    }

    pub fn from_nfa(
        &self,
        nfa: thompson::NFA,
    ) -> anyhow::Result<hybrid::dfa::DFA> {
        hybrid::dfa::Builder::new()
            .configure(self.config.clone())
            .build_from_nfa(nfa)
            .context("failed to build lazy DFA")
    }

    pub fn from_patterns(
        &self,
        table: &mut Table,
        syntax: &Syntax,
        thompson: &Thompson,
        patterns: &Patterns,
    ) -> anyhow::Result<hybrid::dfa::DFA> {
        let patterns = patterns.as_strings();

        let (asts, time) = util::timeitr(|| syntax.asts(patterns))?;
        table.add("parse time", time);
        let (hirs, time) = util::timeitr(|| syntax.hirs(patterns, &asts))?;
        table.add("translate time", time);
        let (nfa, time) = util::timeitr(|| thompson.from_hirs(&hirs))?;
        table.add("compile nfa time", time);
        table.add("nfa memory", nfa.memory_usage());
        let (dfa, time) = util::timeitr(|| self.from_nfa(nfa))?;
        table.add("build hybrid dfa time", time);
        table.add("hybrid dfa memory", dfa.memory_usage());

        Ok(dfa)
    }
}

#[derive(Debug)]
pub struct RegexHybrid {
    /// Hybrid NFA/DFA regexes have no config currently.
    _nothing: (),
}

impl RegexHybrid {
    pub fn define(app: App) -> App {
        app
    }

    pub fn get(_args: &Args) -> anyhow::Result<RegexHybrid> {
        Ok(RegexHybrid { _nothing: () })
    }

    pub fn builder(
        &self,
        syntax: &Syntax,
        thompson: &Thompson,
        hybrid: &Hybrid,
    ) -> hybrid::regex::Builder {
        let mut builder = hybrid::regex::Builder::new();
        builder
            .syntax(syntax.0)
            .thompson(thompson.0.clone())
            .dfa(hybrid.config.clone());
        builder
    }

    pub fn from_patterns(
        &self,
        table: &mut Table,
        syntax: &Syntax,
        thompson: &Thompson,
        hybrid: &Hybrid,
        patterns: &Patterns,
    ) -> anyhow::Result<hybrid::regex::Regex> {
        let patterns = patterns.as_strings();
        let b = self.builder(syntax, thompson, hybrid);
        let (re, time) = util::timeitr(|| b.build_many(patterns))?;
        // let mem_fwd = re.forward().memory_usage();
        // let mem_rev = re.forward().memory_usage();
        table.add("build hybrid regex time", time);
        // table.add("dense regex (forward DFA) memory", mem_fwd);
        // table.add("dense regex (reverse DFA) memory", mem_rev);
        // table.add("hybrid regex memory", mem_fwd + mem_rev);
        Ok(re)
    }
}

#[derive(Debug)]
pub struct Meta {
    config: meta::Config,
}

impl Meta {
    pub fn define(mut app: App) -> App {
        {
            const SHORT: &str = "Choose the match kind.";
            const LONG: &str = "\
Choose the match kind.

This permits setting the match kind to either 'leftmost-first' (the default)
or 'all'. The former will attempt to find the longest match starting at the
leftmost position, but prioritizing alternations in the regex that appear
first. For example, with leftmost-first enabled, 'Sam|Samwise' will match 'Sam'
in 'Samwise' while 'Samwise|Sam' would match 'Samwise'.

'all' match semantics will include all possible matches, including the longest
possible match. 'all' is most commonly used when compiling a reverse DFA to
determine the starting position of a match. Note that when 'all' is used, there
is no distinction between greedy and non-greedy regexes. Everything is greedy
all the time.
";
            app = app.arg(
                flag("match-kind").short("k").help(SHORT).long_help(LONG),
            );
        }
        {
            const SHORT: &str = "Disable UTF-8 handling for iterators.";
            const LONG: &str = "\
Disable UTF-8 handling for match iterators when an empty match is seen.

When UTF-8 mode is enabled for regexes (the default) and an empty match is
seen, the iterators will always start the next search at the next UTF-8 encoded
codepoint when searching valid UTF-8. When UTF-8 mode is disabled, such
searches are started at the next byte offset.

Generally speaking, UTF-8 mode for regexes should only be used when you know
you are searching valid UTF-8. Typically, this should only be disabled in
precisely the cases where the regex itself is permitted to match invalid UTF-8.
This means you usually want to use '--no-utf8-syntax' and '--no-utf8-nfa'
together.

This mode cannot be toggled inside the regex.
";
            app = app.arg(switch("no-utf8-iter").help(SHORT).long_help(LONG));
        }
        {
            const SHORT: &str = "Disable automatic prefilter.";
            const LONG: &str = "\
Disable automatic prefilter.

When set, this causes the meta regex engine to never build or use a prefilter.
By default, the meta regex engine will try to use a prefilter to accelerate
searches.
";
            app = app.arg(switch("no-prefilter").help(SHORT).long_help(LONG));
        }
        {
            const SHORT: &str = "Set a size limit, in bytes, on the NFA size.";
            const LONG: &str = "\
Set a size limit, in bytes, on the NFA size.

This size limit is expressed in bytes of heap memory and is applied while
converting the concrete syntax of a pattern to its corresponding NFA.

The default for this flag is some reasonable size, but 'none' can be used
to disable the limit entirely.
";
            app = app.arg(flag("nfa-size-limit").help(SHORT).long_help(LONG));
        }
        {
            const SHORT: &str =
                "Set a size limit, in bytes, on the one-pass DFA size.";
            const LONG: &str = "\
Set a size limit, in bytes, on the one-pass DFA size.

This size limit is expressed in bytes of heap memory and is applied while
converting the NFA of a regex to its corresponding one-pass DFA. This limit
is only applied when one-pass DFA construction is attempted, which does not
always occur.

The default for this flag is some reasonable size, but 'none' can be used
to disable the limit entirely.
";
            app = app
                .arg(flag("onepass-size-limit").help(SHORT).long_help(LONG));
        }
        {
            const SHORT: &str = "Set the cache capacity of the lazy DFA.";
            const LONG: &str = "\
Set the cache capacity of the lazy DFA.

This size limit is expressed in bytes of heap memory and is applied to the
capacity of the working memory used by the lazy DFA to store states. It can be
used to increase this for especially large regexes or for regexes that generate
many DFA states during powerset construction.

The default for this flag is some reasonable size. Note that unlike other
limits, this must be set to a valid value. It cannot be disabled via a 'none'
value (because it is a capacity), although some large value may be used to
achieve a similar effect.
";
            app = app.arg(
                flag("hybrid-cache-capacity").help(SHORT).long_help(LONG),
            );
        }
        {
            const SHORT: &str = "Disable the hybrid NFA/DFA engine.";
            const LONG: &str = "\
Disable the hybrid NFA/DFA engine.

When set, the meta regex engine will not attempt to use the hybrid NFA/DFA
engine (also known as the lazy DFA).
";
            app = app.arg(switch("no-hybrid").help(SHORT).long_help(LONG));
        }
        {
            const SHORT: &str = "Disable the one-pass DFA engine.";
            const LONG: &str = "\
Disable the one-pass DFA engine.

When set, the meta regex engine will not attempt to use the one-pass DFA
engine.
";
            app = app.arg(switch("no-onepass").help(SHORT).long_help(LONG));
        }
        {
            const SHORT: &str = "Disable the bounded backtracking engine.";
            const LONG: &str = "\
Disable the bounded backtracking engine.

When set, the meta regex engine will not attempt to use the bounded
backtracking engine.
";
            app = app.arg(switch("no-backtrack").help(SHORT).long_help(LONG));
        }
        {
            const SHORT: &str = "Disable the use of equivalence classes.";
            const LONG: &str = "\
Disable the use of equivalence classes.

When disabled, every state in the lazy DFA and one-pass DFA will always
have 256 transitions When enabled (the default), transitions are grouped
into equivalence classes where every byte in the same class cannot possibly
differentiate between a match and a non-match.

Keep byte classes enabled is always a good idea, since it both decreases the
amount of space required and also the amount of time it takes to build the DFA
(since there are fewer transitions to create). The only reason to disable byte
classes is for debugging the representation of a DFA, since equivalence class
identifiers will be used for the transitions instead of the actual bytes.
";
            app = app.arg(
                switch("no-byte-classes")
                    .short("C")
                    .help(SHORT)
                    .long_help(LONG),
            );
        }
        app
    }

    pub fn get(args: &Args) -> anyhow::Result<Meta> {
        let kind = match args.value_of_lossy("match-kind") {
            None => MatchKind::LeftmostFirst,
            Some(value) => match &*value {
                "all" => MatchKind::All,
                "leftmost-first" => MatchKind::LeftmostFirst,
                unk => anyhow::bail!("unrecognized match kind: {:?}", unk),
            },
        };
        let mut c = meta::Config::new()
            .utf8(!args.is_present("no-utf8-iter"))
            .match_kind(kind)
            .auto_prefilter(!args.is_present("no-prefilter"))
            .hybrid(!args.is_present("no-hybrid"))
            .onepass(!args.is_present("no-onepass"))
            .backtrack(!args.is_present("no-backtrack"))
            .byte_classes(!args.is_present("no-byte-classes"));
        if let Some(x) = args.value_of_lossy("nfa-size-limit") {
            if x.to_lowercase() == "none" {
                c = c.nfa_size_limit(None);
            } else {
                let limit =
                    x.parse().context("failed to parse --nfa-size-limit")?;
                c = c.nfa_size_limit(Some(limit));
            }
        }
        if let Some(x) = args.value_of_lossy("onepass-size-limit") {
            if x.to_lowercase() == "none" {
                c = c.onepass_size_limit(None);
            } else {
                let limit = x
                    .parse()
                    .context("failed to parse --onepass-size-limit")?;
                c = c.onepass_size_limit(Some(limit));
            }
        }
        if let Some(x) = args.value_of_lossy("hybrid-cache-capacity") {
            let limit = x
                .parse()
                .context("failed to parse --hybrid-cache-capacity")?;
            c = c.hybrid_cache_capacity(limit);
        }
        Ok(Meta { config: c })
    }

    pub fn builder(&self, syntax: &Syntax) -> meta::Builder {
        let mut builder = meta::Builder::new();
        builder.configure(self.config.clone()).syntax(syntax.0);
        builder
    }

    pub fn from_patterns(
        &self,
        table: &mut Table,
        syntax: &Syntax,
        patterns: &Patterns,
    ) -> anyhow::Result<meta::Regex> {
        let patterns = patterns.as_strings();
        let b = self.builder(syntax);
        let (re, time) = util::timeitr(|| b.build_many(patterns))?;
        // let mem_fwd = re.forward().memory_usage();
        // let mem_rev = re.forward().memory_usage();
        table.add("build hybrid regex time", time);
        // table.add("dense regex (forward DFA) memory", mem_fwd);
        // table.add("dense regex (reverse DFA) memory", mem_rev);
        // table.add("hybrid regex memory", mem_fwd + mem_rev);
        Ok(re)
    }
}

#[derive(Debug)]
pub struct RegexAPI {
    size_limit: Option<usize>,
    dfa_size_limit: Option<usize>,
}

impl RegexAPI {
    pub fn define(mut app: App) -> App {
        {
            const SHORT: &str =
                "Set the approximate size limit for a compiled regex.";
            const LONG: &str = "\
Set the approximate size limit for a compiled regex.
";
            app = app.arg(flag("size-limit").help(SHORT).long_help(LONG));
        }
        {
            const SHORT: &str =
                "Set the approximate size of the cache used by the DFA.";
            const LONG: &str = "\
Set the approximate size of the cache used by the DFA.
";
            app = app.arg(flag("dfa-size-limit").help(SHORT).long_help(LONG));
        }
        app
    }

    pub fn get(args: &Args) -> anyhow::Result<RegexAPI> {
        let mut config = RegexAPI { size_limit: None, dfa_size_limit: None };
        if let Some(x) = args.value_of_lossy("size-limit") {
            let limit = x.parse().context("failed to parse --size-limit")?;
            config.size_limit = Some(limit);
        }
        if let Some(x) = args.value_of_lossy("dfa-size-limit") {
            let limit =
                x.parse().context("failed to parse --dfa-size-limit")?;
            config.dfa_size_limit = Some(limit);
        }
        Ok(config)
    }

    pub fn from_patterns(
        &self,
        table: &mut Table,
        syntax: &Syntax,
        api: &RegexAPI,
        patterns: &Patterns,
    ) -> anyhow::Result<regex::bytes::Regex> {
        if syntax.0.get_utf8() {
            anyhow::bail!(
                "API-level regex requires that UTF-8 syntax mode be disabled",
            );
        }
        let patterns = patterns.as_strings();
        if patterns.len() != 1 {
            anyhow::bail!(
                "API-level regex requires exactly one pattern, \
                 but {} were given",
                patterns.len(),
            );
        }
        let (re, time) = util::timeitr(|| {
            let mut b = regex::bytes::RegexBuilder::new(&patterns[0]);
            b.case_insensitive(syntax.0.get_case_insensitive());
            b.multi_line(syntax.0.get_multi_line());
            b.dot_matches_new_line(syntax.0.get_dot_matches_new_line());
            b.swap_greed(syntax.0.get_swap_greed());
            b.ignore_whitespace(syntax.0.get_ignore_whitespace());
            b.unicode(syntax.0.get_unicode());
            b.octal(syntax.0.get_octal());
            b.nest_limit(syntax.0.get_nest_limit());
            if let Some(limit) = api.size_limit {
                b.size_limit(limit);
            }
            if let Some(limit) = api.dfa_size_limit {
                b.dfa_size_limit(limit);
            }
            b.build().map_err(anyhow::Error::from)
        })?;
        table.add("build API regex time", time);
        Ok(re)
    }
}

#[derive(Debug)]
pub struct RegexSetAPI {
    size_limit: Option<usize>,
    dfa_size_limit: Option<usize>,
}

impl RegexSetAPI {
    pub fn define(mut app: App) -> App {
        {
            const SHORT: &str =
                "Set the approximate size limit for a compiled regex set.";
            const LONG: &str = "\
Set the approximate size limit for a compiled regex set.
";
            app = app.arg(flag("size-limit").help(SHORT).long_help(LONG));
        }
        {
            const SHORT: &str =
                "Set the approximate size of the cache used by the DFA.";
            const LONG: &str = "\
Set the approximate size of the cache used by the DFA.
";
            app = app.arg(flag("dfa-size-limit").help(SHORT).long_help(LONG));
        }
        app
    }

    pub fn get(args: &Args) -> anyhow::Result<RegexSetAPI> {
        let mut config =
            RegexSetAPI { size_limit: None, dfa_size_limit: None };
        if let Some(x) = args.value_of_lossy("size-limit") {
            let limit = x.parse().context("failed to parse --size-limit")?;
            config.size_limit = Some(limit);
        }
        if let Some(x) = args.value_of_lossy("dfa-size-limit") {
            let limit =
                x.parse().context("failed to parse --dfa-size-limit")?;
            config.dfa_size_limit = Some(limit);
        }
        Ok(config)
    }

    pub fn from_patterns(
        &self,
        table: &mut Table,
        syntax: &Syntax,
        api: &RegexSetAPI,
        patterns: &Patterns,
    ) -> anyhow::Result<regex::bytes::RegexSet> {
        if syntax.0.get_utf8() {
            anyhow::bail!(
                "API-level regex set requires that \
                 UTF-8 syntax mode be disabled",
            );
        }
        let patterns = patterns.as_strings();
        let (re, time) = util::timeitr(|| {
            let mut b = regex::bytes::RegexSetBuilder::new(patterns);
            b.case_insensitive(syntax.0.get_case_insensitive());
            b.multi_line(syntax.0.get_multi_line());
            b.dot_matches_new_line(syntax.0.get_dot_matches_new_line());
            b.swap_greed(syntax.0.get_swap_greed());
            b.ignore_whitespace(syntax.0.get_ignore_whitespace());
            b.unicode(syntax.0.get_unicode());
            b.octal(syntax.0.get_octal());
            b.nest_limit(syntax.0.get_nest_limit());
            if let Some(limit) = api.size_limit {
                b.size_limit(limit);
            }
            if let Some(limit) = api.dfa_size_limit {
                b.dfa_size_limit(limit);
            }
            b.build().map_err(anyhow::Error::from)
        })?;
        table.add("build API regexset time", time);
        Ok(re)
    }
}
