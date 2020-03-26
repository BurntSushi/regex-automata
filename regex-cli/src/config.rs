use std::borrow::Borrow;
use std::fs;
use std::path::PathBuf;

use anyhow::Context;
use automata::{
    dfa::{self, dense, sparse},
    nfa::thompson,
    MatchKind, StateID,
};

use crate::app::{self, flag, switch, App, Args};
use crate::util::{self, Table};

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
            app = app.arg(
                app::flag("pattern-file")
                    .short("f")
                    .multiple(true)
                    .number_of_values(1)
                    .help(SHORT),
            );
        }
        app
    }

    /// Reads at least one pattern from either positional arguments (preferred)
    /// or from pattern files. If no patterns could be found, then an error
    /// is returned.
    pub fn get(args: &Args) -> anyhow::Result<Patterns> {
        if let Some(os_patterns) = args.values_of_os("pattern") {
            if args.value_of_os("pattern-file").is_some() {
                anyhow::bail!(
                    "cannot provide both positional patterns and \
                     --pattern-file"
                );
            }
            let mut patterns = vec![];
            for (i, p) in os_patterns.enumerate() {
                let p = match p.to_str() {
                    Some(p) => p,
                    None => anyhow::bail!("pattern {} is not valid UTF-8", i),
                };
                patterns.push(p.to_string());
            }
            Ok(Patterns(patterns))
        } else if let Some(pfile) = args.value_of_os("pattern-file") {
            let path = std::path::Path::new(pfile);
            let contents =
                std::fs::read_to_string(path).with_context(|| {
                    anyhow::anyhow!("failed to read {}", path.display())
                })?;
            Ok(Patterns(contents.lines().map(|x| x.to_string()).collect()))
        } else {
            Err(anyhow::anyhow!("no regex patterns given"))
        }
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
    pub unsafe fn mmap(&self) -> anyhow::Result<memmap::Mmap> {
        let file = fs::File::open(&self.0)
            .with_context(|| format!("failed to open {}", self.0.display()))?;
        memmap::Mmap::map(&file)
            .with_context(|| format!("failed to mmap {}", self.0.display()))
    }
}

/// Flags specific to searching.
#[derive(Debug)]
pub struct Find {
    kind: FindKind,
    matches: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FindKind {
    Earliest,
    Leftmost,
    Overlapping,
}

impl Find {
    pub fn define(mut app: App) -> App {
        {
            const SHORT: &str = "Set the type of search to perform.";
            const LONG: &str = "\
Set the type of search to perform.
";
            app = app
                .arg(flag("find-kind").short("K").help(SHORT).long_help(LONG));
        }
        {
            const SHORT: &str = "Show the offsets of each match found.";
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
        let kind = match args.value_of_lossy("find-kind") {
            None => FindKind::Leftmost,
            Some(value) => match &*value {
                "earliest" => FindKind::Earliest,
                "leftmost" => FindKind::Leftmost,
                "overlapping" => FindKind::Overlapping,
                unk => anyhow::bail!("unrecognized find kind: {:?}", unk),
            },
        };
        Ok(Find { kind, matches: args.is_present("matches") })
    }

    pub fn kind(&self) -> FindKind {
        self.kind
    }

    pub fn matches(&self) -> bool {
        self.matches
    }
}

#[derive(Debug)]
pub struct Syntax(automata::SyntaxConfig);

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
        let mut c = automata::SyntaxConfig::new()
            .case_insensitive(args.is_present("case-insensitive"))
            .multi_line(args.is_present("multi-line"))
            .dot_matches_new_line(args.is_present("dot-matches-new-line"))
            .swap_greed(args.is_present("swap-greed"))
            .ignore_whitespace(args.is_present("ignore-whitespace"))
            .unicode(!args.is_present("no-unicode"))
            .allow_invalid_utf8(args.is_present("no-utf8-syntax"))
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
            .allow_invalid_utf8(self.0.get_allow_invalid_utf8())
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
                "Allow unachored searches through invalid UTF-8.";
            const LONG: &str = "\
Allow unanchored searches through invalid UTF-8.

When UTF-8 mode is enabled (which is the default), unanchored searches will
only match through valid UTF-8. If invalid UTF-8 is seen, then an unanchored
search will stop at that point. This is equivalent to putting a `(?s:.)*` at
the start of the regex.

When UTF-8 mode is disabled, then unanchored searches will match through any
arbitrary byte. This is equivalent to putting a `(?s-u:.)*` at the start of the
regex.

Generally speaking, UTF-8 mode should only be used when you know you are
searching valid UTF-8. Typically, this should only be disabled in precisely
the cases where the regex itself is permitted to match invalid UTF-8. This
means you usually want to use '--no-utf8-syntax' and '--no-utf8-nfa' (or '-bB')
together instead of one or the other.

This mode cannot be toggled inside the regex.
";
            app = app.arg(
                switch("no-utf8-nfa").short("B").help(SHORT).long_help(LONG),
            );
        }
        {
            const SHORT: &str = "Disable NFA shrinking.";
            const LONG: &str = "\
Disable NFA shrinking.

By default, when compiling an NFA, some extra effort is expended to reduce the
size of the NFA. While implementation details may change, currently this only
occurs when compiling large Unicode character classes in reverse. This extra
work can make NFA compilation slower. However, the reduction in size can be
big enough to cause a dramatic performance improvement when the NFA is used to
compile a DFA.
";
            app = app.arg(switch("no-shrink").help(SHORT).long_help(LONG));
        }

        app
    }

    pub fn get(args: &Args) -> anyhow::Result<Thompson> {
        let c = thompson::Config::new()
            .reverse(args.is_present("reverse"))
            .utf8(!args.is_present("no-utf8-nfa"))
            .shrink(!args.is_present("no-shrink"));
        Ok(Thompson(c))
    }

    pub fn from_hirs<H: Borrow<syntax::hir::Hir>>(
        &self,
        exprs: &[H],
    ) -> anyhow::Result<thompson::NFA> {
        thompson::Builder::new()
            .configure(self.0)
            .build_many_from_hir(exprs)
            .context("failed to compile Thompson NFA")
    }
}

#[derive(Debug)]
pub struct Dense {
    config: dense::Config,
    state_id_size: usize,
}

impl Dense {
    pub fn define(mut app: App) -> App {
        {
            const SHORT: &str = "Compile an anchored DFA.";
            const LONG: &str = "\
Compile an anchored DFA.

When enabled, the DFA is anchored. This means that the DFA can only find
matches that begin where the search starts. When disabled (the default), the
DFA will have an \"unanchored\" prefix that permits it to match anywhere.
";
            app = app.arg(
                switch("anchored").short("a").help(SHORT).long_help(LONG),
            );
        }
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
            const SHORT: &str = "Choose the size of the state ID.";
            const LONG: &str = "\
Choose the size of the state ID.

By default, the size of the state identifiers compiled into a DFA is the size
of your target's pointer (i.e., a 'usize'). However, it can be advantageous to
save space by changing the state ID representation to a smaller size. This can
not only decrease space, but also make matching faster by virtue of better
CPU cache utilization.

There is generally no downside to using a smaller state ID representation
other than smaller representations (especially 1 or 2 bytes) being unable to
represent larger DFAs. In that case, an error will be returned.

The only legal values are 1, 2, 4 or 8.
";

            app = app.arg(
                flag("state-id")
                    .help(SHORT)
                    .long_help(LONG)
                    .possible_values(&["1", "2", "4", "8"]),
            );
        }
        app
    }

    pub fn get(args: &Args) -> anyhow::Result<Dense> {
        let kind = match args.value_of_lossy("match-kind") {
            None => MatchKind::LeftmostFirst,
            Some(value) => match &*value {
                "all" => MatchKind::All,
                "leftmost-first" => MatchKind::LeftmostFirst,
                unk => anyhow::bail!("unrecognized match kind: {:?}", unk),
            },
        };
        let mut c = dense::Config::new()
            .anchored(args.is_present("anchored"))
            .accelerate(!args.is_present("no-accelerate"))
            .minimize(args.is_present("minimize"))
            .byte_classes(!args.is_present("no-byte-classes"))
            .match_kind(kind)
            .unicode_word_boundary(args.is_present("unicode-word-boundary"));
        if let Some(quits) = args.value_of_lossy("quit") {
            for ch in quits.chars() {
                if !ch.is_ascii() {
                    anyhow::bail!("quit bytes must be ASCII");
                }
                c = c.quit(ch as u8, true);
            }
        }
        Ok(Dense { config: c, state_id_size: get_state_id_size(args)? })
    }

    pub fn from_nfa<S: StateID>(
        &self,
        nfa: &thompson::NFA,
    ) -> anyhow::Result<dense::DFA<Vec<S>, Vec<u8>, S>> {
        dense::Builder::new()
            .configure(self.config)
            .build_from_nfa_with_size(nfa)
            .context("failed to compile dense DFA")
    }

    pub fn from_patterns_dense<S: StateID>(
        &self,
        table: &mut Table,
        syntax: &Syntax,
        thompson: &Thompson,
        dense: &Dense,
        patterns: &Patterns,
    ) -> anyhow::Result<dense::DFA<Vec<S>, Vec<u8>, S>> {
        let patterns = patterns.as_strings();

        let (asts, time) = util::timeitr(|| syntax.asts(patterns))?;
        table.add("parse time", time);
        let (hirs, time) = util::timeitr(|| syntax.hirs(patterns, &asts))?;
        table.add("translate time", time);
        let (nfa, time) = util::timeitr(|| thompson.from_hirs(&hirs))?;
        table.add("compile nfa time", time);
        let (dfa, time) = util::timeitr(|| dense.from_nfa::<S>(&nfa))?;
        table.add("compile dense dfa time", time);
        table.add("dense dfa memory", dfa.memory_usage());
        table.add("dense alphabet length", dfa.alphabet_len());
        table.add("dense stride", 1 << dfa.stride2());

        Ok(dfa)
    }

    pub fn from_patterns_sparse<S: StateID>(
        &self,
        table: &mut Table,
        syntax: &Syntax,
        thompson: &Thompson,
        dense: &Dense,
        patterns: &Patterns,
    ) -> anyhow::Result<sparse::DFA<Vec<u8>, S>> {
        let dfa = self
            .from_patterns_dense(table, syntax, thompson, dense, patterns)?;
        let (sdfa, time) = util::timeitr(|| dfa.to_sparse())?;
        table.add("compile sparse dfa time", time);
        table.add("sparse dfa memory", sdfa.memory_usage());

        Ok(sdfa)
    }
}

#[derive(Debug)]
pub struct RegexDFA(());

impl RegexDFA {
    pub fn define(app: App) -> App {
        // Currently there are no regex specific configuration options.
        app
    }

    pub fn get(_: &Args) -> anyhow::Result<RegexDFA> {
        Ok(RegexDFA(()))
    }

    pub fn builder(
        &self,
        syntax: &Syntax,
        thompson: &Thompson,
        dense: &Dense,
    ) -> dfa::RegexBuilder {
        let mut builder = dfa::RegexBuilder::new();
        builder.syntax(syntax.0).thompson(thompson.0).dense(dense.config);
        builder
    }

    pub fn from_patterns_dense<S: StateID>(
        &self,
        table: &mut Table,
        syntax: &Syntax,
        thompson: &Thompson,
        dense: &Dense,
        patterns: &Patterns,
    ) -> anyhow::Result<dfa::Regex<dense::DFA<Vec<S>, Vec<u8>, S>>> {
        let patterns = patterns.as_strings();
        let b = self.builder(syntax, thompson, dense);
        let (re, time) =
            util::timeitr(|| b.build_many_with_size::<S, _>(patterns))?;
        let mem_fwd = re.forward().memory_usage();
        let mem_rev = re.forward().memory_usage();
        table.add("compile dense regex time", time);
        table.add("dense regex (forward DFA) memory", mem_fwd);
        table.add("dense regex (reverse DFA) memory", mem_rev);
        table.add("dense regex memory", mem_fwd + mem_rev);
        Ok(re)
    }

    pub fn from_patterns_sparse<S: StateID>(
        &self,
        table: &mut Table,
        syntax: &Syntax,
        thompson: &Thompson,
        dense: &Dense,
        patterns: &Patterns,
    ) -> anyhow::Result<dfa::Regex<sparse::DFA<Vec<u8>, S>>> {
        let re = self
            .from_patterns_dense(table, syntax, thompson, dense, patterns)?;
        let (sre, time) = util::timeitr(|| {
            let (fwd, rev) = (re.forward(), re.reverse());
            fwd.to_sparse().and_then(|f| {
                rev.to_sparse().map(|r| dfa::Regex::from_dfas(f, r))
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

/// A convenience function for retrieving the state ID size.
///
/// It's convenient to sometime access the state ID size outside the context
/// of the DFA configuration. There's a little redundancy since this is also
/// parsed as part of the dense configuration, but oh well.
pub fn get_state_id_size(args: &Args) -> anyhow::Result<usize> {
    args.value_of_lossy("state-id")
        .map(|v| v.parse())
        .unwrap_or(Ok(std::mem::size_of::<usize>()))
        .context("failed to parse --state-id integer")
}
