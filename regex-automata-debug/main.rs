use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;
use regex_automata::{dense, nfa, Regex, RegexBuilder, DFA};
use regex_syntax as syntax;

fn main() -> Result<()> {
    Command::parse().and_then(|c| c.run())
}

#[derive(Debug)]
struct Command {
    kind: CommandKind,
    args: Common,
}

#[derive(Debug)]
enum CommandKind {
    Debug { pattern: String, quiet: bool },
    DebugNFA { pattern: String, quiet: bool },
    Find { pattern: String, path: PathBuf },
    Nothing,
}

#[derive(Debug)]
struct DebugInfo {
    output: Option<String>,
    dense: Stats,
    sparse: Option<Stats>,
}

#[derive(Debug, Default)]
struct Stats {
    compile: Duration,
    memory: usize,
}

impl Command {
    fn parse() -> Result<Command> {
        match app().get_matches().subcommand() {
            ("debug", Some(m)) => {
                let args = Common::new(m);
                let pattern = pattern_from_matches(m)?;
                let quiet = m.is_present("quiet");
                Ok(Command {
                    kind: CommandKind::Debug { pattern, quiet },
                    args,
                })
            }
            ("debug-nfa", Some(m)) => {
                let args = Common::new(m);
                let pattern = pattern_from_matches(m)?;
                let quiet = m.is_present("quiet");
                Ok(Command {
                    kind: CommandKind::DebugNFA { pattern, quiet },
                    args,
                })
            }
            ("find", Some(m)) => {
                let args = Common::new(m);
                let pattern = pattern_from_matches(m)?;
                let path = PathBuf::from(m.value_of_os("path").unwrap());
                Ok(Command { kind: CommandKind::Find { pattern, path }, args })
            }
            ("", _) => {
                app().print_help()?;
                println!("");
                Ok(Command {
                    kind: CommandKind::Nothing,
                    args: Common::default(),
                })
            }
            (unknown, _) => {
                Err(anyhow::anyhow!("unrecognized command: {}", unknown))
            }
        }
    }

    fn run(&self) -> Result<()> {
        match self.kind {
            CommandKind::Debug { .. } => self.run_debug(),
            CommandKind::DebugNFA { .. } => self.run_debug_nfa(),
            CommandKind::Find { .. } => self.run_find(),
            CommandKind::Nothing => Ok(()),
        }
    }

    fn run_debug(&self) -> Result<()> {
        let mut stdout = io::stdout();

        let i = self.debug_info()?;
        writeln!(stdout, " dense compile time: {:?}", i.dense.compile)?;
        writeln!(stdout, "       dense memory: {}", i.dense.memory)?;
        if let Some(sparse) = i.sparse {
            writeln!(stdout, "sparse compile time: {:?}", sparse.compile)?;
            writeln!(stdout, "      sparse memory: {}", sparse.memory)?;
        }
        if let Some(output) = i.output {
            writeln!(stdout, "")?;
            writeln!(stdout, "{}", output)?;
        }
        Ok(())
    }

    fn run_debug_nfa(&self) -> Result<()> {
        let mut stdout = io::stdout();

        let start = Instant::now();
        let expr = self.parser_builder().build().parse(self.pattern())?;
        let parse_time = Instant::now().duration_since(start);

        let start = Instant::now();
        let nfa = self.nfa_builder().build(&expr)?;
        let nfa_time = Instant::now().duration_since(start);

        writeln!(stdout, "       parse time: {:?}", parse_time)?;
        writeln!(stdout, " nfa compile time: {:?}", nfa_time)?;
        if !self.quiet() {
            writeln!(stdout, "")?;
            writeln!(stdout, "{:?}", nfa)?;
        }
        Ok(())
    }

    fn run_find(&self) -> Result<()> {
        let mut stdout = io::stdout();

        let start = Instant::now();
        let (finder, memory_usage) = self.finder()?;
        let compile_time = Instant::now().duration_since(start);
        writeln!(stdout, "compile time: {:?}", compile_time)?;
        writeln!(stdout, "      memory: {}", memory_usage)?;

        let start = Instant::now();
        let data = self.data()?;
        let read_time = Instant::now().duration_since(start);
        writeln!(stdout, "   read time: {:?}", read_time)?;

        let start = Instant::now();
        let matches = finder(&data);
        let match_time = Instant::now().duration_since(start);
        writeln!(stdout, "  match time: {:?}", match_time)?;

        writeln!(stdout, " match count: {}", matches)?;
        Ok(())
    }

    fn data(&self) -> Result<Vec<u8>> {
        let path = match self.kind {
            CommandKind::Find { ref path, .. } => path,
            _ => unreachable!(),
        };
        Ok(fs::read(path)?)
    }

    fn pattern(&self) -> &str {
        match self.kind {
            CommandKind::Debug { ref pattern, .. } => pattern,
            CommandKind::DebugNFA { ref pattern, .. } => pattern,
            CommandKind::Find { ref pattern, .. } => pattern,
            _ => unreachable!(),
        }
    }

    fn quiet(&self) -> bool {
        match self.kind {
            CommandKind::Debug { quiet, .. } => quiet,
            CommandKind::DebugNFA { quiet, .. } => quiet,
            _ => false,
        }
    }

    fn finder(&self) -> Result<(Box<dyn Fn(&[u8]) -> usize>, usize)> {
        let re = self.regex_builder().build(self.pattern())?;
        if self.args.sparse {
            let re = Regex::from_dfas(
                re.forward().to_sparse()?,
                re.reverse().to_sparse()?,
            );
            let m = re.forward().memory_usage() + re.reverse().memory_usage();
            Ok((counter(re), m))
        } else {
            let m = re.forward().memory_usage() + re.reverse().memory_usage();
            Ok((counter(re), m))
        }
    }

    fn debug_info(&self) -> Result<DebugInfo> {
        let mut dense = Stats::default();

        let start = Instant::now();
        let dfa = self.dense_builder().build(self.pattern())?;
        dense.compile = Instant::now().duration_since(start);
        dense.memory = dfa.memory_usage();

        let (sparse, output) = if self.args.sparse {
            let mut sparse = Stats::default();

            let start = Instant::now();
            let sdfa = dfa.to_sparse()?;
            sparse.compile = Instant::now().duration_since(start);
            sparse.memory = sdfa.memory_usage();
            if self.quiet() {
                (Some(sparse), None)
            } else {
                (Some(sparse), Some(format!("{:?}", sdfa)))
            }
        } else {
            if self.quiet() {
                (None, None)
            } else {
                (None, Some(format!("{:?}", dfa)))
            }
        };

        Ok(DebugInfo { output, dense, sparse })
    }

    fn regex_builder(&self) -> RegexBuilder {
        let mut builder = RegexBuilder::new();
        builder
            .anchored(self.args.anchored)
            .case_insensitive(self.args.case_insensitive)
            .unicode(!self.args.no_unicode)
            .allow_invalid_utf8(self.args.no_utf8)
            .minimize(self.args.minimize)
            .premultiply(self.args.premultiply)
            .byte_classes(self.args.classes);
        builder
    }

    fn dense_builder(&self) -> dense::Builder {
        let mut builder = dense::Builder::new();
        builder
            .anchored(self.args.anchored)
            .case_insensitive(self.args.case_insensitive)
            .unicode(!self.args.no_unicode)
            .allow_invalid_utf8(self.args.no_utf8)
            .minimize(self.args.minimize)
            .premultiply(self.args.premultiply)
            .byte_classes(self.args.classes)
            .reverse(self.args.reverse)
            .longest_match(self.args.longest_match)
            .shrink(self.args.shrink_nfa);
        builder
    }

    fn nfa_builder(&self) -> nfa::Builder {
        let mut builder = nfa::Builder::new();
        builder
            .anchored(self.args.anchored)
            .allow_invalid_utf8(self.args.no_utf8)
            .reverse(self.args.reverse)
            .shrink(self.args.shrink_nfa);
        builder
    }

    fn parser_builder(&self) -> syntax::ParserBuilder {
        let mut builder = syntax::ParserBuilder::new();
        builder
            .case_insensitive(self.args.case_insensitive)
            .unicode(!self.args.no_unicode)
            .allow_invalid_utf8(self.args.no_utf8);
        builder
    }
}

#[derive(Debug, Default)]
struct Common {
    sparse: bool,
    anchored: bool,
    case_insensitive: bool,
    no_unicode: bool,
    no_utf8: bool,
    minimize: bool,
    premultiply: bool,
    classes: bool,
    reverse: bool,
    longest_match: bool,
    shrink_nfa: bool,
}

impl Common {
    fn new(m: &clap::ArgMatches) -> Common {
        Common {
            sparse: m.is_present("sparse"),
            anchored: m.is_present("anchored"),
            case_insensitive: m.is_present("case-insensitive"),
            no_unicode: m.is_present("no-unicode"),
            no_utf8: m.is_present("no-utf8"),
            minimize: m.is_present("minimize"),
            premultiply: m.is_present("premultiply"),
            classes: m.is_present("classes"),
            reverse: m.is_present("reverse"),
            longest_match: m.is_present("longest-match"),
            shrink_nfa: m.is_present("shrink-nfa"),
        }
    }
}

fn counter<D: DFA + 'static>(re: Regex<D>) -> Box<dyn Fn(&[u8]) -> usize> {
    Box::new(move |bytes| re.find_iter(bytes).count())
}

fn pattern_from_matches(m: &clap::ArgMatches) -> Result<String> {
    if !m.is_present("file") {
        Ok(m.value_of("pattern").unwrap().to_string())
    } else {
        let path = m.value_of("pattern").unwrap();
        let contents = fs::read_to_string(path)?;
        Ok(contents.lines().collect::<Vec<_>>().join("|"))
    }
}

fn app() -> clap::App<'static, 'static> {
    let cmd = |name| {
        clap::SubCommand::with_name(name)
            .author(clap::crate_authors!())
            .version(clap::crate_version!())
    };
    let pos = |name| clap::Arg::with_name(name);
    let flag = |name| clap::Arg::with_name(name).long(name);

    let common = |app: clap::App<'static, 'static>| {
        app.arg(flag("file").short("f").takes_value(true))
            .arg(flag("sparse").short("s"))
            .arg(flag("anchored").short("a"))
            .arg(flag("case-insensitive").short("i"))
            .arg(flag("no-unicode"))
            .arg(flag("no-utf8").short("u"))
            .arg(flag("minimize").short("m"))
            .arg(flag("premultiply").short("p"))
            .arg(flag("classes").short("c"))
            .arg(flag("reverse").short("r"))
            .arg(flag("longest-match"))
            .arg(flag("shrink-nfa"))
    };

    let cmd_debug = cmd("debug")
        .about("Show output of automata.")
        .arg(pos("pattern").required(true))
        .arg(flag("quiet").short("q"));
    let cmd_debug_nfa = cmd("debug-nfa")
        .about("Show output of NFA automata.")
        .arg(pos("pattern").required(true))
        .arg(flag("quiet").short("q"));
    let cmd_find = cmd("find")
        .about("Search in file with automata.")
        .arg(pos("pattern").required(true))
        .arg(pos("path").required(true));

    clap::App::new("Search using regex-automata")
        .author(clap::crate_authors!())
        .version(clap::crate_version!())
        .max_term_width(100)
        .subcommand(common(cmd_debug))
        .subcommand(common(cmd_debug_nfa))
        .subcommand(common(cmd_find))
}
