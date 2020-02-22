use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::process;
use std::result;
use std::time::Instant;

use regex_automata::{DenseDFA, Regex, RegexBuilder, SparseDFA, DFA};

type Result<T> = result::Result<T, Box<dyn Error>>;

macro_rules! err {
    ($($tt:tt)*) => { Err(From::from(format!($($tt)*))) }
}

fn main() {
    if let Err(err) = try_main() {
        eprintln!("{}", err);
        process::exit(1);
    }
}

fn try_main() -> Result<()> {
    let mut stdout = io::stdout();

    let args = Args::parse()?;

    if args.debug {
        let start = Instant::now();
        let (debug, memory_usage) = args.debug()?;
        let compile_time = Instant::now().duration_since(start);
        writeln!(stdout, "compile time: {:?}", compile_time)?;
        writeln!(stdout, "      memory: {}", memory_usage)?;
        writeln!(stdout, "")?;
        writeln!(stdout, "{}", debug)?;
    } else {
        let start = Instant::now();
        let (finder, memory_usage) = args.finder()?;
        let compile_time = Instant::now().duration_since(start);
        writeln!(stdout, "compile time: {:?}", compile_time)?;
        writeln!(stdout, "      memory: {}", memory_usage)?;

        let start = Instant::now();
        let data = args.data()?;
        let read_time = Instant::now().duration_since(start);
        writeln!(stdout, "   read time: {:?}", read_time)?;

        let start = Instant::now();
        let matches = finder(&data);
        let match_time = Instant::now().duration_since(start);
        writeln!(stdout, "  match time: {:?}", match_time)?;

        writeln!(stdout, " match count: {}", matches)?;
    }
    Ok(())
}

#[derive(Debug)]
struct Args {
    file: PathBuf,
    pattern: String,
    debug: bool,
    sparse: bool,
    anchored: bool,
    minimize: bool,
    classes: bool,
    premultiply: bool,
    no_utf8: bool,
}

impl Args {
    fn parse() -> Result<Args> {
        use clap::{crate_authors, crate_version, App, Arg};

        let parsed = App::new("Search using regex-automata")
            .author(crate_authors!())
            .version(crate_version!())
            .max_term_width(100)
            .arg(Arg::with_name("path").required(true))
            .arg(Arg::with_name("pattern"))
            .arg(Arg::with_name("file").short("f").takes_value(true))
            .arg(Arg::with_name("debug").long("debug").short("d"))
            .arg(Arg::with_name("sparse").long("sparse").short("s"))
            .arg(Arg::with_name("anchored").long("anchored").short("a"))
            .arg(Arg::with_name("minimize").long("minimize").short("m"))
            .arg(Arg::with_name("classes").long("classes").short("c"))
            .arg(Arg::with_name("premultiply").long("premultiply").short("p"))
            .arg(Arg::with_name("no-utf8").long("no-utf8").short("u"))
            .get_matches();

        let pattern = match parsed.value_of("pattern") {
            Some(pattern) => pattern.to_string(),
            None => {
                let pattern_path = match parsed.value_of_os("file") {
                    None => return err!("no patterns found"),
                    Some(p) => Path::new(p),
                };
                let contents = fs::read_to_string(pattern_path)?;
                contents.lines().collect::<Vec<_>>().join("|")
            }
        };

        Ok(Args {
            file: PathBuf::from(parsed.value_of_os("path").unwrap()),
            pattern: pattern,
            debug: parsed.is_present("debug"),
            sparse: parsed.is_present("sparse"),
            anchored: parsed.is_present("anchored"),
            minimize: parsed.is_present("minimize"),
            classes: parsed.is_present("classes"),
            premultiply: parsed.is_present("premultiply"),
            no_utf8: parsed.is_present("no-utf8"),
        })
    }

    fn data(&self) -> Result<Vec<u8>> {
        Ok(fs::read(&self.file)?)
    }

    fn finder(&self) -> Result<(Box<dyn Fn(&[u8]) -> usize>, usize)> {
        let re = self.builder().build(&self.pattern)?;
        if self.sparse {
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

    fn debug(&self) -> Result<(String, usize)> {
        if self.sparse {
            let re = self.builder().build_sparse(&self.pattern)?;
            let m = re.forward().memory_usage()
                + re.reverse().memory_usage()
                + (2 * size_of::<SparseDFA<Vec<u8>, usize>>());
            Ok((format!("{:?}", re), m))
        } else {
            let re = self.builder().build(&self.pattern)?;
            let m = re.forward().memory_usage()
                + re.reverse().memory_usage()
                + (2 * size_of::<DenseDFA<Vec<usize>, usize>>());
            Ok((format!("{:?}", re), m))
        }
    }

    fn builder(&self) -> RegexBuilder {
        let mut builder = RegexBuilder::new();
        builder
            .minimize(self.minimize)
            .anchored(self.anchored)
            .byte_classes(self.classes)
            .premultiply(self.premultiply)
            .allow_invalid_utf8(self.no_utf8);
        builder
    }
}

fn counter<D: DFA + 'static>(re: Regex<D>) -> Box<dyn Fn(&[u8]) -> usize> {
    Box::new(move |bytes| re.find_iter(bytes).count())
}
