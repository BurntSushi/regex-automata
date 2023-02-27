use std::{
    collections::HashSet,
    fs::File,
    io::{BufRead, Read, Write},
    path::{Path, PathBuf},
};

use crate::{
    app::{self, App, Args},
    escape,
};

use anyhow::Context;

const ABOUT: &'static str = "\
Generate TOML tests from Glenn Fowler's regex test suite.

This corresponds to a very sizeable set of regex tests that were written many
moons ago. They have been tweaked slightly by both Russ Cox and myself (Andrew
Gallant).

This tool is the spiritual successor of some hacky Python scripts. Its input is
a bespoke plain text format matching the original test data, and its output are
TOML files mean to work with the 'regex-test' crate.

Example usage from the root of this repository:

    regex-cli generate fowler tests/data/fowler tests/data/fowler/dat/*.dat

See tests/data/fowler/data/README for more context.
";

pub fn define() -> App {
    let outdir = app::arg("outdir")
        .help("Directory to write generated TOML files.")
        .required(true);
    let datfile = app::arg("datfile")
        .help("A AT&T regex test dat file.")
        .required(true)
        .multiple(true);
    app::leaf("fowler")
        .about("Generate TOML tests from Glenn Fowler's regex test suite.")
        .before_help(ABOUT)
        .arg(outdir)
        .arg(datfile)
}

pub fn run(args: &Args) -> anyhow::Result<()> {
    // OK because 'outdir' is marked as required.
    let outdir = PathBuf::from(args.value_of_os("outdir").unwrap());
    // Also OK because 'datfile' is marked as required.
    for datfile in args.values_of_os("datfile").unwrap() {
        let datfile = Path::new(datfile);
        let stem = match datfile.file_stem() {
            Some(stem) => stem.to_string_lossy(),
            None => anyhow::bail!("{}: has no file stem", datfile.display()),
        };
        let tomlfile = outdir.join(format!("{}.toml", stem));

        let mut rdr = File::open(datfile)
            .with_context(|| datfile.display().to_string())?;
        let mut wtr = File::create(&tomlfile)
            .with_context(|| tomlfile.display().to_string())?;
        convert(&stem, &mut rdr, &mut wtr)
            .with_context(|| stem.to_string())?;
    }
    Ok(())
}

fn convert(
    mut group_name: &str,
    src: &mut dyn Read,
    dst: &mut dyn Write,
) -> anyhow::Result<()> {
    log::trace!("processing {}", group_name);
    let src = std::io::BufReader::new(src);

    writeln!(
        dst,
        "\
# !!! DO NOT EDIT !!!
# Automatically generated by 'regex-cli generate fowler'.
# Numbers in the test names correspond to the line number of the test from
# the original dat file.
"
    )?;

    let mut prev = None;
    for (i, result) in src.lines().enumerate() {
        // Every usize can fit into a u64... Right?
        let line_number = u64::try_from(i).unwrap().checked_add(1).unwrap();
        let line = result.with_context(|| format!("line {}", line_number))?;
        // The last group of tests in 'repetition' take quite a lot of time
        // when using them to build and minimize a DFA. So we tag them with
        // 'expensive' so that we can skip those tests when we need to minimize
        // a DFA.
        if group_name == "repetition" && line.contains("Chris Kuklewicz") {
            group_name = "repetition-expensive";
        }
        if line.trim().is_empty() || line.starts_with('#') {
            // Too noisy to log that we're skipping an empty or commented test.
            continue;
        }
        let dat = match DatTest::parse(prev.as_ref(), line_number, &line)? {
            None => continue,
            Some(dat) => dat,
        };
        let toml = TomlTest::from_dat_test(group_name, &dat)?;
        writeln!(dst, "{}", toml)?;
        prev = Some(dat);
    }
    Ok(())
}

#[derive(Debug)]
struct TomlTest {
    group_name: String,
    line_number: u64,
    regex: String,
    haystack: String,
    captures: Vec<Option<(u64, u64)>>,
    unescape: bool,
    case_insensitive: bool,
    comment: Option<String>,
}

impl TomlTest {
    fn from_dat_test(
        group_name: &str,
        dat: &DatTest,
    ) -> anyhow::Result<TomlTest> {
        let mut captures = dat.captures.clone();
        if !captures.is_empty() {
            // Many of the Fowler tests don't actually list out every capturing
            // group match, and they instead stop once all remaining capturing
            // groups are empty. In effect, it makes writing tests terser,
            // but adds more implicitness. The TOML test suite does not make
            // this trade off (to this extent anyway), so it really wants all
            // capturing groups...
            //
            // So what we do here is is look for the number of groups in the
            // pattern and then just pad out the capture matches with None
            // values to make the number of capture matches equal to what we
            // would expect from the pattern. (We actually parse the regex to
            // determine this.)
            //
            // Sadly, this doesn't work for a small subset of tests that
            // actually have more capturing group MATCHES than what is listed
            // explicitly in the test. Instead, the test includes an 'nmatch'
            // instruction that instructs the test runner to only consider the
            // first N capturing groups. Our test runner has no such option...
            // To fix that, I rewrote the tests to use non-capturing groups in
            // order to match the expected number of expected capture matches.
            let numcaps = count_capturing_groups(&dat.regex)?;
            for _ in captures.len()..numcaps {
                captures.push(None);
            }
        }

        let comment = if dat.re2go {
            Some("Test added by RE2/Go project.".to_string())
        } else if dat.rust {
            Some("Test added by Rust regex project.".to_string())
        } else {
            None
        };
        Ok(TomlTest {
            group_name: group_name.to_string(),
            line_number: dat.line_number,
            regex: dat.regex.clone(),
            haystack: dat.haystack.clone(),
            captures,
            unescape: dat.flags.contains(&'$'),
            case_insensitive: dat.flags.contains(&'i'),
            comment,
        })
    }
}

impl core::fmt::Display for TomlTest {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        if let Some(ref comment) = self.comment {
            writeln!(f, "# {}", comment)?;
        }
        writeln!(f, "[[test]]")?;
        writeln!(f, "name = \"{}{}\"", self.group_name, self.line_number)?;
        writeln!(f, "regex = '''{}'''", self.regex)?;
        writeln!(f, "haystack = '''{}'''", self.haystack)?;
        if self.captures.is_empty() {
            writeln!(f, "matches = []")?;
        } else {
            write!(f, "matches = [[")?;
            for (i, &group) in self.captures.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                match group {
                    None => write!(f, "[]")?,
                    Some((start, end)) => write!(f, "[{}, {}]", start, end)?,
                }
            }
            writeln!(f, "]]")?;
        }
        writeln!(f, "match_limit = 1")?;
        // If the match starts at 0, then set anchored=true. This gives us more
        // coverage on the anchored option and lets regex engines like the
        // one-pass DFA participate a bit more in the test suite.
        if self
            .captures
            .get(0)
            .and_then(|&s| s)
            .map_or(false, |span| span.0 == 0)
        {
            writeln!(f, "anchored = true")?;
        }
        if self.unescape {
            writeln!(f, "unescape = true")?;
        }
        if self.case_insensitive {
            writeln!(f, "case_insensitive = true")?;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct DatTest {
    line_number: u64,
    flags: HashSet<char>,
    regex: String,
    haystack: String,
    captures: Vec<Option<(u64, u64)>>,
    re2go: bool,
    rust: bool,
}

impl DatTest {
    fn parse(
        prev: Option<&DatTest>,
        line_number: u64,
        line: &str,
    ) -> anyhow::Result<Option<DatTest>> {
        let fields: Vec<String> = line
            .split('\t')
            .map(|f| f.trim().to_string())
            .filter(|f| !f.is_empty())
            .collect();
        if !(4 <= fields.len() && fields.len() <= 5) {
            log::trace!(
                "skipping {}: too few or too many fields ({})",
                line_number,
                fields.len(),
            );
            return Ok(None);
        }

        // First field contains terse one-letter flags.
        let mut flags: HashSet<char> = fields[0].chars().collect();
        if !flags.contains(&'E') {
            log::trace!("skipping {}: does not contain 'E' flag", line_number);
            return Ok(None);
        }

        // Second field contains the regex pattern or 'SAME' if it's the
        // same as the regex in the previous test.
        let mut regex = fields[1].clone();
        if regex == "SAME" {
            regex = match prev {
                Some(test) => test.regex.clone(),
                None => anyhow::bail!(
                    "line {}: wants previous pattern but none is available",
                    line_number,
                ),
            };
        }

        // Third field contains the text to search or 'NULL'.
        let mut haystack = fields[2].clone();
        if haystack == "NULL" {
            haystack = "".to_string();
        }

        // Some tests have literal control characters in the regex/input
        // instead of using escapes. TOML freaks out at this, so we detect the
        // case, escape them and add '$' to our flags. (Which will ultimately
        // instruct the test harness to unescape the input.)
        if regex.chars().any(|c| c.is_control())
            || haystack.chars().any(|c| c.is_control())
        {
            flags.insert('$');
            regex = escape::escape(regex.as_bytes());
            haystack = escape::escape(haystack.as_bytes());
        }

        // Fourth field contains the capturing groups, or 'NOMATCH' or an
        // error.
        let mut captures = vec![];
        if fields[3] != "NOMATCH" {
            // Some tests check for a compilation error to occur, but we skip
            // these for now. We might consider adding them manually, or better
            // yet, just adding support for them here.
            if !fields[3].contains(',') {
                log::trace!(
                    "skipping {}: malformed capturing group",
                    line_number
                );
                return Ok(None);
            }
            let noparen = fields[3]
                .split(")(")
                .map(|x| x.trim_matches(|c| c == '(' || c == ')'));
            for group in noparen {
                let (start, end) = match group.split_once(',') {
                    Some((start, end)) => (start.trim(), end.trim()),
                    None => anyhow::bail!(
                        "line {}: invalid capturing group '{}' in '{}'",
                        line_number,
                        group,
                        fields[3]
                    ),
                };
                if start == "?" && end == "?" {
                    captures.push(None);
                } else {
                    let start = start.parse()?;
                    let end = end.parse()?;
                    captures.push(Some((start, end)));
                }
            }
        }

        // The fifth field is optional and contains some notes. Currently, this
        // is used to indicate tests added or modified by particular regex
        // implementations.
        let re2go = fields.len() >= 5 && fields[4] == "RE2/Go";
        let rust = fields.len() >= 5 && fields[4] == "Rust";

        Ok(Some(DatTest {
            line_number,
            flags,
            regex,
            haystack,
            captures,
            re2go,
            rust,
        }))
    }
}

fn count_capturing_groups(pattern: &str) -> anyhow::Result<usize> {
    let ast = syntax::ast::parse::Parser::new()
        .parse(pattern)
        .with_context(|| format!("failed to parse '{}'", pattern))?;
    // We add 1 to account for the capturing group for the entire
    // pattern.
    Ok(1 + count_capturing_groups_ast(&ast))
}

fn count_capturing_groups_ast(ast: &syntax::ast::Ast) -> usize {
    use syntax::ast::Ast;
    match *ast {
        Ast::Empty(_)
        | Ast::Flags(_)
        | Ast::Literal(_)
        | Ast::Dot(_)
        | Ast::Assertion(_)
        | Ast::Class(_) => 0,
        Ast::Repetition(ref rep) => count_capturing_groups_ast(&*rep.ast),
        Ast::Group(ref group) => {
            let this = if group.is_capturing() { 1 } else { 0 };
            this + count_capturing_groups_ast(&*group.ast)
        }
        Ast::Alternation(ref alt) => {
            alt.asts.iter().map(count_capturing_groups_ast).sum()
        }
        Ast::Concat(ref concat) => {
            concat.asts.iter().map(count_capturing_groups_ast).sum()
        }
    }
}
