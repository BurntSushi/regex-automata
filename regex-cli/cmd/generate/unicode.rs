use std::{
    fs::File,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

use {
    anyhow::Context,
    bstr::BString,
    lexopt::{Arg, Parser},
};

use crate::{
    args::{self, Usage},
    util,
};

pub fn run(p: &mut Parser) -> anyhow::Result<()> {
    const USAGE: &'static str = "\
Generates all Unicode tables for the regex project.

Most Unicode tables are generated into the regex-syntax library.

Note that this requires that the 'ucd-generate' tool be installed and in your
PATH. The 'ucd-generate' tool is what is responsible for reading from the
Unicode Character Database (UCD) and converting tables of codepoints into Rust
code that is embedded into the regex library.

ucd-generate can be found here https://github.com/BurntSushi/ucd-generate/
and can be installed with:

    cargo install ucd-generate

USAGE:
    regex-cli generate unicode <outdir> <ucddir>

    outdir should be a directory path pointing to the root of the regex
    repository. ucddir should be a directory containing the UCD data
    downloaded from unicode.org, as described in ucd-generate's README:
    https://github.com/BurntSushi/ucd-generate/

";

    let mut config = Config::default();
    args::configure(p, USAGE, &mut [&mut config])?;

    let outdir = config.outdir()?;
    let ucddir = config.ucddir()?;

    // Data tables for regex-automata proper.
    let pre = outdir.join("src").join("util").join("unicode_data");
    let dest = pre.join("perl_word.rs");
    ucdgen_to(&["perl-word", &ucddir, "--chars"], &dest)?;
    util::rustfmt(&dest)?;

    Ok(())
}

#[derive(Debug, Default)]
struct Config {
    outdir: Option<PathBuf>,
    ucddir: Option<PathBuf>,
}

impl Config {
    fn outdir(&self) -> anyhow::Result<&Path> {
        self.outdir
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("missing <outdir>"))
    }

    fn ucddir(&self) -> anyhow::Result<&str> {
        self.ucddir
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("missing <ucddir>"))?
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("ucddir must be valid UTF-8"))
    }
}

impl args::Configurable for Config {
    fn configure(
        &mut self,
        _: &mut Parser,
        arg: &mut Arg,
    ) -> anyhow::Result<bool> {
        match *arg {
            Arg::Value(ref mut value) => {
                if self.outdir.is_none() {
                    self.outdir = Some(PathBuf::from(std::mem::take(value)));
                } else if self.ucddir.is_none() {
                    self.ucddir = Some(PathBuf::from(std::mem::take(value)));
                } else {
                    return Ok(false);
                }
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn usage(&self) -> &[Usage] {
        const USAGES: &'static [Usage] = &[];
        USAGES
    }
}

/// Run 'ucd-generate' with the args given and write its output to the file
/// path given.
fn ucdgen_to<P: AsRef<Path>>(args: &[&str], dest: P) -> anyhow::Result<()> {
    let dest = dest.as_ref();
    let err = || format!("{}", dest.display());
    // The "right" thing would be to connect this to the stdout of
    // ucd-generate, but meh, I got lazy.
    let mut fdest = File::create(dest).with_context(err)?;
    let data = ucdgen(args)?;
    fdest.write_all(&data).with_context(err)?;
    Ok(())
}

/// Run 'ucd-generate' with the args given. Upon success, the contents of
/// stdout are returned. Otherwise, the error will include the contents of
/// stderr.
fn ucdgen(args: &[&str]) -> anyhow::Result<Vec<u8>> {
    let out = Command::new("ucd-generate")
        .args(args)
        .output()
        .context("ucd-generate command failed")?;
    anyhow::ensure!(
        out.status.success(),
        "ucd-generate {}: {}",
        args.join(" "),
        BString::from(out.stderr),
    );
    Ok(out.stdout)
}
