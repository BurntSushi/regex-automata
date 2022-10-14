use std::{
    fs::File,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

use crate::app::{self, App, Args};

use anyhow::Context;
use bstr::BString;

const ABOUT: &'static str = "\
Generate all Unicode tables required for the regex project.
";

pub fn define() -> App {
    let outdir = app::arg("outdir")
        .help("Directory containing the root of the regex repository.")
        .required(true);
    let ucddir = app::arg("ucddir")
        .help("Directory containing the UCD download.")
        .required(true);
    app::leaf("unicode")
        .about("Generate all Unicode tables required for the regex project.")
        .before_help(ABOUT)
        .arg(outdir)
        .arg(ucddir)
}

pub fn run(args: &Args) -> anyhow::Result<()> {
    // OK because both are marked as required. We also require UTF-8 for ucddir
    // because it's a pain otherwise. No biggie in a tool like this.
    let outdir = PathBuf::from(args.value_of_os("outdir").unwrap());
    let ucddir = args
        .value_of_os("ucddir")
        .unwrap()
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("ucddir must be valid UTF-8"))?;

    // Data tables for regex-automata proper.
    let pre = outdir.join("src").join("util").join("unicode_data");
    let dest = pre.join("perl_word.rs");
    ucdgen_to(&["perl-word", &ucddir, "--chars"], &dest)?;
    rustfmt(&dest)?;

    Ok(())
}

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

fn ucdgen_to<P: AsRef<Path>>(args: &[&str], dest: P) -> anyhow::Result<()> {
    let dest = dest.as_ref();
    let err = || format!("{}", dest.display());
    let mut fdest = File::create(dest).with_context(err)?;
    let data = ucdgen(args)?;
    fdest.write_all(&data).with_context(err)?;
    Ok(())
}

fn rustfmt<P: AsRef<Path>>(path: P) -> anyhow::Result<()> {
    let path = path.as_ref();
    let out = Command::new("rustfmt")
        .arg(path)
        .output()
        .context("rustfmt command failed")?;
    anyhow::ensure!(
        out.status.success(),
        "rustfmt {}: {}",
        path.display(),
        BString::from(out.stderr),
    );
    Ok(())
}
