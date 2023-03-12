use std::io::{self, Write};

use unicode_width::UnicodeWidthStr;

/// Time an arbitrary operation.
pub fn timeit<T>(run: impl FnOnce() -> T) -> (T, std::time::Duration) {
    let start = std::time::Instant::now();
    let t = run();
    (t, start.elapsed())
}

/// Convenient time an operation that returns a result by packing the duration
/// into the `Ok` variant.
pub fn timeitr<T, E>(
    run: impl FnOnce() -> Result<T, E>,
) -> Result<(T, std::time::Duration), E> {
    let (result, time) = timeit(run);
    let t = result?;
    Ok((t, time))
}

/// Print the given text with an ASCII art underline beneath it.
///
/// If the given text is empty, then `<empty>` is printed.
pub fn print_with_underline<W: io::Write>(
    mut wtr: W,
    text: &str,
) -> io::Result<()> {
    let toprint = if text.is_empty() { "<empty>" } else { text };
    writeln!(wtr, "{}", toprint)?;
    writeln!(wtr, "{}", "-".repeat(toprint.width()))?;
    Ok(())
}

/// A somewhat silly little thing that prints an aligned table of key-value
/// pairs. Keys can be any string and values can be anything that implements
/// Debug.
///
/// This table is used to print little bits of useful information about stuff.
#[derive(Debug)]
pub struct Table {
    pairs: Vec<(String, Box<dyn std::fmt::Debug>)>,
}

impl Table {
    pub fn empty() -> Table {
        Table { pairs: vec![] }
    }

    pub fn add<D: std::fmt::Debug + 'static>(
        &mut self,
        label: &str,
        value: D,
    ) {
        self.pairs.push((label.to_string(), Box::new(value)));
    }

    pub fn print<W: io::Write>(&self, wtr: W) -> io::Result<()> {
        let mut wtr = tabwriter::TabWriter::new(wtr)
            .alignment(tabwriter::Alignment::Right);
        for (label, value) in self.pairs.iter() {
            writeln!(wtr, "{}:\t{:?}", label, value)?;
        }
        wtr.flush()
    }
}
