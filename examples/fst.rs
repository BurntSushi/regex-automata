// To run this example, use:
//
//     cargo run --manifest-path examples/Cargo.toml --example fst

use fst::{IntoStreamer, Set};
use regex_automata::dfa::dense;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let set = Set::from_iter(&["FoO", "Foo", "fOO", "foo"])?;
    let pattern = r"(?i)foo";
    let config = dense::Config::new().anchored(true);
    let dfa = dense::Builder::new().configure(config).build(pattern)?;

    let keys = set.search(&dfa).into_stream().into_strs()?;
    assert_eq!(keys, vec!["FoO", "Foo", "fOO", "foo"]);
    println!("{:?}", keys);
    Ok(())
}
