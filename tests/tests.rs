mod dfa;
mod hybrid;
mod nfa;
mod util;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn suite() -> Result<ret::RegexTests> {
    let mut tests = ret::RegexTests::new();
    macro_rules! load {
        ($name:expr) => {{
            const DATA: &[u8] =
                include_bytes!(concat!("data/", $name, ".toml"));
            tests.load_slice($name, DATA)?;
        }};
    }

    load!("bytes");
    load!("crazy");
    load!("earliest");
    load!("empty");
    load!("expensive");
    load!("flags");
    load!("iter");
    load!("misc");
    load!("multiline");
    load!("no-unicode");
    load!("overlapping");
    load!("regression");
    load!("set");
    load!("unicode");
    load!("word-boundary");
    load!("fowler/basic");
    load!("fowler/nullsubexpr");
    load!("fowler/repetition");

    Ok(tests)
}
