use regex_automata::{Error, ErrorKind, Regex, RegexBuilder};

use collection::{SUITE, RegexTest, RegexTestOption};

#[test]
fn regex_byte_class() {
    let mut builder = RegexBuilder::new();
    builder.minimize(false).premultiply(false).byte_classes(true);

    SUITE.tester().is_match(&mut *regex_builder_is_match(&builder)).assert();
    SUITE.tester().find(&mut *regex_builder_find(&builder)).assert();
}

fn regex_builder_is_match(
    builder: &RegexBuilder,
) -> Box<FnMut(&RegexTest) -> Option<bool>>
{
    let mut builder = builder.clone();
    Box::new(move |test| {
        build_regex(&mut builder, test).map(|m| m.is_match(&test.input))
    })
}

fn regex_builder_find(
    builder: &RegexBuilder,
) -> Box<FnMut(&RegexTest) -> Option<Option<(usize, usize)>>>
{
    let mut builder = builder.clone();
    Box::new(move |test| {
        build_regex(&mut builder, test).map(|m| m.find(&test.input))
    })
}

fn build_regex(
    builder: &mut RegexBuilder,
    test: &RegexTest,
) -> Option<Regex<'static>> {
    regex_builder_apply_options(builder, &test.options);
    ignore_unsupported(builder.build(&test.pattern))
}

fn regex_builder_apply_options(
    builder: &mut RegexBuilder,
    opts: &[RegexTestOption],
) {
    for opt in opts {
        match *opt {
            RegexTestOption::CaseInsensitive => {
                builder.case_insensitive(true);
            }
        }
    }
}

fn ignore_unsupported<T>(res: Result<T, Error>) -> Option<T> {
    let err = match res {
        Ok(t) => return Some(t),
        Err(err) => err,
    };
    if let ErrorKind::Unsupported(_) = *err.kind() {
        return None;
    }
    panic!("{}", err);
}
