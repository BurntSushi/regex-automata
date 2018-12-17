use regex_automata::{Error, ErrorKind, Matcher, MatcherBuilder};

use load::{SUITE, RegexTest, RegexTestOption};

#[test]
fn matcher_byte_class() {
    let mut builder = MatcherBuilder::new();
    builder.minimize(false).premultiply(false).byte_classes(true);

    SUITE.tester().is_match(&mut *matcher_builder_is_match(&builder)).assert();
    SUITE.tester().find(&mut *matcher_builder_find(&builder)).assert();
}

fn matcher_builder_is_match(
    builder: &MatcherBuilder,
) -> Box<FnMut(&RegexTest) -> Option<bool>>
{
    let mut builder = builder.clone();
    Box::new(move |test| {
        build_matcher(&mut builder, test).map(|m| m.is_match(&test.input))
    })
}

fn matcher_builder_find(
    builder: &MatcherBuilder,
) -> Box<FnMut(&RegexTest) -> Option<Option<(usize, usize)>>>
{
    let mut builder = builder.clone();
    Box::new(move |test| {
        build_matcher(&mut builder, test).map(|m| m.find(&test.input))
    })
}

fn build_matcher(
    builder: &mut MatcherBuilder,
    test: &RegexTest,
) -> Option<Matcher<'static>> {
    matcher_builder_apply_options(builder, &test.options);
    ignore_unsupported(builder.build(&test.pattern))
}

fn matcher_builder_apply_options(
    builder: &mut MatcherBuilder,
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
