use regex_automata::{Error, ErrorKind, Regex, RegexBuilder};

use collection::{SUITE, RegexTest, RegexTestOption};

#[test]
fn unminimized_standard() {
    let mut builder = RegexBuilder::new();
    builder.minimize(false).premultiply(false).byte_classes(false);

    SUITE.tester().is_match(&mut *regex_builder_is_match(&builder)).assert();
    SUITE.tester().find(&mut *regex_builder_find(&builder)).assert();
}

#[test]
fn unminimized_premultiply() {
    let mut builder = RegexBuilder::new();
    builder.minimize(false).premultiply(true).byte_classes(false);

    SUITE.tester().is_match(&mut *regex_builder_is_match(&builder)).assert();
    SUITE.tester().find(&mut *regex_builder_find(&builder)).assert();
}

#[test]
fn unminimized_byte_class() {
    let mut builder = RegexBuilder::new();
    builder.minimize(false).premultiply(false).byte_classes(true);

    SUITE.tester().is_match(&mut *regex_builder_is_match(&builder)).assert();
    SUITE.tester().find(&mut *regex_builder_find(&builder)).assert();
}

#[test]
fn unminimized_premultiply_byte_class() {
    let mut builder = RegexBuilder::new();
    builder.minimize(false).premultiply(true).byte_classes(true);

    SUITE.tester().is_match(&mut *regex_builder_is_match(&builder)).assert();
    SUITE.tester().find(&mut *regex_builder_find(&builder)).assert();
}

#[test]
fn minimized_standard() {
    let mut builder = RegexBuilder::new();
    builder.minimize(true).premultiply(false).byte_classes(false);

    SUITE.tester().is_match(&mut *regex_builder_is_match(&builder)).assert();
    SUITE.tester().find(&mut *regex_builder_find(&builder)).assert();
}

#[test]
fn minimized_premultiply() {
    let mut builder = RegexBuilder::new();
    builder.minimize(true).premultiply(true).byte_classes(false);

    SUITE.tester().is_match(&mut *regex_builder_is_match(&builder)).assert();
    SUITE.tester().find(&mut *regex_builder_find(&builder)).assert();
}

#[test]
fn minimized_byte_class() {
    let mut builder = RegexBuilder::new();
    builder.minimize(true).premultiply(false).byte_classes(true);

    SUITE.tester().is_match(&mut *regex_builder_is_match(&builder)).assert();
    SUITE.tester().find(&mut *regex_builder_find(&builder)).assert();
}

#[test]
fn minimized_premultiply_byte_class() {
    let mut builder = RegexBuilder::new();
    builder.minimize(true).premultiply(true).byte_classes(true);

    SUITE.tester().is_match(&mut *regex_builder_is_match(&builder)).assert();
    SUITE.tester().find(&mut *regex_builder_find(&builder)).assert();
}

/// Create a closure built from the given builder that can be used to run a
/// suite's `is_match` tests.
fn regex_builder_is_match(
    builder: &RegexBuilder,
) -> Box<FnMut(&RegexTest) -> Option<bool>>
{
    let mut builder = builder.clone();
    Box::new(move |test| {
        build_regex(&mut builder, test).map(|m| m.is_match(&test.input))
    })
}

/// Create a closure built from the given builder that can be used to run a
/// suite's `find` tests.
fn regex_builder_find(
    builder: &RegexBuilder,
) -> Box<FnMut(&RegexTest) -> Option<Option<(usize, usize)>>>
{
    let mut builder = builder.clone();
    Box::new(move |test| {
        build_regex(&mut builder, test).map(|m| m.find(&test.input))
    })
}

/// Build a regex from the given builder using the given test's configuration.
///
/// If the test demands unsupported features, then this returns `None`. If the
/// test reports an invalid regex, then this panics with an error and fails
/// the current test.
fn build_regex(
    builder: &mut RegexBuilder,
    test: &RegexTest,
) -> Option<Regex<'static>> {
    regex_builder_apply_options(builder, &test.options);
    ignore_unsupported(builder.build(&test.pattern))
}

/// Apply the given test options to the given builder.
fn regex_builder_apply_options(
    builder: &mut RegexBuilder,
    opts: &[RegexTestOption],
) {
    for opt in opts {
        match *opt {
            RegexTestOption::CaseInsensitive => {
                builder.case_insensitive(true);
            }
            RegexTestOption::NoUnicode => {
                builder.unicode(false);
            }
            RegexTestOption::Escaped => {}
            RegexTestOption::InvalidUTF8 => {
                builder.allow_invalid_utf8(true);
            }
        }
    }
}

/// If the given result is an error and is unsupported, then return `None`.
/// Otherwise, for any other error, panic.
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
