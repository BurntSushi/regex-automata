#[cfg(all(
    // We have to explicitly want to support Unicode word boundaries.
    feature = "unicode-word-boundary",
    any(
        // If we don't have regex-syntax at all, then we definitely need to
        // bring our own \w data table.
        not(feature = "syntax"),
        // If unicode-perl is enabled, then regex-syntax/unicode-perl is
        // also enabled, which in turn means we can use regex-syntax's
        // is_word_character routine (and thus use its data tables). But if
        // unicode-perl is not enabled, even if syntax is, then we need to
        // bring our own.
        not(feature = "unicode-perl"),
    ),
))]
mod perl_word;
