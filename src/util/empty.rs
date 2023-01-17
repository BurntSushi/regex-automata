use crate::util::search::{Input, MatchError};

#[inline]
pub(crate) fn skip_splits_fwd<T, F>(
    input: &Input<'_>,
    init_value: T,
    match_offset: usize,
    find: F,
) -> Result<Option<T>, MatchError>
where
    F: FnMut(&Input<'_>) -> Result<Option<(T, usize)>, MatchError>,
{
    skip_splits(true, input, init_value, match_offset, find)
}

#[inline]
pub(crate) fn skip_splits_rev<T, F>(
    input: &Input<'_>,
    init_value: T,
    match_offset: usize,
    find: F,
) -> Result<Option<T>, MatchError>
where
    F: FnMut(&Input<'_>) -> Result<Option<(T, usize)>, MatchError>,
{
    skip_splits(false, input, init_value, match_offset, find)
}

#[cold]
#[inline(never)]
fn skip_splits<T, F>(
    forward: bool,
    input: &Input<'_>,
    init_value: T,
    mut match_offset: usize,
    mut find: F,
) -> Result<Option<T>, MatchError>
where
    F: FnMut(&Input<'_>) -> Result<Option<(T, usize)>, MatchError>,
{
    // If our config says to do an anchored search, then we're definitely
    // done. We just need to determine whether we have a valid match or
    // not. If we don't, then we're not allowed to continue, so we report
    // no match.
    //
    // This is actually quite a subtle correctness thing. The key here is
    // that if we got an empty match that splits a codepoint after doing an
    // anchored search in UTF-8 mode, then that implies that we must have
    // *started* the search at a location that splits a codepoint. This
    // follows from the fact that if a match is reported from an anchored
    // search, then the start offset of the match *must* match the start
    // offset of the search.
    //
    // It also follows that no other non-empty match is possible. For
    // example, you might write a regex like '(?:)|SOMETHING' and start its
    // search in the middle of a codepoint. The first branch is an empty
    // regex that will bubble up a match at the first position, and then
    // get rejected here and report no match. But what if 'SOMETHING' could
    // have matched? We reason that such a thing is impossible, because
    // if it does, it must report a match the starts in the middle of a
    // codepoint. This in turn implies that a match is reported whose span
    // does not correspond to valid UTF-8, and this breaks the promise
    // made when UTF-8 mode is enabled. (That promise *can* be broken, for
    // example, by enabling UTF-8 mode but building an NFA that produce
    // non-empty matches that span invalid UTF-8. This is an unchecked but
    // documented precondition violation of UTF-8 mode, and is documented
    // to have unspecified behavior.)
    //
    // I believe this actually means that if an anchored search is run, and
    // UTF-8 mode is enabled and the start position splits a codepoint,
    // then it is correct to immediately report no match without even
    // executing the regex engine. But it doesn't really seem worth writing
    // out that case in every regex engine to save a tiny bit of work in an
    // extremely pathological case, so we just handle it here.
    if input.get_anchored().is_anchored() {
        return Ok(if input.is_char_boundary(match_offset) {
            Some(init_value)
        } else {
            None
        });
    }
    // Otherwise, we have an unanchored search, so just keep looking for
    // matches until we have one that does not split a codepoint or we hit
    // EOI.
    let mut value = init_value;
    let mut input = input.clone();
    while !input.is_char_boundary(match_offset) {
        if forward {
            // The unwrap is OK here because overflowing usize while
            // iterating over a slice is impossible, at it would require
            // a slice of length greater than isize::MAX, which is itself
            // impossible.
            input.set_start(input.start().checked_add(1).unwrap());
        } else {
            input.set_end(match input.end().checked_sub(1) {
                None => return Ok(None),
                Some(end) => end,
            });
        }
        match find(&input)? {
            None => return Ok(None),
            Some((new_value, new_match_end)) => {
                value = new_value;
                match_offset = new_match_end;
            }
        }
    }
    Ok(Some(value))
}
