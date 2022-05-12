// BREADCRUMBS: I suppose we should decide whether this iterator helper is
// something we should really commit to. We can use it to impl our own
// iterators but also provide it for others to impl their own. Using a closure
// is unfortunate because it means you can't name it, which in turn means that
// composing this helper requires dynamic dispatch (AFAIK). Probably not a huge
// deal in practice, and if folks really can't afford it, then they can write
// their own damn iterator.
//
// Another idea is whether we can somehow factor out the handling of empty
// matches. That's really the only bit that makes writing an iterator
// complicated.
//
// This should get simpler once we use util::matchtypes::Search.
//
// Also, maybe use H: AsRef<[u8]> for haystack. Which I guess we can do with
// 'Search'. Also permits starting/ending an iterator anywhere and also doing
// 'earliest' searches.
//
// What about the overlapping case? Maybe we don't need a helper for it since
// overlapping searches don't need special handling for empty matches. But maybe
// we should offer it for completeness?
//
// Also, provide an infallible helper, Find, as well.
//
// ... sigh. What about iterators over capturing groups? Can we build
// them on top of this infrastructure? Yes, I believe we can, because of
// Captures::get_match. Although... the captures seem like they'll be bottled
// up inside the closure. So how do we get them out? Either we use some crazy
// iterior mutability scheme or we hand-roll those iterators or we build
// a distinct helper... Sigh....... Yeah, I think the 'F' type has to be
// different. Gah.

use crate::util::{
    matchtypes::{Match, MatchError},
    prefilter,
};

pub struct TryFind<'h, F> {
    find: F,
    haystack: &'h [u8],
    start: usize,
    last_match: Option<usize>,
    utf8: bool,
}

impl<'h, 'c, F> TryFind<'h, F>
where
    F: FnMut(&'h [u8], usize, usize) -> Result<Option<Match>, MatchError> + 'c,
{
    pub fn new(haystack: &'h [u8], find: F) -> TryFind<'h, F> {
        TryFind { find, haystack, start: 0, last_match: None, utf8: true }
    }

    pub fn boxed(
        haystack: &'h [u8],
        find: F,
    ) -> TryFind<
        'h,
        Box<
            dyn FnMut(
                    &'h [u8],
                    usize,
                    usize,
                ) -> Result<Option<Match>, MatchError>
                + 'c,
        >,
    > {
        TryFind::new(haystack, Box::new(find))
    }

    pub fn utf8(self, yes: bool) -> TryFind<'h, F> {
        TryFind { utf8: yes, ..self }
    }

    #[cold]
    #[inline(never)]
    fn handle_empty(
        &mut self,
        mut m: Match,
    ) -> Option<Result<Match, MatchError>> {
        assert!(m.is_empty());
        self.start = if self.utf8 {
            crate::util::next_utf8(self.haystack, m.end())
        } else {
            m.end() + 1
        };
        if Some(m.end()) == self.last_match {
            if self.start > self.haystack.len() {
                return None;
            }
            let result =
                (self.find)(self.haystack, self.start, self.haystack.len());
            m = match result {
                Err(err) => return Some(Err(err)),
                Ok(None) => return None,
                Ok(Some(m)) => m,
            };
            if m.is_empty() {
                self.start = if self.utf8 {
                    crate::util::next_utf8(self.haystack, m.end())
                } else {
                    m.end() + 1
                };
            } else {
                self.start = m.end();
            }
        }
        Some(Ok(m))
    }
}

impl<'h, F> Iterator for TryFind<'h, F>
where
    F: FnMut(&'h [u8], usize, usize) -> Result<Option<Match>, MatchError>,
{
    type Item = Result<Match, MatchError>;

    #[inline]
    fn next(&mut self) -> Option<Result<Match, MatchError>> {
        if self.start > self.haystack.len() {
            return None;
        }
        let result =
            (self.find)(self.haystack, self.start, self.haystack.len());
        let mut m = match result {
            Err(err) => return Some(Err(err)),
            Ok(None) => return None,
            Ok(Some(m)) => m,
        };
        if m.is_empty() {
            m = match self.handle_empty(m)? {
                Err(err) => return Some(Err(err)),
                Ok(m) => m,
            };
        } else {
            self.start = m.end();
        }
        self.last_match = Some(m.end());
        Some(Ok(m))
    }
}
