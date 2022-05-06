macro_rules! log {
    ($($tt:tt)*) => {
        #[cfg(feature = "logging")]
        {
            $($tt)*
        }
    }
}

macro_rules! trace {
    ($($tt:tt)*) => { log!(log::trace!($($tt)*)) }
}

// The common implementation parts for a regex iterator. The handling of empty
// matches is in particular slightly subtle, so it's worth bundling it up into
// a macro like this. Repeating this code a couple times isn't a big deal, but
// there are a lot of iterators defined in this crate.
//
// I did think a bit about how I could do this without a macro, but I couldn't
// come up with any good ideas. The recursive call to 'next' is particularly
// troublesome.
//
// This requires that 'self' correspond to the receiver of a type that
// implements Iterator, and that:
//
//   self.last_end be a usize indicating the beginning of the next search.
//   self.text be a &[u8] corresponding to the bytes to search.
//   self.last_match be a Option<usize> that is the end of the last match.
//
// 'match' should correspond to an expression with type 'Match' and
// 'utf8' should be a boolean indicating whether we should behave as if we are
// searching valid UTF-8.
macro_rules! handle_iter_match {
    ($self:ident, $match:expr, $utf8:expr) => {{
        let (m, utf8) = ($match, $utf8);
        if m.is_empty() {
            // This is an empty match. To ensure we make progress, start
            // the next search at the smallest possible starting position
            // of the next match following this one.
            $self.last_end = if utf8 {
                crate::util::next_utf8($self.text, m.end())
            } else {
                m.end() + 1
            };
            // Don't accept empty matches immediately following a match.
            // Just move on to the next match.
            if Some(m.end()) == $self.last_match {
                // This recursive call might raise alarm bells, but it's
                // guaranteed to only be called once. Namely, every call to
                // 'next' always increases self.last_end by at least 1. Since
                // self.last_end is the start of the next search, it follows
                // that the subsequent value of m.end() cannot be equal to
                // self.last_match since the previous value (this iteration)
                // is equal to self.last_match.
                return $self.next();
            }
        } else {
            $self.last_end = m.end();
        }
        $self.last_match = Some(m.end());
        m
    }};
}

macro_rules! handle_iter_match_fallible {
    ($self:ident, $result:expr, $utf8:expr) => {{
        let m = match $result {
            Err(err) => return Some(Err(err)),
            Ok(None) => return None,
            Ok(Some(m)) => m,
        };
        handle_iter_match!($self, m, $utf8)
    }};
}

macro_rules! handle_iter_match_overlapping {
    ($self:ident, $match:expr) => {{
        let m = $match;
        // Unlike the non-overlapping case, we're OK with empty matches at this
        // level. In particular, the overlapping search algorithm is itself
        // responsible for ensuring that progress is always made.
        $self.last_end = m.end();
        m
    }};
}

macro_rules! handle_iter_match_overlapping_fallible {
    ($self:ident, $result:expr) => {{
        let m = match $result {
            Err(err) => return Some(Err(err)),
            Ok(None) => return None,
            Ok(Some(m)) => m,
        };
        handle_iter_match_overlapping!($self, m)
    }};
}
