#![allow(non_camel_case_types)]

use std::ptr::NonNull;

use {
    automata::{util::iter, Input, Match, PatternID, Span},
    libc::{c_int, c_uchar, c_void},
};

/// Regex wraps an RE2 regular expression.
///
/// It cannot be used safely from multiple threads simultaneously.
pub struct Regex {
    re: NonNull<re2_regexp>,
    pattern: String,
}

// SAFETY: RE2 provides the guarantee that its regex is safe to use from
// multiple threads simultaneously.
unsafe impl Send for Regex {}

impl std::fmt::Debug for Regex {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.pattern)
    }
}

impl Drop for Regex {
    fn drop(&mut self) {
        // SAFETY: We know our 're' pointer is valid by construction.
        // Otherwise, we rely on Rust's guarantee that 'Drop' is called at most
        // once to guarantee safety.
        unsafe {
            re2_regexp_free(self.re.as_ptr());
        }
    }
}

impl Regex {
    /// Create a new RE2 regex with the given configuration. If one could not
    /// be created, then an error is returned.
    ///
    /// Currently, the error returned doesn't include any RE2-specific
    /// diagnostics. However, RE2 is likely configured to log errors to stderr.
    pub fn new(pattern: &str, opts: Options) -> anyhow::Result<Regex> {
        // SAFETY: If compilation fails and/or throws an exception, then
        // nullptr is returned which we convert into a generic error here.
        match NonNull::new(unsafe { re2_regexp_new(pattern.into(), opts) }) {
            Some(re) => Ok(Regex { re, pattern: pattern.to_string() }),
            // We don't make any attempt at extracting the error message
            // from RE2. We probably should, but my C++ skills suck.
            None => Err(anyhow::anyhow!(
                "RE2 regex compilation failed for: {}",
                pattern
            )),
        }
    }

    /// Create a new 'Captures' value that is sized to be able to hold all
    /// possible capturing groups (including the implicit unnamed group) in
    /// this regex.
    pub fn create_captures(&self) -> Captures {
        Captures::new(1 + self.explicit_capture_len())
    }

    /// Return the number of explicit capturing groups for this regex. This
    /// does not include the implicit unnamed capturing group corresponding to
    /// the overall match span of the regex.
    fn explicit_capture_len(&self) -> usize {
        // SAFETY: We know our regex point is valid by construction.
        unsafe { re2_regexp_capture_len(self.re.as_ptr()) as usize }
    }

    /// Return true if the given input matches anywhere.
    pub fn is_match(&self, input: &Input<'_>) -> bool {
        // SAFETY: By construction, self.re is non-null. So is our haystack.
        // We assume RE2 searching can't throw an exception.
        unsafe {
            re2_regexp_is_match(
                self.re.as_ptr(),
                input.haystack().into(),
                input.start() as c_int,
                input.end() as c_int,
            )
        }
    }

    /// Return the first 'Match' found in the given input. If no such match
    /// exists, then return None.
    pub fn find(&self, input: &Input<'_>) -> Option<Match> {
        let (mut match_start, mut match_end): (c_int, c_int) = (0, 0);
        // SAFETY: By construction, self.re is non-null. So is our haystack.
        // We assume RE2 searching can't throw an exception.
        let matched = unsafe {
            re2_regexp_find(
                self.re.as_ptr(),
                input.haystack().into(),
                input.start() as c_int,
                input.end() as c_int,
                &mut match_start,
                &mut match_end,
            )
        };
        if matched {
            let span =
                Span { start: match_start as usize, end: match_end as usize };
            Some(Match::new(PatternID::ZERO, span))
        } else {
            None
        }
    }

    /// Return an iterator over all non-overlapping successive matches in the
    /// given input.
    pub fn find_iter<'r, 'h>(
        &'r self,
        input: Input<'h>,
    ) -> FindMatches<'r, 'h> {
        let it = iter::Searcher::new(input);
        FindMatches { re: self, it }
    }

    /// Write the matching capturing groups in 'caps' if a match could be
    /// found in the given input.
    pub fn captures(&self, input: &Input<'_>, caps: &mut Captures) -> bool {
        // We make sure to reset this to avoid having incorrect haystack
        // addresses in our captures. Also, this being None is a sentinel for
        // indicating that 'Captures' does not represent a match, regardless of
        // the rest of its contents.
        caps.haystack_start_addr = None;
        // SAFETY: By construction, self.re is non-null. So is our haystack. If
        // 'caps' was created for a different regex, then the worst thing that
        // happens is it passes an 'nsubmatch' value to RE2::Match that is too
        // big. RE2's API declares that it handles this case just fine. If the
        // 'caps' value is too small, that's OK too since an 'nsubmatch' value
        // of 0 is fine in all cases. A too small 'caps' will likely lead to
        // logic errors, but should never lead to UB. Finally, we assume RE2
        // searching can't throw an exception.
        let matched = unsafe {
            re2_regexp_captures(
                self.re.as_ptr(),
                input.haystack().into(),
                input.start() as c_int,
                input.end() as c_int,
                caps.caps.as_ptr(),
            )
        };
        if matched {
            // We use this to convert substrings reported by RE2's API into
            // offset spans. We never convert this integer back into a pointer.
            caps.haystack_start_addr =
                Some(input.haystack().as_ptr() as usize);
        }
        matched
    }
}

/// Options that can be passed to Regex::new to configure a subset of RE2
/// knobs.
///
/// Note that since this is such a simple type, we just make it repr(C)
/// directly. It is meant to be equivalent to the re2_options type defined in
/// the shim layer.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct Options {
    /// When disabled, RE2's "latin1" mode is enabled. Otherwise, UTF-8 is
    /// enabled. RE2's "latin1" mode is (I believe) the same as disabling
    /// regex-automata's Unicode and UTF-8 modes. Namely, it permits an RE2
    /// regex to match arbitrary bytes and it caps the range of any character
    /// class to be at most 0xFF.
    pub utf8: bool,
    /// When enabled, RE2's case sensitive mode is enabled. When disabled,
    /// matching is done case insensitively.
    pub case_sensitive: bool,
}

impl Default for Options {
    fn default() -> Options {
        Options { utf8: true, case_sensitive: true }
    }
}

/// An allocation that can hold some number of capturing groups reported by
/// RE2. A 'Captures' may be reused in multiple calls to 'Regex::captures'.
pub struct Captures {
    /// A pointer to an opaque 'captures' type provided by our shim layer.
    caps: NonNull<re2_captures>,
    /// The number of submatches (including the implicit one tracking the
    /// overall match) tracked by these capturing groups. Any call to
    /// 'get_group' with an index less than this number is valid.
    group_len: usize,
    /// The start address of the haystack these capture groups were reported
    /// from. We use this to convert 're2_string' values returned by
    /// 're2_captures_get' into bare spans. We never convert this back into a
    /// pointer.
    ///
    /// This is 'None' when this 'Captures' doesn't correspond to a match.
    haystack_start_addr: Option<usize>,
}

impl Captures {
    /// Create a new 'Captures' value with room for 'group_len' groups. This
    /// length should generally include the 0th group corresponding to the
    /// implicit unnamed group for the whole match.
    fn new(group_len: usize) -> Captures {
        // SAFETY: 're2_captures_new' should be safe for all inputs.
        match NonNull::new(unsafe { re2_captures_new(group_len as c_int) }) {
            None => unreachable!("re2_captures_new should always work"),
            Some(caps) => {
                Captures { caps, group_len, haystack_start_addr: None }
            }
        }
    }

    /// Return the match for this 'Captures' value. The match span always
    /// corresponds to the group span at index 0.
    ///
    /// Since this regex engine only supports matching one pattern, the pattern
    /// ID returned in the match is always `PatternID::ZERO`.
    pub fn get_match(&self) -> Option<Match> {
        self.get_group(0).map(|span| Match::new(PatternID::ZERO, span))
    }

    /// Return the span for the group at the given index, if it participated in
    /// a match. If the index is invalid, then return None. If this 'Captures'
    /// value does not represent a match, then None is always returned.
    ///
    /// The span for the group at index 0 always corresponds to the span
    /// reported in the 'Match' returned by 'get_match'.
    pub fn get_group(&self, index: usize) -> Option<Span> {
        // For an invalid index, we just return None. This matches the behavior
        // of regex_automata::nfa::thompson::Captures.
        if index >= self.group_len {
            return None;
        }
        // Similarly, return None if not a match.
        let haystack_start_addr = self.haystack_start_addr?;
        // SAFETY: We know our pointer is valid and non-null by construction.
        // We also know the index is valid because of the check above.
        let submatch =
            unsafe { re2_captures_get(self.caps.as_ptr(), index as c_int) };
        if submatch.data.is_null() {
            return None;
        }
        let start =
            (submatch.data as usize).checked_sub(haystack_start_addr).unwrap();
        let end = start.checked_add(submatch.length as usize).unwrap();
        Some(Span { start, end })
    }

    /// Return the total number of capturing groups in this allocation. This
    /// always includes all groups (including the implicit group for the
    /// overall match), including groups that may not have participated in a
    /// match.
    pub fn group_len(&self) -> usize {
        self.group_len
    }
}

// SAFETY: A 'Captures' value can only ever be mutated behind a mutable borrow,
// including at the C level, so this is safe to send to other threads.
unsafe impl Send for Captures {}

impl Drop for Captures {
    fn drop(&mut self) {
        // SAFETY: We know our 'caps' pointer is valid by construction.
        // Otherwise, we rely on Rust's guarantee that 'Drop' is called at most
        // once to guarantee safety.
        unsafe {
            re2_captures_free(self.caps.as_ptr());
        }
    }
}

impl std::fmt::Debug for Captures {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut map = f.debug_map();
        for group_index in 0..self.group_len() {
            let span = self.get_group(group_index);
            let debug_span: &dyn core::fmt::Debug = match span {
                None => &None::<()>,
                Some(ref span) => span,
            };
            map.entry(&group_index, debug_span);
        }
        map.finish()
    }
}

/// An iterator over all successive non-overlapping matches in a particular
/// haystack. `'r` represents the lifetime of the regex while `'h` represents
/// the lifetime of the haystack.
#[derive(Debug)]
pub struct FindMatches<'r, 'h> {
    re: &'r Regex,
    it: iter::Searcher<'h>,
}

impl<'r, 'h> Iterator for FindMatches<'r, 'h> {
    type Item = Match;

    #[inline]
    fn next(&mut self) -> Option<Match> {
        let FindMatches { re, ref mut it } = *self;
        it.advance(|input| Ok(re.find(input)))
    }
}

// RE2 FFI is below. Since RE2 is written in C++, we hand-rolled our own
// C API shim that is defined in re2.cpp. CRE2 does exist, but I want to
// eliminate any layers between the benchmark harness and the thing being
// measured. Indeed, in CRE2, its 'cre2_match' routine allocates a new
// std::vector on every call to store submatches. We avoid that here by
// defining our own opaque 're2_captures' type. The downside is that accessing
// a submatch requires calling 're2_captures_get'. (If we can make that
// function inlineable, then that downside would be fully mitigated.)
//
// For docs, see the re2.cpp file.

type re2_regexp = c_void;

type re2_captures = c_void;

#[repr(C)]
struct re2_string {
    data: *const c_uchar,
    length: c_int,
}

impl<'a> From<&'a str> for re2_string {
    fn from(s: &'a str) -> re2_string {
        re2_string { data: s.as_ptr(), length: s.len() as c_int }
    }
}

impl<'a> From<&'a [u8]> for re2_string {
    fn from(s: &'a [u8]) -> re2_string {
        re2_string { data: s.as_ptr(), length: s.len() as c_int }
    }
}

extern "C" {
    fn re2_regexp_new(pat: re2_string, opts: Options) -> *mut re2_regexp;
    fn re2_regexp_free(re: *mut re2_regexp);
    fn re2_regexp_is_match(
        re: *mut re2_regexp,
        haystack: re2_string,
        startpos: c_int,
        endpos: c_int,
    ) -> bool;
    fn re2_regexp_find(
        re: *mut re2_regexp,
        haystack: re2_string,
        startpos: c_int,
        endpos: c_int,
        match_start: *mut c_int,
        match_end: *mut c_int,
    ) -> bool;
    fn re2_regexp_captures(
        re: *mut re2_regexp,
        haystack: re2_string,
        startpos: c_int,
        endpos: c_int,
        caps: *mut re2_captures,
    ) -> bool;
    fn re2_regexp_capture_len(re: *mut re2_regexp) -> c_int;
    fn re2_captures_new(nsubmatch: c_int) -> *mut re2_captures;
    fn re2_captures_free(caps: *mut re2_captures);
    fn re2_captures_get(caps: *mut re2_captures, index: c_int) -> re2_string;
}

#[cfg(test)]
mod tests {
    use super::*;

    // A basic sanity check that our 'captures' works as intended. And in
    // particular, that we correctly handle the case of a capturing group that
    // didn't participate in a match.
    #[test]
    fn captures() {
        let re = Regex::new(r"\W+(?:([a-z]+)|([0-9]+))", Options::default())
            .unwrap();
        let mut caps = re.create_captures();
        assert!(re.captures(&Input::new("ABC!@#123"), &mut caps));
        assert_eq!(Some(Span::from(3..9)), caps.get_group(0));
        assert_eq!(None, caps.get_group(1));
        assert_eq!(Some(Span::from(6..9)), caps.get_group(2));
    }
}
