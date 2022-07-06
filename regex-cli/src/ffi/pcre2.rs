#![allow(non_camel_case_types)]
// TODO: remove this
#![allow(dead_code)]

use std::ptr::{self, NonNull};

use libc::{c_int, c_void};

use automata::Input;

// The PCRE2 docs say that 32KB is the default, and that 1MB should be big
// enough for anything. But let's crank the max to 10MB. We can go bigger if
// necessary, but we should stay somewhere around what is "reasonable" in a
// "real" application. (That sounds pretty weasely.) The max is also what
// ripgrep happens to use and it tends to work well as far as I know, so I
// suppose that's decent justification.
const MIN_JIT_STACK_SIZE: usize = 32 * (1 << 10);
const MAX_JIT_STACK_SIZE: usize = 10 * (1 << 20);

/// A low level representation of a compiled PCRE2 code object.
pub struct Regex {
    code: NonNull<pcre2_code_8>,
    // The pattern string.
    pattern: String,
    // Whether we've successfully JIT compiled this code object.
    compiled_jit: bool,
}

// SAFETY: Compiled PCRE2 code objects are immutable once built and explicitly
// safe to use from multiple threads simultaneously.
//
// One hitch here is that JIT compiling can write into a PCRE2 code object, but
// we only ever JIT compile immediately after first building the code object
// and before making it available to the caller.
unsafe impl Send for Regex {}
unsafe impl Sync for Regex {}

impl std::fmt::Debug for Regex {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.pattern)
    }
}

impl Drop for Regex {
    fn drop(&mut self) {
        // SAFETY: By construction, both the compile context and the code
        // objects are valid.
        unsafe {
            pcre2_code_free_8(self.code.as_ptr());
        }
    }
}

impl Regex {
    /// Compile the given pattern with the given options. If there was a
    /// problem compiling the pattern, then return an error.
    pub fn new(pattern: &str, opts: Options) -> anyhow::Result<Regex> {
        let mut pcre2_opts = 0;
        // Since we support setting an end position past which we shouldn't
        // search, we need to pass this option at compile time.
        pcre2_opts |= PCRE2_USE_OFFSET_LIMIT;
        if opts.ucp {
            pcre2_opts |= PCRE2_UCP;
            pcre2_opts |= PCRE2_MATCH_INVALID_UTF;
        }
        if opts.caseless {
            pcre2_opts |= PCRE2_CASELESS;
        }

        let mut error_code = 0;
        let mut re = match NonNull::new(unsafe {
            pcre2_compile_8(
                pattern.as_ptr(),
                pattern.len(),
                pcre2_opts,
                &mut error_code,
                &mut 0, // don't care about this for now
                ptr::null_mut(),
            )
        }) {
            None => return Err(Error { error_code }.into()),
            Some(code) => Regex {
                code,
                pattern: pattern.to_string(),
                compiled_jit: false,
            },
        };
        if opts.jit {
            anyhow::ensure!(
                is_jit_available(),
                "asked for JIT, but it's unavailable in your build of PCRE2",
            );
            re.jit_compile()?;
        }
        Ok(re)
    }

    /// JIT compile this code object.
    ///
    /// If there was a problem performing JIT compilation, then this returns
    /// an error.
    fn jit_compile(&mut self) -> anyhow::Result<()> {
        let error_code = unsafe {
            pcre2_jit_compile_8(self.code.as_ptr(), PCRE2_JIT_COMPLETE)
        };
        if error_code == 0 {
            self.compiled_jit = true;
            Ok(())
        } else {
            Err(Error { error_code }.into())
        }
    }

    pub fn create_match_data(&self) -> MatchData {
        MatchData::new(self)
    }

    pub fn find(
        &self,
        input: &Input<'_, '_>,
        match_data: &mut MatchData,
    ) -> anyhow::Result<bool> {
        let matched = match_data.find(self, input)?;
        Ok(matched)
    }
}

/// Options that can be passed to Regex::new to configure a subset
/// of PCRE2 knobs.
#[derive(Clone, Debug)]
pub struct Options {
    /// When enabled, PCRE2's JIT will attempt to be used. If this is enabled
    /// and PCRE2's JIT isn't available or the JIT compilation fails, then an
    /// error will be returned by Regex::new.
    pub jit: bool,
    /// When enabled, PCRE2's "UCP" option is enabled. When this option is
    /// enabled, we also set PCRE2_MATCH_INVALID_UTF which in turn enables the
    /// UTF option and permits safely matching subjects that may not be valid
    /// UTF-8. (Any invalid UTF-8 will prevent a match.)
    pub ucp: bool,
    /// When enabled, PCRE2's "caseless" option is enabled when compiling the
    /// regex.
    pub caseless: bool,
}

impl Default for Options {
    fn default() -> Options {
        Options { jit: true, ucp: true, caseless: false }
    }
}

/// A low level representation of a match data block.
///
/// Technically, a single match data block can be used with multiple regexes
/// (not simultaneously), but in practice, we just create a single match data
/// block for each regex.
pub struct MatchData {
    match_context: NonNull<pcre2_match_context_8>,
    match_data: NonNull<pcre2_match_data_8>,
    jit_stack: Option<NonNull<pcre2_jit_stack_8>>,
    ovector_ptr: NonNull<usize>,
    ovector_count: u32,
}

// SAFETY: Match data blocks can be freely sent from one thread to another,
// but they do not support multiple threads using them simultaneously. We still
// implement Sync however, since we require mutable access to use the match
// data block for executing a search, which statically prevents simultaneous
// reading/writing. It is legal to read match data blocks from multiple threads
// simultaneously.
unsafe impl Send for MatchData {}
unsafe impl Sync for MatchData {}

impl Drop for MatchData {
    fn drop(&mut self) {
        // SAFETY: All of our pointers are valid by construction of MatchData.
        unsafe {
            if let Some(stack) = self.jit_stack {
                pcre2_jit_stack_free_8(stack.as_ptr());
            }
            pcre2_match_data_free_8(self.match_data.as_ptr());
            pcre2_match_context_free_8(self.match_context.as_ptr());
            // N.B. The ovector pointer points into the match data block, so it
            // gets freed as part of freeing the match data.
        }
    }
}

impl MatchData {
    /// Create a new match data block from a compiled PCRE2 code object.
    ///
    /// This panics if memory could not be allocated for the block.
    fn new(re: &Regex) -> MatchData {
        // SAFETY: Passing null is OK and causes PCRE2 to use default memory
        // allocation primitives.
        let match_context = NonNull::new(unsafe {
            pcre2_match_context_create_8(ptr::null_mut())
        })
        .expect("failed to allocate match context");

        // SAFETY: 'code' is valid by construction and passing null is OK as
        // a general context, like above.
        let match_data = NonNull::new(unsafe {
            pcre2_match_data_create_from_pattern_8(
                re.code.as_ptr(),
                ptr::null_mut(),
            )
        })
        .expect("failed to allocate match data block");

        let jit_stack = if !re.compiled_jit {
            None
        } else {
            // SAFETY: We pass our min/max, and null for the general context
            // as is allowed. (Same as above.)
            let stack = NonNull::new(unsafe {
                pcre2_jit_stack_create_8(
                    MIN_JIT_STACK_SIZE,
                    MAX_JIT_STACK_SIZE,
                    ptr::null_mut(),
                )
            })
            .expect("failed to allocate JIT stack");

            // SAFETY: Our match context is valid by construction (we panic
            // if it wasn't). We don't give a callback (allowed by PCRE2
            // docs) and give a valid stack, also valid by construction.
            // PCRE2 docs say that a null callback with non-null callback
            // data requires the callback data to be a valid JIT stack,
            // which it is.
            unsafe {
                pcre2_jit_stack_assign_8(
                    match_context.as_ptr(),
                    None,
                    stack.as_ptr() as *mut c_void,
                )
            };
            Some(stack)
        };

        // SAFETY: match_data is valid by construction.
        let ovector_ptr = NonNull::new(unsafe {
            pcre2_get_ovector_pointer_8(match_data.as_ptr())
        })
        .expect("got NULL ovector pointer");
        // SAFETY: match_data is valid by construction.
        let ovector_count =
            unsafe { pcre2_get_ovector_count_8(match_data.as_ptr()) };
        MatchData {
            match_context,
            match_data,
            jit_stack,
            ovector_ptr,
            ovector_count,
        }
    }

    /// Execute PCRE2's primary match routine on the given subject string
    /// starting at the given offset. The provided options are passed to PCRE2
    /// as is.
    ///
    /// This returns false if no match occurred.
    ///
    /// Match offsets can be extracted via `ovector`.
    fn find(
        &mut self,
        re: &Regex,
        input: &Input<'_, '_>,
    ) -> Result<bool, Error> {
        // The regex-automata handle this case correctly, but I'm not sure if
        // PCRE2 does. The regex-automata iterators rely on the regex engine
        // handling this, so we do it here before jumping into PCRE2.
        if input.start() > input.end() {
            return Ok(false);
        }
        let mut haystack = input.haystack();
        // When the haystack is empty, we use an empty slice
        // with a known valid pointer. Otherwise, slices derived
        // from, e.g., an empty `Vec<u8>` may not have a valid
        // pointer, since creating an empty `Vec` is guaranteed
        // to not allocate.
        const EMPTY: &[u8] = &[];
        if haystack.is_empty() {
            haystack = EMPTY;
        }

        // SAFETY: Our match context is valid and 'input.end()' is treated as
        // a limit, so it shouldn't matter if it's a valid index into the
        // haystack.
        unsafe {
            // This always returns 0.
            pcre2_set_offset_limit_8(self.match_context.as_ptr(), input.end());
        }
        // SAFETY: Our 'code', 'haystack', 'match_data' and 'match_context'
        // pointers are all valid, by construction. We don't permit setting any
        // options at match time (we set them all at regex compile time).
        let rc = unsafe {
            pcre2_match_8(
                re.code.as_ptr(),
                haystack.as_ptr(),
                haystack.len(),
                input.start(),
                0,
                self.match_data.as_ptr(),
                self.match_context.as_ptr(),
            )
        };
        if rc == PCRE2_ERROR_NOMATCH {
            Ok(false)
        } else if rc > 0 {
            // We don't care that 'rc' is the highest numbered capturing
            // group that matched, so we throw it away and just return true.
            Ok(true)
        } else {
            // We always create match data with
            // pcre2_match_data_create_from_pattern, so the ovector should
            // always be big enough.
            assert!(rc != 0, "ovector should never be too small");
            Err(Error { error_code: rc })
        }
    }

    /// Return the ovector corresponding to this match data.
    ///
    /// The ovector represents match offsets as pairs. This always returns
    /// N + 1 pairs (so 2*N + 1 offsets), where N is the number of capturing
    /// groups in the original regex.
    pub fn ovector(&self) -> &[usize] {
        // SAFETY: Both our ovector pointer and count are derived directly from
        // the creation of a valid match data block. One interesting question
        // here is whether the contents of the ovector are always initialized.
        // The PCRE2 documentation suggests that they are (so does testing),
        // but this isn't actually 100% clear!
        unsafe {
            std::slice::from_raw_parts(
                self.ovector_ptr.as_ptr(),
                self.ovector_count as usize * 2,
            )
        }
    }
}

/// An error reported by PCRE2.
#[derive(Clone, Debug)]
pub struct Error {
    error_code: c_int,
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // Hopefully 1KB is enough? If it isn't, the worst thing that happens
        // is that we get another error. (In which case, we panic.)
        let mut buf = [0u8; 1 << 10];
        // SAFETY: Our buffer and buffer length are initialized and correct,
        // and the PCRE2 docs imply that any integer code is a valid input.
        // If the code is unrecognized, then we get PCRE2_ERROR_BADDATA.
        let rc = unsafe {
            pcre2_get_error_message_8(
                self.error_code,
                buf.as_mut_ptr(),
                buf.len(),
            )
        };
        // Errors are only ever constructed from codes reported by PCRE2, so
        // our code should always be valid.
        assert!(rc != PCRE2_ERROR_BADDATA, "used an invalid error code");
        // Hopefully 1KB is enough.
        assert!(rc != PCRE2_ERROR_NOMEMORY, "buffer size too small");
        // Sanity check that we do indeed have a non-negative result. 0 is OK.
        assert!(rc >= 0, "expected non-negative but got {}", rc);
        let msg = String::from_utf8(buf[..rc as usize].to_vec())
            .expect("valid UTF-8");
        write!(f, "{}", msg)
    }
}

/// Returns true if and only if PCRE2 believes that JIT is available.
///
/// We use this routine to return an error if the caller requested the JIT
/// and it isn't avaialble.
fn is_jit_available() -> bool {
    let mut rc: u32 = 0;
    let error_code = unsafe {
        pcre2_config_8(PCRE2_CONFIG_JIT, &mut rc as *mut _ as *mut c_void)
    };
    if error_code < 0 {
        // If PCRE2_CONFIG_JIT is a bad option, then there's a bug somewhere.
        panic!("BUG: {}", Error { error_code });
    }
    rc == 1
}

// Below are our FFI declarations. We just hand-write what we need instead of
// trying to generate bindings for everything.

type pcre2_code_8 = c_void;
type pcre2_compile_context_8 = c_void;
type pcre2_general_context_8 = c_void;
type pcre2_jit_stack_8 = c_void;
type pcre2_jit_callback_8 = Option<
    unsafe extern "C" fn(callback_data: *mut c_void) -> *mut pcre2_jit_stack_8,
>;
type pcre2_match_context_8 = c_void;
type pcre2_match_data_8 = c_void;

type PCRE2_UCHAR8 = u8;
type PCRE2_SPTR8 = *const PCRE2_UCHAR8;

const PCRE2_CASELESS: u32 = 8;
const PCRE2_CONFIG_JIT: u32 = 1;
const PCRE2_ERROR_BADDATA: i32 = -29;
const PCRE2_ERROR_NOMEMORY: i32 = -48;
const PCRE2_ERROR_NOMATCH: i32 = -1;
const PCRE2_JIT_COMPLETE: u32 = 1;
const PCRE2_MATCH_INVALID_UTF: u32 = 67108864;
const PCRE2_USE_OFFSET_LIMIT: u32 = 8388608;
const PCRE2_UCP: u32 = 131072;
const PCRE2_UNSET: usize = std::usize::MAX;

extern "C" {
    fn pcre2_code_free_8(code: *mut pcre2_code_8);
    fn pcre2_compile_8(
        pattern: PCRE2_SPTR8,
        pattern_len: usize,
        options: u32,
        error_code: *mut c_int,
        error_offset: *mut usize,
        ctx: *mut pcre2_compile_context_8,
    ) -> *mut pcre2_code_8;
    fn pcre2_config_8(option: u32, code: *mut c_void) -> c_int;
    fn pcre2_get_error_message_8(
        error_code: c_int,
        buf: *mut PCRE2_UCHAR8,
        buflen: usize,
    ) -> c_int;
    fn pcre2_get_ovector_count_8(data: *mut pcre2_match_data_8) -> u32;
    fn pcre2_get_ovector_pointer_8(
        data: *mut pcre2_match_data_8,
    ) -> *mut usize;
    fn pcre2_jit_compile_8(code: *mut pcre2_code_8, options: u32) -> c_int;
    fn pcre2_jit_stack_free_8(stack: *mut pcre2_jit_stack_8);
    fn pcre2_jit_stack_create_8(
        start_size: usize,
        max_size: usize,
        ctx: *mut pcre2_general_context_8,
    ) -> *mut pcre2_jit_stack_8;
    fn pcre2_jit_stack_assign_8(
        ctx: *mut pcre2_match_context_8,
        callback: pcre2_jit_callback_8,
        callback_data: *mut c_void,
    );
    fn pcre2_match_8(
        code: *const pcre2_code_8,
        subject: PCRE2_SPTR8,
        subject_len: usize,
        start: usize,
        options: u32,
        data: *mut pcre2_match_data_8,
        ctx: *mut pcre2_match_context_8,
    ) -> c_int;
    fn pcre2_match_context_create_8(
        ctx: *mut pcre2_general_context_8,
    ) -> *mut pcre2_match_context_8;
    fn pcre2_match_context_free_8(ctx: *mut pcre2_match_context_8);
    fn pcre2_match_data_create_from_pattern_8(
        code: *const pcre2_code_8,
        ctx: *mut pcre2_general_context_8,
    ) -> *mut pcre2_match_data_8;
    fn pcre2_match_data_free_8(data: *mut pcre2_match_data_8);
    fn pcre2_set_offset_limit_8(
        ctx: *mut pcre2_match_context_8,
        offset: usize,
    ) -> c_int;
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
        let mut match_data = re.create_match_data();
        dbg!(match_data.ovector());
        assert!(re.find(&Input::new("ABC!@#123"), &mut match_data).unwrap());
        dbg!(match_data.ovector());
        dbg!(PCRE2_UNSET);
        // assert_eq!(Some(Span::from(3..9)), caps.get_group(0));
        // assert_eq!(None, caps.get_group(1));
        // assert_eq!(Some(Span::from(6..9)), caps.get_group(2));
    }
}
