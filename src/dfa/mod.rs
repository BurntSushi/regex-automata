/*!
A module for building and searching with determinstic finite automata (DFAs).

Like other modules in this crate, DFAs support a rich syntax with Unicode
support, has extensive options for configuring the best space vs time trade off
for your use case and provides support for cheap deserialization of automata
for use in `no_std` environments.

# Overview

This section gives a brief overview of the primary types in this module:

* A [`Regex`] provides a way to search for matches of a regular expression
using DFAs. This includes iterating over matches with both the start and end
positions of each match.
* A [`RegexBuilder`] provides a way configure many compilation options for a
regex.
* A [`dense::DFA`] provides low level access to a DFA that uses a dense
representation (uses lots of space, but fast searching). Low level access to
DFAs only provides access to the end of a match location.
* A [`sparse::DFA`] provides the same API as a `dense::DFA`, but uses a sparse
representation (uses less space, but slower matching). Low level access to
DFAs only provides access to the end of a match location.
* An [`Automaton`] trait that defines an interface that all DFAs must
implement.
* Both dense DFAs and sparse DFAs support serialization to raw bytes (e.g.,
[`dense::DFA::to_bytes_little_endian`]) and cheap deserialization (e.g.,
[`dense::DFA::from_bytes`]).

# Example: basic regex searching

This example shows how to compile a regex using the default configuration
and then use it to find matches in a byte string:

```
use regex_automata::{MultiMatch, dfa::Regex};

let re = Regex::new(r"[0-9]{4}-[0-9]{2}-[0-9]{2}").unwrap();
let text = b"2018-12-24 2016-10-08";
let matches: Vec<MultiMatch> = re.find_leftmost_iter(text).collect();
assert_eq!(matches, vec![
    MultiMatch::new(0, 0, 10),
    MultiMatch::new(0, 11, 21),
]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

# Example: use sparse DFAs

By default, compiling a regex will use dense DFAs internally. This uses more
memory, but executes searches more quickly. If you can abide slower searches
(somewhere around 3-5x), then sparse DFAs might make more sense since they can
use significantly less space.

Using sparse DFAs is as easy as using `Regex::new_sparse` instead of
`Regex::new`:

```
use regex_automata::{MultiMatch, dfa::Regex};

let re = Regex::new_sparse(r"[0-9]{4}-[0-9]{2}-[0-9]{2}").unwrap();
let text = b"2018-12-24 2016-10-08";
let matches: Vec<MultiMatch> = re.find_leftmost_iter(text).collect();
assert_eq!(matches, vec![
    MultiMatch::new(0, 0, 10),
    MultiMatch::new(0, 11, 21),
]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

If you already have dense DFAs for some reason, they can be converted to sparse
DFAs and used to build a new `Regex`. For example:

```
use regex_automata::{MultiMatch, dfa::Regex};

let dense_re = Regex::new(r"[0-9]{4}-[0-9]{2}-[0-9]{2}").unwrap();
let sparse_re = Regex::from_dfas(
    dense_re.forward().to_sparse()?,
    dense_re.reverse().to_sparse()?,
);
let text = b"2018-12-24 2016-10-08";
let matches: Vec<MultiMatch> = sparse_re.find_leftmost_iter(text).collect();
assert_eq!(matches, vec![
    MultiMatch::new(0, 0, 10),
    MultiMatch::new(0, 11, 21),
]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

# Example: deserialize a DFA

This shows how to first serialize a DFA into raw bytes, and then deserialize
those raw bytes back into a DFA. While this particular example is a
bit contrived, this same technique can be used in your program to
deserialize a DFA at start up time or by memory mapping a file.

```
use regex_automata::{MultiMatch, dfa::{dense, Regex}};

let re1 = Regex::new(r"[0-9]{4}-[0-9]{2}-[0-9]{2}").unwrap();
// serialize both the forward and reverse DFAs, see note below
let (fwd_bytes, fwd_pad) =
    re1.forward().to_sized::<u16>()?.to_bytes_native_endian();
let (rev_bytes, rev_pad) =
    re1.reverse().to_sized::<u16>()?.to_bytes_native_endian();
// now deserialize both---we need to specify the correct type!
let fwd: dense::DFA<&[u16], &[u8], u16> =
    dense::DFA::from_bytes(&fwd_bytes[fwd_pad..])?.0;
let rev: dense::DFA<&[u16], &[u8], u16> =
    dense::DFA::from_bytes(&rev_bytes[rev_pad..])?.0;
// finally, reconstruct our regex
let re2 = Regex::from_dfas(fwd, rev);

// we can use it like normal
let text = b"2018-12-24 2016-10-08";
let matches: Vec<MultiMatch> = re2.find_leftmost_iter(text).collect();
assert_eq!(matches, vec![
    MultiMatch::new(0, 0, 10),
    MultiMatch::new(0, 11, 21),
]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

There are a few points worth noting here:

* We need to extract the raw DFAs used by the regex and serialize those. You
can build the DFAs manually yourself using [`dense::Builder`], but using
the DFAs from a `Regex` guarantees that the DFAs are built correctly. (In
particular, a `Regex` constructs a reverse DFA for finding the starting
location of matches.)
* We specifically convert the dense DFA to a representation that uses `u16` for
its state identifiers using [`dense::DFA::to_sized`]. While this isn't strictly
necessary, if we skipped this step, then the serialized bytes would use `usize`
for state identifiers, which does not have a fixed size. Using `u16` ensures
that we can deserialize this DFA even on platforms with a smaller pointer size.
If our DFA is too big for `u16` state identifiers, then one can use `u32` or
`u64`.
* To convert the DFA to raw bytes, we use the `to_bytes_native_endian` method.
In practice, you'll want to use either [`dense::DFA::to_bytes_little_endian`]
or [`dense::DFA::to_bytes_big_endian`], depending on which platform you're
deserializing your DFA from. If you intend to deserialize on either platform,
then you'll need to serialize both and deserialize the right one depending on
your target's endianness.
* Safely deserializing a DFA requires verifying the raw bytes, particularly if
they are untrusted, since an invalid DFA could cause logical errors, panics
or even undefined behavior. This verification step requires visiting all of
the transitions in the DFA, which can be costly. If cheaper verification is
desired, then [`dense::DFA::from_bytes_unchecked`] is available that only does
verification that can be performed in constant time. However, one can only use
this routine if the caller can guarantee that the bytes provided encoded a
valid DFA.

The same process can be achieved with sparse DFAs as well:

```
use regex_automata::{MultiMatch, dfa::{sparse, Regex}};

let re1 = Regex::new(r"[0-9]{4}-[0-9]{2}-[0-9]{2}").unwrap();
// serialize both
let fwd_bytes = re1.forward().to_sized::<u16>()?.to_sparse()?.to_bytes_native_endian();
let rev_bytes = re1.reverse().to_sized::<u16>()?.to_sparse()?.to_bytes_native_endian();
// now deserialize both---we need to specify the correct type!
let fwd: sparse::DFA<&[u8], u16> = sparse::DFA::from_bytes(&fwd_bytes)?.0;
let rev: sparse::DFA<&[u8], u16> = sparse::DFA::from_bytes(&rev_bytes)?.0;
// finally, reconstruct our regex
let re2 = Regex::from_dfas(fwd, rev);

// we can use it like normal
let text = b"2018-12-24 2016-10-08";
let matches: Vec<MultiMatch> = re2.find_leftmost_iter(text).collect();
assert_eq!(matches, vec![
    MultiMatch::new(0, 0, 10),
    MultiMatch::new(0, 11, 21),
]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Note that unlike dense DFAs, sparse DFAs have no alignment requirements.
Conversely, dense DFAs must be be aligned to the same alignment as their
state identifier representation.

# Support for `no_std`

This crate comes with a `std` feature that is enabled by default. When the
`std` feature is enabled, the API of this module will include the facilities
necessary for compiling, serializing, deserializing and searching with DFAs.
When the `std` feature is disabled, the API of this module will shrink such
that it only includes the facilities necessary for deserializing and searching
with DFAs.

The intended workflow for `no_std` environments is thus as follows:

* Write a program with the `std` feature that compiles and serializes a regular
expression. Serialization should only happen after first converting the DFAs to
use a fixed size state identifier instead of the default `usize`. You may also
need to serialize both little and big endian versions of each DFA. (So that's 4
DFAs in total for each regex.)
* In your `no_std` environment, follow the examples above for deserializing
your previously serialized DFAs into regexes. You can then search with them as
you would any regex.

Deserialization can happen anywhere. For example, with bytes embedded into a
binary or with a file memory mapped at runtime.

TODO: Include link to `regex-cli` here pointing out how to generate Rust code
for deserializing DFAs.

# Syntax

This module supports the same syntax as the `regex` crate, since they share the
same parser. You can find an exhaustive list of supported syntax in the
[documentation for the `regex` crate](https://docs.rs/regex/1.4/regex/#syntax).

There are two things that are not supported by the DFAs in this module:

* Capturing groups. The DFAs (and [`Regex`]es built on top of them) can only
find the offsets of an entire match, but cannot resolve the offsets of each
capturing group. This is because DFAs do not have the expressive power to
provide this.
* Unicode word boundaries. These present particularly difficult challenges for
DFA construction and would result in an explosion in the number of states.
One can enable [`dense::Config::unicode_word_boundary`] though, which provides
heuristic support for Unicode word boundaries that only works on ASCII text.
Otherwise, one can use `(?-u:\b)` for an ASCII word boundary, which will work
on any input.

There are no plans to lift either of these limitations.

# Differences with general purpose regexes

The main goal of the [`regex`](https://docs.rs/regex) crate is to serve as a
general purpose regular expression engine. It aims to automatically balance low
compile times, fast search times and low memory usage, while also providing
a convenient API for users. In contrast, this module provides a lower level
regular expression interface based exclusively on DFAs that is a bit less
convenient while providing more explicit control over memory usage and search
times.

Here are some specific negative differences:

* **Compilation can take an exponential amount of time and space** in the size
of the regex pattern. While most patterns do not exhibit worst case exponential
time, such patterns do exist. For example, `[01]*1[01]{N}` will build a DFA
with approximately `2^(N+1)` states. For this reason, untrusted patterns should
not be compiled with this module. (In the future, the API may expose an option
to return an error if the DFA gets too big.)
* This module does not support sub-match extraction via capturing groups, which
can be achieved with the regex crate's "captures" API.
* While the regex crate doesn't necessarily sport fast compilation times,
the regexes in this module are almost universally slow to compile, especially
when they contain large Unicode character classes. For example, on my system,
compiling `\w{50}` with byte classes enabled takes about 1 second and almost
15MB of memory! (Compiling a sparse regex takes about the same time but only
uses about 1.2MB of memory.) Conversly, compiling the same regex without
Unicode support, e.g., `(?-u)\w{50}`, takes under 1 millisecond and about 15KB
of memory. For this reason, you should only use Unicode character classes if
you absolutely need them! (They are enabled by default though.)
* This module does not support Unicode word boundaries. ASCII word bondaries
may be used though by disabling Unicode or selectively doing so in the syntax,
e.g., `(?-u:\b)`.
* As a lower level API, this module does not do literal optimizations
automatically. Although it does provide hooks in its API to make use of the
[`Prefilter`](crate::prefilter::Prefilter) trait. Missing literal optimizations
means that searches may run much slower than what you're accustomed to,
although, it does provide more predictable and consistent performance.
* There is no `&str` API like in the regex crate. In this module, all APIs
operate on `&[u8]`. By default, match indices are guaranteed to fall on UTF-8
boundaries, unless
[`SyntaxConfig::allow_invalid_utf8`](crate::SyntaxConfig::allow_invalid_utf8)
is enabled.

With some of the downsides out of the way, here are some positive differences:

* Both dense and sparse DFAs can be serialized to raw bytes, and then cheaply
deserialized. Deserialization can be done in constant time with the unchecked
APIs, since searching can be performed directly on the raw serialized bytes of
a DFA.
* This module was specifically designed so that the searching phase of a
DFA has minimal runtime requirements, and can therefore be used in `no_std`
environments. While `no_std` environments cannot compile regexes, they can
deserialize pre-compiled regexes.
* Since this module builds DFAs ahead of time, it will generally out-perform
the `regex` crate on equivalent tasks. The performance difference is likely
not large. However, because of a complex set of optimizations in the regex
crate (like literal optimizations), an accurate performance comparison may be
difficult to do.
* Sparse DFAs provide a way to build a DFA ahead of time that sacrifices search
performance a small amount, but uses much less storage space. Potentially even
less than what the regex crate uses.
* This module exposes DFAs directly, such as [`dense::DFA`] and
[`sparse::DFA`], which enables one to do less work in some cases. For example,
if you only need the end of a match and not the start of a match, then you can
use a DFA directly without building a `Regex`, which always requires a second
DFA to find the start of a match.
* This module provides more control over memory usage. Aside from choosing
between dense and sparse DFAs, one can also choose a smaller state identifier
representation to use less space. Also, one can enable DFA minimization
via [`dense::Config::minimize`], but it can increase compilation times
dramatically.
*/

pub use crate::dfa::automaton::{Automaton, HalfMatch, OverlappingState};
#[cfg(feature = "std")]
pub use crate::dfa::error::{Error, ErrorKind};
#[cfg(feature = "std")]
pub use crate::dfa::regex::RegexBuilder;
pub use crate::dfa::regex::{
    FindEarliestMatches, FindLeftmostMatches, FindOverlappingMatches, Regex,
    TryFindEarliestMatches, TryFindLeftmostMatches, TryFindOverlappingMatches,
};

mod accel;
mod automaton;
#[path = "dense.rs"]
mod dense_imp;
#[cfg(feature = "std")]
mod determinize;
#[cfg(feature = "std")]
pub(crate) mod error;
#[cfg(feature = "std")]
mod minimize;
mod regex;
mod search;
#[path = "sparse.rs"]
mod sparse_imp;
mod special;
#[cfg(feature = "transducer")]
mod transducer;

/// Types and routines specific to dense DFAs.
///
/// This module is the home of [`dense::DFA`].
///
/// This module also contains a [`dense::Builder`] and a [`dense::Config`] for
/// configuring and building a dense DFA.
pub mod dense {
    pub use crate::dfa::dense_imp::*;
}

/// Types and routines specific to sparse DFAs.
///
/// This module is the home of [`sparse::DFA`].
///
/// Unlike the [`dense`] module, this module does not contain a builder
/// or configuration specific for sparse DFAs. Instead, the intended way
/// to build a sparse DFA is either by using a default configuration with
/// its constructor [`sparse::DFA::new`], or by first configuring the
/// construction of a dense DFA with [`dense::Builder`] and then calling
/// [`dense::DFA::to_sparse`]. For example, this configures a sparse DFA to do
/// an overlapping search:
///
/// ```
/// use regex_automata::{
///     dfa::{Automaton, HalfMatch, OverlappingState, dense},
///     MatchKind,
/// };
///
/// let dense_re = dense::Builder::new()
///     .configure(dense::Config::new().match_kind(MatchKind::All))
///     .build(r"Samwise|Sam")?;
/// let sparse_re = dense_re.to_sparse()?;
///
/// // Setup our haystack and initial start state.
/// let haystack = b"Samwise";
/// let mut state = OverlappingState::start();
///
/// // First, 'Sam' will match.
/// let end1 = sparse_re.find_overlapping_fwd_at(
///     None, None, haystack, 0, haystack.len(), &mut state,
/// )?;
/// assert_eq!(end1, Some(HalfMatch::new(0, 3)));
///
/// // And now 'Samwise' will match.
/// let end2 = sparse_re.find_overlapping_fwd_at(
///     None, None, haystack, 3, haystack.len(), &mut state,
/// )?;
/// assert_eq!(end2, Some(HalfMatch::new(0, 7)));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub mod sparse {
    pub use crate::dfa::sparse_imp::*;
}
