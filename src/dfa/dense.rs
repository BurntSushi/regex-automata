use core::cmp;
use core::convert::TryInto;
#[cfg(feature = "std")]
use core::fmt;
#[cfg(feature = "std")]
use core::iter;
use core::marker::PhantomData;
use core::mem;
use core::slice;
#[cfg(feature = "std")]
use std::collections::{BTreeMap, BTreeSet};

#[cfg(feature = "std")]
use regex_syntax::ParserBuilder;

use crate::bytes::{self, DeserializeError, Endian, SerializeError};
use crate::classes::{Byte, ByteClasses, ByteSet};
use crate::dfa::accel::{Accel, Accels};
use crate::dfa::automaton::{fmt_state_indicator, Automaton, Start};
#[cfg(feature = "std")]
use crate::dfa::determinize::Determinizer;
#[cfg(feature = "std")]
use crate::dfa::error::Error;
#[cfg(feature = "std")]
use crate::dfa::minimize::Minimizer;
#[cfg(feature = "std")]
use crate::dfa::sparse;
use crate::dfa::special::Special;
#[cfg(feature = "std")]
use crate::nfa::thompson;
use crate::state_id::{dead_id, StateID};
use crate::{MatchKind, PatternID};

const LABEL: &str = "rust-regex-automata-dfa-dense";
const VERSION: u64 = 2;

/// The configuration used for compiling a dense DFA.
#[derive(Clone, Copy, Debug, Default)]
pub struct Config {
    anchored: Option<bool>,
    accelerate: Option<bool>,
    minimize: Option<bool>,
    match_kind: Option<MatchKind>,
    starts_for_each_pattern: Option<bool>,
    byte_classes: Option<bool>,
    unicode_word_boundary: Option<bool>,
    quit: Option<ByteSet>,
}

impl Config {
    /// Return a new default dense DFA compiler configuration.
    pub fn new() -> Config {
        Config::default()
    }

    /// Set whether matching must be anchored at the beginning of the input.
    ///
    /// When enabled, a match must begin at the start of a search. When
    /// disabled, the DFA will act as if the pattern started with a `(?s:.)*?`,
    /// which enables a match to appear anywhere.
    ///
    /// By default this is disabled.
    ///
    /// **WARNING:** this is subtly different than using a `^` at the start of
    /// your regex. A `^` forces a regex to match exclusively at the start of
    /// input, regardless of where you start your search. In contrast, enabling
    /// this option will allow your regex to match anywhere in your input, but
    /// the match must start at the beginning of a search.
    ///
    /// For example, consider the haystack `aba` and the following searches:
    ///
    /// 1. The regex `^a` is compiled with `anchored=false` and searches
    ///    `aba` starting at position `2`. Since `^` requires the match to
    ///    start at the beginning of the input and `2 > 0`, no match is found.
    /// 2. The regex `a` is compiled with `anchored=true` and searches `aba`
    ///    starting at position `2`. This reports a match at `[2, 3]` since
    ///    the match starts where the search started. Since there is no `^`,
    ///    there is no requirement for the match to start at the beginning of
    ///    the input.
    /// 3. The regex `a` is compiled with `anchored=true` and searches `aba`
    ///    starting at position `1`. Since `b` corresponds to position `1` and
    ///    since the regex is anchored, it finds no match.
    /// 4. The regex `a` is compiled with `anchored=false` and searches `aba`
    ///    startting at position `1`. Since the regex is neither anchored nor
    ///    starts with `^`, the regex is compiled with an implicit `(?s:.)*?`
    ///    prefix that permits it to match anywhere. Thus, it reports a match
    ///    at `[2, 3]`.
    pub fn anchored(mut self, yes: bool) -> Config {
        self.anchored = Some(yes);
        self
    }

    pub fn accelerate(mut self, yes: bool) -> Config {
        self.accelerate = Some(yes);
        self
    }

    /// Minimize the DFA.
    ///
    /// When enabled, the DFA built will be minimized such that it is as small
    /// as possible.
    ///
    /// Whether one enables minimization or not depends on the types of costs
    /// you're willing to pay and how much you care about its benefits. In
    /// particular, minimization has worst case `O(n*k*logn)` time and `O(k*n)`
    /// space, where `n` is the number of DFA states and `k` is the alphabet
    /// size. In practice, minimization can be quite costly in terms of both
    /// space and time, so it should only be done if you're willing to wait
    /// longer to produce a DFA. In general, you might want a minimal DFA in
    /// the following circumstances:
    ///
    /// 1. You would like to optimize for the size of the automaton. This can
    ///    manifest in one of two ways. Firstly, if you're converting the
    ///    DFA into Rust code (or a table embedded in the code), then a minimal
    ///    DFA will translate into a corresponding reduction in code  size, and
    ///    thus, also the final compiled binary size. Secondly, if you are
    ///    building many DFAs and putting them on the heap, you'll be able to
    ///    fit more if they are smaller. Note though that building a minimal
    ///    DFA itself requires additional space; you only realize the space
    ///    savings once the minimal DFA is constructed (at which point, the
    ///    space used for minimization is freed).
    /// 2. You've observed that a smaller DFA results in faster match
    ///    performance. Naively, this isn't guaranteed since there is no
    ///    inherent difference between matching with a bigger-than-minimal
    ///    DFA and a minimal DFA. However, a smaller DFA may make use of your
    ///    CPU's cache more efficiently.
    /// 3. You are trying to establish an equivalence between regular
    ///    languages. The standard method for this is to build a minimal DFA
    ///    for each language and then compare them. If the DFAs are equivalent
    ///    (up to state renaming), then the languages are equivalent.
    ///
    /// Typically, minimization only makes sense as an offline process. That
    /// is, one might minimize a DFA before serializing it to persistent
    /// storage.
    ///
    /// This option is disabled by default.
    pub fn minimize(mut self, yes: bool) -> Config {
        self.minimize = Some(yes);
        self
    }

    /// Find the longest possible match.
    ///
    /// This is distinct from the default leftmost-first match semantics in
    /// that it treats all NFA states as having equivalent priority. In other
    /// words, the longest possible match is always found and it is not
    /// possible to implement non-greedy match semantics when this is set. That
    /// is, `a+` and `a+?` are equivalent when this is enabled.
    ///
    /// In particular, a practical issue with this option at the moment is that
    /// it prevents unanchored searches from working correctly, since
    /// unanchored searches are implemented by prepending an non-greedy `.*?`
    /// to the beginning of the pattern. As stated above, non-greedy match
    /// semantics aren't supported. Therefore, if this option is enabled and
    /// an unanchored search is requested, then building a DFA will return an
    /// error.
    ///
    /// This option is principally useful when building a reverse DFA for
    /// finding the start of a match. If you are building a regex with
    /// [`RegexBuilder`](struct.RegexBuilder.html), then this is handled for
    /// you automatically. The reason why this is necessary for start of match
    /// handling is because we want to find the earliest possible starting
    /// position of a match to satisfy leftmost-first match semantics. When
    /// matching in reverse, this means finding the longest possible match,
    /// hence, this option.
    ///
    /// By default this is disabled.
    pub fn match_kind(mut self, kind: MatchKind) -> Config {
        self.match_kind = Some(kind);
        self
    }

    /// Whether to compile a separate start state for each pattern in the
    /// automaton.
    ///
    /// When enabled, a separate anchored start state is added for each pattern
    /// in the DFA. When this start state is used, then the DFA will only
    /// search for matches for the pattern, even if there are other patterns in
    /// the DFA.
    ///
    /// The main downside of this option is that it can potentially increase
    /// the size of the DFA and/or increase the time it takes to build the DFA.
    ///
    /// There are a few reasons one might want to enable this (it's disabled
    /// by default):
    ///
    /// 1. When looking for the start of an overlapping match (using a reverse
    /// DFA), doing it correctly requires starting the reverse search using the
    /// starting state of the pattern that matched in the forward direction.
    /// Indeed, when building a [`Regex`](../struct.Regex.html), it will
    /// automatically enable this option when building the reverse DFA
    /// internally.
    /// 2. When you want to use a DFA with multiple patterns to both search
    /// for matches of any pattern or to search for anchored matches of one
    /// particular pattern while using the same DFA. (Otherwise, you would need
    /// to compile a new DFA for each pattern.)
    /// 3. Since the start states added for each pattern are anchored, if you
    /// compile an unanchored DFA with one pattern while also enabling this
    /// option, then you can use the same DFA to perform anchored or unanchored
    /// searches. The latter you get with the standard search APIs. The former
    /// you get from the various `_at` search methods that allow you specify a
    /// pattern ID to search for.
    ///
    /// By default this is disabled.
    ///
    /// # Example
    ///
    /// This example shows how to use this option to permit the same DFA to
    /// run both anchored and unanchored searches for a single pattern.
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, dense};
    ///
    /// let dfa = dense::Builder::new()
    ///     .configure(dense::Config::new().starts_for_each_pattern(true))
    ///     .build(r"foo[0-9]+")?;
    /// let haystack = b"quux foo123";
    ///
    /// // Here's a normal unanchored search. Notice that we use 'None' for the
    /// // pattern ID. Since the DFA was built as an unanchored machine, it
    /// // use its default unanchored starting state.
    /// let expected = HalfMatch::new(0, 11);
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd_at(
    ///     None, None, haystack, 0, haystack.len(),
    /// )?);
    /// // But now if we explicitly specify the pattern to search ('0' being
    /// // the only pattern in the DFA), then it will use the starting state
    /// // for that specific pattern which is always anchored. Since the
    /// // pattern doesn't have a match at the beginning of the haystack, we
    /// // find nothing.
    /// assert_eq!(None, dfa.find_leftmost_fwd_at(
    ///     None, Some(0), haystack, 0, haystack.len(),
    /// )?);
    /// // And finally, an anchored search is not the same as putting a '^' at
    /// // beginning of the pattern. An anchored search can only match at the
    /// // beginning of the *search*, which we can change:
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd_at(
    ///     None, Some(0), haystack, 5, haystack.len(),
    /// )?);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn starts_for_each_pattern(mut self, yes: bool) -> Config {
        self.starts_for_each_pattern = Some(yes);
        self
    }

    /// Whether to attempt to shrink the size of the DFA's alphabet or not.
    ///
    /// This option is enabled by default and should never by disabled unless
    /// one is debugging a generated DFA.
    ///
    /// When enabled, each DFA will use a map from all possible bytes to their
    /// corresponding equivalence class. Each equivalence class represents a
    /// set of bytes that does not discriminate between a match and a non-match
    /// in the DFA. For example, the pattern `[ab]+` has at least two
    /// equivalence classes: a set containing `a` and `b` and a set containing
    /// every byte except for `a` and `b`. `a` and `b` are in the same
    /// equivalence classes because they never discriminate between a match
    /// and a non-match.
    ///
    /// The advantage of this map is that the size of the transition table can
    /// be reduced drastically from `#states * 256 * sizeof(id)` to `#states *
    /// k * sizeof(id)` where `k` is the number of equivalence classes. As a
    /// result, total space usage can decrease substantially. Moreover, since a
    /// smaller alphabet is used, DFA compilation becomes faster as well.
    ///
    /// **WARNING:** This is only useful for debugging DFAs. Disabling this
    /// does not yield any speed advantages. Namely, even when this is
    /// disabled, a byte class map is still used while searching. The only
    /// difference is that every byte will be forced into its own distinct
    /// equivalence class. This is useful for debugging the actual generated
    /// transitions because it lets one see the transitions defined on actual
    /// bytes instead of the equivalence classes.
    pub fn byte_classes(mut self, yes: bool) -> Config {
        self.byte_classes = Some(yes);
        self
    }

    /// Heuristically enable Unicode word boundaries.
    ///
    /// When set, this will attempt to implement Unicode word boundaries as if
    /// they were ASCII word boundaries. This only works when the search input
    /// is ASCII only. If a non-ASCII byte is observed while searching, then a
    /// [`MatchError::Quit`](../../enum.MatchError.html#variant.Quit)
    /// error is returned.
    ///
    /// Therefore, when enabling this option, callers _must_ be prepared
    /// to handle a `MatchError` error during search. When using a
    /// [`Regex`](../struct.Regex.html), this corresponds to using the `try_`
    /// suite of methods. Alternatively, if callers can guarantee that their
    /// input is ASCII only, then a `MatchError::Quit` error will never be
    /// returned while searching.
    ///
    /// If the regex pattern provided has no Unicode word boundary in it, then
    /// this option has no effect. (That is, quitting on a non-ASCII byte only
    /// occurs when this option is enabled _and_ a Unicode word boundary is
    /// present in the pattern.)
    ///
    /// This is almost equivalent to setting all non-ASCII bytes to be quit
    /// bytes. The only difference is that this will cause non-ASCII bytes to
    /// be quit bytes _only_ when a Unicode word boundary is present in the
    /// regex pattern.
    ///
    /// This is disabled by default.
    pub fn unicode_word_boundary(mut self, yes: bool) -> Config {
        // We have a separate option for this instead of just setting the
        // appropriate quit bytes here because we don't want to set quit bytes
        // for every regex. We only want to set them when the regex contains a
        // Unicode word boundary.
        self.unicode_word_boundary = Some(yes);
        self
    }

    /// Add a "quit" byte to the DFA.
    ///
    /// When a quit byte is seen during search time, then search will return
    /// a [`MatchError::Quit`](../../enum.MatchError.html#variant.Quit) error
    /// indicating the offset at which the search stopped.
    ///
    /// A quit byte will always overrule any other aspects of a regex. For
    /// example, if the `x` byte is added as a quit byte and the regex `\w` is
    /// used, then observing `x` will cause the search to quit immediately
    /// despite the fact that `x` is in the `\w` class.
    ///
    /// This mechanism is primarily useful for heuristically enabling certain
    /// features like Unicode word boundaries in a DFA. Namely, if the input
    /// to search is ASCII, then a Unicode word boundary can be implemented
    /// via an ASCII word boundary with no change in semantics. Thus, a DFA
    /// can attempt to match a Unicode word boundary but give up as soon as it
    /// observes a non-ASCII byte. Indeed, if callers set all non-ASCII bytes
    /// to be quit bytes, then Unicode word boundaries will be permitted when
    /// building DFAs.
    ///
    /// When enabling this option, callers _must_ be prepared to handle a
    /// `MatchError` error during search. When using a
    /// [`Regex`](../struct.Regex.html), this corresponds to using the `try_`
    /// suite of methods.
    ///
    /// By default, there are no quit bytes set.
    ///
    /// # Panics
    ///
    /// This panics if Unicode word boundaries are enabled and any non-ASCII
    /// byte is removed from the set of quit bytes. Namely, enabling Unicode
    /// word boundaries requires setting every non-ASCII byte to a quit byte.
    /// So if the caller attempts to undo any of that, then this will panic.
    pub fn quit(mut self, byte: u8, yes: bool) -> Config {
        if self.get_unicode_word_boundary() && !byte.is_ascii() && !yes {
            panic!(
                "cannot set non-ASCII byte to be non-quit when \
                 Unicode word boundaries are enabled"
            );
        }
        if self.quit.is_none() {
            self.quit = Some(ByteSet::empty());
        }
        if yes {
            self.quit.as_mut().unwrap().add(byte);
        } else {
            self.quit.as_mut().unwrap().remove(byte);
        }
        self
    }

    pub fn get_anchored(&self) -> bool {
        self.anchored.unwrap_or(false)
    }

    pub fn get_accelerate(&self) -> bool {
        self.accelerate.unwrap_or(true)
    }

    pub fn get_minimize(&self) -> bool {
        self.minimize.unwrap_or(false)
    }

    pub fn get_match_kind(&self) -> MatchKind {
        self.match_kind.unwrap_or(MatchKind::LeftmostFirst)
    }

    pub fn get_starts_for_each_pattern(&self) -> bool {
        self.starts_for_each_pattern.unwrap_or(false)
    }

    pub fn get_byte_classes(&self) -> bool {
        self.byte_classes.unwrap_or(true)
    }

    pub fn get_unicode_word_boundary(&self) -> bool {
        self.unicode_word_boundary.unwrap_or(false)
    }

    pub fn get_quit(&self, byte: u8) -> bool {
        self.quit.map_or(false, |q| q.contains(byte))
    }

    pub(crate) fn overwrite(self, o: Config) -> Config {
        Config {
            anchored: o.anchored.or(self.anchored),
            accelerate: o.accelerate.or(self.accelerate),
            minimize: o.minimize.or(self.minimize),
            match_kind: o.match_kind.or(self.match_kind),
            starts_for_each_pattern: o
                .starts_for_each_pattern
                .or(self.starts_for_each_pattern),
            byte_classes: o.byte_classes.or(self.byte_classes),
            unicode_word_boundary: o
                .unicode_word_boundary
                .or(self.unicode_word_boundary),
            quit: o.quit.or(self.quit),
        }
    }
}

/// A builder for constructing a deterministic finite automaton from regular
/// expressions.
///
/// This builder permits configuring several aspects of the construction
/// process such as case insensitivity, Unicode support and various options
/// that impact the size of the generated DFA. In some cases, options (like
/// performing DFA minimization) can come with a substantial additional cost.
///
/// This builder always constructs a *single* DFA. As such, this builder can
/// only be used to construct regexes that either detect the presence of a
/// match or find the end location of a match. A single DFA cannot produce both
/// the start and end of a match. For that information, use a
/// [`Regex`](struct.Regex.html), which can be similarly configured using
/// [`RegexBuilder`](struct.RegexBuilder.html).
#[cfg(feature = "std")]
#[derive(Clone, Debug)]
pub struct Builder {
    config: Config,
    thompson: thompson::Builder,
}

#[cfg(feature = "std")]
impl Builder {
    /// Create a new dense DFA builder with the default configuration.
    pub fn new() -> Builder {
        Builder {
            config: Config::default(),
            thompson: thompson::Builder::new(),
        }
    }

    /// Build a DFA from the given pattern.
    ///
    /// If there was a problem parsing or compiling the pattern, then an error
    /// is returned.
    pub fn build(&self, pattern: &str) -> Result<OwnedDFA<usize>, Error> {
        self.build_many(&[pattern])
    }

    /// Build a DFA from the given patterns.
    pub fn build_many<P: AsRef<str>>(
        &self,
        patterns: &[P],
    ) -> Result<OwnedDFA<usize>, Error> {
        self.build_many_with_size::<usize, _>(patterns)
    }

    /// Build a DFA from the given pattern using a specific representation for
    /// the DFA's state IDs.
    ///
    /// If there was a problem parsing or compiling the pattern, then an error
    /// is returned.
    ///
    /// The representation of state IDs is determined by the `S` type
    /// parameter. In general, `S` is usually one of `u8`, `u16`, `u32`, `u64`
    /// or `usize`, where `usize` is the default used for `build`. The purpose
    /// of specifying a representation for state IDs is to reduce the memory
    /// footprint of a DFA.
    ///
    /// When using this routine, the chosen state ID representation will be
    /// used throughout determinization and minimization, if minimization
    /// was requested. Even if the minimized DFA can fit into the chosen
    /// state ID representation but the initial determinized DFA cannot,
    /// then this will still return an error. To get a minimized DFA with a
    /// smaller state ID representation, first build it with a bigger state ID
    /// representation, and then shrink the size of the DFA using one of its
    /// conversion routines, such as
    /// [`DFA::to_sized`](struct.DFA.html#method.to_sized).
    pub fn build_with_size<S: StateID>(
        &self,
        pattern: &str,
    ) -> Result<OwnedDFA<S>, Error> {
        self.build_many_with_size(&[pattern])
    }

    /// Build a DFA from the given patterns using `S` as the state identifier
    /// representation.
    pub fn build_many_with_size<S: StateID, P: AsRef<str>>(
        &self,
        patterns: &[P],
    ) -> Result<OwnedDFA<S>, Error> {
        let nfa = self.thompson.build_many(patterns).map_err(Error::nfa)?;
        self.build_from_nfa_with_size(&nfa)
    }

    /// Build a DFA from the given NFA.
    pub fn build_from_nfa(
        &self,
        nfa: &thompson::NFA,
    ) -> Result<OwnedDFA<usize>, Error> {
        self.build_from_nfa_with_size::<usize>(nfa)
    }

    /// Build a DFA from the given NFA using a specific representation for
    /// the DFA's state IDs.
    pub fn build_from_nfa_with_size<S: StateID>(
        &self,
        nfa: &thompson::NFA,
    ) -> Result<OwnedDFA<S>, Error> {
        let mut quit = self.config.quit.unwrap_or(ByteSet::empty());
        if self.config.get_unicode_word_boundary()
            && nfa.has_word_boundary_unicode()
        {
            for b in 0x80..=0xFF {
                quit.add(b);
            }
        }
        let classes = if self.config.get_byte_classes() {
            let mut set = nfa.byte_class_set().clone();
            if !quit.is_empty() {
                set.add_set(&quit);
            }
            set.byte_classes()
        } else {
            ByteClasses::singletons()
        };

        let mut dfa = DFA::empty(
            classes,
            nfa.match_len(),
            self.config.get_starts_for_each_pattern(),
        )?;
        Determinizer::new()
            .anchored(self.config.get_anchored())
            .match_kind(self.config.get_match_kind())
            .quit(quit)
            .run(nfa, &mut dfa)?;
        if self.config.get_minimize() {
            dfa.minimize();
        }
        if self.config.get_accelerate() {
            dfa.accelerate();
        }
        Ok(dfa)
    }

    /// Apply the given dense DFA configuration options to this builder.
    pub fn configure(&mut self, config: Config) -> &mut Builder {
        self.config = self.config.overwrite(config);
        self
    }

    /// Set the syntax configuration for this builder using
    /// [`SyntaxConfig`](../struct.SyntaxConfig.html).
    ///
    /// This permits setting things like case insensitivity, Unicode and multi
    /// line mode.
    ///
    /// These settings only apply when constructing a DFA directly from a
    /// pattern.
    pub fn syntax(&mut self, config: crate::SyntaxConfig) -> &mut Builder {
        self.thompson.syntax(config);
        self
    }

    /// Set the Thompson NFA configuration for this builder using
    /// [`nfa::thompson::Config`](../nfa/thompson/struct.Config.html).
    ///
    /// This permits setting things like whether the DFA should match the regex
    /// in reverse or if additional time should be spent shrinking the size of
    /// the NFA.
    pub fn thompson(&mut self, config: thompson::Config) -> &mut Builder {
        self.thompson.configure(config);
        self
    }
}

#[cfg(feature = "std")]
impl Default for Builder {
    fn default() -> Builder {
        Builder::new()
    }
}

/// A convenience alias for an owned DFA. We use this particular instantiation
/// a lot in this crate, so it's worth giving it a name. This instantiation
/// is commonly used for mutable APIs on the DFA while building it. The main
/// reason for making it generic is no_std support, and more generally, making
/// it possible to load a DFA from an arbitrary slice of bytes.
pub(crate) type OwnedDFA<S> = DFA<Vec<S>, Vec<u8>, S>;

/// A dense table-based deterministic finite automaton (DFA).
///
/// All dense DFAs have one or more start states, zero or more match states
/// and a transition table that maps the current state and the current byte
/// of input to the next state. A DFA can use this information to implement
/// fast searching. In particular, the use of a dense DFA generally makes the
/// trade off that match speed is the most valuable characteristic, even if
/// building the DFA may take significant time *and* space. (More concretely,
/// building a DFA takes time and space that is exponential in the size of the
/// pattern in the worst case.) As such, the processing of every byte of input
/// is done with a small constant number of operations that does not vary with
/// the pattern, its size or the size of the alphabet. If your needs don't line
/// up with this trade off, then a dense DFA may not be an adequate solution to
/// your problem.
///
/// In contrast, a [sparse DFA](../sparse/struct.DFA.html) makes the opposite
/// trade off: it uses less space but will execute a variable number of
/// instructions per byte at match time, which makes it slower for matching.
/// (Note that space usage is still exponential in the size of the pattern in
/// the worst case.)
///
/// A DFA can be built using the default configuration via the
/// [`DFA::new`](struct.DFA.html#method.new) constructor. Otherwise, one can
/// configure various aspects via [`dense::Builder`](struct.Builder.html).
///
/// A single DFA fundamentally supports the following operations:
///
/// 1. Detection of a match.
/// 2. Location of the end of a match.
///
/// A notable absence from the above list of capabilities is the location of
/// the *start* of a match. In order to provide both the start and end of
/// a match, *two* DFAs are required. This functionality is provided by a
/// [`Regex`](../struct.Regex.html).
///
/// # Type parameters and state size
///
/// A `DFA` has three type parameters, `T`, `A` and `S`:
///
/// * `T` is the type of the DFA's transition table. `T` is typically
///   `Vec<S>` or `&[S]`.
/// * `A` is the type used for the DFA's acceleration table. `A` is typically
///   `Vec<u8>` or `&[u8]`.
/// * `S` is the representation used for the DFA's state identifiers as
///   described by the [`StateID`](../../trait.StateID.html) trait. `S` must
///   be one of `usize`, `u8`, `u16`, `u32` or `u64`. It defaults to
///   `usize`. The primary reason for choosing a different state identifier
///   representation than the default is to reduce the amount of memory used by
///   a DFA. Note though, that if the chosen representation cannot accommodate
///   the size of your DFA, then building the DFA will fail and return an
///   error.
///
/// While the reduction in heap memory used by a DFA is one reason for choosing
/// a smaller state identifier representation, another possible reason is for
/// decreasing the serialization size of a DFA, as returned by
/// [`to_bytes_little_endian`](struct.DFA.html#method.to_bytes_little_endian),
/// [`to_bytes_big_endian`](struct.DFA.html#method.to_bytes_big_endian)
/// or
/// [`to_bytes_native_endian`](struct.DFA.html#method.to_bytes_native_endian).
/// A smaller DFA also means that more of it will fit in your CPU's cache,
/// potentially leading to overall better search performance.
///
/// # The `Automaton` trait
///
/// This type implements the [`Automaton`](../trait.Automaton.html) trait,
/// which means it can be used for searching. For example:
///
/// ```
/// use regex_automata::dfa::{Automaton, HalfMatch, dense::DFA};
///
/// let dfa = DFA::new("foo[0-9]+")?;
/// let expected = HalfMatch::new(0, 8);
/// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo12345")?);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone)]
pub struct DFA<T, A, S = usize> {
    /// The transition table for this DFA. This includes the transitions
    /// themselves, along with the stride, number of states and the equivalence
    /// class mapping.
    tt: TransitionTable<T, S>,
    /// The set of starting state identifiers for this DFA. The starting state
    /// IDs act as pointers into the transition table. The specific starting
    /// state chosen for each search is dependent on the context at which the
    /// search begins.
    st: StartTable<T, S>,
    /// The set of match states and the patterns that match for each
    /// corresponding match state.
    ///
    /// This structure is technically only needed because of support for
    /// multi-regexes. Namely, multi-regexes require answering not just whether
    /// a match exists, but _which_ patterns match. So we need to store the
    /// matching pattern IDs for each match state.
    ms: MatchStates<T, A, S>,
    /// Information about which states as "special." Special states are states
    /// that are dead, quit, matching, starting or accelerated. For more info,
    /// see the docs for `Special`.
    special: Special<S>,
    /// The accelerators for this DFA.
    ///
    /// If a state is accelerated, then there exist only a small number of
    /// bytes that can cause the DFA to leave the state. This permits searching
    /// to use optimized routines to find those specific bytes instead of using
    /// the transition table.
    ///
    /// All accelerated states exist in a contiguous range in the DFA's
    /// transition table. See dfa/special.rs for more details on how states are
    /// arranged.
    ///
    /// In practice, A is either Vec<u8> or &[u8].
    accels: Accels<A>,
}

#[cfg(feature = "std")]
impl OwnedDFA<usize> {
    /// Parse the given regular expression using a default configuration and
    /// return the corresponding DFA.
    ///
    /// The default configuration uses `usize` for state IDs. The DFA is *not*
    /// minimized.
    ///
    /// If you want a non-default configuration, then use the
    /// [`dense::Builder`](dense/struct.Builder.html)
    /// to set your own configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, dense};
    ///
    /// let dfa = dense::DFA::new("foo[0-9]+bar")?;
    /// let expected = HalfMatch::new(0, 11);
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo12345bar")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(pattern: &str) -> Result<OwnedDFA<usize>, Error> {
        Builder::new().build(pattern)
    }
}

#[cfg(feature = "std")]
impl<S: StateID> OwnedDFA<S> {
    /// Create a new DFA that matches every input.
    ///
    /// # Example
    ///
    /// In order to build a DFA that always matches, callers must provide a
    /// type hint indicating their choice of state identifier representation.
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, dense};
    ///
    /// let dfa: dense::DFA<Vec<_>, _, usize> = dense::DFA::always_match()?;
    ///
    /// let expected = HalfMatch::new(0, 0);
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"")?);
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn always_match() -> Result<OwnedDFA<S>, Error> {
        let nfa = thompson::NFA::always_match();
        Builder::new().build_from_nfa_with_size(&nfa)
    }

    /// Create a new DFA that never matches any input.
    ///
    /// # Example
    ///
    /// In order to build a DFA that never matches, callers must provide a type
    /// hint indicating their choice of state identifier representation.
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, dense};
    ///
    /// let dfa: dense::DFA<Vec<_>, _, usize> = dense::DFA::never_match()?;
    /// assert_eq!(None, dfa.find_leftmost_fwd(b"")?);
    /// assert_eq!(None, dfa.find_leftmost_fwd(b"foo")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn never_match() -> Result<OwnedDFA<S>, Error> {
        let nfa = thompson::NFA::never_match();
        Builder::new().build_from_nfa_with_size(&nfa)
    }

    /// Create a new DFA with the given set of byte equivalence classes, along
    /// with the total number of patterns in this DFA. The DFA contains a
    /// single dead state that never matches any input.
    fn empty(
        classes: ByteClasses,
        pattern_count: usize,
        starts_for_each_pattern: bool,
    ) -> Result<OwnedDFA<S>, Error> {
        let start_pattern_count =
            if starts_for_each_pattern { pattern_count } else { 0 };
        Ok(DFA {
            tt: TransitionTable::minimal(classes)?,
            st: StartTable::dead(start_pattern_count),
            ms: MatchStates::empty(pattern_count),
            special: Special::new(),
            accels: Accels::empty(),
        })
    }
}

impl<T: AsRef<[S]>, A: AsRef<[u8]>, S: StateID> DFA<T, A, S> {
    /// Cheaply return a borrowed version of this dense DFA. Specifically, the
    /// DFA returned always uses `&[S]` for its transition table while keeping
    /// the same state identifier representation.
    pub fn as_ref(&self) -> DFA<&'_ [S], &'_ [u8], S> {
        DFA {
            tt: self.tt.as_ref(),
            st: self.st.as_ref(),
            ms: self.ms.as_ref(),
            special: self.special,
            accels: self.accels(),
        }
    }

    /// Return an owned version of this sparse DFA. Specifically, the DFA
    /// returned always uses `Vec<S>` for its transition table while keeping
    /// the same state identifier representation.
    ///
    /// Effectively, this returns a dense DFA whose transition table lives on
    /// the heap.
    #[cfg(feature = "std")]
    pub fn to_owned(&self) -> OwnedDFA<S> {
        DFA {
            tt: self.tt.to_owned(),
            st: self.st.to_owned(),
            ms: self.ms.to_owned(),
            special: self.special,
            accels: self.accels().to_owned(),
        }
    }

    /// Returns the memory usage, in bytes, of this DFA.
    ///
    /// The memory usage is computed based on the number of bytes used to
    /// represent this DFA's transition table. This corresponds to heap memory
    /// usage.
    ///
    /// This does **not** include the stack size used up by this DFA. To
    /// compute that, used `std::mem::size_of::<dense::DFA>()`.
    pub fn memory_usage(&self) -> usize {
        let state_size = mem::size_of::<S>();
        self.accels().as_bytes().len()
            + self.tt.memory_usage()
            + self.st.memory_usage()
    }

    /// Returns true only if this DFA has starting states for each pattern.
    ///
    /// When a DFA has starting states for each pattern, then a search with the
    /// DFA can be configured to only look for anchored matches of a specific
    /// pattern. Specifically, APIs like
    /// [`Automaton::find_earliest_fwd_at`](../trait.Automaton.html#method.find_earliest_fwd_at)
    /// can accept a non-None `pattern_id` if and only if this method returns
    /// true. Otherwise, calling `find_earliest_fwd_at` will panic.
    ///
    /// Note that if the DFA is empty, this always returns false.
    pub fn has_starts_for_each_pattern(&self) -> bool {
        self.st.patterns > 0
    }

    /// Returns the total number of elements in the alphabet for this DFA.
    ///
    /// That is, this returns the total number of transitions that each state
    /// in this DFA must have. Typically, a normal byte oriented DFA would
    /// always have an alphabet size of 256, corresponding to the number of
    /// unique values in a single byte. However, this implementation has two
    /// peculiarities that impact the alphabet length:
    ///
    /// * Every state has a special "EOF" transition that is only followed
    /// after the end of some haystack is reached. This EOF transition is
    /// necessary to account for one byte of look-ahead when implementing
    /// things like `\b` and `$`.
    /// * Bytes are grouped into equivalence classes such that no two bytes in
    /// the same class can distinguish a match from a non-match. For example,
    /// in the regex `^[a-z]+$`, the ASCII bytes `a-z` could all be in the
    /// same equivalence class. This leads to a massive space savings.
    ///
    /// Note though that the alphabet length does _not_ necessarily equal the
    /// total stride space taken up by a single DFA state in the transition
    /// table. Namely, for performance reasons, the stride is always the
    /// smallest power of two that is greater than or equal to the alphabet
    /// length.
    pub fn alphabet_len(&self) -> usize {
        self.tt.alphabet_len()
    }

    /// Returns the total stride for every state in this DFA, expressed as the
    /// exponent of a power of 2.
    ///
    /// The stride of a DFA is always equivalent to the smallest power of 2
    /// that is greater than or equal to the DFA's alphabet length. This
    /// definition uses extra space, but permits faster translation between
    /// premultiplied state identifiers and contiguous indices (by using shifts
    /// instead of relying on integer division).
    ///
    /// For example, if the DFA's stride is 16 transitions, then its `stride2`
    /// is `4` since `2^4 = 16`.
    ///
    /// The minimum `stride2` value is `1` (corresponding to a stride of `2`)
    /// while the maximum `stride2` value is `9` (corresponding to a stride of
    /// `512`). The maximum is not `8` since the maximum alphabet size is `257`
    /// when accounting for the special EOF transition. However, an alphabet
    /// length of that size is exceptionally rare since the alphabet is shrunk
    /// into equivalence classes.
    pub fn stride2(&self) -> usize {
        self.tt.stride2
    }

    /// Returns the total stride for every state in this DFA. This corresponds
    /// to the total number of transitions used by each state in this DFA's
    /// transition table.
    ///
    /// Please see [`stride2`](struct.DFA.html#method.stride2) for more
    /// information. In particular, this returns the stride as the number of
    /// transitions, where as `stride2` returns it as the exponent of a power
    /// of 2.
    pub fn stride(&self) -> usize {
        self.tt.stride()
    }
}

/// Routines for converting a dense DFA to other representations, such as
/// sparse DFAs, smaller state identifiers or raw bytes suitable for persistent
/// storage.
#[cfg(feature = "std")]
impl<T: AsRef<[S]>, A: AsRef<[u8]>, S: StateID> DFA<T, A, S> {
    /// Convert this dense DFA to a sparse DFA.
    ///
    /// This is a convenience routine for `to_sparse_sized` that fixes the
    /// state identifier representation of the sparse DFA to the same
    /// representation used for this dense DFA.
    ///
    /// If the state identifier representation is too small to represent all
    /// states in the sparse DFA, then this returns an error. In most cases,
    /// if a dense DFA is constructable with `S` then a sparse DFA will be as
    /// well. However, it is not guaranteed.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, dense};
    ///
    /// let dense = dense::DFA::new("foo[0-9]+")?;
    /// let sparse = dense.to_sparse()?;
    ///
    /// let expected = HalfMatch::new(0, 8);
    /// assert_eq!(Some(expected), sparse.find_leftmost_fwd(b"foo12345")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_sparse(&self) -> Result<sparse::DFA<Vec<u8>, S>, Error> {
        self.to_sparse_sized()
    }

    /// Convert this dense DFA to a sparse DFA.
    ///
    /// Using this routine requires supplying a type hint to choose the state
    /// identifier representation for the resulting sparse DFA.
    ///
    /// If the chosen state identifier representation is too small to represent
    /// all states in the sparse DFA, then this returns an error.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, dense};
    ///
    /// let dense = dense::DFA::new("foo[0-9]+")?;
    /// let sparse = dense.to_sparse_sized::<u16>()?;
    ///
    /// let expected = HalfMatch::new(0, 8);
    /// assert_eq!(Some(expected), sparse.find_leftmost_fwd(b"foo12345")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_sparse_sized<S2: StateID>(
        &self,
    ) -> Result<sparse::DFA<Vec<u8>, S2>, Error> {
        sparse::DFA::from_dense_sized(self)
    }

    /// Create a new DFA whose match semantics are equivalent to this DFA, but
    /// attempt to use `S2` for the representation of state identifiers. If
    /// `S2` is insufficient to represent all state identifiers in this DFA,
    /// then this returns an error.
    ///
    /// An alternative way to construct such a DFA is to use
    /// [`dense::Builder::build_with_size`](struct.Builder.html#method.build_with_size).
    /// In general, using the builder is preferred since it will use the given
    /// state identifier representation throughout determinization (and
    /// minimization, if done), and thereby using less memory throughout the
    /// entire construction process. However, this routine is necessary
    /// in cases where, say, a minimized DFA could fit in a smaller state
    /// identifier representation, but the initial determinized DFA would not.
    ///
    /// # Example
    ///
    /// This example shows how to create a DFA with `u16` as the state
    /// identifier representation.
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, dense::DFA};
    ///
    /// let dfa = DFA::new("foo[0-9]+")?.to_sized::<u16>()?;
    ///
    /// let expected = HalfMatch::new(0, 8);
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo12345")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_sized<S2: StateID>(&self) -> Result<OwnedDFA<S2>, Error> {
        // The new DFA is the same as the old one, except all state IDs are
        // represented by `S2` instead of `S`.
        Ok(DFA {
            tt: self.tt.as_ref().to_sized()?,
            st: self.st.to_sized()?,
            ms: self.ms.to_sized()?,
            special: self.special.to_sized()?,
            accels: self.accels().to_owned(),
        })
    }

    /// Serialize this DFA as raw bytes to a `Vec<u8>` in little endian
    /// format. Upon success, the `Vec<u8>` and the initial padding length are
    /// returned.
    ///
    /// The written bytes are guaranteed to be deserialized correctly and
    /// without errors in a semver compatible release of this crate by a
    /// `DFA`'s deserialization APIs (assuming all other criteria for the
    /// deserialization APIs has been satisfied):
    ///
    /// * [`from_bytes`](struct.DFA.html#method.from_bytes)
    /// * [`from_bytes_unchecked`](struct.DFA.html#method.from_bytes_unchecked)
    ///
    /// The padding returned is non-zero if the returned `Vec<u8>` starts at
    /// an address that does not have the same alignment as `S`. The padding
    /// corresponds to the number of leading bytes written to the returned
    /// `Vec<u8>`. The number of padding bytes written is typically zero, but
    /// will never be more than 7.
    ///
    /// # Example
    ///
    /// This example shows how to serialize and deserialize a DFA:
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, dense::DFA};
    ///
    /// // Compile our original DFA. We use 16-bit state identifiers to give
    /// // our state IDs a small fixed size.
    /// let original_dfa = DFA::new("foo[0-9]+")?.to_sized::<u16>()?;
    ///
    /// // N.B. We use native endianness here to make the example work, but
    /// // using to_bytes_little_endian would work on a little endian target.
    /// let (buf, _) = original_dfa.to_bytes_native_endian();
    /// // Even if buf has initial padding, DFA::from_bytes will automatically
    /// // ignore it.
    /// let dfa: DFA<&[u16], &[u8], u16> = DFA::from_bytes(&buf)?.0;
    ///
    /// let expected = HalfMatch::new(0, 8);
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo12345")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_bytes_little_endian(&self) -> (Vec<u8>, usize) {
        self.to_bytes::<bytes::LE>()
    }

    /// Serialize this DFA as raw bytes to a `Vec<u8>` in big endian
    /// format. Upon success, the `Vec<u8>` and the initial padding length are
    /// returned.
    ///
    /// The written bytes are guaranteed to be deserialized correctly and
    /// without errors in a semver compatible release of this crate by a
    /// `DFA`'s deserialization APIs (assuming all other criteria for the
    /// deserialization APIs has been satisfied):
    ///
    /// * [`from_bytes`](struct.DFA.html#method.from_bytes)
    /// * [`from_bytes_unchecked`](struct.DFA.html#method.from_bytes_unchecked)
    ///
    /// The padding returned is non-zero if the returned `Vec<u8>` starts at
    /// an address that does not have the same alignment as `S`. The padding
    /// corresponds to the number of leading bytes written to the returned
    /// `Vec<u8>`. The number of padding bytes written is typically zero, but
    /// will never be more than 7.
    ///
    /// # Example
    ///
    /// This example shows how to serialize and deserialize a DFA:
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, dense::DFA};
    ///
    /// // Compile our original DFA. We use 16-bit state identifiers to give
    /// // our state IDs a small fixed size.
    /// let original_dfa = DFA::new("foo[0-9]+")?.to_sized::<u16>()?;
    ///
    /// // N.B. We use native endianness here to make the example work, but
    /// // using to_bytes_big_endian would work on a big endian target.
    /// let (buf, _) = original_dfa.to_bytes_native_endian();
    /// // Even if buf has initial padding, DFA::from_bytes will automatically
    /// // ignore it.
    /// let dfa: DFA<&[u16], &[u8], u16> = DFA::from_bytes(&buf)?.0;
    ///
    /// let expected = HalfMatch::new(0, 8);
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo12345")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_bytes_big_endian(&self) -> (Vec<u8>, usize) {
        self.to_bytes::<bytes::BE>()
    }

    /// Serialize this DFA as raw bytes to a `Vec<u8>` in native endian
    /// format. Upon success, the `Vec<u8>` and the initial padding length are
    /// returned.
    ///
    /// The written bytes are guaranteed to be deserialized correctly and
    /// without errors in a semver compatible release of this crate by a
    /// `DFA`'s deserialization APIs (assuming all other criteria for the
    /// deserialization APIs has been satisfied):
    ///
    /// * [`from_bytes`](struct.DFA.html#method.from_bytes)
    /// * [`from_bytes_unchecked`](struct.DFA.html#method.from_bytes_unchecked)
    ///
    /// The padding returned is non-zero if the returned `Vec<u8>` starts at
    /// an address that does not have the same alignment as `S`. The padding
    /// corresponds to the number of leading bytes written to the returned
    /// `Vec<u8>`. The number of padding bytes written is typically zero, but
    /// will never be more than 7.
    ///
    /// Generally speaking, native endian format should only be used when
    /// you know that the target you're compiling the DFA for matches the
    /// endianness of the target on which you're compiling DFA. For example,
    /// if serialization and deserialization happen in the same process or on
    /// the same machine. Otherwise, when serializing a DFA for use in a
    /// portable environment, you'll almost certainly want to serialize _both_
    /// a little endian and a big endian version and then load the correct one
    /// based on the target's configuration.
    ///
    /// # Example
    ///
    /// This example shows how to serialize and deserialize a DFA:
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, dense::DFA};
    ///
    /// // Compile our original DFA. We use 16-bit state identifiers to give
    /// // our state IDs a small fixed size.
    /// let original_dfa = DFA::new("foo[0-9]+")?.to_sized::<u16>()?;
    ///
    /// let (buf, _) = original_dfa.to_bytes_native_endian();
    /// // Even if buf has initial padding, DFA::from_bytes will automatically
    /// // ignore it.
    /// let dfa: DFA<&[u16], &[u8], u16> = DFA::from_bytes(&buf)?.0;
    ///
    /// let expected = HalfMatch::new(0, 8);
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo12345")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_bytes_native_endian(&self) -> (Vec<u8>, usize) {
        self.to_bytes::<bytes::NE>()
    }

    /// The implementation of the public `to_bytes` serialization methods,
    /// which is generic over endianness.
    fn to_bytes<E: Endian>(&self) -> (Vec<u8>, usize) {
        let len = self.write_to_len();
        let (mut buf, padding) = bytes::alloc_aligned_buffer::<S>(len);
        // This should always succeed since the only possible serialization
        // error is providing a buffer that's too small, but we've ensured that
        // `buf` is big enough here.
        self.as_ref().write_to::<E>(&mut buf[padding..]).unwrap();
        (buf, padding)
    }

    /// Serialize this DFA as raw bytes to the given slice, in little endian
    /// format. Upon success, the total number of bytes written to `dst` is
    /// returned.
    ///
    /// The written bytes are guaranteed to be deserialized correctly and
    /// without errors in a semver compatible release of this crate by a
    /// `DFA`'s deserialization APIs (assuming all other criteria for the
    /// deserialization APIs has been satisfied):
    ///
    /// * [`from_bytes`](struct.DFA.html#method.from_bytes)
    /// * [`from_bytes_unchecked`](struct.DFA.html#method.from_bytes_unchecked)
    ///
    /// # Errors
    ///
    /// This returns an error if the given destination slice is not big enough
    /// to contain the full serialized DFA. If an error occurs, then nothing
    /// is written to `dst`.
    ///
    /// # Example
    ///
    /// This example shows how to serialize and deserialize a DFA without
    /// dynamic memory allocation.
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, dense::DFA};
    ///
    /// // Compile our original DFA. We use 16-bit state identifiers to give
    /// // our state IDs a small fixed size.
    /// let original_dfa = DFA::new("foo[0-9]+")?.to_sized::<u16>()?;
    ///
    /// // Create a 4KB buffer on the stack to store our serialized DFA.
    /// let mut buf = [0u8; 4 * (1<<10)];
    /// // N.B. We use native endianness here to make the example work, but
    /// // using write_to_little_endian would work on a little endian target.
    /// let written = original_dfa.write_to_native_endian(&mut buf)?;
    /// let dfa: DFA<&[u16], &[u8], u16> = DFA::from_bytes(&buf[..written])?.0;
    ///
    /// let expected = HalfMatch::new(0, 8);
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo12345")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn write_to_little_endian(
        &self,
        dst: &mut [u8],
    ) -> Result<usize, SerializeError> {
        self.as_ref().write_to::<bytes::LE>(dst)
    }

    /// Serialize this DFA as raw bytes to the given slice, in big endian
    /// format. Upon success, the total number of bytes written to `dst` is
    /// returned.
    ///
    /// The written bytes are guaranteed to be deserialized correctly and
    /// without errors in a semver compatible release of this crate by a
    /// `DFA`'s deserialization APIs (assuming all other criteria for the
    /// deserialization APIs has been satisfied):
    ///
    /// * [`from_bytes`](struct.DFA.html#method.from_bytes)
    /// * [`from_bytes_unchecked`](struct.DFA.html#method.from_bytes_unchecked)
    ///
    /// # Errors
    ///
    /// This returns an error if the given destination slice is not big enough
    /// to contain the full serialized DFA. If an error occurs, then nothing
    /// is written to `dst`.
    ///
    /// # Example
    ///
    /// This example shows how to serialize and deserialize a DFA without
    /// dynamic memory allocation.
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, dense::DFA};
    ///
    /// // Compile our original DFA. We use 16-bit state identifiers to give
    /// // our state IDs a small fixed size.
    /// let original_dfa = DFA::new("foo[0-9]+")?.to_sized::<u16>()?;
    ///
    /// // Create a 4KB buffer on the stack to store our serialized DFA.
    /// let mut buf = [0u8; 4 * (1<<10)];
    /// // N.B. We use native endianness here to make the example work, but
    /// // using write_to_big_endian would work on a big endian target.
    /// let written = original_dfa.write_to_native_endian(&mut buf)?;
    /// let dfa: DFA<&[u16], &[u8], u16> = DFA::from_bytes(&buf[..written])?.0;
    ///
    /// let expected = HalfMatch::new(0, 8);
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo12345")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn write_to_big_endian(
        &self,
        dst: &mut [u8],
    ) -> Result<usize, SerializeError> {
        self.as_ref().write_to::<bytes::BE>(dst)
    }

    /// Serialize this DFA as raw bytes to the given slice, in native endian
    /// format. Upon success, the total number of bytes written to `dst` is
    /// returned.
    ///
    /// The written bytes are guaranteed to be deserialized correctly and
    /// without errors in a semver compatible release of this crate by a
    /// `DFA`'s deserialization APIs (assuming all other criteria for the
    /// deserialization APIs has been satisfied):
    ///
    /// * [`from_bytes`](struct.DFA.html#method.from_bytes)
    /// * [`from_bytes_unchecked`](struct.DFA.html#method.from_bytes_unchecked)
    ///
    /// Generally speaking, native endian format should only be used when
    /// you know that the target you're compiling the DFA for matches the
    /// endianness of the target on which you're compiling DFA. For example,
    /// if serialization and deserialization happen in the same process or on
    /// the same machine. Otherwise, when serializing a DFA for use in a
    /// portable environment, you'll almost certainly want to serialize _both_
    /// a little endian and a big endian version and then load the correct one
    /// based on the target's configuration.
    ///
    /// # Errors
    ///
    /// This returns an error if the given destination slice is not big enough
    /// to contain the full serialized DFA. If an error occurs, then nothing
    /// is written to `dst`.
    ///
    /// # Example
    ///
    /// This example shows how to serialize and deserialize a DFA without
    /// dynamic memory allocation.
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, dense::DFA};
    ///
    /// // Compile our original DFA. We use 16-bit state identifiers to give
    /// // our state IDs a small fixed size.
    /// let original_dfa = DFA::new("foo[0-9]+")?.to_sized::<u16>()?;
    ///
    /// // Create a 4KB buffer on the stack to store our serialized DFA.
    /// let mut buf = [0u8; 4 * (1<<10)];
    /// let written = original_dfa.write_to_native_endian(&mut buf)?;
    /// let dfa: DFA<&[u16], &[u8], u16> = DFA::from_bytes(&buf[..written])?.0;
    ///
    /// let expected = HalfMatch::new(0, 8);
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo12345")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn write_to_native_endian(
        &self,
        dst: &mut [u8],
    ) -> Result<usize, SerializeError> {
        self.as_ref().write_to::<bytes::NE>(dst)
    }

    /// Return the total number of bytes required to serialize this DFA.
    ///
    /// This is useful for determining the size of the buffer required to pass
    /// to one of the serialization routines:
    ///
    /// * [`write_to_little_endian`](struct.DFA.html#method.write_to_little_endian)
    /// * [`write_to_big_endian`](struct.DFA.html#method.write_to_big_endian)
    /// * [`write_to_native_endian`](struct.DFA.html#method.write_to_native_endian)
    ///
    /// Passing a buffer smaller than the size returned by this method will
    /// result in a serialization error.
    ///
    /// # Example
    ///
    /// This example shows how to dynamically allocate enough room to serialize
    /// a DFA.
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, dense::DFA};
    ///
    /// // Compile our original DFA. We use 16-bit state identifiers to give
    /// // our state IDs a small fixed size.
    /// let original_dfa = DFA::new("foo[0-9]+")?.to_sized::<u16>()?;
    ///
    /// let mut buf = vec![0; original_dfa.write_to_len()];
    /// let written = original_dfa.write_to_native_endian(&mut buf)?;
    /// let dfa: DFA<&[u16], &[u8], u16> = DFA::from_bytes(&buf[..written])?.0;
    ///
    /// let expected = HalfMatch::new(0, 8);
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo12345")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// Note that this example isn't actually guaranteed to work! In
    /// particular, if `buf` is aligned to a 2-byte boundary, then the
    /// `DFA::from_bytes` call will fail. If you need this to work, then you
    /// either need to deal with adding some initial padding yourself, or use
    /// one of the `to_bytes` methods, which will do it for you.
    pub fn write_to_len(&self) -> usize {
        bytes::write_label_len(LABEL)
        + bytes::write_endianness_check_len()
        + bytes::write_version_len()
        + bytes::write_state_size_len()
        + 8 // unused, intended for future flexibility
        + self.tt.as_ref().write_to_len()
        + self.st.write_to_len()
        + self.ms.write_to_len()
        + self.special.write_to_len()
        + self.accels.write_to_len()
    }
}

impl<'a, S: StateID> DFA<&'a [S], &'a [u8], S> {
    /// Safely deserialize a DFA with a specific state identifier
    /// representation. Upon success, this returns both the deserialized DFA
    /// and the number of bytes read from the given slice. Namely, the contents
    /// of the slice beyond the DFA are not read.
    ///
    /// Deserializing a DFA using this routine will never allocate heap memory.
    /// For safety purposes, the DFA's transition table will be verified such
    /// that every transition points to a valid state. If this verification is
    /// too costly, then a
    /// [`from_bytes_unchecked`](struct.DFA.html#method.from_bytes_unchecked)
    /// API is provided, which will always execute in constant time.
    ///
    /// The bytes given must be generated by one of the serialization APIs
    /// of a `DFA` using a semver compatible release of this crate. Those
    /// include:
    ///
    /// * [`to_bytes_little_endian`](struct.DFA.html#method.to_bytes_little_endian)
    /// * [`to_bytes_big_endian`](struct.DFA.html#method.to_bytes_big_endian)
    /// * [`to_bytes_native_endian`](struct.DFA.html#method.to_bytes_native_endian)
    /// * [`write_to_little_endian`](struct.DFA.html#method.write_to_little_endian)
    /// * [`write_to_big_endian`](struct.DFA.html#method.write_to_big_endian)
    /// * [`write_to_native_endian`](struct.DFA.html#method.write_to_native_endian)
    ///
    /// The `to_bytes` methods allocate and return a `Vec<u8>` for you, along
    /// with handling alignment correctly. The `write_to` methods do not
    /// allocate and write to an existing slice (which may be on the stack).
    /// Since deserialization always uses the native endianness of the target
    /// platform, the serialization API you use should match the endianness of
    /// the target platform. (It's often a good idea to generate serialized
    /// DFAs for both forms of endianness and then load the correct one based
    /// on endianness.)
    ///
    /// If the state identifier representation is `usize`, then deserialization
    /// is dependent on the pointer size. For this reason, it is best to
    /// serialize DFAs using a fixed size representation for your state
    /// identifiers, such as `u8`, `u16`, `u32` or `u64`.
    ///
    /// # Errors
    ///
    /// Generally speaking, it's easier to state the conditions in which an
    /// error is _not_ returned. All of the following must be true:
    ///
    /// * The bytes given must be produced by one of the serialization APIs
    ///   on this DFA, as mentioned above.
    /// * The state ID representation chosen by type inference (that's the `S`
    ///   type parameter) must match the state ID representation in the given
    ///   serialized DFA.
    /// * The endianness of the target platform matches the endianness used to
    ///   serialized the provided DFA.
    /// * The slice given must have the same alignment as `S`.
    ///
    /// If any of the above are not true, then an error will be returned.
    ///
    /// # Panics
    ///
    /// This routine will never panic for any input.
    ///
    /// # Example
    ///
    /// This example shows how to serialize a DFA to raw bytes, deserialize it
    /// and then use it for searching. Note that we first convert the DFA to
    /// using `u16` for its state identifier representation before serializing
    /// it. While this isn't strictly necessary, it's good practice in order to
    /// decrease the size of the DFA and to avoid platform specific pitfalls
    /// such as differing pointer sizes.
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, dense::DFA};
    ///
    /// let initial = DFA::new("foo[0-9]+")?;
    /// let (bytes, _) = initial.to_sized::<u16>()?.to_bytes_native_endian();
    /// let dfa: DFA<&[u16], &[u8], u16> = DFA::from_bytes(&bytes)?.0;
    ///
    /// let expected = HalfMatch::new(0, 8);
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo12345")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: dealing with alignment and padding
    ///
    /// In the above example, we used the `to_bytes_native_endian` method to
    /// serialize a DFA, but we ignored part of its return value corresponding
    /// to padding added to the beginning of the serialized DFA. This is OK
    /// because deserialization will skip this initial padding. What matters
    /// is that the address immediately following the padding has an alignment
    /// that matches `S`. That is, the following is an equivalent but
    /// alternative way to write the above example:
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, dense::DFA};
    ///
    /// let initial = DFA::new("foo[0-9]+")?;
    /// // Serialization returns the number of leading padding bytes added to
    /// // the returned Vec<u8>.
    /// let (bytes, pad) = initial.to_sized::<u16>()?.to_bytes_native_endian();
    /// let dfa: DFA<&[u16], &[u8], u16> = DFA::from_bytes(&bytes[pad..])?.0;
    ///
    /// let expected = HalfMatch::new(0, 8);
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo12345")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// This padding is necessary because Rust's standard library does
    /// not expose any safe and robust way of creating a `Vec<u8>` with a
    /// guaranteed alignment other than 1. Now, in practice, the underlying
    /// allocator is likely to provide a `Vec<u8>` that meets our alignment
    /// requirements, which means `pad` is zero in practice most of the time.
    ///
    /// The purpose of exposing the padding like this is flexibility for the
    /// caller. For example, if one wants to embed a serialized DFA into a
    /// compiled program, then it's important to guarantee that it starts at
    /// an `S`-aligned address. The simplest way to do this is to discard the
    /// padding bytes and set it up so that the serialized DFA itself begins
    /// at a properly aligned address. We can show this in two parts. The first
    /// part is serializing the DFA to a file:
    ///
    /// ```no_run
    /// use regex_automata::dfa::{Automaton, dense::DFA};
    ///
    /// let dfa = DFA::new("foo[0-9]+")?;
    ///
    /// let (bytes, pad) = dfa.to_sized::<u16>()?.to_bytes_big_endian();
    /// // Write the contents of the DFA *without* the initial padding.
    /// std::fs::write("foo.bigendian.dfa", &bytes[pad..])?;
    ///
    /// // Do it again, but this time for little endian.
    /// let (bytes, pad) = dfa.to_sized::<u16>()?.to_bytes_little_endian();
    /// std::fs::write("foo.littleendian.dfa", &bytes[pad..])?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// And now the second part is embedding the DFA into the compiled program
    /// and deserializing it at runtime on first use. We use conditional
    /// compilation to choose the correct endianness.
    ///
    /// ```no_run
    /// use regex_automata::dfa::{Automaton, HalfMatch, dense};
    ///
    /// type S = u16;
    /// type DFA = dense::DFA<&'static [S], &'static [u8], S>;
    ///
    /// fn get_foo() -> &'static DFA {
    ///     use std::cell::Cell;
    ///     use std::mem::MaybeUninit;
    ///     use std::sync::Once;
    ///
    ///     // This struct with a generic B is used to permit unsizing
    ///     // coercions, specifically, where B winds up being a [u8]. We also
    ///     // need repr(C) to guarantee that _align comes first, which forces
    ///     // a correct alignment.
    ///     #[repr(C)]
    ///     struct Aligned<B: ?Sized> {
    ///         _align: [S; 0],
    ///         bytes: B,
    ///     }
    ///
    ///     # const _: &str = stringify! {
    ///     // This assignment is made possible (implicitly) via the
    ///     // CoerceUnsized trait.
    ///     static ALIGNED: &Aligned<[u8]> = &Aligned {
    ///         _align: [],
    ///         #[cfg(target_endian = "big")]
    ///         bytes: *include_bytes!("foo.bigendian.dfa"),
    ///         #[cfg(target_endian = "little")]
    ///         bytes: *include_bytes!("foo.littleendian.dfa"),
    ///     };
    ///     # };
    ///     # static ALIGNED: &Aligned<[u8]> = &Aligned {
    ///     #     _align: [],
    ///     #     bytes: [],
    ///     # };
    ///
    ///     struct Lazy(Cell<MaybeUninit<DFA>>);
    ///     // SAFETY: This is safe because DFA impls Sync.
    ///     unsafe impl Sync for Lazy {}
    ///
    ///     static INIT: Once = Once::new();
    ///     static DFA: Lazy = Lazy(Cell::new(MaybeUninit::uninit()));
    ///
    ///     INIT.call_once(|| {
    ///         let (dfa, _) = DFA::from_bytes(&ALIGNED.bytes)
    ///             .expect("serialized DFA should be valid");
    ///         // SAFETY: This is guaranteed to only execute once, and all
    ///         // we do with the pointer is write the DFA to it.
    ///         unsafe {
    ///             (*DFA.0.as_ptr()).as_mut_ptr().write(dfa);
    ///         }
    ///     });
    ///     // SAFETY: DFA is guaranteed to by initialized via INIT and is
    ///     // stored in static memory.
    ///     unsafe {
    ///         let dfa = (*DFA.0.as_ptr()).as_ptr();
    ///         std::mem::transmute::<*const DFA, &'static DFA>(dfa)
    ///     }
    /// }
    ///
    /// let dfa = get_foo();
    /// let expected = HalfMatch::new(0, 8);
    /// assert_eq!(Ok(Some(expected)), dfa.find_leftmost_fwd(b"foo12345"));
    /// ```
    ///
    /// Alternatively, consider using
    /// [`lazy_static`](https://crates.io/crates/lazy_static)
    /// or
    /// [`once_cell`](https://crates.io/crates/once_cell),
    /// which will guarantee safety for you. You will still need to use the
    /// `Aligned` trick above to force correct alignment, but this is safe to
    /// do and `from_bytes` will return an error if you get it wrong.
    pub fn from_bytes(
        mut slice: &'a [u8],
    ) -> Result<(DFA<&'a [S], &'a [u8], S>, usize), DeserializeError> {
        // SAFETY: This is safe because we validate both the transition table
        // and start state ID list below. If either validation fails, then we
        // return an error.
        let (dfa, nread) = unsafe { DFA::from_bytes_unchecked(slice)? };
        dfa.tt.validate()?;
        dfa.st.validate(&dfa.tt)?;
        dfa.ms.validate(&dfa)?;
        Ok((dfa, nread))
    }

    /// Deserialize a DFA with a specific state identifier representation in
    /// constant time by omitting the verification of the validity of the
    /// transition table.
    ///
    /// This is just like
    /// [`from_bytes`](struct.DFA.html#method.from_bytes),
    /// except it can potentially return a DFA that exhibits undefined behavior
    /// if its transition table contains invalid state identifiers.
    ///
    /// This routine is useful if you need to deserialize a DFA cheaply
    /// and cannot afford the transition table validation performed by
    /// `from_bytes`.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, HalfMatch, dense::DFA};
    ///
    /// let initial = DFA::new("foo[0-9]+")?;
    /// let (bytes, _) = initial.to_sized::<u16>()?.to_bytes_native_endian();
    /// // SAFETY: This is guaranteed to be safe since the bytes given come
    /// // directly from a compatible serialization routine.
    /// let dfa: DFA<&[u16], &[u8], u16> = unsafe {
    ///     DFA::from_bytes_unchecked(&bytes)?.0
    /// };
    ///
    /// let expected = HalfMatch::new(0, 8);
    /// assert_eq!(Some(expected), dfa.find_leftmost_fwd(b"foo12345")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub unsafe fn from_bytes_unchecked(
        mut slice: &'a [u8],
    ) -> Result<(DFA<&'a [S], &'a [u8], S>, usize), DeserializeError> {
        let mut nr = 0;

        nr += bytes::skip_initial_padding(slice);
        bytes::check_alignment::<S>(&slice[nr..])?;
        nr += bytes::read_label(&slice[nr..], LABEL)?;
        nr += bytes::read_endianness_check(&slice[nr..])?;
        nr += bytes::read_version(&slice[nr..], VERSION)?;
        nr += bytes::read_state_size::<S>(&slice[nr..])?;

        let _unused = bytes::try_read_u64(&slice[nr..], "unused space")?;
        nr += 8;

        let (tt, nread) = TransitionTable::from_bytes_unchecked(&slice[nr..])?;
        nr += nread;

        let (st, nread) = StartTable::from_bytes_unchecked(&slice[nr..])?;
        nr += nread;

        let (ms, nread) = MatchStates::from_bytes_unchecked(&slice[nr..])?;
        nr += nread;

        let (special, nread) = Special::from_bytes(&slice[nr..])?;
        nr += nread;
        special.validate_state_count(tt.count, tt.stride2)?;

        let (accels, nread) = Accels::from_bytes(&slice[nr..])?;
        nr += nread;

        Ok((DFA { tt, st, ms, special, accels }, nr))
    }

    /// The implementation of the public `write_to` serialization methods,
    /// which is generic over endianness.
    ///
    /// This is defined only for &[S] to reduce binary size/compilation time.
    fn write_to<E: Endian>(
        &self,
        dst: &mut [u8],
    ) -> Result<usize, SerializeError> {
        let mut nw = 0;
        nw += bytes::write_label(LABEL, &mut dst[nw..])?;
        nw += bytes::write_endianness_check::<E>(&mut dst[nw..])?;
        nw += bytes::write_version::<E>(VERSION, &mut dst[nw..])?;
        nw += bytes::write_state_size::<E, S>(&mut dst[nw..])?;
        nw += {
            // Currently unused, intended for future flexibility
            E::write_u64(0, &mut dst[nw..]);
            8
        };
        nw += self.tt.as_ref().write_to::<E>(&mut dst[nw..])?;
        nw += self.st.write_to::<E>(&mut dst[nw..])?;
        nw += self.ms.write_to::<E>(&mut dst[nw..])?;
        nw += self.special.write_to::<E>(&mut dst[nw..])?;
        nw += self.accels.write_to::<E>(&mut dst[nw..])?;
        Ok(nw)
    }
}

/// The following methods implement mutable routines on the internal
/// representation of a DFA. As such, we must fix the first type parameter to
/// a `Vec<S>` since a generic `T: AsRef<[S]>` does not permit mutation. We
/// can get away with this because these methods are internal to the crate and
/// are exclusively used during construction of the DFA.
#[cfg(feature = "std")]
impl<S: StateID> OwnedDFA<S> {
    /// Add a start state of this DFA.
    pub(crate) fn set_start_state(
        &mut self,
        index: Start,
        pattern_id: Option<PatternID>,
        id: S,
    ) {
        assert!(self.tt.is_valid(id), "invalid start state");
        self.st.set_start(index, pattern_id, id);
    }

    /// Add the given transition to this DFA. Both the `from` and `to` states
    /// must already exist.
    pub(crate) fn add_transition(&mut self, from: S, byte: Byte, to: S) {
        self.tt.set(from, byte, to);
    }

    /// An an empty state (a state where all transitions lead to a dead state)
    /// and return its identifier. The identifier returned is guaranteed to
    /// not point to any other existing state.
    ///
    /// If adding a state would exhaust the state identifier space (given by
    /// `S`), then this returns an error. In practice, this means that the
    /// state identifier representation chosen is too small.
    pub(crate) fn add_empty_state(&mut self) -> Result<S, Error> {
        self.tt.add_empty_state()
    }

    /// Swap the two states given in the transition table.
    ///
    /// This routine does not do anything to check the correctness of this
    /// swap. Callers must ensure that other states pointing to id1 and id2 are
    /// updated appropriately.
    pub(crate) fn swap_states(&mut self, id1: S, id2: S) {
        self.tt.swap(id1, id2);
    }

    /// Truncate the states in this DFA to the given count.
    ///
    /// This routine does not do anything to check the correctness of this
    /// truncation. Callers must ensure that other states pointing to truncated
    /// states are updated appropriately.
    pub(crate) fn truncate_states(&mut self, count: usize) {
        self.tt.truncate(count);
    }

    /// Return a mutable representation of the state corresponding to the given
    /// id. This is useful for implementing routines that manipulate DFA states
    /// (e.g., swapping states).
    pub(crate) fn state_mut(&mut self, id: S) -> StateMut<'_, S> {
        self.tt.state_mut(id)
    }

    /// Minimize this DFA in place using Hopcroft's algorithm.
    pub(crate) fn minimize(&mut self) {
        Minimizer::new(self).run();
    }

    /// Updates the match state pattern ID map to use the one provided.
    pub(crate) fn set_pattern_map(
        &mut self,
        map: &BTreeMap<S, Vec<PatternID>>,
    ) {
        self.ms = self.ms.new_with_map(map);
    }

    /// Find states that has a small number of non-loop transitions and mark
    /// them as candidates for acceleration during search.
    pub(crate) fn accelerate(&mut self) {
        // dead and quit states can never be accelerated.
        if self.tt.count <= 2 {
            return;
        }

        // Go through every state and record their accelerator, if possible.
        let mut accels = BTreeMap::new();
        // Count the number of accelerated match, start and non-match/start
        // states.
        let (mut cmatch, mut cstart, mut cnormal) = (0, 0, 0);
        for state in self.states() {
            if let Some(accel) = state.accelerate(self.byte_classes()) {
                accels.insert(state.id(), accel);
                if self.is_match_state(state.id()) {
                    cmatch += 1;
                } else if self.is_start_state(state.id()) {
                    cstart += 1;
                } else {
                    assert!(!self.is_dead_state(state.id()));
                    assert!(!self.is_quit_state(state.id()));
                    cnormal += 1;
                }
            }
        }
        // If no states were able to be accelerated, then we're done.
        if accels.is_empty() {
            return;
        }
        let original_accels_len = accels.len();

        // A remapper keeps track of state ID changes. Once we're done
        // shuffling, the remapper is used to rewrite all transitions in the
        // DFA based on the new positions of states.
        let mut remapper = Remapper::from_dfa(self);

        // As we swap states, if they are match states, we need to swap their
        // pattern ID lists too (for multi-regexes). We do this by converting
        // the lists to an easily swappable map, and then convert back to
        // MatchStates once we're done.
        let mut new_matches = self.ms.to_map(self);

        // There is at least one state that gets accelerated, so these are
        // guaranteed to get set to sensible values below.
        self.special.min_accel = S::from_usize(S::max_id());
        self.special.max_accel = dead_id();
        let mut update_special_accel =
            |special: &mut Special<S>, accel_id: S| {
                special.min_accel = cmp::min(special.min_accel, accel_id);
                special.max_accel = cmp::max(special.max_accel, accel_id);
            };

        // Start by shuffling match states. Any match states that are
        // accelerated get moved to the end of the match state range.
        if cmatch > 0 && self.special.matches() {
            // N.B. special.{min,max}_match do not need updating, since the
            // range/number of match states does not change. Only the ordering
            // of match states may change.
            let mut next_id = self.special.max_match;
            let mut cur_id = next_id;
            while cur_id >= self.special.min_match {
                if let Some(accel) = accels.remove(&cur_id) {
                    accels.insert(next_id, accel);
                    update_special_accel(&mut self.special, next_id);

                    // No need to do any actual swapping for equivalent IDs.
                    if cur_id != next_id {
                        remapper.swap(self, cur_id, next_id);

                        // Swap pattern IDs for match states.
                        let cur_pids = new_matches.remove(&cur_id).unwrap();
                        let next_pids = new_matches.remove(&next_id).unwrap();
                        new_matches.insert(cur_id, next_pids);
                        new_matches.insert(next_id, cur_pids);
                    }
                    next_id = self.tt.prev_state_id(next_id);
                }
                cur_id = self.tt.prev_state_id(cur_id);
            }
        }

        // This is where it gets tricky. Without acceleration, start states
        // normally come right after match states. But we want accelerated
        // states to be a single contiguous range (to make it very fast
        // to determine whether a state *is* accelerated), while also keeping
        // match and starting states as contiguous ranges for the same reason.
        // So what we do here is shuffle states such that it looks like this:
        //
        //     DQMMMMAAAAASSSSSSNNNNNNN
        //         |         |
        //         |---------|
        //      accelerated states
        //
        // Where:
        //   D - dead state
        //   Q - quit state
        //   M - match state (may be accelerated)
        //   A - normal state that is accelerated
        //   S - start state (may be accelerated)
        //   N - normal state that is NOT accelerated
        //
        // We implement this by shuffling states, which is done by a sequence
        // of pairwise swaps. We start by looking at all normal states to be
        // accelerated. When we find one, we swap it with the earliest starting
        // state, and then swap that with the earliest normal state. This
        // preserves the contiguous property.
        //
        // Once we're done looking for accelerated normal states, now we look
        // for accelerated starting states by moving them to the beginning
        // of the starting state range (just like we moved accelerated match
        // states to the end of the matching state range).
        //
        // For a more detailed/different perspective on this, see the docs
        // in dfa/special.rs.
        if cnormal > 0 {
            // our next available starting and normal states for swapping.
            let mut next_start_id = self.special.min_start;
            let mut cur_id = self.from_index(self.tt.count - 1);
            // This is guaranteed to exist since cnormal > 0.
            let mut next_norm_id =
                self.tt.next_state_id(self.special.max_start);
            while cur_id >= next_norm_id {
                if let Some(accel) = accels.remove(&cur_id) {
                    remapper.swap(self, next_start_id, cur_id);
                    remapper.swap(self, next_norm_id, cur_id);
                    // Keep our accelerator map updated with new IDs if the
                    // states we swapped were also accelerated.
                    if let Some(accel2) = accels.remove(&next_norm_id) {
                        accels.insert(cur_id, accel2);
                    }
                    if let Some(accel2) = accels.remove(&next_start_id) {
                        accels.insert(next_norm_id, accel2);
                    }
                    accels.insert(next_start_id, accel);
                    update_special_accel(&mut self.special, next_start_id);
                    // Our start range shifts one to the right now.
                    self.special.min_start =
                        self.tt.next_state_id(self.special.min_start);
                    self.special.max_start =
                        self.tt.next_state_id(self.special.max_start);
                    next_start_id = self.tt.next_state_id(next_start_id);
                    next_norm_id = self.tt.next_state_id(next_norm_id);
                }
                // This is pretty tricky, but if our 'next_norm_id' state also
                // happened to be accelerated, then the result is that it is
                // now in the position of cur_id, so we need to consider it
                // again. This loop is still guaranteed to terminate though,
                // because when accels contains cur_id, we're guaranteed to
                // increment next_norm_id even if cur_id remains unchanged.
                if !accels.contains_key(&cur_id) {
                    cur_id = self.tt.prev_state_id(cur_id);
                }
            }
        }
        // Just like we did for match states, but we want to move accelerated
        // start states to the beginning of the range instead of the end.
        if cstart > 0 {
            // N.B. special.{min,max}_start do not need updating, since the
            // range/number of start states does not change at this point. Only
            // the ordering of start states may change.
            let mut next_id = self.special.min_start;
            let mut cur_id = next_id;
            while cur_id <= self.special.max_start {
                if let Some(accel) = accels.remove(&cur_id) {
                    remapper.swap(self, cur_id, next_id);
                    accels.insert(next_id, accel);
                    update_special_accel(&mut self.special, next_id);
                    next_id = self.tt.next_state_id(next_id);
                }
                cur_id = self.tt.next_state_id(cur_id);
            }
        }

        // Remap all transitions in our DFA and assert some things.
        remapper.remap(self);
        self.set_pattern_map(&new_matches);
        self.special.set_max();
        self.special.validate().expect("special state ranges should validate");
        self.special
            .validate_state_count(self.tt.count, self.stride2())
            .expect(
                "special state ranges should be consistent with state count",
            );
        assert_eq!(
            self.special.accel_len(self.stride()),
            // We record the number of accelerated states initially detected
            // since the accels map is itself mutated in the process above.
            // If mutated incorrectly, its size may change, and thus can't be
            // trusted as a source of truth of how many accelerated states we
            // expected there to be.
            original_accels_len,
            "mismatch with expected number of accelerated states",
        );

        // And finally record our accelerators. We kept our accels map updated
        // as we shuffled states above, so the accelerators should now
        // correspond to a contiguous range in the state ID space. (Which we
        // assert.)
        let mut prev: Option<S> = None;
        for (id, accel) in accels {
            assert!(prev.map_or(true, |p| self.tt.next_state_id(p) == id));
            prev = Some(id);
            self.accels.add(accel);
        }
    }

    /// Shuffle the states in this DFA so that starting states and match
    /// states are contiguous.
    ///
    /// See dfa/special.rs for more details.
    pub(crate) fn shuffle(
        &mut self,
        mut matches: BTreeMap<S, Vec<PatternID>>,
    ) {
        // The determinizer always adds a quit state and it is always second.
        self.special.quit_id = self.from_index(1);
        // If all we have are the dead and quit states, then we're done and
        // the DFA will never produce a match.
        if self.tt.count <= 2 {
            self.special.set_max();
            return;
        }

        // Collect all our start states into a convenient set and confirm there
        // is no overlap with match states. In the classicl DFA construction,
        // start states can be match states. But because of look-around, we
        // delay all matches by a byte, which prevents start states from being
        // match states.
        let mut is_start: BTreeSet<S> = BTreeSet::new();
        for (start_id, _, _) in self.starts() {
            // While there's nothing theoretically wrong with setting a start
            // state to a dead ID (indeed, it could be an optimization!), the
            // shuffling code below assumes that start states aren't dead. If
            // this assumption is violated, the dead state could be shuffled
            // to a new location, which must never happen. So if we do want
            // to allow start states to be dead, then this assert should be
            // removed and the code below fixed.
            //
            // N.B. Minimization can cause start states to be dead, but that
            // happens after states are shuffled, so it's OK.
            assert_ne!(start_id, dead_id(), "start state cannot be dead");
            assert!(
                !matches.contains_key(&start_id),
                "{:?} is both a start and a match state, which is not allowed",
                start_id,
            );
            is_start.insert(start_id);
        }

        // We implement shuffling by a sequence of pairwise swaps of states.
        // Since we have a number of things referencing states via their
        // IDs and swapping them changes their IDs, we need to record every
        // swap we make so that we can remap IDs. The remapper handles this
        // book-keeping for us.
        let mut remapper = Remapper::from_dfa(self);

        // Shuffle matching states.
        if matches.is_empty() {
            self.special.min_match = dead_id();
            self.special.max_match = dead_id();
        } else {
            // The determinizer guarantees that the first two states are the
            // dead and quit states, respectively. We want our match states to
            // come right after quit.
            let mut next_id = self.from_index(2);
            let mut new_matches = BTreeMap::new();
            self.special.min_match = next_id;
            for (id, pids) in matches {
                remapper.swap(self, next_id, id);
                new_matches.insert(next_id, pids);
                // If we swapped a start state, then update our set.
                if is_start.contains(&next_id) {
                    is_start.remove(&next_id);
                    is_start.insert(id);
                }
                next_id = self.tt.next_state_id(next_id);
            }
            matches = new_matches;
            self.special.max_match = cmp::max(
                self.special.min_match,
                self.tt.prev_state_id(next_id),
            );
        }

        // Shuffle starting states.
        {
            let mut next_id = self.from_index(2);
            if self.special.matches() {
                next_id = self.tt.next_state_id(self.special.max_match);
            }
            self.special.min_start = next_id;
            for id in is_start {
                remapper.swap(self, next_id, id);
                next_id = self.tt.next_state_id(next_id);
            }
            self.special.max_start = cmp::max(
                self.special.min_start,
                self.tt.prev_state_id(next_id),
            );
        }

        // Finally remap all transitions in our DFA.
        remapper.remap(self);
        self.set_pattern_map(&matches);
        self.special.set_max();
        self.special.validate().expect("special state ranges should validate");
        self.special
            .validate_state_count(self.tt.count, self.stride2())
            .expect(
                "special state ranges should be consistent with state count",
            );
    }
}

/// A variety of generic internal methods for accessing DFA internals.
impl<T: AsRef<[S]>, A: AsRef<[u8]>, S: StateID> DFA<T, A, S> {
    /// Return the byte classes used by this DFA.
    pub(crate) fn byte_classes(&self) -> &ByteClasses {
        &self.tt.classes
    }

    /// Return the info about special states.
    pub(crate) fn special(&self) -> &Special<S> {
        &self.special
    }

    /// Return the info about special states as a mutable borrow.
    pub(crate) fn special_mut(&mut self) -> &mut Special<S> {
        &mut self.special
    }

    /// Returns an iterator over all states in this DFA.
    ///
    /// This iterator yields a tuple for each state. The first element of the
    /// tuple corresponds to a state's identifier, and the second element
    /// corresponds to the state itself (comprised of its transitions).
    pub(crate) fn states(&self) -> StateIter<'_, T, S> {
        self.tt.states()
    }

    /// Return the total number of states in this DFA. Every DFA has at least
    /// 1 state, even the empty DFA.
    pub(crate) fn state_count(&self) -> usize {
        self.tt.count
    }

    /// Return an iterator over all pattern IDs for the given match state.
    ///
    /// If the given state is not a match state, then this panics.
    pub(crate) fn match_pattern_ids(&self, id: S) -> PatternIDIter {
        assert!(self.is_match_state(id));
        self.ms.match_pattern_ids(self.match_index(id))
    }

    /// Return the total number of pattern IDs for the given match state.
    ///
    /// If the given state is not a match state, then this panics.
    pub(crate) fn match_pattern_len(&self, id: S) -> usize {
        assert!(self.is_match_state(id));
        self.ms.pattern_len(self.match_index(id))
    }

    /// Returns the total number of patterns matched by this DFA.
    pub(crate) fn pattern_count(&self) -> usize {
        self.ms.patterns
    }

    /// Returns a map from match state ID to a list of pattern IDs that match
    /// in that state.
    pub(crate) fn pattern_map(&self) -> BTreeMap<S, Vec<PatternID>> {
        self.ms.to_map(self)
    }

    /// Returns the ID of the quit state for this DFA.
    pub(crate) fn quit_id(&self) -> S {
        self.from_index(1)
    }

    /// Convert the given state identifier to the state's index. The state's
    /// index corresponds to the position in which it appears in the transition
    /// table. When a DFA is NOT premultiplied, then a state's identifier is
    /// also its index. When a DFA is premultiplied, then a state's identifier
    /// is equal to `index * alphabet_len`. This routine reverses that.
    pub(crate) fn to_index(&self, id: S) -> usize {
        self.tt.to_index(id)
    }

    /// Convert an index to a state (in the range 0..count) to an actual state
    /// identifier.
    ///
    /// This is useful when using a `Vec<T>` as an efficient map keyed by state
    /// to some other information (such as a remapped state ID).
    pub(crate) fn from_index(&self, index: usize) -> S {
        self.tt.from_index(index)
    }

    /// Return the table of state IDs for this DFA's start states.
    pub(crate) fn starts(&self) -> StartStateIter<'_, S> {
        self.st.iter()
    }

    /// Returns the index of the match state for the given ID. If the
    /// given ID does not correspond to a match state, then this may
    /// panic or produce an incorrect result.
    fn match_index(&self, id: S) -> usize {
        // This is one of the places where we rely on the fact that match
        // states are contiguous in the transition table. Namely, that the
        // first match state ID always corresponds to dfa.special.min_start.
        // From there, since we know the stride, we can compute the overall
        // index of any match state given the match state's ID.
        let min = self.special().min_match.as_usize();
        self.to_index(S::from_usize(id.as_usize() - min))
    }

    /// Returns the index of the accelerator state for the given ID. If the
    /// given ID does not correspond to an accelerator state, then this may
    /// panic or produce an incorrect result.
    fn accelerator_index(&self, id: S) -> usize {
        let min = self.special().min_accel.as_usize();
        self.to_index(S::from_usize(id.as_usize() - min))
    }

    /// Return the accelerators for this DFA.
    fn accels(&self) -> Accels<&[u8]> {
        self.accels.as_ref()
    }

    /// Return this DFA's transition table as a slice.
    fn trans(&self) -> &[S] {
        self.tt.table()
    }
}

#[cfg(feature = "std")]
impl<T: AsRef<[S]>, A: AsRef<[u8]>, S: StateID> fmt::Debug for DFA<T, A, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "dense::DFA(")?;
        for state in self.states() {
            fmt_state_indicator(f, self, state.id())?;
            let id = if f.alternate() {
                state.id().as_usize()
            } else {
                self.to_index(state.id())
            };
            write!(f, "{:06}: ", id)?;
            state.fmt(f)?;
            write!(f, "\n")?;
        }
        writeln!(f, "")?;
        for (i, (start_id, sty, pid)) in self.starts().enumerate() {
            let id = if f.alternate() {
                start_id.as_usize()
            } else {
                self.to_index(start_id)
            };
            if i % self.st.stride == 0 {
                let group = match pid {
                    None => "ALL".to_string(),
                    Some(pid) => format!("pattern: {}", pid),
                };
                writeln!(f, "START-GROUP({})", group)?;
            }
            writeln!(f, "  {:?} => {:06}", sty, id)?;
        }
        if self.pattern_count() > 1 {
            writeln!(f, "")?;
            for i in 0..self.ms.count() {
                let id = self.ms.match_state_id(self, i);
                let id = if f.alternate() {
                    id.as_usize()
                } else {
                    self.to_index(id)
                };
                write!(f, "MATCH({:06}): ", id)?;
                for (i, pid) in self.ms.match_pattern_ids(i).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", pid)?;
                }
                writeln!(f, "")?;
            }
        }
        writeln!(f, "state count: {}", self.state_count())?;
        writeln!(f, "pattern count: {}", self.pattern_count())?;
        writeln!(f, ")")?;
        Ok(())
    }
}

unsafe impl<T: AsRef<[S]>, A: AsRef<[u8]>, S: StateID> Automaton
    for DFA<T, A, S>
{
    type ID = S;

    #[inline]
    fn is_special_state(&self, id: S) -> bool {
        self.special.is_special_state(id)
    }

    #[inline]
    fn is_dead_state(&self, id: S) -> bool {
        self.special.is_dead_state(id)
    }

    #[inline]
    fn is_quit_state(&self, id: S) -> bool {
        self.special.is_quit_state(id)
    }

    #[inline]
    fn is_match_state(&self, id: S) -> bool {
        self.special.is_match_state(id)
    }

    #[inline]
    fn is_start_state(&self, id: S) -> bool {
        self.special.is_start_state(id)
    }

    #[inline]
    fn is_accel_state(&self, id: S) -> bool {
        self.special.is_accel_state(id)
    }

    #[inline]
    fn next_state(&self, current: S, input: u8) -> S {
        let input = self.byte_classes().get(input);
        let o = current.as_usize() + input as usize;
        self.trans()[o]
    }

    #[inline]
    unsafe fn next_state_unchecked(&self, current: S, input: u8) -> S {
        let input = self.byte_classes().get_unchecked(input);
        let o = current.as_usize() + input as usize;
        *self.trans().get_unchecked(o)
    }

    #[inline]
    fn next_eof_state(&self, current: S) -> S {
        let eof = self.byte_classes().eof().as_usize();
        let o = current.as_usize() + eof;
        self.trans()[o]
    }

    #[inline]
    fn patterns(&self) -> usize {
        self.ms.patterns
    }

    #[inline]
    fn match_count(&self, id: Self::ID) -> usize {
        self.match_pattern_len(id)
    }

    #[inline]
    fn match_pattern(&self, id: Self::ID, match_index: usize) -> PatternID {
        // This is an optimization for the very common case of a DFA with a
        // single pattern. This conditional avoids a somewhat more costly path
        // that finds the pattern ID from the state machine, which requires
        // a bit of slicing/pointer-chasing. This optimization tends to only
        // matter when matches are frequent.
        if self.ms.patterns == 1 {
            assert_eq!(match_index, 0);
            return 0;
        }
        let state_index = self.match_index(id);
        self.ms.pattern_id(state_index, match_index)
    }

    #[inline]
    fn start_state_forward(
        &self,
        pattern_id: Option<PatternID>,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> S {
        let index = Start::from_position_fwd(bytes, start, end);
        self.st.start(index, pattern_id)
    }

    #[inline]
    fn start_state_reverse(
        &self,
        pattern_id: Option<PatternID>,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> S {
        let index = Start::from_position_rev(bytes, start, end);
        self.st.start(index, pattern_id)
    }

    fn accelerator(&self, id: Self::ID) -> &[u8] {
        if !self.is_accel_state(id) {
            return &[];
        }
        self.accels.needles(self.accelerator_index(id))
    }
}

/// The transition table portion of a dense DFA.
///
/// The transition table is the core part of the DFA in that it describes how
/// to move from one state to another based on the input sequence observed.
#[derive(Clone)]
pub struct TransitionTable<T, S> {
    /// A contiguous region of memory representing the transition table in
    /// row-major order. The representation is dense. That is, every state
    /// has precisely the same number of transitions. The maximum number of
    /// transitions per state is 257 (256 for each possible byte value, plus 1
    /// for the special EOF transition). If a DFA has been instructed to use
    /// byte classes (the default), then the number of transitions is usually
    /// substantially fewer.
    ///
    /// In practice, T is either Vec<S> or &[S].
    table: T,
    /// A set of equivalence classes, where a single equivalence class
    /// represents a set of bytes that never discriminate between a match
    /// and a non-match in the DFA. Each equivalence class corresponds to a
    /// single character in this DFA's alphabet, where the maximum number of
    /// characters is 257 (each possible value of a byte plus the special
    /// EOF transition). Consequently, the number of equivalence classes
    /// corresponds to the number of transitions for each DFA state. Note
    /// though that the *space* used by each DFA state in the transition table
    /// may be larger. The total space used by each DFA state is known as the
    /// stride and is documented above.
    ///
    /// The only time the number of equivalence classes is fewer than 257 is
    /// if the DFA's kind uses byte classes which is the default. Equivalence
    /// classes should generally only be disabled when debugging, so that
    /// the transitions themselves aren't obscured. Disabling them has no
    /// other benefit, since the equivalence class map is always used while
    /// searching. In the vast majority of cases, the number of equivalence
    /// classes is substantially smaller than 257, particularly when large
    /// Unicode classes aren't used.
    classes: ByteClasses,
    /// The stride of each DFA state, expressed as a power-of-two exponent.
    ///
    /// The stride of a DFA corresponds to the total amount of space used by
    /// each DFA state in the transition table. This may be bigger than the
    /// size of a DFA's alphabet, since the stride is always the smallest
    /// power of two greater than or equal to the alphabet size.
    ///
    /// While this wastes space, this avoids the need for integer division
    /// to convert between premultiplied state IDs and their corresponding
    /// indices. Instead, we can use simple logical shifts.
    ///
    /// See the docs for the `stride2` method for more details.
    stride2: usize,
    /// The total number of states in the table. Note that a DFA always has at
    /// least one state---the dead state---even the empty DFA. In particular,
    /// the dead state always has ID 0 and is correspondingly always the first
    /// state. The dead state is never a match state.
    count: usize,
    /// The state ID representation. This is what's actually stored in `table`.
    _state_id: PhantomData<S>,
}

impl<'a, S: StateID> TransitionTable<&'a [S], S> {
    /// Deserialize a transition table starting at the beginning of `slice`.
    /// Upon success, return the total number of bytes read along with the
    /// transition table.
    ///
    /// If there was a problem deserializing any part of the transition table,
    /// then this returns an error. Notably, if the given slice does not have
    /// the same alignment as `S`, then this will return an error (among other
    /// possible errors).
    ///
    /// This is guaranteed to execute in constant time.
    ///
    /// # Safety
    ///
    /// This routine is not safe because it does not check the valdity of the
    /// transition table itself. In particular, the transition table can be
    /// quite large, so checking its validity can be somewhat expensive. An
    /// invalid transition table is not safe because other code may rely on the
    /// transition table being correct (such as explicit bounds check elision).
    /// Therefore, an invalid transition table can lead to undefined behavior.
    ///
    /// Callers that use this function must either pass on the safety invariant
    /// or guarantee that the bytes given contain a valid transition table.
    /// This guarantee is upheld by the bytes written by `write_to`.
    unsafe fn from_bytes_unchecked(
        mut slice: &'a [u8],
    ) -> Result<(TransitionTable<&'a [S], S>, usize), DeserializeError> {
        let count = bytes::try_read_u64_as_usize(slice, "state count")?;
        slice = &slice[8..];

        let stride2 = bytes::try_read_u64_as_usize(slice, "stride2")?;
        slice = &slice[8..];

        let (classes, nread) = ByteClasses::from_bytes(slice)?;
        slice = &slice[nread..];

        // The alphabet length (determined by the byte class map) cannot be
        // bigger than the stride (total space used by each DFA state).
        let stride = 1 << stride2;
        if classes.alphabet_len() > stride {
            return Err(DeserializeError::generic(
                "alphabet size cannot be bigger than transition table stride",
            ));
        }

        let table_bytes_len = (count << stride2) * core::mem::size_of::<S>();
        let nread = 8 + 8 + nread + table_bytes_len;
        if slice.len() < table_bytes_len {
            return Err(DeserializeError::buffer_too_small(
                "transition table",
            ));
        }
        bytes::check_alignment::<S>(slice)?;
        // SAFETY: Since S is always in {usize, u8, u16, u32, u64}, all we need
        // to do is ensure that we have the proper length and alignment. We've
        // checked both above, so the cast below is safe.
        //
        // N.B. This is the only not-safe code in this function, so we mark
        // it explicitly to call it out, even though it is technically
        // superfluous.
        let table = unsafe {
            core::slice::from_raw_parts(
                slice.as_ptr() as *const S,
                count << stride2,
            )
        };
        let tt = TransitionTable {
            table,
            classes,
            stride2,
            count,
            _state_id: PhantomData,
        };
        Ok((tt, nread))
    }

    /// Writes a serialized form of this transition table to the buffer given.
    /// If the buffer is too small, then an error is returned. To determine
    /// how big the buffer must be, use `write_to_len`.
    fn write_to<E: Endian>(
        &self,
        mut dst: &mut [u8],
    ) -> Result<usize, SerializeError> {
        let nwrite = self.write_to_len();
        if dst.len() < nwrite {
            return Err(SerializeError::buffer_too_small("transition table"));
        }
        dst = &mut dst[..nwrite];

        // write state count
        E::write_u64(self.count as u64, dst);
        dst = &mut dst[8..];

        // write state stride (as power of 2)
        E::write_u64(self.stride2 as u64, dst);
        dst = &mut dst[8..];

        // write byte class map
        let n = self.classes.write_to(dst)?;
        dst = &mut dst[n..];

        // write actual transitions
        dst.copy_from_slice(self.table_bytes());
        Ok(nwrite)
    }

    /// Returns the number of bytes the serialized form of this transition
    /// table will use.
    fn write_to_len(&self) -> usize {
        8   // state count
        + 8 // stride2
        + self.classes.write_to_len()
        + self.table_bytes().len()
    }

    /// Validates that every state ID in this transition table is valid.
    ///
    /// That is, every state ID can be used to correctly index a state in this
    /// table.
    fn validate(&self) -> Result<(), DeserializeError> {
        for state in self.states() {
            for (_, to) in state.transitions() {
                if !self.is_valid(to) {
                    return Err(DeserializeError::generic(
                        "found invalid state ID in transition table",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Converts this transition table from a table that uses S as its state
    /// ID representation to one that uses S2. If
    /// `size_of::<S2> >= size_of::<S>()`, then this always succeeds. If
    /// `size_of::<S2> < size_of::<S>()` and if S2 cannot represent every state
    /// ID in this transition table, then an error is returned.
    fn to_sized<S2: StateID>(
        &self,
    ) -> Result<TransitionTable<Vec<S2>, S2>, Error> {
        // Check that this table can fit into S2's representation.
        let last_state_id = (self.count - 1) << self.stride2;
        if last_state_id > S2::max_id() {
            return Err(Error::state_id_overflow(S2::max_id()));
        }
        let mut tt = TransitionTable {
            table: vec![dead_id::<S2>(); self.table().len()],
            classes: self.classes.clone(),
            count: self.count,
            stride2: self.stride2,
            _state_id: PhantomData,
        };
        for (i, id) in tt.table.iter_mut().enumerate() {
            // This is always correct since we've verified above that the
            // maximum state ID can fit into S2.
            *id = S2::from_usize(self.table()[i].as_usize());
        }
        Ok(tt)
    }
}

impl<S: StateID> TransitionTable<Vec<S>, S> {
    /// Create a minimal transition table with just two states: a dead state
    /// and a quit state. The alphabet length and stride of the transition
    /// table is determined by the given set of equivalence classes.
    ///
    /// This returns an error if the resulting transition table's state IDs
    /// cannot fit in `S`. (This can actually occur, e.g., if S = u8 and every
    /// equivalence class is a singleton.)
    fn minimal(
        classes: ByteClasses,
    ) -> Result<TransitionTable<Vec<S>, S>, Error> {
        let mut tt = TransitionTable {
            table: vec![],
            classes,
            stride2: classes.stride2(),
            count: 0,
            _state_id: PhantomData,
        };
        tt.add_empty_state()?; // dead state
        tt.add_empty_state()?; // quit state
        Ok(tt)
    }

    /// Set a transition in this table. Both the `from` and `to` states must
    /// already exist. `byte` should correspond to the transition out of `from`
    /// to set.
    fn set(&mut self, from: S, byte: Byte, to: S) {
        assert!(self.is_valid(from), "invalid 'from' state");
        assert!(self.is_valid(to), "invalid 'to' state");
        let class = match byte {
            Byte::U8(b) => self.classes.get(b) as usize,
            Byte::EOF(b) => b as usize,
        };
        self.table[from.as_usize() + class] = to;
    }

    /// Add an empty state (a state where all transitions lead to a dead state)
    /// and return its identifier. The identifier returned is guaranteed to
    /// not point to any other existing state.
    ///
    /// If adding a state would exhaust the state identifier space (determined
    /// by `S`), then this returns an error. In practice, this means that the
    /// state identifier representation chosen is too small.
    fn add_empty_state(&mut self) -> Result<S, Error> {
        let id = if self.count == 0 {
            S::from_usize(0)
        } else {
            // Normally, to get a fresh state identifier, we would just
            // take the index of the next state added to the transition
            // table. However, we actually perform an optimization here
            // that premultiplies state IDs by the stride, such that they
            // point immediately at the beginning of their transitions in
            // the transition table. This avoids an extra multiplication
            // instruction for state lookup at search time.
            //
            // Premultiplied identifiers means that instead of your matching
            // loop looking something like this:
            //
            //   state = dfa.start
            //   for byte in haystack:
            //       next = dfa.transitions[state * stride + byte]
            //       if dfa.is_match(next):
            //           return true
            //   return false
            //
            // it can instead look like this:
            //
            //   state = dfa.start
            //   for byte in haystack:
            //       next = dfa.transitions[state + byte]
            //       if dfa.is_match(next):
            //           return true
            //   return false
            //
            // In other words, we save a multiplication instruction in the
            // critical path. This turns out to be a decent performance win.
            // The cost of using premultiplied state ids is that they can
            // require a bigger state id representation. (And they also make
            // the code a bit more complex, especially during minimization and
            // when reshuffling states, as one needs to convert back and forth
            // between state IDs and state indices.)
            let next = match self.count.checked_shl(self.stride2 as u32) {
                Some(next) => next,
                None => return Err(Error::state_id_overflow(std::usize::MAX)),
            };
            if next > S::max_id() {
                return Err(Error::state_id_overflow(S::max_id()));
            }
            S::from_usize(next)
        };
        self.table.extend(iter::repeat(dead_id::<S>()).take(self.stride()));
        // This should never panic, since count is a usize. The transition
        // table size would have run out of room long ago.
        self.count = self.count.checked_add(1).unwrap();
        Ok(id)
    }

    /// Swap the two states given in this transition table.
    ///
    /// This routine does not do anything to check the correctness of this
    /// swap. Callers must ensure that other states pointing to id1 and id2 are
    /// updated appropriately.
    ///
    /// Both id1 and id2 must point to valid states.
    fn swap(&mut self, id1: S, id2: S) {
        assert!(self.is_valid(id1), "invalid 'id1' state: {:?}", id1);
        assert!(self.is_valid(id2), "invalid 'id2' state: {:?}", id2);
        for b in 0..self.classes.alphabet_len() {
            self.table.swap(id1.as_usize() + b, id2.as_usize() + b);
        }
    }

    /// Truncate the states in this transition table to the given count.
    ///
    /// This routine does not do anything to check the correctness of this
    /// truncation. Callers must ensure that other states pointing to truncated
    /// states are updated appropriately.
    fn truncate(&mut self, count: usize) {
        self.table.truncate(count << self.stride2);
        self.count = count;
    }

    /// Return a mutable representation of the state corresponding to the given
    /// id. This is useful for implementing routines that manipulate DFA states
    /// (e.g., swapping states).
    fn state_mut(&mut self, id: S) -> StateMut<'_, S> {
        let alphabet_len = self.alphabet_len();
        let i = id.as_usize();
        StateMut {
            id,
            stride2: self.stride2,
            transitions: &mut self.table_mut()[i..i + alphabet_len],
        }
    }
}

impl<T: AsRef<[S]>, S: StateID> TransitionTable<T, S> {
    /// Converts this transition table to a borrowed value.
    fn as_ref(&self) -> TransitionTable<&'_ [S], S> {
        TransitionTable {
            table: self.table(),
            classes: self.classes.clone(),
            count: self.count,
            stride2: self.stride2,
            _state_id: self._state_id,
        }
    }

    /// Converts this transition table to an owned value.
    fn to_owned(&self) -> TransitionTable<Vec<S>, S> {
        TransitionTable {
            table: self.table().to_vec(),
            classes: self.classes.clone(),
            count: self.count,
            stride2: self.stride2,
            _state_id: self._state_id,
        }
    }

    /// Return the state for the given ID. If the given ID is not valid, then
    /// this panics.
    fn state(&self, id: S) -> State<'_, S> {
        assert!(self.is_valid(id));

        let i = id.as_usize();
        State {
            id,
            stride2: self.stride2,
            transitions: &self.table()[i..i + self.alphabet_len()],
        }
    }

    /// Returns an iterator over all states in this transition table.
    ///
    /// This iterator yields a tuple for each state. The first element of the
    /// tuple corresponds to a state's identifier, and the second element
    /// corresponds to the state itself (comprised of its transitions).
    fn states(&self) -> StateIter<'_, T, S> {
        StateIter {
            tt: self,
            it: self.table().chunks(self.stride()).enumerate(),
        }
    }

    /// Convert a state identifier to an index to a state (in the range
    /// 0..count).
    ///
    /// This is useful when using a `Vec<T>` as an efficient map keyed by state
    /// to some other information (such as a remapped state ID).
    fn to_index(&self, id: S) -> usize {
        id.as_usize() >> self.stride2
    }

    /// Convert an index to a state (in the range 0..count) to an actual state
    /// identifier.
    ///
    /// This is useful when using a `Vec<T>` as an efficient map keyed by state
    /// to some other information (such as a remapped state ID).
    fn from_index(&self, index: usize) -> S {
        S::from_usize(index << self.stride2)
    }

    /// Returns the state ID for the state immediately following the one given.
    ///
    /// This does not check whether the state ID returned is invalid. In fact,
    /// if the state ID given is the last state in this DFA, then the state ID
    /// returned is guaranteed to be invalid.
    fn next_state_id(&self, id: S) -> S {
        self.from_index(self.to_index(id).checked_add(1).unwrap())
    }

    /// Returns the state ID for the state immediately preceding the one given.
    ///
    /// If the dead ID given (which is zero), then this panics.
    fn prev_state_id(&self, id: S) -> S {
        self.from_index(self.to_index(id).checked_sub(1).unwrap())
    }

    /// Returns the table as a slice of state IDs.
    fn table(&self) -> &[S] {
        self.table.as_ref()
    }

    /// Returns the transition table in its raw byte representation.
    ///
    /// The length of the slice returned is always equivalent to
    /// `self.table().len() * self.state_size()`.
    ///
    /// This is generally only useful when serializing the transition table
    /// to raw bytes.
    fn table_bytes(&self) -> &[u8] {
        let table = self.table();
        // SAFETY: This is safe because S is guaranteed to be one of {usize,
        // u8, u16, u32, u64}, and because u8 always has a smaller alignment.
        unsafe {
            core::slice::from_raw_parts(
                table.as_ptr() as *const u8,
                table.len() * self.state_size(),
            )
        }
    }

    /// Returns the total stride for every state in this DFA. This corresponds
    /// to the total number of transitions used by each state in this DFA's
    /// transition table.
    fn stride(&self) -> usize {
        1 << self.stride2
    }

    /// Returns the total number of elements in the alphabet for this
    /// transition table. This is always less than or equal to `self.stride()`.
    /// It is only equal when the alphabet length is a power of 2. Otherwise,
    /// it is always strictly less.
    fn alphabet_len(&self) -> usize {
        self.classes.alphabet_len()
    }

    /// Returns true if and only if the given state ID is valid for this
    /// transition table. Validity in this context means that the given ID
    /// can be used to correct index a state with `self.stride()` transitions
    /// in this table.
    fn is_valid(&self, id: S) -> bool {
        let id = id.as_usize();
        id < self.table().len() && id % self.stride() == 0
    }

    /// Returns the size of the specific state ID representation, in bytes.
    ///
    /// This is always 1, 2, 4 or 8.
    fn state_size(&self) -> usize {
        core::mem::size_of::<S>()
    }

    /// Return the memory usage, in bytes, of this transition table.
    ///
    /// This does not include the size of a `TransitionTable` value itself.
    fn memory_usage(&self) -> usize {
        self.table_bytes().len()
    }
}

impl<T: AsMut<[S]>, S: StateID> TransitionTable<T, S> {
    /// Returns the table as a slice of state IDs.
    fn table_mut(&mut self) -> &mut [S] {
        self.table.as_mut()
    }
}

/// The set of all possible starting states in a DFA.
///
/// The set of starting states corresponds to the possible choices one can make
/// in terms of starting a DFA. That is, before following the first transition,
/// you first need to select the state that you start in.
///
/// Normally, a DFA converted from an NFA that has a single starting state
/// would itself just have one starting state. However, our support for look
/// around generally requires more starting states. The correct starting state
/// is chosen based on certain properties of the position at which we begin
/// our search.
///
/// Before listing those properties, we first must define two terms:
///
/// * `haystack` - The bytes to execute the search. The search always starts
///   at the beginning of `haystack` and ends before or at the end of
///   haystack`.
/// * `context` - The (possibly empty) bytes surrounding `haystack`. `haystack`
///   must be contained within `context` such that `context` is at least as big
///   as `haystack`.
///
/// This split is crucial for dealing with look-around. For example, consider
/// the context `foobarbaz`, the haystack `bar` and the regex `^bar$`. This
/// regex should _not_ match the haystack since `bar` does not appear at the
/// beginning of the input. Similarly, the regex `\Bbar\B` should match the
/// haystack because `bar` is not surrounded by word boundaries. But a search
/// that does not take context into account would not permit `\B` to match
/// since the beginning of any string matches a word boundary.
///
/// Thus, it follows that the starting state is chosen based on the following
/// criteria, derived from the position at which the search starts in the
/// `context` (corresponding to the start of `haystack`):
///
/// 1. If the search starts at the beginning of `context`, then the `Text`
///    start state is used. (Since `^` corresponds to
///    `hir::Anchor::StartText`.)
/// 2. If the search starts at a position immediately following a line
///    terminator, then the `Line` start state is used. (Since `(?m:^)`
///    corresponds to `hir::Anchor::StartLine`.)
/// 3. If the search starts at a position immediately following a byte
///    classified as a "word" character ([_0-9a-zA-Z]), then the `WordByte`
///    start state is used. (Since `(?-u:\b)` corresponds to a word boundary.)
/// 4. Otherwise, if the search starts at a position immediately following
///    a byte that is not classified as a "word" character ([^_0-9a-zA-Z]),
///    then the `NonWordByte` start state is used. (Since `(?-u:\B)`
///    corresponds to a not-word-boundary.)
///
/// To further complicate things, we also support constructing individual
/// anchored start states for each pattern in the DFA. (Which is required to
/// implement overlapping regexes correctly, but is also generally useful.)
/// Thus, when individual start states for each pattern is enabled, then the
/// total number of start states represented is 4 + (4 * #patterns), where the
/// 4 comes from each of the 4 possibilities above. The first 4 represents the
/// starting states for the entire DFA, which support searching for multiple
/// patterns simultaneously.
///
/// If individual start states are disabled, then this will only store 4
/// start states. Typically, individual start states are only enabled when
/// constructing the reverse DFA for regex matching. But they are also useful
/// for building DFAs that can search for a specific pattern or even to support
/// both anchored and unanchored searches with the same DFA.
///
/// Note though that while the start table always has either `4` or
/// `4 + (4 * #patterns)` starting state *ids*, the total number of states
/// might be considerably smaller. That is, many of the IDs may just be
/// duplicative. (For example, if a regex doesn't have a `\b` sub-pattern, then
/// there's no reason to generate a unique starting state for handling word
/// boundaries. Similarly for start/end anchors.)
#[derive(Clone)]
pub struct StartTable<T, S> {
    /// The initial start state IDs.
    ///
    /// In practice, T is either Vec<S> or &[S].
    ///
    /// The first `stride` (currently always 4) entries always correspond to
    /// the start states for the entire DFA. After that, there are
    /// `stride * patterns` state IDs, where `patterns` may be zero in the
    /// case of a DFA with no patterns or in the case where the DFA was built
    /// without enabling starting states for each pattern.
    table: T,
    /// The number of starting state IDs per pattern.
    stride: usize,
    /// The total number of patterns for which starting states are encoded.
    /// This may be zero for non-empty DFAs when the DFA was built without
    /// start states for each pattern.
    patterns: usize,
    /// The state ID representation. This is what's actually stored in `list`.
    _state_id: PhantomData<S>,
}

impl<S: StateID> StartTable<Vec<S>, S> {
    /// Create a valid set of start states all pointing to the dead state.
    ///
    /// When the corresponding DFA is constructed with start states for each
    /// pattern, then `patterns` should be the number of patterns. Otherwise,
    /// it should be zero.
    fn dead(patterns: usize) -> StartTable<Vec<S>, S> {
        let stride = Start::count();
        StartTable {
            table: vec![dead_id(); stride + (stride * patterns)],
            stride,
            patterns,
            _state_id: PhantomData,
        }
    }
}

impl<'a, S: StateID> StartTable<&'a [S], S> {
    /// Deserialize a table of start state IDs starting at the beginning of
    /// `slice`. Upon success, return the total number of bytes read along with
    /// the table of starting state IDs.
    ///
    /// If there was a problem deserializing any part of the starting IDs,
    /// then this returns an error. Notably, if the given slice does not have
    /// the same alignment as `S`, then this will return an error (among other
    /// possible errors).
    ///
    /// This is guaranteed to execute in constant time.
    ///
    /// # Safety
    ///
    /// This routine is not safe because it does not check the valdity of the
    /// starting state IDs themselves. In particular, the number of starting
    /// IDs can be of variable length, so it's possible that checking their
    /// validity cannot be done in constant time. An invalid starting state
    /// ID is not safe because other code may rely on the starting IDs being
    /// correct (such as explicit bounds check elision). Therefore, an invalid
    /// start ID can lead to undefined behavior.
    ///
    /// Callers that use this function must either pass on the safety invariant
    /// or guarantee that the bytes given contain valid starting state IDs.
    /// This guarantee is upheld by the bytes written by `write_to`.
    unsafe fn from_bytes_unchecked(
        mut slice: &'a [u8],
    ) -> Result<(StartTable<&'a [S], S>, usize), DeserializeError> {
        let stride =
            bytes::try_read_u64_as_usize(slice, "start table stride")?;
        slice = &slice[8..];
        let patterns =
            bytes::try_read_u64_as_usize(slice, "start table patterns")?;
        slice = &slice[8..];

        if stride != Start::count() {
            return Err(DeserializeError::generic(
                "invalid starting table stride",
            ));
        }
        if patterns > crate::pattern_limit() {
            return Err(DeserializeError::generic(
                "invalid number of patterns",
            ));
        }
        let pattern_table_size = match stride.checked_mul(patterns) {
            Some(x) => x,
            None => {
                return Err(DeserializeError::generic("invalid pattern count"))
            }
        };
        let count = match stride.checked_add(pattern_table_size) {
            Some(x) => x,
            None => {
                return Err(DeserializeError::generic(
                    "invalid pattern+stride",
                ))
            }
        };
        let table_bytes_len = count * core::mem::size_of::<S>();
        let nread = 16 + table_bytes_len;
        if slice.len() < table_bytes_len {
            return Err(DeserializeError::buffer_too_small("start ID table"));
        }
        bytes::check_alignment::<S>(slice)?;
        // SAFETY: Since S is always in {usize, u8, u16, u32, u64}, all we need
        // to do is ensure that we have the proper length and alignment. We've
        // checked both above, so the cast below is safe.
        //
        // N.B. This is the only not-safe code in this function, so we mark
        // it explicitly to call it out, even though it is technically
        // superfluous.
        let table = unsafe {
            core::slice::from_raw_parts(slice.as_ptr() as *const S, count)
        };
        let st =
            StartTable { table, stride, patterns, _state_id: PhantomData };
        Ok((st, nread))
    }
}

impl<T: AsRef<[S]>, S: StateID> StartTable<T, S> {
    /// Writes a serialized form of this start table to the buffer given. If
    /// the buffer is too small, then an error is returned. To determine how
    /// big the buffer must be, use `write_to_len`.
    fn write_to<E: Endian>(
        &self,
        mut dst: &mut [u8],
    ) -> Result<usize, SerializeError> {
        let nwrite = self.write_to_len();
        if dst.len() < nwrite {
            return Err(SerializeError::buffer_too_small(
                "starting table ids",
            ));
        }
        dst = &mut dst[..nwrite];

        // write stride
        E::write_u64(self.stride as u64, dst);
        dst = &mut dst[8..];
        // write pattern count
        E::write_u64(self.patterns as u64, dst);
        dst = &mut dst[8..];
        // write start IDs
        dst.copy_from_slice(self.table_bytes());
        Ok(nwrite)
    }

    /// Returns the number of bytes the serialized form of this start ID table
    /// will use.
    fn write_to_len(&self) -> usize {
        8 // stride
        + 8 // # patterns
        + self.table_bytes().len()
    }

    /// Validates that every state ID in this start table is valid by checking
    /// it against the given transition table (which must be for the same DFA).
    ///
    /// That is, every state ID can be used to correctly index a state.
    fn validate(
        &self,
        tt: &TransitionTable<T, S>,
    ) -> Result<(), DeserializeError> {
        for &id in self.table() {
            if !tt.is_valid(id) {
                return Err(DeserializeError::generic(
                    "found invalid starting state ID",
                ));
            }
        }
        Ok(())
    }

    /// Converts this start list to a borrowed value.
    fn as_ref(&self) -> StartTable<&'_ [S], S> {
        StartTable {
            table: self.table(),
            stride: self.stride,
            patterns: self.patterns,
            _state_id: self._state_id,
        }
    }

    /// Converts this start list to an owned value.
    fn to_owned(&self) -> StartTable<Vec<S>, S> {
        StartTable {
            table: self.table().to_vec(),
            stride: self.stride,
            patterns: self.patterns,
            _state_id: self._state_id,
        }
    }

    /// Converts this table of starting IDs from a list that uses S as its
    /// state ID representation to one that uses S2. If
    /// `size_of::<S2> >= size_of::<S>()`, then this always succeeds. If
    /// `size_of::<S2> < size_of::<S>()` and if S2 cannot represent every state
    /// ID in this list, then an error is returned.
    fn to_sized<S2: StateID>(&self) -> Result<StartTable<Vec<S2>, S2>, Error> {
        // Check that this list can fit into S2's representation.
        let max_state_id = match self.table().iter().cloned().max() {
            None => {
                return Ok(StartTable::dead(self.patterns));
            }
            Some(max_state_id) => max_state_id.as_usize(),
        };
        if max_state_id > S2::max_id() {
            return Err(Error::state_id_overflow(S2::max_id()));
        }
        let mut st = StartTable::dead(self.patterns);
        for (i, id) in st.table.iter_mut().enumerate() {
            // This is always correct since we've verified above that the
            // maximum state ID can fit into S2.
            *id = S2::from_usize(self.table()[i].as_usize());
        }
        Ok(st)
    }

    /// Return the start state for the given index and pattern ID. If the
    /// pattern ID is None, then the corresponding start state for the entire
    /// DFA is returned. If the pattern ID is not None, then the corresponding
    /// starting state for the given pattern is returned. If this start table
    /// does not have individual starting states for each pattern, then this
    /// panics.
    fn start(&self, index: Start, pattern_id: Option<PatternID>) -> S {
        let start_index = index.as_usize();
        let index = match pattern_id {
            None => start_index,
            Some(pid) => {
                self.stride + (self.stride * pid as usize) + start_index
            }
        };
        self.table()[index]
    }

    /// Returns an iterator over all start state IDs in this table.
    ///
    /// Each item is a triple of: start state ID, the start state type and the
    /// pattern ID (if any).
    fn iter(&self) -> StartStateIter<'_, S> {
        StartStateIter { st: self.as_ref(), i: 0 }
    }

    /// Returns the table as a slice of state IDs.
    fn table(&self) -> &[S] {
        self.table.as_ref()
    }

    /// Returns the table of start IDs as its raw byte representation.
    ///
    /// The length of the slice returned is always equivalent to
    /// `self.table().len() * self.state_size()`.
    ///
    /// This is generally only useful when serializing the starting IDs to raw
    /// bytes.
    fn table_bytes(&self) -> &[u8] {
        let table = self.table();
        // SAFETY: This is safe because S is guaranteed to be one of {usize,
        // u8, u16, u32, u64}, and because u8 always has a smaller or
        // equivalent alignment.
        unsafe {
            core::slice::from_raw_parts(
                table.as_ptr() as *const u8,
                table.len() * self.state_size(),
            )
        }
    }

    /// Returns the size of the specific state ID representation, in bytes.
    ///
    /// This is always 1, 2, 4 or 8.
    fn state_size(&self) -> usize {
        core::mem::size_of::<S>()
    }

    /// Return the memory usage, in bytes, of this start list.
    ///
    /// This does not include the size of a `StartList` value itself.
    fn memory_usage(&self) -> usize {
        self.table_bytes().len()
    }
}

impl<T: AsMut<[S]>, S: StateID> StartTable<T, S> {
    /// Set the start state for the given index and pattern.
    fn set_start(
        &mut self,
        index: Start,
        pattern_id: Option<PatternID>,
        id: S,
    ) {
        let start_index = index.as_usize();
        let index = match pattern_id {
            None => start_index,
            Some(pid) => {
                self.stride + (self.stride * pid as usize) + start_index
            }
        };
        self.table_mut()[index] = id;
    }

    /// Returns the table as a mutable slice of state IDs.
    fn table_mut(&mut self) -> &mut [S] {
        self.table.as_mut()
    }
}

/// An iterator over start state IDs.
///
/// This iterator yields a triple of start state ID, the start state type
/// and the pattern ID (if any). The pattern ID is None for start states
/// corresponding to the entire DFA and non-None for start states corresponding
/// to a specific pattern. The latter only occurs when the DFA is compiled with
/// start states for each pattern.
pub(crate) struct StartStateIter<'a, S> {
    st: StartTable<&'a [S], S>,
    i: usize,
}

impl<'a, S: StateID> Iterator for StartStateIter<'a, S> {
    type Item = (S, Start, Option<PatternID>);

    fn next(&mut self) -> Option<(S, Start, Option<PatternID>)> {
        let i = self.i;
        let table = self.st.table();
        if i >= table.len() {
            return None;
        }
        self.i += 1;

        // This unwrap is okay since the stride of any DFA must always match
        // the number of start state types.
        let start_type = Start::from_usize(i % self.st.stride).unwrap();
        let pid = if i < self.st.stride {
            None
        } else {
            Some(((i - self.st.stride) / self.st.stride) as u32)
        };
        Some((table[i], start_type, pid))
    }
}

/// This type represents that patterns that should be reported whenever a DFA
/// enters a match state. This structure exists to support DFAs that search for
/// matches for multiple regexes.
///
/// This structure relies on the fact that all match states in a DFA occur
/// contiguously in the DFA's transition table. (See dfa/special.rs for a more
/// detailed breakdown of the representation.) Namely, when a match occurs, we
/// know its state ID. Since we know the start and end of the contiguous region
/// of match states, we can use that to compute the position at which the match
/// state occurs. That in turn is used as an offset into this structure.
#[derive(Clone, Debug)]
struct MatchStates<T, A, S> {
    /// Slices is a flattened sequence of pairs, where each pair points to a
    /// sub-slice of pattern_ids. The first element of the pair is an offset
    /// into pattern_ids and the second element of the pair is the number
    /// of 32-bit pattern IDs starting at that position. That is, each pair
    /// corresponds to a single DFA match state and its corresponding match
    /// IDs. The number of pairs always corresponds to the number of distinct
    /// DFA match states.
    ///
    /// In practice, T is either Vec<S> or &[S], where S: StateID.
    ///
    /// It's a bit weird to use S for this since these aren't actually state
    /// IDs. And in fact, they don't have anything to do with state IDs. But
    /// we reuse the "state ID" abstraction because the state ID abstraction is
    /// really just an abstraction around pointer sized fields. For example, on
    /// a 16-bit target, S is guaranteed to be no bigger than a u16. And that's
    /// exactly what we want here: to store pointers into some other slice,
    /// which is all state IDs really are at the end of the day.
    slices: T,
    /// A flattened sequence of pattern IDs for each DFA match state. The only
    /// way to correctly read this sequence is indirectly via `slices`.
    ///
    /// In practice, T is either Vec<u8> or &[u8].
    pattern_ids: A,
    /// The total number of unique patterns represented by these match states.
    patterns: usize,
    /// The 'S' type parameter isn't explicitly used above, so we need to fake
    /// it.
    _state_id: PhantomData<S>,
}

impl<'a, S: StateID> MatchStates<&'a [S], &'a [u8], S> {
    unsafe fn from_bytes_unchecked(
        mut slice: &'a [u8],
    ) -> Result<(MatchStates<&'a [S], &'a [u8], S>, usize), DeserializeError>
    {
        let mut nread = 0;

        // Read the total number of match states.
        let count = bytes::try_read_u64_as_usize(slice, "match state count")?;
        nread += 8;
        slice = &slice[8..];

        // Read the slice start/length pairs.
        let slices_bytes_len = 2 * count * core::mem::size_of::<S>();
        if slice.len() < slices_bytes_len {
            return Err(DeserializeError::buffer_too_small(
                "match state slices",
            ));
        }
        bytes::check_alignment::<S>(slice)?;
        // SAFETY: Since S is always in {usize, u8, u16, u32, u64}, all we need
        // to do is ensure that we have the proper length and alignment. We've
        // checked both above, so the cast below is safe.
        //
        // N.B. This is the only not-safe code in this function, so we mark
        // it explicitly to call it out, even though it is technically
        // superfluous.
        let slices = unsafe {
            core::slice::from_raw_parts(slice.as_ptr() as *const S, 2 * count)
        };
        nread += slices_bytes_len;
        slice = &slice[slices_bytes_len..];

        // Read the total number of unique pattern IDs (which is always 1 more
        // than the maximum pattern ID).
        let patterns = bytes::try_read_u64_as_usize(slice, "pattern count")?;
        nread += 8;
        slice = &slice[8..];

        // Now read the pattern ID count. We don't need to store this
        // explicitly, but we need it to know how many pattern IDs to read.
        let idcount = bytes::try_read_u64_as_usize(slice, "pattern ID count")?;
        nread += 8;
        slice = &slice[8..];

        // Read the actual pattern IDs.
        let pattern_ids_len = idcount * 4; // each ID is a u32
        if slice.len() < pattern_ids_len {
            return Err(DeserializeError::buffer_too_small(
                "match pattern IDs",
            ));
        }
        let pattern_ids = &slice[..pattern_ids_len];
        nread += pattern_ids_len;
        slice = &slice[pattern_ids_len..];

        // And finally, make sure there are appropriate padding bytes.
        let pad = bytes::padding_len(pattern_ids.len());
        if slice.len() < pad {
            return Err(DeserializeError::buffer_too_small(
                "match pattern ID padding",
            ));
        }
        nread += pad;
        slice = &slice[pad..];

        let ms = MatchStates {
            slices,
            pattern_ids,
            patterns,
            _state_id: PhantomData,
        };
        Ok((ms, nread))
    }
}

impl<S: StateID> MatchStates<Vec<S>, Vec<u8>, S> {
    fn empty(pattern_count: usize) -> MatchStates<Vec<S>, Vec<u8>, S> {
        MatchStates {
            slices: vec![],
            pattern_ids: vec![],
            patterns: pattern_count,
            _state_id: PhantomData,
        }
    }

    fn new(
        matches: &BTreeMap<S, Vec<PatternID>>,
        pattern_count: usize,
    ) -> MatchStates<Vec<S>, Vec<u8>, S> {
        let mut m = MatchStates::empty(pattern_count);
        for (state_id, pids) in matches.iter() {
            let start = S::from_usize(m.pattern_ids.len());
            m.slices.push(start);
            let len = S::from_usize(pids.len());
            m.slices.push(len);
            for &pid in pids {
                m.pattern_ids.extend_from_slice(&pid.to_ne_bytes());
            }
        }
        m.patterns = pattern_count;
        m
    }

    fn new_with_map(
        &self,
        matches: &BTreeMap<S, Vec<PatternID>>,
    ) -> MatchStates<Vec<S>, Vec<u8>, S> {
        MatchStates::new(matches, self.patterns)
    }
}

impl<T: AsRef<[S]>, A: AsRef<[u8]>, S: StateID> MatchStates<T, A, S> {
    /// Writes a serialized form of these match states to the buffer given. If
    /// the buffer is too small, then an error is returned. To determine how
    /// big the buffer must be, use `write_to_len`.
    fn write_to<E: Endian>(
        &self,
        mut dst: &mut [u8],
    ) -> Result<usize, SerializeError> {
        let nwrite = self.write_to_len();
        if dst.len() < nwrite {
            return Err(SerializeError::buffer_too_small("match states"));
        }
        dst = &mut dst[..nwrite];

        // write state ID count
        E::write_u64(self.count() as u64, dst);
        dst = &mut dst[8..];

        // write slice offset pairs
        let slices = self.slices_bytes();
        dst[..slices.len()].copy_from_slice(slices);
        dst = &mut dst[slices.len()..];

        // write unique pattern ID count
        E::write_u64(self.patterns as u64, dst);
        dst = &mut dst[8..];

        // write pattern ID count
        E::write_u64(self.pattern_id_count() as u64, dst);
        dst = &mut dst[8..];

        // write pattern IDs
        dst[..self.pattern_ids().len()].copy_from_slice(self.pattern_ids());
        dst = &mut dst[self.pattern_ids().len()..];

        // ... and also write padding bytes, just so that we are S-aligned
        // everywhere.
        for _ in 0..bytes::padding_len(self.pattern_ids().len()) {
            dst[0] = 0;
            dst = &mut dst[1..];
        }
        Ok(nwrite)
    }

    /// Returns the number of bytes the serialized form of this transition
    /// table will use.
    fn write_to_len(&self) -> usize {
        8   // match state count
        + self.slices_bytes().len()
        + 8 // unique pattern ID count
        + 8 // pattern ID count
        + self.pattern_ids().len()
        + bytes::padding_len(self.pattern_ids().len())
    }

    /// Valides that the match state info is itself internally consistent and
    /// consistent with the recorded match state region in the given DFA.
    fn validate(&self, dfa: &DFA<T, A, S>) -> Result<(), DeserializeError> {
        if self.count() != dfa.special.match_len(dfa.stride()) {
            return Err(DeserializeError::generic(
                "match state count mismatch",
            ));
        }
        for si in 0..self.count() {
            let start = self.slices()[si * 2].as_usize();
            let len = self.slices()[si * 2 + 1].as_usize();
            if start >= self.pattern_ids().len() {
                return Err(DeserializeError::generic(
                    "invalid pattern ID start offset",
                ));
            }
            if start + len * 4 > self.pattern_ids().len() {
                return Err(DeserializeError::generic(
                    "invalid pattern ID length",
                ));
            }
            for mi in 0..len {
                let pid = self.pattern_id(si, mi);
                if pid as usize >= self.patterns {
                    return Err(DeserializeError::generic(
                        "invalid pattern ID",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Converts these match states back into their map form. This is useful
    /// when shuffling states, as the normal MatchStates representation is not
    /// amenable to easy state swapping. But with this map, to swap id1 and
    /// id2, all you need to do is:
    ///
    /// if let Some(pids) = map.remove(&id1) {
    ///     map.insert(id2, pids);
    /// }
    ///
    /// Once shuffling is done, use MatchStates::new to convert back.
    fn to_map(&self, dfa: &DFA<T, A, S>) -> BTreeMap<S, Vec<PatternID>> {
        let mut map = BTreeMap::new();
        for i in 0..self.count() {
            let mut pids = vec![];
            for j in 0..self.pattern_len(i) {
                pids.push(self.pattern_id(i, j));
            }
            map.insert(self.match_state_id(dfa, i), pids);
        }
        map
    }

    /// Converts these match states to a borrowed value.
    fn as_ref(&self) -> MatchStates<&'_ [S], &'_ [u8], S> {
        MatchStates {
            slices: self.slices(),
            pattern_ids: self.pattern_ids(),
            patterns: self.patterns,
            _state_id: self._state_id,
        }
    }

    /// Converts these match states to an owned value.
    fn to_owned(&self) -> MatchStates<Vec<S>, Vec<u8>, S> {
        MatchStates {
            slices: self.slices().to_vec(),
            pattern_ids: self.pattern_ids().to_vec(),
            patterns: self.patterns,
            _state_id: self._state_id,
        }
    }

    /// Converts these match states from using S as its state ID representation
    /// to using S2. If `size_of::<S2> >= size_of::<S>()`, then this always
    /// succeeds. If `size_of::<S2> < size_of::<S>()` and if S2 cannot
    /// represent every state ID in these match states, then an error is
    /// returned.
    fn to_sized<S2: StateID>(
        &self,
    ) -> Result<MatchStates<Vec<S2>, Vec<u8>, S2>, Error> {
        let mut ms = MatchStates {
            slices: Vec::with_capacity(self.slices().len()),
            pattern_ids: self.pattern_ids().to_vec(),
            patterns: self.patterns,
            _state_id: PhantomData,
        };
        for x in self.slices().iter() {
            if x.as_usize() > S2::max_id() {
                return Err(Error::state_id_overflow(S2::max_id()));
            }
            ms.slices.push(S2::from_usize(x.as_usize()));
        }
        Ok(ms)
    }

    /// Returns the match state ID given the match state index. (Where the
    /// first match state corresponds to index 0.)
    ///
    /// This panics if there is no match state at the given index.
    fn match_state_id(&self, dfa: &DFA<T, A, S>, index: usize) -> S {
        assert!(dfa.special.matches(), "no match states to index");
        // This is one of the places where we rely on the fact that match
        // states are contiguous in the transition table. Namely, that the
        // first match state ID always corresponds to dfa.special.min_start.
        // From there, since we know the stride, we can compute the ID of any
        // match state given its index.
        let id = S::from_usize(
            dfa.special.min_match.as_usize() + (index << dfa.tt.stride2),
        );
        assert!(dfa.is_match_state(id));
        id
    }

    fn match_pattern_ids(&self, state_index: usize) -> PatternIDIter {
        PatternIDIter { pattern_id_bytes: self.pattern_id_slice(state_index) }
    }

    fn pattern_id(&self, state_index: usize, match_index: usize) -> PatternID {
        let pids = self.pattern_id_slice(state_index);
        let pid = &pids[match_index * 4..match_index * 4 + 4];
        u32::from_ne_bytes(pid.try_into().unwrap())
    }

    fn pattern_len(&self, state_index: usize) -> usize {
        self.slices()[state_index * 2 + 1].as_usize()
    }

    fn pattern_id_slice(&self, state_index: usize) -> &[u8] {
        let start = self.slices()[state_index * 2].as_usize();
        let len = self.slices()[state_index * 2 + 1].as_usize();
        &self.pattern_ids()[start..start + 4 * len]
    }

    fn slices(&self) -> &[S] {
        self.slices.as_ref()
    }

    /// Returns the slice pairs as raw bytes.
    ///
    /// The length of the slice returned is always equivalent to
    /// `self.slices().len() * self.state_size()`.
    ///
    /// This is generally only useful when serializing the slices to raw bytes.
    fn slices_bytes(&self) -> &[u8] {
        let slices = self.slices();
        // SAFETY: This is safe because S is guaranteed to be one of {usize,
        // u8, u16, u32, u64}, and because u8 always has a smaller alignment.
        unsafe {
            core::slice::from_raw_parts(
                slices.as_ptr() as *const u8,
                slices.len() * self.state_size(),
            )
        }
    }

    /// Returns the total number of match states.
    fn count(&self) -> usize {
        assert_eq!(0, self.slices().len() % 2);
        self.slices().len() / 2
    }

    fn pattern_ids(&self) -> &[u8] {
        self.pattern_ids.as_ref()
    }

    /// Returns the total number of pattern IDs for all match states.
    fn pattern_id_count(&self) -> usize {
        assert_eq!(0, self.pattern_ids().len() % 4);
        self.pattern_ids().len() / 4
    }

    /// Returns the size of the specific state ID representation, in bytes.
    ///
    /// This is always 1, 2, 4 or 8.
    fn state_size(&self) -> usize {
        core::mem::size_of::<S>()
    }
}

/// An iterator over all states in a DFA.
///
/// This iterator yields a tuple for each state. The first element of the
/// tuple corresponds to a state's identifier, and the second element
/// corresponds to the state itself (comprised of its transitions).
///
/// `'a` corresponding to the lifetime of original DFA, `T` corresponds to
/// the type of the transition table itself and `S` corresponds to the state
/// identifier representation.
#[cfg(feature = "std")]
pub(crate) struct StateIter<'a, T, S> {
    tt: &'a TransitionTable<T, S>,
    it: iter::Enumerate<slice::Chunks<'a, S>>,
}

#[cfg(feature = "std")]
impl<'a, T: AsRef<[S]>, S: StateID> Iterator for StateIter<'a, T, S> {
    type Item = State<'a, S>;

    fn next(&mut self) -> Option<State<'a, S>> {
        self.it.next().map(|(index, _)| {
            let id = self.tt.from_index(index);
            self.tt.state(self.tt.from_index(index))
        })
    }
}

/// An immutable representation of a single DFA state.
///
/// `'a` correspondings to the lifetime of a DFA's transition table and `S`
/// corresponds to the state identifier representation.
#[cfg(feature = "std")]
pub(crate) struct State<'a, S> {
    id: S,
    stride2: usize,
    transitions: &'a [S],
}

#[cfg(feature = "std")]
impl<'a, S: StateID> State<'a, S> {
    /// Return an iterator over all transitions in this state. This yields
    /// a number of transitions equivalent to the alphabet length of the
    /// corresponding DFA.
    ///
    /// Each transition is represented by a tuple. The first element is
    /// the input byte for that transition and the second element is the
    /// transitions itself.
    pub(crate) fn transitions(&self) -> StateTransitionIter<'_, S> {
        StateTransitionIter {
            len: self.transitions.len(),
            it: self.transitions.iter().enumerate(),
        }
    }

    /// Return an iterator over a sparse representation of the transitions in
    /// this state. Only non-dead transitions are returned.
    ///
    /// The "sparse" representation in this case corresponds to a sequence of
    /// triples. The first two elements of the triple comprise an inclusive
    /// byte range while the last element corresponds to the transition taken
    /// for all bytes in the range.
    ///
    /// This is somewhat more condensed than the classical sparse
    /// representation (where you have an element for every non-dead
    /// transition), but in practice, checking if a byte is in a range is very
    /// cheap and using ranges tends to conserve quite a bit more space.
    pub(crate) fn sparse_transitions(
        &self,
    ) -> StateSparseTransitionIter<'_, S> {
        StateSparseTransitionIter { dense: self.transitions(), cur: None }
    }

    /// Returns the identifier for this state.
    pub(crate) fn id(&self) -> S {
        self.id
    }

    /// Returns the number of transitions in this state. This also corresponds
    /// to the alphabet length of this DFA.
    fn len(&self) -> usize {
        self.transitions.len()
    }

    /// Analyzes this state to determine whether it can be accelerated. If so,
    /// it returns an accelerator that contains at least one byte.
    fn accelerate(&self, classes: &ByteClasses) -> Option<Accel> {
        // We just try to add bytes to our accelerator. Once adding fails
        // (because we've added too many bytes), then give up.
        let mut accel = Accel::new();
        for (class, id) in self.transitions() {
            if id == self.id() {
                continue;
            }
            for byte_or_eof in classes.elements(class) {
                if let Byte::U8(byte) = byte_or_eof {
                    if !accel.add(byte) {
                        return None;
                    }
                }
            }
        }
        if accel.is_empty() {
            None
        } else {
            Some(accel)
        }
    }
}

impl<'a, S: StateID> fmt::Debug for State<'a, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, (start, end, id)) in self.sparse_transitions().enumerate() {
            let index = if f.alternate() {
                id.as_usize()
            } else {
                id.as_usize() >> self.stride2
            };
            if i > 0 {
                write!(f, ", ")?;
            }
            if start == end {
                write!(f, "{:?} => {}", start, index)?;
            } else {
                write!(f, "{:?}-{:?} => {}", start, end, index)?;
            }
        }
        Ok(())
    }
}

/// A mutable representation of a single DFA state.
///
/// `'a` correspondings to the lifetime of a DFA's transition table and `S`
/// corresponds to the state identifier representation.
#[cfg(feature = "std")]
pub(crate) struct StateMut<'a, S> {
    id: S,
    stride2: usize,
    transitions: &'a mut [S],
}

#[cfg(feature = "std")]
impl<'a, S: StateID> StateMut<'a, S> {
    /// Return an iterator over all transitions in this state. This yields
    /// a number of transitions equivalent to the alphabet length of the
    /// corresponding DFA.
    ///
    /// Each transition is represented by a tuple. The first element is the
    /// input byte for that transition and the second element is a mutable
    /// reference to the transition itself.
    pub(crate) fn iter_mut(&mut self) -> StateTransitionIterMut<'_, S> {
        StateTransitionIterMut {
            len: self.transitions.len(),
            it: self.transitions.iter_mut().enumerate(),
        }
    }
}

#[cfg(feature = "std")]
impl<'a, S: StateID> fmt::Debug for StateMut<'a, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(
            &State {
                id: self.id,
                stride2: self.stride2,
                transitions: self.transitions,
            },
            f,
        )
    }
}

/// An iterator over all transitions in a single DFA state. This yields
/// a number of transitions equivalent to the alphabet length of the
/// corresponding DFA.
///
/// Each transition is represented by a tuple. The first element is the input
/// byte for that transition and the second element is the transitions itself.
#[cfg(feature = "std")]
#[derive(Debug)]
pub(crate) struct StateTransitionIter<'a, S> {
    len: usize,
    it: iter::Enumerate<slice::Iter<'a, S>>,
}

#[cfg(feature = "std")]
impl<'a, S: StateID> Iterator for StateTransitionIter<'a, S> {
    type Item = (Byte, S);

    fn next(&mut self) -> Option<(Byte, S)> {
        self.it.next().map(|(i, &id)| {
            let b = if i + 1 == self.len {
                Byte::EOF(i as u16)
            } else {
                Byte::U8(i as u8)
            };
            (b, id)
        })
    }
}

/// A mutable iterator over all transitions in a DFA state.
///
/// Each transition is represented by a tuple. The first element is the
/// input byte for that transition and the second element is a mutable
/// reference to the transition itself.
#[cfg(feature = "std")]
#[derive(Debug)]
pub(crate) struct StateTransitionIterMut<'a, S> {
    len: usize,
    it: iter::Enumerate<slice::IterMut<'a, S>>,
}

#[cfg(feature = "std")]
impl<'a, S: StateID> Iterator for StateTransitionIterMut<'a, S> {
    type Item = (Byte, &'a mut S);

    fn next(&mut self) -> Option<(Byte, &'a mut S)> {
        self.it.next().map(|(i, id)| {
            let b = if i + 1 == self.len {
                Byte::EOF(i as u16)
            } else {
                Byte::U8(i as u8)
            };
            (b, id)
        })
    }
}

/// An iterator over all transitions in a single DFA state using a sparse
/// representation.
///
/// Each transition is represented by a triple. The first two elements of the
/// triple comprise an inclusive byte range while the last element corresponds
/// to the transition taken for all bytes in the range.
///
/// As a convenience, this always returns `Byte` values of the same type. That
/// is, you'll never get a (Byte::U8, Byte::EOF) or a (Byte::EOF, Byte::U8).
/// Only (Byte::U8, Byte::U8) and (Byte::EOF, Byte::EOF) values are yielded.
#[cfg(feature = "std")]
#[derive(Debug)]
pub(crate) struct StateSparseTransitionIter<'a, S> {
    dense: StateTransitionIter<'a, S>,
    cur: Option<(Byte, Byte, S)>,
}

#[cfg(feature = "std")]
impl<'a, S: StateID> Iterator for StateSparseTransitionIter<'a, S> {
    type Item = (Byte, Byte, S);

    fn next(&mut self) -> Option<(Byte, Byte, S)> {
        while let Some((b, next)) = self.dense.next() {
            let (prev_start, prev_end, prev_next) = match self.cur {
                Some(t) => t,
                None => {
                    self.cur = Some((b, b, next));
                    continue;
                }
            };
            if prev_next == next && !b.is_eof() {
                self.cur = Some((prev_start, b, prev_next));
            } else {
                self.cur = Some((b, b, next));
                if prev_next != dead_id() {
                    return Some((prev_start, prev_end, prev_next));
                }
            }
        }
        if let Some((start, end, next)) = self.cur.take() {
            if next != dead_id() {
                return Some((start, end, next));
            }
        }
        None
    }
}

/// An iterator over pattern IDs for a single match state.
#[derive(Debug)]
pub(crate) struct PatternIDIter<'a> {
    pattern_id_bytes: &'a [u8],
}

impl<'a> Iterator for PatternIDIter<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        if self.pattern_id_bytes.is_empty() {
            return None;
        }
        let bytes = &self.pattern_id_bytes[..4];
        self.pattern_id_bytes = &self.pattern_id_bytes[4..];
        Some(u32::from_ne_bytes(bytes.try_into().unwrap()))
    }
}

/// Remapper is an abstraction the manages the remapping of state IDs in a
/// dense DFA. This is useful when one wants to shuffle states into different
/// positions in the DFA.
///
/// One of the key complexities this manages is the ability to correctly move
/// one state multiple times.
///
/// Once shuffling is complete, `remap` should be called, which will rewrite
/// all pertinent transitions to updated state IDs.
#[derive(Debug)]
struct Remapper<S> {
    map: Vec<S>,
    // matches: BTreeMap<S, Vec<PatternID>>,
}

impl<S: StateID> Remapper<S> {
    fn from_dfa(dfa: &OwnedDFA<S>) -> Remapper<S> {
        Remapper {
            map: (0..dfa.tt.count).map(|i| dfa.from_index(i)).collect(),
            // matches: dfa.ms.to_map(dfa),
        }
    }

    fn swap(&mut self, dfa: &mut OwnedDFA<S>, id1: S, id2: S) {
        dfa.swap_states(id1, id2);
        self.map.swap(dfa.to_index(id1), dfa.to_index(id2));
    }

    fn remap(mut self, dfa: &mut OwnedDFA<S>) {
        // To work around the borrow checker for converting state IDs to
        // indices. We cannot borrow self while mutably iterating over a
        // state's transitions.
        let stride2 = dfa.stride2();
        let to_index = |id: S| -> usize { id.as_usize() >> stride2 };

        // Update the map to account for states that have been swapped
        // multiple times. For example, if (A, C) and (C, G) are swapped, then
        // transitions previously pointing to A should now point to G. But if
        // we don't update our map, they will erroneously be set to C. All we
        // do is follow the swaps in our map until we see our original state
        // ID.
        let oldmap = self.map.clone();
        for i in 0..dfa.tt.count {
            let cur_id = dfa.from_index(i);
            let mut new = oldmap[i];
            if cur_id == new {
                continue;
            }
            loop {
                let id = oldmap[dfa.to_index(new)];
                if cur_id == id {
                    self.map[i] = new;
                    break;
                }
                new = id;
            }
        }

        // Now that we've finished shuffling, we need to remap all of our
        // transitions. We don't need to handle re-mapping accelerated states
        // since `accels` is only populated after shuffling.
        for &id in self.map.iter() {
            for (_, next_id) in dfa.state_mut(id).iter_mut() {
                *next_id = self.map[to_index(*next_id)];
            }
        }
        for start_id in dfa.st.table_mut().iter_mut() {
            *start_id = self.map[to_index(*start_id)];
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn tiny_dfa_works() {
        let pattern = r"\w";
        Builder::new()
            .configure(Config::new().anchored(true))
            .syntax(crate::SyntaxConfig::new().unicode(false))
            .build_with_size::<u8>(pattern)
            .unwrap();
    }

    #[test]
    fn errors_when_converting_to_smaller_dfa() {
        let pattern = r"\w{10}";
        let dfa = Builder::new()
            .configure(Config::new().anchored(true).byte_classes(false))
            .build_with_size::<u32>(pattern)
            .unwrap();
        assert!(dfa.to_sized::<u16>().is_err());
    }

    #[test]
    fn errors_when_determinization_would_overflow() {
        let pattern = r"\w{10}";

        let mut builder = Builder::new();
        builder.configure(Config::new().anchored(true).byte_classes(false));
        // using u32 is fine
        assert!(builder.build_with_size::<u32>(pattern).is_ok());
        // // ... but u16 results in overflow (because there are >65536 states)
        assert!(builder.build_with_size::<u16>(pattern).is_err());
    }

    #[test]
    fn errors_when_classes_would_overflow() {
        let pattern = r"[a-z]";

        let mut builder = Builder::new();
        builder.configure(Config::new().anchored(true).byte_classes(true));
        // with classes is OK
        assert!(builder.build_with_size::<u8>(pattern).is_ok());
        // ... but without classes, it fails, since states become much bigger.
        builder.configure(Config::new().byte_classes(false));
        assert!(builder.build_with_size::<u8>(pattern).is_err());
    }

    #[test]
    fn errors_with_unicode_word_boundary() {
        let pattern = r"\b";
        assert!(Builder::new().build(pattern).is_err());
    }
}
