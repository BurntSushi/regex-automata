use core::{cmp, convert::TryFrom, fmt, mem, ops::Range};

use alloc::{boxed::Box, format, string::String, sync::Arc, vec, vec::Vec};

use crate::{
    nfa::thompson::{
        builder::Builder,
        compiler::{Compiler, Config},
        error::Error,
    },
    util::{
        alphabet::{self, ByteClassSet},
        captures::{self, GroupInfo, GroupInfoError},
        primitives::{
            IteratorIndexExt, NonMaxUsize, PatternID, PatternIDIter,
            SmallIndex, StateID,
        },
        search::{Match, Span},
        utf8,
    },
};

/// A byte oriented Thompson non-deterministic finite automaton (NFA).
///
/// A Thompson NFA is a finite state machine that permits unconditional epsilon
/// transitions, but guarantees that there exists at most one non-epsilon
/// transition for each element in the alphabet for each state.
///
/// An NFA may be used directly for searching, for analysis or to build
/// a deterministic finite automaton (DFA).
///
/// # Cheap clones
///
/// Since an NFA is a core data type in this crate that many other regex
/// engines are based on top of, it is convenient to give ownership of an NFA
/// to said regex engines. Because of this, an NFA uses reference counting
/// internally. Therefore, it is cheap to clone and it is encouraged to do so.
///
/// # Capabilities
///
/// Using an NFA for searching provides the most amount of "power" of any
/// regex engine in this crate. Namely, it supports the following:
///
/// 1. Detection of a match.
/// 2. Location of a match, including both the start and end offset, in a
/// single pass of the haystack.
/// 3. Location of matching capturing groups.
/// 4. Handles multiple patterns, including (1)-(3) when multiple patterns are
/// present.
///
/// # Capturing Groups
///
/// Groups refer to parenthesized expressions inside a regex pattern. They look
/// like this, where `exp` is an arbitrary regex:
///
/// * `(exp)` - An unnamed capturing group.
/// * `(?P<name>exp) - A named capturing group.
/// * `(?:exp)` - A non-capturing group.
/// * `(?i:exp)` - A non-capturing group that sets flags.
///
/// Only the first two forms are said to be _capturing_, which means that the
/// last position at which they match is reportable. The [`Captures`] type
/// provides convenient access to the match positions of capturing groups,
/// which includes looking up capturing groups by their name.
///
/// # Byte oriented
///
/// This NFA is byte oriented, which means that all of its transitions are
/// defined on bytes. In other words, the alphabet of an NFA consists of the
/// 256 different byte values.
///
/// While DFAs nearly demand that they be byte oriented for performance
/// reasons, an NFA could conceivably be *Unicode codepoint* oriented. Indeed,
/// a previous version of this NFA supported both byte and codepoint oriented
/// modes. A codepoint oriented mode can work because an NFA fundamentally uses
/// a sparse representation of transitions, which works well with the large
/// sparse space of Unicode codepoints.
///
/// Nevertheless, this NFA is only byte oriented. This choice is primarily
/// driven by implementation simplicity, and also in part memory usage. In
/// practice, performance between the two is roughly comparable. However,
/// building a DFA (including a hybrid DFA) really wants a byte oriented NFA.
/// So if we do have a codepoint oriented NFA, then we also need to generate
/// byte oriented NFA in order to build an hybrid NFA/DFA. Thus, by only
/// generating byte oriented NFAs, we can produce one less NFA. In other words,
/// if we made our NFA codepoint oriented, we'd need to *also* make it support
/// a byte oriented mode, which is more complicated. But a byte oriented mode
/// can support everything.
///
/// # Differences with DFAs
///
/// At the theoretical level, the precise difference between an NFA and a DFA
/// is that, in a DFA, for every state, an input symbol unambiguously refers
/// to a single transition _and_ that an input symbol is required for each
/// transition. At a practical level, this permits DFA implementations to be
/// implemented at their core with a small constant number of CPU instructions
/// for each byte of input searched. In practice, this makes them quite a bit
/// faster than NFAs _in general_. Namely, in order to execute a search for any
/// Thompson NFA, one needs to keep track of a _set_ of states, and execute
/// the possible transitions on all of those states for each input symbol.
/// Overall, this results in much more overhead. To a first approximation, one
/// can expect DFA searches to be about an order of magnitude faster.
///
/// So why use an NFA at all? The main advantage of an NFA is that it takes
/// linear time (in the size of the pattern string after repetitions have been
/// expanded) to build and linear memory usage. A DFA, on the other hand, may
/// take exponential time and/or space to build. Even in non-pathological
/// cases, DFAs often take quite a bit more memory than their NFA counterparts,
/// _especially_ if large Unicode character classes are involved. Of course,
/// an NFA also provides additional capabilities. For example, it can match
/// Unicode word boundaries on non-ASCII text and resolve the positions of
/// capturing groups.
///
/// Note that a [`hybrid::regex::Regex`](crate::hybrid::regex::Regex) strikes a
/// good balance between an NFA and a DFA. It avoids the exponential build time
/// of a DFA while maintaining its fast search time. The downside of a hybrid
/// NFA/DFA is that in some cases it can be slower at search time than the NFA.
/// (It also has less functionality than a pure NFA. It cannot handle Unicode
/// word boundaries on non-ASCII text and cannot resolve capturing groups.)
///
/// # Example
///
/// This shows how to build an NFA with the default configuration and execute a
/// search using the Pike VM.
///
/// ```
/// use regex_automata::{nfa::thompson::pikevm::PikeVM, Match};
///
/// let vm = PikeVM::new(r"foo[0-9]+")?;
/// let mut cache = vm.create_cache();
/// let mut caps = vm.create_captures();
///
/// let expected = Some(Match::must(0, 0..8));
/// vm.find(&mut cache, b"foo12345", &mut caps);
/// assert_eq!(expected, caps.get_match());
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Example: resolving capturing groups
///
/// This example shows how to parse some simple dates and extract the
/// components of each date via capturing groups.
///
/// ```
/// use regex_automata::nfa::thompson::{pikevm::PikeVM, Captures};
///
/// let vm = PikeVM::new(r"(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})")?;
/// let mut cache = vm.create_cache();
///
/// let haystack = "2012-03-14, 2013-01-01 and 2014-07-05";
/// let all: Vec<Captures> = vm.captures_iter(
///     &mut cache, haystack.as_bytes()
/// ).collect();
/// // There should be a total of 3 matches.
/// assert_eq!(3, all.len());
/// // The year from the second match is '2013'.
/// let span = all[1].get_group_by_name("y").unwrap();
/// assert_eq!("2013", &haystack[span]);
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// This example shows that only the last match of a capturing group is
/// reported, even if it had to match multiple times for an overall match
/// to occur.
///
/// ```
/// use regex_automata::{nfa::thompson::pikevm::PikeVM, Span};
///
/// let vm = PikeVM::new(r"([a-z]){4}")?;
/// let mut cache = vm.create_cache();
/// let mut caps = vm.create_captures();
///
/// let haystack = b"quux";
/// vm.find(&mut cache, haystack, &mut caps);
/// assert!(caps.is_match());
/// assert_eq!(Some(Span::from(3..4)), caps.get_group(1));
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone)]
pub struct NFA(
    // We make NFAs reference counted primarily for two reasons. First is that
    // the NFA type itself is quite large (at least 0.5KB), and so it makes
    // sense to put it on the heap by default anyway. Second is that, for Arc
    // specifically, this enables cheap clones. This tends to be useful because
    // several structures (the backtracker, the Pike VM, the hybrid NFA/DFA)
    // all want to hang on to an NFA for use during search time. We could
    // provide the NFA at search time via a function argument, but this makes
    // for an unnecessarily annoying API. Instead, we just let each structure
    // share ownership of the NFA. Using a deep clone would not be smart, since
    // the NFA can use quite a bit of heap space.
    pub(super) Arc<Inner>,
);

impl NFA {
    /// Parse the given regular expression using a default configuration and
    /// build an NFA from it.
    ///
    /// If you want a non-default configuration, then use the NFA
    /// [`Compiler`] with a [`Config`].
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::pikevm::PikeVM, Match};
    ///
    /// let vm = PikeVM::new(r"foo[0-9]+")?;
    /// let (mut cache, mut caps) = (vm.create_cache(), vm.create_captures());
    ///
    /// let expected = Some(Match::must(0, 0..8));
    /// vm.find(&mut cache, b"foo12345", &mut caps);
    /// assert_eq!(expected, caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(pattern: &str) -> Result<NFA, Error> {
        NFA::compiler().build(pattern)
    }

    /// Parse the given regular expressions using a default configuration and
    /// build a multi-NFA from them.
    ///
    /// If you want a non-default configuration, then use the NFA
    /// [`Compiler`] with a [`Config`].
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::pikevm::PikeVM, Match};
    ///
    /// let vm = PikeVM::new_many(&["[0-9]+", "[a-z]+"])?;
    /// let (mut cache, mut caps) = (vm.create_cache(), vm.create_captures());
    ///
    /// let expected = Some(Match::must(1, 0..3));
    /// vm.find(&mut cache, b"foo12345bar", &mut caps);
    /// assert_eq!(expected, caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new_many<P: AsRef<str>>(patterns: &[P]) -> Result<NFA, Error> {
        NFA::compiler().build_many(patterns)
    }

    /// Returns an NFA with a single regex pattern that always matches at every
    /// position.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::{NFA, pikevm::PikeVM}, Match};
    ///
    /// let vm = PikeVM::new_from_nfa(NFA::always_match())?;
    /// let (mut cache, mut caps) = (vm.create_cache(), vm.create_captures());
    ///
    /// let expected = Some(Match::must(0, 0..0));
    /// vm.find(&mut cache, b"", &mut caps);
    /// assert_eq!(expected, caps.get_match());
    /// vm.find(&mut cache, b"foo", &mut caps);
    /// assert_eq!(expected, caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn always_match() -> NFA {
        // We could use NFA::new("") here and we'd get the same semantics, but
        // hand-assembling the NFA (as below) does the same thing with a fewer
        // number of states.
        //
        // Technically all we need is the "match" state, but we add the
        // "capture" states so that the PikeVM can use this NFA.
        //
        // The unwraps below are OK because we add so few states that they will
        // never exhaust any default limits in any environment.
        let mut builder = Builder::new();
        let pid = builder.start_pattern().unwrap();
        assert_eq!(pid.as_usize(), 0);
        let start_id =
            builder.add_capture_start(StateID::ZERO, 0, None).unwrap();
        let end_id = builder.add_capture_end(StateID::ZERO, 0).unwrap();
        let match_id = builder.add_match().unwrap();
        builder.patch(start_id, end_id);
        builder.patch(end_id, match_id);
        let pid = builder.finish_pattern(start_id).unwrap();
        assert_eq!(pid.as_usize(), 0);
        builder.build(start_id, start_id).unwrap()
    }

    /// Returns an NFA that never matches at any position.
    ///
    /// This is a convenience routine for creating an NFA with zero patterns.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::nfa::thompson::{NFA, pikevm::PikeVM};
    ///
    /// let vm = PikeVM::new_from_nfa(NFA::never_match())?;
    /// let (mut cache, mut caps) = (vm.create_cache(), vm.create_captures());
    ///
    /// vm.find(&mut cache, b"", &mut caps);
    /// assert!(!caps.is_match());
    /// vm.find(&mut cache, b"foo", &mut caps);
    /// assert!(!caps.is_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn never_match() -> NFA {
        // This always succeeds because it only requires one NFA state, which
        // will never exhaust any (default) limits.
        NFA::new_many::<&str>(&[]).unwrap()
    }

    /// Return a default configuration for an `NFA`.
    ///
    /// This is a convenience routine to avoid needing to import the `Config`
    /// type when customizing the construction of an NFA.
    ///
    /// # Example
    ///
    /// This example shows how to build an NFA with a small size limit that
    /// results in a compilation error for any regex that tries to use more
    /// heap memory than the configured limit.
    ///
    /// ```
    /// use regex_automata::nfa::thompson::{NFA, pikevm::PikeVM};
    ///
    /// let result = PikeVM::builder()
    ///     .thompson(NFA::config().nfa_size_limit(Some(1_000)))
    ///     // Remember, \w is Unicode-aware by default and thus huge.
    ///     .build(r"\w+");
    /// assert!(result.is_err());
    /// ```
    pub fn config() -> Config {
        Config::new()
    }

    /// Return a compiler for configuring the construction of an `NFA`.
    ///
    /// This is a convenience routine to avoid needing to import the
    /// [`Compiler`] type in common cases.
    ///
    /// # Example
    ///
    /// This example shows how to build an NFA that is permitted match invalid
    /// UTF-8. Without the additional syntax configuration here, compilation of
    /// `(?-u:.)` would fail because it is permitted to match invalid UTF-8.
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::pikevm::PikeVM,
    ///     Match, SyntaxConfig
    /// };
    ///
    /// let vm = PikeVM::builder()
    ///     .syntax(SyntaxConfig::new().utf8(false))
    ///     .build(r"[a-z]+(?-u:.)")?;
    /// let (mut cache, mut caps) = (vm.create_cache(), vm.create_captures());
    ///
    /// let expected = Some(Match::must(0, 1..5));
    /// vm.find(&mut cache, b"\xFFabc\xFF", &mut caps);
    /// assert_eq!(expected, caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn compiler() -> Compiler {
        Compiler::new()
    }

    /// Returns an iterator over all pattern identifiers in this NFA.
    ///
    /// Pattern IDs are allocated in sequential order starting from zero,
    /// where the order corresponds to the order of patterns provided to the
    /// [`NFA::new_many`] constructor.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::NFA, PatternID};
    ///
    /// let nfa = NFA::new_many(&["[0-9]+", "[a-z]+", "[A-Z]+"])?;
    /// let pids: Vec<PatternID> = nfa.patterns().collect();
    /// assert_eq!(pids, vec![
    ///     PatternID::must(0),
    ///     PatternID::must(1),
    ///     PatternID::must(2),
    /// ]);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn patterns(&self) -> PatternIter<'_> {
        PatternIter {
            it: PatternID::iter(self.pattern_len()),
            _marker: core::marker::PhantomData,
        }
    }

    /// Returns the total number of regex patterns in this NFA.
    ///
    /// This may return zero if the NFA was constructed with no patterns. In
    /// this case, the NFA can never produce a match for any input.
    ///
    /// This is guaranteed to be no bigger than [`PatternID::LIMIT`] because
    /// NFA construction will fail if too many patterns are added.
    ///
    /// It is always true that `nfa.patterns().count() == nfa.pattern_len()`.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::nfa::thompson::NFA;
    ///
    /// let nfa = NFA::new_many(&["[0-9]+", "[a-z]+", "[A-Z]+"])?;
    /// assert_eq!(3, nfa.pattern_len());
    ///
    /// let nfa = NFA::never_match();
    /// assert_eq!(0, nfa.pattern_len());
    ///
    /// let nfa = NFA::always_match();
    /// assert_eq!(1, nfa.pattern_len());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn pattern_len(&self) -> usize {
        self.0.start_pattern.len()
    }

    /// Return the state identifier of the initial anchored state of this NFA.
    ///
    /// The returned identifier is guaranteed to be a valid index into the
    /// slice returned by [`NFA::states`], and is also a valid argument to
    /// [`NFA::state`].
    ///
    /// # Example
    ///
    /// This example shows a somewhat contrived example where we can easily
    /// predict the anchored starting state.
    ///
    /// ```
    /// use regex_automata::nfa::thompson::{NFA, State};
    ///
    /// let nfa = NFA::compiler()
    ///     .configure(NFA::config().captures(false))
    ///     .build("a")?;
    /// let state = nfa.state(nfa.start_anchored());
    /// match *state {
    ///     State::ByteRange { trans } => {
    ///         assert_eq!(b'a', trans.start);
    ///         assert_eq!(b'a', trans.end);
    ///     }
    ///     _ => unreachable!("unexpected state"),
    /// }
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn start_anchored(&self) -> StateID {
        self.0.start_anchored
    }

    /// Return the state identifier of the initial unanchored state of this
    /// NFA.
    ///
    /// This is equivalent to the identifier returned by
    /// [`NFA::start_anchored`] when the NFA has no unanchored starting state.
    ///
    /// The returned identifier is guaranteed to be a valid index into the
    /// slice returned by [`NFA::states`], and is also a valid argument to
    /// [`NFA::state`].
    ///
    /// # Example
    ///
    /// This example shows that the anchored and unanchored starting states
    /// are equivalent when an anchored NFA is built.
    ///
    /// ```
    /// use regex_automata::nfa::thompson::NFA;
    ///
    /// let nfa = NFA::new("^a")?;
    /// assert_eq!(nfa.start_anchored(), nfa.start_unanchored());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn start_unanchored(&self) -> StateID {
        self.0.start_unanchored
    }

    /// Return the state identifier of the initial anchored state for the given
    /// pattern.
    ///
    /// If one uses the starting state for a particular pattern, then the only
    /// match that can be returned is for the corresponding pattern.
    ///
    /// The returned identifier is guaranteed to be a valid index into the
    /// slice returned by [`NFA::states`], and is also a valid argument to
    /// [`NFA::state`].
    ///
    /// # Panics
    ///
    /// If the pattern doesn't exist in this NFA, then this panics. This
    /// occurs when `pid.as_usize() >= nfa.pattern_len()`.
    ///
    /// # Example
    ///
    /// This example shows that the anchored and unanchored starting states
    /// are equivalent when an anchored NFA is built.
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::NFA, PatternID};
    ///
    /// let nfa = NFA::new_many(&["^a", "^b"])?;
    /// // The anchored and unanchored states for the entire NFA are the same,
    /// // since all of the patterns are anchored.
    /// assert_eq!(nfa.start_anchored(), nfa.start_unanchored());
    /// // But the anchored starting states for each pattern are distinct,
    /// // because these starting states can only lead to matches for the
    /// // corresponding pattern.
    /// let anchored = nfa.start_anchored();
    /// assert_ne!(anchored, nfa.start_pattern(PatternID::must(0)));
    /// assert_ne!(anchored, nfa.start_pattern(PatternID::must(1)));
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn start_pattern(&self, pid: PatternID) -> StateID {
        assert!(pid.as_usize() < self.pattern_len(), "invalid pattern ID");
        self.0.start_pattern[pid]
    }

    /// Get the byte class set for this NFA.
    ///
    /// A byte class set is a partitioning of this NFA's alphabet into
    /// equivalence classes. Any two bytes in the same equivalence class are
    /// guaranteed to never discriminate between a match or a non-match. (The
    /// partitioning may not be minimal.)
    ///
    /// Byte classes are used internally by this crate when building DFAs.
    /// Namely, among other optimizations, they enable a space optimization
    /// where the DFA's internal alphabet is defined over the equivalence
    /// classes of bytes instead of all possible byte values. The former is
    /// often quite a bit smaller than the latter, which permits the DFA to use
    /// less space for its transition table.
    ///
    /// # Example
    ///
    /// Typically the only operation one can perform on a `ByteClassSet` is to
    /// extract the equivalence classes:
    ///
    /// ```
    /// use regex_automata::nfa::thompson::NFA;
    ///
    /// let nfa = NFA::new("[a-z]+")?;
    /// let classes = nfa.byte_class_set().byte_classes();
    /// // 'a' and 'z' are in the same class for this regex.
    /// assert_eq!(classes.get(b'a'), classes.get(b'z'));
    /// // But 'a' and 'A' are not.
    /// assert_ne!(classes.get(b'a'), classes.get(b'A'));
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn byte_class_set(&self) -> &ByteClassSet {
        &self.0.byte_class_set
    }

    /// Return a reference to the NFA state corresponding to the given ID.
    ///
    /// This is a convenience routine for `nfa.states()[id]`.
    ///
    /// # Panics
    ///
    /// This panics when the given identifier does not reference a valid state.
    /// That is, when `id.as_usize() >= nfa.states().len()`.
    ///
    /// # Example
    ///
    /// The anchored state for a pattern will typically correspond to a
    /// capturing state for that pattern. (Although, this is not an API
    /// guarantee!)
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::{NFA, State}, PatternID};
    ///
    /// let nfa = NFA::new("a")?;
    /// let state = nfa.state(nfa.start_pattern(PatternID::ZERO));
    /// match *state {
    ///     State::Capture { slot, .. } => {
    ///         assert_eq!(0, slot.as_usize());
    ///     }
    ///     _ => unreachable!("unexpected state"),
    /// }
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn state(&self, id: StateID) -> &State {
        &self.states()[id]
    }

    /// Returns a slice of all states in this NFA.
    ///
    /// The slice returned is indexed by `StateID`. This provides a convenient
    /// way to access states while following transitions among those states.
    ///
    /// # Example
    ///
    /// This demonstrates that disabling UTF-8 mode can shrink the size of the
    /// NFA considerably in some cases, especially when using Unicode character
    /// classes.
    ///
    /// ```
    /// use regex_automata::nfa::thompson::NFA;
    ///
    /// let nfa_unicode = NFA::new(r"\w")?;
    /// let nfa_ascii = NFA::new(r"(?-u)\w")?;
    /// // Yes, a factor of 45 difference. No lie.
    /// assert!(40 * nfa_ascii.states().len() < nfa_unicode.states().len());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn states(&self) -> &[State] {
        &self.0.states
    }

    /// Returns the capturing group info for this NFA.
    ///
    /// The [`GroupInfo`] provides a way to map to and from capture index
    /// and capture name for each pattern. It also provides a mapping from
    /// each of the capturing groups in every pattern to their corresponding
    /// slot offsets encoded in [`State::Capture`] states.
    ///
    /// Note that `GroupInfo` uses reference counting internally, such that
    /// cloning a `GroupInfo` is very cheap.
    #[inline]
    pub fn group_info(&self) -> &GroupInfo {
        &self.0.group_info()
    }

    /// Returns true if and only if this NFA has at least one
    /// [`Capture`](State::Capture) in its sequence of states.
    ///
    /// This is useful as a way to perform a quick test before attempting
    /// something that does or does not require capture states. For example,
    /// some regex engines (like the PikeVM) require capture states in order to
    /// work at all.
    ///
    /// # Example
    ///
    /// This example shows a few different NFAs and whether they have captures
    /// or not.
    ///
    /// ```
    /// use regex_automata::nfa::thompson::NFA;
    ///
    /// // Obviously has capture states.
    /// let nfa = NFA::new("(a)")?;
    /// assert!(nfa.has_capture());
    ///
    /// // Less obviously has capture states, because every pattern has at
    /// // least one anonymous capture group corresponding to the match for the
    /// // entire pattern.
    /// let nfa = NFA::new("a")?;
    /// assert!(nfa.has_capture());
    ///
    /// // Other than hand building your own NFA, this is the only way to build
    /// // an NFA without capturing groups. In general, you should only do this
    /// // if you don't intend to use any of the NFA-oriented regex engines.
    /// // Overall, capturing groups don't have many downsides. Although they
    /// // can add a bit of noise to simple NFAs, so it can be nice to disable
    /// // them for debugging purposes.
    /// //
    /// // Notice that 'has_capture' is false here even when we have an
    /// // explicit capture group in the pattern.
    /// let nfa = NFA::compiler()
    ///     .configure(NFA::config().captures(false))
    ///     .build("(a)")?;
    /// assert!(!nfa.has_capture());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn has_capture(&self) -> bool {
        self.0.facts.has_capture
    }

    /// Returns true if and only if all starting states for this NFA correspond
    /// to the beginning of an anchored search.
    ///
    /// Typically, an NFA will have both an anchored and an unanchored starting
    /// state. Namely, because it tends to be useful to have both and the cost
    /// of having an unanchored starting state is almost zero (for an NFA).
    /// However, if all patterns in the NFA are themselves anchored, then even
    /// the unanchored starting state will correspond to an anchored search
    /// since the pattern doesn't permit anything else.
    ///
    /// # Example
    ///
    /// This example shows a few different scenarios where this method's
    /// return value varies.
    ///
    /// ```
    /// use regex_automata::nfa::thompson::NFA;
    ///
    /// // The unanchored starting state permits matching this pattern anywhere
    /// // in a haystack, instead of just at the beginning.
    /// let nfa = NFA::new("a")?;
    /// assert!(!nfa.is_always_start_anchored());
    ///
    /// // In this case, the pattern is itself anchored, so there is no way
    /// // to run an unanchored search.
    /// let nfa = NFA::new("^a")?;
    /// assert!(nfa.is_always_start_anchored());
    ///
    /// // When multiline mode is enabled, '^' can match at the start of a line
    /// // in addition to the start of a haystack, so an unanchored search is
    /// // actually possible.
    /// let nfa = NFA::new("(?m)^a")?;
    /// assert!(!nfa.is_always_start_anchored());
    ///
    /// // Weird cases also work. A pattern is only considered anchored if all
    /// // matches may only occur at the start of a haystack.
    /// let nfa = NFA::new("(^a)|a")?;
    /// assert!(!nfa.is_always_start_anchored());
    ///
    /// // When multiple patterns are present, if they are all anchored, then
    /// // the NFA is always anchored too.
    /// let nfa = NFA::new_many(&["^a", "^b", "^c"])?;
    /// assert!(nfa.is_always_start_anchored());
    ///
    /// // But if one pattern is unanchored, then the NFA must permit an
    /// // unanchored search.
    /// let nfa = NFA::new_many(&["^a", "b", "^c"])?;
    /// assert!(!nfa.is_always_start_anchored());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn is_always_start_anchored(&self) -> bool {
        self.start_anchored() == self.start_unanchored()
    }

    /// Returns true if this NFA has any [`Look`](State::Look) states.
    ///
    /// This is useful for cases where you want to use an NFA in contexts that
    /// can't handle look-around.
    ///
    /// # Example
    ///
    /// This example shows how this routine varies based on the regex pattern:
    ///
    /// ```
    /// use regex_automata::nfa::thompson::NFA;
    ///
    /// // No look-around at all.
    /// let nfa = NFA::new("a")?;
    /// assert!(!nfa.has_look());
    ///
    /// // Look-around via an anchor.
    /// let nfa = NFA::new("^")?;
    /// assert!(nfa.has_look());
    ///
    /// // Look-around via a word boundary.
    /// let nfa = NFA::new(r"\b")?;
    /// assert!(nfa.has_look());
    ///
    /// // When multiple patterns are present, this still returns true even
    /// // if only one of them has look-around.
    /// let nfa = NFA::new_many(&["a", "b", "^", "c"])?;
    /// assert!(nfa.has_look());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn has_look(&self) -> bool {
        self.0.facts.has_look
    }

    /// Returns true if this NFA has any [`Look`](State::Look) states that
    /// correspond to an anchor assertion (start/end of haystack or start/end
    /// of line).
    ///
    /// This is useful for cases where you want to use an NFA in contexts that
    /// can't handle anchor assertions.
    ///
    /// # Example
    ///
    /// This example shows how this routine varies based on the regex pattern:
    ///
    /// ```
    /// use regex_automata::nfa::thompson::NFA;
    ///
    /// // With an anchor.
    /// let nfa = NFA::new("^")?;
    /// assert!(nfa.has_anchor());
    ///
    /// // A word boundary isn't an anchor.
    /// let nfa = NFA::new(r"\b")?;
    /// assert!(!nfa.has_anchor());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn has_anchor(&self) -> bool {
        self.0.facts.has_anchor
    }

    /// Returns true if this NFA has any [`Look`](State::Look) states that
    /// correspond to a word boundary assertion.
    ///
    /// This is useful for cases where you want to use an NFA in contexts that
    /// can't handle word boundary assertions.
    ///
    /// # Example
    ///
    /// This example shows how this routine varies based on the regex pattern:
    ///
    /// ```
    /// use regex_automata::nfa::thompson::NFA;
    ///
    /// // An anchor isn't a word boundary.
    /// let nfa = NFA::new("^")?;
    /// assert!(!nfa.has_word_boundary());
    ///
    /// // With a word boundary.
    /// let nfa = NFA::new(r"\b")?;
    /// assert!(nfa.has_word_boundary());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn has_word_boundary(&self) -> bool {
        self.has_word_boundary_unicode() || self.has_word_boundary_ascii()
    }

    /// Returns true if this NFA has any [`Look`](State::Look) states that
    /// correspond to a Unicode word boundary assertion.
    ///
    /// This is useful for cases where you want to use an NFA in contexts that
    /// can't handle Unicode word boundary assertions (such as the DFAs in this
    /// crate).
    ///
    /// # Example
    ///
    /// This example shows how this routine varies based on the regex pattern:
    ///
    /// ```
    /// use regex_automata::nfa::thompson::NFA;
    ///
    /// // With a Unicode word boundary.
    /// let nfa = NFA::new(r"\b")?;
    /// assert!(nfa.has_word_boundary_unicode());
    ///
    /// // When Unicode is disabled, \b is only ASCII-aware.
    /// let nfa = NFA::new(r"(?-u:\b)")?;
    /// assert!(!nfa.has_word_boundary_unicode());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn has_word_boundary_unicode(&self) -> bool {
        self.0.facts.has_word_boundary_unicode
    }

    /// Returns true if this NFA has any [`Look`](State::Look) states that
    /// correspond to an ASCII word boundary assertion.
    ///
    /// This is useful for cases where you want to use an NFA in contexts that
    /// can't handle ASCII word boundary assertions.
    ///
    /// # Example
    ///
    /// This example shows how this routine varies based on the regex pattern:
    ///
    /// ```
    /// use regex_automata::nfa::thompson::NFA;
    ///
    /// // With a Unicode word boundary, this returns false.
    /// let nfa = NFA::new(r"\b")?;
    /// assert!(!nfa.has_word_boundary_ascii());
    ///
    /// // When Unicode is disabled, \b is only ASCII-aware.
    /// let nfa = NFA::new(r"(?-u:\b)")?;
    /// assert!(nfa.has_word_boundary_ascii());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn has_word_boundary_ascii(&self) -> bool {
        self.0.facts.has_word_boundary_ascii
    }

    /// Returns the memory usage, in bytes, of this NFA.
    ///
    /// This does **not** include the stack size used up by this NFA. To
    /// compute that, use `std::mem::size_of::<NFA>()`.
    ///
    /// # Example
    ///
    /// This example shows that large Unicode character classes can use quite
    /// a bit of memory.
    ///
    /// ```
    /// use regex_automata::nfa::thompson::NFA;
    ///
    /// let nfa_unicode = NFA::new(r"\w")?;
    /// let nfa_ascii = NFA::new(r"(?-u:\w)")?;
    ///
    /// assert!(10 * nfa_ascii.memory_usage() < nfa_unicode.memory_usage());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn memory_usage(&self) -> usize {
        use core::mem::size_of as s;

        s::<Inner>()
            + self.0.states.len() * s::<State>()
            + self.0.start_pattern.len() * s::<StateID>()
            + self.0.group_info.memory_usage()
            + self.0.memory_extra
    }
}

impl fmt::Debug for NFA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// The "inner" part of the NFA. We split this part out so that we can easily
/// wrap it in an `Arc` above in the definition of `NFA`.
///
/// See builder.rs for the code that actually builds this type. This module
/// does provide (internal) mutable methods for adding things to this
/// NFA before finalizing it, but the high level construction process is
/// controlled by the builder abstraction. (Which is complicated enough to
/// get its own module.)
#[derive(Default)]
pub(super) struct Inner {
    /// The state sequence. This sequence is guaranteed to be indexable by all
    /// starting state IDs, and it is also guaranteed to contain at most one
    /// `Match` state for each pattern compiled into this NFA. (A pattern may
    /// not have a corresponding `Match` state if a `Match` state is impossible
    /// to reach.)
    states: Vec<State>,
    /// The anchored starting state of this NFA.
    start_anchored: StateID,
    /// The unanchored starting state of this NFA.
    start_unanchored: StateID,
    /// The starting states for each individual pattern. Starting at any
    /// of these states will result in only an anchored search for the
    /// corresponding pattern. The vec is indexed by pattern ID. When the NFA
    /// contains a single regex, then `start_pattern[0]` and `start_anchored`
    /// are always equivalent.
    start_pattern: Vec<StateID>,
    /// Info about the capturing groups in this NFA. This is responsible for
    /// mapping groups to slots, mapping groups to names and names to groups.
    group_info: GroupInfo,
    /// A representation of equivalence classes over the transitions in this
    /// NFA. Two bytes in the same equivalence class must not discriminate
    /// between a match or a non-match. This map can be used to shrink the
    /// total size of a DFA's transition table with a small match-time cost.
    ///
    /// Note that the NFA's transitions are *not* defined in terms of these
    /// equivalence classes. The NFA's transitions are defined on the original
    /// byte values. For the most part, this is because they wouldn't really
    /// help the NFA much since the NFA already uses a sparse representation
    /// to represent transitions. Byte classes are most effective in a dense
    /// representation.
    byte_class_set: ByteClassSet,
    /// Various facts about this NFA, which can be used to improve failure
    /// modes (e.g., rejecting DFA construction if an NFA has Unicode word
    /// boundaries) or for performing optimizations (avoiding an increase in
    /// states if there are no look-around states).
    facts: Facts,
    /// Heap memory used indirectly by NFA states and other things (like the
    /// various capturing group representations above). Since each state
    /// might use a different amount of heap, we need to keep track of this
    /// incrementally.
    memory_extra: usize,
}

impl Inner {
    /// Returns the capturing group info for this NFA.
    pub(super) fn group_info(&self) -> &GroupInfo {
        &self.group_info
    }

    /// Add the given state to this NFA after allocating a fresh identifier for
    /// it.
    ///
    /// This panics if too many states are added such that a fresh identifier
    /// could not be created. (Currently, the only caller of this routine is
    /// a `Builder`, and it upholds this invariant.)
    pub(super) fn add(&mut self, state: State) -> StateID {
        match state {
            State::ByteRange { ref trans } => {
                self.byte_class_set.set_range(trans.start, trans.end);
            }
            State::Sparse(ref sparse) => {
                for trans in sparse.transitions.iter() {
                    self.byte_class_set.set_range(trans.start, trans.end);
                }
            }
            State::Look { ref look, .. } => {
                self.facts.has_look = true;
                look.add_to_byteset(&mut self.byte_class_set);
                match look {
                    Look::StartLine
                    | Look::EndLine
                    | Look::StartText
                    | Look::EndText => {
                        self.facts.has_anchor = true;
                    }
                    Look::WordBoundaryUnicode
                    | Look::WordBoundaryUnicodeNegate => {
                        self.facts.has_word_boundary_unicode = true;
                    }
                    Look::WordBoundaryAscii
                    | Look::WordBoundaryAsciiNegate => {
                        self.facts.has_word_boundary_ascii = true;
                    }
                }
            }
            State::Capture { .. } => {
                self.facts.has_capture = true;
            }
            State::Union { .. }
            | State::BinaryUnion { .. }
            | State::Fail
            | State::Match { .. } => {}
        }

        let id = StateID::new(self.states.len()).unwrap();
        self.memory_extra += state.memory_usage();
        self.states.push(state);
        id
    }

    /// Set the starting state identifiers for this NFA.
    ///
    /// `start_anchored` and `start_unanchored` may be equivalent. When they
    /// are, then the NFA can only execute anchored searches. This might
    /// occur, for example, for patterns that are unconditionally anchored.
    /// e.g., `^foo`.
    pub(super) fn set_starts(
        &mut self,
        start_anchored: StateID,
        start_unanchored: StateID,
        start_pattern: &[StateID],
    ) {
        self.start_anchored = start_anchored;
        self.start_unanchored = start_unanchored;
        self.start_pattern = start_pattern.to_owned();
    }

    /// Set the capturing groups for this NFA.
    ///
    /// The given slice should contain the capturing groups for each pattern,
    /// The capturing groups in turn should correspond to the total number of
    /// capturing groups in the pattern, including the anonymous first capture
    /// group for each pattern. If a capturing group does have a name, then it
    /// should be provided as a Arc<str>.
    ///
    /// This returns an error if a corresponding `GroupInfo` could not be
    /// built.
    pub(super) fn set_captures(
        &mut self,
        captures: &[Vec<Option<Arc<str>>>],
    ) -> Result<(), GroupInfoError> {
        self.group_info = GroupInfo::new(
            captures.iter().map(|x| x.iter().map(|y| y.as_ref())),
        )?;
        Ok(())
    }

    /// Remap the transitions in every state of this NFA using the given map.
    /// The given map should be indexed according to state ID namespace used by
    /// the transitions of the states currently in this NFA.
    ///
    /// This is particularly useful to the NFA builder, since it is convenient
    /// to add NFA states in order to produce their final IDs. Then, after all
    /// of the intermediate "empty" states (unconditional epsilon transitions)
    /// have been removed from the builder's representation, we can re-map all
    /// of the transitions in the states already added to their final IDs.
    pub(super) fn remap(&mut self, old_to_new: &[StateID]) {
        for state in &mut self.states {
            state.remap(old_to_new);
        }
        self.start_anchored = old_to_new[self.start_anchored];
        self.start_unanchored = old_to_new[self.start_unanchored];
        for (pid, id) in self.start_pattern.iter_mut().with_pattern_ids() {
            *id = old_to_new[*id];
        }
    }
}

impl fmt::Debug for Inner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "thompson::NFA(")?;
        for (sid, state) in self.states.iter().with_state_ids() {
            let status = if sid == self.start_anchored {
                '^'
            } else if sid == self.start_unanchored {
                '>'
            } else {
                ' '
            };
            writeln!(f, "{}{:06?}: {:?}", status, sid.as_usize(), state)?;
        }
        let pattern_len = self.start_pattern.len();
        if pattern_len > 1 {
            writeln!(f, "")?;
            for pid in 0..pattern_len {
                let sid = self.start_pattern[pid];
                writeln!(f, "START({:06?}): {:?}", pid, sid.as_usize())?;
            }
        }
        writeln!(f, "")?;
        writeln!(
            f,
            "transition equivalence classes: {:?}",
            self.byte_class_set.byte_classes(),
        )?;
        writeln!(f, ")")?;
        Ok(())
    }
}

/// A map from capture group name to its corresponding capture index.
///
/// Since there are always two slots for each capture index, the pair of slots
/// corresponding to the capture index for a pattern ID of 0 are indexed at
/// `map["<name>"] * 2` and `map["<name>"] * 2 + 1`.
///
/// This type is actually wrapped inside a Vec indexed by pattern ID on the
/// NFA, since multiple patterns may have the same capture group name.
///
/// Note that this is somewhat of a sub-optimal representation, since it
/// requires a hashmap for each pattern. A better representation would be
/// HashMap<(PatternID, Arc<str>), usize>, but this makes it difficult to look
/// up a capture index by name without producing a `Arc<str>`, which requires
/// an allocation. To fix this, I think we'd need to define our own unsized
/// type or something?
#[cfg(feature = "std")]
type CaptureNameMap = std::collections::HashMap<Arc<str>, usize>;
#[cfg(not(feature = "std"))]
type CaptureNameMap = alloc::collections::BTreeMap<Arc<str>, usize>;

/// A state in an NFA.
///
/// In theory, it can help to conceptualize an `NFA` as a graph consisting of
/// `State`s. Each `State` contains its complete set of outgoing transitions.
///
/// In practice, it can help to conceptualize an `NFA` as a sequence of
/// instructions for a virtual machine. Each `State` says what to do and where
/// to go next.
///
/// Strictly speaking, the practical interpretation is the most correct one,
/// because of the [`Capture`](State::Capture) state. Namely, a `Capture`
/// state always forwards execution to another state unconditionally. Its only
/// purpose is to cause a side effect: the recording of the current input
/// position at a particular location in memory. In this sense, an `NFA`
/// has more power than a theoretical non-deterministic finite automaton.
/// (Although, strictly speaking, one could write a search implementation that
/// ignores `Capture` states and reports only the end location of a match, just
/// like the DFAs do. However, such a limited implementation does not exist in
/// this crate.)
///
/// For most uses of this crate, it is likely that one may never even need to
/// be aware of this type at all. The main use cases for looking at `State`s
/// directly are if you need to write your own search implementation or if you
/// need to do some kind of analysis on the NFA.
#[derive(Clone, Eq, PartialEq)]
pub enum State {
    /// A state with a single transition that can only be taken if the current
    /// input symbol is in a particular range of bytes.
    ByteRange { trans: Transition },
    /// A state with possibly many transitions represented in a sparse fashion.
    /// Transitions are non-overlapping and ordered lexicographically by input
    /// range.
    ///
    /// In practice, this is used for encoding UTF-8 automata. Its presence is
    /// primarily an optimization that avoids many additional unconditional
    /// epsilon transitions (via [`Union`](State::Union) states), and thus
    /// decreases the overhead of traversing the NFA. This can improve both
    /// matching time and DFA construction time.
    Sparse(SparseTransitions),
    /// A conditional epsilon transition satisfied via some sort of
    /// look-around. Look-around is limited to anchor and word boundary
    /// assertions.
    ///
    /// Look-around states are meant to be evaluated while performing epsilon
    /// closure (computing the set of states reachable from a particular state
    /// via only epsilon transitions). If the current position in the haystack
    /// satisfies the look-around assertion, then you're permitted to follow
    /// that epsilon transition.
    Look { look: Look, next: StateID },
    /// An alternation such that there exists an epsilon transition to all
    /// states in `alternates`, where matches found via earlier transitions
    /// are preferred over later transitions.
    Union { alternates: Box<[StateID]> },
    /// An alternation such that there exists precisely two unconditional
    /// epsilon transitions, where matches found via `alt1` are preferred over
    /// matches found via `alt2`.
    ///
    /// This state exists as a common special case of Union where there are
    /// only two alternates. In this case, we don't need any allocations to
    /// represent the state. This saves a bit of memory and also saves an
    /// additional memory access when traversing the NFA.
    BinaryUnion { alt1: StateID, alt2: StateID },
    /// An empty state that records a capture location.
    ///
    /// From the perspective of finite automata, this is precisely equivalent
    /// to an epsilon transition, but serves the purpose of instructing NFA
    /// simulations to record additional state when the finite state machine
    /// passes through this epsilon transition.
    ///
    /// These transitions are treated as epsilon transitions with no additional
    /// effects in DFAs.
    ///
    /// 'slot' in this context refers to the specific capture group slot
    /// offset that is being recorded. Each capturing group has two slots
    /// corresponding to the start and end of the matching portion of that
    /// group.
    Capture {
        next: StateID,
        pattern_id: PatternID,
        group_index: SmallIndex,
        slot: SmallIndex,
    },
    /// A state that cannot be transitioned out of. This is useful for cases
    /// where you want to prevent matching from occurring. For example, if your
    /// regex parser permits empty character classes, then one could choose a
    /// `Fail` state to represent it.
    Fail,
    /// A match state. There is at least one such occurrence of this state for
    /// each regex compiled into the NFA. The pattern ID in the state indicates
    /// which pattern matched.
    Match { pattern_id: PatternID },
}

impl State {
    /// Returns true if and only if this state contains one or more epsilon
    /// transitions.
    ///
    /// In practice, a state has no outgoing transitions (like `Match`), has
    /// only non-epsilon transitions (like `ByteRange`) or has only epsilon
    /// transitions (like `Union`).
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::{State, Transition},
    ///     util::primitives::{PatternID, StateID, SmallIndex},
    /// };
    ///
    /// // Capture states are epsilon transitions.
    /// let state = State::Capture {
    ///     next: StateID::ZERO,
    ///     pattern_id: PatternID::ZERO,
    ///     group_index: SmallIndex::ZERO,
    ///     slot: SmallIndex::ZERO,
    /// };
    /// assert!(state.is_epsilon());
    ///
    /// // ByteRange states are not.
    /// let state = State::ByteRange {
    ///     trans: Transition { start: b'a', end: b'z', next: StateID::ZERO },
    /// };
    /// assert!(!state.is_epsilon());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn is_epsilon(&self) -> bool {
        match *self {
            State::ByteRange { .. }
            | State::Sparse { .. }
            | State::Fail
            | State::Match { .. } => false,
            State::Look { .. }
            | State::Union { .. }
            | State::BinaryUnion { .. }
            | State::Capture { .. } => true,
        }
    }

    /// Returns the heap memory usage of this NFA state in bytes.
    fn memory_usage(&self) -> usize {
        match *self {
            State::ByteRange { .. }
            | State::Look { .. }
            | State::BinaryUnion { .. }
            | State::Capture { .. }
            | State::Match { .. }
            | State::Fail => 0,
            State::Sparse(SparseTransitions { ref transitions }) => {
                transitions.len() * mem::size_of::<Transition>()
            }
            State::Union { ref alternates } => {
                alternates.len() * mem::size_of::<StateID>()
            }
        }
    }

    /// Remap the transitions in this state using the given map. Namely, the
    /// given map should be indexed according to the transitions currently
    /// in this state.
    ///
    /// This is used during the final phase of the NFA compiler, which turns
    /// its intermediate NFA into the final NFA.
    fn remap(&mut self, remap: &[StateID]) {
        match *self {
            State::ByteRange { ref mut trans } => {
                trans.next = remap[trans.next]
            }
            State::Sparse(SparseTransitions { ref mut transitions }) => {
                for t in transitions.iter_mut() {
                    t.next = remap[t.next];
                }
            }
            State::Look { ref mut next, .. } => *next = remap[*next],
            State::Union { ref mut alternates } => {
                for alt in alternates.iter_mut() {
                    *alt = remap[*alt];
                }
            }
            State::BinaryUnion { ref mut alt1, ref mut alt2 } => {
                *alt1 = remap[*alt1];
                *alt2 = remap[*alt2];
            }
            State::Capture { ref mut next, .. } => *next = remap[*next],
            State::Fail => {}
            State::Match { .. } => {}
        }
    }
}

impl fmt::Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            State::ByteRange { ref trans } => trans.fmt(f),
            State::Sparse(SparseTransitions { ref transitions }) => {
                let rs = transitions
                    .iter()
                    .map(|t| format!("{:?}", t))
                    .collect::<Vec<String>>()
                    .join(", ");
                write!(f, "sparse({})", rs)
            }
            State::Look { ref look, next } => {
                write!(f, "{:?} => {:?}", look, next.as_usize())
            }
            State::Union { ref alternates } => {
                let alts = alternates
                    .iter()
                    .map(|id| format!("{:?}", id.as_usize()))
                    .collect::<Vec<String>>()
                    .join(", ");
                write!(f, "union({})", alts)
            }
            State::BinaryUnion { alt1, alt2 } => {
                write!(
                    f,
                    "binary-union({}, {})",
                    alt1.as_usize(),
                    alt2.as_usize()
                )
            }
            State::Capture { next, pattern_id, group_index, slot } => {
                write!(
                    f,
                    "capture(pid={:?}, group={:?}, slot={:?}) => {:?}",
                    pattern_id.as_usize(),
                    group_index.as_usize(),
                    slot.as_usize(),
                    next.as_usize(),
                )
            }
            State::Fail => write!(f, "FAIL"),
            State::Match { pattern_id } => {
                write!(f, "MATCH({:?})", pattern_id.as_usize())
            }
        }
    }
}

/// A collection of facts about an NFA.
///
/// There are no real cohesive principles behind what gets put in here. For
/// the most part, it is implementation driven. That is, what we put here
/// depends on what callers want to know cheaply. Most of these things could be
/// computed on the fly, but it's convenient to have cheap access.
#[derive(Clone, Copy, Debug, Default)]
struct Facts {
    has_capture: bool,
    has_look: bool,
    has_anchor: bool,
    has_word_boundary_unicode: bool,
    has_word_boundary_ascii: bool,
}

// THOUGHT: I wonder if it makes sense to add a DenseTransitions too? The main
// problem, of course, is that it would use a lot of space, especially without
// any sort of byte class optimization. (Although, perhaps we can make the
// byte class optimization for the NFA work..?) Naively, if DenseTransitions
// had space for 256 transitions, then it would take 256*sizeof(StateID)=1KB.
// Yikes. So we would really need to figure out *when* to use it, and that
// seems a little tricky...

/// A sequence of transitions used to represent a sparse state.
///
/// This is the primary representation of a [`Sparse`](State::Sparse) state.
/// It corresponds to a sorted sequence of transitions with non-overlapping
/// byte ranges. If the byte at the current position in the haystack matches
/// one of the byte ranges, then the finite state machine should take the
/// corresponding transition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SparseTransitions {
    /// The sorted sequence of non-overlapping transitions.
    pub transitions: Box<[Transition]>,
}

impl SparseTransitions {
    /// This follows the matching transition for a particular byte.
    ///
    /// The matching transition is found by looking for a matching byte
    /// range (there is at most one) corresponding to the position `at` in
    /// `haystack`.
    ///
    /// If `at >= haystack.len()`, then this returns `None`.
    pub fn matches(&self, haystack: &[u8], at: usize) -> Option<StateID> {
        haystack.get(at).and_then(|&b| self.matches_byte(b))
    }

    /// This follows the matching transition for any member of the alphabet.
    ///
    /// The matching transition is found by looking for a matching byte
    /// range (there is at most one) corresponding to the position `at` in
    /// `haystack`. If the given alphabet unit is [`EOI`](alphabet::Unit::EOI),
    /// then this always returns `None`.
    pub fn matches_unit(&self, unit: alphabet::Unit) -> Option<StateID> {
        unit.as_u8().map_or(None, |byte| self.matches_byte(byte))
    }

    /// This follows the matching transition for a particular byte.
    ///
    /// The matching transition is found by looking for a matching byte range
    /// (there is at most one) corresponding to the byte given.
    pub fn matches_byte(&self, byte: u8) -> Option<StateID> {
        for t in self.transitions.iter() {
            if t.start > byte {
                break;
            } else if t.matches_byte(byte) {
                return Some(t.next);
            }
        }
        None

        /*
        // This is an alternative implementation that uses binary search. In
        // some ad hoc experiments, like
        //
        //   smallishru=OpenSubtitles2018.raw.sample.smallish.ru
        //   regex-cli find nfa thompson pikevm -b "@$smallishru" '\b\w+\b'
        //
        // I could not observe any improvement, and in fact, things seemed to
        // be a bit slower. I can see an improvement in at least one benchmark:
        //
        //   allcpssmall=all-codepoints-utf8-10x
        //   regex-cli find nfa thompson pikevm @$allcpssmall '\pL{100}'
        //
        // Where total search time goes from 3.2s to 2.4s when using binary
        // search.
        self.transitions
            .binary_search_by(|t| {
                if t.end < byte {
                    core::cmp::Ordering::Less
                } else if t.start > byte {
                    core::cmp::Ordering::Greater
                } else {
                    core::cmp::Ordering::Equal
                }
            })
            .ok()
            .map(|i| self.transitions[i].next)
        */
    }
}

/// A single transition to another state.
///
/// This transition may only be followed if the current byte in the haystack
/// falls in the inclusive range of bytes specified.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct Transition {
    /// The start of the byte range.
    pub start: u8,
    /// The end of the byte range.
    pub end: u8,
    /// The identifier of the state to transition to.
    pub next: StateID,
}

impl Transition {
    /// Returns true if the position `at` in `haystack` falls in this
    /// transition's range of bytes.
    ///
    /// If `at >= haystack.len()`, then this returns `false`.
    pub fn matches(&self, haystack: &[u8], at: usize) -> bool {
        haystack.get(at).map_or(false, |&b| self.matches_byte(b))
    }

    /// Returns true if the given alphabet unit falls in this transition's
    /// range of bytes. If the given unit is [`EOI`](alphabet::Unit::EOI), then
    /// this returns `false`.
    pub fn matches_unit(&self, unit: alphabet::Unit) -> bool {
        unit.as_u8().map_or(false, |byte| self.matches_byte(byte))
    }

    /// Returns true if the given byte falls in this transition's range of
    /// bytes.
    pub fn matches_byte(&self, byte: u8) -> bool {
        self.start <= byte && byte <= self.end
    }
}

impl fmt::Debug for Transition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::util::escape::DebugByte;

        let Transition { start, end, next } = *self;
        if self.start == self.end {
            write!(f, "{:?} => {:?}", DebugByte(start), next.as_usize())
        } else {
            write!(
                f,
                "{:?}-{:?} => {:?}",
                DebugByte(start),
                DebugByte(end),
                next.as_usize(),
            )
        }
    }
}

/// A look-around assertion.
///
/// A simulation of the NFA can only move through conditional epsilon
/// transitions if the current position satisfies some look-around property.
/// Some assertions are look-behind (`StartLine`, `StartText`), some assertions
/// are look-ahead (`EndLine`, `EndText`) while other assertions are both
/// look-behind and look-ahead (`WordBoundary*`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Look {
    /// The previous position is either `\n` or the current position is the
    /// beginning of the haystack (at position `0`).
    StartLine = 1 << 0,
    /// The next position is either `\n` or the current position is the end of
    /// the haystack (at position `haystack.len()`).
    EndLine = 1 << 1,
    /// The current position is the beginning of the haystack (at position
    /// `0`).
    StartText = 1 << 2,
    /// The current position is the end of the haystack (at position
    /// `haystack.len()`).
    EndText = 1 << 3,
    /// When tested at position `i`, where `p=decode_utf8_rev(&haystack[..i])`
    /// and `n=decode_utf8(&haystack[i..])`, this assertion passes if and only
    /// if `is_word(p) != is_word(n)`. If `i=0`, then `is_word(p)=false` and if
    /// `i=haystack.len()`, then `is_word(n)=false`.
    WordBoundaryUnicode = 1 << 4,
    /// Same as for `WordBoundaryUnicode`, but requires that
    /// `is_word(p) == is_word(n)`.
    WordBoundaryUnicodeNegate = 1 << 5,
    /// When tested at position `i`, where `p=haystack[i-1]` and
    /// `n=haystack[i]`, this assertion passes if and only if `is_word(p)
    /// != is_word(n)`. If `i=0`, then `is_word(p)=false` and if
    /// `i=haystack.len()`, then `is_word(n)=false`.
    WordBoundaryAscii = 1 << 6,
    /// Same as for `WordBoundaryAscii`, but requires that
    /// `is_word(p) == is_word(n)`.
    ///
    /// Note that it is possible for this assertion to match at positions that
    /// split the UTF-8 encoding of a codepoint. For this reason, this may only
    /// be used when UTF-8 mode is disabled in the regex syntax.
    WordBoundaryAsciiNegate = 1 << 7,
}

impl Look {
    /// Returns true when the position `at` in `haystack` satisfies this
    /// look-around assertion.
    ///
    /// This panics if `at > haystack.len()`.
    pub fn matches(&self, haystack: &[u8], at: usize) -> bool {
        match *self {
            Look::StartLine => at == 0 || haystack[at - 1] == b'\n',
            Look::EndLine => at == haystack.len() || haystack[at] == b'\n',
            Look::StartText => at == 0,
            Look::EndText => at == haystack.len(),
            Look::WordBoundaryUnicode => {
                let word_before = utf8::is_word_char_rev(haystack, at);
                let word_after = utf8::is_word_char_fwd(haystack, at);
                word_before != word_after
            }
            Look::WordBoundaryUnicodeNegate => {
                // This is pretty subtle. Why do we need to do UTF-8 decoding
                // here? Well... at time of writing, the is_word_char_{fwd,rev}
                // routines will only return true if there is a valid UTF-8
                // encoding of a "word" codepoint, and false in every other
                // case (including invalid UTF-8). This means that in regions
                // of invalid UTF-8 (which might be a subset of valid UTF-8!),
                // it would result in \B matching. While this would be
                // questionable in the context of truly invalid UTF-8, it is
                // *certainly* wrong to report match boundaries that split the
                // encoding of a codepoint. So to work around this, we ensure
                // that we can decode a codepoint on either side of `at`. If
                // either direction fails, then we don't permit \B to match at
                // all.
                //
                // Now, this isn't exactly optimal from a perf perspective. We
                // could try and detect this in is_word_char_{fwd,rev}, but
                // it's not clear if it's worth it. \B is, after all, rarely
                // used.
                //
                // And in particular, we do *not* have to do this with \b,
                // because \b *requires* that at least one side of `at` be a
                // "word" codepoint, which in turn implies one side of `at`
                // must be valid UTF-8. This in turn implies that \b can never
                // split a valid UTF-8 encoding of a codepoint. In the case
                // where one side of `at` is truly invalid UTF-8 and the other
                // side IS a word codepoint, then we want \b to match since it
                // represents a valid UTF-8 boundary. It also makes sense. For
                // example, you'd want \b\w+\b to match 'abc' in '\xFFabc\xFF'.
                let word_before = at > 0
                    && match utf8::decode_last(&haystack[..at]) {
                        None | Some(Err(_)) => return false,
                        Some(Ok(_)) => utf8::is_word_char_rev(haystack, at),
                    };
                let word_after = at < haystack.len()
                    && match utf8::decode(&haystack[at..]) {
                        None | Some(Err(_)) => return false,
                        Some(Ok(_)) => utf8::is_word_char_fwd(haystack, at),
                    };
                word_before == word_after
            }
            Look::WordBoundaryAscii => {
                let word_before =
                    at > 0 && utf8::is_word_byte(haystack[at - 1]);
                let word_after =
                    at < haystack.len() && utf8::is_word_byte(haystack[at]);
                word_before != word_after
            }
            Look::WordBoundaryAsciiNegate => {
                let word_before =
                    at > 0 && utf8::is_word_byte(haystack[at - 1]);
                let word_after =
                    at < haystack.len() && utf8::is_word_byte(haystack[at]);
                word_before == word_after
            }
        }
    }

    /// Create a look-around assertion from its corresponding integer (as
    /// defined in `Look`). If the given integer does not correspond to any
    /// assertion, then `None` is returned.
    pub fn from_repr(n: u8) -> Option<Look> {
        match n {
            0b0000_0001 => Some(Look::StartLine),
            0b0000_0010 => Some(Look::EndLine),
            0b0000_0100 => Some(Look::StartText),
            0b0000_1000 => Some(Look::EndText),
            0b0001_0000 => Some(Look::WordBoundaryUnicode),
            0b0010_0000 => Some(Look::WordBoundaryUnicodeNegate),
            0b0100_0000 => Some(Look::WordBoundaryAscii),
            0b1000_0000 => Some(Look::WordBoundaryAsciiNegate),
            _ => None,
        }
    }

    /// Return the underlying representation of this look-around enumeration
    /// as an integer. Giving the return value to the [`Look::from_repr`]
    /// constructor is guaranteed to return the same look-around variant that
    /// one started with.
    pub fn as_repr(self) -> u8 {
        self as u8
    }

    /// Flip the look-around assertion to its equivalent for reverse searches.
    /// For example, `StartLine` gets translated to `EndLine`.
    pub fn reversed(&self) -> Look {
        match *self {
            Look::StartLine => Look::EndLine,
            Look::EndLine => Look::StartLine,
            Look::StartText => Look::EndText,
            Look::EndText => Look::StartText,
            Look::WordBoundaryUnicode => Look::WordBoundaryUnicode,
            Look::WordBoundaryUnicodeNegate => Look::WordBoundaryUnicodeNegate,
            Look::WordBoundaryAscii => Look::WordBoundaryAscii,
            Look::WordBoundaryAsciiNegate => Look::WordBoundaryAsciiNegate,
        }
    }

    /// Split up the given byte classes into equivalence classes in a way that
    /// is consistent with this look-around assertion.
    fn add_to_byteset(&self, set: &mut ByteClassSet) {
        match *self {
            Look::StartText | Look::EndText => {}
            Look::StartLine | Look::EndLine => {
                set.set_range(b'\n', b'\n');
            }
            Look::WordBoundaryUnicode
            | Look::WordBoundaryUnicodeNegate
            | Look::WordBoundaryAscii
            | Look::WordBoundaryAsciiNegate => {
                // We need to mark all ranges of bytes whose pairs result in
                // evaluating \b differently. This isn't technically correct
                // for Unicode word boundaries, but DFAs can't handle those
                // anyway, and thus, the byte classes don't need to either
                // since they are themselves only used in DFAs.
                //
                // FIXME: It seems like the calls to 'set_range' here are
                // completely invariant, which means we could just hard-code
                // them here without needing to write a loop. And we only need
                // to do this dance at most once per regex.
                //
                // FIXME: Is this correct for \B?
                let iswb = utf8::is_word_byte;
                // This unwrap is OK because we guard every use of 'asu8' with
                // a check that the input is <= 255.
                let asu8 = |b: u16| u8::try_from(b).unwrap();
                let mut b1: u16 = 0;
                let mut b2: u16;
                while b1 <= 255 {
                    b2 = b1 + 1;
                    while b2 <= 255 && iswb(asu8(b1)) == iswb(asu8(b2)) {
                        b2 += 1;
                    }
                    // The guards above guarantee that b2 can never get any
                    // bigger.
                    assert!(b2 <= 256);
                    // Subtracting 1 from b2 is always OK because it is always
                    // at least 1 greater than b1, and the assert above
                    // guarantees that the asu8 conversion will succeed.
                    set.set_range(asu8(b1), asu8(b2.checked_sub(1).unwrap()));
                    b1 = b2;
                }
            }
        }
    }
}

/// An iterator over all pattern IDs in an NFA.
///
/// This iterator is created by [`NFA::patterns`].
///
/// The lifetime parameter `'a` refers to the lifetime of the NFA from which
/// this pattern iterator was created.
#[derive(Debug)]
pub struct PatternIter<'a> {
    it: PatternIDIter,
    /// We explicitly associate a lifetime with this iterator even though we
    /// don't actually borrow anything from the NFA. We do this for backward
    /// compatibility purposes. If we ever do need to borrow something from
    /// the NFA, then we can and just get rid of this marker without breaking
    /// the public API.
    _marker: core::marker::PhantomData<&'a ()>,
}

impl<'a> Iterator for PatternIter<'a> {
    type Item = PatternID;

    fn next(&mut self) -> Option<PatternID> {
        self.it.next()
    }
}

/// The span offsets of capturing groups after a match has been found.
///
/// This type represents the primary output of NFA-based regex engines like
/// the [`PikeVM`](crate::nfa::thompson::pikevm::PikeVM). When a match occurs,
/// it will at minimum contain the [`PatternID`] of the pattern that matched.
/// Depending upon how it was constructed, it may also contain the start/end
/// offsets of the entire match of the pattern and the start/end offsets of
/// each capturing group that participated in the match.
///
/// Values of this type are always created for a specific NFA. It is
/// unspecified behavior to use a `Captures` value in a search with any NFA
/// other than the one it was created for.
///
/// # Constructors
///
/// There are three constructors for this type that control what kind of
/// information is available upon a match:
///
/// * [`Captures::new`]: Will store overall pattern match offsets in addition
/// to the offsets of capturing groups that participated in the match.
/// * [`Captures::new_for_matches_only`]: Will store only the overall pattern
/// match offsets. The offsets of capturing groups (even ones that participated
/// in the match) are not available.
/// * [`Captures::empty`]: Will only store the pattern ID that matched. No
/// match offsets are available at all.
///
/// If you aren't sure which to choose, then pick the first one. The first one
/// is what the convenience routine,
/// [`PikeVM::create_captures`](crate::nfa::thompson::pikevm::PikeVM::create_captures),
/// will use automatically.
///
/// The main difference between these choices is performance. Namely, if you
/// ask for _less_ information, then the execution of an NFA search may be able
/// to run more quickly.
///
/// # Notes
///
/// It is worth pointing out that while this type is necessary to use if you
/// want to use one of the NFA regex engines provided by this crate, it is
/// _not_ necessary to use this type if you want to build your own NFA-based
/// regex engine. That is, by design, the implementation of this type does not
/// use any private implementation details of `NFA`.
///
/// # Example
///
/// This example shows how to parse a simple date and extract the components of
/// the date via capturing groups:
///
/// ```
/// use regex_automata::{nfa::thompson::pikevm::PikeVM, Span};
///
/// let vm = PikeVM::new(r"^([0-9]{4})-([0-9]{2})-([0-9]{2})$")?;
/// let (mut cache, mut caps) = (vm.create_cache(), vm.create_captures());
///
/// vm.find(&mut cache, "2010-03-14", &mut caps);
/// assert!(caps.is_match());
/// assert_eq!(Some(Span::from(0..4)), caps.get_group(1));
/// assert_eq!(Some(Span::from(5..7)), caps.get_group(2));
/// assert_eq!(Some(Span::from(8..10)), caps.get_group(3));
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Example: named capturing groups
///
/// This example is like the one above, but leverages the ability to name
/// capturing groups in order to make the code a bit clearer:
///
/// ```
/// use regex_automata::{nfa::thompson::pikevm::PikeVM, Span};
///
/// let vm = PikeVM::new(r"^(?P<y>[0-9]{4})-(?P<m>[0-9]{2})-(?P<d>[0-9]{2})$")?;
/// let (mut cache, mut caps) = (vm.create_cache(), vm.create_captures());
///
/// vm.find(&mut cache, "2010-03-14", &mut caps);
/// assert!(caps.is_match());
/// assert_eq!(Some(Span::from(0..4)), caps.get_group_by_name("y"));
/// assert_eq!(Some(Span::from(5..7)), caps.get_group_by_name("m"));
/// assert_eq!(Some(Span::from(8..10)), caps.get_group_by_name("d"));
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone)]
pub struct Captures {
    // FIXME: Consider (again) re-orienting this type such that it only has
    // enough slots to store the pattern with the greatest number of groups.
    // Since the public API only supports accessing groups for a specific
    // pattern (other than the lower level slot API), this is OK.
    //
    // Main advantage is that accessing a group from a group index becomes much
    // more straightforward. You don't need any lookup tables. You just do
    // (group_index * 2, group_index * 2 + 1).
    //
    // Main downside is that you can't write directly to a 'Captures' given a
    // slot index from an 'State::Capture'. The PikeVM doesn't even try that
    // anyway, but the backtracker does. So the backtracker will need an extra
    // copy I guess. Both will need some way to copy matching slots values to a
    // 'Captures', but I think all of the necessary info is already on the NFA.
    nfa: NFA,
    pid: Option<PatternID>,
    slots: Vec<Option<NonMaxUsize>>,
}

impl Captures {
    /// Create new storage for the offsets of all matching capturing groups.
    ///
    /// This routine provides the most information for matches---namely, the
    /// match spans of capturing groups---but also requires the NFA search
    /// routines to do the most work.
    ///
    /// It is unspecified behavior to use the returned `Captures` value in
    /// a search with any NFA other than the one that is provided to this
    /// constructor.
    ///
    /// # Example
    ///
    /// This example shows that all capturing groups---but only ones that
    /// participated in a match---are available to query after a match has
    /// been found:
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::{pikevm::PikeVM, Captures},
    ///     Span, Match,
    /// };
    ///
    /// let vm = PikeVM::new(
    ///     r"^(?:(?P<lower>[a-z]+)|(?P<upper>[A-Z]+))(?P<digits>[0-9]+)$",
    /// )?;
    /// let mut cache = vm.create_cache();
    /// let mut caps = Captures::new(vm.get_nfa().clone());
    ///
    /// vm.find(&mut cache, "ABC123", &mut caps);
    /// assert!(caps.is_match());
    /// assert_eq!(Some(Match::must(0, 0..6)), caps.get_match());
    /// // The 'lower' group didn't match, so it won't have any offsets.
    /// assert_eq!(None, caps.get_group_by_name("lower"));
    /// assert_eq!(Some(Span::from(0..3)), caps.get_group_by_name("upper"));
    /// assert_eq!(Some(Span::from(3..6)), caps.get_group_by_name("digits"));
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(nfa: NFA) -> Captures {
        let slots = nfa.group_info().slot_len();
        Captures { nfa, pid: None, slots: vec![None; slots] }
    }

    /// Create new storage for only the full match spans of a pattern. This
    /// does not include any capturing group offsets.
    ///
    /// It is unspecified behavior to use the returned `Captures` value in
    /// a search with any NFA other than the one that is provided to this
    /// constructor.
    ///
    /// # Example
    ///
    /// This example shows that only overall match offsets are reported when
    /// this constructor is used. Accessing any capturing groups other than
    /// the 0th will always return `None`.
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::{pikevm::PikeVM, Captures},
    ///     Match,
    /// };
    ///
    /// let vm = PikeVM::new(
    ///     r"^(?:(?P<lower>[a-z]+)|(?P<upper>[A-Z]+))(?P<digits>[0-9]+)$",
    /// )?;
    /// let mut cache = vm.create_cache();
    /// let mut caps = Captures::new_for_matches_only(vm.get_nfa().clone());
    ///
    /// vm.find(&mut cache, "ABC123", &mut caps);
    /// assert!(caps.is_match());
    /// assert_eq!(Some(Match::must(0, 0..6)), caps.get_match());
    /// // We didn't ask for capturing group offsets, so they aren't available.
    /// assert_eq!(None, caps.get_group_by_name("lower"));
    /// assert_eq!(None, caps.get_group_by_name("upper"));
    /// assert_eq!(None, caps.get_group_by_name("digits"));
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new_for_matches_only(nfa: NFA) -> Captures {
        // This is OK because we know there are at least this many slots,
        // and NFA construction guarantees that the number of slots fits
        // into a usize.
        let slots = nfa.pattern_len().checked_mul(2).unwrap();
        Captures { nfa, pid: None, slots: vec![None; slots] }
    }

    /// Create new storage for only tracking which pattern matched. No offsets
    /// are stored at all.
    ///
    /// It is unspecified behavior to use the returned `Captures` value in
    /// a search with any NFA other than the one that is provided to this
    /// constructor.
    ///
    /// # Example
    ///
    /// This example shows that only the pattern that matched can be accessed
    /// from a `Captures` value created via this constructor.
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::{pikevm::PikeVM, Captures},
    ///     PatternID,
    /// };
    ///
    /// let vm = PikeVM::new_many(&[r"[a-z]+", r"[A-Z]+"])?;
    /// let mut cache = vm.create_cache();
    /// let mut caps = Captures::empty(vm.get_nfa().clone());
    ///
    /// vm.find(&mut cache, "aABCz", &mut caps);
    /// assert!(caps.is_match());
    /// assert_eq!(Some(PatternID::must(0)), caps.pattern());
    /// // We didn't ask for any offsets, so they aren't available.
    /// assert_eq!(None, caps.get_match());
    ///
    /// vm.find(&mut cache, &"aABCz"[1..], &mut caps);
    /// assert!(caps.is_match());
    /// assert_eq!(Some(PatternID::must(1)), caps.pattern());
    /// // We didn't ask for any offsets, so they aren't available.
    /// assert_eq!(None, caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn empty(nfa: NFA) -> Captures {
        Captures { nfa, pid: None, slots: vec![] }
    }

    /// Returns true if and only if this capturing group represents a match.
    ///
    /// This is a convenience routine for `caps.pattern().is_some()`.
    ///
    /// # Example
    ///
    /// When using the PikeVM (for example), the lightest weight way of
    /// detecting whether a match exists is to create capturing groups that
    /// only track the ID of the pattern that match (if any):
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::{Captures, pikevm::PikeVM}};
    ///
    /// let vm = PikeVM::new(r"[a-z]+")?;
    /// let mut cache = vm.create_cache();
    /// let mut caps = Captures::empty(vm.get_nfa().clone());
    ///
    /// vm.find(&mut cache, "aABCz", &mut caps);
    /// assert!(caps.is_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn is_match(&self) -> bool {
        self.pid.is_some()
    }

    /// Returns the identifier of the pattern that matched when this
    /// capturing group represents a match. If no match was found, then this
    /// always returns `None`.
    ///
    /// This returns a pattern ID in precisely the cases in which `is_match`
    /// returns `true`.
    ///
    /// # Example
    ///
    /// When using the PikeVM (for example), the lightest weight way of
    /// detecting which pattern matched is to create capturing groups that only
    /// track the ID of the pattern that match (if any):
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::{Captures, pikevm::PikeVM},
    ///     PatternID,
    /// };
    ///
    /// let vm = PikeVM::new_many(&[r"[a-z]+", r"[A-Z]+"])?;
    /// let mut cache = vm.create_cache();
    /// let mut caps = Captures::empty(vm.get_nfa().clone());
    ///
    /// vm.find(&mut cache, "ABC", &mut caps);
    /// assert_eq!(Some(PatternID::must(1)), caps.pattern());
    /// // Recall that offsets are only available when using a non-empty
    /// // Captures value. So even though a match occurred, this returns None!
    /// assert_eq!(None, caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn pattern(&self) -> Option<PatternID> {
        self.pid
    }

    /// Returns the pattern ID and the span of the match, if one occurred.
    ///
    /// This always returns `None` when `Captures` was created with
    /// [`Captures::empty`], even if a match was found.
    ///
    /// If this routine returns a non-`None` value, then `is_match` is
    /// guaranteed to return `true` and `pattern` is also guaranteed to return
    /// a non-`None` value.
    ///
    /// # Example
    ///
    /// This example shows how to get the full match from a search:
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::pikevm::PikeVM, Match};
    ///
    /// let vm = PikeVM::new_many(&[r"[a-z]+", r"[A-Z]+"])?;
    /// let (mut cache, mut caps) = (vm.create_cache(), vm.create_captures());
    ///
    /// vm.find(&mut cache, "ABC", &mut caps);
    /// assert_eq!(Some(Match::must(1, 0..3)), caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn get_match(&self) -> Option<Match> {
        Some(Match::new(self.pattern()?, self.get_group(0)?))
    }

    /// Returns the span of a capturing group match corresponding to the group
    /// index given, only if both the overall pattern matched and the capturing
    /// group participated in that match.
    ///
    /// This returns `None` if `index` is invalid. `index` is valid if and
    /// only if it's less than [`Captures::group_len`].
    ///
    /// This always returns `None` when `Captures` was created with
    /// [`Captures::empty`], even if a match was found. This also always
    /// returns `None` for any `index > 0` when `Captures` was created with
    /// [`Captures::new_for_matches_only`].
    ///
    /// If this routine returns a non-`None` value, then `is_match` is
    /// guaranteed to return `true`, `pattern` is guaranteed to return a
    /// non-`None` value and `get_match` is guaranteed to return a non-`None`
    /// value.
    ///
    /// Note that for NFAs built by a [`Compiler`] with capturing groups
    /// enabled, the 0th capture group will always return the same span as
    /// the span returned by `get_match`. This is because the 0th capture
    /// group always corresponds to the entirety of the pattern's match.
    /// (It is similarly always unnamed because it is implicit.) This isn't
    /// necessarily true of all NFAs however, since NFAs can be hand-compiled
    /// via a [`Builder`], which isn't technically forced to make the 0th
    /// capturing group always correspond to the entire match.
    ///
    /// # Example
    ///
    /// This example shows how to get the capturing groups, by index, from a
    /// match:
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::pikevm::PikeVM, Span, Match};
    ///
    /// let vm = PikeVM::new(r"^(?P<first>\pL+)\s+(?P<last>\pL+)$")?;
    /// let (mut cache, mut caps) = (vm.create_cache(), vm.create_captures());
    ///
    /// vm.find(&mut cache, "Bruce Springsteen", &mut caps);
    /// assert_eq!(Some(Match::must(0, 0..17)), caps.get_match());
    /// assert_eq!(Some(Span::from(0..5)), caps.get_group(1));
    /// assert_eq!(Some(Span::from(6..17)), caps.get_group(2));
    /// // Looking for a non-existent capturing group will return None:
    /// assert_eq!(None, caps.get_group(3));
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn get_group(&self, index: usize) -> Option<Span> {
        let pid = self.pattern()?;
        let (slot_start, slot_end) =
            self.nfa.group_info().slots(pid, index)?;
        let start = self.slots.get(slot_start).copied()??;
        let end = self.slots.get(slot_end).copied()??;
        Some(Span { start: start.get(), end: end.get() })
    }

    /// Returns the span of a capturing group match corresponding to the group
    /// name given, only if both the overall pattern matched and the capturing
    /// group participated in that match.
    ///
    /// This returns `None` if `name` does not correspond to a valid capturing
    /// group for the pattern that matched.
    ///
    /// This always returns `None` when `Captures` was created with
    /// [`Captures::empty`], even if a match was found. This also always
    /// returns `None` for any `index > 0` when `Captures` was created with
    /// [`Captures::new_for_matches_only`].
    ///
    /// If this routine returns a non-`None` value, then `is_match` is
    /// guaranteed to return `true`, `pattern` is guaranteed to return a
    /// non-`None` value and `get_match` is guaranteed to return a non-`None`
    /// value.
    ///
    /// # Example
    ///
    /// This example shows how to get the capturing groups, by name, from a
    /// match:
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::pikevm::PikeVM, Span, Match};
    ///
    /// let vm = PikeVM::new(r"^(?P<first>\pL+)\s+(?P<last>\pL+)$")?;
    /// let (mut cache, mut caps) = (vm.create_cache(), vm.create_captures());
    ///
    /// vm.find(&mut cache, "Bruce Springsteen", &mut caps);
    /// assert_eq!(Some(Match::must(0, 0..17)), caps.get_match());
    /// assert_eq!(Some(Span::from(0..5)), caps.get_group_by_name("first"));
    /// assert_eq!(Some(Span::from(6..17)), caps.get_group_by_name("last"));
    /// // Looking for a non-existent capturing group will return None:
    /// assert_eq!(None, caps.get_group_by_name("middle"));
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn get_group_by_name(&self, name: &str) -> Option<Span> {
        let index = self.nfa.group_info().to_index(self.pattern()?, name)?;
        self.get_group(index)
    }

    /// Returns an iterator of possible spans for every capturing group in the
    /// matching pattern.
    ///
    /// If this `Captures` value does not correspond to a match, then the
    /// iterator returned yields no elements.
    ///
    /// Note that the iterator returned yields elements of type `Option<Span>`.
    /// A span is present if and only if it corresponds to a capturing group
    /// that participated in a match.
    ///
    /// # Example
    ///
    /// This example shows how to collect all capturing groups:
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::pikevm::PikeVM, Span};
    ///
    /// let vm = PikeVM::new(
    ///     // Matches first/last names, with an optional middle name.
    ///     r"^(?P<first>\pL+)\s+(?:(?P<middle>\pL+)\s+)?(?P<last>\pL+)$",
    /// )?;
    /// let (mut cache, mut caps) = (vm.create_cache(), vm.create_captures());
    ///
    /// vm.find(&mut cache, "Harry James Potter", &mut caps);
    /// assert!(caps.is_match());
    /// let groups: Vec<Option<Span>> = caps.iter().collect();
    /// assert_eq!(groups, vec![
    ///     Some(Span::from(0..18)),
    ///     Some(Span::from(0..5)),
    ///     Some(Span::from(6..11)),
    ///     Some(Span::from(12..18)),
    /// ]);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// This example uses the same regex as the previous example, but with a
    /// haystack that omits the middle name. This results in a capturing group
    /// that is present in the elements yielded by the iterator but without a
    /// match:
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::pikevm::PikeVM, Span};
    ///
    /// let vm = PikeVM::new(
    ///     // Matches first/last names, with an optional middle name.
    ///     r"^(?P<first>\pL+)\s+(?:(?P<middle>\pL+)\s+)?(?P<last>\pL+)$",
    /// )?;
    /// let (mut cache, mut caps) = (vm.create_cache(), vm.create_captures());
    ///
    /// vm.find(&mut cache, "Harry Potter", &mut caps);
    /// assert!(caps.is_match());
    /// let groups: Vec<Option<Span>> = caps.iter().collect();
    /// assert_eq!(groups, vec![
    ///     Some(Span::from(0..12)),
    ///     Some(Span::from(0..5)),
    ///     None,
    ///     Some(Span::from(6..12)),
    /// ]);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn iter(&self) -> CapturesPatternIter<'_> {
        let names = self
            .pattern()
            .map(|pid| self.nfa.group_info().pattern_names(pid).enumerate());
        CapturesPatternIter { caps: self, names }
    }

    /// Return the total number of capturing groups for the matching pattern.
    ///
    /// If this `Captures` value does not correspond to a match, then this
    /// always returns `0`.
    ///
    /// This always returns the same number of elements yielded by
    /// [`Captures::iter`]. That is, the number includes capturing groups even
    /// if they don't participate in the match.
    ///
    /// # Example
    ///
    /// This example shows how to count the total number of capturing groups
    /// associated with a pattern. Notice that it includes groups that did not
    /// participate in a match (just like `Captures::iter` does).
    ///
    /// ```
    /// use regex_automata::nfa::thompson::pikevm::PikeVM;
    ///
    /// let vm = PikeVM::new(
    ///     // Matches first/last names, with an optional middle name.
    ///     r"^(?P<first>\pL+)\s+(?:(?P<middle>\pL+)\s+)?(?P<last>\pL+)$",
    /// )?;
    /// let (mut cache, mut caps) = (vm.create_cache(), vm.create_captures());
    ///
    /// vm.find(&mut cache, "Harry Potter", &mut caps);
    /// assert_eq!(4, caps.group_len());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn group_len(&self) -> usize {
        let pid = match self.pattern() {
            None => return 0,
            Some(pid) => pid,
        };
        self.nfa.group_info().group_len(pid)
    }
}

/// Lower level "slot" oriented APIs. One does not typically need to use these
/// when executing a search. They are instead mostly intended for folks that
/// are writing their own NFA regex engine while reusing this `Captures` type.
impl Captures {
    /// Clear this `Captures` value.
    ///
    /// After clearing, all slots inside this `Captures` value will be set to
    /// `None`. Similarly, any pattern ID that it was previously associated
    /// with (for a match) is erased.
    ///
    /// It is not usually necessary to call this routine. Namely, a `Captures`
    /// value only provides high level access to the capturing groups of the
    /// pattern that matched, and only low level access to individual slots.
    /// Thus, even if slots corresponding to groups that aren't associated
    /// with the matching pattern are set, then it won't impact the higher
    /// level APIs. Namely, higher level APIs like [`Captures::get_group`] will
    /// return `None` if no pattern ID is present, even if there are spans set
    /// in the underlying slots.
    ///
    /// Thus, to "clear" a `Captures` value of a match, it is usually only
    /// necessary to call [`Captures::set_pattern`] with `None`.
    ///
    /// # Example
    ///
    /// This example shows what happens when a `Captures` value is cleared.
    ///
    /// ```
    /// use regex_automata::nfa::thompson::pikevm::PikeVM;
    ///
    /// let vm = PikeVM::new(r"^(?P<first>\pL+)\s+(?P<last>\pL+)$")?;
    /// let (mut cache, mut caps) = (vm.create_cache(), vm.create_captures());
    ///
    /// vm.find(&mut cache, "Bruce Springsteen", &mut caps);
    /// assert!(caps.is_match());
    /// let slots: Vec<Option<usize>> =
    ///     caps.slots().iter().map(|s| s.map(|x| x.get())).collect();
    /// // Note that the following ordering is not considered an API guarantee.
    /// // The only valid way of mapping a capturing group index to a slot
    /// // index is with the NFA::slot or NFA::slots routines.
    /// assert_eq!(slots, vec![
    ///     Some(0),
    ///     Some(17),
    ///     Some(0),
    ///     Some(5),
    ///     Some(6),
    ///     Some(17),
    /// ]);
    ///
    /// // Now clear the slots. Everything is gone and it is no longer a match.
    /// caps.clear();
    /// assert!(!caps.is_match());
    /// let slots: Vec<Option<usize>> =
    ///     caps.slots().iter().map(|s| s.map(|x| x.get())).collect();
    /// assert_eq!(slots, vec![
    ///     None,
    ///     None,
    ///     None,
    ///     None,
    ///     None,
    ///     None,
    /// ]);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.pid = None;
        for slot in self.slots.iter_mut() {
            *slot = None;
        }
    }

    /// Set the pattern on this `Captures` value.
    ///
    /// When the pattern ID is `None`, then this `Captures` value does not
    /// correspond to a match (`is_match` will return `false`). Otherwise, it
    /// corresponds to a match.
    ///
    /// # Example
    ///
    /// This example shows that `set_pattern` merely overwrites the pattern ID.
    /// It does not actually change the underlying slot values.
    ///
    /// ```
    /// use regex_automata::nfa::thompson::pikevm::PikeVM;
    ///
    /// let vm = PikeVM::new(r"^(?P<first>\pL+)\s+(?P<last>\pL+)$")?;
    /// let (mut cache, mut caps) = (vm.create_cache(), vm.create_captures());
    ///
    /// vm.find(&mut cache, "Bruce Springsteen", &mut caps);
    /// assert!(caps.is_match());
    /// assert!(caps.pattern().is_some());
    /// let slots: Vec<Option<usize>> =
    ///     caps.slots().iter().map(|s| s.map(|x| x.get())).collect();
    /// // Note that the following ordering is not considered an API guarantee.
    /// // The only valid way of mapping a capturing group index to a slot
    /// // index is with the NFA::slot or NFA::slots routines.
    /// assert_eq!(slots, vec![
    ///     Some(0),
    ///     Some(17),
    ///     Some(0),
    ///     Some(5),
    ///     Some(6),
    ///     Some(17),
    /// ]);
    ///
    /// // Now set the pattern to None. Note that the slot values remain.
    /// caps.set_pattern(None);
    /// assert!(!caps.is_match());
    /// assert!(!caps.pattern().is_some());
    /// let slots: Vec<Option<usize>> =
    ///     caps.slots().iter().map(|s| s.map(|x| x.get())).collect();
    /// // Note that the following ordering is not considered an API guarantee.
    /// // The only valid way of mapping a capturing group index to a slot
    /// // index is with the NFA::slot or NFA::slots routines.
    /// assert_eq!(slots, vec![
    ///     Some(0),
    ///     Some(17),
    ///     Some(0),
    ///     Some(5),
    ///     Some(6),
    ///     Some(17),
    /// ]);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn set_pattern(&mut self, pid: Option<PatternID>) {
        self.pid = pid;
    }

    #[inline]
    pub fn slots(&self) -> &[Option<NonMaxUsize>] {
        &self.slots
    }

    #[inline]
    pub fn slots_mut(&mut self) -> &mut [Option<NonMaxUsize>] {
        &mut self.slots
    }
}

impl core::fmt::Debug for Captures {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        let mut dstruct = f.debug_struct("Captures");
        dstruct.field("pid", &self.pid);
        if let Some(pid) = self.pid {
            dstruct.field("spans", &CapturesDebugMap { pid, caps: self });
        }
        dstruct.finish()
    }
}

/// A little helper type to provide a nice map-like debug representation for
/// our capturing group spans.
struct CapturesDebugMap<'a> {
    pid: PatternID,
    caps: &'a Captures,
}

impl<'a> core::fmt::Debug for CapturesDebugMap<'a> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        let mut map = f.debug_map();
        let mut names = self.caps.nfa.group_info().pattern_names(self.pid);
        for (group_index, maybe_name) in names.enumerate() {
            let span = self.caps.get_group(group_index);
            let debug_span: &dyn core::fmt::Debug = match span {
                None => &None::<()>,
                Some(ref span) => span,
            };
            if let Some(name) = maybe_name {
                map.entry(&format!("{}/{}", group_index, name), debug_span);
            } else {
                map.entry(&group_index, debug_span);
            }
        }
        map.finish()
    }
}

/// An iterator over all capturing groups in a `Captures` value.
///
/// This iterator includes capturing groups that did not participate in a
/// match. See the [`Captures::iter`] method documentation for more details
/// and examples.
///
/// The lifetime parameter `'a` refers to the lifetime of the underlying
/// `Captures` value.
#[derive(Debug)]
pub struct CapturesPatternIter<'a> {
    caps: &'a Captures,
    names: Option<core::iter::Enumerate<captures::GroupInfoPatternNames<'a>>>,
}

impl<'a> Iterator for CapturesPatternIter<'a> {
    type Item = Option<Span>;

    fn next(&mut self) -> Option<Option<Span>> {
        let (group_index, _) = self.names.as_mut()?.next()?;
        Some(self.caps.get_group(group_index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{nfa::thompson::pikevm::PikeVM, Input};

    // This asserts that an NFA state doesn't have its size changed. It is
    // *really* easy to accidentally increase the size, and thus potentially
    // dramatically increase the memory usage of every NFA.
    //
    // This assert doesn't mean we absolutely cannot increase the size of an
    // NFA state. We can. It's just here to make sure we do it knowingly and
    // intentionally.
    #[test]
    fn state_has_small_size() {
        assert_eq!(24, core::mem::size_of::<State>());
    }

    #[test]
    fn always_match() {
        let vm = PikeVM::new_from_nfa(NFA::always_match()).unwrap();
        let mut cache = vm.create_cache();
        let mut caps = vm.create_captures();
        let mut find = |haystack, start, end| {
            let input = Input::new(haystack).range(start..end);
            vm.search(&mut cache, &input, &mut caps);
            caps.get_match().map(|m| m.end())
        };

        assert_eq!(Some(0), find("", 0, 0));
        assert_eq!(Some(0), find("a", 0, 1));
        assert_eq!(Some(1), find("a", 1, 1));
        assert_eq!(Some(0), find("ab", 0, 2));
        assert_eq!(Some(1), find("ab", 1, 2));
        assert_eq!(Some(2), find("ab", 2, 2));
    }

    #[test]
    fn never_match() {
        let vm = PikeVM::new_from_nfa(NFA::never_match()).unwrap();
        let mut cache = vm.create_cache();
        let mut caps = vm.create_captures();
        let mut find = |haystack, start, end| {
            let input = Input::new(haystack).range(start..end);
            vm.search(&mut cache, &input, &mut caps);
            caps.get_match().map(|m| m.end())
        };

        assert_eq!(None, find("", 0, 0));
        assert_eq!(None, find("a", 0, 1));
        assert_eq!(None, find("a", 1, 1));
        assert_eq!(None, find("ab", 0, 2));
        assert_eq!(None, find("ab", 1, 2));
        assert_eq!(None, find("ab", 2, 2));
    }

    #[test]
    fn look_matches_start_line() {
        let look = Look::StartLine;

        assert!(look.matches(B(""), 0));
        assert!(look.matches(B("\n"), 0));
        assert!(look.matches(B("\n"), 1));
        assert!(look.matches(B("a"), 0));
        assert!(look.matches(B("\na"), 1));

        assert!(!look.matches(B("a"), 1));
        assert!(!look.matches(B("a\na"), 1));
    }

    #[test]
    fn look_matches_end_line() {
        let look = Look::EndLine;

        assert!(look.matches(B(""), 0));
        assert!(look.matches(B("\n"), 1));
        assert!(look.matches(B("\na"), 0));
        assert!(look.matches(B("\na"), 2));
        assert!(look.matches(B("a\na"), 1));

        assert!(!look.matches(B("a"), 0));
        assert!(!look.matches(B("\na"), 1));
        assert!(!look.matches(B("a\na"), 0));
        assert!(!look.matches(B("a\na"), 2));
    }

    #[test]
    fn look_matches_start_text() {
        let look = Look::StartText;

        assert!(look.matches(B(""), 0));
        assert!(look.matches(B("\n"), 0));
        assert!(look.matches(B("a"), 0));

        assert!(!look.matches(B("\n"), 1));
        assert!(!look.matches(B("\na"), 1));
        assert!(!look.matches(B("a"), 1));
        assert!(!look.matches(B("a\na"), 1));
    }

    #[test]
    fn look_matches_end_text() {
        let look = Look::EndText;

        assert!(look.matches(B(""), 0));
        assert!(look.matches(B("\n"), 1));
        assert!(look.matches(B("\na"), 2));

        assert!(!look.matches(B("\na"), 0));
        assert!(!look.matches(B("a\na"), 1));
        assert!(!look.matches(B("a"), 0));
        assert!(!look.matches(B("\na"), 1));
        assert!(!look.matches(B("a\na"), 0));
        assert!(!look.matches(B("a\na"), 2));
    }

    #[test]
    #[cfg(not(miri))]
    fn look_matches_word_unicode() {
        let look = Look::WordBoundaryUnicode;

        // \xF0\x9D\x9B\x83 = 𝛃 (in \w)
        // \xF0\x90\x86\x80 = 𐆀 (not in \w)

        // Simple ASCII word boundaries.
        assert!(look.matches(B("a"), 0));
        assert!(look.matches(B("a"), 1));
        assert!(look.matches(B("a "), 1));
        assert!(look.matches(B(" a "), 1));
        assert!(look.matches(B(" a "), 2));

        // Unicode word boundaries with a non-ASCII codepoint.
        assert!(look.matches(B("𝛃"), 0));
        assert!(look.matches(B("𝛃"), 4));
        assert!(look.matches(B("𝛃 "), 4));
        assert!(look.matches(B(" 𝛃 "), 1));
        assert!(look.matches(B(" 𝛃 "), 5));

        // Unicode word boundaries between non-ASCII codepoints.
        assert!(look.matches(B("𝛃𐆀"), 0));
        assert!(look.matches(B("𝛃𐆀"), 4));

        // Non word boundaries for ASCII.
        assert!(!look.matches(B(""), 0));
        assert!(!look.matches(B("ab"), 1));
        assert!(!look.matches(B("a "), 2));
        assert!(!look.matches(B(" a "), 0));
        assert!(!look.matches(B(" a "), 3));

        // Non word boundaries with a non-ASCII codepoint.
        assert!(!look.matches(B("𝛃b"), 4));
        assert!(!look.matches(B("𝛃 "), 5));
        assert!(!look.matches(B(" 𝛃 "), 0));
        assert!(!look.matches(B(" 𝛃 "), 6));
        assert!(!look.matches(B("𝛃"), 1));
        assert!(!look.matches(B("𝛃"), 2));
        assert!(!look.matches(B("𝛃"), 3));

        // Non word boundaries with non-ASCII codepoints.
        assert!(!look.matches(B("𝛃𐆀"), 1));
        assert!(!look.matches(B("𝛃𐆀"), 2));
        assert!(!look.matches(B("𝛃𐆀"), 3));
        assert!(!look.matches(B("𝛃𐆀"), 5));
        assert!(!look.matches(B("𝛃𐆀"), 6));
        assert!(!look.matches(B("𝛃𐆀"), 7));
        assert!(!look.matches(B("𝛃𐆀"), 8));
    }

    #[test]
    fn look_matches_word_ascii() {
        let look = Look::WordBoundaryAscii;

        // \xF0\x9D\x9B\x83 = 𝛃 (in \w)
        // \xF0\x90\x86\x80 = 𐆀 (not in \w)

        // Simple ASCII word boundaries.
        assert!(look.matches(B("a"), 0));
        assert!(look.matches(B("a"), 1));
        assert!(look.matches(B("a "), 1));
        assert!(look.matches(B(" a "), 1));
        assert!(look.matches(B(" a "), 2));

        // Unicode word boundaries with a non-ASCII codepoint. Since this is
        // an ASCII word boundary, none of these match.
        assert!(!look.matches(B("𝛃"), 0));
        assert!(!look.matches(B("𝛃"), 4));
        assert!(!look.matches(B("𝛃 "), 4));
        assert!(!look.matches(B(" 𝛃 "), 1));
        assert!(!look.matches(B(" 𝛃 "), 5));

        // Unicode word boundaries between non-ASCII codepoints. Again, since
        // this is an ASCII word boundary, none of these match.
        assert!(!look.matches(B("𝛃𐆀"), 0));
        assert!(!look.matches(B("𝛃𐆀"), 4));

        // Non word boundaries for ASCII.
        assert!(!look.matches(B(""), 0));
        assert!(!look.matches(B("ab"), 1));
        assert!(!look.matches(B("a "), 2));
        assert!(!look.matches(B(" a "), 0));
        assert!(!look.matches(B(" a "), 3));

        // Non word boundaries with a non-ASCII codepoint.
        assert!(look.matches(B("𝛃b"), 4));
        assert!(!look.matches(B("𝛃 "), 5));
        assert!(!look.matches(B(" 𝛃 "), 0));
        assert!(!look.matches(B(" 𝛃 "), 6));
        assert!(!look.matches(B("𝛃"), 1));
        assert!(!look.matches(B("𝛃"), 2));
        assert!(!look.matches(B("𝛃"), 3));

        // Non word boundaries with non-ASCII codepoints.
        assert!(!look.matches(B("𝛃𐆀"), 1));
        assert!(!look.matches(B("𝛃𐆀"), 2));
        assert!(!look.matches(B("𝛃𐆀"), 3));
        assert!(!look.matches(B("𝛃𐆀"), 5));
        assert!(!look.matches(B("𝛃𐆀"), 6));
        assert!(!look.matches(B("𝛃𐆀"), 7));
        assert!(!look.matches(B("𝛃𐆀"), 8));
    }

    #[test]
    #[cfg(not(miri))]
    fn look_matches_word_unicode_negate() {
        let look = Look::WordBoundaryUnicodeNegate;

        // \xF0\x9D\x9B\x83 = 𝛃 (in \w)
        // \xF0\x90\x86\x80 = 𐆀 (not in \w)

        // Simple ASCII word boundaries.
        assert!(!look.matches(B("a"), 0));
        assert!(!look.matches(B("a"), 1));
        assert!(!look.matches(B("a "), 1));
        assert!(!look.matches(B(" a "), 1));
        assert!(!look.matches(B(" a "), 2));

        // Unicode word boundaries with a non-ASCII codepoint.
        assert!(!look.matches(B("𝛃"), 0));
        assert!(!look.matches(B("𝛃"), 4));
        assert!(!look.matches(B("𝛃 "), 4));
        assert!(!look.matches(B(" 𝛃 "), 1));
        assert!(!look.matches(B(" 𝛃 "), 5));

        // Unicode word boundaries between non-ASCII codepoints.
        assert!(!look.matches(B("𝛃𐆀"), 0));
        assert!(!look.matches(B("𝛃𐆀"), 4));

        // Non word boundaries for ASCII.
        assert!(look.matches(B(""), 0));
        assert!(look.matches(B("ab"), 1));
        assert!(look.matches(B("a "), 2));
        assert!(look.matches(B(" a "), 0));
        assert!(look.matches(B(" a "), 3));

        // Non word boundaries with a non-ASCII codepoint.
        assert!(look.matches(B("𝛃b"), 4));
        assert!(look.matches(B("𝛃 "), 5));
        assert!(look.matches(B(" 𝛃 "), 0));
        assert!(look.matches(B(" 𝛃 "), 6));
        // These don't match because they could otherwise return an offset that
        // splits the UTF-8 encoding of a codepoint.
        assert!(!look.matches(B("𝛃"), 1));
        assert!(!look.matches(B("𝛃"), 2));
        assert!(!look.matches(B("𝛃"), 3));

        // Non word boundaries with non-ASCII codepoints. These also don't
        // match because they could otherwise return an offset that splits the
        // UTF-8 encoding of a codepoint.
        assert!(!look.matches(B("𝛃𐆀"), 1));
        assert!(!look.matches(B("𝛃𐆀"), 2));
        assert!(!look.matches(B("𝛃𐆀"), 3));
        assert!(!look.matches(B("𝛃𐆀"), 5));
        assert!(!look.matches(B("𝛃𐆀"), 6));
        assert!(!look.matches(B("𝛃𐆀"), 7));
        // But this one does, since 𐆀 isn't a word codepoint, and 8 is the end
        // of the haystack. So the "end" of the haystack isn't a word and 𐆀
        // isn't a word, thus, \B matches.
        assert!(look.matches(B("𝛃𐆀"), 8));
    }

    #[test]
    fn look_matches_word_ascii_negate() {
        let look = Look::WordBoundaryAsciiNegate;

        // \xF0\x9D\x9B\x83 = 𝛃 (in \w)
        // \xF0\x90\x86\x80 = 𐆀 (not in \w)

        // Simple ASCII word boundaries.
        assert!(!look.matches(B("a"), 0));
        assert!(!look.matches(B("a"), 1));
        assert!(!look.matches(B("a "), 1));
        assert!(!look.matches(B(" a "), 1));
        assert!(!look.matches(B(" a "), 2));

        // Unicode word boundaries with a non-ASCII codepoint. Since this is
        // an ASCII word boundary, none of these match.
        assert!(look.matches(B("𝛃"), 0));
        assert!(look.matches(B("𝛃"), 4));
        assert!(look.matches(B("𝛃 "), 4));
        assert!(look.matches(B(" 𝛃 "), 1));
        assert!(look.matches(B(" 𝛃 "), 5));

        // Unicode word boundaries between non-ASCII codepoints. Again, since
        // this is an ASCII word boundary, none of these match.
        assert!(look.matches(B("𝛃𐆀"), 0));
        assert!(look.matches(B("𝛃𐆀"), 4));

        // Non word boundaries for ASCII.
        assert!(look.matches(B(""), 0));
        assert!(look.matches(B("ab"), 1));
        assert!(look.matches(B("a "), 2));
        assert!(look.matches(B(" a "), 0));
        assert!(look.matches(B(" a "), 3));

        // Non word boundaries with a non-ASCII codepoint.
        assert!(!look.matches(B("𝛃b"), 4));
        assert!(look.matches(B("𝛃 "), 5));
        assert!(look.matches(B(" 𝛃 "), 0));
        assert!(look.matches(B(" 𝛃 "), 6));
        assert!(look.matches(B("𝛃"), 1));
        assert!(look.matches(B("𝛃"), 2));
        assert!(look.matches(B("𝛃"), 3));

        // Non word boundaries with non-ASCII codepoints.
        assert!(look.matches(B("𝛃𐆀"), 1));
        assert!(look.matches(B("𝛃𐆀"), 2));
        assert!(look.matches(B("𝛃𐆀"), 3));
        assert!(look.matches(B("𝛃𐆀"), 5));
        assert!(look.matches(B("𝛃𐆀"), 6));
        assert!(look.matches(B("𝛃𐆀"), 7));
        assert!(look.matches(B("𝛃𐆀"), 8));
    }

    fn B<'a, T: 'a + ?Sized + AsRef<[u8]>>(string: &'a T) -> &'a [u8] {
        string.as_ref()
    }
}
