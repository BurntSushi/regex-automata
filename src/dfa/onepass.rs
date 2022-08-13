/*!
A DFA that can return spans for matching capturing groups.

This module is the home of a [one-pass DFA](DFA).

This module also contains a [`Builder`] and a [`Config`] for building and
configuring a one-pass DFA.
*/

// A note on naming and credit:
//
// As far as I know, Russ Cox came up with the practical vision and
// implementation of a "one-pass regex engine." He mentions and describes it
// briefly in the third article of his regexp article series:
// https://swtch.com/~rsc/regexp/regexp3.html
//
// That first implementation is in RE2, and the implementation below is most
// heavily inspired by RE2's. The key thing they have in common is that their
// transitions are defined over an alphabet of bytes. In contrast, Go's
// regex engine also has a one-pass engine, but its transitions are more firmly
// rooted on Unicode codepoints. The ideas are the same, but the implementations
// are different.
//
// So, RE2 tends to call this a "one-pass NFA." Here, we call it a "one-pass
// DFA." They're both true in their own ways:
//
// * The "one-pass" criterion is generally a property of the NFA itself. In
// particular, it is said that an NFA is one-pass if, after each byte of input
// during a search, there is at most one "VM thread" remaining to take for the
// next byte of input. That is, there is never any ambiguity as to the path to
// take through the NFA during a search.
//
// * On the other hand, once a one-pass NFA has its representation converted
// to something where a constant number of instructions is used for each byte
// of input, the implementation looks a lot more like a DFA. It's technically
// more powerful than a DFA since it has side effects (storing offsets inside
// of slots activated by a transition), but it is far closer to a DFA than an
// NFA simulation.
//
// Thus, in this crate, we call it a one-pass DFA.

use core::convert::TryFrom;

use alloc::{vec, vec::Vec};

use crate::{
    dfa::{error::Error, remapper::Remapper, DEAD},
    nfa::thompson::{self, NFA},
    util::{
        alphabet::{self, ByteClasses},
        captures::Captures,
        escape::DebugByte,
        int::{Usize, U32, U64, U8},
        iter,
        look::{Look, LookSet},
        primitives::{NonMaxUsize, PatternID, SmallIndex, StateID},
        search::{Anchored, Input, Match, MatchError, MatchKind},
        sparse_set::SparseSet,
    },
};

/// The configuration used for building a [one-pass DFA](DFA).
///
/// A one-pass DFA configuration is a simple data object that is typically used
/// with [`Builder::configure`]. It can be cheaply cloned.
///
/// A default configuration can be created either with `Config::new`, or
/// perhaps more conveniently, with [`DFA::config`].
#[derive(Clone, Debug, Default)]
pub struct Config {
    match_kind: Option<MatchKind>,
    utf8: Option<bool>,
    starts_for_each_pattern: Option<bool>,
    byte_classes: Option<bool>,
    size_limit: Option<Option<usize>>,
}

impl Config {
    /// Return a new default one-pass DFA configuration.
    pub fn new() -> Config {
        Config::default()
    }

    /// Set the desired match semantics.
    ///
    /// The default is [`MatchKind::LeftmostFirst`], which corresponds to the
    /// match semantics of Perl-like regex engines. That is, when multiple
    /// patterns would match at the same leftmost position, the pattern that
    /// appears first in the concrete syntax is chosen.
    ///
    /// Currently, the only other kind of match semantics supported is
    /// [`MatchKind::All`]. This corresponds to "classical DFA" construction
    /// where all possible matches are visited.
    ///
    /// When it comes to the one-pass DFA, it is rarer for preference order and
    /// "longest match" to actually disagree. Since if they did disagree, then
    /// the regex typically isn't one-pass. For example, searching `Samwise`
    /// for `Sam|Samwise` will report `Sam` for leftmost-first matching and
    /// `Samwise` for "longest match" or "all" matching. However, this regex is
    /// not one-pass if taken literally. The equivalent regex, `Sam(?:|wise)`
    /// is one-pass and `Sam|Samwise` may be optimized to it.
    ///
    /// The other main difference is that "all" match semantics don't support
    /// non-greedy matches. "All" match semantics always try to match as much
    /// as possible.
    pub fn match_kind(mut self, kind: MatchKind) -> Config {
        self.match_kind = Some(kind);
        self
    }

    /// Whether to enable UTF-8 mode or not.
    ///
    /// When UTF-8 mode is enabled (the default) and an empty match is seen,
    /// the search APIs of a one-pass DFA will never report a match that would
    /// otherwise split a valid UTF-8 code unit sequence.
    ///
    /// If this mode is enabled and invalid UTF-8 is given to search, then
    /// behavior is unspecified.
    ///
    /// Generally speaking, one should enable this when
    /// [`syntax::Config::utf8`](crate::util::syntax::Config::utf8)
    /// is enabled, and disable it otherwise.
    ///
    /// # Example
    ///
    /// This example demonstrates the differences between when this option is
    /// enabled and disabled. The differences only arise when the one-pass DFA
    /// can return matches of length zero.
    ///
    /// In this first snippet, we show the results when UTF-8 mode is disabled.
    ///
    /// ```
    /// use regex_automata::{dfa::onepass::DFA, Match};
    ///
    /// let re = DFA::builder()
    ///     .configure(DFA::config().utf8(false))
    ///     .build(r"")?;
    /// let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    /// let mut input = re.create_input("a☃z");
    ///
    /// // The empty string matches at every position.
    ///
    /// re.search(&mut cache, &input, &mut caps);
    /// assert_eq!(Some(Match::must(0, 0..0)), caps.get_match());
    ///
    /// input.set_start(1);
    /// re.search(&mut cache, &input, &mut caps);
    /// assert_eq!(Some(Match::must(0, 1..1)), caps.get_match());
    ///
    /// input.set_start(2);
    /// re.search(&mut cache, &input, &mut caps);
    /// assert_eq!(Some(Match::must(0, 2..2)), caps.get_match());
    ///
    /// input.set_start(3);
    /// re.search(&mut cache, &input, &mut caps);
    /// assert_eq!(Some(Match::must(0, 3..3)), caps.get_match());
    ///
    /// input.set_start(4);
    /// re.search(&mut cache, &input, &mut caps);
    /// assert_eq!(Some(Match::must(0, 4..4)), caps.get_match());
    ///
    /// input.set_start(5);
    /// re.search(&mut cache, &input, &mut caps);
    /// assert_eq!(Some(Match::must(0, 5..5)), caps.get_match());
    ///
    /// // 6 > input.haystack.len(), so there's no match here.
    /// input.set_start(6);
    /// re.search(&mut cache, &input, &mut caps);
    /// assert_eq!(None, caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// And in this snippet, we execute the same search on the same haystack,
    /// but with UTF-8 mode enabled. Notice that when we search at offsets that
    /// would otherwise return a match that splits the encoding of `☃`, we
    /// get no match.
    ///
    /// ```
    /// use regex_automata::{dfa::onepass::DFA, Match};
    ///
    /// let re = DFA::builder()
    ///     .configure(DFA::config().utf8(true))
    ///     .build(r"")?;
    /// let mut cache = re.create_cache();
    /// let mut caps = re.create_captures();
    /// let mut input = re.create_input("a☃z");
    ///
    /// // 0 occurs just before 'a', where the empty string matches.
    /// re.search(&mut cache, &input, &mut caps);
    /// assert_eq!(Some(Match::must(0, 0..0)), caps.get_match());
    ///
    /// // 1 occurs just before the snowman.
    /// input.set_start(1);
    /// re.search(&mut cache, &input, &mut caps);
    /// assert_eq!(Some(Match::must(0, 1..1)), caps.get_match());
    ///
    /// // 2 splits the first and second bytes of the snowman.
    /// input.set_start(2);
    /// re.search(&mut cache, &input, &mut caps);
    /// assert_eq!(None, caps.get_match());
    ///
    /// // 3 splits the second and third bytes of the snowman.
    /// input.set_start(3);
    /// re.search(&mut cache, &input, &mut caps);
    /// assert_eq!(None, caps.get_match());
    ///
    /// // 4 is right past the snowman and before the 'z'
    /// input.set_start(4);
    /// re.search(&mut cache, &input, &mut caps);
    /// assert_eq!(Some(Match::must(0, 4..4)), caps.get_match());
    ///
    /// // 5 == input.haystack.len(), at which point, the empty string matches.
    /// input.set_start(5);
    /// re.search(&mut cache, &input, &mut caps);
    /// assert_eq!(Some(Match::must(0, 5..5)), caps.get_match());
    ///
    /// // 6 > input.haystack.len(), so there's no match here.
    /// input.set_start(6);
    /// re.search(&mut cache, &input, &mut caps);
    /// assert_eq!(None, caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn utf8(mut self, yes: bool) -> Config {
        self.utf8 = Some(yes);
        self
    }

    /// Whether to compile a separate start state for each pattern in the
    /// one-pass DFA.
    ///
    /// When enabled, a separate **anchored** start state is added for each
    /// pattern in the DFA. When this start state is used, then the DFA will
    /// only search for matches for the pattern specified, even if there are
    /// other patterns in the DFA.
    ///
    /// The main downside of this option is that it can potentially increase
    /// the size of the DFA and/or increase the time it takes to build the DFA.
    ///
    /// You might want to enable this option when you want to both search for
    /// anchored matches of any pattern or to search for anchored matches of
    /// one particular pattern while using the same DFA. (Otherwise, you would
    /// need to compile a new DFA for each pattern.)
    ///
    /// By default this is disabled.
    ///
    /// # Example
    ///
    /// This example shows how to build a multi-regex and then search for
    /// matches for a any of the patterns or matches for a specific pattern.
    ///
    /// ```
    /// use regex_automata::{dfa::onepass::DFA, Anchored, Match, PatternID};
    ///
    /// let re = DFA::builder()
    ///     .configure(DFA::config().starts_for_each_pattern(true))
    ///     .build_many(&["[a-z]+", "[0-9]+"])?;
    /// let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    /// let haystack = "123abc";
    ///
    /// // A normal multi-pattern search will show pattern 1 matches.
    /// re.search(&mut cache, &re.create_input(haystack), &mut caps);
    /// assert_eq!(Some(Match::must(1, 0..3)), caps.get_match());
    ///
    /// // If we only want to report pattern 0 matches, then we'll get no
    /// // match here.
    /// let input = re.create_input(haystack)
    ///     .anchored(Anchored::Pattern(PatternID::must(0)));
    /// re.search(&mut cache, &input, &mut caps);
    /// assert_eq!(None, caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn starts_for_each_pattern(mut self, yes: bool) -> Config {
        self.starts_for_each_pattern = Some(yes);
        self
    }

    /// Whether to attempt to shrink the size of the DFA's alphabet or not.
    ///
    /// This option is enabled by default and should never be disabled unless
    /// one is debugging a one-pass DFA.
    ///
    /// When enabled, the DFA will use a map from all possible bytes to their
    /// corresponding equivalence class. Each equivalence class represents a
    /// set of bytes that does not discriminate between a match and a non-match
    /// in the DFA. For example, the pattern `[ab]+` has at least two
    /// equivalence classes: a set containing `a` and `b` and a set containing
    /// every byte except for `a` and `b`. `a` and `b` are in the same
    /// equivalence class because they never discriminate between a match and a
    /// non-match.
    ///
    /// The advantage of this map is that the size of the transition table
    /// can be reduced drastically from (approximately) `#states * 256 *
    /// sizeof(StateID)` to `#states * k * sizeof(StateID)` where `k` is the
    /// number of equivalence classes (rounded up to the nearest power of 2).
    /// As a result, total space usage can decrease substantially. Moreover,
    /// since a smaller alphabet is used, DFA compilation becomes faster as
    /// well.
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

    /// Set a size limit on the total heap used by a one-pass DFA.
    ///
    /// This size limit is expressed in bytes and is applied during
    /// construction of a one-pass DFA. If the DFA's heap usage exceeds
    /// this configured limit, then construction is stopped and an error is
    /// returned.
    ///
    /// The default is no limit.
    ///
    /// # Example
    ///
    /// This example shows a one-pass DFA that fails to build because of
    /// a configured size limit. This particular example also serves as a
    /// cautionary tale demonstrating just how big DFAs with large Unicode
    /// character classes can get.
    ///
    /// ```
    /// use regex_automata::{dfa::onepass::DFA, Match};
    ///
    /// // 6MB isn't enough!
    /// DFA::builder()
    ///     .configure(DFA::config().size_limit(Some(6_000_000)))
    ///     .build(r"\w{20}")
    ///     .unwrap_err();
    ///
    /// // ... but 7MB probably is!
    /// // (Note that DFA sizes aren't necessarily stable between releases.)
    /// let re = DFA::builder()
    ///     .configure(DFA::config().size_limit(Some(7_000_000)))
    ///     .build(r"\w{20}")?;
    /// let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    /// let haystack = "A".repeat(20);
    /// re.find(&mut cache, &haystack, &mut caps);
    /// assert_eq!(Some(Match::must(0, 0..20)), caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// While one needs a little more than 3MB to represent `\w{20}`, it
    /// turns out that you only need a little more than 4KB to represent
    /// `(?-u:\w{20})`. So only use Unicode if you need it!
    pub fn size_limit(mut self, limit: Option<usize>) -> Config {
        self.size_limit = Some(limit);
        self
    }

    /// Returns the match semantics set in this configuration.
    pub fn get_match_kind(&self) -> MatchKind {
        self.match_kind.unwrap_or(MatchKind::LeftmostFirst)
    }

    /// Returns whether UTF-8 mode should be enabled for searches.
    pub fn get_utf8(&self) -> bool {
        self.utf8.unwrap_or(true)
    }

    /// Returns whether this configuration has enabled anchored starting states
    /// for every pattern in the DFA.
    pub fn get_starts_for_each_pattern(&self) -> bool {
        self.starts_for_each_pattern.unwrap_or(false)
    }

    /// Returns whether this configuration has enabled byte classes or not.
    /// This is typically a debugging oriented option, as disabling it confers
    /// no speed benefit.
    pub fn get_byte_classes(&self) -> bool {
        self.byte_classes.unwrap_or(true)
    }

    /// Returns the DFA size limit of this configuration if one was set.
    /// The size limit is total number of bytes on the heap that a DFA is
    /// permitted to use. If the DFA exceeds this limit during construction,
    /// then construction is stopped and an error is returned.
    pub fn get_size_limit(&self) -> Option<usize> {
        self.size_limit.unwrap_or(None)
    }

    /// Overwrite the default configuration such that the options in `o` are
    /// always used. If an option in `o` is not set, then the corresponding
    /// option in `self` is used. If it's not set in `self` either, then it
    /// remains not set.
    pub(crate) fn overwrite(&self, o: Config) -> Config {
        Config {
            match_kind: o.match_kind.or(self.match_kind),
            utf8: o.utf8.or(self.utf8),
            starts_for_each_pattern: o
                .starts_for_each_pattern
                .or(self.starts_for_each_pattern),
            byte_classes: o.byte_classes.or(self.byte_classes),
            size_limit: o.size_limit.or(self.size_limit),
        }
    }
}

/// A builder for a [one-pass DFA](DFA).
///
/// This builder permits configuring options for the syntax of a pattern, the
/// NFA construction and the DFA construction. This builder is different from a
/// general purpose regex builder in that it permits fine grain configuration
/// of the construction process. The trade off for this is complexity, and
/// the possibility of setting a configuration that might not make sense. For
/// example, there are two different UTF-8 modes:
///
/// * [`syntax::Config::utf8`](crate::util::syntax::Config::utf8) controls
/// whether the pattern itself can contain sub-expressions that match invalid
/// UTF-8.
/// * [`Config::utf8`] controls whether empty matches that split a Unicode
/// codepoint are reported or not.
///
/// Generally speaking, callers will want to either enable all of these or
/// disable all of these.
///
/// # Example
///
/// This example shows how to disable UTF-8 mode in the syntax and the regex
/// itself. This is generally what you want for matching on arbitrary bytes.
///
/// ```
/// use regex_automata::{dfa::onepass::DFA, util::syntax, Match};
///
/// let re = DFA::builder()
///     .configure(DFA::config().utf8(false))
///     .syntax(syntax::Config::new().utf8(false))
///     .build(r"foo(?-u:[^b])ar.*")?;
/// let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
///
/// let haystack = b"foo\xFFarzz\xE2\x98\xFF\n";
/// re.find(&mut cache, haystack, &mut caps);
/// // Notice that `(?-u:[^b])` matches invalid UTF-8,
/// // but the subsequent `.*` does not! Disabling UTF-8
/// // on the syntax permits this.
/// //
/// // N.B. This example does not show the impact of
/// // disabling UTF-8 mode on a one-pass DFA Config,
/// //  since that only impacts regexes that can
/// // produce matches of length 0.
/// assert_eq!(Some(Match::must(0, 0..8)), caps.get_match());
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug)]
pub struct Builder {
    config: Config,
    thompson: thompson::Compiler,
}

impl Builder {
    /// Create a new one-pass DFA builder with the default configuration.
    pub fn new() -> Builder {
        Builder {
            config: Config::default(),
            thompson: thompson::Compiler::new(),
        }
    }

    /// Build a one-pass DFA from the given pattern.
    ///
    /// If there was a problem parsing or compiling the pattern, then an error
    /// is returned.
    pub fn build(&self, pattern: &str) -> Result<DFA, Error> {
        self.build_many(&[pattern])
    }

    /// Build a one-pass DFA from the given patterns.
    ///
    /// When matches are returned, the pattern ID corresponds to the index of
    /// the pattern in the slice given.
    pub fn build_many<P: AsRef<str>>(
        &self,
        patterns: &[P],
    ) -> Result<DFA, Error> {
        let nfa = self.thompson.build_many(patterns).map_err(Error::nfa)?;
        self.build_from_nfa(nfa)
    }

    /// Build a DFA from the given NFA.
    ///
    /// # Example
    ///
    /// This example shows how to build a DFA if you already have an NFA in
    /// hand.
    ///
    /// ```
    /// use regex_automata::{dfa::onepass::DFA, nfa::thompson::NFA, Match};
    ///
    /// // This shows how to set non-default options for building an NFA.
    /// let nfa = NFA::compiler()
    ///     .configure(NFA::config().shrink(true))
    ///     .build(r"[a-z0-9]+")?;
    /// let re = DFA::builder().build_from_nfa(nfa)?;
    /// let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    /// re.find(&mut cache, "foo123bar", &mut caps);
    /// assert_eq!(Some(Match::must(0, 0..9)), caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn build_from_nfa(&self, nfa: NFA) -> Result<DFA, Error> {
        // Why take ownership if we're just going to pass a reference to the
        // NFA to our internal builder? Well, the first thing to note is that
        // an NFA uses reference counting internally, so either choice is going
        // to be cheap. So there isn't much cost either way.
        //
        // The real reason is that a one-pass DFA, semantically, shares
        // ownership of an NFA. This is unlike other DFAs that don't share
        // ownership of an NFA at all, primarily because they want to be
        // self-contained in order to support cheap (de)serialization.
        //
        // But then why pass a '&nfa' below if we want to share ownership?
        // Well, it turns out that using a '&NFA' in our internal builder
        // separates its lifetime from the DFA we're building, and this turns
        // out to make code a bit more composable. e.g., We can iterate over
        // things inside the NFA while borrowing the builder as mutable because
        // we know the NFA cannot be mutated. So TL;DR --- this weirdness is
        // "because borrow checker."
        InternalBuilder::new(self.config.clone(), &nfa).build()
    }

    /// Apply the given one-pass DFA configuration options to this builder.
    pub fn configure(&mut self, config: Config) -> &mut Builder {
        self.config = self.config.overwrite(config);
        self
    }

    /// Set the syntax configuration for this builder using
    /// [`syntax::Config`](crate::util::syntax::Config).
    ///
    /// This permits setting things like case insensitivity, Unicode and multi
    /// line mode.
    ///
    /// These settings only apply when constructing a one-pass DFA directly
    /// from a pattern.
    pub fn syntax(
        &mut self,
        config: crate::util::syntax::Config,
    ) -> &mut Builder {
        self.thompson.syntax(config);
        self
    }

    /// Set the Thompson NFA configuration for this builder using
    /// [`nfa::thompson::Config`](crate::nfa::thompson::Config).
    ///
    /// This permits setting things like whether additional time should be
    /// spent shrinking the size of the NFA.
    ///
    /// These settings only apply when constructing a DFA directly from a
    /// pattern.
    pub fn thompson(&mut self, config: thompson::Config) -> &mut Builder {
        self.thompson.configure(config);
        self
    }
}

/// An internal builder for encapsulating the state necessary to build a
/// one-pass DFA. Typical use is just `InternalBuilder::new(..).build()`.
///
/// There is no separate pass for determining whether the NFA is one-pass or
/// not. We just try to build the DFA. If during construction we discover that
/// it is not one-pass, we bail out. This is likely to lead to some undesirable
/// expense in some cases, so it might make sense to try an identify common
/// patterns in the NFA that make it definitively not one-pass. That way, we
/// can avoid ever trying to build a one-pass DFA in the first place. For
/// example, '\w*\s' is not one-pass, and since '\w' is Unicode-aware by
/// default, it's probably not a trivial cost to try and build a one-pass DFA
/// for it and then fail.
///
/// Note that some (immutable) fields are duplicated here. For example, the
/// 'nfa' and 'classes' fields are both in the 'DFA'. They are the same thing,
/// but we duplicate them because it makes composition easier below. Otherwise,
/// since the borrow checker can't see through method calls, the mutable borrow
/// we use to mutate the DFA winds up preventing borrowing from any other part
/// of the DFA, even though we aren't mutating those parts. We only do this
/// because the duplication is cheap.
#[derive(Debug)]
struct InternalBuilder<'a> {
    /// The DFA we're building.
    dfa: DFA,
    /// An unordered collection of NFA state IDs that we haven't yet tried to
    /// build into a DFA state yet.
    ///
    /// This collection does not ultimately wind up including every NFA state
    /// ID. Instead, each ID represents a "start" state for a sub-graph of the
    /// NFA. The set of NFA states we then use to build a DFA state consists
    /// of that "start" state and all states reachable from it via epsilon
    /// transitions.
    uncompiled_nfa_ids: Vec<StateID>,
    /// A map from NFA state ID to DFA state ID. This is useful for easily
    /// determining whether an NFA state has been used as a "starting" point
    /// to build a DFA state yet. If it hasn't, then it is mapped to DEAD,
    /// and since DEAD is specially added and never corresponds to any NFA
    /// state, it follows that a mapping to DEAD implies the NFA state has
    /// no corresponding DFA state yet.
    nfa_to_dfa_id: Vec<StateID>,
    /// A stack used to traverse the NFA states that make up a single DFA
    /// state. Traversal occurs until the stack is empty, and we only push to
    /// the stack when the state ID isn't in 'seen'. Actually, even more than
    /// that, if we try to push something on to this stack that is already in
    /// 'seen', then we bail out on construction completely, since it implies
    /// that the NFA is not one-pass.
    stack: Vec<(StateID, Epsilons)>,
    /// The set of NFA states that we've visited via 'stack'.
    seen: SparseSet,
    /// Whether a match NFA state has been observed while constructing a
    /// one-pass DFA state. Once a match state is seen, assuming we are using
    /// leftmost-first match semantics, then we don't add any more transitions
    /// to the DFA state we're building.
    matched: bool,
    /// The config passed to the builder.
    ///
    /// This is duplicated in dfa.config.
    config: Config,
    /// The NFA we're building a one-pass DFA from.
    ///
    /// This is duplicated in dfa.nfa.
    nfa: &'a NFA,
    /// The equivalence classes that make up the alphabet for this DFA>
    ///
    /// This is duplicated in dfa.classes.
    classes: ByteClasses,
}

impl<'a> InternalBuilder<'a> {
    /// Create a new builder with an initial empty DFA.
    fn new(config: Config, nfa: &'a NFA) -> InternalBuilder {
        let classes = if !config.get_byte_classes() {
            // A one-pass DFA will always use the equivalence class map, but
            // enabling this option is useful for debugging. Namely, this will
            // cause all transitions to be defined over their actual bytes
            // instead of an opaque equivalence class identifier. The former is
            // much easier to grok as a human.
            ByteClasses::singletons()
        } else {
            nfa.byte_class_set().byte_classes()
        };
        // Normally a DFA alphabet includes the EOI symbol, but we don't need
        // that in the one-pass DFA since we handle look-around explicitly
        // without encoding it into the DFA. Thus, we don't need to delay
        // matches by 1 byte. However, we reuse the space that *would* be used
        // by the EOI transition by putting match information there (like which
        // pattern matches and which look-around assertions need to hold). So
        // this means our real alphabet length is 1 fewer than what the byte
        // classes report, since we don't use EOI.
        let alphabet_len = classes.alphabet_len().checked_sub(1).unwrap();
        let stride2 = classes.stride2();
        let dfa = DFA {
            config: config.clone(),
            nfa: nfa.clone(),
            table: vec![],
            starts: vec![],
            // Since one-pass DFAs have a smaller state ID max than
            // StateID::MAX, it follows that StateID::MAX is a valid initial
            // value for min_match_id since no state ID can ever be greater
            // than it. In the case of a one-pass DFA with no match states, the
            // min_match_id will keep this sentinel value.
            min_match_id: StateID::MAX,
            classes: classes.clone(),
            alphabet_len,
            stride2,
            pateps_offset: alphabet_len,
            // OK because PatternID::MAX*2 is guaranteed not to overflow.
            explicit_slot_start: nfa.pattern_len().checked_mul(2).unwrap(),
        };
        InternalBuilder {
            dfa,
            uncompiled_nfa_ids: vec![],
            nfa_to_dfa_id: vec![DEAD; nfa.states().len()],
            stack: vec![],
            seen: SparseSet::new(nfa.states().len()),
            matched: false,
            config,
            nfa,
            classes,
        }
    }

    /// Build the DFA from the NFA given to this builder. If the NFA is not
    /// one-pass, then return an error. An error may also be returned if a
    /// particular limit is exceeded. (Some limits, like the total heap memory
    /// used, are configurable. Others, like the total patterns or slots, are
    /// hard-coded based on representational limitations.)
    fn build(mut self) -> Result<DFA, Error> {
        if self.nfa.pattern_len().as_u64() > PatternEpsilons::PATTERN_ID_LIMIT
        {
            return Err(Error::one_pass_too_many_patterns(
                PatternEpsilons::PATTERN_ID_LIMIT,
            ));
        }
        if self.nfa.group_info().explicit_slot_len() > Slots::LIMIT {
            return Err(Error::one_pass_fail(
                "too many explicit capturing groups (max is 24)",
            ));
        }
        assert_eq!(DEAD, self.add_empty_state()?);

        // This is where the explicit slots start. We care about this because
        // we only need to track explicit slots. The implicit slots---two for
        // each pattern---are tracked as part of the search routine itself.
        let explicit_slot_start = self.nfa.pattern_len() * 2;
        self.add_start_state(None, self.nfa.start_anchored())?;
        if self.config.get_starts_for_each_pattern() {
            for pid in self.nfa.patterns() {
                self.add_start_state(Some(pid), self.nfa.start_pattern(pid))?;
            }
        }
        // NOTE: One wonders what the effects of treating 'uncompiled_nfa_ids'
        // as a stack are. It is really an unordered *set* of NFA state IDs.
        // If it, for example, in practice led to discovering whether a regex
        // was or wasn't one-pass later than if we processed NFA state IDs in
        // ascending order, then that would make this routine more costly in
        // the somewhat common case of a regex that isn't one-pass.
        while let Some(nfa_id) = self.uncompiled_nfa_ids.pop() {
            let dfa_id = self.nfa_to_dfa_id[nfa_id];
            // Once we see a match, we keep going, but don't add any new
            // transitions. Normally we'd just stop, but we have to keep
            // going in order to verify that our regex is actually one-pass.
            self.matched = false;
            // The NFA states we've already explored for this DFA state.
            self.seen.clear();
            // The NFA states to explore via epsilon transitions. If we ever
            // try to push an NFA state that we've already seen, then the NFA
            // is not one-pass because it implies there are multiple epsilon
            // transition paths that lead to the same NFA state. In other
            // words, there is ambiguity.
            self.stack_push(nfa_id, Epsilons::empty())?;
            while let Some((id, epsilons)) = self.stack.pop() {
                match *self.nfa.state(id) {
                    thompson::State::ByteRange { ref trans } => {
                        self.compile_transition(dfa_id, trans, epsilons)?;
                    }
                    thompson::State::Sparse(ref sparse) => {
                        for trans in sparse.transitions.iter() {
                            self.compile_transition(dfa_id, trans, epsilons)?;
                        }
                    }
                    thompson::State::Look { look, next } => {
                        let looks = epsilons.looks().insert(look);
                        self.stack_push(next, epsilons.set_looks(looks))?;
                    }
                    thompson::State::Union { ref alternates } => {
                        for &sid in alternates.iter().rev() {
                            self.stack_push(sid, epsilons)?;
                        }
                    }
                    thompson::State::BinaryUnion { alt1, alt2 } => {
                        self.stack_push(alt2, epsilons)?;
                        self.stack_push(alt1, epsilons)?;
                    }
                    thompson::State::Capture { next, slot, .. } => {
                        let slot = slot.as_usize();
                        let epsilons = if slot < explicit_slot_start {
                            // If this is an implicit slot, we don't care
                            // about it, since we handle implicit slots in
                            // the search routine. We can get away with that
                            // because there are 2 implicit slots for every
                            // pattern.
                            epsilons
                        } else {
                            // Offset our explicit slots so that they start
                            // at index 0.
                            let offset = slot - explicit_slot_start;
                            epsilons.set_slots(epsilons.slots().insert(offset))
                        };
                        self.stack_push(next, epsilons)?;
                    }
                    thompson::State::Fail => {
                        continue;
                    }
                    thompson::State::Match { pattern_id } => {
                        // If we found two different paths to a match state
                        // for the same DFA state, then we have ambiguity.
                        // Thus, it's not one-pass.
                        if self.matched {
                            return Err(Error::one_pass_fail(
                                "multiple epsilon transitions to match state",
                            ));
                        }
                        self.matched = true;
                        // Shove the matching pattern ID and the 'epsilons'
                        // into the current DFA state's pattern epsilons. The
                        // 'epsilons' includes the slots we need to capture
                        // before reporting the match and also the conditional
                        // epsilon transitions we need to check before we can
                        // report a match.
                        self.dfa.set_pattern_epsilons(
                            dfa_id,
                            PatternEpsilons::empty()
                                .set_pattern_id(pattern_id)
                                .set_epsilons(epsilons),
                        );
                        // N.B. It is tempting to just bail out here when
                        // compiling a leftmost-first DFA, since we will never
                        // compile any more transitions in that case. But we
                        // actually need to keep going in order to verify that
                        // we actually have a one-pass regex. e.g., We might
                        // see more Match states (e.g., for other patterns)
                        // that imply that we don't have a one-pass regex.
                        // So instead, we mark that we've found a match and
                        // continue on. When we go to compile a new DFA state,
                        // we just skip that part. But otherwise check that the
                        // one-pass property is upheld.
                    }
                }
            }
        }
        self.shuffle_states();
        Ok(self.dfa)
    }

    /// Shuffle all match states to the end of the transition table and set
    /// 'min_match_id' to the ID of the first such match state.
    ///
    /// The point of this is to make it extremely cheap to determine whether
    /// a state is a match state or not. We need to check on this on every
    /// transition during a search, so it being cheap is important. This
    /// permits us to check it by simply comparing two state identifiers, as
    /// opposed to looking for the pattern ID in the state's `PatternEpsilons`.
    /// (Which requires a memory load and some light arithmetic.)
    fn shuffle_states(&mut self) {
        let mut remapper = Remapper::new(&self.dfa);
        let mut next_dest = self.dfa.last_state_id();
        for i in (0..self.dfa.state_len()).rev() {
            let id = self.dfa.to_state_id(i);
            let is_match =
                self.dfa.pattern_epsilons(id).pattern_id().is_some();
            if !is_match {
                continue;
            }
            remapper.swap(&mut self.dfa, next_dest, id);
            self.dfa.min_match_id = next_dest;
            next_dest = self.dfa.prev_state_id(next_dest).expect(
                "match states should be a proper subset of all states",
            );
        }
        remapper.remap(&mut self.dfa);
    }

    /// Compile the given NFA transition into the DFA state given.
    ///
    /// 'Epsilons' corresponds to any conditional epsilon transitions that need
    /// to be satisfied to follow this transition, and any slots that need to
    /// be saved if the transition is followed.
    ///
    /// If this transition indicates that the NFA is not one-pass, then
    /// this returns an error. (This occurs, for example, if the DFA state
    /// already has a transition defined for the same input symbols as the
    /// given transition, *and* the result of the old and new transitions is
    /// different.)
    fn compile_transition(
        &mut self,
        dfa_id: StateID,
        trans: &thompson::Transition,
        epsilons: Epsilons,
    ) -> Result<(), Error> {
        // If we already have seen a match and we are compiling a leftmost
        // first DFA, then we shouldn't add any more transitions. This is
        // effectively how preference order and non-greediness is implemented.
        if !self.config.get_match_kind().continue_past_first_match()
            && self.matched
        {
            return Ok(());
        }
        let next_dfa_id = self.add_dfa_state_for_nfa_state(trans.next)?;
        for byte in self
            .classes
            .representatives(trans.start..=trans.end)
            .filter_map(|r| r.as_u8())
        {
            let oldtrans = self.dfa.transition(dfa_id, byte);
            let newtrans = Transition::new(next_dfa_id, epsilons);
            // If the old transition points to the DEAD state, then we know
            // 'byte' has not been mapped to any transition for this DFA state
            // yet. So set it unconditionally. Otherwise, we require that the
            // old and new transitions are equivalent. Otherwise, there is
            // ambiguity and thus the regex is not one-pass.
            if oldtrans.state_id() == DEAD {
                self.dfa.set_transition(dfa_id, byte, newtrans);
            } else if oldtrans != newtrans {
                return Err(Error::one_pass_fail("conflicting transition"));
            }
        }
        Ok(())
    }

    /// Add a start state to the DFA corresponding to the given NFA starting
    /// state ID.
    ///
    /// If adding a state would blow any limits (configured or hard-coded),
    /// then an error is returned.
    ///
    /// If the starting state is an anchored state for a particular pattern,
    /// then callers must provide the pattern ID for that starting state.
    /// Callers must also ensure that the first starting state added is the
    /// start state for all patterns, and then each anchored starting state for
    /// each pattern (if necessary) added in order. Otherwise, this panics.
    fn add_start_state(
        &mut self,
        pid: Option<PatternID>,
        nfa_id: StateID,
    ) -> Result<StateID, Error> {
        match pid {
            // With no pid, this should be the start state for all patterns
            // and thus be the first one.
            None => assert!(self.dfa.starts.is_empty()),
            // With a pid, we want it to be at self.dfa.starts[pid+1].
            Some(pid) => assert!(self.dfa.starts.len() == pid.one_more()),
        }
        let dfa_id = self.add_dfa_state_for_nfa_state(nfa_id)?;
        self.dfa.starts.push(dfa_id);
        Ok(dfa_id)
    }

    /// Add a new DFA state corresponding to the given NFA state. If adding a
    /// state would blow any limits (configured or hard-coded), then an error
    /// is returned. If a DFA state already exists for the given NFA state,
    /// then that DFA state's ID is returned and no new states are added.
    ///
    /// It is not expected that this routine is called for every NFA state.
    /// Instead, an NFA state ID will usually correspond to the "start" state
    /// for a sub-graph of the NFA, where all states in the sub-graph are
    /// reachable via epsilon transitions (conditional or unconditional). That
    /// sub-graph of NFA states is ultimately what produces a single DFA state.
    fn add_dfa_state_for_nfa_state(
        &mut self,
        nfa_id: StateID,
    ) -> Result<StateID, Error> {
        // If we've already built a DFA state for the given NFA state, then
        // just return that. We definitely do not want to have more than one
        // DFA state in existence for the same NFA state, since all but one of
        // them will likely become unreachable. And at least some of them are
        // likely to wind up being incomplete.
        let existing_dfa_id = self.nfa_to_dfa_id[nfa_id];
        if existing_dfa_id != DEAD {
            return Ok(existing_dfa_id);
        }
        // If we don't have any DFA state yet, add it and then add the given
        // NFA state to the list of states to explore.
        let dfa_id = self.add_empty_state()?;
        self.nfa_to_dfa_id[nfa_id] = dfa_id;
        self.uncompiled_nfa_ids.push(nfa_id);
        Ok(dfa_id)
    }

    /// Unconditionally add a new empty DFA state. If adding it would exceed
    /// any limits (configured or hard-coded), then an error is returned. The
    /// ID of the new state is returned on success.
    ///
    /// The added state is *not* a match state.
    fn add_empty_state(&mut self) -> Result<StateID, Error> {
        let next = self.dfa.table.len();
        let id = StateID::new(next).map_err(|_| Error::too_many_states())?;
        if id.as_u64() > Transition::STATE_ID_LIMIT {
            let state_limit =
                Transition::STATE_ID_LIMIT / self.dfa.stride().as_u64();
            return Err(Error::one_pass_too_many_states(state_limit));
        }
        self.dfa
            .table
            .extend(core::iter::repeat(Transition(0)).take(self.dfa.stride()));
        // The default empty value for 'PatternEpsilons' is sadly not all
        // zeroes. Instead, a special sentinel is used to indicate that there
        // is no pattern. So we need to explicitly set the pattern epsilons to
        // the correct "empty" PatternEpsilons.
        self.dfa.set_pattern_epsilons(id, PatternEpsilons::empty());
        if let Some(size_limit) = self.config.get_size_limit() {
            if self.dfa.memory_usage() > size_limit {
                return Err(Error::one_pass_exceeded_size_limit(size_limit));
            }
        }
        Ok(id)
    }

    /// Push the given NFA state ID and its corresponding epsilons (slots and
    /// conditional epsilon transitions) on to a stack for use in a depth first
    /// traversal of a sub-graph of the NFA.
    ///
    /// If the given NFA state ID has already been pushed on to the stack, then
    /// it indicates the regex is not one-pass and this correspondingly returns
    /// an error.
    fn stack_push(
        &mut self,
        nfa_id: StateID,
        epsilons: Epsilons,
    ) -> Result<(), Error> {
        if !self.seen.insert(nfa_id) {
            return Err(Error::one_pass_fail(
                "multiple epsilon transitions to same state",
            ));
        }
        self.stack.push((nfa_id, epsilons));
        Ok(())
    }
}

/// A one-pass DFA for executing a subset of anchored regex searches while
/// resolving capturing groups.
///
/// A one-pass DFA can be built from an NFA that is one-pass. An NFA is
/// one-pass when there is never any ambiguity about how to continue a search.
/// For example, `a*a` is not one-pass becuase during a search, it's not
/// possible to know whether to continue matching the `a*` or to move on to
/// the single `a`. However, `a*b` is one-pass, because for every byte in the
/// input, it's always clear when to move on from `a*` to `b`.
///
/// # Only anchored searches are supported
///
/// In this crate, especially for DFAs, unanchored searches are implemented by
/// treating the pattern as if it had a `(?s-u:.)*?` prefix. While the prefix
/// is one-pass on its own, adding anything after it, e.g., `(?s-u:.)*?a` will
/// make the overall pattern not one-pass. Why? Because the `(?s-u:.)` matches
/// any byte, and there is therefore ambiguity as to when the prefix should
/// stop matching and something else should start matching.
///
/// Therefore, one-pass DFAs do not support unanchored searches. In addition
/// to many regexes simply not being one-pass, it implies that one-pass DFAs
/// have limited utility. With that said, when a one-pass DFA can be used, it
/// can potentially provide a dramatic speed up over alternatives like the
/// [`BoundedBacktracker`](crate::nfa::thompson::backtrack::BoundedBacktracker)
/// and the [`PikeVM`](crate::nfa::thompson::pikevm::PikeVM). In particular,
/// a one-pass DFA is the only DFA capable of reporting the spans of matching
/// capturing groups.
///
/// # Other limitations
///
/// In addition to the [configurable heap limit](Config::size_limit) and
/// the requirement that a regex pattern be one-pass, there are some other
/// limitations:
///
/// * There is an internal limit on the total number of explicit capturing
/// groups that appear across all patterns. It is somewhat small and there is
/// no way to configure it. If your pattern(s) exceed this limit, then building
/// a one-pass DFA will fail.
/// * If the number of patterns exceeds an internal unconfigurable limit, then
/// building a one-pass DFA will fail. This limit is quite large and you're
/// unlikely to hit it.
/// * If the total number of states exceeds an internal unconfigurable limit,
/// then building a one-pass DFA will fail. This limit is quite large and
/// you're unlikely to hit it.
///
/// # Other examples of regexes that aren't one-pass
///
/// One particularly unfortunate example is that enabling Unicode can cause
/// regexes that were one-pass to no longer be one-pass. Consider the regex
/// `(?-u)\w*\s` for example. It is one-pass because there is exactly no
/// overlap between the ASCII definitions of `\w` and `\s`. But `\w*\s` (i.e.,
/// with Unicode enabled) is *not* one-pass because `\w` and `\s` are actually
/// UTF-8 automatons. And while the *codepoints* in `\w` and `\s` do not
/// overlap, the underlying UTF-8 encodings do. Indeed, because of the overlap
/// between UTF-8 automata, the use of Unicode character classes will tend to
/// vastly increase the likelihood of a regex not being one-pass.
///
/// # How does one know if a regex is one-pass or not?
///
/// At the time of writing, the only way to know is to try and build a one-pass
/// DFA. The one-pass property is checked while constructing the DFA.
///
/// This does mean that you might potentially waste some CPU cycles and memory
/// by optimistically trying to build a one-pass DFA. But this is currently the
/// only way. In the future, building a one-pass DFA might be able to use some
/// heuristics to detect common violations of the one-pass property and bail
/// more quickly.
///
/// # Example
///
/// This example shows that the one-pass DFA implements Unicode word boundaries
/// correctly while simultaneously reporting spans for capturing groups that
/// participate in a match. (This is the only DFA that implements full support
/// for Unicode word boundaries.)
///
/// ```
/// use regex_automata::{dfa::onepass::DFA, Match, Span};
///
/// let re = DFA::new(r"\b(?P<first>\w+)[[:space:]]+(?P<last>\w+)\b")?;
/// let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
///
/// re.find(&mut cache, "Шерлок Холмс", &mut caps);
/// assert_eq!(Some(Match::must(0, 0..23)), caps.get_match());
/// assert_eq!(Some(Span::from(0..12)), caps.get_group_by_name("first"));
/// assert_eq!(Some(Span::from(13..23)), caps.get_group_by_name("last"));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Example: iteration
///
/// Unlike other regex engines in this crate, this one does not provide
/// iterator search functions. This is because a one-pass DFA only supports
/// anchored searches, and so iterator functions are generally not applicable.
///
/// However, if you know that all of your matches are
/// directly adjacent, then an iterator can be used. The
/// [`util::iter::Searcher`](crate::util::iter::Searcher) type can be used for
/// this purpose:
///
/// ```
/// use regex_automata::{dfa::onepass::DFA, util::iter::Searcher, Span};
///
/// let re = DFA::new(r"\w(\d)\w")?;
/// let (mut cache, caps) = (re.create_cache(), re.create_captures());
/// let input = re.create_input("a1zb2yc3x");
///
/// let mut it = Searcher::new(input).into_captures_iter(caps, |input, caps| {
///     Ok(re.search(&mut cache, input, caps))
/// }).infallible();
/// let caps0 = it.next().unwrap();
/// assert_eq!(Some(Span::from(1..2)), caps0.get_group(1));
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone)]
pub struct DFA {
    /// The configuration provided by the caller.
    config: Config,
    /// The NFA used to build this DFA.
    ///
    /// NOTE: We probably don't need to store the NFA here, but we use enough
    /// bits from it that it's convenient to do so. And there really isn't much
    /// cost to doing so either, since an NFA is reference counted internally.
    nfa: NFA,
    /// The transition table. Given a state ID 's' and a byte of haystack 'b',
    /// the next state is 'table[sid + classes[byte]]'.
    ///
    /// The stride of this table (i.e., the number of columns) is always
    /// a power of 2, even if the alphabet length is smaller. This makes
    /// converting between state IDs and state indices very cheap.
    ///
    /// Note that the stride always includes room for one extra "transition"
    /// that isn't actually a transition. It is a 'PatternEpsilons' that is
    /// used for match states only. Because of this, the maximum number of
    /// active columns in the transition table is 257, which means the maximum
    /// stride is 512 (the next power of 2 greater than or equal to 257).
    table: Vec<Transition>,
    /// The DFA state IDs of the starting states.
    ///
    /// starts[0] is always present and corresponds to the starting state when
    /// searching for matches of any pattern in the DFA.
    ///
    /// starts[i] where i>0 corresponds to the starting state for the pattern
    /// ID 'i-1'. These starting states are optional.
    starts: Vec<StateID>,
    /// Every state ID >= this value corresponds to a match state.
    ///
    /// This is what a search uses to detect whether a state is a match state
    /// or not. It requires only a simple comparison instead of bit-unpacking
    /// the PatternEpsilons from every state.
    min_match_id: StateID,
    /// The alphabet of this DFA, split into equivalence classes. Bytes in the
    /// same equivalence class can never discriminate between a match and a
    /// non-match.
    classes: ByteClasses,
    /// The number of elements in each state in the transition table. This may
    /// be less than the stride, since the stride is always a power of 2 and
    /// the alphabet length can be anything up to and including 256.
    alphabet_len: usize,
    /// The number of columns in the transition table, expressed as a power of
    /// 2.
    stride2: usize,
    /// The offset at which the PatternEpsilons for a match state is stored in
    /// the transition table.
    ///
    /// PERF: One wonders whether it would be better to put this in a separate
    /// allocation, since only match states have a non-empty PatternEpsilons
    /// and the number of match states tends be dwarfed by the number of
    /// non-match states. So this would save '8*len(non_match_states)' for each
    /// DFA. The question is whether moving this to a different allocation will
    /// lead to a perf hit during searches. You might think dealing with match
    /// states is rare, but some regexes spend a lot of time in match states
    /// gobbling up input. But... match state handling is already somewhat
    /// expensive, so maybe this wouldn't do much? Either way, it's worth
    /// experimenting.
    pateps_offset: usize,
    /// The first explicit slot index. This refers to the first slot appearing
    /// immediately after the last implicit slot. It is always 'patterns.len()
    /// * 2'.
    ///
    /// We record this because we only store the explicit slots in our DFA
    /// transition table that need to be saved. Implicit slots are handled
    /// automatically as part of the search.
    explicit_slot_start: usize,
}

impl DFA {
    /// Parse the given regular expression using the default configuration and
    /// return the corresponding one-pass DFA.
    ///
    /// If you want a non-default configuration, then use the [`Builder`] to
    /// set your own configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{dfa::onepass::DFA, Match};
    ///
    /// let re = DFA::new("foo[0-9]+bar")?;
    /// let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    ///
    /// re.find(&mut cache, "foo12345barzzz", &mut caps);
    /// assert_eq!(Some(Match::must(0, 0..11)), caps.get_match());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn new(pattern: &str) -> Result<DFA, Error> {
        DFA::builder().build(pattern)
    }

    /// Like `new`, but parses multiple patterns into a single "multi regex."
    /// This similarly uses the default regex configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{dfa::onepass::DFA, Match};
    ///
    /// let re = DFA::new_many(&["[a-z]+", "[0-9]+"])?;
    /// let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    ///
    /// re.find(&mut cache, "abc123", &mut caps);
    /// assert_eq!(Some(Match::must(0, 0..3)), caps.get_match());
    ///
    /// re.find(&mut cache, "123abc", &mut caps);
    /// assert_eq!(Some(Match::must(1, 0..3)), caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn new_many<P: AsRef<str>>(patterns: &[P]) -> Result<DFA, Error> {
        DFA::builder().build_many(patterns)
    }

    /// Like `new`, but builds a one-pass DFA directly from an NFA. This is
    /// useful if you already have an NFA, or even if you hand-assembled the
    /// NFA.
    ///
    /// # Example
    ///
    /// This shows how to hand assemble a regular expression via its HIR,
    /// compile an NFA from it and build a one-pass DFA from the NFA.
    ///
    /// ```
    /// use regex_automata::{
    ///     dfa::onepass::DFA,
    ///     nfa::thompson::NFA,
    ///     Match,
    /// };
    /// use regex_syntax::hir::{Hir, Class, ClassBytes, ClassBytesRange};
    ///
    /// let hir = Hir::class(Class::Bytes(ClassBytes::new(vec![
    ///     ClassBytesRange::new(b'0', b'9'),
    ///     ClassBytesRange::new(b'A', b'Z'),
    ///     ClassBytesRange::new(b'_', b'_'),
    ///     ClassBytesRange::new(b'a', b'z'),
    /// ])));
    ///
    /// let config = NFA::config().nfa_size_limit(Some(1_000));
    /// let nfa = NFA::compiler().configure(config).build_from_hir(&hir)?;
    ///
    /// let re = DFA::new_from_nfa(nfa)?;
    /// let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    /// let expected = Some(Match::must(0, 0..1));
    /// re.find(&mut cache, "A", &mut caps);
    /// assert_eq!(expected, caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new_from_nfa(nfa: NFA) -> Result<DFA, Error> {
        DFA::builder().build_from_nfa(nfa)
    }

    /// Create a new one-pass DFA that matches every input.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{dfa::onepass::DFA, Match};
    ///
    /// let dfa = DFA::always_match()?;
    /// let mut cache = dfa.create_cache();
    /// let mut caps = dfa.create_captures();
    ///
    /// let expected = Match::must(0, 0..0);
    /// dfa.find(&mut cache, "", &mut caps);
    /// assert_eq!(Some(expected), caps.get_match());
    /// dfa.find(&mut cache, "foo", &mut caps);
    /// assert_eq!(Some(expected), caps.get_match());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn always_match() -> Result<DFA, Error> {
        let nfa = thompson::NFA::always_match();
        Builder::new().build_from_nfa(nfa)
    }

    /// Create a new one-pass DFA that never matches any input.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::dfa::onepass::DFA;
    ///
    /// let dfa = DFA::never_match()?;
    /// let mut cache = dfa.create_cache();
    /// let mut caps = dfa.create_captures();
    ///
    /// dfa.find(&mut cache, "", &mut caps);
    /// assert_eq!(None, caps.get_match());
    /// dfa.find(&mut cache, "foo", &mut caps);
    /// assert_eq!(None, caps.get_match());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn never_match() -> Result<DFA, Error> {
        let nfa = thompson::NFA::never_match();
        Builder::new().build_from_nfa(nfa)
    }

    /// Return a default configuration for a DFA.
    ///
    /// This is a convenience routine to avoid needing to import the `Config`
    /// type when customizing the construction of a DFA.
    ///
    /// # Example
    ///
    /// This example shows how to change the match semantics of this DFA from
    /// its default "leftmost first" to "all." When using "all," non-greediness
    /// doesn't apply and neither does preference order matching. Instead, the
    /// longest match possible is always returned. (Although, by construction,
    /// it's impossible for a one-pass DFA to have a different answer for
    /// "preference order" vs "longest match.")
    ///
    /// ```
    /// use regex_automata::{dfa::onepass::DFA, Match, MatchKind};
    ///
    /// let re = DFA::builder()
    ///     .configure(DFA::config().match_kind(MatchKind::All))
    ///     .build(r"(abc)+?")?;
    /// let mut cache = re.create_cache();
    /// let mut caps = re.create_captures();
    ///
    /// re.find(&mut cache, "abcabc", &mut caps);
    /// // Normally, the non-greedy repetition would give us a 0..3 match.
    /// assert_eq!(Some(Match::must(0, 0..6)), caps.get_match());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn config() -> Config {
        Config::new()
    }

    /// Return a builder for configuring the construction of a DFA.
    ///
    /// This is a convenience routine to avoid needing to import the
    /// [`Builder`] type in common cases.
    ///
    /// # Example
    ///
    /// This example shows how to use the builder to disable UTF-8 mode.
    ///
    /// ```
    /// use regex_automata::{
    ///     dfa::onepass::DFA,
    ///     util::syntax,
    ///     Match,
    /// };
    ///
    /// let vm = DFA::builder()
    ///     .configure(DFA::config().utf8(false))
    ///     .syntax(syntax::Config::new().utf8(false))
    ///     .build(r"foo(?-u:[^b])ar.*")?;
    /// let (mut cache, mut caps) = (vm.create_cache(), vm.create_captures());
    ///
    /// let haystack = b"foo\xFFarzz\xE2\x98\xFF\n";
    /// let expected = Some(Match::must(0, 0..8));
    /// vm.find(&mut cache, haystack, &mut caps);
    /// assert_eq!(expected, caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn builder() -> Builder {
        Builder::new()
    }

    /// Create a new `Input` for the given haystack.
    ///
    /// The `Input` returned is configured to match the configuration of this
    /// DFA. Notably, since the one-pass DFA does not support unanchored
    /// searches, this sets the [`Input::anchored`] mode to [`Anchored::No`].
    ///
    /// This routine is useful when using the lower-level [`DFA::search`] API.
    #[inline]
    pub fn create_input<'h, 'p, H: ?Sized + AsRef<[u8]>>(
        &'p self,
        haystack: &'h H,
    ) -> Input<'h, 'p> {
        let c = self.get_config();
        Input::new(haystack.as_ref())
            .anchored(Anchored::Yes)
            .utf8(c.get_utf8())
    }

    /// Create a new empty set of capturing groups that is guaranteed to be
    /// valid for the search APIs on this DFA.
    ///
    /// A `Captures` value created for a specific DFA cannot be used with any
    /// other DFA.
    ///
    /// This is a convenience function for [`Captures::all`]. See the
    /// [`Captures`] documentation for an explanation of its alternative
    /// constructors that permit the DFA to do less work during a search, and
    /// thus might make it faster.
    #[inline]
    pub fn create_captures(&self) -> Captures {
        Captures::all(self.nfa.group_info().clone())
    }

    /// Create a new cache for this DFA.
    ///
    /// The cache returned should only be used for searches for this
    /// DFA. If you want to reuse the cache for another DFA, then you
    /// must call [`Cache::reset`] with that DFA (or, equivalently,
    /// [`DFA::reset_cache`]).
    #[inline]
    pub fn create_cache(&self) -> Cache {
        Cache::new(self)
    }

    /// Reset the given cache such that it can be used for searching with the
    /// this DFA (and only this DFA).
    ///
    /// A cache reset permits reusing memory already allocated in this cache
    /// with a different DFA.
    ///
    /// # Example
    ///
    /// This shows how to re-purpose a cache for use with a different DFA.
    ///
    /// ```
    /// use regex_automata::{dfa::onepass::DFA, Match};
    ///
    /// let re1 = DFA::new(r"\w")?;
    /// let re2 = DFA::new(r"\W")?;
    /// let mut caps1 = re1.create_captures();
    /// let mut caps2 = re2.create_captures();
    ///
    /// let mut cache = re1.create_cache();
    /// assert_eq!(
    ///     Some(Match::must(0, 0..2)),
    ///     { re1.find(&mut cache, "Δ", &mut caps1); caps1.get_match() },
    /// );
    ///
    /// // Using 'cache' with re2 is not allowed. It may result in panics or
    /// // incorrect results. In order to re-purpose the cache, we must reset
    /// // it with the one-pass DFA we'd like to use it with.
    /// //
    /// // Similarly, after this reset, using the cache with 're1' is also not
    /// // allowed.
    /// re2.reset_cache(&mut cache);
    /// assert_eq!(
    ///     Some(Match::must(0, 0..3)),
    ///     { re2.find(&mut cache, "☃", &mut caps2); caps2.get_match() },
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn reset_cache(&self, cache: &mut Cache) {
        cache.reset(self);
    }

    /// Return the config for this one-pass DFA.
    #[inline]
    pub fn get_config(&self) -> &Config {
        &self.config
    }

    /// Returns a reference to the underlying NFA.
    #[inline]
    pub fn get_nfa(&self) -> &NFA {
        &self.nfa
    }

    /// Returns the total number of patterns compiled into this DFA.
    ///
    /// In the case of a DFA that contains no patterns, this returns `0`.
    #[inline]
    pub fn pattern_len(&self) -> usize {
        self.get_nfa().pattern_len()
    }

    /// Returns the total number of states in this one-pass DFA.
    ///
    /// Note that unlike dense or sparse DFAs, a one-pass DFA does not expose
    /// a low level DFA API. Therefore, this routine has little use other than
    /// being informational.
    #[inline]
    pub fn state_len(&self) -> usize {
        self.table.len() >> self.stride2()
    }

    /// Returns the total number of elements in the alphabet for this DFA.
    ///
    /// That is, this returns the total number of transitions that each
    /// state in this DFA must have. The maximum alphabet size is 256, which
    /// corresponds to each possible byte value.
    ///
    /// The alphabet size may be less than 256 though, and unless
    /// [`Config::byte_classes`] is disabled, it is typically must less than
    /// 256. Namely, bytes are grouped into equivalence classes such that no
    /// two bytes in the same class can distinguish a match from a non-match.
    /// For example, in the regex `^[a-z]+$`, the ASCII bytes `a-z` could
    /// all be in the same equivalence class. This leads to a massive space
    /// savings.
    ///
    /// Note though that the alphabet length does _not_ necessarily equal the
    /// total stride space taken up by a single DFA state in the transition
    /// table. Namely, for performance reasons, the stride is always the
    /// smallest power of two that is greater than or equal to the alphabet
    /// length. For this reason, [`DFA::stride`] or [`DFA::stride2`] are
    /// often more useful. The alphabet length is typically useful only for
    /// informational purposes.
    ///
    /// Note also that unlike dense or sparse DFAs, a one-pass DFA does
    /// not have a special end-of-input (EOI) transition. This is because
    /// a one-pass DFA handles look-around assertions explicitly (like the
    /// [`PikeVM`](crate::nfa::thompson::pikevm::PikeVM)) and does not build
    /// them into the transitions of the DFA.
    #[inline]
    pub fn alphabet_len(&self) -> usize {
        self.alphabet_len
    }

    /// Returns the total stride for every state in this DFA, expressed as the
    /// exponent of a power of 2. The stride is the amount of space each state
    /// takes up in the transition table, expressed as a number of transitions.
    /// (Unused transitions map to dead states.)
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
    /// while the maximum `stride2` value is `9` (corresponding to a stride
    /// of `512`). The maximum in theory should be `8`, but because of some
    /// implementation quirks that may be relaxed in the future, it is one more
    /// than `8`. (Do note that a maximal stride is incredibly rare, as it
    /// would imply that there is almost no redundant in the regex pattern.)
    ///
    /// Note that unlike dense or sparse DFAs, a one-pass DFA does not expose
    /// a low level DFA API. Therefore, this routine has little use other than
    /// being informational.
    #[inline]
    pub fn stride2(&self) -> usize {
        self.stride2
    }

    /// Returns the total stride for every state in this DFA. This corresponds
    /// to the total number of transitions used by each state in this DFA's
    /// transition table.
    ///
    /// Please see [`DFA::stride2`] for more information. In particular, this
    /// returns the stride as the number of transitions, where as `stride2`
    /// returns it as the exponent of a power of 2.
    ///
    /// Note that unlike dense or sparse DFAs, a one-pass DFA does not expose
    /// a low level DFA API. Therefore, this routine has little use other than
    /// being informational.
    #[inline]
    pub fn stride(&self) -> usize {
        1 << self.stride2()
    }

    /// Returns the memory usage, in bytes, of this DFA.
    ///
    /// The memory usage is computed based on the number of bytes used to
    /// represent this DFA.
    ///
    /// This does **not** include the stack size used up by this DFA. To
    /// compute that, use `std::mem::size_of::<onepass::DFA>()`.
    #[inline]
    pub fn memory_usage(&self) -> usize {
        use core::mem::size_of;

        self.table.len() * size_of::<Transition>()
            + self.starts.len() * size_of::<StateID>()
    }
}

impl DFA {
    /// Executes an anchored leftmost forward search, and returns true if and
    /// only if this one-pass DFA matches the given haystack.
    ///
    /// This routine may short circuit if it knows that scanning future
    /// input will never lead to a different result. In particular, if the
    /// underlying DFA enters a match state, then this routine will return
    /// `true` immediately without inspecting any future input. (Consider how
    /// this might make a difference given the regex `a+` on the haystack
    /// `aaaaaaaaaaaaaaa`. This routine can stop after it sees the first `a`,
    /// but routines like `find` need to continue searching because `+` is
    /// greedy by default.)
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::dfa::onepass::DFA;
    ///
    /// let re = DFA::new("foo[0-9]+bar")?;
    /// let mut cache = re.create_cache();
    ///
    /// assert!(re.is_match(&mut cache, "foo12345bar"));
    /// assert!(!re.is_match(&mut cache, "foobar"));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn is_match<H: AsRef<[u8]>>(
        &self,
        cache: &mut Cache,
        haystack: H,
    ) -> bool {
        let input = self.create_input(haystack.as_ref()).earliest(true);
        self.search_slots(cache, &input, &mut []).is_some()
    }

    /// Executes an anchored leftmost forward search and writes the spans
    /// of capturing groups that participated in a match into the provided
    /// [`Captures`] value. If no match was found, then [`Captures::is_match`]
    /// is guaranteed to return `false`.
    ///
    /// For more control over the input parameters, see [`DFA::search`].
    ///
    /// # Example
    ///
    /// This shows a simple example of a one-pass regex that extracts
    /// capturing group spans.
    ///
    /// ```
    /// use regex_automata::{dfa::onepass::DFA, Match, Span};
    ///
    /// let re = DFA::new(
    ///     // Notice that we use ASCII here. The corresponding Unicode regex
    ///     // is sadly not one-pass.
    ///     "(?P<first>[[:alpha:]]+)[[:space:]]+(?P<last>[[:alpha:]]+)",
    /// )?;
    /// let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    ///
    /// re.find(&mut cache, "Bruce Springsteen", &mut caps);
    /// assert_eq!(Some(Match::must(0, 0..17)), caps.get_match());
    /// assert_eq!(Some(Span::from(0..5)), caps.get_group(1));
    /// assert_eq!(Some(Span::from(6..17)), caps.get_group_by_name("last"));
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn find<H: AsRef<[u8]>>(
        &self,
        cache: &mut Cache,
        haystack: H,
        caps: &mut Captures,
    ) {
        let input = self.create_input(haystack.as_ref());
        self.search(cache, &input, caps)
    }

    /// Executes an anchored leftmost forward search and writes the spans
    /// of capturing groups that participated in a match into the provided
    /// [`Captures`] value. If no match was found, then [`Captures::is_match`]
    /// is guaranteed to return `false`.
    ///
    /// This is like [`DFA::find`], except it provides some additional control
    /// over how the search is executed. Those parameters are configured via a
    /// [`Input`].
    ///
    /// The examples below demonstrate each of these additional parameters.
    ///
    /// Note that not all parameters on an `Input` are used by a one-pass
    /// DFA. For example, a one-pass DFA only executes anchored searches,
    /// and thus does not use any prefilter even if one is configured.
    ///
    /// # Example: specific pattern search
    ///
    /// This example shows how to build a multi-regex that permits searching
    /// for specific patterns. Note that this is somewhat less useful than
    /// in other regex engines, since a one-pass DFA by definition has no
    /// ambiguity about which pattern can match at a position. That is, if it
    /// were possible for two different patterns to match at the same starting
    /// position, then the multi-regex would not be one-pass and construction
    /// would have failed.
    ///
    /// Nevertheless, this can still be useful if you only care about matches
    /// for a specific pattern, and want the DFA to report "no match" even if
    /// some other pattern would have matched.
    ///
    /// Note that in order to make use of this functionality,
    /// [`Config::starts_for_each_pattern`] must be enabled. It is disabled
    /// by default since it may result in higher memory usage.
    ///
    /// ```
    /// use regex_automata::{dfa::onepass::DFA, Anchored, Match, PatternID};
    ///
    /// let re = DFA::builder()
    ///     .configure(DFA::config().starts_for_each_pattern(true))
    ///     .build_many(&["[a-z]+", "[0-9]+"])?;
    /// let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    /// let haystack = "123abc";
    ///
    /// // A normal multi-pattern search will show pattern 1 matches.
    /// re.search(&mut cache, &re.create_input(haystack), &mut caps);
    /// assert_eq!(Some(Match::must(1, 0..3)), caps.get_match());
    ///
    /// // If we only want to report pattern 0 matches, then we'll get no
    /// // match here.
    /// let input = re.create_input(haystack)
    ///     .anchored(Anchored::Pattern(PatternID::must(0)));
    /// re.search(&mut cache, &input, &mut caps);
    /// assert_eq!(None, caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: specifying the bounds of a search
    ///
    /// This example shows how providing the bounds of a search can produce
    /// different results than simply sub-slicing the haystack.
    ///
    /// ```
    /// use regex_automata::{dfa::onepass::DFA, Match};
    ///
    /// // one-pass DFAs fully support Unicode word boundaries!
    /// // A sad joke is that a Unicode aware regex like \w+\s is not one-pass.
    /// // :-(
    /// let re = DFA::new(r"\b[0-9]{3}\b")?;
    /// let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    /// let haystack = "foo123bar";
    ///
    /// // Since we sub-slice the haystack, the search doesn't know about
    /// // the larger context and assumes that `123` is surrounded by word
    /// // boundaries. And of course, the match position is reported relative
    /// // to the sub-slice as well, which means we get `0..3` instead of
    /// // `3..6`.
    /// let expected = Some(Match::must(0, 0..3));
    /// let input = re.create_input(&haystack[3..6]);
    /// re.search(&mut cache, &input, &mut caps);
    /// assert_eq!(expected, caps.get_match());
    ///
    /// // But if we provide the bounds of the search within the context of the
    /// // entire haystack, then the search can take the surrounding context
    /// // into account. (And if we did find a match, it would be reported
    /// // as a valid offset into `haystack` instead of its sub-slice.)
    /// let expected = None;
    /// let input = re.create_input(haystack).range(3..6);
    /// re.search(&mut cache, &input, &mut caps);
    /// assert_eq!(expected, caps.get_match());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn search(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        caps: &mut Captures,
    ) {
        let pid = self.search_slots(cache, input, caps.slots_mut());
        caps.set_pattern(pid);
    }

    /// Executes an anchored leftmost forward search and writes the spans
    /// of capturing groups that participated in a match into the provided
    /// `slots`, and returns the matching pattern ID. The contents of the
    /// slots for patterns other than the matching pattern are unspecified. If
    /// no match was found, then `None` is returned and the contents of all
    /// `slots` is unspecified.
    ///
    /// This is like [`DFA::search`], but it accepts a raw slots slice
    /// instead of a `Captures` value. This is useful in contexts where you
    /// don't want or need to allocate a `Captures`.
    ///
    /// It is legal to pass _any_ number of slots to this routine. If the regex
    /// engine would otherwise write a slot offset that doesn't fit in the
    /// provided slice, then it is simply skipped. In general though, there are
    /// usually three slice lengths you might want to use:
    ///
    /// * An empty slice, if you only care about which pattern matched.
    /// * A slice with
    /// [`pattern_len() * 2`](crate::dfa::onepass::DFA::pattern_len)
    /// slots, if you only care about the overall match spans for each matching
    /// pattern.
    /// * A slice with
    /// [`slot_len()`](crate::util::captures::GroupInfo::slot_len) slots, which
    /// permits recording match offsets for every capturing group in every
    /// pattern.
    ///
    /// # Example
    ///
    /// This example shows how to find the overall match offsets in a
    /// multi-pattern search without allocating a `Captures` value. Indeed, we
    /// can put our slots right on the stack.
    ///
    /// ```
    /// use regex_automata::{dfa::onepass::DFA, PatternID};
    ///
    /// let re = DFA::new_many(&[
    ///     r"[a-zA-Z]+",
    ///     r"[0-9]+",
    /// ])?;
    /// let mut cache = re.create_cache();
    /// let input = re.create_input("123");
    ///
    /// // We only care about the overall match offsets here, so we just
    /// // allocate two slots for each pattern. Each slot records the start
    /// // and end of the match.
    /// let mut slots = [None; 4];
    /// let pid = re.search_slots(&mut cache, &input, &mut slots);
    /// assert_eq!(Some(PatternID::must(1)), pid);
    ///
    /// // The overall match offsets are always at 'pid * 2' and 'pid * 2 + 1'.
    /// // See 'GroupInfo' for more details on the mapping between groups and
    /// // slot indices.
    /// let slot_start = pid.unwrap().as_usize() * 2;
    /// let slot_end = slot_start + 1;
    /// assert_eq!(Some(0), slots[slot_start].map(|s| s.get()));
    /// assert_eq!(Some(3), slots[slot_end].map(|s| s.get()));
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn search_slots(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Option<PatternID> {
        match self.search_imp(cache, input, slots) {
            None => return None,
            Some(pid) if !input.get_utf8() => return Some(pid),
            Some(pid) => {
                // These slot indices are always correct because we know our
                // 'pid' is valid and thus we know that the slot indices for it
                // are valid.
                let slot_start = pid.as_usize().wrapping_mul(2);
                let slot_end = slot_start.wrapping_add(1);
                if slot_end >= slots.len() {
                    return Some(pid);
                }
                // These unwraps are OK because we know we have a match and
                // we know our caller provided slots are big enough.
                let start = slots[slot_start].unwrap().get();
                let end = slots[slot_end].unwrap().get();
                // If our match splits a codepoint, then we cannot report is
                // as a match. And since one-pass DFAs only support anchored
                // searches, we don't try to skip ahead to find the next match.
                if start == end && !input.is_char_boundary(start) {
                    return None;
                }
                Some(pid)
            }
        }
    }
}

impl DFA {
    fn search_imp(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Option<PatternID> {
        // PERF: Some ideas. I ran out of steam after my initial impl to try
        // many of these.
        //
        // 1) Try doing more state shuffling. Right now, all we do is push
        // match states to the end of the transition table so that we can do
        // 'if sid >= self.min_match_id' to know whether we're in a match
        // state or not. But what about doing something like dense DFAs and
        // pushing dead, match and states with captures/looks all toward the
        // beginning of the transition table. Then we could do 'if sid <=
        // self.max_special_id', in which case, we need to do some special
        // handling of some sort. Otherwise, we get the happy path, just
        // like in a DFA search. The main argument against this is that the
        // one-pass DFA is likely to be used most often with capturing groups
        // and if capturing groups are common, then this might wind up being a
        // pessimization.
        //
        // 2) Consider moving 'PatternEpsilons' out of the transition table.
        // It is only needed for match states and usually a small minority of
        // states are match states. Therefore, we're using an extra 'u64' for
        // most states.
        //
        // 3) I played around with the match state handling and it seems like
        // there is probably a lot left on the table for improvement. The
        // key tension is that the 'find_match' routine is a giant mess, but
        // splitting it out into a non-inlineable function is a non-starter
        // because the match state might consume input. In theory, we could
        // detect whether a match state consumes input and then specialize our
        // search routine based on that. In that case, maybe an extra function
        // call is OK, but even then, it might be too much of a latency hit.
        // Another idea is to just try and figure out how to reduce the code
        // size of 'find_match'. RE2 has a trick here where the match handling
        // isn't done if we know the next byte of input yields a match too.
        // Maybe we adopt that?
        //
        // This just might be a tricky DFA to optimize.

        if input.is_done() {
            return None;
        }
        // We unfortunately have a bit of book-keeping to do to set things
        // up. We do have to setup our cache and clear all of our slots. In
        // particular, clearing the slots is necessary for the case where we
        // report a match, but one of the capturing groups didn't participate
        // in the match but had a span set from a previous search. That would
        // be bad. In theory, we could avoid all this slot clearing if we knew
        // that every slot was always activated for every match. Then we would
        // know they would always be overwritten when a match is found.
        let explicit_slots_len = core::cmp::min(
            Slots::LIMIT,
            slots.len().saturating_sub(self.explicit_slot_start),
        );
        cache.setup_search(explicit_slots_len);
        for slot in cache.explicit_slots() {
            *slot = None;
        }
        for slot in slots.iter_mut() {
            *slot = None;
        }
        // We set the starting slots for every pattern up front. This does
        // increase our latency somewhat, but it avoids having to do it every
        // time we see a match state (which could be many times in a single
        // search if the match state consumes input).
        for pid in self.nfa.patterns() {
            let i = pid.as_usize() * 2;
            if i >= slots.len() {
                break;
            }
            slots[i] = NonMaxUsize::new(input.start());
        }
        let mut pid = None;
        let mut sid = match input.get_anchored() {
            Anchored::Yes => self.start(),
            Anchored::Pattern(pid) => self.start_pattern(pid),
            Anchored::No => {
                panic!("one-pass DFA does not support unanchored searches")
            }
        };
        for at in input.start()..input.end() {
            if sid >= self.min_match_id {
                if self.find_match(cache, input, at, sid, slots, &mut pid) {
                    if input.get_earliest() {
                        return pid;
                    }
                }
            }

            let trans = self.transition(sid, input.haystack()[at]);
            sid = trans.state_id();
            let epsilons = trans.epsilons();
            if sid == DEAD || !epsilons.look_matches(input.haystack(), at) {
                return pid;
            }
            epsilons.slots().apply(at, cache.explicit_slots());
        }
        if sid >= self.min_match_id {
            self.find_match(cache, input, input.end(), sid, slots, &mut pid);
        }
        pid
    }

    /// Assumes 'sid' is a match state and looks for whether a match can
    /// be reported. If so, appropriate offsets are written to 'slots' and
    /// 'matched_pid' is set to the matching pattern ID.
    ///
    /// Even when 'sid' is a match state, it's possible that a match won't
    /// be reported. For example, when the conditional epsilon transitions
    /// leading to the match state aren't satisfied at the given position in
    /// the haystack.
    #[inline(always)]
    fn find_match(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        at: usize,
        sid: StateID,
        slots: &mut [Option<NonMaxUsize>],
        matched_pid: &mut Option<PatternID>,
    ) -> bool {
        debug_assert!(sid >= self.min_match_id);
        let pateps = self.pattern_epsilons(sid);
        let epsilons = pateps.epsilons();
        if !epsilons.look_matches(input.haystack(), at) {
            return false;
        }
        let pid = pateps.pattern_id_unchecked();
        // This calculation is always correct because we know our 'pid' is
        // valid and thus we know that the slot indices for it are valid.
        let slot_end = pid.as_usize().wrapping_mul(2).wrapping_add(1);
        // Set the implicit 'end' slot for the matching pattern. (The 'start'
        // slot was set at the beginning of the search.)
        if slot_end < slots.len() {
            slots[slot_end] = NonMaxUsize::new(at);
        }
        // If the caller provided enough room, copy the previously recorded
        // explicit slots from our scratch space to the caller provided slots.
        // We *also* need to set any explicit slots that are active as part of
        // the path to the match state.
        if self.explicit_slot_start < slots.len() {
            // NOTE: The 'cache.explicit_slots()' slice is setup at the
            // beginning of every search such that it is guaranteed to return a
            // slice of length equivalent to 'slots[explicit_slot_start..]'.
            slots[self.explicit_slot_start..]
                .copy_from_slice(cache.explicit_slots());
            epsilons.slots().apply(at, &mut slots[self.explicit_slot_start..]);
        }
        *matched_pid = Some(pid);
        true
    }
}

impl DFA {
    /// Returns the anchored start state for matching any pattern in this DFA.
    fn start(&self) -> StateID {
        self.starts[0]
    }

    /// Returns the anchored start state for matching the given pattern. If
    /// the given pattern is not in this DFA or if 'starts_for_each_pattern'
    /// was not enabled, then this panics.
    fn start_pattern(&self, pid: PatternID) -> StateID {
        assert!(pid.as_usize() < self.pattern_len(), "invalid pattern ID");
        self.starts.get(pid.one_more()).copied().expect(
            "starts_for_each_pattern must be enabled to \
             search by a specific pattern",
        )
    }

    /// Returns the transition from the given state ID and byte of input. The
    /// transition includes the next state ID, the slots that should be saved
    /// and any conditional epsilon transitions that must be satisfied in order
    /// to take this transition.
    fn transition(&self, sid: StateID, byte: u8) -> Transition {
        let class = self.classes.get(byte);
        self.table[sid.as_usize() + class.as_usize()]
    }

    /// Set the transition from the given state ID and byte of input to the
    /// transition given.
    fn set_transition(&mut self, sid: StateID, byte: u8, to: Transition) {
        let class = self.classes.get(byte);
        self.table[sid.as_usize() + class.as_usize()] = to;
    }

    /// Return an iterator of "sparse" transitions for the given state ID.
    /// "sparse" in this context means that consecutive transitions that are
    /// equivalent are returned as one group, and transitions to the DEAD state
    /// are ignored.
    ///
    /// This winds up being useful for debug printing, since it's much terser
    /// to display runs of equivalent transitions than the transition for every
    /// possible byte value. Indeed, in practice, it's very common for runs
    /// of equivalent transitions to appear.
    fn sparse_transitions(&self, sid: StateID) -> SparseTransitionIter<'_> {
        let start = sid.as_usize();
        let end = start + self.alphabet_len();
        SparseTransitionIter {
            it: self.table[start..end].iter().enumerate(),
            cur: None,
        }
    }

    /// Return the pattern epsilons for the given state ID.
    ///
    /// If the given state ID does not correspond to a match state ID, then the
    /// pattern epsilons returned is empty.
    fn pattern_epsilons(&self, sid: StateID) -> PatternEpsilons {
        PatternEpsilons(self.table[sid.as_usize() + self.pateps_offset].0)
    }

    /// Set the pattern epsilons for the given state ID.
    fn set_pattern_epsilons(&mut self, sid: StateID, pateps: PatternEpsilons) {
        self.table[sid.as_usize() + self.pateps_offset] = Transition(pateps.0);
    }

    /// Map the given state ID to its index. The index of a state is an
    /// incrementing integer starting at 0 determined by the order in which the
    /// state was added.
    fn to_index(&self, id: StateID) -> usize {
        id.as_usize() >> self.stride2()
    }

    /// Map the given state index to its ID. The ID is a "pre-multiplied"
    /// index into the DFA transition table. This avoids a multiplication
    /// at search time.
    fn to_state_id(&self, index: usize) -> StateID {
        // CORRECTNESS: If the given index is not valid, then it is not
        // required for this to panic or return a valid state ID. We'll "just"
        // wind up with panics or silent logic errors at some other point.
        StateID::new_unchecked(index << self.stride2())
    }

    /// Returns the state ID prior to the one given. This returns None if the
    /// given ID is the first DFA state.
    fn prev_state_id(&self, id: StateID) -> Option<StateID> {
        if id == DEAD {
            None
        } else {
            // CORRECTNESS: Since 'id' is not the first state and since all
            // states have the same stride, it follows that subtracting the
            // stride from a valid state ID must yield the previous state ID.
            Some(StateID::new_unchecked(
                id.as_usize().checked_sub(self.stride()).unwrap(),
            ))
        }
    }

    /// Returns the state ID of the last state in this DFA's transition table.
    /// "last" in this context means the last state to appear in memory, i.e.,
    /// the one with the greatest ID.
    fn last_state_id(&self) -> StateID {
        // CORRECTNESS: A DFA table is always non-empty since it always at
        // least contains a DEAD state. Since every state has the same stride,
        // it follows that subtracting it from the length of the table will
        // give us the ID of the last state in the table.
        StateID::new_unchecked(
            self.table.len().checked_sub(self.stride()).unwrap(),
        )
    }

    /// Move the transitions from 'id1' to 'id2' and vice versa.
    ///
    /// WARNING: This does not update the rest of the transition table to have
    /// transitions to 'id1' changed to 'id2' and vice versa. This merely moves
    /// the states in memory.
    pub(super) fn swap_states(&mut self, id1: StateID, id2: StateID) {
        for b in 0..self.stride() {
            self.table.swap(id1.as_usize() + b, id2.as_usize() + b);
        }
    }

    /// Map all state IDs in this DFA (transition table + start states)
    /// according to the closure given.
    pub(super) fn remap(&mut self, map: impl Fn(StateID) -> StateID) {
        for i in 0..self.state_len() {
            let id = StateID::new_unchecked(i << self.stride2());
            for b in 0..self.alphabet_len() {
                let next = self.table[id.as_usize() + b].state_id();
                self.table[id.as_usize() + b].set_state_id(map(next));
            }
        }
        for i in 0..self.starts.len() {
            self.starts[i] = map(self.starts[i]);
        }
    }
}

impl core::fmt::Debug for DFA {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        fn debug_id(
            f: &core::fmt::Formatter,
            dfa: &DFA,
            sid: StateID,
        ) -> usize {
            if f.alternate() {
                sid.as_usize()
            } else {
                sid.as_usize() >> dfa.stride2()
            }
        }

        fn debug_state_transitions(
            f: &mut core::fmt::Formatter,
            dfa: &DFA,
            sid: StateID,
        ) -> core::fmt::Result {
            for (i, (start, end, trans)) in
                dfa.sparse_transitions(sid).enumerate()
            {
                let next = trans.state_id();
                if i > 0 {
                    write!(f, ", ")?;
                }
                if start == end {
                    write!(
                        f,
                        "{:?} => {:?}",
                        DebugByte(start),
                        debug_id(f, dfa, next)
                    )?;
                } else {
                    write!(
                        f,
                        "{:?}-{:?} => {:?}",
                        DebugByte(start),
                        DebugByte(end),
                        debug_id(f, dfa, next),
                    )?;
                }
                if !trans.epsilons().is_empty() {
                    write!(f, " ({:?})", trans.epsilons())?;
                }
            }
            Ok(())
        }

        writeln!(f, "onepass::DFA(")?;
        for index in 0..self.state_len() {
            let sid = StateID::must(index << self.stride2());
            let pateps = self.pattern_epsilons(sid);
            if sid == DEAD {
                write!(f, "D ")?;
            } else if pateps.pattern_id().is_some() {
                write!(f, "* ")?;
            } else {
                write!(f, "  ")?;
            }
            write!(f, "{:06?}", debug_id(f, self, sid))?;
            if !pateps.is_empty() {
                write!(f, " ({:?})", pateps)?;
            }
            write!(f, ": ")?;
            debug_state_transitions(f, self, sid)?;
            write!(f, "\n")?;
        }
        writeln!(f, "")?;
        for (i, &sid) in self.starts.iter().enumerate() {
            if i == 0 {
                writeln!(f, "START(ALL): {:?}", debug_id(f, self, sid))?;
            } else {
                writeln!(
                    f,
                    "START(pattern: {:?}): {:?}",
                    i - 1,
                    debug_id(f, self, sid)
                )?;
            }
        }
        writeln!(f, "state length: {:?}", self.state_len())?;
        writeln!(f, "pattern length: {:?}", self.pattern_len())?;
        writeln!(f, ")")?;
        Ok(())
    }
}

/// An iterator over groups of consecutive equivalent transitions in a single
/// state.
#[derive(Debug)]
struct SparseTransitionIter<'a> {
    it: core::iter::Enumerate<core::slice::Iter<'a, Transition>>,
    cur: Option<(u8, u8, Transition)>,
}

impl<'a> Iterator for SparseTransitionIter<'a> {
    type Item = (u8, u8, Transition);

    fn next(&mut self) -> Option<(u8, u8, Transition)> {
        while let Some((b, &trans)) = self.it.next() {
            // Fine because we'll never have more than u8::MAX transitions in
            // one state.
            let b = b.as_u8();
            let (prev_start, prev_end, prev_trans) = match self.cur {
                Some(t) => t,
                None => {
                    self.cur = Some((b, b, trans));
                    continue;
                }
            };
            if prev_trans == trans {
                self.cur = Some((prev_start, b, prev_trans));
            } else {
                self.cur = Some((b, b, trans));
                if prev_trans.state_id() != DEAD {
                    return Some((prev_start, prev_end, prev_trans));
                }
            }
        }
        if let Some((start, end, trans)) = self.cur.take() {
            if trans.state_id() != DEAD {
                return Some((start, end, trans));
            }
        }
        None
    }
}

/// A cache represents mutable state that a one-pass [`DFA`] requires during a
/// search.
///
/// For a given one-pass DFA, its corresponding cache may be created either via
/// [`DFA::create_cache`], or via [`Cache::new`]. They are equivalent in every
/// way, except the former does not require explicitly importing `Cache`.
///
/// A particular `Cache` is coupled with the one-pass DFA from which it was
/// created. It may only be used with that one-pass DFA. A cache and its
/// allocations may be re-purposed via [`Cache::reset`], in which case, it can
/// only be used with the new one-pass DFA (and not the old one).
#[derive(Clone, Debug)]
pub struct Cache {
    /// Scratch space used to store slots during a search. Basically, we use
    /// the caller provided slots to store slots known when a match occurs.
    /// But after a match occurs, we might continue a search but ultimately
    /// fail to extend the match. When continuing the search, we need some
    /// place to store candidate capture offsets without overwriting the slot
    /// offsets recorded for the most recently seen match.
    explicit_slots: Vec<Option<NonMaxUsize>>,
    /// The number of slots in the caller-provided 'Captures' value for the
    /// current search. This is always at most 'explicit_slots.len()', but
    /// might be less than it, if the caller provided fewer slots to fill.
    explicit_slot_len: usize,
}

impl Cache {
    /// Create a new [`onepass::DFA`](DFA) cache.
    ///
    /// A potentially more convenient routine to create a cache is
    /// [`DFA::create_cache`], as it does not require also importing the
    /// `Cache` type.
    ///
    /// If you want to reuse the returned `Cache` with some other one-pass DFA,
    /// then you must call [`Cache::reset`] with the desired one-pass DFA.
    pub fn new(re: &DFA) -> Cache {
        let mut cache = Cache { explicit_slots: vec![], explicit_slot_len: 0 };
        cache.reset(re);
        cache
    }

    /// Reset this cache such that it can be used for searching with a
    /// different [`onepass::DFA`](DFA).
    ///
    /// A cache reset permits reusing memory already allocated in this cache
    /// with a different one-pass DFA.
    ///
    /// # Example
    ///
    /// This shows how to re-purpose a cache for use with a different one-pass
    /// DFA.
    ///
    /// ```
    /// use regex_automata::{dfa::onepass::DFA, Match};
    ///
    /// let re1 = DFA::new(r"\w")?;
    /// let re2 = DFA::new(r"\W")?;
    /// let mut caps1 = re1.create_captures();
    /// let mut caps2 = re2.create_captures();
    ///
    /// let mut cache = re1.create_cache();
    /// assert_eq!(
    ///     Some(Match::must(0, 0..2)),
    ///     { re1.find(&mut cache, "Δ", &mut caps1); caps1.get_match() },
    /// );
    ///
    /// // Using 'cache' with re2 is not allowed. It may result in panics or
    /// // incorrect results. In order to re-purpose the cache, we must reset
    /// // it with the one-pass DFA we'd like to use it with.
    /// //
    /// // Similarly, after this reset, using the cache with 're1' is also not
    /// // allowed.
    /// re2.reset_cache(&mut cache);
    /// assert_eq!(
    ///     Some(Match::must(0, 0..3)),
    ///     { re2.find(&mut cache, "☃", &mut caps2); caps2.get_match() },
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reset(&mut self, re: &DFA) {
        let explicit_slot_len = re.get_nfa().group_info().explicit_slot_len();
        self.explicit_slots.resize(explicit_slot_len, None);
        self.explicit_slot_len = explicit_slot_len;
    }

    /// Returns the heap memory usage, in bytes, of this cache.
    ///
    /// This does **not** include the stack size used up by this cache. To
    /// compute that, use `std::mem::size_of::<Cache>()`.
    pub fn memory_usage(&self) -> usize {
        self.explicit_slots.len() * core::mem::size_of::<Option<NonMaxUsize>>()
    }

    fn explicit_slots(&mut self) -> &mut [Option<NonMaxUsize>] {
        &mut self.explicit_slots[..self.explicit_slot_len]
    }

    fn setup_search(&mut self, explicit_slot_len: usize) {
        self.explicit_slot_len = explicit_slot_len;
    }
}

/// Represents a single transition in a one-pass DFA.
///
/// The high 24 bits corresponds to the state ID. The low 48 bits corresponds
/// to the transition epsilons, which contains the slots that should be saved
/// when this transition is followed and the conditional epsilon transitions
/// that must be satisfied in order to follow this transition.
#[derive(Clone, Copy, Eq, PartialEq)]
struct Transition(u64);

impl Transition {
    const STATE_ID_BITS: u64 = 24;
    const STATE_ID_SHIFT: u64 = 64 - Transition::STATE_ID_BITS;
    const STATE_ID_LIMIT: u64 = 1 << Transition::STATE_ID_BITS;
    const STATE_ID_MASK: u64 = 0xFFFFFF00_00000000;
    const INFO_MASK: u64 = 0x000000FF_FFFFFFFF;

    /// Return a new transition to the given state ID with the given epsilons.
    fn new(sid: StateID, epsilons: Epsilons) -> Transition {
        Transition((sid.as_u64() << Transition::STATE_ID_SHIFT) | epsilons.0)
    }

    /// Returns true if and only if this transition points to the DEAD state.
    fn is_dead(self) -> bool {
        self.state_id() == DEAD
    }

    /// Return the "next" state ID that this transition points to.
    fn state_id(&self) -> StateID {
        // OK because a Transition has a valid StateID in its upper bits by
        // construction. The cast to usize is also correct, even on 16-bit
        // targets because, again, we know the upper bits is a valid StateID,
        // which can never overflow usize on any supported target.
        StateID::new_unchecked(
            (self.0 >> Transition::STATE_ID_SHIFT).as_usize(),
        )
    }

    /// Set the "next" state ID in this transition.
    fn set_state_id(&mut self, sid: StateID) {
        *self = Transition::new(sid, self.epsilons());
    }

    /// Return the epsilons embedded in this transition.
    fn epsilons(&self) -> Epsilons {
        Epsilons(self.0 & Transition::INFO_MASK)
    }
}

impl core::fmt::Debug for Transition {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        if self.is_dead() {
            return write!(f, "0");
        }
        write!(f, "{}", self.state_id().as_usize())?;
        if !self.epsilons().is_empty() {
            write!(f, "-{:?}", self.epsilons())?;
        }
        Ok(())
    }
}

/// A representation of a match state's pattern ID along with the epsilons for
/// when a match occurs.
///
/// A match state in a one-pass DFA, unlike in a more general DFA, has exactly
/// one pattern ID. If it had more, then the original NFA would not have been
/// one-pass.
///
/// The "epsilons" part of this corresponds to what was found in the epsilon
/// transitions between the transition taken in the last byte of input and the
/// ultimate match state. This might include saving slots and/or conditional
/// epsilon transitions that must be satisfied before one can report the match.
///
/// Technically, every state has room for a 'PatternEpsilons', but it is only
/// ever non-empty for match states.
#[derive(Clone, Copy)]
struct PatternEpsilons(u64);

impl PatternEpsilons {
    const PATTERN_ID_BITS: u64 = 24;
    const PATTERN_ID_SHIFT: u64 = 64 - PatternEpsilons::PATTERN_ID_BITS;
    // A sentinel value indicating that this is not a match state. We don't
    // use 0 since 0 is a valid pattern ID.
    const PATTERN_ID_NONE: u64 = 0x00000000_00FFFFFF;
    const PATTERN_ID_LIMIT: u64 = PatternEpsilons::PATTERN_ID_NONE;
    const PATTERN_ID_MASK: u64 = 0xFFFFFF00_00000000;
    const INFO_MASK: u64 = 0x000000FF_FFFFFFFF;

    /// Return a new empty pattern epsilons that has no pattern ID and has no
    /// epsilons. This is suitable for non-match states.
    fn empty() -> PatternEpsilons {
        PatternEpsilons(
            PatternEpsilons::PATTERN_ID_NONE
                << PatternEpsilons::PATTERN_ID_SHIFT,
        )
    }

    /// Whether this pattern epsilons is empty or not. It's empty when it has
    /// no pattern ID and an empty epsilons.
    fn is_empty(self) -> bool {
        self.pattern_id().is_none() && self.epsilons().is_empty()
    }

    /// Return the pattern ID in this pattern epsilons if one exists.
    fn pattern_id(self) -> Option<PatternID> {
        let pid = self.0 >> PatternEpsilons::PATTERN_ID_SHIFT;
        if pid == PatternEpsilons::PATTERN_ID_LIMIT {
            None
        } else {
            Some(PatternID::new_unchecked(pid.as_usize()))
        }
    }

    /// Returns the pattern ID without checking whether it's valid. If this is
    /// called and there is no pattern ID in this `PatternEpsilons`, then this
    /// will likely produce an incorrect result or possibly even a panic or
    /// an overflow. But safety will not be violated.
    ///
    /// This is useful when you know a particular state is a match state. If
    /// it's a match state, then it must have a pattern ID.
    fn pattern_id_unchecked(self) -> PatternID {
        let pid = self.0 >> PatternEpsilons::PATTERN_ID_SHIFT;
        PatternID::new_unchecked(pid.as_usize())
    }

    /// Return a new pattern epsilons with the given pattern ID, but the same
    /// epsilons.
    fn set_pattern_id(self, pid: PatternID) -> PatternEpsilons {
        PatternEpsilons(
            (pid.as_u64() << PatternEpsilons::PATTERN_ID_SHIFT)
                | (self.0 & PatternEpsilons::INFO_MASK),
        )
    }

    /// Return the epsilons part of this pattern epsilons.
    fn epsilons(self) -> Epsilons {
        Epsilons(self.0 & PatternEpsilons::INFO_MASK)
    }

    /// Return a new pattern epsilons with the given epsilons, but the same
    /// pattern ID.
    fn set_epsilons(self, epsilons: Epsilons) -> PatternEpsilons {
        PatternEpsilons(
            (self.0 & PatternEpsilons::PATTERN_ID_MASK)
                | u64::from(epsilons.0),
        )
    }
}

impl core::fmt::Debug for PatternEpsilons {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        if self.is_empty() {
            return write!(f, "N/A");
        }
        if let Some(pid) = self.pattern_id() {
            write!(f, "{}", pid.as_usize())?;
        }
        if !self.epsilons().is_empty() {
            if self.pattern_id().is_some() {
                write!(f, "/")?;
            }
            write!(f, "{:?}", self.epsilons())?;
        }
        Ok(())
    }
}

/// Epsilons represents all of the NFA epsilons transitions that went into a
/// single transition in a single DFA state. In this case, it only represents
/// the epsilon transitions that have some kind of non-consumin side effect:
/// either the transition requires storing the current position of the search
/// into a slot, or the transition is conditional and requires the current
/// position in the input to satisfy an assertion before the transition may be
/// taken.
///
/// This folds the cumulative effect of a group of NFA states (all connected
/// by epsilon transitions) down into a single set of bits. While these bits
/// can represent all possible conditional epsilon transitions, it only permits
/// storing up to a somewhat small number of slots.
///
/// Epsilons is represented as a 48-bit integer. For example, it is packed into
/// the lower 48 bits of a `Transition`. (Where the high 24 bits contains a
/// `StateID`.)
#[derive(Clone, Copy)]
struct Epsilons(u64);

impl Epsilons {
    const SLOT_MASK: u64 = 0x000000FF_FFFFFF00;
    const SLOT_SHIFT: u64 = 8;
    const LOOK_MASK: u64 = 0x00000000_000000FF;

    /// Create a new empty epsilons. It has no slots and no assertions that
    /// need to be satisfied.
    fn empty() -> Epsilons {
        Epsilons(0)
    }

    /// Returns true if this epsilons contains no slots and no assertions.
    fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns the slot epsilon transitions.
    fn slots(self) -> Slots {
        Slots((self.0 >> Epsilons::SLOT_SHIFT).low_u32())
    }

    /// Set the slot epsilon transitions.
    fn set_slots(self, slots: Slots) -> Epsilons {
        Epsilons(
            (u64::from(slots.0) << Epsilons::SLOT_SHIFT)
                | (self.0 & Epsilons::LOOK_MASK),
        )
    }

    /// Return the set of look-around assertions in these epsilon transitions.
    fn looks(self) -> LookSet {
        LookSet::from_repr(self.0.low_u8())
    }

    /// Set the look-around assertions on these epsilon transitions.
    fn set_looks(self, look_set: LookSet) -> Epsilons {
        Epsilons(
            (self.0 & Epsilons::SLOT_MASK) | u64::from(look_set.to_repr()),
        )
    }

    /// A light wrapper around 'looks().matches()' that avoids the 'matches()'
    /// call when the look set is empty. In theory, if 'looks().matches()'
    /// would always get inlined, then this wouldn't be a problem. But since
    /// 'looks().matches()' is a public API, it isn't appropriate to tag it
    /// with 'inline(always)'. So we write this little hack instead.
    #[inline(always)]
    fn look_matches(self, haystack: &[u8], at: usize) -> bool {
        if self.looks().is_empty() {
            return true;
        }
        self.looks().matches(haystack, at)
    }
}

impl core::fmt::Debug for Epsilons {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        let mut wrote = false;
        if !self.slots().is_empty() {
            write!(f, "{:?}", self.slots())?;
            wrote = true;
        }
        if !self.looks().is_empty() {
            if wrote {
                write!(f, "/")?;
            }
            write!(f, "{:?}", self.looks())?;
            wrote = true;
        }
        if !wrote {
            write!(f, "N/A")?;
        }
        Ok(())
    }
}

/// The set of epsilon transitions indicating that the current position in a
/// search should be saved to a slot.
///
/// This *only* represents explicit slots. So for example, the pattern
/// `[a-z]+([0-9]+)([a-z]+)` has:
///
/// * 3 capturing groups, thus 6 slots.
/// * 1 implicit capturing group, thus 2 implicit slots.
/// * 2 explicit capturing groups, thus 4 explicit slots.
///
/// While implicit slots are represented by epsilon transitions in an NFA, we
/// do not explicitly represent them here. Instead, implicit slots are assumed
/// to be present and handled automatically in the search code. Therefore,
/// that means we only need to represent explicit slots in our epsilon
/// transitions.
///
/// Its representation is a bit set. The bit 'i' is set if and only if there
/// exists an explicit slot at index 'c', where 'c = (#patterns * 2) + i'. That
/// is, the bit 'i' corresponds to the first explicit slot and the first
/// explicit slot appears immediately following the last implicit slot. (If
/// this is confusing, see `GroupInfo` for more details on how slots works.)
///
/// A single `Slots` represents all the active slots in a sub-graph of an NFA,
/// where all the states are connected by epsilon transitions. In effect, when
/// traversing the one-pass DFA during a search, all slots set in a particular
/// transition must be captured by recording the current search position.
///
/// The API of `Slots` requires the caller to handle the explicit slot offset.
/// That is, a `Slots` doesn't know where the explicit slots start for a
/// particular NFA. Thus, if the callers see's the bit 'i' is set, then they
/// need to do the arithmetic above to find 'c', which is the real actual slot
/// index in the corresponding NFA.
#[derive(Clone, Copy)]
struct Slots(u32);

impl Slots {
    const LIMIT: usize = 32;

    /// Return an empty set of slots.
    fn empty() -> Slots {
        Slots(0)
    }

    /// Insert the slot at the given bit index.
    fn insert(self, slot: usize) -> Slots {
        debug_assert!(slot < Slots::LIMIT);
        Slots(self.0 | (1 << slot.as_u32()))
    }

    /// Remove the slot at the given bit index.
    fn remove(self, slot: usize) -> Slots {
        debug_assert!(slot < Slots::LIMIT);
        Slots(self.0 & !(1 << slot.as_u32()))
    }

    /// Returns true if and only if the given bit index is set.
    fn contains(self, slot: usize) -> bool {
        debug_assert!(slot < Slots::LIMIT);
        self.0 & (1 << slot.as_u32()) != 0
    }

    /// Returns true if and only if this set contains no slots.
    fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns the number of slots in this set.
    fn len(self) -> usize {
        self.0.count_ones().as_usize()
    }

    /// Returns an iterator over all of the set bits in this set.
    fn iter(self) -> SlotsIter {
        SlotsIter { slots: self }
    }

    /// For the position `at` in the current haystack, copy it to
    /// `caller_explicit_slots` for all slots that are in this set.
    ///
    /// Callers may pass a slice of any length. Slots in this set bigger than
    /// the length of the given explicit slots are simply skipped.
    ///
    /// The slice *must* correspond only to the explicit slots and the first
    /// element of the slice must always correspond to the first explicit slot
    /// in the corresponding NFA.
    fn apply(
        self,
        at: usize,
        caller_explicit_slots: &mut [Option<NonMaxUsize>],
    ) {
        if self.is_empty() {
            return;
        }
        let at = NonMaxUsize::new(at);
        for slot in self.iter() {
            if slot >= caller_explicit_slots.len() {
                break;
            }
            caller_explicit_slots[slot] = at;
        }
    }
}

impl core::fmt::Debug for Slots {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "S")?;
        for slot in self.iter() {
            write!(f, "-{:?}", slot)?;
        }
        Ok(())
    }
}

/// An iterator over all of the bits set in a slot set.
///
/// This returns the bit index that is set, so callers may need to offset it
/// to get the actual NFA slot index.
#[derive(Debug)]
struct SlotsIter {
    slots: Slots,
}

impl Iterator for SlotsIter {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        // Number of zeroes here is always <= u8::MAX, and so fits in a usize.
        let slot = self.slots.0.trailing_zeros().as_usize();
        if slot >= Slots::LIMIT {
            return None;
        }
        self.slots = self.slots.remove(slot);
        Some(slot)
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;

    use super::*;

    #[test]
    fn fail_conflicting_transition() {
        let predicate = |err: &str| err.contains("conflicting transition");

        let err = DFA::new(r"a*[ab]").unwrap_err().to_string();
        assert!(predicate(&err), "{}", err);
    }

    #[test]
    fn fail_multiple_epsilon() {
        let predicate = |err: &str| {
            err.contains("multiple epsilon transitions to same state")
        };

        let err = DFA::new(r"(^|$)a").unwrap_err().to_string();
        assert!(predicate(&err), "{}", err);
    }

    #[test]
    fn fail_multiple_match() {
        let predicate = |err: &str| {
            err.contains("multiple epsilon transitions to match state")
        };

        let err = DFA::new_many(&[r"^", r"$"]).unwrap_err().to_string();
        assert!(predicate(&err), "{}", err);
    }

    // This test is meant to build a one-pass regex with the maximum number of
    // possible slots.
    //
    // NOTE: Remember that the slot limit only applies to explicit capturing
    // groups. Any number of implicit capturing groups is supported (up to the
    // maximum number of supported patterns), since implicit groups are handled
    // by the search loop itself.
    #[test]
    fn max_slots() {
        // One too many...
        let pat = r"(a)(b)(c)(d)(e)(f)(g)(h)(i)(j)(k)(l)(m)(n)(o)(p)(q)";
        assert!(DFA::new(pat).is_err());
        // Just right.
        let pat = r"(a)(b)(c)(d)(e)(f)(g)(h)(i)(j)(k)(l)(m)(n)(o)(p)";
        assert!(DFA::new(pat).is_ok());
    }

    #[test]
    fn is_one_pass() {
        use crate::util::syntax;

        assert!(DFA::new(r"a*b").is_ok());
        assert!(DFA::new(r"\w").is_ok());
        assert!(DFA::new(r"(?-u)\w*\s").is_ok());
        assert!(DFA::new(r"(?s:.)*?").is_ok());
        assert!(DFA::builder()
            .syntax(syntax::Config::new().utf8(false))
            .build(r"(?s-u:.)*?")
            .is_ok());
    }

    #[test]
    fn is_not_one_pass() {
        assert!(DFA::new(r"a*a").is_err());
        assert!(DFA::new(r"\w*\s").is_err());
        assert!(DFA::new(r"(?s-u:.)*?").is_err());
        assert!(DFA::new(r"(?s:.)*?a").is_err());
    }
}
