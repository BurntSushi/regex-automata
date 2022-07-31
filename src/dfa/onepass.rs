#![allow(warnings)]

// BREADCRUMBS:
//
// Finish documenting module. Including internal docs.
//
// Clean up search routine... Benchmark carefully.
//
// Write down future perf notes: 1) more state shuffling, e.g., separate
// states with captures/looks, 2) maybe move match states to beginning, 3) see
// if we can arrange search loop to have one 'if sid < specialmax' like other
// DFAs, 4) move 'PatternInfo' out of transition table, since it's only set
// for match states.

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
        search::{Input, Match, MatchError, MatchKind},
        sparse_set::SparseSet,
    },
};

#[derive(Clone, Debug, Default)]
pub struct Config {
    match_kind: Option<MatchKind>,
    utf8: Option<bool>,
    starts_for_each_pattern: Option<bool>,
    byte_classes: Option<bool>,
    size_limit: Option<Option<usize>>,
}

impl Config {
    pub fn new() -> Config {
        Config::default()
    }

    pub fn match_kind(mut self, kind: MatchKind) -> Config {
        self.match_kind = Some(kind);
        self
    }

    pub fn utf8(mut self, yes: bool) -> Config {
        self.utf8 = Some(yes);
        self
    }

    pub fn starts_for_each_pattern(mut self, yes: bool) -> Config {
        self.starts_for_each_pattern = Some(yes);
        self
    }

    pub fn byte_classes(mut self, yes: bool) -> Config {
        self.byte_classes = Some(yes);
        self
    }

    pub fn size_limit(mut self, limit: Option<usize>) -> Config {
        self.size_limit = Some(limit);
        self
    }

    pub fn get_match_kind(&self) -> MatchKind {
        self.match_kind.unwrap_or(MatchKind::LeftmostFirst)
    }

    pub fn get_utf8(&self) -> bool {
        self.utf8.unwrap_or(true)
    }

    pub fn get_starts_for_each_pattern(&self) -> bool {
        self.starts_for_each_pattern.unwrap_or(false)
    }

    pub fn get_byte_classes(&self) -> bool {
        self.byte_classes.unwrap_or(true)
    }

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

#[derive(Clone, Debug)]
pub struct Builder {
    config: Config,
    thompson: thompson::Compiler,
}

impl Builder {
    pub fn new() -> Builder {
        Builder {
            config: Config::default(),
            thompson: thompson::Compiler::new(),
        }
    }

    pub fn build(&self, pattern: &str) -> Result<DFA, Error> {
        self.build_many(&[pattern])
    }

    pub fn build_many<P: AsRef<str>>(
        &self,
        patterns: &[P],
    ) -> Result<DFA, Error> {
        let nfa = self.thompson.build_many(patterns).map_err(Error::nfa)?;
        self.build_from_nfa(nfa)
    }

    pub fn build_from_nfa(&self, nfa: NFA) -> Result<DFA, Error> {
        InternalBuilder::new(self.config.clone(), &nfa).build()
    }

    pub fn configure(&mut self, config: Config) -> &mut Builder {
        self.config = self.config.overwrite(config);
        self
    }

    pub fn syntax(
        &mut self,
        config: crate::util::syntax::SyntaxConfig,
    ) -> &mut Builder {
        self.thompson.syntax(config);
        self
    }

    pub fn thompson(&mut self, config: thompson::Config) -> &mut Builder {
        self.thompson.configure(config);
        self
    }
}

#[derive(Debug)]
struct InternalBuilder<'a> {
    config: Config,
    nfa: &'a NFA,
    dfa: DFA,
    classes: ByteClasses,
    nfa_to_dfa_id: Vec<StateID>,
    uncompiled_nfa_ids: Vec<StateID>,
    seen: SparseSet,
    stack: Vec<(StateID, Info)>,
    matched: bool,
}

impl<'a> InternalBuilder<'a> {
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
            min_match_id: DEAD,
            classes: classes.clone(),
            alphabet_len,
            stride2,
            patinfo_offset: classes.alphabet_len() - 1,
        };
        InternalBuilder {
            config,
            nfa,
            dfa,
            classes,
            nfa_to_dfa_id: vec![DEAD; nfa.states().len()],
            uncompiled_nfa_ids: vec![],
            seen: SparseSet::new(nfa.states().len()),
            stack: vec![],
            matched: false,
        }
    }

    fn build(mut self) -> Result<DFA, Error> {
        if self.nfa.pattern_len().as_u64() > PatternInfo::PATTERN_ID_LIMIT {
            return Err(Error::one_pass_too_many_patterns(
                PatternInfo::PATTERN_ID_LIMIT,
            ));
        }
        if self.nfa.group_info().explicit_slot_len() > Slots::LIMIT {
            return Err(Error::one_pass_fail(
                "too many explicit capturing groups (max is 24)",
            ));
        }
        assert_eq!(DEAD, self.add_empty_state()?);

        let explicit_slot_start = self.nfa.pattern_len() * 2;
        self.add_start_state(self.nfa.start_anchored())?;
        if self.config.get_starts_for_each_pattern() {
            for pid in self.nfa.patterns() {
                self.add_start_state(self.nfa.start_pattern(pid))?;
            }
        }
        while let Some(nfa_id) = self.uncompiled_nfa_ids.pop() {
            let dfa_id = self.nfa_to_dfa_id[nfa_id];
            self.matched = false;
            self.seen.clear();
            self.stack_push(nfa_id, Info::empty())?;
            while let Some((id, info)) = self.stack.pop() {
                match *self.nfa.state(id) {
                    thompson::State::ByteRange { ref trans } => {
                        self.compile_transition(dfa_id, trans, info)?;
                    }
                    thompson::State::Sparse(ref sparse) => {
                        for trans in sparse.transitions.iter() {
                            self.compile_transition(dfa_id, trans, info)?;
                        }
                    }
                    thompson::State::Look { look, next } => {
                        let looks = info.looks().insert(look);
                        self.stack_push(next, info.set_looks(looks))?;
                    }
                    thompson::State::Union { ref alternates } => {
                        for &sid in alternates.iter().rev() {
                            self.stack_push(sid, info)?;
                        }
                    }
                    thompson::State::BinaryUnion { alt1, alt2 } => {
                        self.stack_push(alt2, info)?;
                        self.stack_push(alt1, info)?;
                    }
                    thompson::State::Capture { next, slot, .. } => {
                        let slot = slot.as_usize();
                        let info = if slot < explicit_slot_start {
                            info
                        } else {
                            let offset = slot - explicit_slot_start;
                            info.set_slots(info.slots().insert(offset))
                        };
                        self.stack_push(next, info)?;
                    }
                    thompson::State::Fail => {
                        continue;
                    }
                    thompson::State::Match { pattern_id } => {
                        if self.matched {
                            return Err(Error::one_pass_fail(
                                "multiple epsilon transitions to match state",
                            ));
                        }
                        self.matched = true;
                        self.dfa.set_pattern_info(
                            dfa_id,
                            PatternInfo::empty()
                                .set_pattern_id(pattern_id)
                                .set_info(info),
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
        self.dfa.min_match_id = StateID::new_unchecked(self.dfa.table.len());
        let mut match_ids = vec![];
        for i in 0..self.dfa.state_len() {
            let id = StateID::new_unchecked(i << self.dfa.stride2());
            let is_match = self.dfa.pattern_info(id).pattern_id().is_some();
            if is_match {
                match_ids.push(id);
            }
        }
        let mut remapper = Remapper::new(&self.dfa);
        let mut next_dest =
            StateID::new_unchecked(self.dfa.table.len() - self.dfa.stride());
        for id in match_ids.into_iter().rev() {
            remapper.swap(&mut self.dfa, next_dest, id);
            self.dfa.min_match_id = next_dest;
            next_dest = StateID::new_unchecked(
                next_dest.as_usize() - self.dfa.stride(),
            );
        }
        remapper.remap(&mut self.dfa);
        Ok(self.dfa)
    }

    fn stack_push(
        &mut self,
        nfa_id: StateID,
        info: Info,
    ) -> Result<(), Error> {
        if !self.seen.insert(nfa_id) {
            return Err(Error::one_pass_fail(
                "multiple epsilon transitions to same state",
            ));
        }
        self.stack.push((nfa_id, info));
        Ok(())
    }

    fn compile_transition(
        &mut self,
        dfa_id: StateID,
        trans: &thompson::Transition,
        info: Info,
    ) -> Result<(), Error> {
        // If we already have seen a match and we are compiling a leftmost
        // first DFA, then we shouldn't add any more transitions. This is
        // effectively how preference order and non-greediness is implemented.
        if !self.config.get_match_kind().continue_past_first_match()
            && self.matched
        {
            return Ok(());
        }
        let mut next_dfa_id = self.nfa_to_dfa_id[trans.next];
        if next_dfa_id == DEAD {
            next_dfa_id = self.add_dfa_state_for_nfa_state(trans.next)?;
        }
        for byte in self
            .classes
            .representatives(trans.start..=trans.end)
            .filter_map(|r| r.as_u8())
        {
            let oldtrans = self.dfa.transition(dfa_id, byte);
            let newtrans = Transition::new(next_dfa_id, info);
            if oldtrans.state_id() == DEAD {
                self.dfa.set_transition(dfa_id, byte, newtrans);
            } else if oldtrans != newtrans {
                return Err(Error::one_pass_fail("conflicting transition"));
            }
        }
        Ok(())
    }

    fn add_start_state(&mut self, nfa_id: StateID) -> Result<StateID, Error> {
        let dfa_id = self.add_dfa_state_for_nfa_state(nfa_id)?;
        self.dfa.starts.push(dfa_id);
        Ok(dfa_id)
    }

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

    fn add_empty_state(&mut self) -> Result<StateID, Error> {
        let next = self.dfa.table.len();
        let id = StateID::new(next).map_err(|_| Error::too_many_states())?;
        let id_limit = 1 << Transition::STATE_ID_BITS;
        if id.as_u64() > id_limit {
            let state_limit = id_limit / self.dfa.stride().as_u64();
            return Err(Error::one_pass_too_many_states(state_limit));
        }
        self.dfa
            .table
            .extend(core::iter::repeat(Transition(0)).take(self.dfa.stride()));
        // The default empty value for 'PatternInfo' is sadly not all zeroes.
        // Instead, the special sentinel u32::MAX<<32 is used to indicate that
        // there is no pattern. So we need to explicitly set the pattern info
        // to the correct "empty" state.
        self.dfa.set_pattern_info(id, PatternInfo::empty());
        if let Some(size_limit) = self.config.get_size_limit() {
            if self.dfa.memory_usage() > size_limit {
                return Err(Error::one_pass_exceeded_size_limit(size_limit));
            }
        }
        Ok(id)
    }
}

#[derive(Clone)]
pub struct DFA {
    config: Config,
    nfa: NFA,
    table: Vec<Transition>,
    starts: Vec<StateID>,
    /// Every state ID >= this value corresponds to a match state.
    min_match_id: StateID,
    classes: ByteClasses,
    alphabet_len: usize,
    stride2: usize,
    patinfo_offset: usize,
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
    ///     Match, SyntaxConfig,
    /// };
    ///
    /// let vm = DFA::builder()
    ///     .configure(DFA::config().utf8(false))
    ///     .syntax(SyntaxConfig::new().utf8(false))
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
    /// DFA.
    ///
    /// This routine is useful when using the lower-level [`DFA::search`] API.
    #[inline]
    pub fn create_input<'h, 'p, H: ?Sized + AsRef<[u8]>>(
        &'p self,
        haystack: &'h H,
    ) -> Input<'h, 'p> {
        let c = self.get_config();
        Input::new(haystack.as_ref()).utf8(c.get_utf8())
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
        self.table.len() >> self.stride2
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
        1 << self.stride2
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
    /// use regex_automata::{dfa::onepass::DFA, Input, Match, PatternID};
    ///
    /// let re = DFA::builder()
    ///     .configure(DFA::config().starts_for_each_pattern(true))
    ///     .build_many(&["[a-z]+", "[0-9]+"])?;
    /// let (mut cache, mut caps) = (re.create_cache(), re.create_captures());
    /// let haystack = "123abc";
    ///
    /// // A normal multi-pattern search will show pattern 1 matches.
    /// re.search(&mut cache, &Input::new(haystack), &mut caps);
    /// assert_eq!(Some(Match::must(1, 0..3)), caps.get_match());
    ///
    /// // If we only want to report pattern 0 matches, then we'll get no
    /// // match here.
    /// re.search(
    ///     &mut cache,
    ///     &Input::new(haystack).pattern(Some(PatternID::must(0))),
    ///     &mut caps,
    /// );
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
    /// use regex_automata::{dfa::onepass::DFA, Match, Input};
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
    /// re.search(&mut cache, &Input::new(&haystack[3..6]), &mut caps);
    /// assert_eq!(expected, caps.get_match());
    ///
    /// // But if we provide the bounds of the search within the context of the
    /// // entire haystack, then the search can take the surrounding context
    /// // into account. (And if we did find a match, it would be reported
    /// // as a valid offset into `haystack` instead of its sub-slice.)
    /// let expected = None;
    /// re.search(&mut cache, &Input::new(haystack).range(3..6), &mut caps);
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
        caps.set_pattern(None);
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
    /// use regex_automata::{dfa::onepass::DFA, PatternID, Input};
    ///
    /// let re = DFA::new_many(&[
    ///     r"[a-zA-Z]+",
    ///     r"[0-9]+",
    /// ])?;
    /// let mut cache = re.create_cache();
    /// let input = Input::new("123");
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
        let m = match self.search_imp(cache, input, slots) {
            None => return None,
            Some(pid) if !input.get_utf8() => return Some(pid),
            Some(pid) => {
                let slot_start = pid.as_usize() * 2;
                let slot_end = slot_start + 1;
                if slot_end >= slots.len() {
                    return Some(pid);
                }
                // These unwraps are OK because we know we have a match and
                // we know our caller provided slots are big enough.
                let start = slots[slot_start].unwrap().get();
                let end = slots[slot_end].unwrap().get();
                if start < end {
                    return Some(pid);
                }
                Match::new(pid, start..end)
            }
        };
        input
            .skip_empty_utf8_splits(m, |search| {
                let pid = match self.search_imp(cache, search, slots) {
                    None => return Ok(None),
                    Some(pid) => pid,
                };
                let slot_start = pid.as_usize() * 2;
                let slot_end = slot_start + 1;
                let start = slots[slot_start].unwrap().get();
                let end = slots[slot_end].unwrap().get();
                Ok(Some(Match::new(pid, start..end)))
            })
            .unwrap()
            .map(|m| m.pattern())
    }
}

impl DFA {
    fn search_imp(
        &self,
        cache: &mut Cache,
        input: &Input<'_, '_>,
        slots: &mut [Option<NonMaxUsize>],
    ) -> Option<PatternID> {
        let explicit_start = self.nfa.pattern_len() * 2;
        let explicit_slots = core::cmp::min(
            Slots::LIMIT,
            slots.len().saturating_sub(explicit_start),
        );
        cache.setup_search(explicit_slots);
        for slot in slots.iter_mut() {
            *slot = None;
        }
        for slot in cache.explicit_slots() {
            *slot = None;
        }
        if input.is_done() {
            return None;
        }
        for pid in self.nfa.patterns() {
            let i = pid.as_usize() * 2;
            if i >= slots.len() {
                break;
            }
            slots[i] = NonMaxUsize::new(input.start());
        }
        let haystack = input.haystack();
        let mut pid = None;
        let mut sid = match input.get_pattern() {
            None => self.start(),
            Some(pid) => self.start_pattern(pid),
        };
        for at in input.start()..input.end() {
            if sid >= self.min_match_id {
                let patinfo = self.pattern_info(sid);
                let foundpid = patinfo.pattern_id_unchecked();
                if patinfo.info().look_matches(haystack, at) {
                    pid = Some(foundpid);
                    let i = foundpid.as_usize() * 2 + 1;
                    if i < slots.len() {
                        slots[i] = NonMaxUsize::new(at);
                    }
                    if explicit_start < slots.len() {
                        slots[explicit_start..]
                            .copy_from_slice(cache.explicit_slots());
                        patinfo
                            .info()
                            .slots()
                            .apply(at, &mut slots[explicit_start..]);
                    }
                    if input.get_earliest() {
                        return pid;
                    }
                }
            }

            let trans = self.transition(sid, haystack[at]);
            sid = trans.state_id();
            let info = trans.info();
            if sid == DEAD || !info.look_matches(haystack, at) {
                return pid;
            }
            info.slots().apply(at, cache.explicit_slots());
        }
        if sid >= self.min_match_id {
            let patinfo = self.pattern_info(sid);
            let foundpid = patinfo.pattern_id_unchecked();
            if patinfo.info().look_matches(haystack, input.end()) {
                pid = Some(foundpid);
                let i = foundpid.as_usize() * 2 + 1;
                if i < slots.len() {
                    slots[i] = NonMaxUsize::new(input.end());
                }
                if explicit_start < slots.len() {
                    slots[explicit_start..]
                        .copy_from_slice(cache.explicit_slots());
                    patinfo
                        .info()
                        .slots()
                        .apply(input.end(), &mut slots[explicit_start..]);
                }
            }
        }
        pid
    }
}

impl DFA {
    fn start(&self) -> StateID {
        self.starts[0]
    }

    fn start_pattern(&self, pid: PatternID) -> StateID {
        assert!(pid.as_usize() < self.pattern_len(), "invalid pattern ID");
        self.starts.get(pid.one_more()).copied().expect(
            "starts_for_each_pattern must be enabled to \
             search by a specific pattern",
        )
    }

    fn transition(&self, sid: StateID, byte: u8) -> Transition {
        let class = self.classes.get(byte);
        self.table[sid.as_usize() + class.as_usize()]
    }

    fn set_transition(&mut self, sid: StateID, byte: u8, to: Transition) {
        let class = self.classes.get(byte);
        self.table[sid.as_usize() + class.as_usize()] = to;
    }

    fn sparse_transitions(&self, sid: StateID) -> SparseTransitionIter<'_> {
        let start = sid.as_usize();
        let end = start + self.alphabet_len();
        SparseTransitionIter {
            it: self.table[start..end].iter().enumerate(),
            cur: None,
        }
    }

    fn pattern_info(&self, sid: StateID) -> PatternInfo {
        PatternInfo(self.table[sid.as_usize() + self.patinfo_offset].0)
    }

    fn set_pattern_info(&mut self, sid: StateID, patinfo: PatternInfo) {
        self.table[sid.as_usize() + self.patinfo_offset] =
            Transition(patinfo.0);
    }

    pub(crate) fn swap_states(&mut self, id1: StateID, id2: StateID) {
        for b in 0..self.stride() {
            self.table.swap(id1.as_usize() + b, id2.as_usize() + b);
        }
    }

    pub(crate) fn remap(&mut self, map: impl Fn(StateID) -> StateID) {
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
                if !trans.info().is_empty() {
                    write!(f, " ({:?})", trans.info())?;
                }
            }
            Ok(())
        }

        writeln!(f, "onepass::DFA(")?;
        for index in 0..self.state_len() {
            let sid = StateID::must(index << self.stride2());
            let patinfo = self.pattern_info(sid);
            if sid == DEAD {
                write!(f, "D ")?;
            } else if patinfo.pattern_id().is_some() {
                write!(f, "* ")?;
            } else {
                write!(f, "  ")?;
            }
            write!(f, "{:06?}", debug_id(f, self, sid))?;
            if !patinfo.is_empty() {
                write!(f, " ({:?})", patinfo)?;
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

#[derive(Clone, Copy, Eq, PartialEq)]
struct Transition(u64);

impl Transition {
    const STATE_ID_BITS: u64 = 24;
    const STATE_ID_SHIFT: u64 = 64 - Transition::STATE_ID_BITS;
    const MASK_STATE_ID: u64 = 0xFFFFFF00_00000000;
    const MASK_INFO: u64 = 0x000000FF_FFFFFFFF;

    fn new(sid: StateID, info: Info) -> Transition {
        Transition((sid.as_u64() << Transition::STATE_ID_SHIFT) | info.0)
    }

    fn is_dead(self) -> bool {
        self.state_id() == DEAD
    }

    fn state_id(&self) -> StateID {
        // OK because a Transition has a valid StateID in its upper 32 bits
        // by construction. The cast to usize is also correct, even on 16-bit
        // targets because, again, we know the upper 32 bits is a valid
        // StateID, which can never overflow usize on any supported target.
        StateID::new_unchecked(
            (self.0 >> Transition::STATE_ID_SHIFT).as_usize(),
        )
    }

    fn set_state_id(&mut self, sid: StateID) {
        *self = Transition::new(sid, self.info());
    }

    fn info(&self) -> Info {
        Info(self.0 & Transition::MASK_INFO)
    }
}

impl core::fmt::Debug for Transition {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        if self.is_dead() {
            return write!(f, "0");
        }
        write!(f, "{}", self.state_id().as_usize())?;
        if !self.info().is_empty() {
            write!(f, "-{:?}", self.info())?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct PatternInfo(u64);

impl PatternInfo {
    const PATTERN_ID_BITS: u64 = 24;
    const PATTERN_ID_SHIFT: u64 = 64 - PatternInfo::PATTERN_ID_BITS;
    const PATTERN_ID_LIMIT: u64 = (1 << PatternInfo::PATTERN_ID_BITS) - 1;
    const MASK_PATTERN_ID: u64 = 0xFFFFFF00_00000000;
    const MASK_INFO: u64 = 0x000000FF_FFFFFFFF;

    fn empty() -> PatternInfo {
        PatternInfo(
            PatternInfo::PATTERN_ID_LIMIT << PatternInfo::PATTERN_ID_SHIFT,
        )
    }

    fn is_empty(self) -> bool {
        self.pattern_id().is_none() && self.info().is_empty()
    }

    fn pattern_id(self) -> Option<PatternID> {
        let pid = self.0 >> PatternInfo::PATTERN_ID_SHIFT;
        if pid == PatternInfo::PATTERN_ID_LIMIT {
            None
        } else {
            Some(PatternID::new_unchecked(pid.as_usize()))
        }
    }

    /// Returns the pattern ID without checking whether it's valid. If this is
    /// called and there is no pattern ID in this `PatternInfo`, then this
    /// will likely produce an incorrect result or possibly even a panic or
    /// an overflow. But safety will not be violated.
    fn pattern_id_unchecked(self) -> PatternID {
        let pid = self.0 >> PatternInfo::PATTERN_ID_SHIFT;
        PatternID::new_unchecked(pid.as_usize())
    }

    fn set_pattern_id(self, pid: PatternID) -> PatternInfo {
        PatternInfo(
            (pid.as_u64() << PatternInfo::PATTERN_ID_SHIFT)
                | (self.0 & PatternInfo::MASK_INFO),
        )
    }

    fn info(self) -> Info {
        Info(self.0 & PatternInfo::MASK_INFO)
    }

    fn set_info(self, info: Info) -> PatternInfo {
        PatternInfo(
            (self.0 & PatternInfo::MASK_PATTERN_ID) | u64::from(info.0),
        )
    }
}

impl core::fmt::Debug for PatternInfo {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        if self.is_empty() {
            return write!(f, "N/A");
        }
        if let Some(pid) = self.pattern_id() {
            write!(f, "{}", pid.as_usize())?;
        }
        if !self.info().is_empty() {
            if self.pattern_id().is_some() {
                write!(f, "/")?;
            }
            write!(f, "{:?}", self.info())?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Info(u64);

impl Info {
    const SLOT_MASK: u64 = 0x000000FF_FFFFFF00;
    const SLOT_SHIFT: u64 = 8;
    const LOOK_MASK: u64 = 0x00000000_000000FF;

    fn empty() -> Info {
        Info(0)
    }

    fn is_empty(self) -> bool {
        self.0 == 0
    }

    fn slots(self) -> Slots {
        Slots((self.0 >> Info::SLOT_SHIFT).low_u32())
    }

    fn set_slots(self, slots: Slots) -> Info {
        Info(
            (u64::from(slots.0) << Info::SLOT_SHIFT)
                | (self.0 & Info::LOOK_MASK),
        )
    }

    fn looks(self) -> LookSet {
        LookSet::from_repr(self.0.low_u8())
    }

    fn set_looks(self, look_set: LookSet) -> Info {
        Info((self.0 & Info::SLOT_MASK) | u64::from(look_set.to_repr()))
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

impl core::fmt::Debug for Info {
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

#[derive(Clone, Copy)]
struct Slots(u32);

impl Slots {
    const LIMIT: usize = 32;

    fn empty() -> Slots {
        Slots(0)
    }

    fn insert(self, slot: usize) -> Slots {
        debug_assert!(slot < Slots::LIMIT);
        Slots(self.0 | (1 << slot.as_u32()))
    }

    fn remove(self, slot: usize) -> Slots {
        debug_assert!(slot < Slots::LIMIT);
        Slots(self.0 & !(1 << slot.as_u32()))
    }

    fn contains(self, slot: usize) -> bool {
        debug_assert!(slot < Slots::LIMIT);
        self.0 & (1 << slot.as_u32()) != 0
    }

    fn is_empty(self) -> bool {
        self.0 == 0
    }

    fn len(self) -> usize {
        self.0.count_ones().as_usize()
    }

    fn iter(self) -> SlotsIter {
        SlotsIter { slots: self }
    }

    fn apply(self, at: usize, caller_slots: &mut [Option<NonMaxUsize>]) {
        if self.is_empty() {
            return;
        }
        let at = NonMaxUsize::new(at);
        for slot in self.iter() {
            if slot >= caller_slots.len() {
                break;
            }
            caller_slots[slot] = at;
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
}
