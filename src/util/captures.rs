use alloc::sync::Arc;

use crate::util::{
    primitives::{
        NonMaxUsize, PatternID, PatternIDError, PatternIDIter, SmallIndex,
        SmallIndexError,
    },
    search::{Match, Span},
};

// BREADCRUMBS: Bring 'Captures' type in here, since we should now be able
// to couple it with 'GroupInfo' instead of 'thompson::NFA'. Indeed, all
// references to 'nfa' inside of the 'Captures' impl are just to call
// 'nfa.group_info()'. So the de-coupling is done. Now we just need to
// rejigger things.
//
// We should also continue to think about whether to offer search APIs
// that just take a `&mut [Option<NonMaxUsize>]` and also return a
// `Option<PatternID>`. It does mean you don't need a heap alloc for simple
// cases or when you statically know the number of slots you need based on the
// regex. And that API can be used to implement the higher level one that
// uses `&mut Captures`.

/// The span offsets of capturing groups after a match has been found.
///
/// This type represents the output of regex engines that can report the
/// offsets at which capturing groups matches or "submatches" occur. For
/// example, the [`PikeVM`](crate::nfa::thompson::pikevm::PikeVM). When a match
/// occurs, it will at minimum contain the [`PatternID`] of the pattern that
/// matched. Depending upon how it was constructed, it may also contain the
/// start/end offsets of the entire match of the pattern and the start/end
/// offsets of each capturing group that participated in the match.
///
/// Values of this type are always created for a specific [`GroupInfo`]. It is
/// unspecified behavior to use a `Captures` value in a search with any regex
/// engine that has a different `GroupInfo` than the one it was created with.
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
/// ask for _less_ information, then the execution of regex search may be able
/// to run more quickly.
///
/// # Notes
///
/// It is worth pointing out that this type is not coupled to any one specific
/// regex engine. Instead, its coupling is with [`GroupInfo`], which is the
/// thing that is responsible for mapping capturing groups to "slot" offsets.
/// Slot offsets are indices into a single sequence of memory at which offsets
/// are written by regex engines.
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
    group_info: GroupInfo,
    pid: Option<PatternID>,
    slots: Vec<Option<NonMaxUsize>>,
}

impl Captures {
    /// Create new storage for the offsets of all matching capturing groups.
    ///
    /// This routine provides the most information for matches---namely, the
    /// match spans of capturing groups---but also requires the regex search
    /// routines to do the most work.
    ///
    /// It is unspecified behavior to use the returned `Captures` value in a
    /// search with a `GroupInfo` other than the one that is provided to this
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
    ///     nfa::thompson::pikevm::PikeVM,
    ///     util::captures::Captures,
    ///     Span, Match,
    /// };
    ///
    /// let vm = PikeVM::new(
    ///     r"^(?:(?P<lower>[a-z]+)|(?P<upper>[A-Z]+))(?P<digits>[0-9]+)$",
    /// )?;
    /// let mut cache = vm.create_cache();
    /// let mut caps = Captures::new(vm.get_nfa().group_info().clone());
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
    pub fn new(group_info: GroupInfo) -> Captures {
        let slots = group_info.slot_len();
        Captures { group_info, pid: None, slots: vec![None; slots] }
    }

    /// Create new storage for only the full match spans of a pattern. This
    /// does not include any capturing group offsets.
    ///
    /// It is unspecified behavior to use the returned `Captures` value in a
    /// search with a `GroupInfo` other than the one that is provided to this
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
    ///     nfa::thompson::pikevm::PikeVM,
    ///     util::captures::Captures,
    ///     Match,
    /// };
    ///
    /// let vm = PikeVM::new(
    ///     r"^(?:(?P<lower>[a-z]+)|(?P<upper>[A-Z]+))(?P<digits>[0-9]+)$",
    /// )?;
    /// let mut cache = vm.create_cache();
    /// let mut caps = Captures::new_for_matches_only(
    ///     vm.get_nfa().group_info().clone(),
    /// );
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
    pub fn new_for_matches_only(group_info: GroupInfo) -> Captures {
        // This is OK because we know there are at least this many slots,
        // and GroupInfo construction guarantees that the number of slots fits
        // into a usize.
        let slots = group_info.pattern_len().checked_mul(2).unwrap();
        Captures { group_info, pid: None, slots: vec![None; slots] }
    }

    /// Create new storage for only tracking which pattern matched. No offsets
    /// are stored at all.
    ///
    /// It is unspecified behavior to use the returned `Captures` value in a
    /// search with a `GroupInfo` other than the one that is provided to this
    /// constructor.
    ///
    /// # Example
    ///
    /// This example shows that only the pattern that matched can be accessed
    /// from a `Captures` value created via this constructor.
    ///
    /// ```
    /// use regex_automata::{
    ///     nfa::thompson::pikevm::PikeVM,
    ///     util::captures::Captures,
    ///     PatternID,
    /// };
    ///
    /// let vm = PikeVM::new_many(&[r"[a-z]+", r"[A-Z]+"])?;
    /// let mut cache = vm.create_cache();
    /// let mut caps = Captures::empty(vm.get_nfa().group_info().clone());
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
    pub fn empty(group_info: GroupInfo) -> Captures {
        Captures { group_info, pid: None, slots: vec![] }
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
    /// use regex_automata::{
    ///     nfa::thompson::pikevm::PikeVM,
    ///     util::captures::Captures,
    /// };
    ///
    /// let vm = PikeVM::new(r"[a-z]+")?;
    /// let mut cache = vm.create_cache();
    /// let mut caps = Captures::empty(vm.get_nfa().group_info().clone());
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
    ///     nfa::thompson::pikevm::PikeVM,
    ///     util::captures::Captures,
    ///     PatternID,
    /// };
    ///
    /// let vm = PikeVM::new_many(&[r"[a-z]+", r"[A-Z]+"])?;
    /// let mut cache = vm.create_cache();
    /// let mut caps = Captures::empty(vm.get_nfa().group_info().clone());
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
    /// By convention, the 0th capture group will always return the same span
    /// as the span returned by `get_match`. This is because the 0th capture
    /// group always corresponds to the entirety of the pattern's match.
    /// (It is similarly always unnamed because it is implicit.) This isn't
    /// necessarily true of all regex engines. For example, one can hand-compile
    /// a [`thompson::NFA`](crate::nfa::thompson::NFA) via a
    /// [`thompson::Builder`](crate::nfa::thompson::Builder), which isn't
    /// technically forced to make the 0th capturing group always correspond to
    /// the entire match.
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
        // There's a little bit of work needed to map captures to slots in the
        // fully general case. But in the overwhelming common case of a single
        // pattern, we can just do some simple arithmetic.
        let (slot_start, slot_end) = if self.group_info().pattern_len() == 1 {
            (index * 2, index * 2 + 1)
        } else {
            self.group_info().slots(pid, index)?
        };
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
        let index = self.group_info().to_index(self.pattern()?, name)?;
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
            .map(|pid| self.group_info().pattern_names(pid).enumerate());
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
        self.group_info().group_len(pid)
    }

    /// Returns a reference to the underlying group info on which these
    /// captures are based.
    ///
    /// Note that a `GroupInfo` uses reference counting internally, so it may
    /// be cloned cheaply.
    pub fn group_info(&self) -> &GroupInfo {
        &self.group_info
    }
}

/// Lower level "slot" oriented APIs. One does not typically need to use these
/// when executing a search. They are instead mostly intended for folks that
/// are writing their own regex engine while reusing this `Captures` type.
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
    /// // Note that the following ordering is considered an API guarantee.
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
    /// // Note that the following ordering is considered an API guarantee.
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
    /// // Note that the following ordering is considered an API guarantee.
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
        let mut names = self.caps.group_info().pattern_names(self.pid);
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
    names: Option<core::iter::Enumerate<GroupInfoPatternNames<'a>>>,
}

impl<'a> Iterator for CapturesPatternIter<'a> {
    type Item = Option<Span>;

    fn next(&mut self) -> Option<Option<Span>> {
        let (group_index, _) = self.names.as_mut()?.next()?;
        Some(self.caps.get_group(group_index))
    }
}

/// Represents information about capturing groups in a compiled regex.
///
/// The information encapsulated by this type consists of the following. For
/// each pattern:
///
/// * A map from every capture group name to its corresponding capture group
/// index.
/// * A map from every capture group index to its corresponding capture group
/// name.
/// * A map from capture group index to its corresponding slot index. A slot
/// refers to one half of a capturing group. That is, a capture slot is either
/// the start or end of a capturing group. A slot is usually the mechanism
/// by which a regex engine records offsets for each capturing group during a
/// search.
#[derive(Clone, Debug, Default)]
pub struct GroupInfo(Arc<GroupInfoInner>);

impl GroupInfo {
    pub fn new<P, G, N>(pattern_groups: P) -> Result<GroupInfo, GroupInfoError>
    where
        P: IntoIterator<Item = G>,
        G: IntoIterator<Item = Option<N>>,
        N: AsRef<str>,
    {
        let mut group_info = GroupInfoInner {
            slot_ranges: vec![],
            name_to_index: vec![],
            index_to_name: vec![],
            memory_extra: 0,
        };
        for (pattern_index, groups) in pattern_groups.into_iter().enumerate() {
            // If we can't convert the pattern index to an ID, then the caller
            // tried to build capture info for too many patterns.
            let pid = PatternID::new(pattern_index)
                .map_err(GroupInfoError::too_many_patterns)?;

            let mut groups_iter = groups.into_iter().enumerate();
            match groups_iter.next() {
                None => return Err(GroupInfoError::missing_groups(pid)),
                Some((_, Some(_))) => {
                    return Err(GroupInfoError::first_must_be_unnamed(pid))
                }
                Some((_, None)) => {}
            }
            group_info.add_first_group(pid);
            // Now iterate over the rest, which correspond to all of the
            // (conventionally) explicit capture groups in a regex pattern.
            for (group_index, maybe_name) in groups_iter {
                // Just like for patterns, if the group index can't be
                // converted to a "small" index, then the caller has given too
                // many groups for a particular pattern.
                let group = SmallIndex::new(group_index).map_err(|_| {
                    GroupInfoError::too_many_groups(pid, group_index)
                })?;
                group_info.add_explicit_group(pid, group, maybe_name)?;
            }
        }
        group_info.fixup_slot_ranges()?;
        Ok(GroupInfo(Arc::new(group_info)))
    }

    /// Return the capture group index corresponding to the given name in the
    /// given pattern. If no such capture group name exists in the given
    /// pattern, then this returns `None`.
    ///
    /// If the given pattern ID is invalid, then this returns `None`.
    ///
    /// This also returns `None` for all inputs if these captures are empty
    /// (e.g., built from an empty [`GroupInfo`]). To check whether captures
    /// are are present for a specific pattern, use [`GroupInfo::group_len`].
    ///
    /// # Example
    ///
    /// This example shows how to find the capture index for the given pattern
    /// and group name.
    ///
    /// Remember that capture indices are relative to the pattern, such that
    /// the same capture index value may refer to different capturing groups
    /// for distinct patterns.
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::NFA, PatternID};
    ///
    /// let (pid0, pid1) = (PatternID::must(0), PatternID::must(1));
    ///
    /// let nfa = NFA::new_many(&[
    ///     r"a(?P<quux>\w+)z(?P<foo>\s+)",
    ///     r"a(?P<foo>\d+)z",
    /// ])?;
    /// let groups = nfa.group_info();
    /// assert_eq!(Some(2), groups.to_index(pid0, "foo"));
    /// // Recall that capture index 0 is always unnamed and refers to the
    /// // entire pattern. So the first capturing group present in the pattern
    /// // itself always starts at index 1.
    /// assert_eq!(Some(1), groups.to_index(pid1, "foo"));
    ///
    /// // And if a name does not exist for a particular pattern, None is
    /// // returned.
    /// assert!(groups.to_index(pid0, "quux").is_some());
    /// assert!(groups.to_index(pid1, "quux").is_none());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn to_index(&self, pid: PatternID, name: &str) -> Option<usize> {
        let indices = self.0.name_to_index.get(pid.as_usize())?;
        indices.get(name).cloned().map(|i| i.as_usize())
    }

    /// Return the capture name for the given index and given pattern. If the
    /// corresponding group does not have a name, then this returns `None`.
    ///
    /// If the pattern ID is invalid, then this returns `None`.
    ///
    /// If the group index is invalid for the given pattern, then this returns
    /// `None`. A group `index` is valid for a pattern `pid` in an `nfa` if and
    /// only if `index < nfa.pattern_capture_len(pid)`.
    ///
    /// This also returns `None` for all inputs if these captures are empty
    /// (e.g., built from an empty [`GroupInfo`]). To check whether captures
    /// are are present for a specific pattern, use [`GroupInfo::group_len`].
    ///
    /// # Example
    ///
    /// This example shows how to find the capture group name for the given
    /// pattern and group index.
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::NFA, PatternID};
    ///
    /// let (pid0, pid1) = (PatternID::must(0), PatternID::must(1));
    ///
    /// let nfa = NFA::new_many(&[
    ///     r"a(?P<foo>\w+)z(\s+)x(\d+)",
    ///     r"a(\d+)z(?P<foo>\s+)",
    /// ])?;
    /// let groups = nfa.group_info();
    /// assert_eq!(None, groups.to_name(pid0, 0));
    /// assert_eq!(Some("foo"), groups.to_name(pid0, 1));
    /// assert_eq!(None, groups.to_name(pid0, 2));
    /// assert_eq!(None, groups.to_name(pid0, 3));
    ///
    /// assert_eq!(None, groups.to_name(pid1, 0));
    /// assert_eq!(None, groups.to_name(pid1, 1));
    /// assert_eq!(Some("foo"), groups.to_name(pid1, 2));
    /// // '3' is not a valid capture index for the second pattern.
    /// assert_eq!(None, groups.to_name(pid1, 3));
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn to_name(&self, pid: PatternID, group_index: usize) -> Option<&str> {
        let pattern_names = self.0.index_to_name.get(pid.as_usize())?;
        pattern_names.get(group_index)?.as_deref()
    }

    /// Return an iterator of all capture groups and their names (if present)
    /// for a particular pattern.
    ///
    /// If the given pattern ID is invalid or if this `GroupInfo` is empty,
    /// then the iterator yields no elements.
    ///
    /// The number of elements yielded by this iterator is always equal to
    /// the result of calling [`GroupInfo::group_len`] with the same
    /// `PatternID`.
    ///
    /// # Example
    ///
    /// This example shows how to get a list of all capture group names for
    /// a particular pattern.
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::NFA, PatternID};
    ///
    /// let nfa = NFA::new(r"(a)(?P<foo>b)(c)(d)(?P<bar>e)")?;
    /// // The first is the implicit group that is always unnammed. The next
    /// // 5 groups are the explicit groups found in the concrete syntax above.
    /// let expected = vec![None, None, Some("foo"), None, None, Some("bar")];
    /// let got: Vec<Option<&str>> =
    ///     nfa.group_info().pattern_names(PatternID::ZERO).collect();
    /// assert_eq!(expected, got);
    ///
    /// // Using an invalid pattern ID will result in nothing yielded.
    /// let got = nfa.group_info().pattern_names(PatternID::must(999)).count();
    /// assert_eq!(0, got);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn pattern_names(&self, pid: PatternID) -> GroupInfoPatternNames<'_> {
        GroupInfoPatternNames {
            it: self
                .0
                .index_to_name
                .get(pid.as_usize())
                .map(|indices| indices.iter())
                .unwrap_or([].iter()),
        }
    }

    /// Return an iterator of all capture groups for all patterns supported by
    /// this `GroupInfo`. Each item yielded is a triple of the group's pattern
    /// ID, index in the pattern and the group's name, if present.
    ///
    /// # Example
    ///
    /// This example shows how to get a list of all capture groups found in
    /// one NFA, potentially spanning multiple patterns.
    ///
    /// ```
    /// use regex_automata::{nfa::thompson::NFA, PatternID};
    ///
    /// let nfa = NFA::new_many(&[
    ///     r"(?P<foo>a)",
    ///     r"a",
    ///     r"(a)",
    /// ])?;
    /// let expected = vec![
    ///     (PatternID::must(0), 0, None),
    ///     (PatternID::must(0), 1, Some("foo")),
    ///     (PatternID::must(1), 0, None),
    ///     (PatternID::must(2), 0, None),
    ///     (PatternID::must(2), 1, None),
    /// ];
    /// let got: Vec<(PatternID, usize, Option<&str>)> =
    ///     nfa.group_info().all_names().collect();
    /// assert_eq!(expected, got);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// Unlike other capturing group related routines, this routine doesn't
    /// panic even if captures aren't enabled on this NFA:
    ///
    /// ```
    /// use regex_automata::nfa::thompson::NFA;
    ///
    /// let nfa = NFA::compiler()
    ///     .configure(NFA::config().captures(false))
    ///     .build_many(&[
    ///         r"(?P<foo>a)",
    ///         r"a",
    ///         r"(a)",
    ///     ])?;
    /// // When captures aren't enabled, there's nothing to return.
    /// assert_eq!(0, nfa.group_info().all_names().count());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn all_names(&self) -> GroupInfoAllNames<'_> {
        GroupInfoAllNames {
            group_info: self,
            pids: PatternID::iter(self.pattern_len()),
            current_pid: None,
            names: None,
        }
    }

    #[inline]
    pub fn slots(
        &self,
        pid: PatternID,
        group_index: usize,
    ) -> Option<(usize, usize)> {
        // Since 'slot' only even returns valid starting slots, we know that
        // there must also be an end slot and that end slot is always one more
        // than the start slot.
        self.slot(pid, group_index).map(|start| (start, start + 1))
    }

    #[inline]
    pub fn slot(&self, pid: PatternID, group_index: usize) -> Option<usize> {
        if group_index >= self.group_len(pid) {
            return None;
        }
        // At this point, we know that 'pid' refers to a real pattern and that
        // 'group_index' refers to a real group. We therefore also know that
        // the pattern and group can be combined to return a correct slot.
        // That's why we don't need to use checked arithmetic below.
        if group_index == 0 {
            Some(pid.as_usize() * 2)
        } else {
            // As above, we don't need to check that our slot is less than the
            // end of our range since we already know the group index is a
            // valid index for the given pattern.
            let (start, _) = self.0.slot_ranges[pid];
            Some(start.as_usize() + ((group_index - 1) * 2))
        }
    }

    #[inline]
    pub fn pattern_len(&self) -> usize {
        self.0.pattern_len()
    }

    #[inline]
    pub fn group_len(&self, pid: PatternID) -> usize {
        self.0.group_len(pid)
    }

    #[inline]
    pub fn slot_len(&self) -> usize {
        self.0.small_slot_len().as_usize()
    }

    #[inline]
    pub fn memory_usage(&self) -> usize {
        use core::mem::size_of as s;

        s::<GroupInfoInner>()
            + self.0.slot_ranges.len() * s::<(SmallIndex, SmallIndex)>()
            + self.0.name_to_index.len() * s::<CaptureNameMap>()
            + self.0.index_to_name.len() * s::<Vec<Option<Arc<str>>>>()
            + self.0.memory_extra
    }
}

/// A map from capture group name to its corresponding capture group index.
///
/// This type is actually wrapped inside a Vec indexed by pattern ID on a
/// `GroupInfo`, since multiple patterns may have the same capture group name.
/// That is, each pattern gets its own namespace of capture group names.
///
/// Perhaps a more memory efficient representation would be
/// HashMap<(PatternID, Arc<str>), usize>, but this makes it difficult to look
/// up a capture index by name without producing a `Arc<str>`, which requires
/// an allocation. To fix this, I think we'd need to define our own unsized
/// type or something? Anyway, I didn't give this much thought since it
/// probably doesn't matter much in the grand scheme of things. But it did
/// stand out to me as mildly wasteful.
#[cfg(feature = "std")]
type CaptureNameMap = std::collections::HashMap<Arc<str>, SmallIndex>;
#[cfg(not(feature = "std"))]
type CaptureNameMap = alloc::collections::BTreeMap<Arc<str>, SmallIndex>;

/// The inner guts of `GroupInfo`. This type only exists so that it can
/// be wrapped in an `Arc` to make `GroupInfo` reference counted.
#[derive(Debug, Default)]
struct GroupInfoInner {
    slot_ranges: Vec<(SmallIndex, SmallIndex)>,
    name_to_index: Vec<CaptureNameMap>,
    index_to_name: Vec<Vec<Option<Arc<str>>>>,
    memory_extra: usize,
}

impl GroupInfoInner {
    /// This adds the first unnamed group for the given pattern ID. The given
    /// pattern ID must be zero if this is the first time this method is
    /// called, or must be exactly one more than the pattern ID supplied to the
    /// previous call to this method. (This method panics if this rule is
    /// violated.)
    ///
    /// This can be thought of as initializing the GroupInfo state for the
    /// given pattern and closing off the state for any previous pattern.
    fn add_first_group(&mut self, pid: PatternID) {
        assert_eq!(pid.as_usize(), self.slot_ranges.len());
        assert_eq!(pid.as_usize(), self.name_to_index.len());
        assert_eq!(pid.as_usize(), self.index_to_name.len());
        // This is the start of our slots for the explicit capturing groups.
        // Note that since the slots for the 0th group for every pattern appear
        // before any slots for the nth group (where n > 0) in any pattern, we
        // will have to fix up the slot ranges once we know how many patterns
        // we've added capture groups for.
        let slot_start = self.small_slot_len();
        self.slot_ranges.push((slot_start, slot_start));
        self.name_to_index.push(CaptureNameMap::new());
        self.index_to_name.push(vec![None]);
        self.memory_extra += core::mem::size_of::<Option<Arc<str>>>();
    }

    /// Add an explicit capturing group for the given pattern with the given
    /// index. If the group has a name, then that must be given as well.
    ///
    /// Note that every capturing group except for the first or zeroth group is
    /// explicit.
    ///
    /// This returns an error if adding this group would result in overflowing
    /// slot indices or if a capturing group with the same name for this
    /// pattern has already been added.
    fn add_explicit_group<N: AsRef<str>>(
        &mut self,
        pid: PatternID,
        group: SmallIndex,
        maybe_name: Option<N>,
    ) -> Result<(), GroupInfoError> {
        // We also need to check that the slot index generated for
        // this group is also valid. Although, this is a little weird
        // because we offset these indices below, at which point, we'll
        // have to recheck them. Gosh this is annoying. Note that
        // the '+2' below is OK because 'end' is guaranteed to be less
        // than isize::MAX.
        let end = &mut self.slot_ranges[pid].1;
        *end = SmallIndex::new(end.as_usize() + 2).map_err(|_| {
            GroupInfoError::too_many_groups(pid, group.as_usize())
        })?;
        if let Some(name) = maybe_name {
            let name = Arc::<str>::from(name.as_ref());
            if self.name_to_index[pid].contains_key(&*name) {
                return Err(GroupInfoError::duplicate(pid, &name));
            }
            let len = name.len();
            self.name_to_index[pid].insert(Arc::clone(&name), group);
            self.index_to_name[pid].push(Some(name));
            // Adds the memory used by the Arc<str> in both maps.
            self.memory_extra +=
                2 * (len + core::mem::size_of::<Option<Arc<str>>>());
            // And also the value entry for the 'name_to_index' map.
            // This is probably an underestimate for 'name_to_index' since
            // hashmaps/btrees likely have some non-zero overhead, but we
            // assume here that they have zero overhead.
            self.memory_extra += core::mem::size_of::<SmallIndex>();
        } else {
            self.index_to_name[pid].push(None);
            self.memory_extra += core::mem::size_of::<Option<Arc<str>>>();
        }
        // This is a sanity assert that checks that our group index
        // is in line with the number of groups added so far for this
        // pattern.
        assert_eq!(group.one_more(), self.group_len(pid));
        // And is also in line with the 'index_to_name' map.
        assert_eq!(group.one_more(), self.index_to_name[pid].len());
        Ok(())
    }

    /// This corrects the slot ranges to account for the slots corresponding
    /// to the zeroth group of each pattern. That is, every slot range is
    /// offset by 'pattern_len() * 2', since each pattern uses two slots to
    /// represent the zeroth group.
    fn fixup_slot_ranges(&mut self) -> Result<(), GroupInfoError> {
        use crate::util::primitives::IteratorIndexExt;
        // Since we know number of patterns fits in PatternID and
        // PatternID::MAX < isize::MAX, it follows that multiplying by 2 will
        // never overflow usize.
        let offset = self.pattern_len().checked_mul(2).unwrap();
        for (pid, &mut (ref mut start, ref mut end)) in
            self.slot_ranges.iter_mut().with_pattern_ids()
        {
            let group_len = 1 + ((end.as_usize() - start.as_usize()) / 2);
            let new_end = match end.as_usize().checked_add(offset) {
                Some(new_end) => new_end,
                None => {
                    return Err(GroupInfoError::too_many_groups(
                        pid, group_len,
                    ))
                }
            };
            *end = SmallIndex::new(new_end).map_err(|_| {
                GroupInfoError::too_many_groups(pid, group_len)
            })?;
            // Since start <= end, if end is valid then start must be too.
            *start = SmallIndex::new(start.as_usize() + offset).unwrap();
        }
        Ok(())
    }

    /// Return the total number of patterns represented by this capture slot
    /// info.
    fn pattern_len(&self) -> usize {
        self.slot_ranges.len()
    }

    /// Return the total number of capturing groups for the given pattern. If
    /// the given pattern isn't valid for this capture slot info, then 0 is
    /// returned.
    fn group_len(&self, pid: PatternID) -> usize {
        let (start, end) = match self.slot_ranges.get(pid.as_usize()) {
            None => return 0,
            Some(range) => range,
        };
        // The difference between any two SmallIndex values always fits in a
        // usize since we know that SmallIndex::MAX <= isize::MAX-1. We also
        // know that start<=end by construction and that the number of groups
        // never exceeds SmallIndex and thus never overflows usize.
        1 + ((end.as_usize() - start.as_usize()) / 2)
    }

    /// Return the total number of slots in this capture slot info as a
    /// "small index."
    fn small_slot_len(&self) -> SmallIndex {
        // Since slots are allocated in order of pattern (starting at 0) and
        // then in order of capture group, it follows that the number of slots
        // is the end of the range of slots for the last pattern. This is
        // true even when the last pattern has no capturing groups, since
        // 'slot_ranges' will still represent it explicitly with an empty
        // range.
        self.slot_ranges.last().map_or(SmallIndex::ZERO, |&(_, end)| end)
    }
}

/// An error that may occur when building a `GroupInfo`.
///
/// Building a `GroupInfo` does a variety of checks to make sure the
/// capturing groups satisfy a number of invariants. This includes, but is not
/// limited to, ensuring that the first capturing group is unnamed and that
/// there are no duplicate capture groups for a specific pattern.
#[derive(Clone, Debug)]
pub struct GroupInfoError {
    kind: GroupInfoErrorKind,
}

/// The kind of error that occurs when building a `GroupInfo` fails.
///
/// We keep this un-exported because it's not clear how useful it is to
/// export it.
#[derive(Clone, Debug)]
enum GroupInfoErrorKind {
    /// This occurs when too many patterns have been added. i.e., It would
    /// otherwise overflow a `PatternID`.
    TooManyPatterns { err: PatternIDError },
    /// This occurs when too many capturing groups have been added for a
    /// particular pattern.
    TooManyGroups {
        /// The ID of the pattern that had too many groups.
        pattern: PatternID,
        /// The minimum number of groups that the caller has tried to add for
        /// a pattern.
        minimum: usize,
    },
    /// An error that occurs when a pattern has no capture groups. Either
    /// all patterns must have no groups or all patterns must have at least
    /// one group (corresponding to the unnamed group for the entire pattern).
    MissingGroups {
        /// The ID of the pattern that had no capturing groups.
        pattern: PatternID,
    },
    /// An error that occurs when one tries to provide a name for the capture
    /// group at index 0. This capturing group must currently always be
    /// unnamed.
    FirstMustBeUnnamed {
        /// The ID of the pattern that was found to have a named first
        /// capturing group.
        pattern: PatternID,
    },
    /// An error that occurs when duplicate capture group names for the same
    /// pattern are added.
    ///
    /// NOTE: At time of writing, this error can never occur if you're using
    /// regex-syntax, since the parser itself will reject patterns with
    /// duplicate capture group names. This error can only occur when the
    /// builder is used to hand construct NFAs.
    Duplicate {
        /// The pattern in which the duplicate capture group name was found.
        pattern: PatternID,
        /// The duplicate name.
        name: String,
    },
}

impl GroupInfoError {
    fn too_many_patterns(err: PatternIDError) -> GroupInfoError {
        GroupInfoError { kind: GroupInfoErrorKind::TooManyPatterns { err } }
    }

    fn too_many_groups(pattern: PatternID, minimum: usize) -> GroupInfoError {
        GroupInfoError {
            kind: GroupInfoErrorKind::TooManyGroups { pattern, minimum },
        }
    }

    fn missing_groups(pattern: PatternID) -> GroupInfoError {
        GroupInfoError { kind: GroupInfoErrorKind::MissingGroups { pattern } }
    }

    fn first_must_be_unnamed(pattern: PatternID) -> GroupInfoError {
        GroupInfoError {
            kind: GroupInfoErrorKind::FirstMustBeUnnamed { pattern },
        }
    }

    fn duplicate(pattern: PatternID, name: &str) -> GroupInfoError {
        GroupInfoError {
            kind: GroupInfoErrorKind::Duplicate {
                pattern,
                name: name.to_string(),
            },
        }
    }
}

impl std::error::Error for GroupInfoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self.kind {
            GroupInfoErrorKind::TooManyPatterns { .. }
            | GroupInfoErrorKind::TooManyGroups { .. }
            | GroupInfoErrorKind::MissingGroups { .. }
            | GroupInfoErrorKind::FirstMustBeUnnamed { .. }
            | GroupInfoErrorKind::Duplicate { .. } => None,
        }
    }
}

impl core::fmt::Display for GroupInfoError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        use self::GroupInfoErrorKind::*;

        match self.kind {
            TooManyPatterns { ref err } => {
                write!(f, "too many patterns to build capture info: {}", err)
            }
            TooManyGroups { pattern, minimum } => {
                write!(
                    f,
                    "too many capture groups (at least {}) were \
                     found for pattern {}",
                    minimum,
                    pattern.as_usize()
                )
            }
            MissingGroups { pattern } => write!(
                f,
                "no capturing groups found for pattern {} \
                 (either all patterns have zero groups or all patterns have \
                  at least one group)",
                pattern.as_usize(),
            ),
            FirstMustBeUnnamed { pattern } => write!(
                f,
                "first capture group (at index 0) for pattern {} has a name \
                 (it must be unnamed)",
                pattern.as_usize(),
            ),
            Duplicate { pattern, ref name } => write!(
                f,
                "duplicate capture group name '{}' found for pattern {}",
                name,
                pattern.as_usize(),
            ),
        }
    }
}

/// An iterator over capturing groups and their names for a specific pattern.
///
/// This iterator is created by [`GroupInfo::pattern_names`].
///
/// The lifetime parameter `'a` refers to the lifetime of the `GroupInfo`
/// from which this iterator was created.
#[derive(Debug)]
pub struct GroupInfoPatternNames<'a> {
    it: core::slice::Iter<'a, Option<Arc<str>>>,
}

impl<'a> Iterator for GroupInfoPatternNames<'a> {
    type Item = Option<&'a str>;

    fn next(&mut self) -> Option<Option<&'a str>> {
        self.it.next().map(|x| x.as_deref())
    }
}

/// An iterator over capturing groups and their names for a `GroupInfo`.
///
/// This iterator is created by [`GroupInfo::all_names`].
///
/// The lifetime parameter `'a` refers to the lifetime of the `GroupInfo`
/// from which this iterator was created.
#[derive(Debug)]
pub struct GroupInfoAllNames<'a> {
    group_info: &'a GroupInfo,
    pids: PatternIDIter,
    current_pid: Option<PatternID>,
    names: Option<core::iter::Enumerate<GroupInfoPatternNames<'a>>>,
}

impl<'a> Iterator for GroupInfoAllNames<'a> {
    type Item = (PatternID, usize, Option<&'a str>);

    fn next(&mut self) -> Option<(PatternID, usize, Option<&'a str>)> {
        // If the group info has no captures, then we never have anything
        // to yield. We need to consider this case explicitly (at time of
        // writing) because 'pattern_capture_names' will panic if captures
        // aren't enabled.
        if self.group_info.0.index_to_name.is_empty() {
            return None;
        }
        if self.current_pid.is_none() {
            self.current_pid = Some(self.pids.next()?);
        }
        let pid = self.current_pid.unwrap();
        if self.names.is_none() {
            self.names = Some(self.group_info.pattern_names(pid).enumerate());
        }
        let (group_index, name) = match self.names.as_mut().unwrap().next() {
            Some((group_index, name)) => (group_index, name),
            None => {
                self.current_pid = None;
                self.names = None;
                return self.next();
            }
        };
        Some((pid, group_index, name))
    }
}
