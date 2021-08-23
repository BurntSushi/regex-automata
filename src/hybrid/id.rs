// BREADCRUMBS: This is quite tricky, particularly with respect to exposing it
// in the public API. We could just expose it as-is, but then computing the
// next state would ALWAYS have to mask out the match/start bits, which is
// an unnecessary expense in some cases. So then we're forced to either find
// another way to determine the type of a state, or to expose a more complex
// API that uses both StateID and LazyStateID. The latter case means we have
// no hope of implementing Automaton, but the former case almost certainly
// leaves perf on the table. Perf is everything, so we'll probably forgo the
// Automaton trait... Sigh.
#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, PartialEq, PartialOrd, Ord,
)]
pub struct LazyStateID(u32);

impl LazyStateID {
    #[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
    const MAX_BIT: usize = 31;

    #[cfg(target_pointer_width = "16")]
    const MAX_BIT: usize = 15;

    const MASK_UNKNOWN: usize = 1 << (LazyStateID::MAX_BIT);
    const MASK_DEAD: usize = 1 << (LazyStateID::MAX_BIT - 1);
    const MASK_QUIT: usize = 1 << (LazyStateID::MAX_BIT - 2);
    const MASK_START: usize = 1 << (LazyStateID::MAX_BIT - 3);
    const MASK_MATCH: usize = 1 << (LazyStateID::MAX_BIT - 4);
    const MAX: usize = LazyStateID::MASK_MATCH - 1;

    /// Create a new lazy state ID.
    ///
    /// If the given identifier exceeds [`LazyStateID::MAX`], then this returns
    /// an error.
    #[inline]
    pub(crate) fn new(id: usize) -> Result<LazyStateID, LazyStateIDError> {
        if id > LazyStateID::MAX {
            return Err(LazyStateIDError { attempted: id as u64 });
        }
        Ok(LazyStateID::new_unchecked(id))
    }

    /// Create a new lazy state ID without checking whether the given value
    /// exceeds [`LazyStateID::MAX`].
    ///
    /// While this is unchecked, providing an incorrect value must never
    /// sacrifice memory safety.
    #[inline]
    const fn new_unchecked(id: usize) -> LazyStateID {
        LazyStateID(id as u32)
    }

    /// Return this lazy state ID as its raw value if and only if it is not
    /// tagged (and thus not an unknown, dead, quit, start or match state ID).
    #[inline]
    pub(crate) fn as_usize(&self) -> Option<usize> {
        if self.is_tagged() {
            None
        } else {
            Some(self.as_usize_unchecked())
        }
    }

    /// Return this lazy state ID as an untagged `usize`.
    ///
    /// If this lazy state ID is tagged, then the usize returned is the state
    /// ID without the tag. If the ID was not tagged, then the usize returned
    /// is equivalent to the state ID.
    #[inline]
    pub(crate) fn as_usize_untagged(&self) -> usize {
        self.as_usize_unchecked() & LazyStateID::MAX
    }

    /// Return this lazy state ID as its raw internal `usize` value, which may
    /// be tagged (and thus greater than LazyStateID::MAX).
    #[inline]
    pub(crate) const fn as_usize_unchecked(&self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub(crate) const fn to_unknown(&self) -> LazyStateID {
        LazyStateID::new_unchecked(
            self.as_usize_unchecked() | LazyStateID::MASK_UNKNOWN,
        )
    }

    #[inline]
    pub(crate) const fn to_dead(&self) -> LazyStateID {
        LazyStateID::new_unchecked(
            self.as_usize_unchecked() | LazyStateID::MASK_DEAD,
        )
    }

    #[inline]
    pub(crate) const fn to_quit(&self) -> LazyStateID {
        LazyStateID::new_unchecked(
            self.as_usize_unchecked() | LazyStateID::MASK_QUIT,
        )
    }

    /// Return this lazy state ID as a state ID that is tagged as a start
    /// state.
    #[inline]
    pub(crate) const fn to_start(&self) -> LazyStateID {
        LazyStateID::new_unchecked(
            self.as_usize_unchecked() | LazyStateID::MASK_START,
        )
    }

    /// Return this lazy state ID as a lazy state ID that is tagged as a match
    /// state.
    #[inline]
    pub(crate) const fn to_match(&self) -> LazyStateID {
        LazyStateID::new_unchecked(
            self.as_usize_unchecked() | LazyStateID::MASK_MATCH,
        )
    }

    /// Return true if and only if this lazy state ID is tagged.
    ///
    /// When a lazy state ID is tagged, then one can conclude that it is one
    /// of a match, start, dead, quit or unknown state.
    #[inline]
    pub const fn is_tagged(&self) -> bool {
        self.as_usize_unchecked() > LazyStateID::MAX
    }

    /// Return true if and only if this represents a lazy state ID that is
    /// "unknown." That is, the state has not yet been created. When a caller
    /// sees this state ID, it generally means that a state has to be computed
    /// in order to proceed.
    #[inline]
    pub const fn is_unknown(&self) -> bool {
        self.as_usize_unchecked() & LazyStateID::MASK_UNKNOWN > 0
    }

    /// Return true if and only if this represents a dead state. A dead state
    /// is a state that can never transition to any other state except the
    /// dead state. When a dead state is seen, it generally indicates that a
    /// search should stop.
    #[inline]
    pub const fn is_dead(&self) -> bool {
        self.as_usize_unchecked() & LazyStateID::MASK_DEAD > 0
    }

    /// Return true if and only if this represents a quit state. A quit state
    /// is a state that is representationally equivalent to a dead state,
    /// except it indicates the automaton has reached a point at which it can
    /// no longer determine whether a match exists or not. In general, this
    /// indicates an error during search and the caller must either pass this
    /// error up or use a different search technique.
    #[inline]
    pub const fn is_quit(&self) -> bool {
        self.as_usize_unchecked() & LazyStateID::MASK_QUIT > 0
    }

    /// Return true if and only if this lazy state ID has been tagged as a
    /// start state.
    #[inline]
    pub const fn is_start(&self) -> bool {
        self.as_usize_unchecked() & LazyStateID::MASK_START > 0
    }

    /// Return true if and only if this lazy state ID has been tagged as a
    /// match state.
    #[inline]
    pub const fn is_match(&self) -> bool {
        self.as_usize_unchecked() & LazyStateID::MASK_MATCH > 0
    }
}

/// This error occurs when a lazy state ID could not be constructed.
///
/// This occurs when given an integer exceeding the maximum lazy state ID
/// value.
///
/// When the `std` feature is enabled, this implements the `Error` trait.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LazyStateIDError {
    attempted: u64,
}

impl LazyStateIDError {
    /// Returns the value that failed to constructed a lazy state ID.
    pub(crate) fn attempted(&self) -> u64 {
        self.attempted
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LazyStateIDError {}

impl core::fmt::Display for LazyStateIDError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(
            f,
            "failed to create LazyStateID from {:?}, which exceeds {:?}",
            self.attempted(),
            LazyStateID::MAX,
        )
    }
}

/// Represents the current state of an overlapping search.
///
/// This is used for overlapping searches since they need to know something
/// about the previous search. For example, when multiple patterns match at the
/// same position, this state tracks the last reported pattern so that the next
/// search knows whether to report another matching pattern or continue with
/// the search at the next position. Additionally, it also tracks which state
/// the last search call terminated in.
///
/// This type provides no introspection capabilities. The only thing a caller
/// can do is construct it and pass it around to permit search routines to use
/// it to track state.
///
/// Callers should always provide a fresh state constructed via
/// [`OverlappingState::start`] when starting a new search. Reusing state from
/// a previous search may result in incorrect results.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OverlappingState {
    /// The state ID of the state at which the search was in when the call
    /// terminated. When this is a match state, `last_match` must be set to a
    /// non-None value.
    ///
    /// A `None` value indicates the start state of the corresponding
    /// automaton. We cannot use the actual ID, since any one automaton may
    /// have many start states, and which one is in use depends on several
    /// search-time factors.
    id: Option<LazyStateID>,
    /// Information associated with a match when `id` corresponds to a match
    /// state.
    last_match: Option<StateMatch>,
}

/// Internal state about the last match that occurred. This records both the
/// offset of the match and the match index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StateMatch {
    /// The index into the matching patterns for the current match state.
    pub(crate) match_index: usize,
    /// The offset in the haystack at which the match occurred. This is used
    /// when reporting multiple matches at the same offset. That is, when
    /// an overlapping search runs, the first thing it checks is whether it's
    /// already in a match state, and if so, whether there are more patterns
    /// to report as matches in that state. If so, it increments `match_index`
    /// and returns the pattern and this offset. Once `match_index` exceeds the
    /// number of matching patterns in the current state, the search continues.
    pub(crate) offset: usize,
}

impl OverlappingState {
    /// Create a new overlapping state that begins at the start state of any
    /// automaton.
    pub fn start() -> OverlappingState {
        OverlappingState { id: None, last_match: None }
    }

    pub(crate) fn id(&self) -> Option<LazyStateID> {
        self.id
    }

    pub(crate) fn set_id(&mut self, id: LazyStateID) {
        self.id = Some(id);
    }

    pub(crate) fn last_match(&mut self) -> Option<&mut StateMatch> {
        self.last_match.as_mut()
    }

    pub(crate) fn set_last_match(&mut self, last_match: StateMatch) {
        self.last_match = Some(last_match);
    }
}
