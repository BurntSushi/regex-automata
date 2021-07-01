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

    const SENTINEL_UNKNOWN: LazyStateID =
        LazyStateID::new_unchecked(1 << LazyStateID::MAX_BIT);

    const SENTINEL_DEAD: LazyStateID = LazyStateID::new_unchecked(
        LazyStateID::SENTINEL_UNKNOWN.as_usize_unchecked() + 1,
    );

    const SENTINEL_QUIT: LazyStateID = LazyStateID::new_unchecked(
        LazyStateID::SENTINEL_DEAD.as_usize_unchecked() + 1,
    );

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

    /// Create a new lazy state ID that always represents an "unknown" state.
    ///
    /// An unknown state is a placeholder for a DFA state that has not yet
    /// been computed. Like a dead state, all transitions on the "unknown"
    /// state lead right back to itself.
    #[inline]
    pub(crate) const fn unknown() -> LazyStateID {
        LazyStateID::SENTINEL_UNKNOWN
    }

    #[inline]
    pub(crate) const fn dead() -> LazyStateID {
        LazyStateID::SENTINEL_DEAD
    }

    #[inline]
    pub(crate) const fn quit() -> LazyStateID {
        LazyStateID::SENTINEL_QUIT
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

    /// Return this lazy state ID as its raw value if and only if it is
    /// unmasked. When the lazy state ID is unknown, dead, quit or tagged as
    /// a start or a match state, then it is masked and thus this returns None.
    #[inline]
    pub(crate) fn as_usize(&self) -> Option<usize> {
        if self.is_unmasked() {
            Some(self.as_usize_unchecked())
        } else {
            None
        }
    }

    /// Return this lazy state ID as an unmasked `usize`.
    #[inline]
    pub(crate) fn as_usize_unmasked(&self) -> usize {
        self.as_usize_unchecked() & LazyStateID::MAX
    }

    /// Return this lazy state ID as its raw internal `usize` value, which may
    /// be masked (and thus greater than LazyStateID::MAX).
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

    /// Return true if and only if this lazy state ID is completely unmasked.
    ///
    /// When a lazy state ID is unmasked, then one can conclude that it is NOT
    /// a match, state, dead, quit or unknown state.
    #[inline]
    pub(crate) fn is_unmasked(&self) -> bool {
        self.as_usize_unchecked() <= LazyStateID::MAX
    }

    /// Return true if and only if this represents a lazy state ID that is
    /// "unknown." That is, the state has not yet been created. When a caller
    /// sees this state ID, it generally means that a state has to be computed
    /// in order to proceed.
    #[inline]
    pub(crate) fn is_unknown(&self) -> bool {
        self.as_usize_unchecked() & LazyStateID::MASK_UNKNOWN > 0
    }

    /// Return true if and only if this represents a dead state. A dead state
    /// is a state that can never transition to any other state except the
    /// dead state. When a dead state is seen, it generally indicates that a
    /// search should stop.
    #[inline]
    pub(crate) fn is_dead(&self) -> bool {
        self.as_usize_unchecked() & LazyStateID::MASK_DEAD > 0
    }

    /// Return true if and only if this represents a quit state. A quit state
    /// is a state that is representationally equivalent to a dead state,
    /// except it indicates the automaton has reached a point at which it can
    /// no longer determine whether a match exists or not. In general, this
    /// indicates an error during search and the caller must either pass this
    /// error up or use a different search technique.
    #[inline]
    pub(crate) fn is_quit(&self) -> bool {
        self.as_usize_unchecked() & LazyStateID::MASK_QUIT > 0
    }

    /// Return true if and only if this lazy state ID has been tagged as a
    /// start state.
    #[inline]
    pub(crate) fn is_start(&self) -> bool {
        self.as_usize_unchecked() & LazyStateID::MASK_START > 0
    }

    /// Return true if and only if this lazy state ID has been tagged as a
    /// match state.
    #[inline]
    pub(crate) const fn is_match(&self) -> bool {
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
    pub fn attempted(&self) -> u64 {
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
