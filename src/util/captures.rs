use alloc::sync::Arc;

use crate::util::primitives::{
    PatternID, PatternIDError, PatternIDIter, SmallIndex, SmallIndexError,
};

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
#[cfg(feature = "alloc")]
#[derive(Clone, Debug)]
pub struct GroupInfo(Arc<GroupInfoInner>);

#[cfg(feature = "alloc")]
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
    /// This also returns `None` for all inputs if captures are not enabled for
    /// this NFA or are not present for the given pattern. To check whether
    /// captures are both enabled for the NFA and are present for a specific
    /// pattern, use [`NFA::pattern_capture_len`].
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
    /// assert_eq!(Some(2), nfa.capture_name_to_index(pid0, "foo"));
    /// // Recall that capture index 0 is always unnamed and refers to the
    /// // entire pattern. So the first capturing group present in the pattern
    /// // itself always starts at index 1.
    /// assert_eq!(Some(1), nfa.capture_name_to_index(pid1, "foo"));
    ///
    /// // And if a name does not exist for a particular pattern, None is
    /// // returned.
    /// assert!(nfa.capture_name_to_index(pid0, "quux").is_some());
    /// assert!(nfa.capture_name_to_index(pid1, "quux").is_none());
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
    /// This also returns `None` for all inputs if captures are not enabled for
    /// this NFA or are not present for the given pattern. To check whether
    /// captures are both enabled for the NFA and are present for a specific
    /// pattern, use [`NFA::pattern_capture_len`].
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
    /// assert_eq!(None, nfa.capture_index_to_name(pid0, 0));
    /// assert_eq!(Some("foo"), nfa.capture_index_to_name(pid0, 1));
    /// assert_eq!(None, nfa.capture_index_to_name(pid0, 2));
    /// assert_eq!(None, nfa.capture_index_to_name(pid0, 3));
    ///
    /// assert_eq!(None, nfa.capture_index_to_name(pid1, 0));
    /// assert_eq!(None, nfa.capture_index_to_name(pid1, 1));
    /// assert_eq!(Some("foo"), nfa.capture_index_to_name(pid1, 2));
    /// // '3' is not a valid capture index for the second pattern.
    /// assert_eq!(None, nfa.capture_index_to_name(pid1, 3));
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
    ///     nfa.pattern_capture_names(PatternID::ZERO).collect();
    /// assert_eq!(expected, got);
    ///
    /// // Using an invalid pattern ID will result in nothing yielded.
    /// assert_eq!(0, nfa.pattern_capture_names(PatternID::must(999)).count());
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

    /// Return an iterator of all capture groups for all patterns in this NFA.
    /// Each item yield is a triple of the group's pattern ID, index in the
    /// pattern and the group's name, if present.
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
    ///     nfa.all_capture_names().collect();
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
    /// assert_eq!(0, nfa.all_capture_names().count());
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
/// This type is actually wrapped inside a Vec indexed by pattern ID on the
/// NFA, since multiple patterns may have the same capture group name. That is,
/// each pattern gets its own namespace of capture group names.
///
/// Perhaps a more memory efficient representation would be
/// HashMap<(PatternID, Arc<str>), usize>, but this makes it difficult to look
/// up a capture index by name without producing a `Arc<str>`, which requires
/// an allocation. To fix this, I think we'd need to define our own unsized
/// type or something? Anyway, I didn't give this much thought since it
/// probably doesn't matter much in the grand scheme of things. But it did
/// stand out to me as mildly wasteful.
#[cfg(all(feature = "alloc", feature = "std"))]
type CaptureNameMap = std::collections::HashMap<Arc<str>, SmallIndex>;
#[cfg(all(feature = "alloc", not(feature = "std")))]
type CaptureNameMap = alloc::collections::BTreeMap<Arc<str>, SmallIndex>;

/// The inner guts of `GroupInfo`. This type only exists so that it can
/// be wrapped in an `Arc` to make `GroupInfo` reference counted.
#[cfg(feature = "alloc")]
#[derive(Debug)]
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
#[cfg(feature = "alloc")]
#[derive(Clone, Debug)]
pub struct GroupInfoError {
    kind: GroupInfoErrorKind,
}

/// The kind of error that occurs when building a `GroupInfo` fails.
///
/// We keep this un-exported because it's not clear how useful it is to
/// export it.
#[cfg(feature = "alloc")]
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

#[cfg(feature = "alloc")]
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

#[cfg(feature = "std")]
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
        // If the NFA has no captures, then we never have anything to yield. We
        // need to consider this case explicitly (at time of writing) because
        // 'pattern_capture_names' will panic if captures aren't enabled.
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
