use alloc::{boxed::Box, vec, vec::Vec};

use crate::util::id::StateID;

/// A pairse of sparse sets.
///
/// This is useful when one needs to compute NFA epsilon closures from a
/// previous set of states derived from an epsilon closure. One set can be the
/// starting states where as the other set can be the destination states after
/// following the transitions for a particular byte of input.
///
/// There is no significance to 'set1' or 'set2'. They are both sparse sets of
/// the same size.
///
/// The members of this struct are exposed so that callers may borrow 'set1'
/// and 'set2' individually without being force to borrow both at the same
/// time.
#[derive(Clone)]
pub(crate) struct SparseSets<T> {
    pub(crate) set1: SparseSet<T>,
    pub(crate) set2: SparseSet<T>,
}

impl<T: Clone> SparseSets<T> {
    /// Create a new pair of sparse sets where each set has the given capacity.
    ///
    /// This panics if the capacity given is bigger than `StateID::LIMIT`.
    pub(crate) fn new(capacity: usize) -> SparseSets<T> {
        SparseSets {
            set1: SparseSet::new(capacity),
            set2: SparseSet::new(capacity),
        }
    }

    /// Resizes these sparse sets to have the new capacity given.
    ///
    /// The sets are automatically cleared.
    ///
    /// This panics if the capacity given is bigger than `StateID::LIMIT`.
    #[inline]
    pub(crate) fn resize(&mut self, new_capacity: usize) {
        self.set1.resize(new_capacity);
        self.set2.resize(new_capacity);
    }

    /// Clear both sparse sets.
    pub(crate) fn clear(&mut self) {
        self.set1.clear();
        self.set2.clear();
    }

    /// Swap set1 with set2.
    pub(crate) fn swap(&mut self) {
        core::mem::swap(&mut self.set1, &mut self.set2);
    }

    /// Returns the memory usage, in bytes, used by this pair of sparse sets.
    pub(crate) fn memory_usage(&self) -> usize {
        self.set1.memory_usage() + self.set2.memory_usage()
    }
}

impl<T: core::fmt::Debug> core::fmt::Debug for SparseSets<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("SparseSets")
            .field("set1", &self.set1)
            .field("set2", &self.set2)
            .finish()
    }
}

// TODO: We need a slightly more sophisticated SparseSet for the NFA, where we
// can create a spot for something with additional data aside from the state
// ID.

/// A sparse set used for representing ordered NFA states.
///
/// This supports constant time addition and membership testing. Clearing an
/// entire set can also be done in constant time. Iteration yields elements
/// in the order in which they were inserted.
///
/// The data structure is based on: https://research.swtch.com/sparse
/// Note though that we don't actually use uninitialized memory. We generally
/// reuse sparse sets, so the initial allocation cost is bareable. However, its
/// other properties listed above are extremely useful.
#[derive(Clone)]
pub(crate) struct SparseSet<T = ()> {
    /// The number of elements currently in this set.
    len: usize,
    /// Dense contains the ids in the order in which they were inserted.
    dense: Vec<Element<T>>,
    /// Sparse maps ids to their location in dense.
    ///
    /// A state ID is in the set if and only if
    /// sparse[id] < len && id == dense[sparse[id]].
    ///
    /// Note that these are indices into 'dense'. It's a little weird to use
    /// StateID here, but we know our length can never exceed the bounds of
    /// StateID (enforced by 'resize') and StateID will be at most 4 bytes
    /// where as a usize is likely double that in most cases.
    sparse: Vec<StateID>,
}

#[derive(Clone, Debug)]
struct Element<T> {
    id: StateID,
    value: Option<T>,
}

impl<T: Clone> SparseSet<T> {
    /// Create a new sparse set with the given capacity.
    ///
    /// Sparse sets have a fixed size and they cannot grow. Attempting to
    /// insert more distinct elements than the total capacity of the set will
    /// result in a panic.
    ///
    /// This panics if the capacity given is bigger than `StateID::LIMIT`.
    #[inline]
    pub(crate) fn new(capacity: usize) -> SparseSet<T> {
        let mut set = SparseSet { len: 0, dense: vec![], sparse: vec![] };
        set.resize(capacity);
        set
    }

    /// Resizes this sparse set to have the new capacity given.
    ///
    /// This set is automatically cleared.
    ///
    /// This panics if the capacity given is bigger than `StateID::LIMIT`.
    #[inline]
    pub(crate) fn resize(&mut self, new_capacity: usize) {
        assert!(
            new_capacity <= StateID::LIMIT,
            "sparse set capacity cannot excced {:?}",
            StateID::LIMIT
        );
        self.clear();
        self.dense
            .resize(new_capacity, Element { id: StateID::ZERO, value: None });
        self.sparse.resize(new_capacity, StateID::ZERO);
    }
}

impl<T> SparseSet<T> {
    /// Returns the capacity of this set.
    ///
    /// The capacity represents a fixed limit on the number of distinct
    /// elements that are allowed in this set. The capacity cannot be changed.
    #[inline]
    pub(crate) fn capacity(&self) -> usize {
        self.dense.len()
    }

    /// Returns the number of elements in this set.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Returns true if and only if this set is empty.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Insert the state ID value into this set and return true if the given
    /// state ID was not previously in this set.
    ///
    /// This operation is idempotent. If the given value is already in this
    /// set, then this is a no-op.
    ///
    /// If more than `capacity` ids are inserted, then this panics.
    ///
    /// This is marked as inline(always) since the compiler won't inline it
    /// otherwise, and it's a fairly hot piece of code in DFA determinization.
    #[inline(always)]
    pub(crate) fn insert(&mut self, id: StateID) -> bool {
        if self.contains(id) {
            return false;
        }

        let i = self.len();
        assert!(
            i < self.capacity(),
            "{:?} exceeds capacity of {:?} when inserting {:?}",
            i,
            self.capacity(),
            id,
        );
        // OK since i < self.capacity() and self.capacity() is guaranteed to
        // be <= StateID::LIMIT.
        let index = StateID::new_unchecked(i);
        self.dense[index] = Element { id, value: None };
        self.sparse[id] = index;
        self.len += 1;
        true
    }

    #[inline]
    pub(crate) fn set(&mut self, id: StateID, value: T) {
        debug_assert!(self.contains(id));
        self.dense[self.sparse[id]].value = Some(value);
    }

    /// Returns true if and only if this set contains the given value.
    #[inline]
    pub(crate) fn contains(&self, id: StateID) -> bool {
        let index = self.sparse[id];
        index.as_usize() < self.len() && self.dense[index].id == id
    }

    /// Returns a reference to the value associated with the given ID.
    #[inline]
    pub(crate) fn get(&self, id: StateID) -> Option<&T> {
        self.dense[self.sparse[id]].value.as_ref()
    }

    /// Returns a mutable reference to the value associated with the given ID.
    #[inline]
    pub(crate) fn get_mut(&mut self, id: StateID) -> Option<&mut T> {
        self.dense[self.sparse[id]].value.as_mut()
    }

    /// Returns the ith inserted ID from this set.
    ///
    /// Panics when i >= self.len().
    #[inline]
    pub(crate) fn get_id(&self, index: usize) -> StateID {
        assert!(index < self.len());
        self.dense[index].id
    }

    /// Clear this set such that it has no members.
    #[inline]
    pub(crate) fn clear(&mut self) {
        self.len = 0;
    }

    /// Returns the heap memory usage, in bytes, used by this sparse set.
    #[inline]
    pub(crate) fn memory_usage(&self) -> usize {
        self.dense.len() * StateID::SIZE + self.sparse.len() * StateID::SIZE
    }
}

impl<T: core::fmt::Debug> core::fmt::Debug for SparseSet<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        let elements: Vec<StateID> = self.into_iter().collect();
        f.debug_tuple("SparseSet").field(&elements).finish()
    }
}

/// An iterator over all elements in a sparse set.
///
/// The lifetime `'a` refers to the lifetime of the set being iterated over.
#[derive(Debug)]
pub(crate) struct SparseSetIter<'a, T>(core::slice::Iter<'a, Element<T>>);

impl<'a, T> IntoIterator for &'a SparseSet<T> {
    type Item = StateID;
    type IntoIter = SparseSetIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        SparseSetIter(self.dense[..self.len()].iter())
    }
}

impl<'a, T> Iterator for SparseSetIter<'a, T> {
    type Item = StateID;

    #[inline(always)]
    fn next(&mut self) -> Option<StateID> {
        self.0.next().map(|element| element.id)
    }
}
