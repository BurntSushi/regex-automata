use alloc::{boxed::Box, vec, vec::Vec};

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
#[derive(Debug)]
pub(crate) struct SparseSets {
    pub(crate) set1: SparseSet,
    pub(crate) set2: SparseSet,
}

impl SparseSets {
    /// Create a new pair of sparse sets where each set has the given capacity.
    pub(crate) fn new(capacity: usize) -> SparseSets {
        SparseSets {
            set1: SparseSet::new(capacity),
            set2: SparseSet::new(capacity),
        }
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
}

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
pub(crate) struct SparseSet {
    /// The number of elements currently in this set.
    len: usize,
    /// Dense contains the instruction pointers in the order in which they
    /// were inserted.
    dense: Box<[usize]>,
    /// Sparse maps instruction pointers to their location in dense.
    ///
    /// An instruction pointer is in the set if and only if
    /// sparse[ip] < dense.len() && ip == dense[sparse[ip]].
    sparse: Box<[usize]>,
}

impl SparseSet {
    /// Create a new sparse set with the given capacity.
    ///
    /// Sparse sets have a fixed size and they cannot grow. Attempting to
    /// insert more distinct elements than the total capacity of the set will
    /// result in a panic.
    #[inline]
    pub(crate) fn new(capacity: usize) -> SparseSet {
        SparseSet {
            len: 0,
            dense: vec![0; capacity].into_boxed_slice(),
            sparse: vec![0; capacity].into_boxed_slice(),
        }
    }

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

    /// Insert the given value into this set and return true if the given value
    /// was not previously in this set.
    ///
    /// This operation is idempotent. If the given value is already in this
    /// set, then this is a no-op.
    ///
    /// If more than `capacity` elements are inserted, then this panics.
    ///
    /// This is marked as inline(always) since the compiler won't inline it
    /// otherwise, and it's a fairly hot piece of code in DFA determinization.
    #[inline(always)]
    pub(crate) fn insert(&mut self, value: usize) -> bool {
        if self.contains(value) {
            return false;
        }

        let i = self.len();
        assert!(
            i < self.capacity(),
            "{} exceeds capacity of {} when inserting {}",
            i,
            self.capacity(),
            value,
        );
        self.dense[i] = value;
        self.sparse[value] = i;
        self.len += 1;
        true
    }

    /// Returns true if and only if this set contains the given value.
    #[inline]
    pub(crate) fn contains(&self, value: usize) -> bool {
        let i = self.sparse[value];
        i < self.len() && self.dense[i] == value
    }

    /// Clear this set such that it has no members.
    #[inline]
    pub(crate) fn clear(&mut self) {
        self.len = 0;
    }
}

impl core::fmt::Debug for SparseSet {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        let elements: Vec<usize> = self.into_iter().collect();
        f.debug_tuple("SparseSet").field(&elements).finish()
    }
}

/// An iterator over all elements in a sparse set.
///
/// The lifetime `'a` refers to the lifetime of the set being iterated over.
#[derive(Debug)]
pub(crate) struct SparseSetIter<'a>(core::slice::Iter<'a, usize>);

impl<'a> IntoIterator for &'a SparseSet {
    type Item = usize;
    type IntoIter = SparseSetIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        SparseSetIter(self.dense[..self.len()].iter())
    }
}

impl<'a> Iterator for SparseSetIter<'a> {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        self.0.next().map(|value| *value)
    }
}
