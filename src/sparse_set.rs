use std::slice;

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
pub struct SparseSet {
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
    pub fn new(size: usize) -> SparseSet {
        SparseSet {
            len: 0,
            dense: vec![0; size].into_boxed_slice(),
            sparse: vec![0; size].into_boxed_slice(),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn insert(&mut self, value: usize) {
        let i = self.len();
        assert!(
            i < self.dense.len(),
            "{} exceeds capacity of {}",
            i,
            self.dense.len()
        );
        self.dense[i] = value;
        self.sparse[value] = i;
        self.len += 1;
    }

    pub fn contains(&self, value: usize) -> bool {
        let i = self.sparse[value];
        i < self.len() && self.dense[i] == value
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }
}

impl std::fmt::Debug for SparseSet {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let elements: Vec<&usize> = self.into_iter().collect();
        f.debug_tuple("SparseSet").field(&elements).finish()
    }
}

#[derive(Debug)]
pub struct SparseSetIter<'a> {
    set: &'a SparseSet,
    i: usize,
}

impl<'a> IntoIterator for &'a SparseSet {
    type Item = &'a usize;
    type IntoIter = SparseSetIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        SparseSetIter { set: self, i: 0 }
    }
}

impl<'a> Iterator for SparseSetIter<'a> {
    type Item = &'a usize;

    fn next(&mut self) -> Option<&'a usize> {
        if self.i < self.set.len() {
            let element = &self.set.dense[self.i];
            self.i += 1;
            Some(element)
        } else {
            None
        }
    }
}
