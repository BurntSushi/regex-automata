use core::num::NonZeroUsize;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
#[repr(transparent)]
pub(crate) struct NonMaxUsize(NonZeroUsize);

impl NonMaxUsize {
    pub(crate) fn new(value: usize) -> Option<NonMaxUsize> {
        NonZeroUsize::new(value.wrapping_add(1)).map(NonMaxUsize)
    }

    pub(crate) fn get(self) -> usize {
        self.0.get().wrapping_sub(1)
    }
}
