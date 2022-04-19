use core::num::NonZeroUsize;

#[derive(Clone, Copy, Eq, Hash, PartialEq, PartialOrd, Ord)]
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

// We provide our own Debug impl because seeing the internal repr can be quite
// surprising if you aren't expecting it.
impl core::fmt::Debug for NonMaxUsize {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "{:?}", self.get())
    }
}
