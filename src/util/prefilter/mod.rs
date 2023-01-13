pub use self::imp::*;

#[cfg(feature = "alloc")]
mod imp;

#[cfg(not(feature = "alloc"))]
mod imp {
    use crate::util::search::{MatchKind, Span};

    #[derive(Clone, Debug)]
    pub struct Prefilter(());

    impl Prefilter {
        pub fn new<B: AsRef<[u8]>>(
            _kind: MatchKind,
            _needles: &[B],
        ) -> Option<Prefilter> {
            None
        }

        pub fn find(&self, _haystack: &[u8], _span: Span) -> Option<Span> {
            unreachable!("prefilters unsupported in no-std no-alloc")
        }

        pub fn prefix(&self, _haystack: &[u8], _span: Span) -> Option<Span> {
            unreachable!("prefilters unsupported in no-std no-alloc")
        }

        pub fn memory_usage(&self) -> usize {
            unreachable!("prefilters unsupported in no-std no-alloc")
        }
    }
}
