pub use self::imp::*;

mod aho_corasick;
mod byteset;
#[cfg(feature = "alloc")]
mod imp;
mod memchr;
mod memmem;
mod teddy;

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
