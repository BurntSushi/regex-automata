pub use self::vm::{
    Builder, Cache, CapturesEarliestMatches, CapturesLeftmostMatches,
    CapturesOverlappingMatches, Config, FindEarliestMatches,
    FindLeftmostMatches, FindOverlappingMatches, OverlappingState, PikeVM,
};

pub mod regex;
mod vm;
