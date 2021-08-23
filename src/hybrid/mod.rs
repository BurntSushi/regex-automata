/*!
# When should I use this?

Generally speaking, if you can abide the use of mutable state during search,
and you don't need things like capturing groups or Unicode word boundaries
support in non-ASCII text, then a lazy DFA is likely a robust choice with
respect to both search speed and memory usage. Note however that its speed
may be worse than a general purpose regex engine if you don't select a good
[prefilter].

If you know ahead of time that your pattern would result in a very large DFA
if it was fully compiled, it may be better to use an NFA simulation instead
of a lazy DFA. Either that, or increase the cache capacity of your lazy DFA
to something that is big enough to hold the state machine (likely through
experimentation). The issue here is that if the cache is too small, then it
could wind up being reset too frequently and this might decrease searching
speed significantly.

# Differences with fully compiled DFAs

A [`hybrid::regex::Regex`](crate::hybrid::regex::Regex) and a
[`dfa::regex::Regex`](crate::dfa::regex::Regex) both have the same capabilities
(and similarly for their underlying DFAs), but they achieve them through
different means. The main difference is that a hybrid or "lazy" regex builds
its DFA lazily during search, where as a fully compiled regex will build its
DFA at construction time. While building a DFA at search time might sound like
it's slow, it tends to work out where most bytes seen during a search will
reuse pre-built parts of the DFA and thus can be almost as fast as a fully
compiled DFA. The main downside is that searching requires mutable space to
store the DFA, and, in the worst case, a search can result in a new state being
created for each byte seen, which would make searching quite a bit slower.

A fully compiled DFA never has to worry about searches being slower once
it's built. (Aside from, say, the transition table being so large that it
is subject to harsh CPU cache effects.) However, of course, building a full
DFA can be quite time consuming and memory hungry. Particularly when it's
so easy to build large DFAs when Unicode mode is enabled.

A lazy DFA strikes a nice balance _in practice_, particularly in the
presence of Unicode mode, by only building what is needed. It avoids the
worst case exponential time complexity of DFA compilation by guaranteeing that
it will only build at most one state per byte searched. While the worst
case here can lead to a very high constant, it will never be exponential.
*/

pub use self::{
    error::{BuildError, CacheError},
    id::{LazyStateID, OverlappingState},
};

pub mod dfa;
mod error;
mod id;
pub mod regex;
mod search;
