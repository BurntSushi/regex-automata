use core::{
    fmt::Debug,
    panic::{RefUnwindSafe, UnwindSafe},
};

use crate::Span;

// TODO: Putting a Box<dyn Prefilter> into something else (like a Regex), makes
// cloning difficult. I believe Aho-Corasick resolves this with some machinery.
// Is that really our only option though? See hybrid::regex and PikeVM as
// things that can't be cloned currently because of this.
//
// Maybe it's just as simple as using Arc<dyn Prefilter>? Why didn't we do that
// in aho-corasick?

// BREADCRUMBS: What if the presumed prefilter design is just all wrong? Up
// until this point, I've been assuming that every regex engine needs to accept
// a prefilter and then weave it into its own search execution. But what if the
// prefilter should actually be the responsibility of the caller? This would
// considerably simplify the regex engines, and seems plausibly necessary for
// full flexibility. (For example, if we have a regex like `\wFOO`, it would be
// nice to centralize the logic in how we deal with an offset prefilter.)
//
// I think the main challenge with moving prefilters to the caller is to ensure
// that it is as fast as possible and that we don't lose anything by making
// this move.
//
// For the NFA at least, this seems likely true. We only execute a prefilter
// when the set of current states becomes empty and we aren't executing an
// anchored search. (Actually, in the current impl, we aren't even detecting
// whether we're in a starting state or not, since emulating the `(?s:.)*?`
// outside of the NFA itself is actually quite tricky.)
//
// For the DFA, it seems a little trickier in the case where there are
// many false positives reported by the prefilter. But using a prefilter
// in cases like this will always lead to some kind of slowdown. If we
// push the prefilter down into the search execution, then there should be
// less overhead. But if it's in the caller, then the DFA search has to be
// repeatedly stopped and started. Either way, both techniques require a
// heuristic to disable the prefilter. That heuristic threshold might be lower
// when the prefilter is in the caller (since the overhead is higher).
//
// In effect, the way this would work is that the caller would execute a
// prefilter, and for each match, run the appropriate regex engine in anchored
// mode. If a match is found, great, we're done. If a match is not found, then
// we have to re-execute the prefilter at some position. I believe the position
// should be the position at which the regex search gave up. (But we should
// verify this.) If we instead re-execute the prefilter at the end of the
// previous candidate match, then we open ourselves up to easy quadratic search
// time (in the size of the input). That's bad. The idea behind this is that if
// we didn't have an anchored search, then instead of the search failing, it
// would loop back into the start state otherwise. Thus, we do this manually
// when a prefilter is present.
//
// This does unfortunately mean that we have to change all of the regex engines
// to report the position at which it stopped a search. So, Option<Span> has
// to become SearchResult, where SearchResult is an enum. That's... annoying.
// But, frightfully, seems worth it.
//
// ... some time passes where I think about the above ...
//
// Welp, nope, the above is all bunk. It turns out that the "we should verify
// this" warning above was astute, because it doesn't work. The issue is what
// happens when we find a candidate from a prefilter but there is no regex
// match at that position. We can either restart the search after the prefilter
// candidate or we can restart it where the regex engine failed. We already
// dispensed with the former case above: it can too easily lead to quadratic
// behavior. (e.g., matching 'foo\w+Z' on 'foofoofoofoofoo'.) It turns out that
// the other strategy is bunk too, because we might miss matches. For example,
// given the regex 'foo\war' and we search 'foofoobar'. We find the first 'foo'
// from the prefilter, but the regex match doesn't fail until we get to the
// third 'o' (at position 4). So if we restart the search at position 4, the
// prefilter won't match the second 'foo' (because it's starting after the
// second 'f'), and thus, the search will incorrectly report no-match.
//
// The underlying issue here is that the prefilter can really only be executed
// when the finite state machine is in a 'start' state, because the finite
// state machine automatically accounts for the fact that the '\w' in 'foo\war'
// might be matching an 'f' in a way that brings us back around to the
// beginning of the pattern without actually entering the start state. But if
// we try to subvert this by putting the prefilter outside the context of the
// finite state machine, then we lose the FSM's context and expose ourselves to
// bad things.
//
// It seems like we are forever doomed to having to embed the prefilter into
// the state machine traversal itself. But maybe there is a simpler way to do
// it than what I'm doing now...
//
// ... more time passes ...
//
// OK, so the above strongly argues that we need to build prefilter support
// into every regex engine. (I have not thought of a better way.) If so,
// then we need to think about what we actually want. That is, what are our
// requirements.
//
// * Prefilter searches should have very low latency. Striving for the lowest
// possible latency is not plausible. That is, if a prefilter is inside the
// regex engine, then the lowest latency thing is probably to specialize every
// regex engine for every prefilter. But the code bloat from this would be
// spectacular. Instead, we sacrifice some latency by using a dyn Prefilter.
//
// * Prefilter searches should accept a haystack and a context, just like all
// regex engines. Any reported offsets should be valid with respect to the
// haystack.
//
// * Prefilter searches should NOT be limited to "prefix" searches. We want
// to support suffix searches and, to the extent possible, inner literal
// searches too.
//
// * For prefix searches, it is tempting to find a way to avoid repeating the
// word of matching the prefix via the regex engine once a candidate is found.
// But handling this correctly---particularly in the presence of capturing
// groups---seems quite tricky. This also seems tricky to handle if the prefix
// consists of multiple distinct literals. In such a case, I imagine each
// literal would need to map to its own distinct starting state. This might be
// pretty easy to pull off with a DFA, but is trickier with an NFA since you
// can't just jump into the middle of an NFA. You might have a bunch of states
// built up by that point, in addition to capturing group offsets. In theory,
// it's pre-computable though... We should keep an open mind for at least
// making this work for the DFA though, since I think that really only actually
// needs a LiteralID |--> StateID mapping.
//
// * Suffix searches are tricky to pull off because they can easily lead
// to quadratic behavior if your reverse search keeps visiting parts of
// the haystack you've already seen. The regex crate has this optimization
// currently, and it uses an alternative lazy DFA API that reports the position
// at which the reverse search failed. Using this information, we can prevent
// the reverse search from going too far backwards. Our revised lazy DFA search
// API doesn't expose this feature, but we could pretty easily write our own
// (safe) search routine for this since the low level DFA transition APIs are
// made available.
//
// * The inner searches are the hardest but awesomest. Going to avoid thinking
// too deeply about them right now, probably because as far as prefilters
// are concerned, I think that if we can do a suffix literal search, then we
// should be fine with inner searches too. The main idea is avoiding quadratic
// behavior. Another thing to think about here is whether we can do additional
// analysis to avoid search time shenanigans, i.e., when we know we won't
// revisit any text, e.g., if the inner literal doesn't match the prefix-regex
// of the pattern.
//
// ... more time passes ...
//
// OK, maybe the trick here is that prefilters are actually only limited to
// "simple" prefix accelerated searches. And anything else that's more
// complicated is really just a "meta" regex engine. Because the non-prefix
// prefilters described above can't really be implemented inside of each
// regex engine, since they require special handling. e.g., Running different
// parts in reverse or splitting the FSM into two parts.
//
// This entire enterprise also really wants an NFA reverse search that supports
// all options. Otherwise, the literal optimizations that require running a
// regex engine reverse will only be limited to the DFA engines, which in turn
// means that Unicode word boundaries will disqualify those optimizations
// entirely. And since Unicode word boundaries aren't supported by the faster
// regex engines, it's all that more important to make sure we can accelerate
// them with whatever tricks we have.
//
// So what's next here? Do we finally take a detour and write our literal
// extraction routines and then work from there? Perhaps. Or maybe we should
// just run with what we have instead of trying to invent new literal
// extraction routines now. That is, let's try to get regex-automata merged
// into regex first. But...
//
// And how do we handle the *prefix* prefilters in the APIs of the regex
// engines. We have a few different approaches thus far:
//
// * The `dfa` module bakes prefilters into the DFA as a type parameter. This
// was principally done so that prefilters were supportable in no_std mode,
// where an Arc<dyn Prefilter> might not be available. We *could* ask for a
// &dyn Prefilter (or some such) at every search call instead.
// * The `hybrid::dfa` module asks for a prefilter at every search call. The
// thinking here is that using lazy DFAs directly (instead of the regex API)
// is low level enough that you want to be able to explicitly control the
// prefilter scanner.
// * The `hybrid::regex` API stores the prefilter as part of the Regex and
// automatically handles the creation of prefilter scanners for each search
// call.
//
// Is there a One True Way of handling prefilters? Or are we doomed to support
// all three approaches..? Sigh.
//
// What if we just require that the prefilter be set in the builder? We can
// let the 'dfa' module be a special little snowflake because of no_std, but
// everything else should be configurable with the builder. If so, should we
// apply this to hybrid::dfa too? Such that the caller can't pass in a scanner?
// Hmmm, no, probably not. Because higher level iterators might want to use
// the same scanner across one search. Bummer.
//
// So I guess the rule here is that "regex level APIs" can have prefilters set
// via their builders, but the rest need it passed as an explicit parameter.
//
// No, wait, that doesn't work either... Because the PikeVM is a "regex level
// API," but there is no lower level API. So its lowest level 'find' routines
// want a prefilter given to them too. So we could ask for a prefilter in the
// builder but then ask for a scanner explicitly in the lower level routines,
// but that's weird because it means the prefilter given in the builder isn't
// used in the lower level routines.
//
// So maybe we just require a prefilter to be passed in explicitly
// everywhere. But that's a bummer, because it means things like
// hybrid::regex::Regex::find_leftmost_iter can *never* use a prefilter...
//
// Oh interesting, for hybrid::regex::Regex, we permit setting a prefilter
// on the Regex object, but expose nothing for setting the scanner on an
// individual search. Indeed, the iterator impls use internal routines that
// permit explicitly setting a scanner. Which means that callers can't actually
// implement their own iterators correctly... Welp, the same is true for
// dfa::regex::Regex too.
//
// So maybe the simplest thing to do is to just demand prefilters for the
// lowest level '*_at' routines. That's the most explicit and consistent
// approach. It works for everything, including dfa::regex::Regex.
//
// ... AHHHH but no! This means the iterators provided by the crate can never
// use a prefilter because our iterators veer towards the simple APIs. Maybe
// that's okay? In order to use a prefilter with an iterator you have to roll
// your own? Really? Either that, or we make the iterator constructors accept
// a prefilter.
//
// What if we got rid of iterators and "higher level" APIs altogether? I guess
// that would "fix" the problem too. The problem is that, because of the
// possibility of zero-width matches, writing the iterator is itself a little
// tricky. It would remove a lot of code from the crate and simplify the APIs
// quite a bit though... Ug.
//
// Now I guess I've come full circle. We should just bake the prefilter into
// the regex objects themselves. If callers really need to control the scanner
// passed into the search routine, they can drop down a level and assemble the
// DFAs themselves. They won't be able to do this for the PikeVM though. That's
// the issue.
//
// Unless we handle passing the prefilter to the higher level APIs and allow
// the caller to specify the scanner in the lower level APIs, thus ignoring
// the prefilter baked into the regex. If we do that though, we need to expose
// a convenience routine or some such for returning a scanner from the regex
// object so that it can be passed to the lower level APIs.
//
// I suppose another option here is to split Regex objects in two. One half is
// only the higher level APIs while the other half is only the lower level APIs
// with no iterators. But that feels pretty annoying to me. And what would the
// PikeVM look like? I guess the PikeVM would be the lower level API, and
// nfa::thompson::pikevm::Regex would be the higher level API? And I guess for
// the DFA-based regex engines, we just lop off the '*_at' routines entirely,
// and say that the DFAs themselves are the lower level API? I don't know, this
// just seems silly. Why lop off the '*_at' APIs at all?
//
// OK, here's a thing: so far, I've been focusing on each regex engine as
// their own individual units. I haven't been thinking too much about the end
// goal, which is that the regex engines can all be combined into on giant
// "meta" regex engine. That meta regex engine is going to mix and match stuff
// and probably won't use the higher level Regex objects at all. And almost
// certainly won't use the iterators either I think. If that's true, then the
// PikeVM at least really MUST expose a way to pass down a prefilter at search
// time. I guess otherwise the meta regex engine will have to stuff prefilters
// down into all of the regex objects?
//
// ... I think.
//
// Popping up a level here, what do iterators really want? Why do they care
// about prefilters at all? The issue here is that a prefilter::Scanner bundles
// some mutable state with an actual prefilter. That mutable state is used
// to determine whether the prefilter should continue to be used or not, via
// the 'is_effective' method. And so the thinking is that an iterator reflects
// a "single search," even with multiple possible matches. So we keep the same
// scanner state throughout the lifetime of the iterator. So if the prefilter
// gets disabled early, it doesn't continually try to re-evaluate itself. It
// gets disabled and stays disabled. So there is an inherent advantage to the
// iterators. Although... Some benchmarks I've run (can't remember which) have
// suggested that re-enabling the prefilter might actually be worth it, since
// the disabling may no longer be the correct choice. Hard call.
//
// Also, if we don't couple the scanner together with the prefilter, then we're
// adding YET ANOTHER parameter to the lower level search routines, which are
// already busting at the seams.
//
// How much simpler would things get if we said, "nah there's no scanner, we
// just always run the prefilter if one is present." Basically, commit to our
// fate. It's worth noting that memmem and aho-corasick already have heuristics
// in place to quit using their *own* prefilters. So maybe we shouldn't do it
// ourselves too? This would remove the complication around iterators being
// a special snowflake, but it also poses a risk: if we do wind up needing
// something like a scanner to heuristically disable a prefilter, then we will
// have built up an architecture around the idea that we don't need one. And
// then it will be hard to retrofit.
//
// So maybe the PikeVM is kind of unique here in that it's trying to mix low
// level with high level. The DFA-based regex engines aren't so bad because
// callers can always build their own forward/reverse DFAs and piece things
// together that way. Not ideal, but not the end of the world. Certainly
// nowhere as difficult as building your own PikeVM. So maybe the answer
// really is that the PikeVM is low-level APIs only, and then we add a
// nfa::thompson::pikevm::regex module that gives the higher level niceties.
//
// I guess the same will be true for the backtracker as well. And onepass.

// DREAM: When writing the prefilter APIs below, I mostly looked at what the
// regex crate was already doing in order to get regex-automata merged into
// the regex crate expeditiously. However, I would very much like to improve
// how prefilters are done, especially for the multi-regex case. I suspect the
// interface for that will look quite a bit different, since you really want
// the prefilter to report for which patterns there was a match. For now, we
// just require that the prefilter report match spans.

#[derive(Clone, Debug)]
pub enum Candidate {
    None,
    Match(Span),
    PossibleMatch(Span),
}

impl Candidate {
    /// Convert this candidate into an option.
    ///
    /// This is useful when callers do not distinguish between true positives
    /// and false positives (i.e., the caller must always confirm the match in
    /// order to update some other state).
    ///
    /// The byte offset in the option returned corresponds to the starting
    /// position of the possible match.
    pub fn into_option(self) -> Option<usize> {
        match self {
            Candidate::None => None,
            Candidate::Match(ref m) => Some(m.start()),
            Candidate::PossibleMatch(ref m) => Some(m.start()),
        }
    }
}

pub trait Prefilter: Debug + Send + Sync + RefUnwindSafe + UnwindSafe {
    fn find(&self, state: &mut State, haystack: &[u8], at: usize)
        -> Candidate;

    fn memory_usage(&self) -> usize;

    fn reports_false_positives(&self) -> bool {
        true
    }
}

impl<'a, P: Prefilter + ?Sized> Prefilter for &'a P {
    #[inline]
    fn find(
        &self,
        state: &mut State,
        haystack: &[u8],
        at: usize,
    ) -> Candidate {
        (**self).find(state, haystack, at)
    }

    #[inline]
    fn memory_usage(&self) -> usize {
        (**self).memory_usage()
    }

    #[inline]
    fn reports_false_positives(&self) -> bool {
        (**self).reports_false_positives()
    }
}

#[derive(Clone, Debug)]
pub struct Scanner<'p> {
    prefilter: &'p dyn Prefilter,
    state: State,
}

impl<'p> Scanner<'p> {
    #[inline]
    pub fn new(prefilter: &'p dyn Prefilter) -> Scanner<'p> {
        Scanner { prefilter, state: State::new() }
    }

    pub(crate) fn is_effective(&mut self, at: usize) -> bool {
        self.state.is_effective(at)
    }

    pub(crate) fn reports_false_positives(&self) -> bool {
        self.prefilter.reports_false_positives()
    }

    pub(crate) fn find(&mut self, bytes: &[u8], at: usize) -> Candidate {
        self.prefilter.find(&mut self.state, bytes, at)
    }
}

/// State tracks state associated with the effectiveness of a prefilter.
///
/// While the specifics on how it works are an implementation detail, the
/// idea here is that this will make heuristic decisions about whether it's
/// advantageuous to continue executing a prefilter or not, typically based on
/// how many bytes the prefilter tends to skip.
///
/// A prefilter state should be created for each search. (Where creating an
/// iterator is typically treated as a single search.)
#[derive(Clone, Debug)]
pub struct State {
    /// We currently don't keep track of anything and always execute
    /// prefilters. This may change in the future.
    _priv: (),
}

impl State {
    /// Create a fresh prefilter state.
    fn new() -> State {
        State { _priv: () }
    }

    /// Return true if and only if this state indicates that a prefilter is
    /// still effective. If the prefilter is not effective, then this state
    /// is rendered "inert." At which point, all subsequent calls to
    /// `is_effective` on this state will return `false`.
    ///
    /// `at` should correspond to the current starting position of the search.
    ///
    /// Callers typically do not need to use this, as it represents the
    /// default implementation of
    /// [`Prefilter::is_effective`](trait.Prefilter.html#tymethod.is_effective).
    fn is_effective(&mut self, _at: usize) -> bool {
        true
    }
}

/// A `Prefilter` implementation that reports a possible match at every
/// position.
///
/// This should generally not be used as an actual prefilter. It is only
/// useful when one needs to represent the absence of a prefilter in a generic
/// context. For example, a [`dfa::regex::Regex`](crate::dfa::regex::Regex)
/// uses this prefilter by default to indicate that no prefilter should be
/// used.
///
/// A `None` prefilter value cannot be constructed.
#[derive(Clone, Debug)]
pub struct None {
    _priv: (),
}

impl Prefilter for None {
    fn find(&self, _: &mut State, _: &[u8], at: usize) -> Candidate {
        Candidate::PossibleMatch(Span::new(at, at))
    }

    fn memory_usage(&self) -> usize {
        0
    }
}
