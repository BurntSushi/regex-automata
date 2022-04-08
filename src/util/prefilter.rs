use crate::Match;

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
// to report the position at which it stopped a search. So, Option<Match> has
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

/// A candidate is the result of running a prefilter on a haystack at a
/// particular position. The result is one of no match, a confirmed match or
/// a possible match.
///
/// When no match is returned, the prefilter is guaranteeing that no possible
/// match can be found in the haystack, and the caller may trust this. That is,
/// all correct prefilters must never report false negatives.
///
/// In some cases, a prefilter can confirm a match very quickly, in which case,
/// the caller may use this to stop what it's doing and report the match. In
/// this case, prefilter implementations must never report a false positive.
/// In other cases, the prefilter can only report a potential match, in which
/// case the callers must attempt to confirm the match. In this case, prefilter
/// implementations are permitted to return false positives.
#[derive(Clone, Debug)]
pub enum Candidate {
    /// The prefilter reports that no match is possible. Prefilter
    /// implementations will never report false negatives.
    None,
    /// The prefilter reports that a match has been confirmed at the provided
    /// byte offsets. When this variant is reported, the prefilter is
    /// guaranteeing a match. No false positives are permitted.
    Match(Match),
    /// The prefilter reports that a match *may* start at the given position.
    /// When this variant is reported, it may correspond to a false positive.
    PossibleStartOfMatch(usize),
}

impl Candidate {
    /// Convert this candidate into an option. This is useful when callers do
    /// not distinguish between true positives and false positives (i.e., the
    /// caller must always confirm the match in order to update some other
    /// state).
    ///
    /// The byte offset in the option returned corresponds to the starting
    /// position of the possible match.
    pub fn into_option(self) -> Option<usize> {
        match self {
            Candidate::None => None,
            Candidate::Match(ref m) => Some(m.start()),
            Candidate::PossibleStartOfMatch(start) => Some(start),
        }
    }
}

/// A prefilter describes the behavior of fast literal scanners for quickly
/// skipping past bytes in the haystack that we know cannot possibly
/// participate in a match.
pub trait Prefilter: core::fmt::Debug {
    /// Returns the next possible match candidate. This may yield false
    /// positives, so callers must confirm a match starting at the position
    /// returned. This, however, must never produce false negatives. That is,
    /// this must, at minimum, return the starting position of the next match
    /// in the given haystack after or at the given position.
    fn next_candidate(
        &self,
        state: &mut State,
        haystack: &[u8],
        at: usize,
    ) -> Candidate;

    /// Returns the approximate total amount of heap used by this prefilter, in
    /// units of bytes.
    fn heap_bytes(&self) -> usize;

    /// Returns true if and only if this prefilter may return false positives
    /// via the `Candidate::PossibleStartOfMatch` variant. This is most useful
    /// when false positives are not posssible (in which case, implementations
    /// should return false), which may allow completely avoiding heavier regex
    /// machinery when the prefilter can quickly confirm its own matches.
    ///
    /// By default, this returns true, which is conservative; it is always
    /// correct to return `true`. Returning `false` here and reporting a false
    /// positive will result in incorrect searches.
    fn reports_false_positives(&self) -> bool {
        true
    }
}

impl<'a, P: Prefilter + ?Sized> Prefilter for &'a P {
    #[inline]
    fn next_candidate(
        &self,
        state: &mut State,
        haystack: &[u8],
        at: usize,
    ) -> Candidate {
        (**self).next_candidate(state, haystack, at)
    }

    fn heap_bytes(&self) -> usize {
        (**self).heap_bytes()
    }

    fn reports_false_positives(&self) -> bool {
        (**self).reports_false_positives()
    }
}

#[derive(Clone)]
pub struct Scanner<'p> {
    prefilter: &'p dyn Prefilter,
    state: State,
}

impl<'p> Scanner<'p> {
    pub fn new(prefilter: &'p dyn Prefilter) -> Scanner<'p> {
        Scanner { prefilter, state: State::new() }
    }

    pub(crate) fn is_effective(&mut self, at: usize) -> bool {
        self.state.is_effective(at)
    }

    pub(crate) fn reports_false_positives(&self) -> bool {
        self.prefilter.reports_false_positives()
    }

    pub(crate) fn next_candidate(
        &mut self,
        bytes: &[u8],
        at: usize,
    ) -> Candidate {
        let cand = self.prefilter.next_candidate(&mut self.state, bytes, at);
        match cand {
            Candidate::None => {
                self.state.update_skipped_bytes(bytes.len() - at);
            }
            Candidate::Match(ref m) => {
                self.state.update_skipped_bytes(m.start() - at);
            }
            Candidate::PossibleStartOfMatch(i) => {
                self.state.update_skipped_bytes(i - at);
            }
        }
        cand
    }
}

impl<'p> core::fmt::Debug for Scanner<'p> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("Scanner").field("state", &self.state).finish()
    }
}

/// State tracks state associated with the effectiveness of a
/// prefilter. It is used to track how many bytes, on average, are skipped by
/// the prefilter. If this average dips below a certain threshold over time,
/// then the state renders the prefilter inert and stops using it.
///
/// A prefilter state should be created for each search. (Where creating an
/// iterator via, e.g., `find_iter`, is treated as a single search.)
#[derive(Clone, Debug)]
pub struct State {
    /// The number of skips that has been executed.
    skips: usize,
    /// The total number of bytes that have been skipped.
    skipped: usize,
    /// Once this heuristic has been deemed permanently ineffective, it will be
    /// inert throughout the rest of its lifetime. This serves as a cheap way
    /// to check inertness.
    inert: bool,
    /// The last (absolute) position at which a prefilter scanned to.
    /// Prefilters can use this position to determine whether to re-scan or
    /// not.
    ///
    /// Unlike other things that impact effectiveness, this is a fleeting
    /// condition. That is, a prefilter can be considered ineffective if it is
    /// at a position before `last_scan_at`, but can become effective again
    /// once the search moves past `last_scan_at`.
    ///
    /// The utility of this is to both avoid additional overhead from calling
    /// the prefilter and to avoid quadratic behavior. This ensures that a
    /// prefilter will scan any particular byte at most once. (Note that some
    /// prefilters, like the start-byte prefilter, do not need to use this
    /// field at all, since it only looks for starting bytes.)
    last_scan_at: usize,
}

impl State {
    /// The minimum number of skip attempts to try before considering whether
    /// a prefilter is effective or not.
    const MIN_SKIPS: usize = 40;

    /// The minimum amount of bytes that skipping must average.
    ///
    /// That is, after MIN_SKIPS have occurred, if the average number of bytes
    /// skipped ever falls below MIN_AVG_SKIP, then the prefilter will be
    /// rendered inert.
    const MIN_AVG_SKIP: usize = 16;

    /// Create a fresh prefilter state.
    pub fn new() -> State {
        State { skips: 0, skipped: 0, inert: false, last_scan_at: 0 }
    }

    /// Updates the position at which the last scan stopped. This may be
    /// greater than the position of the last candidate reported. For example,
    /// searching for the byte `z` in `abczdef` for the pattern `abcz` will
    /// report a candidate at position `0`, but the end of its last scan will
    /// be at position `3`.
    ///
    /// This position factors into the effectiveness of this prefilter. If the
    /// current position is less than the last position at which a scan ended,
    /// then the prefilter should not be re-run until the search moves past
    /// that position.
    ///
    /// It is always correct to never update the last scan position. In fact,
    /// it is also always correct to set the last scan position to an arbitrary
    /// value. The key is setting it to a position in the future at which it
    /// makes sense to restart the prefilter.
    pub fn update_last_scan(&mut self, at: usize) {
        if at > self.last_scan_at {
            self.last_scan_at = at;
        }
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
    fn is_effective(&mut self, at: usize) -> bool {
        if self.inert {
            return false;
        }
        if at < self.last_scan_at {
            return false;
        }
        if self.skips < State::MIN_SKIPS {
            return true;
        }

        if self.skipped >= State::MIN_AVG_SKIP * self.skips {
            return true;
        }

        // We're inert.
        self.inert = true;
        false
    }

    /// Update this state with the number of bytes skipped on the last
    /// invocation of the prefilter.
    fn update_skipped_bytes(&mut self, skipped: usize) {
        self.skips += 1;
        self.skipped += skipped;
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
    fn next_candidate(&self, _: &mut State, _: &[u8], at: usize) -> Candidate {
        Candidate::PossibleStartOfMatch(at)
    }

    fn heap_bytes(&self) -> usize {
        0
    }
}
