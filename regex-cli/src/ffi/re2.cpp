/*
This is a shim layer that effectively exposes a small portion of RE2's C++ API
through a C API. We do our best to keep the layer as thin as possible. For
example, sneaking in an extra allocation for every search call would be very
inappropriate.

Note that CRE2[1] is a thing, but unless it's otherwise impractical, we choose
to own everything between a regex engine and the code that uses it. This is
especially important for the benchmarking harness. A third party binding
layer might do something under our noses that would be bad for benchmarking
for example, and it might be hard to know about it. For example, CRE2 does
allocate[2] in every 'cre2_match' call. Of course, bringing in a third party
binding layer also has complexities of our own. Since we only need a small
portion of a regex engine's public API, we can reasonably justifying hand
rolling it.

[1]: https://github.com/marcomaggi/cre2
[2]: https://github.com/marcomaggi/cre2/blob/6687e7eee83189ddc2b226e7c58adb360b468492/src/cre2.cpp#L278
*/

#include <iostream>
#include <stdio.h>

#include "re2/re2.h"

using namespace re2;

// An example that sets RE2 options. We should expose Latin1 and case
// insensitivty, as those options are tweakable in the benchmark suite.
// TEST(RE2, Bug18391750) {
  // // Stray write past end of match_ in nfa.cc, caught by fuzzing + address sanitizer.
  // const char t[] = {
      // (char)0x28, (char)0x28, (char)0xfc, (char)0xfc, (char)0x08, (char)0x08,
      // (char)0x26, (char)0x26, (char)0x28, (char)0xc2, (char)0x9b, (char)0xc5,
      // (char)0xc5, (char)0xd4, (char)0x8f, (char)0x8f, (char)0x69, (char)0x69,
      // (char)0xe7, (char)0x29, (char)0x7b, (char)0x37, (char)0x31, (char)0x31,
      // (char)0x7d, (char)0xae, (char)0x7c, (char)0x7c, (char)0xf3, (char)0x29,
      // (char)0xae, (char)0xae, (char)0x2e, (char)0x2a, (char)0x29, (char)0x00,
  // };
  // RE2::Options opt;
  // opt.set_encoding(RE2::Options::EncodingLatin1);
  // opt.set_longest_match(true);
  // opt.set_dot_nl(true);
  // opt.set_case_sensitive(false);
  // RE2 re(t, opt);
  // ASSERT_TRUE(re.ok());
  // RE2::PartialMatch(t, re);
// }

extern "C" {
    // An opaque type representing an RE2 regex. Internally, this is a RE2*.
    typedef void re2_regexp;

    // An opaque type representing a sequence of RE2 StringPieces. Internally,
    // this is a std::vector<re2::StringPiece>*.
    //
    // This exists because RE2's 'Match' API wants the C++ StringPiece type
    // in order to record submatches. We could of course just allocate our
    // StringPieces at search time, but then this prevents the caller from
    // amortizing allocation. So 're2_captures' represents this allocation and
    // may be reused for calls to 're2_regexp_captures'.
    typedef void re2_captures;

    // A transparent type representing a string and its length. We frequently
    // convert between this and an RE2 StringPiece (which effectively has the
    // same representation, but I don't believe can be legally casted to this
    // type, so the conversion must be explicit).
    //
    // This also maps very cleanly to a Rust &[u8], which is also just a
    // pointer and a length.
    typedef struct re2_string {
        const char *data;
        int length;
    } re2_string;

    // Create a new RE2 regexp. If one could not be created, return a null
    // pointer.
    re2_regexp* re2_regexp_new(re2_string pat) {
        try {
            re2::StringPiece re2_pat(pat.data, pat.length);
            return reinterpret_cast<re2_regexp*>(new RE2(re2_pat));
        } catch (...) {
            return nullptr;
        }
    }

    // Free the given regexp.
    void re2_regexp_free(re2_regexp *re) {
        delete reinterpret_cast<RE2*>(re);
    }

    // Check whether the given haystack at the given positions matches the
    // given regex. Use this when all you care about is "match or not."
    bool re2_regexp_is_match(
        re2_regexp *re,
        re2_string haystack,
        int startpos,
        int endpos
    ) {
        RE2 *re2_re = reinterpret_cast<RE2*>(re);
        re2::StringPiece re2_haystack(haystack.data, haystack.length);

        // This is a somewhat degenerate case that RE2 errors on, but
        // regex-automata regex engines specifically handle. We don't
        // technically have to do the same here, but the iterator helpers in
        // regex-automata rely on the regex engine handling this case, so we do
        // it here for RE2.
        if (startpos > endpos) {
            return false;
        }
        return re2_re->Match(
            re2_haystack,
            startpos,
            endpos,
            RE2::UNANCHORED,
            NULL,
            0
        );
    }

    // Check whether a match exists in the haystack in the given sub-range, and
    // if so, write the start and end offsets of that match into 'match_start'
    // and 'match_end'. Use this when all you care about is the overall match
    // position.
    bool re2_regexp_find(
        re2_regexp *re,
        re2_string haystack,
        int startpos,
        int endpos,
        int *match_start,
        int *match_end
    ) {
        RE2 *re2_re = reinterpret_cast<RE2*>(re);
        re2::StringPiece re2_haystack(haystack.data, haystack.length);
        re2::StringPiece re2_submatch;
        bool matched;

        // See comments in re2_regexp_is_match for why we do this.
        if (startpos > endpos) {
            return false;
        }
        matched = re2_re->Match(
            re2_haystack,
            startpos,
            endpos,
            RE2::UNANCHORED,
            &re2_submatch,
            1
        );
        if (matched) {
            *match_start = re2_submatch.data() - re2_haystack.data();
            *match_end = *match_start + re2_submatch.length();
        }
        return matched;
    }

    // Check whether a match exists in the haystack in the given sub-range,
    // and if so, write the start and end offsets of each capturing group
    // (including the implicit unnamed group representing the overall match) to
    // the given 'caps' value.
    //
    // When creating a 'caps' value, you'll usually want to create it with
    // 'nsubmatch=1+re2_regexp_capture_len(re)', since the capture length
    // reported by RE2 doesn't include the implicit unnamed group.
    bool re2_regexp_captures(
        re2_regexp *re,
        re2_string haystack,
        int startpos,
        int endpos,
        re2_captures *caps
    ) {
        RE2 *re2_re = reinterpret_cast<RE2*>(re);
        std::vector<re2::StringPiece> *vec =
            reinterpret_cast<std::vector<re2::StringPiece>*>(caps);
        re2::StringPiece re2_haystack(haystack.data, haystack.length);
        bool matched;

        // See comments in re2_regexp_is_match for why we do this.
        if (startpos > endpos) {
            return false;
        }
        matched = re2_re->Match(
            re2_haystack,
            startpos,
            endpos,
            RE2::UNANCHORED,
            vec->data(),
            vec->size()
        );
        return matched;
    }

    // Returns the total number of capturing groups in the given regex. Note
    // that this only includes explicit capturing groups, and doesn't include
    // the implicit group corresponding to the overall match.
    int re2_regexp_capture_len(re2_regexp *re) {
        RE2 *re2_re = reinterpret_cast<RE2*>(re);
        return re2_re->NumberOfCapturingGroups();
    }

    // Create a new set of capturing groups with room for 'nsubmatch' groups.
    //
    // You'll usually want to call this with
    // 'nsubmatch=1+re2_regexp_capture_len(re)', since the capture length
    // reported by RE2 doesn't include the implicit unnamed group.
    //
    // The captures returned may be reused for multiple search calls.
    re2_captures* re2_captures_new(int nsubmatch) {
        return reinterpret_cast<re2_captures*>(
            new std::vector<re2::StringPiece>(nsubmatch)
        );
    }

    // Free the given capturing groups.
    void re2_captures_free(re2_captures *caps) {
        delete reinterpret_cast<std::vector<re2::StringPiece>*>(caps);
    }

    // Return the substring of the haystack that matched the given capturing
    // group. If the group at the given index didn't participate in the match,
    // then the 'data' pointer in the string returned is NULL. If the index
    // given is invalid, then this aborts the process.
    re2_string re2_captures_get(re2_captures *caps, int index) {
        std::vector<re2::StringPiece> *vec =
            reinterpret_cast<std::vector<re2::StringPiece>*>(caps);
        if ((long unsigned int)index >= vec->size()) {
            abort();
        }
        re2_string submatch;
        submatch.data = vec->at(index).data();
        submatch.length = vec->at(index).length();
        return submatch;
    }
}
