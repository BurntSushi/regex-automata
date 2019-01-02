* Look into being smarter about generating NFA states for large Unicode
  character classes. These can create a lot of additional work for both the
  determinizer and the minimizer, and I suspect this is the key thing we'll
  want to improve if we want to make DFA compile times faster. I *believe*
  it's possible to potentially build minimal or nearly minimal NFAs for the
  special case of Unicode character classes by leveraging Daciuk's algorithms
  for building minimal automata in linear time for sets of strings. See
  https://blog.burntsushi.net/transducers/#construction for more details. The
  key adaptation I think we need to make is to modify the algorithm to operate
  on byte ranges instead of enumerating every codepoint in the set. Otherwise,
  it might not be worth doing.
* Add support for regex sets. It should be possible to do this by "simply"
  introducing more match states. I think we can also report the positions at
  each match, similar to how Aho-Corasick works. I think the long pole in the
  tent here is probably the API design work and arranging it so that we don't
  introduce extra overhead into the non-regex-set case without duplicating a
  lot of code. It seems doable.
* Stretch goal: support capturing groups by implementing "tagged" DFA
  (transducers). Laurikari's paper is the usual reference here, but Trofimovich
  has a much more thorough treatment here:
  http://re2c.org/2017_trofimovich_tagged_deterministic_finite_automata_with_lookahead.pdf
  I've only read the paper once. I suspect it will require at least a few more
  read throughs before I understand it.
  See also: http://re2c.org/
* Possibly less ambitious goal: can we select a portion of Trofimovich's work
  to make small fixed length look-around work? It would be really nice to
  support ^, $ and \b, especially the Unicode variant of \b and CRLF aware $.
* Experiment with code generating Rust code. There is an early experiment in
  src/codegen.rs that is thoroughly bit-rotted. At the time, I was
  experimenting with whether or not codegen would significant decrease the size
  of a DFA, since if you squint hard enough, it's kind of like a sparse
  representation. However, it didn't shrink as much as I thought it would, so
  I gave up. The other problem is that Rust doesn't support gotos, so I don't
  even know whether the "match on each state" in a loop thing will be fast
  enough. Either way, it's probably a good option to have. For one thing, it
  would be endian independent where as the serialization format of the DFAs in
  this crate are endian dependent (so you need two versions of every DFA, but
  you only need to compile one of them for any given arch).
* Experiment with unrolling the match loops.
* Add some kind of streaming API. I believe users of the library can already
  implement something for this outside of the crate, but it would be good to
  provide an official API. The key thing here is figuring out the API. I
  suspect we might want to support several variants.
* Make a decision on whether or not there is room for literal optimizations
  in this crate. My original intent was to not let this crate sink down into
  that very very very deep rabbit hole. But instead, we might want to provide
  some way for literal optimizations to hook into the match routines. The right
  path forward here is to probably build something outside of the crate and
  then see about integrating it. After all, users can implement their own
  match routines just as efficiently as what the crate provides.
* A key downside of DFAs is that they can take up a lot of memory and can be
  quite costly to build. Their worst case compilation time is O(2^n), where
  n is the number of NFA states. A paper by Yang and Prasanna (2011) actually
  seems to provide a way to character state blow up such that it is detectable.
  If we could know whether a regex will exhibit state explosion or not, then
  we could make an intelligent decision about whether to ahead-of-time compile
  a DFA.
  See: https://www.researchgate.net/profile/XU_Shutu/publication/229032602_Characterization_of_a_global_germplasm_collection_and_its_potential_utilization_for_analysis_of_complex_quantitative_traits_in_maize/links/02bfe50f914d04c837000000.pdf
