## Possible future work

* The only statistics gathered are mean (with standard deviation), median,
minimum and maximum. Arguably, we should support confidence intervals and
perhaps percentiles as well.
* While we do initially collect benchmark iteration duration internally,
externally, the only units we support are time and throughput. We likely also
want to support instruction counts too. Wall clock time is still the ultimate
thing we are interested in, but instruction counts can be quite beneficial as
an additional view on measurements to help understand whether differences over
time are a result of noise, or if something about the underlying codegen has
changed.
* We should find a way to include Hyperscan in this benchmark. It's an annoying
dependency (about half the times I try to use it I have build problems), but
it's also likely to do very well on many of these benchmarks and so is an
important and relevant addition to the suite. I started with PCRE2 and RE2
because they're ubiquitous regex engines. RE2 in particular uses very similar
techniques as the regex crate, and so it's a useful "standard candle" in the
space of regex engines. PCRE2, on the other hand, is a super optimized
backtracking regex engine. It's very useful to compare and contrast the
different approaches.
