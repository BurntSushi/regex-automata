use regex_automata::{DenseDFA, Regex, RegexBuilder, SparseDFA, DFA};
use regex_automata::dense;

use collection::{RegexTester, SUITE};

#[test]
fn unminimized_standard() {
    let mut builder = RegexBuilder::new();
    builder.minimize(false).premultiply(false).byte_classes(false);

    let mut tester = RegexTester::new().skip_expensive();
    tester.test_all(builder, SUITE.tests());
    tester.assert();
}

#[test]
fn unminimized_premultiply() {
    let mut builder = RegexBuilder::new();
    builder.minimize(false).premultiply(true).byte_classes(false);

    let mut tester = RegexTester::new().skip_expensive();
    tester.test_all(builder, SUITE.tests());
    tester.assert();
}

#[test]
fn unminimized_byte_class() {
    let mut builder = RegexBuilder::new();
    builder.minimize(false).premultiply(false).byte_classes(true);

    let mut tester = RegexTester::new();
    tester.test_all(builder, SUITE.tests());
    tester.assert();
}

#[test]
fn unminimized_premultiply_byte_class() {
    let mut builder = RegexBuilder::new();
    builder.minimize(false).premultiply(true).byte_classes(true);

    let mut tester = RegexTester::new();
    tester.test_all(builder, SUITE.tests());
    tester.assert();
}

#[test]
fn minimized_standard() {
    let mut builder = RegexBuilder::new();
    builder.minimize(true).premultiply(false).byte_classes(false);

    let mut tester = RegexTester::new().skip_expensive();
    tester.test_all(builder, SUITE.tests());
    tester.assert();
}

#[test]
fn minimized_premultiply() {
    let mut builder = RegexBuilder::new();
    builder.minimize(true).premultiply(true).byte_classes(false);

    let mut tester = RegexTester::new().skip_expensive();
    tester.test_all(builder, SUITE.tests());
    tester.assert();
}

#[test]
fn minimized_byte_class() {
    let mut builder = RegexBuilder::new();
    builder.minimize(true).premultiply(false).byte_classes(true);

    let mut tester = RegexTester::new();
    tester.test_all(builder, SUITE.tests());
    tester.assert();
}

#[test]
fn minimized_premultiply_byte_class() {
    let mut builder = RegexBuilder::new();
    builder.minimize(true).premultiply(true).byte_classes(true);

    let mut tester = RegexTester::new();
    tester.test_all(builder, SUITE.tests());
    tester.assert();
}

// A basic sanity test that checks we can convert a regex to a smaller
// representation and that the resulting regex still passes our tests.
//
// If tests grow minimal regexes that cannot be represented in 16 bits, then
// we'll either want to skip those or increase the size to test to u32.
#[test]
fn u16() {
    let mut builder = RegexBuilder::new();
    builder.minimize(true).premultiply(false).byte_classes(true);

    let mut tester = RegexTester::new().skip_expensive();
    for test in SUITE.tests() {
        let builder = builder.clone();
        let re: Regex = match tester.build_regex(builder, test) {
            None => continue,
            Some(re) => re,
        };
        let small_re = Regex::from_dfas(
            re.forward().to_u16().unwrap(),
            re.reverse().to_u16().unwrap(),
        );

        tester.test(test, &small_re);
    }
    tester.assert();
}

// Test that sparse DFAs work using the standard configuration.
#[test]
fn sparse_unminimized_standard() {
    let mut builder = RegexBuilder::new();
    builder.minimize(false).premultiply(false).byte_classes(false);

    let mut tester = RegexTester::new().skip_expensive();
    for test in SUITE.tests() {
        let builder = builder.clone();
        let re: Regex = match tester.build_regex(builder, test) {
            None => continue,
            Some(re) => re,
        };
        let fwd = re.forward().to_sparse().unwrap();
        let rev = re.reverse().to_sparse().unwrap();
        let sparse_re = Regex::from_dfas(fwd, rev);

        tester.test(test, &sparse_re);
    }
    tester.assert();
}

// Test that sparse DFAs work after converting them to a different state ID
// representation.
#[test]
fn sparse_u16() {
    let mut builder = RegexBuilder::new();
    builder.minimize(true).premultiply(false).byte_classes(false);

    let mut tester = RegexTester::new().skip_expensive();
    for test in SUITE.tests() {
        let builder = builder.clone();
        let re: Regex = match tester.build_regex(builder, test) {
            None => continue,
            Some(re) => re,
        };
        let fwd = re.forward().to_sparse().unwrap().to_u16().unwrap();
        let rev = re.reverse().to_sparse().unwrap().to_u16().unwrap();
        let sparse_re = Regex::from_dfas(fwd, rev);

        tester.test(test, &sparse_re);
    }
    tester.assert();
}

// Another basic sanity test that checks we can serialize and then deserialize
// a regex, and that the resulting regex can be used for searching correctly.
#[test]
fn serialization_roundtrip() {
    let mut builder = RegexBuilder::new();
    builder.premultiply(false).byte_classes(true);

    let mut tester = RegexTester::new().skip_expensive();
    for test in SUITE.tests() {
        let builder = builder.clone();
        let re: Regex = match tester.build_regex(builder, test) {
            None => continue,
            Some(re) => re,
        };

        let fwd_bytes = re.forward().to_bytes_native_endian().unwrap();
        let rev_bytes = re.reverse().to_bytes_native_endian().unwrap();
        let fwd: DenseDFA<&[usize], usize> =
            unsafe { DenseDFA::from_bytes(&fwd_bytes) };
        let rev: DenseDFA<&[usize], usize> =
            unsafe { DenseDFA::from_bytes(&rev_bytes) };
        let re = Regex::from_dfas(fwd, rev);

        tester.test(test, &re);
    }
    tester.assert();
}

// A basic sanity test that checks we can serialize and then deserialize a
// regex using sparse DFAs, and that the resulting regex can be used for
// searching correctly.
#[test]
fn sparse_serialization_roundtrip() {
    let mut builder = RegexBuilder::new();
    builder.byte_classes(true);

    let mut tester = RegexTester::new().skip_expensive();
    for test in SUITE.tests() {
        let builder = builder.clone();
        let re: Regex = match tester.build_regex(builder, test) {
            None => continue,
            Some(re) => re,
        };

        let fwd_bytes = re
            .forward()
            .to_sparse()
            .unwrap()
            .to_bytes_native_endian()
            .unwrap();
        let rev_bytes = re
            .reverse()
            .to_sparse()
            .unwrap()
            .to_bytes_native_endian()
            .unwrap();
        let fwd: SparseDFA<&[u8], usize> =
            unsafe { SparseDFA::from_bytes(&fwd_bytes) };
        let rev: SparseDFA<&[u8], usize> =
            unsafe { SparseDFA::from_bytes(&rev_bytes) };
        let re = Regex::from_dfas(fwd, rev);

        tester.test(test, &re);
    }
    tester.assert();
}

#[test]
fn multi_is_match() {
    let mut builder = dense::Builder::new();
    
    // A single matching pattern
    let dfa = builder.build_multi_with_size::<usize>(vec!["a"]).unwrap();
    assert!(dfa.is_match(b"a"));
    
    // A single non-matching pattern
    let dfa = builder.build_multi_with_size::<usize>(vec!["b"]).unwrap();
    assert!(!dfa.is_match(b"a"));
    
    // A single matching pattern not at start
    let dfa = builder.build_multi_with_size::<usize>(vec!["a"]).unwrap();
    assert!(dfa.is_match(b"ba"));
    
    // Multiple matching patterns
    let dfa = builder.build_multi_with_size::<usize>(vec!["a", "b"]).unwrap();
    assert!(dfa.is_match(b"ab"));
    
    // First pattern matches
    let dfa = builder.build_multi_with_size::<usize>(vec!["a", "b"]).unwrap();
    assert!(dfa.is_match(b"a"));
    
    // Second pattern matches
    let dfa = builder.build_multi_with_size::<usize>(vec!["a", "b"]).unwrap();
    assert!(dfa.is_match(b"b"));
    
    // Neither pattern matches
    let dfa = builder.build_multi_with_size::<usize>(vec!["a", "b"]).unwrap();
    assert!(!dfa.is_match(b"c"));
    
    // Determinization edge case
    builder.anchored(true);
    let dfa = builder.build_multi_with_size::<usize>(vec!["ab", "ac"]).unwrap();
    assert!(dfa.is_match(b"ac"));
    
    // Anchored respected
    let dfa = builder.build_multi_with_size::<usize>(vec!["ab", "ac"]).unwrap();
    assert!(!dfa.is_match(b"bac"));
}

#[test]
fn multi_shortest_match() {
    let mut builder = dense::Builder::new();
    
    // A single matching pattern
    let dfa = builder.build_multi_with_size::<usize>(vec!["a"]).unwrap();
    assert_eq!(Some(1), dfa.shortest_match(b"a"));
    
    // A single non-matching pattern
    let dfa = builder.build_multi_with_size::<usize>(vec!["b"]).unwrap();
    assert_eq!(None, dfa.shortest_match(b"a"));
    
    // A single matching pattern not at start
    let dfa = builder.build_multi_with_size::<usize>(vec!["a"]).unwrap();
    assert_eq!(Some(2), dfa.shortest_match(b"ba"));
    
    // Multiple matching patterns
    let dfa = builder.build_multi_with_size::<usize>(vec!["a", "b"]).unwrap();
    assert_eq!(Some(4), dfa.shortest_match(b"dddadddb"));
    
    // First pattern matches
    let dfa = builder.build_multi_with_size::<usize>(vec!["a", "b"]).unwrap();
    assert_eq!(Some(1), dfa.shortest_match(b"a"));
    
    // Second pattern matches
    let dfa = builder.build_multi_with_size::<usize>(vec!["a", "b"]).unwrap();
    assert_eq!(Some(1), dfa.shortest_match(b"b"));
    
    // Neither pattern matches
    let dfa = builder.build_multi_with_size::<usize>(vec!["a", "b"]).unwrap();
    assert_eq!(None, dfa.shortest_match(b"c"));
    
    // Determinization edge case
    builder.anchored(true);
    let dfa = builder.build_multi_with_size::<usize>(vec!["ab", "ac"]).unwrap();
    assert_eq!(Some(2), dfa.shortest_match(b"ac"));
    
    // Anchored respected
    let dfa = builder.build_multi_with_size::<usize>(vec!["ab", "ac"]).unwrap();
    assert_eq!(None, dfa.shortest_match(b"bac"));
}

#[test]
fn multi_find() {
    let mut builder = dense::Builder::new();
    
    // A single matching pattern
    let dfa = builder.build_multi_with_size::<usize>(vec!["a"]).unwrap();
    assert_eq!(Some(1), dfa.find(b"a"));
    
    // A single non-matching pattern
    let dfa = builder.build_multi_with_size::<usize>(vec!["b"]).unwrap();
    assert_eq!(None, dfa.find(b"a"));
    
    // A single matching pattern not at start
    let dfa = builder.build_multi_with_size::<usize>(vec!["a"]).unwrap();
    assert_eq!(Some(2), dfa.find(b"ba"));
    
    // Same pattern multiple times
    let dfa = builder.build_multi_with_size::<usize>(vec!["a"]).unwrap();
    assert_eq!(Some(2), dfa.find(b"bababa"));
    
    // Multiple matching patterns
    let dfa = builder.build_multi_with_size::<usize>(vec!["a", "b"]).unwrap();
    assert_eq!(Some(8), dfa.find(b"dddadddb"));
    
    // First pattern matches
    let dfa = builder.build_multi_with_size::<usize>(vec!["a", "b"]).unwrap();
    assert_eq!(Some(1), dfa.find(b"a"));
    
    // Second pattern matches
    let dfa = builder.build_multi_with_size::<usize>(vec!["a", "b"]).unwrap();
    assert_eq!(Some(1), dfa.find(b"b"));
    
    // Neither pattern matches
    let dfa = builder.build_multi_with_size::<usize>(vec!["a", "b"]).unwrap();
    assert_eq!(None, dfa.find(b"c"));
    
    // Determinization edge case
    builder.anchored(true);
    let dfa = builder.build_multi_with_size::<usize>(vec!["ab", "ac"]).unwrap();
    assert_eq!(Some(2), dfa.find(b"ac"));
    
    // Anchored respected
    let dfa = builder.build_multi_with_size::<usize>(vec!["ab", "ac"]).unwrap();
    assert_eq!(None, dfa.find(b"bac"));
}

#[test]
fn multi_rfind() {
    let mut builder = dense::Builder::new();
    builder.reverse(true);
    
    // A single matching pattern
    let dfa = builder.build_multi_with_size::<usize>(vec!["a"]).unwrap();
    assert_eq!(Some(0), dfa.rfind(b"a"));
    
    // A single non-matching pattern
    let dfa = builder.build_multi_with_size::<usize>(vec!["b"]).unwrap();
    assert_eq!(None, dfa.rfind(b"a"));
    
    // A single matching pattern not at start
    let dfa = builder.build_multi_with_size::<usize>(vec!["a"]).unwrap();
    assert_eq!(Some(1), dfa.rfind(b"ba"));
    
    // Multiple matching patterns
    let dfa = builder.build_multi_with_size::<usize>(vec!["a", "b"]).unwrap();
    assert_eq!(Some(3), dfa.rfind(b"dddadddb"));
    
    // First pattern matches
    let dfa = builder.build_multi_with_size::<usize>(vec!["a", "b"]).unwrap();
    assert_eq!(Some(0), dfa.rfind(b"a"));
    
    // Second pattern matches
    let dfa = builder.build_multi_with_size::<usize>(vec!["a", "b"]).unwrap();
    assert_eq!(Some(0), dfa.rfind(b"b"));
    
    // Neither pattern matches
    let dfa = builder.build_multi_with_size::<usize>(vec!["a", "b"]).unwrap();
    assert_eq!(None, dfa.rfind(b"c"));
    
    // Determinization edge case
    builder.anchored(true);
    let dfa = builder.build_multi_with_size::<usize>(vec!["ab", "ac"]).unwrap();
    assert_eq!(Some(0), dfa.rfind(b"ac"));
    
    // Anchored respected
    let dfa = builder.build_multi_with_size::<usize>(vec!["ab", "ac"]).unwrap();
    assert_eq!(None, dfa.rfind(b"acb"));
}

#[test]
fn overlapping_find_at() {
    let mut builder = dense::Builder::new();
    builder.reverse(true);
    
    // A single matching pattern
//    let dfa = builder.build_multi_with_size::<usize>(vec!["a"]).unwrap();
//    let mut state = dfa.start_state();
//    let mut match_index = 0;
//    let (input_idx, match_idx) = dfa.overlapping_find_at(b"a", 0, &mut state, &mut match_index).unwrap();
//    assert_eq!((1, 0), (input_idx, match_idx));
//    assert_eq!(None, dfa.overlapping_find_at(b"a", input_idx, &mut state, &mut match_index));
//    
//    // A single non-matching pattern
//    let dfa = builder.build_multi_with_size::<usize>(vec!["b"]).unwrap();
//    let mut state = dfa.start_state();
//    let mut match_index = 0;
//    assert_eq!(None, dfa.overlapping_find_at(b"a", 0, &mut state, &mut match_index));
//    
//    // A single matching pattern not at start
//    let dfa = builder.build_multi_with_size::<usize>(vec!["a"]).unwrap();
//    let mut state = dfa.start_state();
//    let mut match_index = 0;
//    let (input_idx, match_idx) = dfa.overlapping_find_at(b"ba", 0, &mut state, &mut match_index).unwrap();
//    assert_eq!((2, 0), (input_idx, match_idx));
//    assert_eq!(None, dfa.overlapping_find_at(b"ba", input_idx, &mut state, &mut match_index));
//    
//    // Same pattern multiple times
//    let dfa = builder.build_multi_with_size::<usize>(vec!["a"]).unwrap();
//    let mut state = dfa.start_state();
//    let mut match_index = 0;
//    let input_idx = 0;
//    let (input_idx, match_idx) = dfa.overlapping_find_at(b"bababa", input_idx, &mut state, &mut match_index).unwrap();
//    assert_eq!((2, 0), (input_idx, match_idx));
//    let (input_idx, match_idx) = dfa.overlapping_find_at(b"bababa", input_idx, &mut state, &mut match_index).unwrap();
//    assert_eq!((4, 0), (input_idx, match_idx));
//    let (input_idx, match_idx) = dfa.overlapping_find_at(b"bababa", input_idx, &mut state, &mut match_index).unwrap();
//    assert_eq!((6, 0), (input_idx, match_idx));
//    assert_eq!(None, dfa.overlapping_find_at(b"bababa", input_idx, &mut state, &mut match_index));
//    
//    // multiple matching patterns
//    let dfa = builder.build_multi_with_size::<usize>(vec!["a", "b"]).unwrap();
//    let mut state = dfa.start_state();
//    let mut match_index = 0;
//    let input_idx = 0;
//    let (input_idx, match_idx) = dfa.overlapping_find_at(b"dddadddb", input_idx, &mut state, &mut match_index).unwrap();
//    assert_eq!((4, 0), (input_idx, match_idx));
//    let (input_idx, match_idx) = dfa.overlapping_find_at(b"dddadddb", input_idx, &mut state, &mut match_index).unwrap();
//    assert_eq!((8, 1), (input_idx, match_idx));
//    assert_eq!(None, dfa.overlapping_find_at(b"dddadddb", input_idx, &mut state, &mut match_index));
//    
//    // First pattern matches
//    let dfa = builder.build_multi_with_size::<usize>(vec!["a", "b"]).unwrap();
//    let mut state = dfa.start_state();
//    let mut match_index = 0;
//    let input_idx = 0;
//    let (input_idx, match_idx) = dfa.overlapping_find_at(b"a", input_idx, &mut state, &mut match_index).unwrap();
//    assert_eq!((1, 0), (input_idx, match_idx));
//    assert_eq!(None, dfa.overlapping_find_at(b"a", input_idx, &mut state, &mut match_index));
//    
//    // Second pattern matches
//    let dfa = builder.build_multi_with_size::<usize>(vec!["a", "b"]).unwrap();
//    let mut state = dfa.start_state();
//    let mut match_index = 0;
//    let input_idx = 0;
//    let (input_idx, match_idx) = dfa.overlapping_find_at(b"b", input_idx, &mut state, &mut match_index).unwrap();
//    assert_eq!((1, 1), (input_idx, match_idx));
//    assert_eq!(None, dfa.overlapping_find_at(b"b", input_idx, &mut state, &mut match_index));
//    
//    // Neither pattern matches
//    let dfa = builder.build_multi_with_size::<usize>(vec!["a", "b"]).unwrap();
//    let mut state = dfa.start_state();
//    let mut match_index = 0;
//    let input_idx = 0;
//    assert_eq!(None, dfa.overlapping_find_at(b"c", input_idx, &mut state, &mut match_index));
    
    // Legit overlap
    builder.premultiply(false);
    let dfa = builder.build_multi_with_size::<usize>(vec!["ab", "bc"]).unwrap();
    dbg!(&dfa);
    dfa.dbg();
    let mut state = dfa.start_state();
    let mut match_index = 0;
    let input_idx = 0;
    let (input_idx, match_idx) = dfa.overlapping_find_at(b"abcabc", input_idx, &mut state, &mut match_index).unwrap();
    assert_eq!((2, 0), (input_idx, match_idx));
    let (input_idx, match_idx) = dfa.overlapping_find_at(b"abcabc", input_idx, &mut state, &mut match_index).unwrap();
    assert_eq!((3, 1), (input_idx, match_idx));
    let (input_idx, match_idx) = dfa.overlapping_find_at(b"abcabc", input_idx, &mut state, &mut match_index).unwrap();
    assert_eq!((3, 2), (input_idx, match_idx));
    let (input_idx, match_idx) = dfa.overlapping_find_at(b"abcabc", input_idx, &mut state, &mut match_index).unwrap();
    assert_eq!((5, 0), (input_idx, match_idx));
    let (input_idx, match_idx) = dfa.overlapping_find_at(b"abcabc", input_idx, &mut state, &mut match_index).unwrap();
    assert_eq!((6, 1), (input_idx, match_idx));
    let (input_idx, match_idx) = dfa.overlapping_find_at(b"abcabc", input_idx, &mut state, &mut match_index).unwrap();
    assert_eq!((6, 2), (input_idx, match_idx));
    assert_eq!(None, dfa.overlapping_find_at(b"abcabc", input_idx, &mut state, &mut match_index));
    
}
