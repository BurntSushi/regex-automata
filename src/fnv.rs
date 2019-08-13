// Basic FNV-1a hash as described:
// https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
//
// This improves benchmarks that compile large Unicode character classes, since
// it makes the generation of the minimal UTF-8 automaton faster. Specifically,
// one can observe the difference with std's hashmap via something like the
// following benchmark:
//
//   hyperfine "regex-automata-debug debug -acqr '\w{40} ecurB'"

use std::collections::HashMap as StdHashMap;
use std::hash::{self, BuildHasherDefault};

const PRIME: u64 = 1099511628211;
const INIT: u64 = 14695981039346656037;

pub type HashMap<K, V> = StdHashMap<K, V, BuildHasherDefault<Hasher>>;

#[derive(Debug)]
pub struct Hasher(u64);

impl Default for Hasher {
    fn default() -> Hasher {
        Hasher(INIT)
    }
}

impl hash::Hasher for Hasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes.iter() {
            self.0 = self.0 ^ (*byte as u64);
            self.0 = self.0.wrapping_mul(PRIME);
        }
    }
}
