#[macro_use]
extern crate lazy_static;
extern crate regex_automata;
extern crate serde;
extern crate serde_bytes;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;

mod fowler;
mod collection;
mod matching;
mod suite;
mod unescape;
