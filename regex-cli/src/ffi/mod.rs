/*!
This module provides safe abstractions over an FFI layer to other regex
engines. None of these are built by default when compiling regex-cli. Instead,
each regex engine must be enabled via the corresponding Cargo feature.
*/

#[cfg(feature = "extre-pcre2")]
pub mod pcre2;
#[cfg(feature = "extre-re2")]
pub mod re2;
