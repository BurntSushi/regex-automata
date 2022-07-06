fn main() {
    #[cfg(feature = "extre-pcre2")]
    if std::env::var("CARGO_FEATURE_EXTRE_PCRE2").is_ok() {
        pkg_config::probe_library("libpcre2-8").unwrap();
    }
    #[cfg(feature = "extre-re2")]
    if std::env::var("CARGO_FEATURE_EXTRE_RE2").is_ok() {
        // RE2 is a C++ library, so we need to compile our shim layer.
        cc::Build::new()
            .cpp(true)
            .debug(true)
            .file("src/ffi/re2.cpp")
            .compile("libcre2.a");
        // If our shim layer changes, make sure Cargo sees it.
        println!("cargo:rerun-if-changed=src/ffi/re2.cpp");
        // It's important this comes after compiling the shim, which results
        // in the correct order of arguments given to the linker.
        pkg_config::probe_library("re2").unwrap();
    }
}
