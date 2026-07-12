fn main() {
    // Only relevant when linking the real engine. `CACTUS_LIB_DIR` should point
    // at the directory containing `libcactus_engine.a` (produced by
    // `cactus-engine/build.sh` in the Cactus repo).
    println!("cargo:rerun-if-env-changed=CACTUS_LIB_DIR");
    if std::env::var("CARGO_FEATURE_LINK_CACTUS").is_ok() {
        if let Ok(dir) = std::env::var("CACTUS_LIB_DIR") {
            println!("cargo:rustc-link-search=native={dir}");
        }
    }
}
