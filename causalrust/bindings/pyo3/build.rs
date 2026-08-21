//! Link configuration for the Python extension module.
//!
//! A Python extension has unresolved symbols by design: `_PyExc_ValueError` and
//! the rest of the CPython API are supplied by the interpreter when the module
//! is loaded, not by the linker when it is built. Linux tolerates that and links
//! the `cdylib` anyway; macOS refuses, so `cargo build --workspace` fails there
//! and nowhere else.
//!
//! `-undefined dynamic_lookup` tells the macOS linker to defer them, which is
//! what maturin passes when it builds a wheel.
//!
//! # Why this is a build script and not `.cargo/config.toml`
//!
//! It was `.cargo/config.toml` first, and it silently did nothing. The
//! `RUSTFLAGS` environment variable — which this project's CI sets to
//! `-D warnings` — **replaces** `target.*.rustflags` from config rather than
//! merging with it, so the link args never reached the linker and macOS kept
//! failing with the same undefined symbols as before the "fix".
//!
//! `cargo::rustc-link-arg` from a build script is not subject to that
//! precedence, and it is scoped to this package rather than to every crate
//! built for the target.

fn main() {
    println!("cargo::rerun-if-changed=build.rs");

    // `CARGO_CFG_TARGET_OS` is the *target* OS, which is what matters when
    // cross-compiling. `cfg!(target_os = "macos")` inside a build script
    // reports the host and would be wrong.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo::rustc-link-arg=-undefined");
        println!("cargo::rustc-link-arg=dynamic_lookup");
    }
}
