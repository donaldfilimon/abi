//! Build script for ABI Rust bindings.
//!
//! This script generates Rust FFI bindings from the C header files.

use std::env;
use std::path::PathBuf;

fn main() {
    // Tell cargo to look for the library in the parent bindings/c/zig-out/lib directory
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let lib_path = PathBuf::from(&manifest_dir)
        .join("..")
        .join("c")
        .join("zig-out")
        .join("lib");

    println!("cargo:rustc-link-search=native={}", lib_path.display());
    println!("cargo:rustc-link-lib=dylib=abi");

    // Tell cargo to rerun if the header changes
    let header_path = PathBuf::from(&manifest_dir)
        .join("..")
        .join("c")
        .join("zig-out")
        .join("include")
        .join("abi.h");

    println!("cargo:rerun-if-changed={}", header_path.display());

    // macOS-specific: set rpath
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_path.display());
    }

    // Linux-specific: set rpath
    #[cfg(target_os = "linux")]
    {
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}",
            lib_path.display()
        );
    }
}
