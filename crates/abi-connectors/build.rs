//! Build the Apple `FoundationModels` Swift shim when enabled.
//!
//! `FoundationModels` is a Swift-only framework, so on-device inference needs a
//! small `@c` dylib (`native/fm_shim.swift`). This matches the Zig gate's
//! approach: compile with `xcrun swiftc -emit-library` only on arm64 macOS
//! when the `foundationmodels` feature is on.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=native/fm_shim.swift");
    println!("cargo:rerun-if-env-changed=ABI_FM_SHIM_FORCE");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let fm = env::var("CARGO_FEATURE_FOUNDATIONMODELS").is_ok();

    if !(fm && target_os == "macos" && target_arch == "aarch64") {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let shim = manifest_dir.join("native/fm_shim.swift");
    let dylib = out_dir.join("libabi_fm_shim.dylib");

    let sdk = Command::new("xcrun")
        .args(["--sdk", "macosx", "--show-sdk-path"])
        .output()
        .expect("xcrun --show-sdk-path");
    assert!(
        sdk.status.success(),
        "xcrun failed: {}",
        String::from_utf8_lossy(&sdk.stderr)
    );
    let sdk_path = String::from_utf8_lossy(&sdk.stdout).trim().to_owned();

    // Prefer xcrun swiftc (Xcode) over a broken Swiftly PATH pin.
    let status = Command::new("xcrun")
        .args([
            "swiftc",
            "-emit-library",
            "-O",
            "-parse-as-library",
            "-target",
            "arm64-apple-macosx26.0",
            "-sdk",
            &sdk_path,
            // @loader_path so `./target/debug/abi` finds the dylib when it sits
            // next to the binary (copied below). Avoids depending on LC_RPATH
            // propagation from a dependency build script into the final exe.
            "-Xlinker",
            "-install_name",
            "-Xlinker",
            "@loader_path/libabi_fm_shim.dylib",
            shim.to_str().expect("utf8 path"),
            "-o",
            dylib.to_str().expect("utf8 path"),
        ])
        .status()
        .expect("spawn xcrun swiftc");
    assert!(
        status.success(),
        "swiftc failed to build libabi_fm_shim.dylib (status {status})"
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=dylib=abi_fm_shim");
    // Also search/copy beside target/{debug,release} for `./target/debug/abi`.
    if env::var("PROFILE").is_ok() {
        // OUT_DIR is .../target/<profile>/build/<pkg>/out
        if let Some(dir) = out_dir.ancestors().nth(3).map(PathBuf::from) {
            let dest = dir.join("libabi_fm_shim.dylib");
            std::fs::copy(&dylib, &dest).unwrap_or_else(|err| {
                panic!("copy libabi_fm_shim.dylib to {}: {err}", dest.display());
            });
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
    }
}
