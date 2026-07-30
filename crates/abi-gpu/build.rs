//! Compile the Metal DOT shim on arm64 macOS when `metal-kernels` is enabled.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=native/metal_dot.swift");
    println!("cargo:rerun-if-env-changed=ABI_METAL_KERNELS_FORCE");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let metal = env::var("CARGO_FEATURE_METAL_KERNELS").is_ok();

    if !(metal && target_os == "macos" && (target_arch == "aarch64" || target_arch == "x86_64")) {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let shim = manifest_dir.join("native/metal_dot.swift");
    let dylib = out_dir.join("libabi_metal_dot.dylib");

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

    let triple = if target_arch == "aarch64" {
        "arm64-apple-macosx14.0"
    } else {
        "x86_64-apple-macosx14.0"
    };

    let status = Command::new("xcrun")
        .args([
            "swiftc",
            "-emit-library",
            "-O",
            "-parse-as-library",
            "-target",
            triple,
            "-sdk",
            &sdk_path,
            "-framework",
            "Metal",
            "-Xlinker",
            "-install_name",
            "-Xlinker",
            "@loader_path/libabi_metal_dot.dylib",
            shim.to_str().expect("utf8 path"),
            "-o",
            dylib.to_str().expect("utf8 path"),
        ])
        .status()
        .expect("spawn xcrun swiftc for metal_dot");
    assert!(
        status.success(),
        "swiftc failed to build libabi_metal_dot.dylib (status {status})"
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=dylib=abi_metal_dot");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");

    if let Some(dir) = out_dir.ancestors().nth(3).map(PathBuf::from) {
        let dest = dir.join("libabi_metal_dot.dylib");
        let _ = std::fs::copy(&dylib, &dest);
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
}
