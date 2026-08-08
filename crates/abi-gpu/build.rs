//! Compile Apple accelerator helpers and record optional toolchain detection.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=native/metal_dot.swift");
    println!("cargo:rerun-if-env-changed=ABI_METAL_KERNELS_FORCE");
    println!("cargo:rerun-if-env-changed=PATH");

    let cuda_detected = command_succeeds("nvcc", &["--version"]);
    let vulkan_detected =
        command_succeeds("glslc", &["--version"]) || command_succeeds("vulkaninfo", &["--summary"]);
    println!(
        "cargo:rustc-env=ABI_CUDA_TOOLCHAIN_DETECTED={}",
        if cuda_detected { "true" } else { "false" }
    );
    println!(
        "cargo:rustc-env=ABI_VULKAN_TOOLCHAIN_DETECTED={}",
        if vulkan_detected { "true" } else { "false" }
    );

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let metal = env::var("CARGO_FEATURE_METAL_KERNELS").is_ok();
    let coreml = env::var("CARGO_FEATURE_COREML_ANE").is_ok();

    if !((metal || coreml)
        && target_os == "macos"
        && (target_arch == "aarch64" || target_arch == "x86_64"))
    {
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
            "-framework",
            "CoreML",
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
    println!("cargo:rustc-link-lib=framework=CoreML");
    println!("cargo:rustc-link-lib=framework=Foundation");

    // Copy beside target/{debug,release} so the plain `./target/debug/abi`
    // binary can resolve `@loader_path/...`. A discarded error here would have
    // no build-time symptom — the first sign would be a dyld abort at runtime —
    // so fail loudly instead.
    if let Some(dir) = profile_dir(&out_dir) {
        let dest = dir.join("libabi_metal_dot.dylib");
        std::fs::copy(&dylib, &dest).unwrap_or_else(|err| {
            panic!("copy libabi_metal_dot.dylib to {}: {err}", dest.display());
        });
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
}

fn command_succeeds(program: &str, arguments: &[&str]) -> bool {
    Command::new(program)
        .args(arguments)
        .output()
        .is_ok_and(|output| output.status.success())
}

/// `target/<profile>` for an `OUT_DIR`, located by structure rather than depth.
///
/// See the twin helper in `abi-connectors/build.rs`: cargo nests `OUT_DIR`
/// differently across releases, so a fixed ancestor count silently drops the
/// dylib one directory away from where `@loader_path` resolves it.
fn profile_dir(out_dir: &Path) -> Option<PathBuf> {
    out_dir
        .ancestors()
        .find(|dir| dir.file_name().is_some_and(|name| name == "build"))
        .and_then(Path::parent)
        .map(PathBuf::from)
}
