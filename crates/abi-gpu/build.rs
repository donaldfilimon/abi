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
    let model_source = out_dir.join("tiny_coreml_model.swift");
    let dylib = out_dir.join("libabi_metal_dot.dylib");

    write_tiny_coreml_model_source(&model_source);

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
            model_source.to_str().expect("utf8 path"),
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

fn write_tiny_coreml_model_source(path: &Path) {
    let bytes = tiny_coreml_model();
    let mut source = String::from(
        "// Generated deterministically by abi-gpu/build.rs.\n\
         let abiTinyCoreMlModelBytes: [UInt8] = [\n",
    );
    for chunk in bytes.chunks(16) {
        source.push_str("    ");
        for byte in chunk {
            use std::fmt::Write as _;
            write!(source, "0x{byte:02x}, ").expect("write generated Swift literal");
        }
        source.push('\n');
    }
    source.push_str("]\n");
    std::fs::write(path, source).unwrap_or_else(|err| {
        panic!(
            "write generated CoreML model source {}: {err}",
            path.display()
        )
    });
}

/// A minimal `CoreML` neural-network model implementing `output = 2 * input + 1`.
///
/// `CoreML`'s model format is protobuf. Keeping this encoder local avoids a
/// build-time Python/`coremltools` dependency while making every byte auditable.
fn tiny_coreml_model() -> Vec<u8> {
    let array_type = message(|message| {
        packed_varints_field(message, 1, &[1]);
        varint_field(message, 2, 65_568); // ArrayFeatureType.FLOAT32
    });
    let feature_type = message(|message| message_field(message, 5, &array_type));
    let input = feature_description("input", &feature_type);
    let output = feature_description("output", &feature_type);
    let description = message(|message| {
        message_field(message, 1, &input);
        message_field(message, 10, &output);
    });

    let weights = message(|message| packed_f32_field(message, 1, &[2.0]));
    let bias = message(|message| packed_f32_field(message, 1, &[1.0]));
    let inner_product = message(|message| {
        varint_field(message, 1, 1); // inputChannels
        varint_field(message, 2, 1); // outputChannels
        varint_field(message, 10, 1); // hasBias
        message_field(message, 20, &weights);
        message_field(message, 21, &bias);
    });
    let layer = message(|message| {
        string_field(message, 1, "twice_plus_one");
        string_field(message, 2, "input");
        string_field(message, 3, "output");
        message_field(message, 140, &inner_product);
    });
    let network = message(|message| message_field(message, 1, &layer));

    message(|model| {
        varint_field(model, 1, 1); // CoreML specification version 1
        message_field(model, 2, &description);
        message_field(model, 500, &network);
    })
}

fn feature_description(name: &str, feature_type: &[u8]) -> Vec<u8> {
    message(|description| {
        string_field(description, 1, name);
        message_field(description, 3, feature_type);
    })
}

fn message(build: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
    let mut bytes = Vec::new();
    build(&mut bytes);
    bytes
}

fn key(output: &mut Vec<u8>, field: u64, wire_type: u64) {
    put_varint(output, (field << 3) | wire_type);
}

fn put_varint(output: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        let low = u8::try_from(value & 0x7f).expect("masked protobuf byte");
        output.push(low | 0x80);
        value >>= 7;
    }
    output.push(u8::try_from(value).expect("final protobuf byte"));
}

fn varint_field(output: &mut Vec<u8>, field: u64, value: u64) {
    key(output, field, 0);
    put_varint(output, value);
}

fn bytes_field(output: &mut Vec<u8>, field: u64, value: &[u8]) {
    key(output, field, 2);
    put_varint(output, value.len() as u64);
    output.extend_from_slice(value);
}

fn string_field(output: &mut Vec<u8>, field: u64, value: &str) {
    bytes_field(output, field, value.as_bytes());
}

fn message_field(output: &mut Vec<u8>, field: u64, value: &[u8]) {
    bytes_field(output, field, value);
}

fn packed_varints_field(output: &mut Vec<u8>, field: u64, values: &[u64]) {
    let packed = message(|bytes| {
        for value in values {
            put_varint(bytes, *value);
        }
    });
    bytes_field(output, field, &packed);
}

fn packed_f32_field(output: &mut Vec<u8>, field: u64, values: &[f32]) {
    let packed = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    bytes_field(output, field, &packed);
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
