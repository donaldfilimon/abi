//! Truthful Rust feature and compute-backend diagnostics.

use std::fmt::Write as _;

use abi_gpu::mlir::{Dialect, ModuleSpec};
use abi_gpu::{
    ShaderLanguage, ShaderSource, compile_shader, detect_backend, lower_mlir,
    mlir_toolchain_status, mobile_status_line, shader_compiler_status,
};

use crate::app::Outcome;

const USAGE: &str = "usage: abi backends";
const ENABLED: &str = "\x1b[32m✓\x1b[0m";
const PENDING: &str = "\x1b[90m○\x1b[0m";

struct Feature {
    name: &'static str,
    implemented: bool,
    detail: &'static str,
}

const FEATURES: &[Feature] = &[
    Feature {
        name: "ai",
        implemented: true,
        detail: "personas, routing, completion, training, modulator; MCP AI tools live",
    },
    Feature {
        name: "wdbx",
        implemented: true,
        detail: "Rust vector store, HNSW, persistence, and reference services",
    },
    Feature {
        name: "sea",
        implemented: true,
        detail: "evidence recall, adaptive learn loop; MCP ai_learn live",
    },
    Feature {
        name: "nn",
        implemented: true,
        detail: "char-LM demo trainer; CLI nn train|sample live",
    },
    Feature {
        name: "gpu",
        implemented: true,
        detail: "detection + MCP gpu_status ported; native kernels not linked",
    },
    Feature {
        name: "accelerator",
        implemented: true,
        detail: "compute backend selection via abi-wdbx (CPU SIMD + ANE metadata); native kernels not linked",
    },
    Feature {
        name: "shaders",
        implemented: true,
        detail: "source validation + local artifact; external compiler toolchains not linked",
    },
    Feature {
        name: "mlir",
        implemented: true,
        detail: "textual IR lowering; external MLIR/LLVM toolchains not linked",
    },
    Feature {
        name: "tui",
        implemented: true,
        detail: "dashboard one-shot + TTY raw-mode refresh; agent tui line-mode REPL",
    },
    Feature {
        name: "os_control",
        implemented: true,
        detail: "agent os dry-run/execute --confirm; allowlist true/pwd/ls/whoami/date",
    },
    Feature {
        name: "telemetry",
        implemented: true,
        detail: "bounded process-wide counters and Prometheus text",
    },
    Feature {
        name: "foundationmodels",
        implemented: true,
        detail: "Swift @c shim + on-device complete; ready only when Apple Intelligence model is available",
    },
    Feature {
        name: "hash",
        implemented: true,
        detail: "portable Wyhash (Zig-compatible) + FNV-1a + wyhash128",
    },
    Feature {
        name: "metrics",
        implemented: true,
        detail: "owned counter/gauge registry + process-wide Prometheus telemetry",
    },
    Feature {
        name: "mobile",
        implemented: true,
        detail: "platform detection + simulated profile; native_dispatch=false",
    },
];

fn build_mode() -> &'static str {
    if cfg!(debug_assertions) {
        "Debug"
    } else {
        "Release"
    }
}

fn report() -> String {
    let mut output = format!(
        "ABI Framework  {}\nRust nightly (workspace min 1.99)  {}  {}  {}\n\nFeatures:\n",
        env!("CARGO_PKG_VERSION"),
        build_mode(),
        abi_foundation::system::platform(),
        abi_foundation::system::arch(),
    );
    for feature in FEATURES {
        writeln!(
            output,
            "  {:<18} {}  {}",
            feature.name,
            if feature.implemented {
                ENABLED
            } else {
                PENDING
            },
            feature.detail
        )
        .expect("writing to a String cannot fail");
    }

    let best = abi_wdbx::best_cpu_backend();
    let gpu = detect_backend();
    let shader_status = shader_compiler_status();
    let shader = compile_shader(ShaderSource {
        name: "status",
        language: ShaderLanguage::ZigKernel,
        source: "fn main() void {}",
    })
    .expect("sample shader validation is deterministic");
    let mlir_status = mlir_toolchain_status();
    let lowered = lower_mlir(&ModuleSpec {
        name: "status",
        dialect: Dialect::Linalg,
        operations: &["matmul"],
    })
    .expect("sample MLIR lowering is deterministic");
    let hash_smoke = abi_foundation::hash64(b"abi-backends");
    let mut metrics = abi_telemetry::Metrics::new();
    metrics.increment("backends.report", 1);
    let fm_bridge = abi_connectors::fm_bridge_linked();
    let fm_ready = abi_connectors::fm_available();

    writeln!(
        output,
        "\nCompute Backends:\n  GPU:            {}  {}  accelerated={}\n  Native kernels  not linked  vectorized CPU fallback active\n  Effective CPU:  {}  portable_simd_lanes={}\n  Native accelerator kernels: not linked",
        gpu.backend.name(),
        if gpu.available { ENABLED } else { PENDING },
        if gpu.accelerated {
            "\x1b[32myes\x1b[0m"
        } else {
            "\x1b[90mno\x1b[0m"
        },
        best.name(),
        abi_wdbx::simd_lanes()
    )
    .expect("writing to a String cannot fail");
    for capability in abi_wdbx::capabilities() {
        let selection = abi_wdbx::select(capability.backend);
        writeln!(
            output,
            "  {:<10} class={:<3} usable={} native={} effective={}",
            capability.backend.name(),
            capability.backend.class(),
            capability.available,
            capability.native,
            selection.effective.name(),
        )
        .expect("writing to a String cannot fail");
    }
    writeln!(
        output,
        "  Apple Neural Engine: hardware_present={} native_dispatch=false\n  Remote compute endpoint: {} (reference transport; local fallback, not production TPU)\n  Shaders         {}  {}  (compiler available={})\n  MLIR            {} → {}  (toolchain available={})\n  Hash            wyhash64=0x{hash_smoke:x}  (Zig-compatible)\n  Metrics         registry counters={}  (enabled={})\n  Mobile          {}\n  FoundationModels bridge_linked={} model_ready={}",
        abi_wdbx::ane_hardware_present(),
        abi_wdbx::remote_compute_endpoint()
            .as_deref()
            .unwrap_or("none"),
        shader.language.name(),
        if shader_status.available {
            ENABLED
        } else {
            PENDING
        },
        shader_status.available,
        lowered.dialect.name(),
        lowered.target_backend,
        mlir_status.available,
        metrics.get_counter("backends.report").unwrap_or(0),
        metrics.enabled(),
        mobile_status_line(),
        fm_bridge,
        fm_ready,
    )
    .expect("writing to a String cannot fail");
    output
}

/// Dispatch `abi backends`, excluding the top-level command token.
pub(crate) fn run(args: &[String]) -> Outcome {
    if args.is_empty() {
        Outcome::stderr(report(), 0)
    } else {
        Outcome::stderr(format!("error: {USAGE}\n"), 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_is_claim_honest_about_the_partial_rust_cutover() {
        let output = run(&[]);
        assert_eq!(output.exit_code, 0);
        assert!(output.stdout.is_empty());
        assert!(output.stderr.contains("Rust nightly"));
        assert!(output.stderr.contains("wdbx               \u{1b}[32m✓"));
        assert!(output.stderr.contains("gpu                \u{1b}[32m✓"));
        assert!(output.stderr.contains("shaders            \u{1b}[32m✓"));
        assert!(output.stderr.contains("mlir               \u{1b}[32m✓"));
        assert!(output.stderr.contains("hash               \u{1b}[32m✓"));
        assert!(output.stderr.contains("metrics            \u{1b}[32m✓"));
        assert!(output.stderr.contains("mobile             \u{1b}[32m✓"));
        assert!(output.stderr.contains("foundationmodels   \u{1b}[32m✓"));
        assert!(output.stderr.contains("FoundationModels bridge_linked="));
        assert!(
            output
                .stderr
                .contains("Native accelerator kernels: not linked")
        );
        assert!(output.stderr.contains("native_dispatch=false"));
        assert!(!output.stderr.contains("native=true"));
        assert!(!output.stderr.contains("accelerated=\u{1b}[32myes"));
    }

    #[test]
    fn extra_arguments_are_usage_errors() {
        let output = run(&["extra".to_owned()]);
        assert_eq!(output.exit_code, 2);
        assert_eq!(output.stderr, format!("error: {USAGE}\n"));
    }
}
