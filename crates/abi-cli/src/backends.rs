//! Truthful Rust feature and compute-backend diagnostics.

use std::fmt::Write as _;

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
        implemented: false,
        detail: "Rust shader validation pending",
    },
    Feature {
        name: "mlir",
        implemented: false,
        detail: "Rust textual MLIR lowering pending",
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
        implemented: false,
        detail: "Apple Foundation Models Rust bridge pending",
    },
    Feature {
        name: "hash",
        implemented: false,
        detail: "standalone Rust hashing feature pending",
    },
    Feature {
        name: "metrics",
        implemented: false,
        detail: "Rust metrics feature pending",
    },
    Feature {
        name: "mobile",
        implemented: false,
        detail: "Rust mobile platform feature pending",
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
    writeln!(
        output,
        "\nCompute Backends:\n  Effective CPU:  {}  portable_simd_lanes={}\n  Native accelerator kernels: not linked",
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
        "  Apple Neural Engine: hardware_present={} native_dispatch=false\n  Remote compute endpoint: {} (reference transport; local fallback, not production TPU)",
        abi_wdbx::ane_hardware_present(),
        abi_wdbx::remote_compute_endpoint().as_deref().unwrap_or("none"),
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
        assert!(
            output
                .stderr
                .contains("Native accelerator kernels: not linked")
        );
        assert!(!output.stderr.contains("native=true"));
    }

    #[test]
    fn extra_arguments_are_usage_errors() {
        let output = run(&["extra".to_owned()]);
        assert_eq!(output.exit_code, 2);
        assert_eq!(output.stderr, format!("error: {USAGE}\n"));
    }
}
