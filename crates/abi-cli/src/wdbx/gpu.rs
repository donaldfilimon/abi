//! WDBX `gpu` subcommand: GPU backend capability report.
//!
//! Split from the flat `wdbx` CLI module; dispatch lives in `super::run`.

use crate::app::Outcome;

pub(crate) const GPU_HELP: &str =
    "usage: abi wdbx gpu info\n\nReport GPU backend capability and native-kernel status.\n";

fn gpu_info() -> Outcome {
    let gpu = abi_gpu::detect_backend();
    let linked = abi_gpu::metal_kernels::kernels_active();
    let note = if linked {
        "note: Metal DOT kernel linked and initialized; other vector ops may still use CPU SIMD"
    } else {
        "note: native kernels inactive or not linked; vector ops use deterministic CPU SIMD fallback"
    };
    Outcome::stderr(
        format!(
            "gpu backend={} accelerated={} native_linked={}\nmessage={}\n{note}\n",
            gpu.backend.name(),
            gpu.accelerated,
            linked,
            gpu.message,
        ),
        0,
    )
}

pub(crate) fn run_gpu(args: &[String]) -> Outcome {
    match args {
        [operation] if operation == "info" => gpu_info(),
        _ => super::usage(),
    }
}
