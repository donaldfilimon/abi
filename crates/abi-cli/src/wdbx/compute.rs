//! WDBX `compute` subcommand: CPU/GPU/NPU/TPU backend selection report.
//!
//! Split from the flat `wdbx` CLI module; dispatch lives in `super::run`.

use crate::app::Outcome;
use std::fmt::Write;

pub(crate) const COMPUTE_HELP: &str = "usage: abi wdbx compute info\n\nReport CPU/GPU/NPU/TPU backend selection and fallback state.\n";

fn compute_info() -> Outcome {
    let mut report = String::from(
        "compute backends (native dispatch not linked in this build; CPU fallback active):\n",
    );
    for capability in abi_wdbx::capabilities() {
        writeln!(
            report,
            "  {:<10} class={:<3} available={} native={}",
            capability.backend.name(),
            capability.backend.class(),
            capability.available,
            capability.native
        )
        .expect("writing to a String cannot fail");
    }
    let best = abi_wdbx::best_cpu_backend();
    let selection = abi_wdbx::select(abi_wdbx::Backend::NpuAne);
    writeln!(
        report,
        "dynamic selection: best_cpu={}; request npu-ane -> effective={} ({})",
        best.name(),
        selection.effective.name(),
        selection.message
    )
    .expect("writing to a String cannot fail");
    writeln!(
        report,
        "apple neural engine: hardware_present={} native_dispatch=false (CoreML/ANE path requires Apple frameworks, not linked; CPU fallback)",
        abi_wdbx::ane_hardware_present()
    )
    .expect("writing to a String cannot fail");
    let endpoint = abi_wdbx::remote_compute_endpoint();
    writeln!(
        report,
        "remote compute: endpoint={} ({}; dotOrLocal attempts dial then CPU fallback — not production TPU)",
        endpoint.as_deref().unwrap_or("none"),
        abi_wdbx::REMOTE_COMPUTE_ENDPOINT_ENV
    )
    .expect("writing to a String cannot fail");
    if endpoint.is_some() {
        match abi_wdbx::dot_or_local(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) {
            Ok(value) => {
                writeln!(
                    report,
                    "remote compute probe: dot([1,2,3],[4,5,6])={value:.4} (local ref=32)"
                )
                .expect("writing to a String cannot fail");
            }
            Err(detail) => {
                writeln!(report, "remote compute probe: error={detail}")
                    .expect("writing to a String cannot fail");
                report
                    .push_str("remote compute probe: dot([1,2,3],[4,5,6])=0.0000 (local ref=32)\n");
            }
        }
    }
    Outcome::stderr(report, 0)
}

pub(crate) fn run_compute(args: &[String]) -> Outcome {
    match args {
        [operation] if operation == "info" => compute_info(),
        _ => super::usage(),
    }
}
