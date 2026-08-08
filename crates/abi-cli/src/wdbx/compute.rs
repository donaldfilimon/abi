//! WDBX `compute` subcommand: CPU/GPU/NPU/TPU backend selection report.
//!
//! Split from the flat `wdbx` CLI module; dispatch lives in `super::run`.

use crate::app::Outcome;
use abi_compute::{Accelerator as _, CapabilityState};
use std::fmt::Write;

pub(crate) const COMPUTE_HELP: &str = "usage: abi wdbx compute info\n\nReport CPU/GPU/NPU/TPU backend selection and fallback state.\n";

fn compute_info() -> Outcome {
    let metal = abi_gpu::MetalAccelerator::new();
    let coreml = abi_gpu::CoreMlAneAccelerator::new();
    let query = [1.0_f32, 0.0];
    let matching = [1.0_f32, 0.0];
    let unrelated = [0.0_f32, 1.0];
    let probe = metal.top_k(&query, &[&matching, &unrelated], 2);

    let mut report = String::from("compute capability evidence (CPU fallback is deterministic):\n");
    for capability in abi_wdbx::capabilities() {
        writeln!(
            report,
            "  compat {:<10} class={:<3} service_available={} native={}",
            capability.backend.name(),
            capability.backend.class(),
            capability.available,
            capability.native
        )
        .expect("writing to a String cannot fail");
    }
    for state in [
        metal.capability(),
        abi_gpu::CudaAccelerator::new().capability(),
        abi_gpu::VulkanAccelerator::new().capability(),
        coreml.capability(),
    ] {
        write_capability_state(&mut report, state);
    }
    if let Err(detail) = probe {
        writeln!(report, "metal oracle probe: error={detail}")
            .expect("writing to a String cannot fail");
    }
    let best = abi_wdbx::best_cpu_backend();
    let selection = abi_wdbx::select(abi_wdbx::Backend::NpuAne);
    writeln!(
        report,
        "legacy selection: best_cpu={}; request npu-ane -> effective={} ({})",
        best.name(),
        selection.effective.name(),
        selection.message
    )
    .expect("writing to a String cannot fail");
    let coreml_state = coreml.capability();
    writeln!(
        report,
        "apple neural engine: hardware_present={} requested_compute_units=cpuAndNeuralEngine; inference_executed={} runtime_residency_verified={}",
        abi_wdbx::ane_hardware_present(),
        coreml_state.executed(),
        coreml_state.runtime_verified()
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

fn write_capability_state(report: &mut String, state: CapabilityState) {
    writeln!(
        report,
        "  evidence {:<10} compiled={} available={} initialized={} executed={} runtime_verified={} message={}",
        state.backend().name(),
        state.compiled(),
        state.available(),
        state.initialized(),
        state.executed(),
        state.runtime_verified(),
        state.message(),
    )
    .expect("writing to a String cannot fail");
}

pub(crate) fn run_compute(args: &[String]) -> Outcome {
    match args {
        [operation] if operation == "info" => compute_info(),
        _ => super::usage(),
    }
}
