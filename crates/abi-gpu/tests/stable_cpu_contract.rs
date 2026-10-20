//! Exact numerical contracts at the ABI/shared-compute boundary.
use abi_compute::{Backend, ComputeError, CpuBackend, cosine, dot, norm, select, top_k};

#[test]
fn dimensions_and_zero_vectors_have_defined_results() {
    let cpu = CpuBackend::scalar();
    assert_eq!(cpu.dot(&[1.0], &[]), Err(ComputeError::DimensionMismatch));
    assert_eq!(
        cpu.cosine(&[], &[1.0]),
        Err(ComputeError::DimensionMismatch)
    );
    assert!(!cpu.capability().executed());
    assert_eq!(cpu.dot(&[], &[]).unwrap().to_bits(), 0);
    assert_eq!(norm(&[]).to_bits(), 0);
    assert_eq!(cosine(&[], &[]), Ok(0.0));
    assert_eq!(cosine(&[0.0, -0.0], &[3.0, 4.0]), Ok(0.0));
    assert_eq!(cpu.dot(&[-0.0], &[1.0]).unwrap().to_bits(), 0);
    assert_eq!(norm(&[3.0, 4.0]), 5.0);
    assert_eq!(cosine(&[3.0, 4.0], &[-3.0, -4.0]), Ok(-1.0));
}

#[test]
fn ragged_tails_and_frozen_reduction_order_are_preserved() {
    let cpu = CpuBackend::scalar();
    // Sum 1..=17 times two is exactly 306, including the seventeenth tail.
    let values: Vec<f32> = (1_u8..=17).map(f32::from).collect();
    assert_eq!(cpu.dot(&values, &[2.0; 17]), Ok(306.0));
    // Frozen four-lane reference: the five-element tail cancels after
    // lane reduction and loses the unit. At eight elements, cancellation
    // occurs inside lane zero and lane one retains its unit (serial sum: zero).
    let cancellation = [16_777_216.0, 1.0, 0.0, 0.0, -16_777_216.0];
    assert_eq!(cpu.dot(&cancellation, &[1.0; 5]), Ok(0.0));
    let grouped = [16_777_216.0, 1.0, 0.0, 0.0, -16_777_216.0, 0.0, 0.0, 0.0];
    assert_eq!(cpu.dot(&grouped, &[1.0; 8]), Ok(1.0));
}

#[test]
fn nonfinite_results_keep_ieee_classification() {
    let cpu = CpuBackend::scalar();
    assert!(cpu.dot(&[f32::NAN], &[1.0]).unwrap().is_nan());
    assert!(cpu.dot(&[f32::INFINITY], &[0.0]).unwrap().is_nan());
    assert_eq!(cpu.dot(&[f32::INFINITY], &[-1.0]), Ok(f32::NEG_INFINITY));
    assert_eq!(cpu.norm(&[f32::INFINITY]), f32::INFINITY);
    assert!(cpu.cosine(&[f32::INFINITY], &[1.0]).unwrap().is_nan());
}

#[test]
fn ranking_boundaries_and_equal_score_indices_are_stable() {
    let candidates: &[&[f32]] = &[&[1.0, 0.0], &[2.0, 0.0], &[0.0, 1.0], &[-1.0, 0.0]];
    assert_eq!(top_k(&[1.0, 0.0], candidates, 0).unwrap(), []);
    assert_eq!(top_k(&[1.0, 0.0], &[], 8).unwrap(), []);
    let all = top_k(&[1.0, 0.0], candidates, usize::MAX).unwrap();
    assert_eq!(
        all.iter().map(|hit| hit.index).collect::<Vec<_>>(),
        [0, 1, 2, 3]
    );
    assert_eq!(
        all.iter().map(|hit| hit.score).collect::<Vec<_>>(),
        [1.0, 1.0, 0.0, -1.0]
    );
    assert_eq!(
        top_k(&[1.0], &[&[]], 0),
        Err(ComputeError::DimensionMismatch)
    );
}

#[test]
fn unsupported_accelerator_selection_runs_honest_cpu_fallback() {
    let selection = select(Backend::GpuMetal);
    assert!(!selection.native);
    assert!(selection.effective.is_cpu());
    assert_eq!(
        dot(
            selection,
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[5.0, 4.0, 3.0, 2.0, 1.0]
        ),
        Ok(35.0)
    );
    let caps = abi_gpu::backend_capabilities(abi_gpu::Backend::Simulated);
    assert!(caps.available);
    assert!(!caps.accelerated);
    assert!(!caps.native_kernels);
}
