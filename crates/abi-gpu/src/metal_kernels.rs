//! Optional linked Metal DOT kernels (macOS + feature `metal-kernels`).
//!
//! When the native dylib initializes a Metal device and pipeline,
//! [`kernels_active`] is true and [`dot`] dispatches on GPU. Otherwise
//! callers must use the CPU path and report `accelerated=false`.

#![allow(unsafe_code)]

/// Whether this build linked the Metal DOT dylib (compile-time feature).
#[must_use]
pub const fn kernels_linked() -> bool {
    cfg!(feature = "metal-kernels") && cfg!(target_os = "macos")
}

#[cfg(all(
    any(feature = "metal-kernels", feature = "coreml-ane"),
    target_os = "macos"
))]
mod ffi {
    unsafe extern "C" {
        pub fn abi_metal_available() -> bool;
        pub fn abi_metal_dot(a: *const f32, b: *const f32, n: usize) -> f32;
        pub fn abi_metal_batch_dot(
            query: *const f32,
            candidates: *const f32,
            count: usize,
            dimensions: usize,
            output: *mut f32,
        ) -> bool;
        pub fn abi_metal_cosine(a: *const f32, b: *const f32, n: usize) -> f32;
        pub fn abi_metal_norm(values: *const f32, n: usize) -> f32;
        pub fn abi_metal_top_k(
            query: *const f32,
            candidates: *const f32,
            count: usize,
            dimensions: usize,
            limit: usize,
            output_indices: *mut usize,
            output_scores: *mut f32,
        ) -> isize;
        pub fn abi_coreml_cpu_and_neural_engine_requested() -> bool;
    }
}

/// Whether the `CoreML` compute-unit request helper is compiled into this build.
#[must_use]
pub const fn coreml_helper_compiled() -> bool {
    cfg!(all(feature = "coreml-ane", target_os = "macos"))
}

/// Ask `CoreML` for `.cpuAndNeuralEngine` compute units.
///
/// A true result proves only that the request was accepted. It does not prove
/// inference, ANE residency, or runtime acceleration.
#[must_use]
pub fn coreml_cpu_and_neural_engine_requested() -> bool {
    #[cfg(all(feature = "coreml-ane", target_os = "macos"))]
    {
        // SAFETY: the symbol is built from the checked-in Swift shim.
        unsafe { ffi::abi_coreml_cpu_and_neural_engine_requested() }
    }
    #[cfg(not(all(feature = "coreml-ane", target_os = "macos")))]
    {
        false
    }
}

/// Whether Metal kernels are linked **and** a device/pipeline is ready.
#[must_use]
pub fn kernels_active() -> bool {
    #[cfg(all(feature = "metal-kernels", target_os = "macos"))]
    {
        // SAFETY: symbols come from libabi_metal_dot built by build.rs.
        unsafe { ffi::abi_metal_available() }
    }
    #[cfg(not(all(feature = "metal-kernels", target_os = "macos")))]
    {
        false
    }
}

/// GPU DOT product when kernels are active.
///
/// Returns `None` when kernels are inactive or the native call fails (NaN).
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || !kernels_active() {
        return None;
    }
    #[cfg(all(feature = "metal-kernels", target_os = "macos"))]
    {
        // SAFETY: pointers are valid for `a.len()` floats; n matches.
        let result = unsafe { ffi::abi_metal_dot(a.as_ptr(), b.as_ptr(), a.len()) };
        if result.is_finite() {
            Some(result)
        } else {
            None
        }
    }
    #[cfg(not(all(feature = "metal-kernels", target_os = "macos")))]
    {
        let _ = (a, b);
        None
    }
}

fn flatten_candidates(query: &[f32], candidates: &[&[f32]]) -> Option<Vec<f32>> {
    if candidates
        .iter()
        .any(|candidate| candidate.len() != query.len())
    {
        return None;
    }
    let capacity = candidates.len().checked_mul(query.len())?;
    let mut flattened = Vec::with_capacity(capacity);
    for candidate in candidates {
        flattened.extend_from_slice(candidate);
    }
    Some(flattened)
}

/// Native batch DOT scores when Metal is active.
#[must_use]
pub fn batch_dot(query: &[f32], candidates: &[&[f32]]) -> Option<Vec<f32>> {
    if !kernels_active() {
        return None;
    }
    u32::try_from(query.len()).ok()?;
    u32::try_from(candidates.len()).ok()?;
    let flattened = flatten_candidates(query, candidates)?;
    #[cfg(all(feature = "metal-kernels", target_os = "macos"))]
    {
        let mut output = vec![0.0; candidates.len()];
        // SAFETY: buffers cover the lengths supplied to the Swift shim.
        let succeeded = unsafe {
            ffi::abi_metal_batch_dot(
                query.as_ptr(),
                flattened.as_ptr(),
                candidates.len(),
                query.len(),
                output.as_mut_ptr(),
            )
        };
        succeeded.then_some(output)
    }
    #[cfg(not(all(feature = "metal-kernels", target_os = "macos")))]
    {
        let _ = flattened;
        None
    }
}

/// Native Metal cosine similarity when available.
#[must_use]
pub fn cosine(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || !kernels_active() {
        return None;
    }
    #[cfg(all(feature = "metal-kernels", target_os = "macos"))]
    {
        // SAFETY: both pointers are valid for `a.len()` elements.
        let result = unsafe { ffi::abi_metal_cosine(a.as_ptr(), b.as_ptr(), a.len()) };
        result.is_finite().then_some(result)
    }
    #[cfg(not(all(feature = "metal-kernels", target_os = "macos")))]
    {
        let _ = (a, b);
        None
    }
}

/// Native Metal Euclidean norm when available.
#[must_use]
pub fn norm(values: &[f32]) -> Option<f32> {
    if !kernels_active() {
        return None;
    }
    #[cfg(all(feature = "metal-kernels", target_os = "macos"))]
    {
        // SAFETY: the pointer is valid for `values.len()` elements.
        let result = unsafe { ffi::abi_metal_norm(values.as_ptr(), values.len()) };
        result.is_finite().then_some(result)
    }
    #[cfg(not(all(feature = "metal-kernels", target_os = "macos")))]
    {
        let _ = values;
        None
    }
}

/// Native Metal batch cosine followed by stable host-side top-k selection.
#[must_use]
pub fn top_k(
    query: &[f32],
    candidates: &[&[f32]],
    limit: usize,
) -> Option<Vec<abi_compute::ScoredIndex>> {
    if !kernels_active() {
        return None;
    }
    u32::try_from(query.len()).ok()?;
    u32::try_from(candidates.len()).ok()?;
    let flattened = flatten_candidates(query, candidates)?;
    let returned_limit = limit.min(candidates.len());
    #[cfg(all(feature = "metal-kernels", target_os = "macos"))]
    {
        let mut indices = vec![0_usize; returned_limit];
        let mut scores = vec![0.0; returned_limit];
        // SAFETY: all input and output buffers cover the supplied counts.
        let returned = unsafe {
            ffi::abi_metal_top_k(
                query.as_ptr(),
                flattened.as_ptr(),
                candidates.len(),
                query.len(),
                returned_limit,
                indices.as_mut_ptr(),
                scores.as_mut_ptr(),
            )
        };
        if returned < 0 || usize::try_from(returned).ok()? != returned_limit {
            return None;
        }
        Some(
            indices
                .into_iter()
                .zip(scores)
                .map(|(index, score)| abi_compute::ScoredIndex { index, score })
                .collect(),
        )
    }
    #[cfg(not(all(feature = "metal-kernels", target_os = "macos")))]
    {
        let _ = (flattened, returned_limit);
        None
    }
}

/// CPU reference DOT (oracle for Metal tests).
#[must_use]
pub fn cpu_dot(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }
    Some(a.iter().zip(b).map(|(x, y)| x * y).sum())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inactive_when_feature_off_or_non_macos() {
        if !kernels_linked() {
            assert!(!kernels_active());
            assert!(dot(&[1.0, 2.0], &[3.0, 4.0]).is_none());
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn metal_dot_matches_cpu_when_active() {
        if !kernels_active() {
            return;
        }
        let a: Vec<f32> = (0..257).map(|i| (i as f32) * 0.125).collect();
        let b: Vec<f32> = (0..257).map(|i| 1.0 - (i as f32) * 0.01).collect();
        let gpu = dot(&a, &b).expect("metal dot");
        let cpu = cpu_dot(&a, &b).expect("cpu dot");
        let err = (gpu - cpu).abs();
        assert!(
            err < 1e-3 * cpu.abs().max(1.0),
            "gpu={gpu} cpu={cpu} err={err}"
        );
    }

    #[test]
    fn coreml_request_never_claims_inference_or_residency() {
        #[cfg(all(feature = "coreml-ane", target_os = "macos", target_arch = "aarch64"))]
        assert!(coreml_cpu_and_neural_engine_requested());
        if coreml_cpu_and_neural_engine_requested() {
            assert!(coreml_helper_compiled());
        }
    }

    #[test]
    fn metal_batch_operations_match_cpu_when_active() {
        if !kernels_active() {
            return;
        }
        let query = [1.0, 2.0, 3.0, 4.0];
        let first = [1.0, 0.0, 0.0, 0.0];
        let second = [0.0, 1.0, 0.0, 0.0];
        let third = [2.0, 0.0, 0.0, 0.0];
        let candidates = [&first[..], &second[..], &third[..]];
        let native = batch_dot(&query, &candidates).expect("batch dot");
        for (score, candidate) in native.iter().zip(candidates) {
            let expected = cpu_dot(&query, candidate).expect("CPU dot");
            assert!((score - expected).abs() < 1e-5);
        }
        assert!((norm(&query).expect("norm") - abi_compute::norm(&query)).abs() < 1e-5);
        assert!(
            (cosine(&query, &first).expect("cosine")
                - abi_compute::cosine(&query, &first).expect("CPU cosine"))
            .abs()
                < 1e-5
        );
        let native_top = top_k(&query, &candidates, 3).expect("top k");
        let cpu_top = abi_compute::top_k(&query, &candidates, 3).expect("CPU top k");
        assert_eq!(native_top.len(), cpu_top.len());
        for (native, cpu) in native_top.iter().zip(cpu_top) {
            assert_eq!(native.index, cpu.index);
            assert!((native.score - cpu.score).abs() < 1e-5);
        }
    }
}
