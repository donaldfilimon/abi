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

#[cfg(all(feature = "metal-kernels", target_os = "macos"))]
mod ffi {
    unsafe extern "C" {
        pub fn abi_metal_available() -> bool;
        pub fn abi_metal_dot(a: *const f32, b: *const f32, n: usize) -> f32;
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
}
