//! Honest WDBX compute-backend selection with a deterministic CPU SIMD path.

use std::simd::{Simd, num::SimdFloat};

/// A compute backend named by the WDBX public surface.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    /// Portable scalar CPU.
    CpuScalar,
    /// x86 AVX2-width CPU SIMD.
    CpuAvx2,
    /// x86 AVX-512-width CPU SIMD.
    CpuAvx512,
    /// `AArch64` NEON-width CPU SIMD.
    CpuNeon,
    /// CUDA capability name; no native kernel is linked.
    GpuCuda,
    /// Metal capability name; no native kernel is linked here.
    GpuMetal,
    /// Vulkan capability name; no native kernel is linked.
    GpuVulkan,
    /// Apple Neural Engine capability name; execution is not linked.
    NpuAne,
    /// Reference remote TPU transport.
    TpuRemote,
}

impl Backend {
    /// All backends in the frozen display order.
    pub const ALL: [Self; 9] = [
        Self::CpuScalar,
        Self::CpuAvx2,
        Self::CpuAvx512,
        Self::CpuNeon,
        Self::GpuCuda,
        Self::GpuMetal,
        Self::GpuVulkan,
        Self::NpuAne,
        Self::TpuRemote,
    ];

    /// Stable CLI-facing backend name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::CpuScalar => "cpu-scalar",
            Self::CpuAvx2 => "cpu-avx2",
            Self::CpuAvx512 => "cpu-avx512",
            Self::CpuNeon => "cpu-neon",
            Self::GpuCuda => "gpu-cuda",
            Self::GpuMetal => "gpu-metal",
            Self::GpuVulkan => "gpu-vulkan",
            Self::NpuAne => "npu-ane",
            Self::TpuRemote => "tpu-remote",
        }
    }

    /// Stable backend class.
    #[must_use]
    pub const fn class(self) -> &'static str {
        match self {
            Self::CpuScalar | Self::CpuAvx2 | Self::CpuAvx512 | Self::CpuNeon => "CPU",
            Self::GpuCuda | Self::GpuMetal | Self::GpuVulkan => "GPU",
            Self::NpuAne => "NPU",
            Self::TpuRemote => "TPU",
        }
    }

    const fn is_accelerator(self) -> bool {
        matches!(
            self,
            Self::GpuCuda | Self::GpuMetal | Self::GpuVulkan | Self::NpuAne | Self::TpuRemote
        )
    }
}

/// Availability and dispatch truth for one backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Capability {
    /// Described backend.
    pub backend: Backend,
    /// Whether a native dispatch implementation is linked.
    pub native: bool,
    /// Whether the selection is usable, including deterministic CPU fallback.
    pub available: bool,
}

/// Requested and effective backend selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Selection {
    /// Caller-requested backend.
    pub requested: Backend,
    /// Backend that performs the work.
    pub effective: Backend,
    /// Whether dispatch is native to the requested accelerator.
    pub native: bool,
    /// Honest operator-facing explanation.
    pub message: &'static str,
}

/// A compute operation failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComputeError {
    /// Input vectors have different dimensions.
    DimensionMismatch,
}

impl std::fmt::Display for ComputeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("vector dimensions differ")
    }
}

impl std::error::Error for ComputeError {}

/// Best host CPU path, using runtime feature detection where available.
#[must_use]
pub fn best_cpu_backend() -> Backend {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return Backend::CpuAvx512;
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return Backend::CpuAvx2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Backend::CpuNeon;
        }
    }
    Backend::CpuScalar
}

/// Whether the target is Apple-Silicon macOS.
///
/// This reports physical presence only. No CoreML/ANE execution path is linked.
#[must_use]
pub const fn ane_hardware_present() -> bool {
    cfg!(all(target_arch = "aarch64", target_os = "macos"))
}

/// Capability table in frozen backend order.
#[must_use]
pub fn capabilities() -> [Capability; 9] {
    let best = best_cpu_backend();
    Backend::ALL.map(|backend| {
        let cpu_available = match backend {
            Backend::CpuScalar => true,
            Backend::CpuAvx2 => matches!(best, Backend::CpuAvx2 | Backend::CpuAvx512),
            Backend::CpuAvx512 => best == Backend::CpuAvx512,
            Backend::CpuNeon => best == Backend::CpuNeon,
            _ => false,
        };
        Capability {
            backend,
            native: false,
            available: backend.is_accelerator() || cpu_available,
        }
    })
}

/// Select a backend, deterministically falling accelerators back to the host CPU.
#[must_use]
pub fn select(requested: Backend) -> Selection {
    if requested.is_accelerator() {
        Selection {
            requested,
            effective: best_cpu_backend(),
            native: false,
            message: "native accelerator dispatch not linked; deterministic CPU fallback active",
        }
    } else {
        Selection {
            requested,
            effective: requested,
            native: false,
            message: "CPU SIMD path active",
        }
    }
}

/// Host-selected SIMD lane count for `f32`.
#[must_use]
pub fn simd_lanes() -> usize {
    match best_cpu_backend() {
        Backend::CpuAvx512 => 16,
        Backend::CpuAvx2 => 8,
        Backend::CpuNeon | Backend::CpuScalar => 4,
        _ => unreachable!("best_cpu_backend always returns a CPU backend"),
    }
}

fn dot_lanes<const LANES: usize>(a: &[f32], b: &[f32]) -> f32 {
    let mut accumulator = Simd::<f32, LANES>::splat(0.0);
    let mut offset = 0;
    while offset + LANES <= a.len() {
        let left = Simd::from_slice(&a[offset..offset + LANES]);
        let right = Simd::from_slice(&b[offset..offset + LANES]);
        accumulator += left * right;
        offset += LANES;
    }
    let mut sum = accumulator.reduce_sum();
    for index in offset..a.len() {
        sum += a[index] * b[index];
    }
    sum
}

/// Execute a dot product on the selection's deterministic CPU path.
pub fn dot(selection: Selection, a: &[f32], b: &[f32]) -> Result<f32, ComputeError> {
    if a.len() != b.len() {
        return Err(ComputeError::DimensionMismatch);
    }
    let lanes = match selection.effective {
        Backend::CpuAvx512 => 16,
        Backend::CpuAvx2 => 8,
        Backend::CpuNeon | Backend::CpuScalar => 4,
        _ => simd_lanes(),
    };
    Ok(match lanes {
        16 => dot_lanes::<16>(a, b),
        8 => dot_lanes::<8>(a, b),
        _ => dot_lanes::<4>(a, b),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accelerator_selection_falls_back_to_the_best_cpu() {
        for requested in [
            Backend::GpuCuda,
            Backend::GpuMetal,
            Backend::GpuVulkan,
            Backend::NpuAne,
            Backend::TpuRemote,
        ] {
            let selection = select(requested);
            assert!(!selection.native);
            assert_eq!(selection.effective.class(), "CPU");
        }
        assert_eq!(select(Backend::CpuScalar).effective, Backend::CpuScalar);
    }

    #[test]
    fn capability_table_has_every_backend_without_native_accelerator_claims() {
        let capabilities = capabilities();
        assert_eq!(capabilities.len(), 9);
        assert!(capabilities.iter().all(|capability| !capability.native));
        assert!(
            capabilities
                .iter()
                .any(|capability| capability.backend == Backend::NpuAne)
        );
        assert!(
            capabilities
                .iter()
                .any(|capability| capability.backend == Backend::TpuRemote)
        );
    }

    #[test]
    fn backend_names_and_classes_match_the_oracle() {
        assert_eq!(Backend::CpuAvx512.name(), "cpu-avx512");
        assert_eq!(Backend::GpuMetal.class(), "GPU");
        assert_eq!(Backend::NpuAne.class(), "NPU");
        assert_eq!(Backend::TpuRemote.class(), "TPU");
    }

    #[test]
    fn cpu_and_accelerator_selections_have_dot_parity() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [5.0, 4.0, 3.0, 2.0, 1.0];
        let expected = 35.0;
        let cpu = dot(select(Backend::CpuScalar), &a, &b).expect("cpu dot");
        let gpu = dot(select(Backend::GpuMetal), &a, &b).expect("gpu fallback dot");
        let npu = dot(select(Backend::NpuAne), &a, &b).expect("npu fallback dot");
        assert!((cpu - expected).abs() < 1e-5);
        assert!((cpu - gpu).abs() < 1e-6);
        assert!((cpu - npu).abs() < 1e-6);
    }

    #[test]
    fn ragged_tail_and_dimension_rejection_match_the_oracle() {
        let a: Vec<f32> = (1_u8..=17).map(f32::from).collect();
        let b = vec![2.0; 17];
        let expected: f32 = a.iter().zip(&b).map(|(left, right)| left * right).sum();
        let actual = dot(select(Backend::CpuScalar), &a, &b).expect("ragged dot");
        assert!((actual - expected).abs() < 1e-4);
        assert_eq!(
            dot(select(Backend::CpuScalar), &[1.0], &[1.0, 2.0]),
            Err(ComputeError::DimensionMismatch)
        );
    }

    #[test]
    fn ane_presence_is_exactly_the_build_target_predicate() {
        assert_eq!(
            ane_hardware_present(),
            cfg!(all(target_arch = "aarch64", target_os = "macos"))
        );
    }
}
