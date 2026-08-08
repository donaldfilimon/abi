//! `abi-compute` accelerator adapters with claim-honest runtime evidence.

use std::sync::atomic::{AtomicBool, Ordering};

use abi_compute::{
    Accelerator, Backend, CapabilityEvidence, CapabilityState, ComputeError, CpuBackend,
    ScoredIndex,
};

use crate::metal_kernels;

fn close(left: f32, right: f32) -> bool {
    (left - right).abs() <= 1e-3 * left.abs().max(right.abs()).max(1.0)
}

fn same_ranking(left: &[ScoredIndex], right: &[ScoredIndex]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.index == right.index && close(left.score, right.score))
}

/// Native Metal numerical adapter with deterministic CPU fallback and oracle.
#[derive(Debug, Default)]
pub struct MetalAccelerator {
    cpu: CpuBackend,
    executed: AtomicBool,
    verified: AtomicBool,
}

impl MetalAccelerator {
    /// Construct an adapter. Metal initialization remains lazy in the Swift shim.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn record(&self, verified: bool) {
        self.executed.store(true, Ordering::Relaxed);
        if verified {
            self.verified.store(true, Ordering::Relaxed);
        }
    }
}

impl Accelerator for MetalAccelerator {
    fn backend(&self) -> Backend {
        Backend::GpuMetal
    }

    fn capability(&self) -> CapabilityState {
        let initialized = metal_kernels::kernels_active();
        let executed = self.executed.load(Ordering::Relaxed);
        let verified = self.verified.load(Ordering::Relaxed);
        CapabilityState::new(
            Backend::GpuMetal,
            CapabilityEvidence::new()
                .with_compiled(metal_kernels::kernels_linked())
                .with_available(initialized)
                .with_initialized(initialized)
                .with_executed(executed)
                .with_runtime_verified(verified),
            if verified {
                "Metal numerical execution matched the deterministic CPU oracle"
            } else if initialized {
                "Metal device and numerical pipelines initialized; no verified execution yet"
            } else if metal_kernels::kernels_linked() {
                "Metal adapter compiled but runtime initialization failed; CPU fallback active"
            } else {
                "Metal adapter not compiled for this target; CPU fallback active"
            },
        )
        .expect("Metal evidence ladder is valid")
    }

    fn dot(&self, left: &[f32], right: &[f32]) -> Result<f32, ComputeError> {
        let oracle = self.cpu.dot(left, right)?;
        let Some(native) = metal_kernels::dot(left, right) else {
            return Ok(oracle);
        };
        let verified = close(native, oracle);
        self.record(verified);
        Ok(if verified { native } else { oracle })
    }

    fn cosine(&self, left: &[f32], right: &[f32]) -> Result<f32, ComputeError> {
        let oracle = self.cpu.cosine(left, right)?;
        let Some(native) = metal_kernels::cosine(left, right) else {
            return Ok(oracle);
        };
        let verified = close(native, oracle);
        self.record(verified);
        Ok(if verified { native } else { oracle })
    }

    fn norm(&self, values: &[f32]) -> Result<f32, ComputeError> {
        let oracle = self.cpu.norm(values);
        let Some(native) = metal_kernels::norm(values) else {
            return Ok(oracle);
        };
        let verified = close(native, oracle);
        self.record(verified);
        Ok(if verified { native } else { oracle })
    }

    fn top_k(
        &self,
        query: &[f32],
        candidates: &[&[f32]],
        limit: usize,
    ) -> Result<Vec<ScoredIndex>, ComputeError> {
        let oracle = self.cpu.top_k(query, candidates, limit)?;
        if limit == 0 {
            return Ok(oracle);
        }
        let Some(native) = metal_kernels::top_k(query, candidates, limit) else {
            return Ok(oracle);
        };
        let verified = same_ranking(&native, &oracle);
        self.record(verified);
        Ok(if verified { native } else { oracle })
    }
}

/// `CoreML` adapter with deterministic tiny-model inference evidence.
///
/// The helper requests `.cpuAndNeuralEngine`, loads an embedded model, runs a
/// fixed prediction, and compares the output to an oracle. `CoreML` does not
/// expose device placement here, so successful inference is not ANE-residency
/// evidence and never sets `runtime_verified`.
#[derive(Debug, Default)]
pub struct CoreMlAneAccelerator {
    cpu: CpuBackend,
}

impl CoreMlAneAccelerator {
    /// Construct the adapter. The native inference probe remains lazy.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether `CoreML` accepted `.cpuAndNeuralEngine` on Apple Silicon.
    #[must_use]
    pub fn requested_cpu_and_neural_engine(&self) -> bool {
        abi_compute::ane_hardware_present()
            && metal_kernels::coreml_cpu_and_neural_engine_requested()
    }

    /// Whether deterministic `CoreML` inference matched the expected output.
    ///
    /// This records verified model execution under the requested compute-unit
    /// policy, not proof that `CoreML` placed the operation on the ANE.
    #[must_use]
    pub fn tiny_model_inference_verified(&self) -> bool {
        self.requested_cpu_and_neural_engine()
            && metal_kernels::coreml_tiny_model_inference_verified()
    }
}

impl Accelerator for CoreMlAneAccelerator {
    fn backend(&self) -> Backend {
        Backend::NpuAne
    }

    fn capability(&self) -> CapabilityState {
        let requested = self.requested_cpu_and_neural_engine();
        let compiled = metal_kernels::coreml_helper_compiled();
        let available = compiled && abi_compute::ane_hardware_present();
        let inference_verified = available && requested && self.tiny_model_inference_verified();
        CapabilityState::new(
            Backend::NpuAne,
            CapabilityEvidence::new()
                .with_compiled(compiled)
                .with_available(available)
                .with_initialized(inference_verified)
                .with_executed(inference_verified),
            if inference_verified {
                "CoreML tiny-model inference matched its oracle under cpuAndNeuralEngine; ANE residency is not verified"
            } else if requested {
                "CoreML accepted cpuAndNeuralEngine but tiny-model inference did not verify; ANE residency is not verified"
            } else if abi_compute::ane_hardware_present() {
                "ANE hardware detected but CoreML request helper unavailable; CPU fallback active"
            } else {
                "CoreML/ANE adapter unavailable on this host; CPU fallback active"
            },
        )
        .expect("CoreML evidence ladder is valid")
    }

    fn dot(&self, left: &[f32], right: &[f32]) -> Result<f32, ComputeError> {
        self.cpu.dot(left, right)
    }
}

/// Whether an NVIDIA CUDA compiler was detected during this build.
#[must_use]
pub fn cuda_toolchain_detected() -> bool {
    option_env!("ABI_CUDA_TOOLCHAIN_DETECTED") == Some("true")
}

/// Whether a Vulkan shader/runtime inspection tool was detected during build.
#[must_use]
pub fn vulkan_toolchain_detected() -> bool {
    option_env!("ABI_VULKAN_TOOLCHAIN_DETECTED") == Some("true")
}

macro_rules! unavailable_adapter {
    ($name:ident, $backend:expr, $feature:literal, $detected:ident, $compiled_message:literal, $disabled_message:literal) => {
        #[doc = $compiled_message]
        #[derive(Debug, Default)]
        pub struct $name {
            cpu: CpuBackend,
        }

        impl $name {
            /// Construct a deterministic CPU-fallback adapter.
            #[must_use]
            pub fn new() -> Self {
                Self::default()
            }

            /// Whether a related build tool was detected. This is not runtime evidence.
            #[must_use]
            pub fn toolchain_detected(&self) -> bool {
                $detected()
            }
        }

        impl Accelerator for $name {
            fn backend(&self) -> Backend {
                $backend
            }

            fn capability(&self) -> CapabilityState {
                CapabilityState::new(
                    $backend,
                    CapabilityEvidence::new().with_compiled(cfg!(feature = $feature)),
                    if cfg!(feature = $feature) {
                        $compiled_message
                    } else {
                        $disabled_message
                    },
                )
                .expect("unavailable adapter evidence is valid")
            }

            fn dot(&self, left: &[f32], right: &[f32]) -> Result<f32, ComputeError> {
                self.cpu.dot(left, right)
            }
        }
    };
}

unavailable_adapter!(
    CudaAccelerator,
    Backend::GpuCuda,
    "cuda-adapter",
    cuda_toolchain_detected,
    "CUDA adapter compiled, but no CUDA runtime initialized or executed; CPU fallback active",
    "CUDA adapter not compiled; CPU fallback active"
);

unavailable_adapter!(
    VulkanAccelerator,
    Backend::GpuVulkan,
    "vulkan-adapter",
    vulkan_toolchain_detected,
    "Vulkan adapter compiled, but no Vulkan runtime initialized or executed; CPU fallback active",
    "Vulkan adapter not compiled; CPU fallback active"
);

/// Rich states for the four local accelerator adapters.
#[must_use]
pub fn accelerator_capability_states() -> [CapabilityState; 4] {
    [
        MetalAccelerator::new().capability(),
        CudaAccelerator::new().capability(),
        VulkanAccelerator::new().capability(),
        CoreMlAneAccelerator::new().capability(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_adapter_matches_cpu_and_advances_only_after_native_execution() {
        let adapter = MetalAccelerator::new();
        let before = adapter.capability();
        assert!(!before.executed());
        let query = [1.0, 2.0, 3.0, 4.0];
        let first = [1.0, 0.0, 0.0, 0.0];
        let second = [0.0, 1.0, 0.0, 0.0];
        let native_or_fallback = adapter.top_k(&query, &[&first, &second], 2).expect("top k");
        let oracle = abi_compute::top_k(&query, &[&first, &second], 2).expect("oracle");
        assert!(same_ranking(&native_or_fallback, &oracle));
        let after = adapter.capability();
        if metal_kernels::kernels_active() {
            assert!(after.executed());
            assert!(after.runtime_verified());
        } else {
            assert!(!after.executed());
            assert!(!after.runtime_verified());
        }
    }

    #[test]
    fn coreml_inference_advances_execution_but_not_residency_evidence() {
        let adapter = CoreMlAneAccelerator::new();
        let state = adapter.capability();
        assert!(!state.runtime_verified());
        if !metal_kernels::coreml_helper_compiled() {
            assert!(!state.available());
        }
        let _ = adapter.dot(&[1.0], &[2.0]).expect("CPU fallback");
        let after_fallback = adapter.capability();
        assert_eq!(after_fallback.executed(), state.executed());
        assert!(!after_fallback.runtime_verified());
        if adapter.tiny_model_inference_verified() {
            assert!(state.initialized());
            assert!(state.executed());
            assert!(state.message().contains("residency"));
            assert!(!state.runtime_verified());
        } else {
            assert!(!state.initialized());
            assert!(!state.executed());
        }
    }

    #[test]
    fn cuda_and_vulkan_never_promote_build_evidence_to_runtime_evidence() {
        let cuda = CudaAccelerator::new();
        let vulkan = VulkanAccelerator::new();
        assert_eq!(cuda.capability().compiled(), cfg!(feature = "cuda-adapter"));
        assert_eq!(
            vulkan.capability().compiled(),
            cfg!(feature = "vulkan-adapter")
        );
        for state in [cuda.capability(), vulkan.capability()] {
            assert!(!state.available());
            assert!(!state.initialized());
            assert!(!state.executed());
            assert!(!state.runtime_verified());
            assert!(!state.can_dispatch());
        }
    }

    #[test]
    fn compute_and_renderer_backend_enums_remain_distinct() {
        assert_eq!(Backend::GpuMetal.name(), "gpu-metal");
        assert_eq!(crate::Backend::Metal.name(), "metal");
    }
}
