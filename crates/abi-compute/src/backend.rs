//! Backend identities, selection, and claim-honest capability states.

/// A compute backend named by ABI's public surfaces.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Backend {
    /// Portable scalar CPU.
    CpuScalar,
    /// x86 AVX2-width CPU SIMD.
    CpuAvx2,
    /// x86 AVX-512-width CPU SIMD.
    CpuAvx512,
    /// `AArch64` NEON-width CPU SIMD.
    CpuNeon,
    /// CUDA accelerator adapter.
    GpuCuda,
    /// Metal accelerator adapter.
    GpuMetal,
    /// Vulkan accelerator adapter.
    GpuVulkan,
    /// Apple Neural Engine adapter.
    NpuAne,
    /// Authenticated remote accelerator adapter.
    TpuRemote,
}

impl Backend {
    /// All backends in stable display order.
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

    /// Whether this is a CPU implementation rather than an accelerator.
    #[must_use]
    pub const fn is_cpu(self) -> bool {
        matches!(
            self,
            Self::CpuScalar | Self::CpuAvx2 | Self::CpuAvx512 | Self::CpuNeon
        )
    }

    /// Whether this backend represents an accelerator adapter.
    #[must_use]
    pub const fn is_accelerator(self) -> bool {
        !self.is_cpu()
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

/// An invalid capability-state transition or combination.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapabilityInvariantError {
    /// Initialization requires compiled adapter code and an available runtime.
    InitializedWithoutAvailability,
    /// Execution requires successful initialization.
    ExecutedWithoutInitialization,
    /// Runtime verification requires at least one successful execution.
    VerifiedWithoutExecution,
}

impl std::fmt::Display for CapabilityInvariantError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::InitializedWithoutAvailability => {
                "an initialized backend must be compiled and available"
            }
            Self::ExecutedWithoutInitialization => "an executed backend must first be initialized",
            Self::VerifiedWithoutExecution => "a runtime-verified backend must have executed",
        })
    }
}

impl std::error::Error for CapabilityInvariantError {}

/// Independently reported capability facts.
///
/// Named setters avoid an error-prone positional list of booleans. The final
/// combination is validated by [`CapabilityState::new`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CapabilityEvidence(u8);

impl CapabilityEvidence {
    const COMPILED: u8 = 1 << 0;
    const AVAILABLE: u8 = 1 << 1;
    const INITIALIZED: u8 = 1 << 2;
    const EXECUTED: u8 = 1 << 3;
    const RUNTIME_VERIFIED: u8 = 1 << 4;

    /// Construct empty evidence.
    #[must_use]
    pub const fn new() -> Self {
        Self(0)
    }

    const fn with_flag(self, flag: u8, value: bool) -> Self {
        if value {
            Self(self.0 | flag)
        } else {
            Self(self.0 & !flag)
        }
    }

    /// Record whether adapter code is compiled.
    #[must_use]
    pub const fn with_compiled(self, value: bool) -> Self {
        self.with_flag(Self::COMPILED, value)
    }

    /// Record whether the device, service, and runtime are available.
    #[must_use]
    pub const fn with_available(self, value: bool) -> Self {
        self.with_flag(Self::AVAILABLE, value)
    }

    /// Record whether initialization succeeded.
    #[must_use]
    pub const fn with_initialized(self, value: bool) -> Self {
        self.with_flag(Self::INITIALIZED, value)
    }

    /// Record whether at least one operation executed.
    #[must_use]
    pub const fn with_executed(self, value: bool) -> Self {
        self.with_flag(Self::EXECUTED, value)
    }

    /// Record whether execution was verified on the current runtime.
    #[must_use]
    pub const fn with_runtime_verified(self, value: bool) -> Self {
        self.with_flag(Self::RUNTIME_VERIFIED, value)
    }

    const fn has(self, flag: u8) -> bool {
        self.0 & flag != 0
    }

    /// Whether adapter code is compiled.
    #[must_use]
    pub const fn compiled(self) -> bool {
        self.has(Self::COMPILED)
    }

    /// Whether the required device, service, and runtime are available.
    #[must_use]
    pub const fn available(self) -> bool {
        self.has(Self::AVAILABLE)
    }

    /// Whether initialization succeeded.
    #[must_use]
    pub const fn initialized(self) -> bool {
        self.has(Self::INITIALIZED)
    }

    /// Whether at least one operation executed.
    #[must_use]
    pub const fn executed(self) -> bool {
        self.has(Self::EXECUTED)
    }

    /// Whether execution was verified on the current runtime.
    #[must_use]
    pub const fn runtime_verified(self) -> bool {
        self.has(Self::RUNTIME_VERIFIED)
    }
}

/// Capability and runtime evidence for one backend.
///
/// Construction validates the evidence ladder. In particular, compiled code
/// alone can never imply initialization, execution, or runtime verification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CapabilityState {
    backend: Backend,
    evidence: CapabilityEvidence,
    message: &'static str,
}

impl CapabilityState {
    /// Construct a capability record while enforcing honest state transitions.
    pub const fn new(
        backend: Backend,
        evidence: CapabilityEvidence,
        message: &'static str,
    ) -> Result<Self, CapabilityInvariantError> {
        if evidence.initialized() && (!evidence.compiled() || !evidence.available()) {
            return Err(CapabilityInvariantError::InitializedWithoutAvailability);
        }
        if evidence.executed() && !evidence.initialized() {
            return Err(CapabilityInvariantError::ExecutedWithoutInitialization);
        }
        if evidence.runtime_verified() && !evidence.executed() {
            return Err(CapabilityInvariantError::VerifiedWithoutExecution);
        }
        Ok(Self {
            backend,
            evidence,
            message,
        })
    }

    /// Described backend.
    #[must_use]
    pub const fn backend(self) -> Backend {
        self.backend
    }

    /// Whether adapter code is compiled into the current binary.
    #[must_use]
    pub const fn compiled(self) -> bool {
        self.evidence.compiled()
    }

    /// Whether the required device, service, and runtime are available.
    #[must_use]
    pub const fn available(self) -> bool {
        self.evidence.available()
    }

    /// Whether adapter initialization succeeded.
    #[must_use]
    pub const fn initialized(self) -> bool {
        self.evidence.initialized()
    }

    /// Whether at least one operation executed on this backend.
    #[must_use]
    pub const fn executed(self) -> bool {
        self.evidence.executed()
    }

    /// Whether execution was verified on the current runtime.
    #[must_use]
    pub const fn runtime_verified(self) -> bool {
        self.evidence.runtime_verified()
    }

    /// Operator-facing explanation of the evidence state.
    #[must_use]
    pub const fn message(self) -> &'static str {
        self.message
    }

    /// Whether the backend may receive native work.
    #[must_use]
    pub const fn can_dispatch(self) -> bool {
        self.compiled() && self.available() && self.initialized()
    }
}

/// Compatibility availability and dispatch truth for one backend.
///
/// `available` includes service through deterministic CPU fallback. Use
/// [`CapabilityState`] when the caller needs compilation, initialization,
/// execution, and runtime-verification evidence separately.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Capability {
    /// Described backend.
    pub backend: Backend,
    /// Whether native dispatch is linked and initialized.
    pub native: bool,
    /// Whether the request is usable, including deterministic CPU fallback.
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
/// This reports physical presence only. It is not evidence of `CoreML` or ANE
/// execution.
#[must_use]
pub const fn ane_hardware_present() -> bool {
    cfg!(all(target_arch = "aarch64", target_os = "macos"))
}

fn cpu_available(backend: Backend, best: Backend) -> bool {
    match backend {
        Backend::CpuScalar => true,
        Backend::CpuAvx2 => matches!(best, Backend::CpuAvx2 | Backend::CpuAvx512),
        Backend::CpuAvx512 => best == Backend::CpuAvx512,
        Backend::CpuNeon => best == Backend::CpuNeon,
        _ => false,
    }
}

/// Rich baseline capability states before optional adapters are attached.
#[must_use]
pub fn baseline_capability_states() -> [CapabilityState; 9] {
    let best = best_cpu_backend();
    Backend::ALL.map(|backend| {
        let is_available = cpu_available(backend, best);
        let hardware_present = match backend {
            Backend::GpuMetal => cfg!(target_os = "macos"),
            Backend::NpuAne => ane_hardware_present(),
            _ => false,
        };
        let (compiled, available, initialized, message) = if backend.is_cpu() {
            (
                true,
                is_available,
                is_available,
                if is_available {
                    "deterministic CPU SIMD implementation available"
                } else {
                    "CPU SIMD width is not available on this host"
                },
            )
        } else {
            (
                false,
                hardware_present,
                false,
                "accelerator adapter is not attached; deterministic CPU fallback active",
            )
        };
        CapabilityState::new(
            backend,
            CapabilityEvidence::new()
                .with_compiled(compiled)
                .with_available(available)
                .with_initialized(initialized),
            message,
        )
        .expect("baseline capability state is valid")
    })
}

/// Compatibility capability table in stable backend order.
///
/// Accelerator rows are available because their requests have deterministic
/// CPU fallback. None is native in this abstraction-only crate.
#[must_use]
pub fn capabilities() -> [Capability; 9] {
    let states = baseline_capability_states();
    Backend::ALL.map(|backend| Capability {
        backend,
        native: false,
        available: backend.is_accelerator()
            || states
                .iter()
                .find(|state| state.backend() == backend)
                .is_some_and(|state| state.available()),
    })
}

/// Select a backend without an attached accelerator implementation.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_evidence_combinations_are_rejected() {
        assert_eq!(
            CapabilityState::new(
                Backend::GpuMetal,
                CapabilityEvidence::new()
                    .with_compiled(true)
                    .with_available(true)
                    .with_executed(true),
                "invalid",
            ),
            Err(CapabilityInvariantError::ExecutedWithoutInitialization)
        );
        assert_eq!(
            CapabilityState::new(
                Backend::GpuMetal,
                CapabilityEvidence::new()
                    .with_compiled(true)
                    .with_available(true)
                    .with_initialized(true)
                    .with_runtime_verified(true),
                "invalid",
            ),
            Err(CapabilityInvariantError::VerifiedWithoutExecution)
        );
    }

    #[test]
    fn compiled_backend_does_not_imply_acceleration() {
        let selection = select(Backend::GpuMetal);
        assert!(!selection.native);
        assert!(selection.effective.is_cpu());
    }

    #[test]
    fn baseline_capabilities_never_claim_accelerator_execution() {
        for capability in baseline_capability_states() {
            assert!(!capability.executed());
            assert!(!capability.runtime_verified());
            if capability.backend().is_accelerator() {
                assert!(!capability.initialized());
            }
        }
    }

    #[test]
    fn compatibility_capabilities_keep_fallback_availability() {
        for capability in capabilities() {
            assert!(!capability.native);
            if capability.backend.is_accelerator() {
                assert!(capability.available);
            }
        }
    }

    #[test]
    fn backend_names_and_classes_are_stable() {
        assert_eq!(Backend::CpuAvx512.name(), "cpu-avx512");
        assert_eq!(Backend::GpuMetal.class(), "GPU");
        assert_eq!(Backend::NpuAne.class(), "NPU");
        assert_eq!(Backend::TpuRemote.class(), "TPU");
    }
}
