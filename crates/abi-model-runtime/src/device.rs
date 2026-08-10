//! Explicit device selection and initialization evidence.

use crate::ModelRuntimeError;
use abi_compute::{Backend, CapabilityEvidence, CapabilityState};
use candle_core::Device;

/// Caller-selected device. No variant means "automatic".
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DevicePreference {
    /// Candle CPU execution.
    Cpu,
    /// Candle Metal execution, requiring the `metal` feature and runtime.
    Metal,
    /// Candle CUDA execution, requiring the `cuda` feature and runtime.
    Cuda,
}

impl DevicePreference {
    /// Stable lowercase label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }
}

impl std::fmt::Display for DevicePreference {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Device path actually initialized or executed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionPath {
    /// Candle CPU tensors and operations.
    Cpu,
    /// Candle Metal tensors and operations.
    Metal,
    /// Candle CUDA tensors and operations.
    Cuda,
}

impl ExecutionPath {
    /// Stable lowercase label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }

    /// ABI compute capability identity for this execution path.
    #[must_use]
    pub const fn backend(self) -> Backend {
        match self {
            Self::Cpu => Backend::CpuScalar,
            Self::Metal => Backend::GpuMetal,
            Self::Cuda => Backend::GpuCuda,
        }
    }
}

impl std::fmt::Display for ExecutionPath {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Initialized Candle device plus claim-honest capability evidence.
pub(crate) struct InitializedDevice {
    pub(crate) device: Device,
    pub(crate) path: ExecutionPath,
    pub(crate) state: CapabilityState,
}

/// Initialize exactly the requested device, never substituting CPU.
pub(crate) fn initialize(
    preference: DevicePreference,
) -> Result<InitializedDevice, ModelRuntimeError> {
    match preference {
        DevicePreference::Cpu => Ok(initialized(
            Device::Cpu,
            ExecutionPath::Cpu,
            "Candle CPU device initialized; no accelerator placement is claimed",
        )),
        DevicePreference::Metal => initialize_metal(),
        DevicePreference::Cuda => initialize_cuda(),
    }
}

/// Build an initialized, not-yet-executed capability state.
fn initialized(device: Device, path: ExecutionPath, message: &'static str) -> InitializedDevice {
    let state = CapabilityState::new(
        path.backend(),
        CapabilityEvidence::new()
            .with_compiled(true)
            .with_available(true)
            .with_initialized(true),
        message,
    )
    .expect("initialized device evidence satisfies ABI invariants");
    InitializedDevice {
        device,
        path,
        state,
    }
}

#[cfg(feature = "metal")]
fn initialize_metal() -> Result<InitializedDevice, ModelRuntimeError> {
    let device = Device::new_metal(0).map_err(|error| ModelRuntimeError::DeviceUnavailable {
        requested: DevicePreference::Metal.as_str(),
        detail: error.to_string(),
    })?;
    Ok(initialized(
        device,
        ExecutionPath::Metal,
        "Candle Metal device initialized; execution is not evidenced until inference succeeds",
    ))
}

#[cfg(not(feature = "metal"))]
fn initialize_metal() -> Result<InitializedDevice, ModelRuntimeError> {
    Err(ModelRuntimeError::DeviceUnavailable {
        requested: DevicePreference::Metal.as_str(),
        detail: "abi-model-runtime was built without the metal feature".to_owned(),
    })
}

#[cfg(feature = "cuda")]
fn initialize_cuda() -> Result<InitializedDevice, ModelRuntimeError> {
    let device = Device::new_cuda(0).map_err(|error| ModelRuntimeError::DeviceUnavailable {
        requested: DevicePreference::Cuda.as_str(),
        detail: error.to_string(),
    })?;
    Ok(initialized(
        device,
        ExecutionPath::Cuda,
        "Candle CUDA device initialized; execution is not evidenced until inference succeeds",
    ))
}

#[cfg(not(feature = "cuda"))]
fn initialize_cuda() -> Result<InitializedDevice, ModelRuntimeError> {
    Err(ModelRuntimeError::DeviceUnavailable {
        requested: DevicePreference::Cuda.as_str(),
        detail: "abi-model-runtime was built without the cuda feature".to_owned(),
    })
}

/// Capability evidence after at least one native tensor operation succeeded.
pub(crate) fn executed_state(path: ExecutionPath) -> CapabilityState {
    CapabilityState::new(
        path.backend(),
        CapabilityEvidence::new()
            .with_compiled(true)
            .with_available(true)
            .with_initialized(true)
            .with_executed(true)
            .with_runtime_verified(true),
        "Candle tensor operations executed and returned a verified typed result",
    )
    .expect("executed device evidence satisfies ABI invariants")
}
