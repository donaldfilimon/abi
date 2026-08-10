//! Model loading and inference evidence reports.

use crate::{DevicePreference, ExecutionPath};
use abi_compute::CapabilityState;
use std::time::Duration;

/// Evidence produced after artifacts, tokenizer, tensor shape, and device load.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoadReport {
    /// Exact model identifier selected by the caller.
    pub model_id: String,
    /// Immutable manifest revision.
    pub revision: String,
    /// Device requested by the caller.
    pub requested_device: DevicePreference,
    /// Device actually initialized.
    pub initialized_path: ExecutionPath,
    /// ABI capability evidence at load time; execution is still false.
    pub capability: CapabilityState,
    /// Verified artifact bytes read for the model.
    pub artifact_bytes: u64,
    /// Number of tensors in the safetensors document.
    pub tensor_count: usize,
    /// Number of transition parameters loaded by the fixture architecture.
    pub parameter_count: usize,
}

/// Evidence produced by the most recent successful provider run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InferenceReport {
    /// Exact loaded model identifier.
    pub model_id: String,
    /// Device originally requested at load.
    pub requested_device: DevicePreference,
    /// Device that actually executed tensor operations.
    pub executed_path: ExecutionPath,
    /// ABI capability evidence after native operations returned results.
    pub capability: CapabilityState,
    /// Tokenized prompt length.
    pub prompt_tokens: usize,
    /// Tokens emitted before EOS, cancellation, or budget stop.
    pub generated_tokens: usize,
    /// Native Candle tensor operations reached during generation.
    pub native_operations: u64,
    /// Whether a different device was substituted. Always false in this crate.
    pub fallback_used: bool,
    /// Whether more than one execution path participated. Always false here.
    pub mixed_execution: bool,
    /// Measured provider-run duration.
    pub duration: Duration,
}
