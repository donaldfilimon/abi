//! Explicit local model loading and evidenced execution for ABI.
//!
//! The crate connects [`abi_models`] storage and consent contracts to
//! [`abi_agent_runtime::ModelProvider`] through Candle. Callers must select an
//! exact model identifier and a [`DevicePreference`]; there is no global model,
//! device default, or silent substitution.
//!
//! The first supported architecture, [`FIXTURE_BIGRAM_ARCHITECTURE`], is a
//! deliberately tiny transition-logit model used to prove the complete
//! safetensors/tokenizer/device/provider path without committing model data to
//! this repository. Gemma and other production architectures remain separate
//! future adapters and must not be inferred from this foundation.
//!
//! CPU execution is always compiled. Metal and CUDA are target features:
//! compilation alone is reported separately from successful initialization and
//! native operations. Exact accelerator requests fail when unavailable; they
//! never fall back to CPU.

mod device;
mod error;
mod provider;
mod report;

pub use device::{DevicePreference, ExecutionPath};
pub use error::ModelRuntimeError;
pub use provider::{
    DEFAULT_MAX_GENERATED_TOKENS, DEFAULT_MAX_MODEL_BYTES, DEFAULT_MAX_PROMPT_BYTES,
    DEFAULT_MAX_TOKENIZER_BYTES, FIXTURE_BIGRAM_ARCHITECTURE, LoadConfig, LocalModelProvider,
};
pub use report::{InferenceReport, LoadReport};
