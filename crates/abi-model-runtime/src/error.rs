//! Typed local-model loading and inference failures.

use abi_models::ModelError;
use std::path::PathBuf;

/// Failures raised before or during local model execution.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ModelRuntimeError {
    /// Registry, license, storage, or hash verification failed.
    #[error(transparent)]
    Model(#[from] ModelError),
    /// An artifact could not be read.
    #[error("io error at {path}: {source}")]
    Io {
        /// Path being read.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },
    /// The manifest names an architecture this crate does not implement.
    #[error("model '{model}' declares unsupported architecture '{architecture}'")]
    UnsupportedArchitecture {
        /// Exact requested model identifier.
        model: String,
        /// Architecture declared by its manifest.
        architecture: String,
    },
    /// The selected architecture requires a different tensor format.
    #[error("model '{model}' requires safetensors, not {format:?}")]
    UnsupportedTensorFormat {
        /// Exact requested model identifier.
        model: String,
        /// Declared tensor format.
        format: abi_models::TensorFormat,
    },
    /// The fixture architecture requires exactly one weights artifact.
    #[error("model '{model}' declares {actual} weight artifacts; expected exactly one")]
    WeightArtifactCount {
        /// Exact requested model identifier.
        model: String,
        /// Number of weight artifacts declared.
        actual: usize,
    },
    /// Model artifacts exceed the caller's load bound.
    #[error("model '{model}' declares {actual} artifact bytes, above maximum {maximum}")]
    ModelTooLarge {
        /// Exact requested model identifier.
        model: String,
        /// Sum of declared artifact sizes.
        actual: u64,
        /// Configured ceiling.
        maximum: u64,
    },
    /// Tokenizer artifact exceeds its parser-allocation boundary.
    #[error("model '{model}' declares a {actual}-byte tokenizer, above maximum {maximum}")]
    TokenizerTooLarge {
        /// Exact requested model identifier.
        model: String,
        /// Manifest-declared tokenizer bytes.
        actual: u64,
        /// Configured tokenizer ceiling.
        maximum: u64,
    },
    /// Requested device feature or runtime is unavailable.
    #[error("requested device {requested} is unavailable: {detail}")]
    DeviceUnavailable {
        /// Explicit caller selection.
        requested: &'static str,
        /// Initialization failure or compile boundary.
        detail: String,
    },
    /// A safetensors document was invalid or lacked the required tensor.
    #[error("invalid safetensors for model '{model}': {detail}")]
    Safetensors {
        /// Exact requested model identifier.
        model: String,
        /// Parser or schema failure.
        detail: String,
    },
    /// Tokenizer parsing, encoding, or decoding failed.
    #[error("tokenizer failure for model '{model}': {detail}")]
    Tokenizer {
        /// Exact requested model identifier.
        model: String,
        /// Tokenizer-supplied failure.
        detail: String,
    },
    /// Candle loading or tensor execution failed.
    #[error("Candle failure for model '{model}': {detail}")]
    Candle {
        /// Exact requested model identifier.
        model: String,
        /// Candle-supplied failure.
        detail: String,
    },
    /// Request selected a model other than the one loaded by this provider.
    #[error(
        "provider loaded model '{loaded}' but request selected '{requested}'; substitution is forbidden"
    )]
    ModelMismatch {
        /// Model resident in the provider.
        loaded: String,
        /// Exact request model.
        requested: String,
    },
    /// A provider run requires at least one conversation message.
    #[error("model '{model}' cannot run an empty conversation")]
    EmptyRequest {
        /// Loaded model identifier.
        model: String,
    },
    /// The fixture architecture does not implement structured tool calls.
    #[error("model '{model}' does not support offered tools")]
    UnsupportedTools {
        /// Loaded model identifier.
        model: String,
    },
    /// Rendered request bytes exceed the pre-tokenization boundary.
    #[error("model '{model}' prompt has {actual} bytes, above maximum {maximum}")]
    PromptTooLarge {
        /// Loaded model identifier.
        model: String,
        /// Exact rendered prompt byte count.
        actual: usize,
        /// Configured pre-tokenization ceiling.
        maximum: usize,
    },
    /// Prompt cannot fit the manifest's declared context.
    #[error(
        "model '{model}' prompt has {prompt_tokens} tokens but context limit is {context_tokens}"
    )]
    ContextExceeded {
        /// Exact requested model identifier.
        model: String,
        /// Tokenized prompt length.
        prompt_tokens: usize,
        /// Manifest-declared context ceiling.
        context_tokens: u32,
    },
    /// Internal inference-report mutex was poisoned by a prior panic.
    #[error("inference report state for model '{model}' is poisoned")]
    ReportStatePoisoned {
        /// Loaded model identifier.
        model: String,
    },
}

impl ModelRuntimeError {
    /// Wrap a filesystem failure with its path.
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}
