//! Candle-backed local provider and the first deterministic fixture architecture.

use crate::device::{executed_state, initialize};
use crate::{DevicePreference, ExecutionPath, InferenceReport, LoadReport, ModelRuntimeError};
use abi_agent_runtime::{Flow, ModelEvent, ModelProvider, ModelRequest, RunContext, RuntimeError};
use abi_models::{
    AcceptanceLedger, ModelManifest, ModelRegistry, StorageRoot, TensorFormat, verify_artifact,
};
use candle_core::{DType, Device, Tensor};
use safetensors::SafeTensors;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;
use tokenizers::Tokenizer;

struct ArtifactSummary {
    weights_path: std::path::PathBuf,
    tokenizer_path: std::path::PathBuf,
    artifact_bytes: u64,
}

struct LoadedFixture {
    tokenizer: Tokenizer,
    transition: Tensor,
    eos_token: u32,
    tensor_count: usize,
    parameter_count: usize,
}

/// Tiny transition-logit architecture used for offline execution fixtures.
pub const FIXTURE_BIGRAM_ARCHITECTURE: &str = "abi-bigram-v1";

/// Default maximum verified artifact bytes loaded for one model: 16 GiB.
pub const DEFAULT_MAX_MODEL_BYTES: u64 = 16 * 1024 * 1024 * 1024;

/// Default hard generation ceiling independent of the request hint.
pub const DEFAULT_MAX_GENERATED_TOKENS: u32 = 256;

/// Explicit local-model load policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadConfig {
    /// Exact device request; no automatic selection exists.
    pub device: DevicePreference,
    /// Maximum sum of manifest-declared artifact bytes.
    pub max_model_bytes: u64,
    /// Hard provider generation ceiling.
    pub max_generated_tokens: u32,
}

impl LoadConfig {
    /// Start a load policy with an explicit device selection.
    #[must_use]
    pub const fn new(device: DevicePreference) -> Self {
        Self {
            device,
            max_model_bytes: DEFAULT_MAX_MODEL_BYTES,
            max_generated_tokens: DEFAULT_MAX_GENERATED_TOKENS,
        }
    }

    /// Override the maximum aggregate artifact bytes.
    #[must_use]
    pub const fn with_max_model_bytes(mut self, maximum: u64) -> Self {
        self.max_model_bytes = maximum;
        self
    }

    /// Override the generation ceiling.
    #[must_use]
    pub const fn with_max_generated_tokens(mut self, maximum: u32) -> Self {
        self.max_generated_tokens = maximum;
        self
    }
}

/// Loaded, explicitly selected local model provider.
pub struct LocalModelProvider {
    model_id: String,
    requested_device: DevicePreference,
    executed_path: ExecutionPath,
    tokenizer: Tokenizer,
    transition: Tensor,
    eos_token: u32,
    context_tokens: u32,
    manifest_output_tokens: u32,
    max_generated_tokens: u32,
    load_report: LoadReport,
    last_inference: Mutex<Option<InferenceReport>>,
}

impl std::fmt::Debug for LocalModelProvider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LocalModelProvider")
            .field("model_id", &self.model_id)
            .field("requested_device", &self.requested_device)
            .field("executed_path", &self.executed_path)
            .field("load_report", &self.load_report)
            .finish_non_exhaustive()
    }
}

impl LocalModelProvider {
    /// Verify, authorize, and load one exact model from external storage.
    pub fn load(
        registry: &ModelRegistry,
        ledger: &AcceptanceLedger,
        storage: &StorageRoot,
        model_id: &str,
        principal: &str,
        config: LoadConfig,
    ) -> Result<Self, ModelRuntimeError> {
        let usable = registry.resolve_for(model_id, ledger, principal)?;
        let manifest = usable.manifest();
        let artifacts = verify_model_artifacts(manifest, storage, config.max_model_bytes)?;
        let initialized = initialize(config.device)?;
        let loaded = load_fixture(manifest, &artifacts, &initialized.device)?;
        let load_report = LoadReport {
            model_id: manifest.id.clone(),
            revision: manifest.revision.as_str().to_owned(),
            requested_device: config.device,
            initialized_path: initialized.path,
            capability: initialized.state,
            artifact_bytes: artifacts.artifact_bytes,
            tensor_count: loaded.tensor_count,
            parameter_count: loaded.parameter_count,
        };
        Ok(Self {
            model_id: manifest.id.clone(),
            requested_device: config.device,
            executed_path: initialized.path,
            tokenizer: loaded.tokenizer,
            transition: loaded.transition,
            eos_token: loaded.eos_token,
            context_tokens: manifest.context.max_context_tokens,
            manifest_output_tokens: manifest.context.max_output_tokens,
            max_generated_tokens: config.max_generated_tokens,
            load_report,
            last_inference: Mutex::new(None),
        })
    }

    /// Immutable evidence captured when this provider loaded.
    #[must_use]
    pub const fn load_report(&self) -> &LoadReport {
        &self.load_report
    }

    /// Most recent successful inference evidence, if any.
    pub fn last_inference_report(&self) -> Result<Option<InferenceReport>, ModelRuntimeError> {
        self.last_inference
            .lock()
            .map(|report| report.clone())
            .map_err(|_| ModelRuntimeError::ReportStatePoisoned {
                model: self.model_id.clone(),
            })
    }

    /// Drive the fixture architecture and retain execution evidence.
    fn infer(
        &self,
        request: &ModelRequest,
        run: &mut RunContext<'_>,
    ) -> Result<(), ModelRuntimeError> {
        if request.model() != self.model_id {
            return Err(ModelRuntimeError::ModelMismatch {
                loaded: self.model_id.clone(),
                requested: request.model().to_owned(),
            });
        }
        if request.messages().is_empty() {
            return Err(ModelRuntimeError::EmptyRequest {
                model: self.model_id.clone(),
            });
        }
        if !request.tools().is_empty() {
            return Err(ModelRuntimeError::UnsupportedTools {
                model: self.model_id.clone(),
            });
        }
        let started = Instant::now();
        if run.emit(&ModelEvent::started(&self.model_id)) == Flow::Stop {
            return Ok(());
        }
        let (prompt_ids, generation_limit) = self.prepare(request, run)?;
        let (generated_tokens, native_operations) =
            self.generate(prompt_ids.last().copied(), generation_limit, run)?;
        let capability = if native_operations == 0 {
            self.load_report.capability
        } else {
            executed_state(self.executed_path)
        };
        let report = InferenceReport {
            model_id: self.model_id.clone(),
            requested_device: self.requested_device,
            executed_path: self.executed_path,
            capability,
            prompt_tokens: prompt_ids.len(),
            generated_tokens,
            native_operations,
            fallback_used: false,
            mixed_execution: false,
            duration: started.elapsed(),
        };
        *self
            .last_inference
            .lock()
            .map_err(|_| ModelRuntimeError::ReportStatePoisoned {
                model: self.model_id.clone(),
            })? = Some(report);
        Ok(())
    }

    fn prepare(
        &self,
        request: &ModelRequest,
        run: &mut RunContext<'_>,
    ) -> Result<(Vec<u32>, u64), ModelRuntimeError> {
        let encoding = self
            .tokenizer
            .encode(render_prompt(request), false)
            .map_err(|error| self.tokenizer_error(error))?;
        let prompt_ids = encoding.get_ids().to_vec();
        if prompt_ids.is_empty() {
            return Err(self.tokenizer_error("prompt encoded to zero tokens"));
        }
        if prompt_ids.len() > self.context_tokens as usize {
            return Err(ModelRuntimeError::ContextExceeded {
                model: self.model_id.clone(),
                prompt_tokens: prompt_ids.len(),
                context_tokens: self.context_tokens,
            });
        }
        run.record_input_tokens(u64::try_from(prompt_ids.len()).unwrap_or(u64::MAX));
        let requested = request
            .max_output_tokens()
            .unwrap_or(u64::from(self.manifest_output_tokens));
        let remaining = u64::from(self.context_tokens)
            .saturating_sub(u64::try_from(prompt_ids.len()).unwrap_or(u64::MAX));
        let generation_limit = requested
            .min(u64::from(self.manifest_output_tokens))
            .min(u64::from(self.max_generated_tokens))
            .min(remaining);
        Ok((prompt_ids, generation_limit))
    }

    fn generate(
        &self,
        start: Option<u32>,
        limit: u64,
        run: &mut RunContext<'_>,
    ) -> Result<(usize, u64), ModelRuntimeError> {
        let mut current =
            usize::try_from(start.expect("prepared prompt is non-empty")).unwrap_or(usize::MAX);
        let mut generated = Vec::new();
        let mut decoded = String::new();
        let mut native_operations = 0u64;
        for _ in 0..limit {
            let row = self
                .transition
                .get(current)
                .and_then(|row| row.argmax(0))
                .map_err(|error| self.candle_error(error))?;
            native_operations = native_operations.saturating_add(2);
            let next = row
                .to_scalar::<u32>()
                .map_err(|error| self.candle_error(error))?;
            native_operations = native_operations.saturating_add(1);
            if next == self.eos_token {
                break;
            }
            generated.push(next);
            let next_decoded = self
                .tokenizer
                .decode(&generated, true)
                .map_err(|error| self.tokenizer_error(error))?;
            let delta = next_decoded
                .strip_prefix(&decoded)
                .ok_or_else(|| self.tokenizer_error("incremental decode changed emitted text"))?;
            if run.emit(&ModelEvent::text(delta)) == Flow::Stop {
                generated.pop();
                break;
            }
            run.record_output_tokens(1);
            decoded = next_decoded;
            current = usize::try_from(next).unwrap_or(usize::MAX);
        }
        Ok((generated.len(), native_operations))
    }

    fn tokenizer_error(&self, error: impl std::fmt::Display) -> ModelRuntimeError {
        ModelRuntimeError::Tokenizer {
            model: self.model_id.clone(),
            detail: error.to_string(),
        }
    }

    fn candle_error(&self, error: impl std::fmt::Display) -> ModelRuntimeError {
        ModelRuntimeError::Candle {
            model: self.model_id.clone(),
            detail: error.to_string(),
        }
    }
}

impl ModelProvider for LocalModelProvider {
    fn id(&self) -> &'static str {
        "abi-local-model"
    }

    fn run(&self, request: &ModelRequest, run: &mut RunContext<'_>) -> Result<(), RuntimeError> {
        self.infer(request, run)
            .map_err(|error| RuntimeError::Provider {
                provider: self.id().to_owned(),
                message: error.to_string(),
            })
    }
}

fn verify_model_artifacts(
    manifest: &ModelManifest,
    storage: &StorageRoot,
    maximum: u64,
) -> Result<ArtifactSummary, ModelRuntimeError> {
    if manifest.architecture != FIXTURE_BIGRAM_ARCHITECTURE {
        return Err(ModelRuntimeError::UnsupportedArchitecture {
            model: manifest.id.clone(),
            architecture: manifest.architecture.clone(),
        });
    }
    if manifest.tensor_format != TensorFormat::Safetensors {
        return Err(ModelRuntimeError::UnsupportedTensorFormat {
            model: manifest.id.clone(),
            format: manifest.tensor_format,
        });
    }
    let artifact_bytes = manifest
        .artifacts
        .iter()
        .try_fold(0u64, |sum, artifact| sum.checked_add(artifact.size_bytes))
        .unwrap_or(u64::MAX);
    if artifact_bytes > maximum {
        return Err(ModelRuntimeError::ModelTooLarge {
            model: manifest.id.clone(),
            actual: artifact_bytes,
            maximum,
        });
    }
    let weights: Vec<_> = manifest.weights().collect();
    if weights.len() != 1 {
        return Err(ModelRuntimeError::WeightArtifactCount {
            model: manifest.id.clone(),
            actual: weights.len(),
        });
    }
    for artifact in &manifest.artifacts {
        verify_artifact(&storage.artifact_path(manifest, artifact), artifact)?;
    }
    Ok(ArtifactSummary {
        weights_path: storage.artifact_path(manifest, weights[0]),
        tokenizer_path: storage.artifact_path(manifest, manifest.tokenizer()),
        artifact_bytes,
    })
}

fn load_fixture(
    manifest: &ModelManifest,
    artifacts: &ArtifactSummary,
    device: &Device,
) -> Result<LoadedFixture, ModelRuntimeError> {
    let tokenizer = Tokenizer::from_file(&artifacts.tokenizer_path).map_err(|error| {
        ModelRuntimeError::Tokenizer {
            model: manifest.id.clone(),
            detail: error.to_string(),
        }
    })?;
    let weights_bytes = read(&artifacts.weights_path)?;
    let safetensors = SafeTensors::deserialize(&weights_bytes).map_err(|error| {
        ModelRuntimeError::Safetensors {
            model: manifest.id.clone(),
            detail: error.to_string(),
        }
    })?;
    let tensors =
        candle_core::safetensors::load_buffer(&weights_bytes, device).map_err(|error| {
            ModelRuntimeError::Candle {
                model: manifest.id.clone(),
                detail: error.to_string(),
            }
        })?;
    let transition = tensors
        .get("transition")
        .ok_or_else(|| ModelRuntimeError::Safetensors {
            model: manifest.id.clone(),
            detail: "required tensor 'transition' is absent".to_owned(),
        })?;
    validate_transition(manifest, &tokenizer, transition)?;
    let eos_token = tokenizer
        .token_to_id("<eos>")
        .ok_or_else(|| ModelRuntimeError::Tokenizer {
            model: manifest.id.clone(),
            detail: "required <eos> token is absent".to_owned(),
        })?;
    Ok(LoadedFixture {
        tokenizer,
        transition: transition.clone(),
        eos_token,
        tensor_count: safetensors.len(),
        parameter_count: transition.elem_count(),
    })
}

fn validate_transition(
    manifest: &ModelManifest,
    tokenizer: &Tokenizer,
    transition: &Tensor,
) -> Result<(), ModelRuntimeError> {
    if transition.dtype() != DType::F32 {
        return Err(ModelRuntimeError::Safetensors {
            model: manifest.id.clone(),
            detail: format!(
                "transition tensor must be f32, found {:?}",
                transition.dtype()
            ),
        });
    }
    let vocabulary = tokenizer.get_vocab_size(true);
    if transition.dims() != [vocabulary, vocabulary] {
        return Err(ModelRuntimeError::Safetensors {
            model: manifest.id.clone(),
            detail: format!(
                "transition shape {:?} does not match tokenizer vocabulary {vocabulary}",
                transition.dims()
            ),
        });
    }
    Ok(())
}

/// Read an artifact without an unsafe memory map.
fn read(path: &Path) -> Result<Vec<u8>, ModelRuntimeError> {
    std::fs::read(path).map_err(|source| ModelRuntimeError::io(path, source))
}

/// Deterministically render all request context for tokenization.
fn render_prompt(request: &ModelRequest) -> String {
    let capacity = request.messages().len() + usize::from(request.system().is_some());
    let mut parts = Vec::with_capacity(capacity);
    if let Some(system) = request.system() {
        parts.push(system);
    }
    parts.extend(
        request
            .messages()
            .iter()
            .map(|message| message.content.as_str()),
    );
    parts.join("\n")
}
