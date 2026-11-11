//! Candle-backed local provider: the deterministic fixture architecture and
//! the Gemma 3 text adapter share one load, budget, and evidence path.

use crate::device::{executed_state, initialize};
use crate::gemma::{GEMMA3_TEXT_ARCHITECTURE, GemmaEngine, GemmaSession, MAX_CONFIG_BYTES};
use crate::gemma4::{GEMMA4_TEXT_ARCHITECTURE, Gemma4Engine, Gemma4Session};
use crate::{DevicePreference, ExecutionPath, InferenceReport, LoadReport, ModelRuntimeError};
use abi_agent_runtime::{
    Flow, ModelEvent, ModelProvider, ModelRequest, RunContext, RuntimeError, ToolCall, ToolChoice,
};
use abi_models::{
    AcceptanceLedger, ArtifactKind, ModelManifest, ModelRegistry, StorageRoot, TensorFormat,
    verify_artifact,
};
use candle_core::{DType, Device, Tensor};
use safetensors::SafeTensors;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;
use tokenizers::Tokenizer;

struct ArtifactSummary {
    weights_paths: Vec<std::path::PathBuf>,
    config_path: Option<std::path::PathBuf>,
    tokenizer_path: std::path::PathBuf,
    artifact_bytes: u64,
}

/// The architecture-specific half of a loaded model.
enum Engine {
    /// Transition-logit fixture: one tensor, one row lookup per token.
    Bigram { transition: Tensor, eos_token: u32 },
    /// Gemma 3 text decoder.
    Gemma(Box<GemmaEngine>),
    /// Gemma 4 text decoder with the tool grammar.
    Gemma4(Box<Gemma4Engine>),
}

/// What one generated token means for the caller-visible stream.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Step {
    /// The model ended its turn; the token is not part of the output.
    Stop,
    /// Answer text.
    Text(u32),
    /// Generated and fed back, but never shown: control tokens, an
    /// unrequested thought channel, or the inside of a tool call.
    Hidden(u32),
    /// Requested reasoning from the thought channel.
    Reasoning(u32),
    /// The token closed a complete tool call.
    Call(u32, ToolCall),
}

struct Loaded {
    tokenizer: Tokenizer,
    engine: Engine,
    tensor_count: usize,
    parameter_count: usize,
}

/// Tiny transition-logit architecture used for offline execution fixtures.
pub const FIXTURE_BIGRAM_ARCHITECTURE: &str = "abi-bigram-v1";

/// Default maximum verified artifact bytes loaded for one model: 16 GiB.
pub const DEFAULT_MAX_MODEL_BYTES: u64 = 16 * 1024 * 1024 * 1024;

/// Default maximum tokenizer artifact size: 64 MiB.
pub const DEFAULT_MAX_TOKENIZER_BYTES: u64 = 64 * 1024 * 1024;

/// Default maximum rendered prompt size before tokenization: 1 MiB.
pub const DEFAULT_MAX_PROMPT_BYTES: usize = 1024 * 1024;

/// Default hard generation ceiling independent of the request hint.
pub const DEFAULT_MAX_GENERATED_TOKENS: u32 = 256;

/// Explicit local-model load policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadConfig {
    /// Exact device request; no automatic selection exists.
    pub device: DevicePreference,
    /// Maximum sum of manifest-declared artifact bytes.
    pub max_model_bytes: u64,
    /// Maximum manifest-declared tokenizer bytes.
    pub max_tokenizer_bytes: u64,
    /// Maximum rendered prompt bytes accepted before tokenization.
    pub max_prompt_bytes: usize,
    /// Hard provider generation ceiling.
    pub max_generated_tokens: u32,
    /// Precision the floating-point weights are held in.
    pub weight_precision: WeightPrecision,
}

/// Precision the floating-point weights are converted to at load.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WeightPrecision {
    /// 32-bit floats.
    #[default]
    Single,
    /// 16-bit floats: half the weight memory. Whether a device executes every
    /// decoder operation at this precision is only known by running it.
    Half,
}

impl WeightPrecision {
    const fn dtype(self) -> DType {
        match self {
            Self::Single => DType::F32,
            Self::Half => DType::F16,
        }
    }

    /// Stable lowercase label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Single => "f32",
            Self::Half => "f16",
        }
    }
}

impl LoadConfig {
    /// Start a load policy with an explicit device selection.
    #[must_use]
    pub const fn new(device: DevicePreference) -> Self {
        Self {
            device,
            max_model_bytes: DEFAULT_MAX_MODEL_BYTES,
            max_tokenizer_bytes: DEFAULT_MAX_TOKENIZER_BYTES,
            max_prompt_bytes: DEFAULT_MAX_PROMPT_BYTES,
            max_generated_tokens: DEFAULT_MAX_GENERATED_TOKENS,
            weight_precision: WeightPrecision::Single,
        }
    }

    /// Hold the Gemma decoder weights at `precision`. The fixture
    /// architecture ignores this.
    #[must_use]
    pub const fn with_weight_precision(mut self, precision: WeightPrecision) -> Self {
        self.weight_precision = precision;
        self
    }

    /// Override the maximum aggregate artifact bytes.
    #[must_use]
    pub const fn with_max_model_bytes(mut self, maximum: u64) -> Self {
        self.max_model_bytes = maximum;
        self
    }

    /// Override the tokenizer-artifact byte ceiling.
    #[must_use]
    pub const fn with_max_tokenizer_bytes(mut self, maximum: u64) -> Self {
        self.max_tokenizer_bytes = maximum;
        self
    }

    /// Override the rendered-prompt byte ceiling.
    #[must_use]
    pub const fn with_max_prompt_bytes(mut self, maximum: usize) -> Self {
        self.max_prompt_bytes = maximum;
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
    engine: Engine,
    context_tokens: u32,
    manifest_output_tokens: u32,
    max_generated_tokens: u32,
    max_prompt_bytes: usize,
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
        let artifacts = verify_model_artifacts(manifest, storage, config)?;
        let initialized = initialize(config.device)?;
        let loaded = load_engine(
            manifest,
            &artifacts,
            &initialized.device,
            config.weight_precision.dtype(),
        )?;
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
            engine: loaded.engine,
            context_tokens: manifest.context.max_context_tokens,
            manifest_output_tokens: manifest.context.max_output_tokens,
            max_generated_tokens: config.max_generated_tokens,
            max_prompt_bytes: config.max_prompt_bytes,
            load_report,
            last_inference: Mutex::new(None),
        })
    }

    /// Whether the loaded architecture and tokenizer can take tool definitions.
    #[must_use]
    pub fn supports_tools(&self) -> bool {
        match &self.engine {
            Engine::Bigram { .. } | Engine::Gemma(_) => false,
            Engine::Gemma4(engine) => engine.supports_tools(),
        }
    }

    /// Whether the loaded model can be asked to reason and report it.
    #[must_use]
    pub fn supports_reasoning(&self) -> bool {
        match &self.engine {
            Engine::Bigram { .. } | Engine::Gemma(_) => false,
            Engine::Gemma4(engine) => engine.supports_reasoning(),
        }
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

    /// Drive the loaded architecture and retain execution evidence.
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
        if request.reasoning() && !self.supports_reasoning() {
            return Err(ModelRuntimeError::UnsupportedReasoning {
                model: self.model_id.clone(),
            });
        }
        if *request.tool_choice() != ToolChoice::Auto && !self.supports_tools() {
            return Err(ModelRuntimeError::UnsupportedToolChoice {
                model: self.model_id.clone(),
            });
        }
        if !request.tools().is_empty() && !self.supports_tools() {
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
            self.generate(request, &prompt_ids, generation_limit, run)?;
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
        let rendered = self.render_prompt(request)?;
        let prompt_ids = match &self.engine {
            Engine::Bigram { .. } => self
                .tokenizer
                .encode(rendered, false)
                .map_err(|error| self.tokenizer_error(error))?
                .get_ids()
                .to_vec(),
            Engine::Gemma(engine) => {
                engine.encode_chat(&self.model_id, &self.tokenizer, request)?
            }
            Engine::Gemma4(engine) => {
                engine.encode_chat(&self.model_id, &self.tokenizer, request)?
            }
        };
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
        request: &ModelRequest,
        prompt_ids: &[u32],
        limit: u64,
        run: &mut RunContext<'_>,
    ) -> Result<(usize, u64), ModelRuntimeError> {
        let mut native_operations = 0u64;
        let generated = match &self.engine {
            Engine::Bigram {
                transition,
                eos_token,
            } => {
                let start = prompt_ids.last().copied();
                self.stream(limit, run, |previous| {
                    let current = previous.or(start).expect("prepared prompt is non-empty");
                    let row = transition
                        .get(usize::try_from(current).unwrap_or(usize::MAX))
                        .and_then(|row| row.argmax(0))
                        .map_err(|error| self.candle_error(error))?;
                    native_operations = native_operations.saturating_add(2);
                    let next = row
                        .to_scalar::<u32>()
                        .map_err(|error| self.candle_error(error))?;
                    native_operations = native_operations.saturating_add(1);
                    Ok(if next == *eos_token {
                        Step::Stop
                    } else {
                        Step::Text(next)
                    })
                })?
            }
            Engine::Gemma(engine) => {
                let mut session = engine.session(&self.model_id)?;
                self.stream(limit, run, |previous| {
                    let next = match previous {
                        None => session.advance(prompt_ids),
                        Some(token) => session.advance(&[token]),
                    }
                    .map_err(|error| self.candle_error(error))?;
                    native_operations =
                        native_operations.saturating_add(GemmaSession::OPERATIONS_PER_STEP);
                    Ok(next.map_or(Step::Stop, Step::Text))
                })?
            }
            Engine::Gemma4(engine) => {
                let mut session = engine.session(&self.model_id, &self.tokenizer, request)?;
                self.stream(limit, run, |previous| {
                    let step = match previous {
                        None => session.advance(prompt_ids),
                        Some(token) => session.advance(&[token]),
                    }?;
                    native_operations =
                        native_operations.saturating_add(Gemma4Session::OPERATIONS_PER_STEP);
                    Ok(step)
                })?
            }
        };
        Ok((generated, native_operations))
    }

    /// Shared emission loop. `step` receives the previously generated token
    /// (`None` on the first call) and classifies the next one. Every generated
    /// token counts against `limit` and the output-token usage, shown or not.
    fn stream(
        &self,
        limit: u64,
        run: &mut RunContext<'_>,
        mut step: impl FnMut(Option<u32>) -> Result<Step, ModelRuntimeError>,
    ) -> Result<usize, ModelRuntimeError> {
        let mut previous = None;
        let mut generated = 0usize;
        let mut shown: Vec<u32> = Vec::new();
        let mut decoded = String::new();
        let mut thought: Vec<u32> = Vec::new();
        let mut thought_shown = String::new();
        for _ in 0..limit {
            let event = match step(previous)? {
                Step::Stop => break,
                Step::Hidden(token) => {
                    previous = Some(token);
                    None
                }
                Step::Call(token, call) => {
                    previous = Some(token);
                    Some(ModelEvent::ToolCall(call))
                }
                Step::Reasoning(token) => {
                    previous = Some(token);
                    thought.push(token);
                    let text = self
                        .tokenizer
                        .decode(&thought, true)
                        .map_err(|error| self.tokenizer_error(error))?;
                    let visible = reasoning_body(&text);
                    let delta = visible.strip_prefix(&thought_shown).ok_or_else(|| {
                        self.tokenizer_error("incremental decode changed emitted reasoning")
                    })?;
                    if delta.is_empty() {
                        None
                    } else {
                        let event = ModelEvent::reasoning(delta);
                        visible.clone_into(&mut thought_shown);
                        Some(event)
                    }
                }
                Step::Text(token) => {
                    previous = Some(token);
                    shown.push(token);
                    let next_decoded = self
                        .tokenizer
                        .decode(&shown, true)
                        .map_err(|error| self.tokenizer_error(error))?;
                    let delta = next_decoded.strip_prefix(&decoded).ok_or_else(|| {
                        self.tokenizer_error("incremental decode changed emitted text")
                    })?;
                    let event = ModelEvent::text(delta);
                    decoded = next_decoded;
                    Some(event)
                }
            };
            if let Some(event) = event
                && run.emit(&event) == Flow::Stop
            {
                break;
            }
            generated += 1;
            run.record_output_tokens(1);
        }
        Ok(generated)
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

    fn render_prompt(&self, request: &ModelRequest) -> Result<String, ModelRuntimeError> {
        let bytes = request
            .system()
            .into_iter()
            .chain(
                request
                    .messages()
                    .iter()
                    .map(|message| message.content.as_str()),
            )
            .try_fold(0usize, |sum, part| {
                sum.checked_add(part.len())?.checked_add(1)
            })
            .unwrap_or(usize::MAX)
            .saturating_sub(1);
        if bytes > self.max_prompt_bytes {
            return Err(ModelRuntimeError::PromptTooLarge {
                model: self.model_id.clone(),
                actual: bytes,
                maximum: self.max_prompt_bytes,
            });
        }
        Ok(render_prompt(request))
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
    config: LoadConfig,
) -> Result<ArtifactSummary, ModelRuntimeError> {
    let gemma = match manifest.architecture.as_str() {
        FIXTURE_BIGRAM_ARCHITECTURE => false,
        GEMMA3_TEXT_ARCHITECTURE | GEMMA4_TEXT_ARCHITECTURE => true,
        _ => {
            return Err(ModelRuntimeError::UnsupportedArchitecture {
                model: manifest.id.clone(),
                architecture: manifest.architecture.clone(),
            });
        }
    };
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
    if artifact_bytes > config.max_model_bytes {
        return Err(ModelRuntimeError::ModelTooLarge {
            model: manifest.id.clone(),
            actual: artifact_bytes,
            maximum: config.max_model_bytes,
        });
    }
    let tokenizer = manifest.tokenizer();
    if tokenizer.size_bytes > config.max_tokenizer_bytes {
        return Err(ModelRuntimeError::TokenizerTooLarge {
            model: manifest.id.clone(),
            actual: tokenizer.size_bytes,
            maximum: config.max_tokenizer_bytes,
        });
    }
    let weights: Vec<_> = manifest.weights().collect();
    // The fixture is one tensor file; Gemma publications ship ordered shards.
    if weights.is_empty() || (!gemma && weights.len() != 1) {
        return Err(ModelRuntimeError::WeightArtifactCount {
            model: manifest.id.clone(),
            actual: weights.len(),
        });
    }
    let configs: Vec<_> = manifest
        .artifacts
        .iter()
        .filter(|artifact| artifact.kind == ArtifactKind::Config)
        .collect();
    if gemma && configs.len() != 1 {
        return Err(ModelRuntimeError::ConfigArtifactCount {
            model: manifest.id.clone(),
            actual: configs.len(),
        });
    }
    if let Some(config) = configs
        .first()
        .filter(|config| config.size_bytes > MAX_CONFIG_BYTES)
    {
        return Err(ModelRuntimeError::ConfigTooLarge {
            model: manifest.id.clone(),
            actual: config.size_bytes,
            maximum: MAX_CONFIG_BYTES,
        });
    }
    for artifact in &manifest.artifacts {
        verify_artifact(&storage.artifact_path(manifest, artifact), artifact)?;
    }
    Ok(ArtifactSummary {
        weights_paths: weights
            .iter()
            .map(|artifact| storage.artifact_path(manifest, artifact))
            .collect(),
        config_path: gemma.then(|| storage.artifact_path(manifest, configs[0])),
        tokenizer_path: storage.artifact_path(manifest, tokenizer),
        artifact_bytes,
    })
}

/// The thought channel opens with a `thought` label line; reasoning is what
/// follows it. While only part of the label has arrived, nothing is shown.
pub(crate) fn reasoning_body(decoded: &str) -> &str {
    const LABEL: &str = "thought";
    let text = decoded.trim_start();
    match text.strip_prefix(LABEL) {
        Some(rest) => rest.trim_start(),
        None if LABEL.starts_with(text) => "",
        None => text,
    }
}

fn load_engine(
    manifest: &ModelManifest,
    artifacts: &ArtifactSummary,
    device: &Device,
    precision: DType,
) -> Result<Loaded, ModelRuntimeError> {
    let tokenizer = Tokenizer::from_file(&artifacts.tokenizer_path).map_err(|error| {
        ModelRuntimeError::Tokenizer {
            model: manifest.id.clone(),
            detail: error.to_string(),
        }
    })?;
    if let Some(config_path) = &artifacts.config_path
        && manifest.architecture == GEMMA4_TEXT_ARCHITECTURE
    {
        let loaded = Gemma4Engine::load(
            manifest,
            config_path,
            &artifacts.weights_paths,
            &tokenizer,
            device,
            precision,
        )?;
        return Ok(Loaded {
            tokenizer,
            engine: Engine::Gemma4(Box::new(loaded.engine)),
            tensor_count: loaded.tensor_count,
            parameter_count: loaded.parameter_count,
        });
    }
    if let Some(config_path) = &artifacts.config_path {
        let loaded = GemmaEngine::load(
            manifest,
            config_path,
            &artifacts.weights_paths,
            &tokenizer,
            device,
            precision,
        )?;
        return Ok(Loaded {
            tokenizer,
            engine: Engine::Gemma(Box::new(loaded.engine)),
            tensor_count: loaded.tensor_count,
            parameter_count: loaded.parameter_count,
        });
    }
    let weights_bytes = read(&artifacts.weights_paths[0])?;
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
    Ok(Loaded {
        engine: Engine::Bigram {
            transition: transition.clone(),
            eos_token,
        },
        tokenizer,
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
