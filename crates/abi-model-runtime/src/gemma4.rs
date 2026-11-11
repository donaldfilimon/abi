//! Gemma 4 text adapter: Candle's Gemma 4 text decoder with the Gemma 4 turn,
//! tool-declaration, tool-call, and tool-response grammar.
//!
//! Only the language model is loaded; vision and audio tensors in a published
//! checkpoint are never read. A request that asks for reasoning puts the
//! thinking switch at the start of the system turn and reports the thought
//! channel as reasoning; otherwise a thought channel the model opens is
//! consumed without being shown. A forced tool choice opens the call in the
//! prompt, so the model can only continue it.
//!
//! NOTE(SDK): Google's page shows an empty thought channel as an optional
//! stabilizer for larger models when thinking is off. This adapter does not
//! add it and ends the prompt at the open model turn.

use crate::gemma4_format::{self as format, Control, Piece};
use crate::provider::Step;
use crate::{ModelRuntimeError, shards};
use abi_agent_runtime::{ModelRequest, Role, ToolCall, ToolChoice};
use abi_models::ModelManifest;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::gemma4::{Gemma4TextConfig, text::TextModel};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};
use tokenizers::Tokenizer;

/// Gemma 4 text decoder architecture loaded from safetensors shards.
pub const GEMMA4_TEXT_ARCHITECTURE: &str = "gemma4-text-v1";

/// Tensor-name prefix of the language model inside a Gemma 4 checkpoint.
const LANGUAGE_MODEL_PREFIX: &str = "model.language_model.";

/// Control-token ids resolved from the verified tokenizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tokens {
    bos: u32,
    eos: u32,
    turn_start: u32,
    turn_end: u32,
    tools: Option<ToolTokens>,
    channel: Option<(u32, u32)>,
    think: Option<u32>,
}

/// The seven tokens the tool grammar needs. Tools are offered only when the
/// tokenizer carries all of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ToolTokens {
    tool_start: u32,
    tool_end: u32,
    call_start: u32,
    call_end: u32,
    response_start: u32,
    response_end: u32,
    quote: u32,
}

impl Tokens {
    fn resolve(manifest: &ModelManifest, tokenizer: &Tokenizer) -> Result<Self, ModelRuntimeError> {
        let required = |token: &str| {
            tokenizer
                .token_to_id(token)
                .ok_or_else(|| ModelRuntimeError::Tokenizer {
                    model: manifest.id.clone(),
                    detail: format!("required {token} token is absent"),
                })
        };
        let optional = |token: &str| tokenizer.token_to_id(token);
        let tools = (|| {
            Some(ToolTokens {
                tool_start: optional("<|tool>")?,
                tool_end: optional("<tool|>")?,
                call_start: optional("<|tool_call>")?,
                call_end: optional("<tool_call|>")?,
                response_start: optional("<|tool_response>")?,
                response_end: optional("<tool_response|>")?,
                quote: optional(format::QUOTE_TEXT)?,
            })
        })();
        Ok(Self {
            bos: required("<bos>")?,
            eos: required("<eos>")?,
            turn_start: required("<|turn>")?,
            turn_end: required("<turn|>")?,
            tools,
            channel: optional("<|channel>").zip(optional("<channel|>")),
            think: optional("<|think|>"),
        })
    }

    fn is_control(self, token: u32) -> bool {
        let tool = self.tools.is_some_and(|tools| {
            [
                tools.tool_start,
                tools.tool_end,
                tools.call_start,
                tools.call_end,
                tools.response_start,
                tools.response_end,
                tools.quote,
            ]
            .contains(&token)
        });
        let channel = self
            .channel
            .is_some_and(|(open, close)| token == open || token == close);
        tool || channel
            || self.think == Some(token)
            || [self.bos, self.eos, self.turn_start, self.turn_end].contains(&token)
    }

    fn control(self, control: Control) -> Option<u32> {
        Some(match control {
            Control::Bos => self.bos,
            Control::TurnStart => self.turn_start,
            Control::TurnEnd => self.turn_end,
            Control::ToolStart => self.tools?.tool_start,
            Control::ToolEnd => self.tools?.tool_end,
            Control::CallStart => self.tools?.call_start,
            Control::CallEnd => self.tools?.call_end,
            Control::ResponseStart => self.tools?.response_start,
            Control::ResponseEnd => self.tools?.response_end,
            Control::Quote => self.tools?.quote,
            Control::Think => self.think?,
        })
    }
}

/// Loaded Gemma 4 text decoder.
pub(crate) struct Gemma4Engine {
    model: Mutex<TextModel>,
    device: Device,
    tokens: Tokens,
}

/// Counts reported in the load evidence.
pub(crate) struct Gemma4Load {
    pub(crate) engine: Gemma4Engine,
    pub(crate) tensor_count: usize,
    pub(crate) parameter_count: usize,
}

impl Gemma4Engine {
    /// Build the decoder from already hash-verified artifacts.
    pub(crate) fn load(
        manifest: &ModelManifest,
        config_path: &Path,
        weights_paths: &[PathBuf],
        tokenizer: &Tokenizer,
        device: &Device,
        precision: DType,
    ) -> Result<Gemma4Load, ModelRuntimeError> {
        let config = read_config(manifest, config_path)?;
        let vocabulary = tokenizer.get_vocab_size(true);
        if vocabulary > config.vocab_size {
            return Err(config_error(
                manifest,
                format!(
                    "tokenizer vocabulary {vocabulary} exceeds config vocab_size {}",
                    config.vocab_size
                ),
            ));
        }
        if config.layer_types.len() != config.num_hidden_layers {
            return Err(config_error(
                manifest,
                format!(
                    "layer_types has {} entries for {} layers",
                    config.layer_types.len(),
                    config.num_hidden_layers
                ),
            ));
        }
        let tokens = Tokens::resolve(manifest, tokenizer)?;
        let shards = shards::read(
            manifest,
            weights_paths,
            Some(LANGUAGE_MODEL_PREFIX),
            precision,
        )?;
        let (tensor_count, parameter_count) = (shards.tensor_count, shards.parameter_count);
        let builder = VarBuilder::from_tensors(shards.tensors, precision, device)
            .pp("model")
            .pp("language_model");
        let model =
            TextModel::new(&config, builder).map_err(|error| candle_error(manifest, &error))?;
        Ok(Gemma4Load {
            engine: Self {
                model: Mutex::new(model),
                device: device.clone(),
                tokens,
            },
            tensor_count,
            parameter_count,
        })
    }

    /// Whether the tokenizer carries the full tool grammar.
    pub(crate) const fn supports_tools(&self) -> bool {
        self.tokens.tools.is_some()
    }

    /// Whether the tokenizer carries the thinking switch and thought channel.
    pub(crate) const fn supports_reasoning(&self) -> bool {
        self.tokens.think.is_some() && self.tokens.channel.is_some()
    }

    /// Render and tokenize the conversation, ending in an open model turn.
    pub(crate) fn encode_chat(
        &self,
        model_id: &str,
        tokenizer: &Tokenizer,
        request: &ModelRequest,
    ) -> Result<Vec<u32>, ModelRuntimeError> {
        let pieces = render_chat(model_id, request)?;
        let mut ids = Vec::new();
        for piece in &pieces {
            match piece {
                Piece::Control(control) => {
                    ids.push(self.tokens.control(*control).ok_or_else(|| {
                        if *control == Control::Think {
                            ModelRuntimeError::UnsupportedReasoning {
                                model: model_id.to_owned(),
                            }
                        } else {
                            ModelRuntimeError::UnsupportedTools {
                                model: model_id.to_owned(),
                            }
                        }
                    })?);
                }
                Piece::Structure(text) => ids.extend(encode(model_id, tokenizer, text)?),
                Piece::Text(text) => ids.extend(
                    encode(model_id, tokenizer, text)?
                        .into_iter()
                        .filter(|token| !self.tokens.is_control(*token)),
                ),
            }
        }
        Ok(ids)
    }

    /// Take exclusive use of the decoder and reset its key/value cache.
    pub(crate) fn session<'a>(
        &'a self,
        model_id: &'a str,
        tokenizer: &'a Tokenizer,
        request: &ModelRequest,
    ) -> Result<Gemma4Session<'a>, ModelRuntimeError> {
        let mut model = self
            .model
            .lock()
            .map_err(|_| ModelRuntimeError::ReportStatePoisoned {
                model: model_id.to_owned(),
            })?;
        model.clear_kv_cache();
        let mut filter = OutputFilter::new(
            self.tokens,
            request
                .tools()
                .iter()
                .map(|tool| tool.name.clone())
                .collect(),
            request.reasoning(),
        );
        if let Some(prefix) = forced_call_prefix(model_id, request)? {
            // The prompt already opened this call, so generation starts inside it.
            filter.mode = Mode::Call(encode(model_id, tokenizer, &prefix)?);
        }
        Ok(Gemma4Session {
            model,
            device: &self.device,
            position: 0,
            filter,
            model_id,
            tokenizer,
        })
    }
}

/// One exclusive generation over the decoder's key/value cache.
pub(crate) struct Gemma4Session<'a> {
    model: MutexGuard<'a, TextModel>,
    device: &'a Device,
    position: usize,
    filter: OutputFilter,
    model_id: &'a str,
    tokenizer: &'a Tokenizer,
}

impl Gemma4Session<'_> {
    /// Native Candle operations per [`Self::advance`]: decoder forward,
    /// argmax, and scalar extraction.
    pub(crate) const OPERATIONS_PER_STEP: u64 = 3;

    /// Feed `tokens`, greedily select the next token, and classify it.
    pub(crate) fn advance(&mut self, tokens: &[u32]) -> Result<Step, ModelRuntimeError> {
        let next = self
            .select(tokens)
            .map_err(|error| ModelRuntimeError::Candle {
                model: self.model_id.to_owned(),
                detail: error.to_string(),
            })?;
        let tokenizer = self.tokenizer;
        self.filter
            .accept(next, |body| {
                tokenizer
                    .decode(body, false)
                    .map_err(|error| error.to_string())
            })
            .map_err(|detail| ModelRuntimeError::MalformedToolCall {
                model: self.model_id.to_owned(),
                detail,
            })
    }

    fn select(&mut self, tokens: &[u32]) -> Result<u32, candle_core::Error> {
        let input = Tensor::new(tokens, self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, self.position)?;
        self.position = self.position.saturating_add(tokens.len());
        logits.flatten_all()?.argmax(0)?.to_scalar::<u32>()
    }
}

/// Where the generated stream currently is.
#[derive(Debug, PartialEq, Eq)]
enum Mode {
    /// Answer text.
    Answer,
    /// Inside a thought channel: consumed, never shown.
    Channel,
    /// Inside a tool call: body tokens collected until the call closes.
    Call(Vec<u32>),
}

/// Token-level classifier for generated output. It holds no model, so the
/// tool and channel grammar is testable on plain token sequences.
#[derive(Debug)]
struct OutputFilter {
    tokens: Tokens,
    offered: Vec<String>,
    mode: Mode,
    calls: usize,
    /// Whether thought-channel tokens are reported instead of hidden.
    reasoning: bool,
}

impl OutputFilter {
    const fn new(tokens: Tokens, offered: Vec<String>, reasoning: bool) -> Self {
        Self {
            tokens,
            offered,
            mode: Mode::Answer,
            calls: 0,
            reasoning,
        }
    }

    /// Classify one generated token. `decode` renders a call body with its
    /// string-delimiter tokens spelled out.
    fn accept(
        &mut self,
        token: u32,
        decode: impl FnOnce(&[u32]) -> Result<String, String>,
    ) -> Result<Step, String> {
        let tokens = self.tokens;
        let ends_turn = token == tokens.eos
            || token == tokens.turn_end
            || tokens
                .tools
                .is_some_and(|tools| token == tools.response_start);
        match &mut self.mode {
            Mode::Call(_) if ends_turn => Err("tool call was not closed".to_owned()),
            Mode::Call(body) => {
                if tokens.tools.is_some_and(|tools| token == tools.call_end) {
                    let text = decode(body)?;
                    self.mode = Mode::Answer;
                    let (name, arguments) = format::parse_call(&text)?;
                    if !self.offered.contains(&name) {
                        return Err(format!("tool '{name}' was not offered"));
                    }
                    self.calls += 1;
                    let id = format!("call_{}", self.calls);
                    return Ok(Step::Call(
                        token,
                        ToolCall::new(id, name, arguments.to_string()),
                    ));
                }
                body.push(token);
                Ok(Step::Hidden(token))
            }
            _ if ends_turn => Ok(Step::Stop),
            Mode::Channel => {
                if tokens.channel.is_some_and(|(_, close)| token == close) {
                    self.mode = Mode::Answer;
                    return Ok(Step::Hidden(token));
                }
                Ok(if self.reasoning && !tokens.is_control(token) {
                    Step::Reasoning(token)
                } else {
                    Step::Hidden(token)
                })
            }
            Mode::Answer => {
                if tokens.channel.is_some_and(|(open, _)| token == open) {
                    self.mode = Mode::Channel;
                    return Ok(Step::Hidden(token));
                }
                // With nothing offered there is no call to make: the token is
                // an ordinary hidden control token.
                if !self.offered.is_empty()
                    && tokens.tools.is_some_and(|tools| token == tools.call_start)
                {
                    self.mode = Mode::Call(Vec::new());
                    return Ok(Step::Hidden(token));
                }
                if tokens.is_control(token) {
                    return Ok(Step::Hidden(token));
                }
                Ok(Step::Text(token))
            }
        }
    }
}

/// Render the conversation as Gemma 4 pieces.
///
/// Tool results follow the agent host's convention: a [`Role::Tool`] message
/// whose content is a JSON object with `tool` and `payload`, plus optional
/// `input` (the call's JSON arguments) and `status`. Each is rendered as a
/// call/response pair inside a model turn. When the conversation ends on tool
/// results, that model turn is left open so the model continues it.
fn render_chat(model_id: &str, request: &ModelRequest) -> Result<Vec<Piece>, ModelRuntimeError> {
    let mut pieces = vec![Piece::Control(Control::Bos)];
    let mut system: Vec<&str> = request.system().into_iter().collect();
    let leading = request
        .messages()
        .iter()
        .take_while(|message| message.role == Role::System)
        .count();
    system.extend(
        request.messages()[..leading]
            .iter()
            .map(|message| message.content.as_str()),
    );
    if !system.is_empty() || !request.tools().is_empty() || request.reasoning() {
        pieces.push(Piece::Control(Control::TurnStart));
        pieces.push(Piece::Structure("system\n".to_owned()));
        if request.reasoning() {
            pieces.push(Piece::Control(Control::Think));
        }
        pieces.push(Piece::Text(system.join("\n\n").trim().to_owned()));
        for tool in request.tools() {
            if !format::valid_tool_name(&tool.name) {
                return Err(ModelRuntimeError::InvalidTool {
                    model: model_id.to_owned(),
                    detail: format!("tool name '{}' is not grammar-safe", tool.name),
                });
            }
            let parameters = if tool.input_schema.trim().is_empty() {
                None
            } else {
                Some(json(model_id, &tool.name, &tool.input_schema)?)
            };
            format::declaration(
                &mut pieces,
                &tool.name,
                &tool.description,
                parameters.as_ref(),
            );
        }
        pieces.push(Piece::Control(Control::TurnEnd));
        pieces.push(Piece::Structure("\n".to_owned()));
    }

    // `open` is true while a model turn carrying tool results awaits its text.
    let mut open = false;
    for message in &request.messages()[leading..] {
        if open && message.role != Role::Tool && message.role != Role::Assistant {
            pieces.push(Piece::Control(Control::TurnEnd));
            pieces.push(Piece::Structure("\n".to_owned()));
            open = false;
        }
        match message.role {
            Role::Tool => {
                if !open {
                    pieces.push(Piece::Control(Control::TurnStart));
                    pieces.push(Piece::Structure("model\n".to_owned()));
                    open = true;
                }
                tool_result(model_id, &mut pieces, &message.content)?;
            }
            Role::Assistant => {
                if !open {
                    pieces.push(Piece::Control(Control::TurnStart));
                    pieces.push(Piece::Structure("model\n".to_owned()));
                }
                pieces.push(Piece::Text(message.content.clone()));
                pieces.push(Piece::Control(Control::TurnEnd));
                pieces.push(Piece::Structure("\n".to_owned()));
                open = false;
            }
            Role::User | Role::System => {
                let role = if message.role == Role::User {
                    "user\n"
                } else {
                    "system\n"
                };
                pieces.push(Piece::Control(Control::TurnStart));
                pieces.push(Piece::Structure(role.to_owned()));
                pieces.push(Piece::Text(message.content.trim().to_owned()));
                pieces.push(Piece::Control(Control::TurnEnd));
                pieces.push(Piece::Structure("\n".to_owned()));
            }
        }
    }
    if !open {
        pieces.push(Piece::Control(Control::TurnStart));
        pieces.push(Piece::Structure("model\n".to_owned()));
    }
    if let Some(prefix) = forced_call_prefix(model_id, request)? {
        pieces.push(Piece::Control(Control::CallStart));
        pieces.push(Piece::Structure(prefix));
    }
    Ok(pieces)
}

/// The call text a forced tool choice writes into the open model turn, so the
/// model can only continue a call: `call:` for any tool, `call:NAME{` for one.
fn forced_call_prefix(
    model_id: &str,
    request: &ModelRequest,
) -> Result<Option<String>, ModelRuntimeError> {
    let invalid = |detail: String| ModelRuntimeError::InvalidTool {
        model: model_id.to_owned(),
        detail,
    };
    match request.tool_choice() {
        ToolChoice::Auto => Ok(None),
        ToolChoice::Required if request.tools().is_empty() => Err(invalid(
            "a tool call is required but no tools were offered".to_owned(),
        )),
        ToolChoice::Required => Ok(Some("call:".to_owned())),
        ToolChoice::Named(name) => {
            if request.tools().iter().any(|tool| &tool.name == name) {
                Ok(Some(format!("call:{name}{{")))
            } else {
                Err(invalid(format!("required tool '{name}' was not offered")))
            }
        }
    }
}

fn tool_result(
    model_id: &str,
    pieces: &mut Vec<Piece>,
    content: &str,
) -> Result<(), ModelRuntimeError> {
    let invalid = |detail: String| ModelRuntimeError::InvalidTool {
        model: model_id.to_owned(),
        detail,
    };
    let envelope: Value = serde_json::from_str(content).map_err(|error| {
        invalid(format!(
            "tool result is not the host JSON envelope: {error}"
        ))
    })?;
    let name = envelope
        .get("tool")
        .and_then(Value::as_str)
        .filter(|name| format::valid_tool_name(name))
        .ok_or_else(|| invalid("tool result has no grammar-safe 'tool' name".to_owned()))?;
    let payload = envelope
        .get("payload")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let arguments = envelope
        .get("input")
        .and_then(Value::as_str)
        .and_then(|input| serde_json::from_str::<Value>(input).ok())
        .filter(Value::is_object)
        .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
    let succeeded = envelope
        .get("status")
        .and_then(Value::as_str)
        .is_none_or(|status| status == "ok");
    let response = if succeeded {
        // A JSON object payload keeps its fields; any other payload is text.
        serde_json::from_str::<Value>(payload)
            .ok()
            .filter(Value::is_object)
            .unwrap_or_else(|| Value::String(payload.to_owned()))
    } else {
        serde_json::json!({ "error": payload })
    };
    format::call_and_response(pieces, name, &arguments, &response);
    Ok(())
}

fn json(model_id: &str, tool: &str, document: &str) -> Result<Value, ModelRuntimeError> {
    serde_json::from_str(document).map_err(|error| ModelRuntimeError::InvalidTool {
        model: model_id.to_owned(),
        detail: format!("tool '{tool}' has an input schema that is not JSON: {error}"),
    })
}

fn encode(
    model_id: &str,
    tokenizer: &Tokenizer,
    text: &str,
) -> Result<Vec<u32>, ModelRuntimeError> {
    tokenizer
        .encode(text, false)
        .map(|encoding| encoding.get_ids().to_vec())
        .map_err(|error| ModelRuntimeError::Tokenizer {
            model: model_id.to_owned(),
            detail: error.to_string(),
        })
}

fn read_config(
    manifest: &ModelManifest,
    path: &Path,
) -> Result<Gemma4TextConfig, ModelRuntimeError> {
    let bytes = std::fs::read(path).map_err(|source| ModelRuntimeError::io(path, source))?;
    let document: Value =
        serde_json::from_slice(&bytes).map_err(|error| config_error(manifest, error))?;
    let text = match document.get("text_config") {
        Some(nested) => nested.clone(),
        None => document,
    };
    serde_json::from_value(text).map_err(|error| config_error(manifest, error))
}

fn config_error(manifest: &ModelManifest, detail: impl std::fmt::Display) -> ModelRuntimeError {
    ModelRuntimeError::Config {
        model: manifest.id.clone(),
        detail: detail.to_string(),
    }
}

fn candle_error(manifest: &ModelManifest, error: &candle_core::Error) -> ModelRuntimeError {
    ModelRuntimeError::Candle {
        model: manifest.id.clone(),
        detail: error.to_string(),
    }
}

#[cfg(test)]
#[path = "gemma4_tests.rs"]
mod tests;
