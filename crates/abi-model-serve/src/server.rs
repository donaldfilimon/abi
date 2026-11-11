//! Loopback listener: authentication, routing, and provider dispatch.

use crate::http::{self, HttpError, Request};
use crate::protocol::{self, Capabilities, ProtocolError, Turn};
use abi_agent_runtime::{
    CancellationToken, CollectingSink, EventSink, ModelEvent, ModelProvider, RunBudget,
    run_provider,
};
use serde_json::json;
use std::io::Write;
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Default loopback port.
pub const DEFAULT_PORT: u16 = 8090;

/// Default request-body ceiling: 1 MiB.
pub const DEFAULT_MAX_BODY_BYTES: usize = 1024 * 1024;

const JSON: &str = "application/json";
const ACCEPT_POLL: Duration = Duration::from_millis(20);

/// Listener policy. Every limit is explicit.
#[derive(Clone, Debug)]
pub struct ServeConfig {
    /// The one model identifier this server answers for.
    pub model_id: String,
    /// Bearer token required on `/v1/*` when set.
    pub bearer_token: Option<String>,
    /// Largest accepted request body.
    pub max_body_bytes: usize,
    /// Socket read and write timeout.
    pub io_timeout: Duration,
    /// Budget applied to every generation.
    pub budget: RunBudget,
    /// Whether the loaded model takes tool definitions.
    pub tools: bool,
    /// Whether the loaded model can be asked to reason.
    pub reasoning: bool,
}

impl ServeConfig {
    /// Policy for `model_id` with default limits and no bearer token.
    #[must_use]
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            bearer_token: None,
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            io_timeout: Duration::from_secs(30),
            budget: RunBudget::unlimited(),
            tools: false,
            reasoning: false,
        }
    }

    /// Accept `reasoning_effort` and report reasoning as `reasoning_content`.
    #[must_use]
    pub const fn with_reasoning(mut self, reasoning: bool) -> Self {
        self.reasoning = reasoning;
        self
    }

    /// Accept tool definitions and report generated tool calls.
    #[must_use]
    pub const fn with_tools(mut self, tools: bool) -> Self {
        self.tools = tools;
        self
    }

    /// Require `Authorization: Bearer <token>` on `/v1/*`.
    #[must_use]
    pub fn with_bearer_token(mut self, token: impl Into<String>) -> Self {
        self.bearer_token = Some(token.into());
        self
    }

    /// Override the request-body ceiling.
    #[must_use]
    pub const fn with_max_body_bytes(mut self, maximum: usize) -> Self {
        self.max_body_bytes = maximum;
        self
    }

    /// Override the socket timeout.
    #[must_use]
    pub const fn with_io_timeout(mut self, timeout: Duration) -> Self {
        self.io_timeout = timeout;
        self
    }

    /// Override the per-generation budget.
    #[must_use]
    pub const fn with_budget(mut self, budget: RunBudget) -> Self {
        self.budget = budget;
        self
    }
}

/// A bound loopback server over one provider.
pub struct Server {
    listener: TcpListener,
    provider: Arc<dyn ModelProvider>,
    config: ServeConfig,
    sequence: AtomicU64,
}

impl std::fmt::Debug for Server {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Server")
            .field("model_id", &self.config.model_id)
            .field("authenticated", &self.config.bearer_token.is_some())
            .finish_non_exhaustive()
    }
}

impl Server {
    /// Bind `127.0.0.1:port`. Port `0` asks the OS for a free port.
    pub fn bind(
        port: u16,
        provider: Arc<dyn ModelProvider>,
        config: ServeConfig,
    ) -> std::io::Result<Self> {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, port))?;
        Ok(Self {
            listener,
            provider,
            config,
            sequence: AtomicU64::new(0),
        })
    }

    /// The bound loopback address.
    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.listener.local_addr()
    }

    /// Accept and answer connections, one at a time, until `stop` is set.
    pub fn serve(&self, stop: &AtomicBool) -> std::io::Result<()> {
        self.listener.set_nonblocking(true)?;
        while !stop.load(Ordering::SeqCst) {
            match self.listener.accept() {
                Ok((stream, _peer)) => {
                    // A failed connection is the peer's problem, not the
                    // listener's: report it and keep serving.
                    if let Err(error) = self.answer(stream) {
                        eprintln!("abi-model-serve: connection ended early: {error}");
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(ACCEPT_POLL);
                }
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    fn answer(&self, mut stream: TcpStream) -> std::io::Result<()> {
        stream.set_nonblocking(false)?;
        stream.set_read_timeout(Some(self.config.io_timeout))?;
        stream.set_write_timeout(Some(self.config.io_timeout))?;
        let request = match http::read_request(&mut stream, self.config.max_body_bytes) {
            Ok(request) => request,
            Err(error) => return refuse_framing(&mut stream, error),
        };
        match self.route(&request, &mut stream) {
            Ok(()) => Ok(()),
            Err(Reply::Refusal(error)) => http::write_response(
                &mut stream,
                (error.status, error.reason),
                JSON,
                error.body().as_bytes(),
            ),
            Err(Reply::Io(error)) => Err(error),
        }
    }

    fn route(&self, request: &Request, stream: &mut TcpStream) -> Result<(), Reply> {
        let created = unix_seconds();
        match (request.method.as_str(), request.path.as_str()) {
            ("GET", "/health") => {
                let body = json!({ "status": "ok", "model": self.config.model_id }).to_string();
                Ok(http::write_response(
                    stream,
                    (200, "OK"),
                    JSON,
                    body.as_bytes(),
                )?)
            }
            ("GET", "/v1/models") => {
                self.authorize(request)?;
                let body = protocol::models_body(&self.config.model_id, created);
                Ok(http::write_response(
                    stream,
                    (200, "OK"),
                    JSON,
                    body.as_bytes(),
                )?)
            }
            ("POST", "/v1/chat/completions") => {
                self.authorize(request)?;
                self.complete(request, stream, created)
            }
            (_, "/health" | "/v1/models" | "/v1/chat/completions") => Err(ProtocolError::new(
                405,
                "Method Not Allowed",
                "invalid_request_error",
                "method_not_allowed",
                "method is not allowed for this path",
            )
            .into()),
            _ => Err(ProtocolError::new(
                404,
                "Not Found",
                "invalid_request_error",
                "not_found",
                "unknown path",
            )
            .into()),
        }
    }

    fn authorize(&self, request: &Request) -> Result<(), ProtocolError> {
        let Some(expected) = &self.config.bearer_token else {
            return Ok(());
        };
        let presented = request
            .header("authorization")
            .and_then(|value| value.strip_prefix("Bearer "))
            .unwrap_or_default();
        if constant_time_eq(presented.as_bytes(), expected.as_bytes()) {
            Ok(())
        } else {
            Err(ProtocolError::new(
                401,
                "Unauthorized",
                "authentication_error",
                "invalid_api_key",
                "missing or incorrect bearer token",
            ))
        }
    }

    fn complete(
        &self,
        request: &Request,
        stream: &mut TcpStream,
        created: u64,
    ) -> Result<(), Reply> {
        let capabilities = Capabilities {
            tools: self.config.tools,
            reasoning: self.config.reasoning,
        };
        let chat = protocol::parse_chat(&request.body, capabilities)?;
        let model = chat.request.model();
        // Exact selection only: a different name is never served by this model.
        if model != self.config.model_id {
            return Err(ProtocolError::new(
                404,
                "Not Found",
                "invalid_request_error",
                "model_not_found",
                format!("model '{model}' is not loaded by this server"),
            )
            .into());
        }
        let id = format!(
            "chatcmpl-{created}-{}",
            self.sequence.fetch_add(1, Ordering::Relaxed)
        );
        let cancel = CancellationToken::new();
        if !chat.stream {
            let mut sink = CollectingSink::new();
            let report = run_provider(
                self.provider.as_ref(),
                &chat.request,
                &mut sink,
                &cancel,
                self.config.budget,
            )
            .map_err(|error| provider_failure(&error))?;
            let calls: Vec<_> = sink
                .events()
                .iter()
                .filter_map(|event| match event {
                    ModelEvent::ToolCall(call) => Some(call),
                    _ => None,
                })
                .collect();
            let reasoning: String = sink
                .events()
                .iter()
                .filter_map(|event| match event {
                    ModelEvent::ReasoningDelta { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            let body = protocol::completion_body(
                &id,
                created,
                model,
                &Turn {
                    text: &sink.text(),
                    reasoning: &reasoning,
                    calls: &calls,
                },
                report.stop,
                &report.usage,
            );
            return Ok(http::write_response(
                stream,
                (200, "OK"),
                JSON,
                body.as_bytes(),
            )?);
        }

        http::write_sse_head(stream)?;
        let role = json!({ "role": "assistant", "content": "" });
        http::write_sse_event(
            stream,
            &protocol::chunk_body(&id, created, model, &role, None),
        )?;
        let mut sink = SseSink {
            stream: &mut *stream,
            cancel: &cancel,
            id: &id,
            created,
            model,
            calls: 0,
        };
        let outcome = run_provider(
            self.provider.as_ref(),
            &chat.request,
            &mut sink,
            &cancel,
            self.config.budget,
        );
        let made_calls = sink.calls > 0;
        match outcome {
            Ok(report) => {
                let reason = protocol::finish_reason(report.stop, made_calls);
                let finish = Some((reason, &report.usage));
                let last = protocol::chunk_body(&id, created, model, &json!({}), finish);
                http::write_sse_event(stream, &last)?;
                http::write_sse_event(stream, "[DONE]")?;
            }
            // The head is already sent, so the failure travels in-band and the
            // stream ends without the terminal marker.
            Err(error) => http::write_sse_event(stream, &provider_failure(&error).body())?,
        }
        Ok(())
    }
}

/// Forwards text deltas and tool calls as chunks; a failed write cancels the
/// generation.
struct SseSink<'a> {
    stream: &'a mut TcpStream,
    cancel: &'a CancellationToken,
    id: &'a str,
    created: u64,
    model: &'a str,
    calls: usize,
}

impl EventSink for SseSink<'_> {
    fn emit(&mut self, event: &ModelEvent) {
        let delta = match event {
            ModelEvent::TextDelta { text } => json!({ "content": text }),
            ModelEvent::ReasoningDelta { text } => json!({ "reasoning_content": text }),
            ModelEvent::ToolCall(call) => {
                let entry = protocol::tool_call_value(call, Some(self.calls));
                self.calls += 1;
                json!({ "tool_calls": [entry] })
            }
            _ => return,
        };
        let chunk = protocol::chunk_body(self.id, self.created, self.model, &delta, None);
        if http::write_sse_event(self.stream, &chunk).is_err() {
            self.cancel.cancel();
        }
    }
}

enum Reply {
    Refusal(ProtocolError),
    Io(std::io::Error),
}

impl From<ProtocolError> for Reply {
    fn from(error: ProtocolError) -> Self {
        Self::Refusal(error)
    }
}

impl From<std::io::Error> for Reply {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

fn provider_failure(error: &abi_agent_runtime::RuntimeError) -> ProtocolError {
    ProtocolError::new(
        500,
        "Internal Server Error",
        "server_error",
        "provider_error",
        error.to_string(),
    )
}

fn refuse_framing(stream: &mut impl Write, error: HttpError) -> std::io::Result<()> {
    let (status, reason) = error.status();
    let refusal = ProtocolError::new(
        status,
        reason,
        "invalid_request_error",
        "invalid_http",
        reason,
    );
    http::write_response(stream, (status, reason), JSON, refusal.body().as_bytes())
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

/// Compare without an early exit on the first differing byte.
fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut difference = left.len() ^ right.len();
    for index in 0..left.len().max(right.len()) {
        let a = left.get(index).copied().unwrap_or(0);
        let b = right.get(index).copied().unwrap_or(0);
        difference |= usize::from(a ^ b);
    }
    difference == 0
}

#[cfg(test)]
mod tests {
    use super::constant_time_eq;

    #[test]
    fn token_comparison_is_exact() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secreT"));
        assert!(!constant_time_eq(b"secret", b"secret1"));
        assert!(!constant_time_eq(b"", b"secret"));
        assert!(constant_time_eq(b"", b""));
    }
}
