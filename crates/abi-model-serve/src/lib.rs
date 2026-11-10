//! Loopback `OpenAI`-compatible chat-completions server over one loaded
//! [`abi_agent_runtime::ModelProvider`].
//!
//! The server binds `127.0.0.1` only, serves exactly the model it was started
//! with, and never substitutes another. It answers `GET /health`,
//! `GET /v1/models`, and `POST /v1/chat/completions` (buffered JSON or
//! server-sent events). Requests are handled one at a time: one loaded model
//! runs one generation.
//!
//! Tool definitions are accepted only when the loaded model takes them;
//! generated calls come back as `tool_calls`, and `tool` messages are passed to
//! the provider in the agent host's JSON envelope. Unsupported tools, forced
//! `tool_choice`, multiple choices, and non-text content parts are refused
//! with an `OpenAI`-shaped `400` instead of being ignored. An optional bearer
//! token is loopback hardening, not a TLS substitute.

mod http;
mod protocol;
mod server;

pub use http::{HttpError, MAX_HEADER_BYTES, Request};
pub use protocol::{ChatRequest, ProtocolError, finish_reason, parse_chat};
pub use server::{DEFAULT_MAX_BODY_BYTES, DEFAULT_PORT, ServeConfig, Server};
