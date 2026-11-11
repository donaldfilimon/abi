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
//! the provider in the agent host's JSON envelope. `tool_choice` may be `auto`,
//! `none`, `required`, or a named function. `reasoning_effort` other than
//! `none` asks the model to reason; the reasoning comes back as
//! `reasoning_content`, a widely used extension that is not part of the
//! `OpenAI` chat-completions schema. Anything the loaded model cannot take,
//! multiple choices, and non-text content parts are refused with an
//! `OpenAI`-shaped `400` instead of being ignored. An optional bearer
//! token is loopback hardening, not a TLS substitute.

mod http;
mod protocol;
mod server;

pub use http::{HttpError, MAX_HEADER_BYTES, Request};
pub use protocol::{Capabilities, ChatRequest, ProtocolError, finish_reason, parse_chat};
pub use server::{DEFAULT_MAX_BODY_BYTES, DEFAULT_PORT, ServeConfig, Server};
