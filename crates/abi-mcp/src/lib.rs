//! The ABI MCP server: JSON-RPC protocol, the frozen 12-tool surface, and the
//! stdio + loopback HTTP/SSE transports.
//!
//! Ported from `src/mcp/`. HTTP/SSE listens on `127.0.0.1` only
//! (`ABI_MCP_HTTP_PORT`, optional `ABI_MCP_HTTP_TOKEN`).

pub mod ai_tools;
pub mod connector_tools;
pub mod handlers;
pub mod http;
pub mod middleware;
pub mod plugin_tools;
pub mod protocol;
pub mod rpc;
pub mod state;
pub mod stdio;

pub use http::{DEFAULT_HTTP_PORT, HttpConfig, McpHttpServer};
pub use state::McpState;
