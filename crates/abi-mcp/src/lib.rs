//! The ABI MCP server: JSON-RPC protocol, the frozen 12-tool surface, and the
//! stdio transport.
//!
//! Ported from `src/mcp/`. The HTTP/SSE transport is not yet ported — see
//! `RUST-REWRITE-PLAN.md` step 9 — so this crate only offers stdio today,
//! which is the transport Claude Code and other local MCP clients use.

pub mod ai_tools;
pub mod connector_tools;
pub mod handlers;
pub mod middleware;
pub mod plugin_tools;
pub mod protocol;
pub mod rpc;
pub mod state;
pub mod stdio;

pub use state::McpState;
