//! `abi-mcp`: the stdio MCP server entry point.
//!
//! Ported from `src/mcp/main.zig`, stdio only — the HTTP/SSE transport is not
//! yet ported (`RUST-REWRITE-PLAN.md` step 9).

fn main() {
    let state = abi_mcp::McpState::new();
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    abi_mcp::stdio::run_loop(state, stdin.lock(), stdout.lock());
}
