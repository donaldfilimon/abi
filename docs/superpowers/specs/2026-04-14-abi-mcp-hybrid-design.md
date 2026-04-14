# Hybrid abi-mcp Integration Design

**Goal:** Enable interactive manual testing and shell-based convenience for the `abi-mcp` server.

**Architecture:**
- **Interactive Mode:** Extend `abi-mcp` binary with a `--debug` mode that handles JSON-RPC framing for manual testing.
- **Convenience Layer:** Shell script utility for common ABI framework operations, providing a CLI abstraction for MCP tools.

**Components:**
1. **Debug Mode:** Read-Eval-Print Loop (REPL) processing raw JSON-RPC requests, handling framing for stdin/stdout.
2. **Shell Utils:** A `tools/abi-mcp-utils.sh` file with semantic functions translating simple CLI commands to JSON-RPC payloads.

**Testing Strategy:**
- Manual verification of REPL interaction with `abi-mcp --debug`.
- Integration testing for shell aliases using a health-check tool.
