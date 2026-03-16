---
title: mcp API
purpose: Generated API reference for mcp
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2905+5d71e3051
---

# mcp

> MCP (Model Context Protocol) Service

Provides a JSON-RPC 2.0 server over stdio for exposing ABI framework
tools to MCP-compatible AI clients (Claude Desktop, Cursor, etc.).

## Usage
```bash
abi mcp serve                          # Start MCP server (stdio)
echo '{"jsonrpc":"2.0","method":"initialize","id":1,"params":{}}' | abi mcp serve
```

## Exposed Tools
- `db_*` — Database tools
- `zls_*` — ZLS LSP tools (hover, completion, definition, etc.)

**Source:** [`src/services/mcp/mod.zig`](../../src/services/mcp/mod.zig)

**Availability:** Always enabled

---

## API

### <a id="pub-fn-createcombinedserver-allocator-std-mem-allocator-version-const-u8-server"></a>`pub fn createCombinedServer(allocator: std.mem.Allocator, version: []const u8) !Server`

<sup>**fn**</sup> | [source](../../src/services/mcp/mod.zig#L26)

Create an MCP server pre-configured with both database and ZLS tools

### <a id="pub-fn-createdatabaseserver-allocator-std-mem-allocator-version-const-u8-server"></a>`pub fn createDatabaseServer(allocator: std.mem.Allocator, version: []const u8) !Server`

<sup>**fn**</sup> | [source](../../src/services/mcp/mod.zig#L47)

Create an MCP server pre-configured with database tools



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` on supported hosts. On Darwin 25+ / 26+, use `zig fmt --check ...` plus `./tools/scripts/run_build.sh <step>`. For docs generation, use `zig build gendocs` or `./tools/scripts/run_build.sh gendocs` on Darwin.
