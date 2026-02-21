# mcp

> MCP (Model Context Protocol) server for WDBX database.

**Source:** [`src/services/mcp/mod.zig`](../../src/services/mcp/mod.zig)

**Availability:** Always enabled

---

MCP (Model Context Protocol) Service

Provides a JSON-RPC 2.0 server over stdio for exposing ABI framework
tools to MCP-compatible AI clients (Claude Desktop, Cursor, etc.).

## Usage
```bash
abi mcp serve                          # Start MCP server (stdio)
echo '{"jsonrpc":"2.0","method":"initialize","id":1,"params":{}}' | abi mcp serve
```

## Exposed Tools
- `db_query` — Vector similarity search
- `db_insert` — Insert vectors with metadata
- `db_stats` — Database statistics
- `db_list` — List stored vectors
- `db_delete` — Delete a vector by ID

---

## API

### `pub fn createWdbxServer(allocator: std.mem.Allocator, version: []const u8) !Server`

<sup>**fn**</sup>

Create an MCP server pre-configured with WDBX database tools

---

*Generated automatically by `zig build gendocs`*
