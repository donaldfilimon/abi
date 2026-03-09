# mcp

> MCP (Model Context Protocol) Service

Provides a JSON-RPC 2.0 server over stdio for exposing ABI framework
tools to MCP-compatible AI clients (Claude Desktop, Cursor, etc.). The default
server now combines WDBX database tools and ZLS tools in one process.

## Usage
```bash
abi mcp serve                          # Start MCP server (stdio)
echo '{"jsonrpc":"2.0","method":"initialize","id":1,"params":{}}' | abi mcp serve
```

## Exposed Tools
- `db_*` — WDBX database tools
- `zls_*` — ZLS LSP tools (hover, completion, definition, etc.)

**Source:** [`src/services/mcp/mod.zig`](../../src/services/mcp/mod.zig)

**Availability:** Always enabled

---

## API

### <a id="pub-fn-createcombinedserver-allocator-std-mem-allocator-version-const-u8-server"></a>`pub fn createCombinedServer(allocator: std.mem.Allocator, version: []const u8) !Server`

<sup>**fn**</sup> | [source](../../src/services/mcp/mod.zig#L30)

Create the default MCP server with both WDBX and ZLS tools registered.



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use the repo-local CEL toolchain for ABI Zig validation, docs generation, and build-wiring changes when Darwin blocks stock Zig.
