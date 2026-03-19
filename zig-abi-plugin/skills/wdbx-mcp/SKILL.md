---
name: wdbx-mcp
description: Use when interacting with the WDBX vector database via MCP tools, setting up the WDBX MCP server, troubleshooting database connectivity, or learning about available database operations and their parameters
---

# WDBX MCP Integration

Connect to the ABI Framework's WDBX vector database through the Model Context Protocol. This skill covers server setup, available tools, query patterns, and troubleshooting.

## Quick Start

1. Ensure the `zig-abi` plugin is installed and the `.mcp.json` is loaded
2. Run `/mcp` to verify the `wdbx` server appears
3. Use `db_stats` to confirm connectivity

## Server Configuration

The WDBX MCP server is configured in `zig-abi-plugin/.mcp.json`:

```json
{
  "wdbx": {
    "command": "abi",
    "args": ["mcp", "serve", "--db"],
    "env": {
      "ABI_DB_PATH": "${ABI_DB_PATH}"
    }
  }
}
```

**Transport:** stdio (JSON-RPC 2.0 over stdin/stdout)
**Binary:** `abi` must be on PATH or built via `zig build run`
**Env var:** `ABI_DB_PATH` sets the database storage directory (optional, defaults to `./data/wdbx/`)

## Available Tools

### Core CRUD

| Tool | Description | Required Params | Optional Params |
|------|-------------|-----------------|-----------------|
| `db_insert` | Insert vector with metadata | `id`, `vector` | `metadata`, `db_name` |
| `db_get` | Retrieve vector by ID | `id` | `db_name` |
| `db_update` | Update existing vector | `id`, `vector` | `db_name` |
| `db_delete` | Remove vector by ID | `id` | `db_name` |
| `db_list` | List stored vectors | — | `limit`, `db_name` |

### Search

| Tool | Description | Required Params | Optional Params |
|------|-------------|-----------------|-----------------|
| `db_query` | Cosine similarity search | `vector` | `top_k`, `db_name` |

### Operations

| Tool | Description | Required Params | Optional Params |
|------|-------------|-----------------|-----------------|
| `db_stats` | Vector count, dimensions, memory | — | `db_name` |
| `db_backup` | Save database to file | `path` | `db_name` |
| `db_diagnostics` | Performance metrics | — | `db_name` |

## Usage Patterns

### Semantic Search

```
Tool: db_query
Params: {"vector": [0.1, 0.2, ...], "top_k": 5, "db_name": "default"}

Score interpretation:
  > 0.8  — Strong semantic match
  0.5-0.8 — Related content
  < 0.5  — Weak or coincidental similarity
```

### Store Knowledge

```
Tool: db_insert
Params: {
  "id": 42,
  "vector": [0.1, 0.2, ...],
  "metadata": "architecture-decision: use types.zig for shared types across mod/stub",
  "db_name": "decisions"
}
```

### Health Check

```
Tool: db_stats
Params: {"db_name": "default"}

Returns: vector count, dimensions, memory bytes, norm cache status
```

### Backup Before Destructive Operations

```
Tool: db_backup
Params: {"path": "/tmp/wdbx-backup-2026-03-19.db", "db_name": "default"}
```

## Multi-Database Support

All tools accept `db_name` (defaults to `"default"`). Suggested databases:

| Database | Purpose |
|----------|---------|
| `default` | General-purpose vector store |
| `code-patterns` | Reusable code pattern embeddings |
| `decisions` | Architecture decision records |
| `sessions` | Cross-session conversation context |

## Troubleshooting

### Server not appearing in `/mcp`

1. Check that `abi` binary is on PATH: `which abi`
2. If not built: `zig build run -- mcp serve --db` to test directly
3. Verify `.mcp.json` is at plugin root: `zig-abi-plugin/.mcp.json`
4. Restart Claude Code after config changes

### Tools fail with connection error

1. The `abi` binary may not be built — run `zig build` first
2. On Darwin 25+, use `./tools/scripts/run_build.sh run -- mcp serve --db`
3. Check `ABI_DB_PATH` if set — directory must exist and be writable

### Database appears empty

1. Run `db_stats` to confirm — dimension=0 means no vectors inserted yet
2. The database creates lazily on first insert
3. Check `db_name` — you may be querying a different database than expected

## Implementation Details

**Source:** `src/services/mcp/real.zig` — tool definitions and handlers
**Server:** `src/services/mcp/server.zig` — JSON-RPC 2.0 protocol
**Types:** `src/services/mcp/types.zig` — protocol message types
**Database:** `src/core/database/mod.zig` — WDBX engine (SIMD-accelerated cosine similarity)

The MCP server is gated by `build_options.feat_mcp` and the database tools are gated by `build_options.feat_database`. Both default to `true`.
