# zig-abi-plugin v0.6.0

Claude Code plugin for ABI Framework development. Provides smart build routing, Zig 0.16 patterns, feature module scaffolding, WDBX vector database MCP integration, and real-time verification.

## Installation

```bash
claude --plugin-dir zig-abi-plugin
```

## Components

### Commands

| Command | Purpose |
|---------|---------|
| `/zig-abi:build [step]` | Smart build with Darwin workaround detection |
| `/zig-abi:check [scope]` | Verification checks (format, imports, stub-sync, deprecated) |
| `/zig-abi:new-feature <name>` | Scaffold new feature module (9-step process) |

### Skills

| Skill | Trigger |
|-------|---------|
| `zig-016-patterns` | Writing Zig code, compilation errors, API questions |
| `abi-architecture` | Feature modules, build system, comptime gating |
| `abi-code-review` | Code review with ABI-specific heuristics |
| `cel-language` | CEL expression syntax, evaluation, policy rules |
| `wdbx-mcp` | WDBX vector database operations, MCP server setup, query patterns |

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/load-context.sh` | SessionStart hook: platform detection, Zig version, task reminders |
| `scripts/check-flag-sync.sh` | Validate feature flag counts across build files |
| `scripts/audit-darwin-targets.sh` | Audit darwinRelink() wiring on executables |
| `skills/abi-code-review/scripts/review_prep.py` | Prepare ABI-specific review context from diffs |

### Agents

| Agent | Purpose |
|--------|---------|
| `stub-sync-validator` | Proactive mod/stub signature checking after feature edits |
| `wdbx-mcp` | Semantic search over WDBX knowledge base, vector data management |

### MCP Servers

The plugin ships an `.mcp.json` that registers the `wdbx` MCP server for WDBX vector database access via Claude Code's tool system.

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

#### WDBX Tools

| Tool | Description |
|------|-------------|
| `db_query` | Cosine similarity search over stored vectors |
| `db_insert` | Insert vector with metadata |
| `db_get` | Retrieve vector by ID |
| `db_update` | Update existing vector |
| `db_delete` | Remove vector by ID |
| `db_list` | List stored vectors |
| `db_stats` | Vector count, dimensions, memory usage |
| `db_backup` | Save database to file |
| `db_diagnostics` | Performance metrics and health info |

### Hooks

| Event | Action |
|-------|--------|
| `SessionStart` | Loads platform context, checks Zig version and pinned version match |
| `PostToolUse` (Edit/Write) | Warns about stub.zig sync and module import violations |
| `PreToolUse` (Bash) | Warns against `zig fmt .` from root (use specific dirs) |
| `PreToolUse` (Edit) | Warns about `@import("abi")` inside `src/features/` |
| `Stop` | Advisory checklist: stub sync, formatting, CLI registry |

## Quick Reference

```bash
# Build with platform detection
/zig-abi:build full-check

# Verify all aspects
/zig-abi:check all

# Scaffold new feature
/zig-abi:new-feature scheduling

# WDBX database operations (via MCP)
db_stats                          # check connectivity and vector count
db_query --vector [...] --top-k 5 # semantic similarity search
db_insert --id doc1 --vector [...] --metadata '{"source": "code"}' # store vector
db_backup --path ./backup/wdbx    # snapshot database
db_diagnostics                    # performance metrics
```

## Feature Flags

The ABI Framework uses comptime feature gating with 27 `feat_*` flags and 56 validated combos. Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `feat_ai` | true | AI/LLM features |
| `feat_gpu` | true | GPU acceleration |
| `feat_database` | true | Vector database (WDBX) |
| `feat_network` | true | Distributed compute |

All flags default to `true`. Disable with `-Dfeat-<name>=false`.

## Platform Notes

On macOS 26+ (Darwin 25+), the stock Zig linker fails. The plugin auto-detects this and routes to:
1. `run_build.sh` wrapper
2. Fallback validation (`zig fmt --check`, `zig test -fno-emit-bin`)
3. Linux CI when a build step still needs binary emission
