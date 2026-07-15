# GEMINI.md — ABI Framework

Quick reference for Google Gemini and compatible agents. Trust executable config and source over prose when they disagree. `AGENTS.md` is the canonical instruction file; `tasks/lessons.md` is the session checklist and `tasks/todo.md` tracks active work.

Three sibling instruction files share repo conventions — `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`. When commands, contracts, feature flags, or Zig patterns change, update all three.

## Toolchain

Pinned by `.zigversion` to `0.17.0-dev.1275+59a628c6d`. `build.sh`/`tools/build.sh` invoke whatever `zig` is on PATH — they do **not** switch. Zig 0.16 fails on WDBX/MCP listeners. On macOS: `./build.sh ...` for the documented Metal-linking workflow.

## Commands

| Command | What it does |
|---------|-------------|
| `./build.sh check` | Primary gate: build, tests, lint, parity, feature-off stubs, CLI smoke |
| `./build.sh full-check` | check + integration + benchmarks + dashboard smoke |
| `./build.sh cli` | Build `zig-out/bin/abi` |
| `./build.sh mcp` | Build `zig-out/bin/abi-mcp` |
| `zig build test -Dtest-filter="<pattern>"` | Single test (post-`--` form silently ignored) |
| `zig build test-cli` / `test-plugins` / `test-contracts` / `test-mcp-server` / `test-integration` | Focused test suites |
| `zig build benchmarks` | Benchmark suite |
| `zig build lint` / `fix` | Check/apply formatting |
| `zig build check-parity` | Verify mod/stub public declaration-name parity |
| `zig build cross-smoke` | Opt-in cross-compile (Linux/Windows/macOS; slow) |
| `npx mint@latest validate` | Docs site validation (not in CI) |

## Architecture

Layered modular codebase. The executable config (`build.zig`, `tools/build.sh`) owns linking and feature selection; trust it over prose.

| Layer | Path | Role |
|-------|------|------|
| Public API | `src/root.zig` | Exposes the `abi` module to consumers (`@import("abi")`). |
| CLI | `src/main.zig`, `src/cli/` | Arg parsing, sub-command dispatch, and handlers. Entry: `pub fn main(init: std.process.Init) !void`. |
| MCP server | `src/mcp/main.zig` + `handlers.zig` group (`handlers.zig`, `ai_tools.zig`, `connector_tools.zig`, `plugin_tools.zig`, `state.zig`) | JSON-RPC 2.0 over stdio + optional loopback HTTP/SSE. `src/mcp/middleware.zig` runs declarative argument validation on every `tools/call` before dispatch. |
| Feature selection | `src/features/mod.zig` | Each `-Dfeat-*` flag selects between a real `mod.zig` and a disabled `stub.zig`. |
| AI | `src/features/ai/` | Profiles, router, constitution, training, and model catalog (`models.zig`). |
| Vector store | `src/features/wdbx/` | In-memory KV + vector storage, HNSW index, MVCC snapshots, WAL/segment checkpoints. |
| GPU | `src/features/gpu/` | Runtime capability report; Metal attempt on macOS; deterministic CPU fallback. |
| Connectors | `src/connectors/` | Local/live adapters: openai, anthropic, grok, discord, twilio, fm, http, json. |
| Plugins | `src/plugins/`, `src/plugin_registry.zig` | Manifest validation + generated metadata registry. |
| Core/Foundation | `src/core/`, `src/foundation/` | Scheduler, registry, memory, time, sync, logger, IO, credentials, OS abstractions. |

Repo-root `mcp/` holds launcher scripts and `.mcp.json` host wiring — it is **not** the Zig MCP implementation.

- **Generated**: `src/plugin_registry.zig` — never hand-edit. Regenerated from `src/plugins/*/abi-plugin.json` at build time.
- **`feat-foundationmodels`**: comptime-gated to arm64 macOS. Requires Xcode + macOS 26 SDK for `xcrun swiftc`. Use `-Dfeat-foundationmodels=false` to skip.

## Import Rules

Inside `src/`: relative `.zig` imports only. **Only** the MCP handler group (`src/mcp/main.zig`, `handlers.zig`, `ai_tools.zig`, `connector_tools.zig`, `plugin_tools.zig`, `state.zig`) may `@import("abi")`.

## CLI Surface (frozen, contract-tested)

13 commands: `help`, `complete`, `train`, `agent`, `backends`, `plugin`, `auth`, `twilio`, `tui`, `dashboard`, `wdbx`, `scheduler`, `nn`. Full specs in `src/cli/usage.zig`.
- `help --json` / `--completion <bash|zsh|fish>`
- `complete` supports `--live`, `--model`, `--confirm` (apple-fm), `--learn` (SEA)
- `agent` supports `plan`, `train`, `tui`, `multi`, `spawn`, `browser`, `os`; `browser` is reviewed local planning only and never embeds or launches a browser
- `tui`/`dashboard` flags: `--help` documents them. `abi --tui` is a shortcut.
- Malformed numeric args → usage + exit 2
- **Do not** resurrect legacy names: `version`, `doctor`, `features`, `platform`, `connectors`, `search`, `info`, `chat`, `db`, `serve`

## MCP Surface (12 tools)

`ai_run`, `ai_complete`, `ai_train`, `ai_learn`, `wdbx_query`, `scheduler_stats`, `scheduler_info`, `connector_test`, `gpu_status`, `plugin_list`, `wdbx_stats`, `plugin_run`. JSON-RPC 2.0 over stdio (64 KB cap + `MAX_JSON_DEPTH` nesting bound). Optional HTTP/SSE on `127.0.0.1:8080` (`ABI_MCP_HTTP_PORT`, `ABI_MCP_HTTP_TOKEN`). Loopback-only hardening. `ai_train` paths confined under cwd or `ABI_TRAIN_DATA_ROOT`.

## API & Contract Rules

- Public API change → update both `mod.zig` + `stub.zig`; run `zig build check-parity`.
- Parity tool: scans column-0 `pub const`/`pub fn` (not `pub var`, threadlocal, nested).
- 5 contract test suites: `surface.zig`, `feature_modules.zig`, `mcp_tools.zig`, `plugin_registry.zig`, `public_docs.zig`.

## Zig 0.17 Patterns

- `pub fn main(init: std.process.Init) !void`
- `ArrayListUnmanaged(T).empty` (not `.init(allocator)`)
- `std.mem.trimEnd` (not `trimRight`); `splitScalar`/`splitAny`/`splitSequence`
- `foundation.time.unixMs()` for ms timestamps
- Tests: inline `test {}`; end modules with `std.testing.refAllDecls(@This())`
- No silent `catch {}` in persistence/inference/connector/data paths — propagate or log
- Conditional compilation: `build_options.feat_*`
- Pass an explicit `std.mem.Allocator`; no global/hidden allocator
- Naming: `camelCase` functions/vars, `PascalCase` types, `SCREAMING_SNAKE_CASE` constants, `snake_case` enum variants

## WDBX / GPU / Connectors

- WDBX: in-process store + segment/WAL persistence; ambient durable parent dirs `0700` on POSIX. Cluster RPC is real TCP RequestVote/AppendEntries (`ABI_WDBX_CLUSTER_TOKEN` for non-loopback; `ABI_WDBX_CLUSTER_PEERS` allowlist). **Not** production multi-host or sharding.
- GPU: capability report + CPU fallback. No `-Dgpu-backend` option.
- Live connectors require explicit credentials + `.live` transport + `https://` base URL. POSIX `auth signin` no-echo; credentials still plaintext JSON.

## AI Subsystem

- **SEA loop**: evidence-augmented self-learning completion with 8-signal scorer + budgeted greedy selection. Persists `AdaptiveModulator` weights in WDBX.
- **Connector validation**: Discord validates numeric snowflake IDs; Twilio validates `AC`+32-hex SIDs, 32-hex tokens, explicit `.live` transport.

## Claims & Docs

No unproven claims (distributed sharding, production FHE/AES/RBAC, non-loopback hardening, K8s/H100, Swift/Python/TF stacks, QPS/latency/accuracy/energy/certifications). Public wording: `docs/contracts/external-claims-audit.mdx`. Security: `abi-threat-model.md`.

## OpenCode Setup

`opencode.json` auto-loaded. `.opencode/skills/` is a symlink to `.agents/skills/`. MCP servers: `abi-mcp`, `skill-loop`. Sync canonical skills: `.agents/skills/sync-clis/launch.sh`.
