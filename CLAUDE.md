# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Trust executable config over prose: when this file conflicts with `build.zig`, `tools/build.sh`, or source, trust the executable source. `AGENTS.md` is a richer companion to this file; `tasks/lessons.md` is the session-start checklist and `tasks/todo.md` tracks active work and known failures.

This file has two sibling instruction files at the repo root — `AGENTS.md` (for Codex) and `GEMINI.md` (for Gemini) — that restate the same repository conventions. When you change a durable convention here (commands, contracts, feature flags, Zig patterns), propagate it to both so the three stay consistent.

Toolchain is pinned to Zig `0.17.0-dev.978+a078d55a2` (see `.zigversion`). On macOS/Darwin, prefer `./build.sh ...` (a thin wrapper over `tools/build.sh` → `zig build`) over raw `zig build` for the documented workflow. Builds are incremental against `.zig-cache/`; the first cold build/check is slow, subsequent runs are fast.

## Common Development Commands

### Build & Validation
- `./build.sh check` – Primary validation gate for build integrity and API parity checks.
- `./build.sh full-check` – Runs `check`, integration tests, benchmarks, and TUI smoke.
- `./build.sh cli` – Builds the main executable (`zig-out/bin/abi`).
- `./build.sh mcp` – Builds the MCP server binary (`zig-out/bin/abi-mcp`).
- `zig build run` – Builds and runs the app.
- `zig build lint` – Runs `zig fmt --check` on all source files for formatting compliance.
- `zig build fix` – Automatically formats source files based on project standards.
- `zig build check-parity` – Verifies top-level public declaration-name parity for feature/plugin `mod.zig` and `stub.zig` pairs.
- Run a single test: `zig build test -Dtest-filter="<pattern>"` (the `test-filter` build option feeds `.filters` on every `addTest`; on macOS use `./build.sh test -Dtest-filter="<pattern>"`). Note: the post-`--` form `zig build test -- --test-filter …` is **not** wired up and is silently ignored.

### Running Tests
- `zig build test-integration` – Executes the integration test suite.
- `zig build benchmarks` – Runs the benchmark suite.
- `zig build test` – Module + connector tests.
- `zig build test-feature-contracts` – Feature module contracts.
- `zig build test-contracts` – Surface/MCP/plugin/docs contracts.
- `zig build test-mcp-contracts` – MCP tool contract tests.
- `zig build test-mcp-server` – MCP server transport tests (stdio + HTTP/SSE).

## Architecture Overview

The ABI framework is a modular Zig codebase with a clear separation of concerns across the following layers:

| Layer | Path | Responsibility |
|-------|------|----------------|
| **Public API** | `src/root.zig` | Exposes the `abi` module to consumers. This is the primary entry point. |
| **CLI** | `src/main.zig`, `src/cli/` | Parses command-line arguments and delegates to sub-commands. |
| **MCP Server** | `src/mcp/main.zig` | Implements a JSON-RPC 2.0 server over stdio and optional HTTP/SSE transport. |
| **Feature Selection** | `src/features/mod.zig` | Enables/disables features via Zig build options (`-Dfeat-*`). Uses the *mod/stub* pattern. |
| **AI Sub-system** | `src/features/ai/` | Implements AI profiles (Abbey, Aviva, Abi), routing, constitution, and the model catalog (`models.zig`). |
| **Vector Store (WDBX)** | `src/features/wdbx/` | Provides in-memory key-value and vector storage with HNSW index and MVCC-style snapshot chain. |
| **GPU Backend** | `src/features/gpu/` | Reports GPU status, attempts Metal initialization on macOS, falls back to vectorized CPU implementation. The backend is runtime-selected; there is **no** `-Dgpu-backend` option. |
| **Connectors** | `src/connectors/` | Provides local/live adapters for external services (OpenAI, Anthropic, Grok, Discord, Twilio, HTTP, JSON). |
| **Plugin System** | `src/plugins/`, `src/plugin_registry.zig` | Validates plugin manifests and generates metadata registry. |
| **Core Utilities** | `src/core/` + `src/foundation/` | Scheduler, memory, config, registry, time, sync, logger, IO, credentials, OS abstractions. |

Note: `src/mcp/` is the Zig MCP server; the **repo-root `mcp/`** directory holds host launcher scripts (`mcp/launcher.sh` → `zig-out/bin/abi-mcp`), not Zig code. The repo-root `.mcp.json` wires `abi-mcp` for host MCP clients via that launcher. Within `src/mcp/`, `middleware.zig` runs declarative argument validation (NUL/length/path-traversal/enum checks) on every `tools/call` before dispatch, and `handlers.errorMessage` normalizes any `anyerror` to a stable, non-leaking client string on both transports.

### Key Areas to Focus On

- **Mod/Stub Pattern**: Ensure public API stability by checking mod/stub parity frequently. Every feature has real `mod.zig` and disabled `stub.zig`; update both when changing public APIs.
- **Build Flow**: `./build.sh check` includes contract tests plus focused feature-off and feature-aware public contracts for every `-Dfeat-*` stub.
- **Feature Flags**: 
  - Enabled by default: `feat-ai`, `feat-wdbx`, `feat-gpu`, `feat-accelerator`, `feat-shader`, `feat-mlir`, `feat-os-control`, `feat-tui`, `feat-hash`, `feat-telemetry`
  - Disabled by default: `feat-mobile`, `feat-metrics`, `feat-sea` (Sparse Evidence Attention self-learning loop), `feat-foundationmodels` (Apple on-device FoundationModels connector, macOS-only)
  - Each flag selects between a feature's `mod.zig` (enabled) and `stub.zig` (disabled) in `src/features/mod.zig`; keep both in declaration-name parity (`zig build check-parity`). (`feat-sea` lives in `src/features/sea/`; the FoundationModels connector is in `src/connectors/fm.zig` and links `FoundationModels.framework` only under `-Dfeat-foundationmodels` on macOS.)
- **Import Rules**: Within `src/`, use relative `.zig` imports. `@import("abi")` is only allowed from the MCP executable + handler module graph — `src/mcp/main.zig` plus the `handlers.zig` group (`handlers.zig`, `ai_tools.zig`, `connector_tools.zig`, `plugin_tools.zig`, `state.zig`), which `build.zig` wires the `abi` package into — never from modules re-exported by `src/root.zig`. Always include `.zig` extension on path imports.
- **CLI Contracts** (frozen, contract-tested in `tests/contracts/`): top-level commands are `help`, `complete`, `train`, `agent`, `backends`, `plugin`, `auth`, `twilio`, `tui`, `dashboard`, `wdbx`, `scheduler`, `nn`, plus the `abi --tui` shortcut handled outside `src/cli/usage.zig`. `nn` subcommands: `train "<text>" | train --jsonl <path> [--field <name>] | sample --text "<corpus>" --seed <char> --n <k>` — a **miniature character-level demo trainer** (`feat-nn`; real manual-backprop char-LM, **not** a production/LLM/distributed trainer). `agent` subcommands: `plan | train <profile|all> | tui | os <dry-run|execute --confirm>` (`agent tui` is now an interactive REPL — line-at-a-time with raw-mode fallback, `/help /model /history /reset /quit`). `complete` takes additive flags `[--live] [--model <id>] [--confirm] [--learn]` (`--confirm` is required for on-device `apple-fm`; `--learn` routes through the SEA loop). `wdbx` subcommands: `db <init|verify> | block <insert|get> | query | benchmark | cluster <status|demo|serve> | compute info | secure demo | gpu info | api serve` (see `src/cli/handlers/wdbx.zig`). Do **not** dispatch legacy names: `version`, `doctor`, `features`, `platform`, `connectors`, `search`, `info`, `chat`, `db`, `serve`.
- **MCP Contracts**: Tools (12): `ai_run`, `ai_complete`, `ai_train`, `ai_learn`, `wdbx_query`, `scheduler_stats`, `scheduler_info`, `connector_test`, `gpu_status`, `plugin_list`, `wdbx_stats`, `plugin_run`. (`ai_learn` dispatches the SEA self-learning loop — `input` required, optional `model`/`evidence_limit`; degrades to a stored completion when `feat-sea` is off. `tests/contracts/surface.zig` asserts the tool count.) JSON-RPC 2.0 over stdio with a 64 KB request cap; optional HTTP on `127.0.0.1:8080` (`ABI_MCP_HTTP_PORT` to override; empty/invalid/zero/out-of-range falls back to 8080; bind failure leaves stdio running). Set `ABI_MCP_HTTP_TOKEN` to require `Authorization: Bearer <token>` on the loopback HTTP/SSE transport (stdio JSON-RPC stays tokenless local IPC). HTTP endpoints: `GET /sse`, `POST /message`. The WDBX REST listener (`abi wdbx api serve`) honors `ABI_WDBX_REST_TOKEN` for the same bearer scheme; both are loopback-only hardening, not a substitute for a TLS fronting layer.
- **Model Catalog** (`src/features/ai/models.zig`, `std`-only so `mod.zig`/`stub.zig` keep parity): single source of truth for recognized model ids, short aliases, and provider routing. Default model is `claude-fable-5` (`fable5`); `abi-local` stays selectable; `apple-fm` (`fm-local`/`fm`, provider `fm`) routes to the macOS on-device FoundationModels connector (framework-linked via a Swift `@c` shim, SE-0495, in `src/connectors/fm_shim.swift`, compiled+linked under `-Dfeat-foundationmodels`; on-device generation requires Apple-Intelligence hardware at runtime — see `src/connectors/fm.zig`). `complete [--live] [--model <id>]` (CLI) and the `ai_complete` `model` arg (MCP) both run `models.canonical(...)` so aliases (e.g. `fable-5` → `claude-fable-5`) are recorded canonically and unknown ids pass through unchanged (the CLI prints a one-line stderr warning on an unrecognized id). `complete --live` serves anthropic-provider models over the explicit live transport using stored credentials. Remote providers are only reachable across the explicit live-transport boundary in `connectors/`.
- **Generated Code**: Do not manually edit `src/plugin_registry.zig`; the build regenerates it from `src/plugins/*/abi-plugin.json` via `tools/generate_plugin_registry.zig`. Plugin manifests must include `name`, `version`, `description`, `target_feature`, and a safe relative `.zig` `entry_point` that exists under the plugin directory (`targetFeature` / `entryPoint` aliases accepted).
- **Zig 0.17 Patterns**: 
  - Entry: `pub fn main(init: std.process.Init) !void`
  - Use `ArrayListUnmanaged(T).empty` (not `.init(allocator)`)
  - Use `std.mem.trimEnd` (not `trimRight`)
  - Use `std.mem.splitScalar`, `splitAny`, or `splitSequence`
  - Use `foundation.time.unixMs()` for timestamps (not `std.time.milliTimestamp`)
  - Avoid silent empty `catch {}` in data, inference, or persistence paths; `@panic` only for unrecoverable invariants, `unreachable` only for provably impossible branches
  - Tests are inline `test {}`; each module ends with `std.testing.refAllDecls(@This())`
  - Naming: `camelCase` fns/vars, `PascalCase` types, `SCREAMING_SNAKE_CASE` constants, `snake_case` enum variants
- **External Claims**: Do not assert unproven capabilities (distributed sharding, AES/RBAC, Swift/Python stacks, Kubernetes/H100, certifications, QPS/latency/accuracy numbers) in docs unless a repo test, benchmark artifact, or source file proves them. See `docs/contracts/external-claims-audit.md`.
- **Connector Validation**: 
  - Discord: validates printable non-whitespace credentials, numeric snowflake-like IDs, author IDs, message size.
  - Twilio: validates `AC` + 32-hex account SIDs, 32-hex auth tokens, base URL, timeout, explicit `.live` transport, XML/form escaping, ConversationRelay aliases.

### Verification

Before finishing code changes, run:
```bash
./build.sh check
```

For full validation including integration tests, benchmarks, and TUI smoke:
```bash
./build.sh full-check
```

## Important Files

- `tasks/lessons.md` – Startup checklist and conventions
- `tasks/todo.md` – Current work items and known failures
- `tasks/` – Working-notes directory beyond the two above (e.g. `roadmap-next.md`, `scheduler-memory-wireup.md`); consult for in-flight context
- `TASKS.md` – Top-level active task board (what's currently on deck); detailed history lives under `tasks/`
- `docs/index.md` – Architecture, public API contracts, onboarding, and development guides
- `CHANGELOG.md` – Release-note style modernization highlights
- `walkthrough.md` – Guided tour of the Zig 0.17 modernization and current-branch expansion surfaces
- `abi-threat-model.md` – AppSec-grade, repo-path-anchored threat model (MCP/WDBX listeners, credentials at rest); pairs with the External Claims guidance and `docs/contracts/external-claims-audit.md`
- `memory/` – Durable project context checked into the repo (`glossary.md`, `context/project.md`), distinct from the scratch caches
