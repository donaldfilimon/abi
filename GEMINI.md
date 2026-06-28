# GEMINI.md - ABI Framework

Quick reference for Google Gemini and compatible agents working on this Zig 0.17 ABI framework. Prefer executable config and source over prose when they disagree.

## First Commands

```bash
./build.sh check              # primary gate on macOS/Darwin
./build.sh full-check         # check + integration tests + benchmarks + TUI smoke
./build.sh cli                # build zig-out/bin/abi
./build.sh mcp                # build zig-out/bin/abi-mcp
zig build test-integration    # explicit integration suite
zig build benchmarks          # explicit benchmark suite
```

Zig is pinned by `.zigversion` to `0.17.0-dev.978+a078d55a2`; `build.zig.zon` keeps `0.17.0-dev.978+a078d55a2` as the package minimum. Plain `zig build` may work with a compatible local toolchain, but use `./build.sh ...` on macOS for the documented Darwin workflow.

## Current CLI Examples

```bash
./zig-out/bin/abi backends
./zig-out/bin/abi complete "Summarize the ABI runtime"
./zig-out/bin/abi train "Summarize the ABI runtime"
./zig-out/bin/abi agent plan "Plan a safe refactor"
./zig-out/bin/abi agent train all
./zig-out/bin/abi plugin list
./zig-out/bin/abi dashboard
./zig-out/bin/abi twilio simulate "I need support"
```

Supported top-level commands are `help`, `complete`, `train`, `agent`, `backends`, `plugin`, `auth`, `twilio`, `tui`, `dashboard`, `wdbx`, and `scheduler`. The top-level `abi --tui` shortcut also renders the dashboard.

Subcommand grammar mirrors `src/cli/`: `complete [--live] [--model <id>] [--confirm] [--learn] <input>` (the model id alias-resolves through the catalog, `--live` serves anthropic models over the live transport, an unrecognized id prints a stderr warning before passing through, `--confirm` is required for on-device `apple-fm`, and `--learn` routes through the SEA self-learning loop; `agent tui` is now an interactive REPL); `agent <plan | train <profile|all> | tui | os <dry-run|execute --confirm>>`; and `wdbx <db <init|verify> | block <insert|get> | query | benchmark | cluster <status|demo|serve> | compute info | secure demo | gpu info | api serve>`. Malformed numeric arguments return usage with exit code 2 rather than silently using a default.

Do not assume old command names exist: `version`, `doctor`, `features`, `platform`, `connectors`, `search`, `info`, `chat`, `db`, and `serve` are not currently dispatched.

## Project Map

| Path | Purpose |
| --- | --- |
| `src/root.zig` | Public `abi` module root |
| `src/main.zig` | CLI entry |
| `src/cli/` | CLI usage, dispatch, handlers |
| `src/mcp/main.zig` | MCP entry point (spawns HTTP, runs stdio loop) |
| `src/features/mod.zig` | Feature flag mod/stub selection |
| `src/features/ai/` | AI profiles, router, constitution, training, local streaming helpers |
| `src/features/wdbx/` | In-memory vector store, HNSW, block chain |
| `src/features/gpu/` | GPU status, Metal attempt on macOS, CPU fallback |
| `src/features/tui/` | Diagnostics dashboard renderer |
| `src/connectors/` | OpenAI, Anthropic, Discord, Grok, Twilio connector surfaces |
| `src/foundation/` | Time, sync, logger, utils, errors, OS, IO, credentials |
| `src/core/registry.zig` | Generated plugin registry loading and metadata accessors |
| `src/plugins/plugin_manager.zig` | Required manifest validation and local plugin manager API |
| `tools/` | Build helpers, plugin registry generation, parity checker |

## Feature Flags

Default enabled: `feat-ai`, `feat-gpu`, `feat-tui`, `feat-accelerator`, `feat-shader`, `feat-mlir`, `feat-wdbx`, `feat-os-control`, `feat-hash`, `feat-telemetry`.

Default disabled: `feat-mobile`, `feat-metrics`, `feat-sea` (`src/features/sea/`, Sparse Evidence Attention self-learning loop), `feat-foundationmodels` (`src/connectors/fm.zig`, Apple on-device FoundationModels — macOS-only; links `FoundationModels.framework` + a `swiftc`-built `libabi_fm_shim.dylib`; on-device generation is wired through a Swift `@c` shim (SE-0495) and requires Apple-Intelligence hardware at runtime).

Use `-Dfeat-<name>=false|true`, for example:

```bash
./build.sh check -Dfeat-tui=false
./build.sh check -Dfeat-gpu=false
```

There is no `-Dgpu-backend` build option. GPU status is runtime behavior.

## MCP Facts

- Build with `./build.sh mcp`.
- Binary: `zig-out/bin/abi-mcp`.
- Primary transport: JSON-RPC 2.0 over stdio.
- Secondary transport: loopback HTTP/SSE on `127.0.0.1:8080` when available; set `ABI_MCP_HTTP_PORT=<port>` to avoid local port conflicts. Empty, invalid, zero, or out-of-range values fall back to `8080`; bind failures leave stdio running. Set `ABI_MCP_HTTP_TOKEN` to require `Authorization: Bearer <token>` on the HTTP/SSE transport (stdio JSON-RPC stays tokenless local IPC); the WDBX REST listener (`abi wdbx api serve`) honors `ABI_WDBX_REST_TOKEN` for the same bearer scheme. Both are loopback-only hardening, not a TLS substitute.
- HTTP endpoints: `GET /sse`, `POST /message`.
- Request size limit: 64KB.
- Methods: `initialize`, `tools/list`, `tools/call`, `resources/list`, `prompts/list`, `ping`, `shutdown`.
- Tools (12, count asserted in `tests/contracts/surface.zig`): `ai_run`, `ai_complete`, `ai_train`, `ai_learn`, `wdbx_query`, `scheduler_stats`, `scheduler_info`, `connector_test`, `gpu_status`, `plugin_list`, `wdbx_stats`, `plugin_run`. (`ai_learn` runs the SEA loop; degrades to a stored completion when `feat-sea` is off.)

## Development Rules

- Read `tasks/lessons.md` and `tasks/todo.md` before substantial work.
- Public feature API changes require matching `mod.zig` and `stub.zig` updates.
- Run `zig build check-parity` after public API changes.
- Do not edit generated `src/plugin_registry.zig`; change plugin source/manifests or `tools/generate_plugin_registry.zig`.
- Plugin manifests require `name`, `version`, `description`, `target_feature`, and a safe relative `.zig` `entry_point` whose file exists under the plugin directory; `targetFeature` / `entryPoint` aliases are accepted. Generated multi-plugin registry metadata is covered by `tests/contracts/plugin_registry.zig`.
- Discord connector IDs are validated as numeric snowflake-like IDs, and local/live paths enforce printable non-whitespace credentials, author ID validation, and message-size checks.
- Twilio validates account SIDs as `AC` + 32 hex characters, auth tokens as 32 hex characters, non-empty base URL, non-zero timeout, explicit `.live` transport selection, XML/form escaping, and ConversationRelay aliases before local/live dispatch.
- Only the MCP executable + handler module graph (`src/mcp/main.zig` plus the `handlers.zig` group: `handlers.zig`, `ai_tools.zig`, `connector_tools.zig`, `plugin_tools.zig`, `state.zig`) may import `@import("abi")` from inside `src/` — never modules re-exported by `src/root.zig`; other `src` imports should be relative `.zig` imports.
- Do not use plain `rm`; use safe alternatives.

## Zig 0.17 Notes

- Entry point: `pub fn main(init: std.process.Init) !void`.
- Use `ArrayListUnmanaged(T).empty`.
- Use `std.mem.trimEnd`.
- Use `std.mem.splitScalar`, `splitAny`, or `splitSequence`.
- Use `foundation.time.unixMs()` for timestamps.
- Avoid silent empty `catch` blocks in persistence, inference, and data-access code.

## Verification

Before finishing code changes, run:

```bash
./build.sh check
```

`./build.sh check` includes feature-off compile smoke tests, focused `test-feature-contracts` behavior coverage, feature-aware `test-contracts` coverage for every `-Dfeat-*` flag, and `-Dfeat-mobile=true` coverage through `tools/check_feature_stubs.sh`.

Use `./build.sh full-check` when you need the full local gate, including integration tests, benchmarks, and TUI smoke.

Run these explicitly when touched:

```bash
zig build test-integration
zig build benchmarks
zig build test-mcp-server     # MCP server transport tests (stdio + HTTP/SSE)
```

`tasks/todo.md` is the current source of truth for known failures. At the current modernization point, no known test failures are reproduced locally.
