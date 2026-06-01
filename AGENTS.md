# AGENTS.md

Trust executable config over prose. When docs conflict with `build.zig`, `tools/build.sh`, or source, trust the executable source.

Read first: `CLAUDE.md`, `GEMINI.md`, `tasks/lessons.md` (session-start checklist), `tasks/todo.md` (active work + known failures).

## First Commands

```bash
./build.sh check            # primary gate: build CLI+MCP, module/connector/contract/stub tests, fmt, parity
./build.sh full-check       # check + integration tests + benchmarks + TUI smoke
./build.sh cli              # -> zig-out/bin/abi
./build.sh mcp              # -> zig-out/bin/abi-mcp
zig build test              # module + connector tests
zig build test-integration  # integration suite (run when touching AI/WDBX/scheduler/MCP/connectors)
zig build benchmarks        # benchmark suite (run when touching GPU/HNSW/router/chain)
zig build lint | zig build fix        # zig fmt --check | auto-format
zig build check-parity      # mod.zig / stub.zig public-decl parity
zig build test -- --test-filter "<pattern>"  # single test
```

`./build.sh` is a thin wrapper over `tools/build.sh` -> `zig build`. On macOS, prefer it over raw `zig build` for the documented Darwin workflow.

## Feature Flags & Stubs

Defaults: `feat-ai`, `feat-gpu`, `feat-tui`, `feat-accelerator`, `feat-shader`, `feat-mlir`, `feat-wdbx`, `feat-os-control`, `feat-hash` enabled; `feat-mobile`, `feat-metrics` disabled. There is **no** `-Dgpu-backend` option; GPU is runtime-selected.

`./build.sh check` smoke-builds every feature-off stub and the real `feat-mobile` module via `tools/check_feature_stubs.sh` (focused feature contracts + public-contracts under every `-Dfeat-*`).

Every feature in `src/features/` has a real `mod.zig` and a disabled `stub.zig`. **Update both** when changing public APIs; disabled paths must return `error.FeatureDisabled`, never fabricate success. Gate with `build_options.feat_*` at compile time, not runtime.

## Architecture

| Path | Role |
|------|------|
| `src/root.zig` | Public `abi` module root (entry for consumers) |
| `src/main.zig` -> `src/abi_cli/dispatch.zig` | CLI entry; also handles `abi --tui` shortcut outside `usage.commands` |
| `src/abi_cli/{usage,handlers}.zig` | Frozen CLI command list and handlers |
| `src/mcp/` | JSON-RPC 2.0 server (stdio + loopback HTTP/SSE) |
| `src/features/{ai,wdbx,gpu,tui,shaders,mlir,accelerator,os_control,hash,metrics,mobile}/` | Feature modules (mod + stub each) |
| `src/connectors/` | OpenAI, Anthropic, Discord, Grok, Twilio, HTTP, JSON |
| `src/core/` + `src/foundation/` | Scheduler, memory, config, registry, time, sync, logger, IO, credentials, OS |
| `src/plugins/` + `src/plugin_registry.zig` | Plugin manifests (generated into registry by the build) |
| `mcp/` (repo root) | **Host** launcher scripts (`launcher.sh` -> `zig-out/bin/abi-mcp`); not Zig code |
| `tools/` | `build.sh`, `check_feature_stubs.sh`, `check_parity.zig`, `generate_plugin_registry.zig`, `run_contract_cli.sh` |
| `tests/contracts/` | Feature, surface, MCP-tool, plugin-registry, and public-docs contract tests |

## Import Rules

- Inside `src/`: relative `.zig` imports only.
- The **only** files in `src/` that may use `@import("abi")` are `src/mcp/main.zig` and `src/mcp/handlers.zig`.
- Outside `src/`: `@import("abi")` and `@import("build_options")`.
- Every path import must end in `.zig`.

## CLI & MCP Contracts (frozen, contract-tested)

CLI top-level: `help`, `complete`, `train`, `agent`, `backends`, `plugin`, `auth`, `twilio`, `tui`, `dashboard` (+ `abi --tui`). `agent` subcommands: `plan | train <profile|all> | tui | os <dry-run|execute --confirm>`. Do **not** dispatch legacy names: `version`, `doctor`, `features`, `platform`, `connectors`, `search`, `info`, `chat`, `db`, `serve`.

MCP tools: `ai_run`, `ai_complete`, `ai_train`, `wdbx_query`, `scheduler_stats`, `scheduler_info`, `connector_test`, `gpu_status`, `plugin_list`, `wdbx_stats`, `plugin_run`. JSON-RPC 2.0 over stdio, request cap 64 KB, optional HTTP on `127.0.0.1:8080` (`ABI_MCP_HTTP_PORT` to override; empty/invalid/zero/out-of-range -> 8080; bind failure leaves stdio running). HTTP endpoints: `GET /sse`, `POST /message`. The repo-root `.mcp.json` already wires `abi-mcp` for host MCP clients via `mcp/launcher.sh`.

## Generated Code

`src/plugin_registry.zig` is regenerated automatically: the build depends on a `gen_plugin_registry` step (see `build.zig:43-45`) that runs `tools/generate_plugin_registry.zig` over `src/plugins/*/abi-plugin.json` before linking the CLI and the plugin-registry contract test. **Do not hand-edit** the generated file. Plugin manifests must include `name`, `version`, `description`, `target_feature`, and a safe relative `.zig` `entry_point` whose file exists under the plugin directory; `targetFeature` / `entryPoint` aliases are accepted.

## Zig 0.17 Patterns

- Entry: `pub fn main(init: std.process.Init) !void`.
- `ArrayListUnmanaged(T).empty` (not `.init(allocator)`); `std.mem.Allocator` is explicit, no global.
- `std.mem.trimEnd` (not `trimRight`); `splitScalar` / `splitAny` / `splitSequence` (not `split`).
- Timestamps: `foundation.time.unixMs()` (not `std.time.milliTimestamp`).
- Tests are inline `test {}`; each module ends with `std.testing.refAllDecls(@This())`.
- No silent empty `catch {}` in data, inference, or persistence paths.
- `@panic` only for unrecoverable invariants; `unreachable` only for provably impossible branches.
- Naming: `camelCase` fns/vars, `PascalCase` types, `SCREAMING_SNAKE_CASE` constants, `snake_case` enum variants.

## Connector Validation

Local paths are deterministic; live paths require explicit `.live` transport calls.

- **Discord**: printable non-whitespace tokens, numeric snowflake-like client/channel/author IDs, 2000-byte message limit.
- **Twilio**: account SID `AC` + 32 hex, auth token 32 hex, non-empty base URL, non-zero timeout, explicit `.live` transport, TwiML/XML + URL-form escaping, ConversationRelay aliases.
- **OpenAI / Anthropic / Grok**: local streaming helpers are deterministic and never hit the network; live methods need credentials. `grok_setup.sh` reads `ABI_GROK_API_KEY` from the env and refuses to run without it.

## External Claims

Do not assert distributed sharding, AES/RBAC, Swift, Python/TensorFlow stacks, Kubernetes/H100 deployments, regulatory certifications, QPS/latency/accuracy, energy efficiency, or model-benchmark numbers in docs or collateral unless a repo test, benchmark artifact, or documented source file proves them. See `docs/contracts/external-claims-audit.md`.

## Onboarding

- `walkthrough.md` - full CLI/MCP/connector walkthrough with examples.
- `tasks/lessons.md` - session-start checklist, conventions, common pitfalls.
- `tasks/todo.md` - current work items and known failures.
- `docs/index.md` - architecture and development guide.
