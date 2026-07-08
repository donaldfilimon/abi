# Repository Guidelines

## Session Start
- Trust executables (`build.zig`, `tools/*.sh`, `src/`, `tests/`) over prose when they disagree.
- Read `tasks/lessons.md` and `tasks/todo.md` (loaded via `opencode.json` `instructions`).
- `git status --short --branch`; never revert unrelated dirty work.
- Run `./build.sh check` before and after changes.

## Commands That Matter
- Toolchain **pinned** by `.zigversion` to `0.17.0-dev.1252+e4b325c19`. `build.sh`/`tools/build.sh` invoke whatever `zig` is on PATH — they do **not** switch. Zig 0.16 fails on WDBX/MCP listeners.
- On macOS: `./build.sh` — Metal linking + documented workflow; plain `zig build` works but bypasses the wrapper.
- Primary gate: `./build.sh check` (build both exes, all tests, lint, parity, feature-off stubs, CLI contract smoke).
- Full gate: `./build.sh full-check` (adds integration, benchmarks, dashboard smoke).
- Single test: `zig build test -Dtest-filter="<pattern>"`. The post-`--` form is **silently ignored**.
- Focused steps: `zig build test-cli`, `test-plugins`, `test-contracts`, `test-mcp-server`, `test-integration`, `benchmarks`, `check-parity`, `lint`, `fix`.
- Opt-in: `zig build cross-smoke` (Linux/Windows/macOS compile; slow).
- Binaries: `./build.sh cli` → `zig-out/bin/abi`, `./build.sh mcp` → `zig-out/bin/abi-mcp`.
- Docs: `npx mint@latest validate` (`docs/docs.json`; not in CI).

## Architecture (what filenames don't show)
- Public API root: `src/root.zig` → `@import("abi")`. CLI entry: `src/main.zig` → `cli/dispatch.zig`.
- MCP server: `src/mcp/`. Repo-root `mcp/` + `.mcp.json` is **only** host launcher glue.
- Features use mod/stub pattern: `src/features/mod.zig` selects `mod.zig` or `stub.zig` per `-Dfeat-*`. All default **on**; disable with `-Dfeat-<name>=false`.
- **`feat-foundationmodels`**: comptime-gated to arm64 macOS. Requires Xcode + macOS 26 SDK for `xcrun swiftc`; use `-Dfeat-foundationmodels=false` to skip.
- **Generated**: `src/plugin_registry.zig` (auto-regenerated at build time from `src/plugins/*/abi-plugin.json` via `tools/generate_plugin_registry.zig`).
- Inside `src/`: relative `.zig` imports only. **Only** the MCP handler group (`src/mcp/main.zig`, `handlers.zig`, `ai_tools.zig`, `connector_tools.zig`, `plugin_tools.zig`, `state.zig`) may `@import("abi")`.

## CLI Surface (frozen, contract-tested)
13 top-level commands: `help`, `complete`, `train`, `agent`, `backends`, `plugin`, `auth`, `twilio`, `tui`, `dashboard`, `wdbx`, `scheduler`, `nn`. Full specs in `src/cli/usage.zig`.
- `help --json` emits typed command metadata; `help --completion <bash|zsh|fish>` emits shell scripts.
- `complete` supports `--live`, `--model`, `--confirm` (apple-fm), `--learn` (SEA loop).
- `tui`/`dashboard` flags documented by `--help`. `abi --tui` shortcuts to `abi tui`.
- Malformed numeric args → usage + exit 2.
- **Do not** resurrect legacy names: `version`, `doctor`, `features`, `platform`, `connectors`, `search`, `info`, `chat`, `db`, `serve`.

## MCP Surface (12 tools, asserted in `tests/contracts/surface.zig`)
`ai_run`, `ai_complete`, `ai_train`, `ai_learn`, `wdbx_query`, `scheduler_stats`, `scheduler_info`, `connector_test`, `gpu_status`, `plugin_list`, `wdbx_stats`, `plugin_run`.
- Primary: JSON-RPC 2.0 over stdio (64 KB request cap).
- Optional: HTTP/SSE on `127.0.0.1:8080` (`ABI_MCP_HTTP_PORT`, `ABI_MCP_HTTP_TOKEN`). WDBX REST uses same bearer scheme via `ABI_WDBX_REST_TOKEN`. Loopback-only hardening.

## API & Contract Rules
- Public feature API change → update both `mod.zig` + `stub.zig`; run `zig build check-parity`.
- Parity tool scans only column-0 `pub const`/`pub fn` names (not `pub var`, threadlocal, nested).
- 5 contract test suites: `surface.zig` (CLI+MCP command/tool freeze), `feature_modules.zig` (mod/stub feature-off behavior), `mcp_tools.zig` (per-tool validation specs), `plugin_registry.zig` (generated metadata), `public_docs.zig` (external claims).

## Zig 0.17 Conventions Agents Miss
- `pub fn main(init: std.process.Init) !void`
- `ArrayListUnmanaged(T).empty` (not `.init(allocator)`)
- `std.mem.trimEnd` (not `trimRight`); `splitScalar`/`splitAny`/`splitSequence`
- `foundation.time.unixMs()` for ms timestamps
- Tests: inline `test {}`; end modules with `std.testing.refAllDecls(@This())`
- Silent empty `catch {}` forbidden in persistence, inference, connector, data-access paths — propagate or log
- Conditional compilation: `build_options.feat_*` (not runtime checks)

## WDBX / GPU / Connectors
- WDBX: in-process store + segment/WAL persistence. Cluster RPC is real TCP RequestVote/AppendEntries (`ABI_WDBX_CLUSTER_TOKEN` for non-loopback; `ABI_WDBX_CLUSTER_PEERS` allowlist). **Not** production multi-host or sharding.
- GPU: capability report + CPU fallback. No `-Dgpu-backend` option.
- Live connectors require explicit credentials + `.live` transport. Local test paths must stay offline.

## AI Subsystem
- **SEA loop** (`feat-sea`): evidence-augmented self-learning completion with 8-signal scorer + budgeted greedy candidate selection. Persists `AdaptiveModulator` weights in WDBX.
- **Connector validation**: Discord validates numeric snowflake IDs; Twilio validates `AC`+32-hex SIDs, 32-hex tokens, explicit `.live` transport.

## Claims & Docs
- No unproven capability claims (distributed sharding, production FHE/AES/RBAC, non-loopback hardening, K8s/H100, Swift/Python/TF stacks, QPS/latency/accuracy/energy/certifications).
- Public wording: `docs/contracts/external-claims-audit.mdx`. Security surface: `abi-threat-model.md`.

## OpenCode Setup
- `opencode.json` auto-loaded (schema `https://opencode.ai/config.json`).
- `.opencode/skills/` is a symlink to `.agents/skills/` (canonical source).
- MCP servers: `abi-mcp` (`./mcp/launcher.sh stdio`), `skill-loop` (`@stylusnexus/skill-loop-cli@0.3.3`).
- OpenCode MCP entries use `type: "local"`, `enabled: true`, single `command` array — do not copy `.mcp.json`'s `command` + `args` shape.

## Keep in Sync
Root siblings `CLAUDE.md` and `GEMINI.md` restate the same conventions. When commands, contracts, feature flags, or Zig patterns change, update all three.
