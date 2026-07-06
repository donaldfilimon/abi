# Repository Guidelines

## Session Start (do this first)
- Trust executable sources (`build.zig`, `tools/*.sh`, `src/`, `tests/`) over prose when they disagree.
- Read `tasks/lessons.md` and `tasks/todo.md`.
- `git status --short --branch`; never revert unrelated dirty work.
- Run `./build.sh check` before and after changes.

## Commands That Matter
- Toolchain **pinned** by `.zigversion` to `0.17.0-dev.978+a078d55a2`. `build.zig.zon` only declares minimum. `build.sh`/`tools/build.sh` invoke whatever `zig` is on PATH (they do **not** switch). Zig 0.16 fails on WDBX/MCP listeners (use 0.17 `std.Io.net.Stream.read`).
- On macOS/Darwin: prefer `./build.sh ...` (keeps documented Metal + workflow).
- Primary gate: `./build.sh check` (CLI+MCP build, module+connector+contract tests, CLI contract smoke, feature-off stubs, `zig fmt --check`, parity).
- Full local gate: `./build.sh full-check` (adds integration, benchmarks, dashboard smoke, `agent tui` line-mode smoke).
- Single test: `zig build test -Dtest-filter="<pattern>"` (or `./build.sh test ...`). The form `zig build test -- --test-filter` is **not wired** and is silently ignored.
- Focused: `zig build test-cli`, `test-plugins`, `test-feature-contracts`, `test-contracts`, `test-mcp-contracts`, `test-mcp-server`, `test-integration`, `benchmarks`, `check-parity`, `lint`, `fix`.
- Opt-in locally: `zig build cross-smoke` (Linux/Windows/macOS cross; deliberately **not** in `check`; slow). CI runs both `zig build check` and `zig build cross-smoke` on `macos-latest` with the pinned Zig.
- Binaries: `./build.sh cli` (`zig-out/bin/abi`), `./build.sh mcp` (`zig-out/bin/abi-mcp`).
- Docs site (optional, outside `check`): `npx mint@latest validate` (`docs/docs.json`).

## Architecture & Boundaries (what filenames do not make obvious)
- Public API root: `src/root.zig`.
- CLI entry: `src/main.zig` → dispatch/usage/registry/handlers under `src/cli/`.
- MCP server (Zig impl): `src/mcp/` (main + handlers group). Repo-root `mcp/` + `.mcp.json` + `mcp/launcher.sh` is **only** host glue/launcher, not the implementation.
- Feature selection: `src/features/mod.zig` picks real `mod.zig` or disabled `stub.zig` via `-Dfeat-*`.
- **All** `-Dfeat-*` default **on** (`ai`, `wdbx`, `gpu`, `accelerator`, `shader`, `mlir`, `os-control`, `tui`, `hash`, `telemetry`, `nn`, `mobile`, `metrics`, `sea`, `foundationmodels`). Turn off with `-Dfeat-<name>=false`.
- `feat-foundationmodels`: comptime-gated to arm64 macOS (`os.tag == .macos and cpu.arch == .aarch64`). Default build still runs `xcrun swiftc` for the `@c` shim + links `FoundationModels.framework`; without Xcode/macOS 26 SDK use `-Dfeat-foundationmodels=false`. On-device needs Apple Intelligence hardware at runtime; `apple-fm` reports `FMUnavailable` elsewhere.
- Generated: `src/plugin_registry.zig` (from `src/plugins/*/abi-plugin.json` + `tools/generate_plugin_registry.zig`). Build step injects it; **never hand-edit**.
- Inside `src/`: use **relative** `.zig` imports. Only the MCP executable + handler module graph (`src/mcp/main.zig` + `handlers.zig`, `ai_tools.zig`, `connector_tools.zig`, `plugin_tools.zig`, `state.zig`) may do `@import("abi")`. Never from modules re-exported by `src/root.zig`.

## CLI Surface (frozen + contract-tested)
Top-level commands (13, order from `src/cli/usage.zig` + `registry.zig`): `help`, `complete`, `train`, `agent`, `backends`, `plugin`, `auth`, `twilio`, `tui`, `dashboard`, `wdbx`, `scheduler`, `nn`.
- `complete [--live] [--model <id>] [--confirm] [--learn] <input>` (`--live` = anthropic over explicit live transport; `--confirm` required for `apple-fm`; `--learn` = SEA loop; aliases resolved via model catalog).
- `agent <plan | train <profile|all> | tui | os <dry-run|execute --confirm>>` (`agent tui` is interactive REPL).
- `wdbx <db <init|verify|compact> | block <insert|get> | query | benchmark | cluster <status|demo|serve <port> [node] [host]> | compute info | secure demo | gpu info | api serve [port]>`.
- `nn train "<text>" | train --jsonl <path> [--field <name>] | sample --text "<corpus>" --seed <char> --n <k>` (miniature pure-Zig char-LM demo; not production).
- Malformed numeric args (ports, counts, node ids) → usage + exit 2.
- **Do not** resurrect legacy names: `version`, `doctor`, `features`, `platform`, `connectors`, `search`, `info`, `chat`, `db`, `serve`.

## MCP Surface (12 tools, asserted in `tests/contracts/surface.zig`)
`ai_run`, `ai_complete`, `ai_train`, `ai_learn`, `wdbx_query`, `scheduler_stats`, `scheduler_info`, `connector_test`, `gpu_status`, `plugin_list`, `wdbx_stats`, `plugin_run`.
- Primary: JSON-RPC 2.0 over stdio (64 KB request cap).
- Optional: HTTP/SSE on loopback `127.0.0.1:8080` (`ABI_MCP_HTTP_PORT` to override; `ABI_MCP_HTTP_TOKEN` for `Authorization: Bearer`). Same bearer scheme for WDBX REST (`abi wdbx api serve`) via `ABI_WDBX_REST_TOKEN`. Both are loopback hardening only.

## API & Contract Rules
- Public feature API change → keep `mod.zig` + `stub.zig` in declaration parity and return `error.FeatureDisabled` when off. Run `zig build check-parity`.
- Parity tool only scans column-0 `pub const`/`pub fn` names (ignores `pub var`, threadlocal, nested); builds host-only std checker so it runs even under mismatched Zig.
- `zig build check` runs feature-off contract smoke + feature-aware public contracts for every `-Dfeat-*`.

## Zig 0.17 Conventions Agents Miss
- Entry: `pub fn main(init: std.process.Init) !void`.
- `ArrayListUnmanaged(T).empty` (not `.init(allocator)`).
- `std.mem.trimEnd` (not `trimRight`); `splitScalar`/`splitAny`/`splitSequence`.
- `foundation.time.unixMs()` for ms timestamps.
- Tests: inline `test {}`; end modules with `std.testing.refAllDecls(@This())`.
- No silent empty `catch {}` in persistence, inference, connector, or data-access paths — propagate or log.

## WDBX / GPU / Connectors
- WDBX: in-process store + segment/WAL persistence + hybrid retrieval. Cluster RPC is real TCP RequestVote/AppendEntries (`ABI_WDBX_CLUSTER_TOKEN` required for non-loopback binds; `ABI_WDBX_CLUSTER_PEERS` allowlist), but still **not** production multi-host deployment or sharding.
- GPU/vector: reports capability and falls back deterministically to CPU. No `-Dgpu-backend` option.
- Live connectors: require explicit credentials + live transport selection. Local helpers and `connector_test` must stay offline.

## Claims & Docs
- Do not claim unproven capabilities (distributed sharding, full production FHE, AES/RBAC, K8s/H100, Swift/Python/TF stacks, QPS/latency/accuracy, energy, non-loopback hardening, regulatory certs) unless source/tests prove it.
- Public wording: consult `docs/contracts/external-claims-audit.mdx`.
- Security surface: `abi-threat-model.md` (MCP/WDBX listeners, creds at rest).

## Keep in Sync
- Root siblings `CLAUDE.md` and `GEMINI.md` restate the same durable conventions. When commands, contracts, feature flags, or Zig patterns change, update all three.
