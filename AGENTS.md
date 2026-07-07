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
- `help [--json|--completion <bash|zsh|fish>] [command] [subcommand]` (`--json` emits typed command/subcommand metadata plus shortcut and completion-shell metadata for automation/docs tooling; `--completion` emits metadata-driven shell completions).
- `complete [--live] [--model <id>] [--confirm] [--learn] <input>` (`--live` = anthropic over explicit live transport; `--confirm` required for `apple-fm`; `--learn` = SEA loop; aliases resolved via model catalog). Completion output includes `audit_escore=` (weighted constitutional E-score 0-1) and `audit_vetoed=` (hard veto flag) from the six-principle constitution audit.
- `agent <plan | train <profile|all> | tui | os <dry-run|execute --confirm>>` (`agent tui` is interactive REPL with `/help`, `/model`, `/status`, `/history`, `/reset`, `/quit`; `/model` accepts printable non-whitespace ids only).
- `tui` / `dashboard` render the diagnostics dashboard; flags: `--pane <pane>`, `--plain`/`--no-color`, `--compact`, `--once`, `--interval <ms>` (100-60000), `--json`, `--list-panes`. JSON snapshots include layout metadata (`compact`, color, visible panes, pane titles/hotkeys). The legacy top-level `--tui` shortcut routes through the same parser and accepts the same dashboard flags.
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
- WDBX: in-process store + segment/WAL persistence + hybrid retrieval. Cluster RPC is real TCP RequestVote/AppendEntries (`ABI_WDBX_CLUSTER_TOKEN` required for non-loopback binds; `ABI_WDBX_CLUSTER_PEERS` allowlist), but still **not** production multi-host deployment or sharding. `runAuthenticatedLoopbackRound()` provides a deterministic multi-node vote+append helper with quorum verification (tested with 4-node round).
- GPU/vector: reports capability and falls back deterministically to CPU. No `-Dgpu-backend` option.

## AI Subsystem Details
- **SEA scorer** (`src/features/sea/scorer.zig`): 8-signal weighted scorer (`SeaSignals`: semantic, keyword, metadata, recency, authority, graph, contradiction, task_fit), `DEFAULT_SEA_WEIGHTS` (sum to 1.0), `seaScore()` weighted sum clamped to [0,1], `adjustWeightsForTask()` for per-task tuning (code_repair, project_recall, benchmark_review), `selectSeaCandidates()` budgeted greedy selection (token/record/cluster budgets with ≥0.92 high-score escape hatch), `contextPack()` evidence renderer.
- **SEA types** (`src/features/sea/types.zig`): `MemoryKind` (9 variants: note, user_preference, project_decision, code_fact, tool_output, benchmark, constraint, contradiction, summary), `Authority` (5 rungs: inferred, user_stated, tool_verified, file_verified, system_pinned) with parse/text/score round-trips.
- **IoT monitor** (`src/features/ai/iot_monitor.zig`): `IotMonitor` with z-score anomaly detection (Welford's online mean/variance, configurable threshold default 2.5), history tracking, `feed()` → bool, `reset()`.
- **Multimodal fusion** (`src/features/ai/multimodal_fusion.zig`): `VisionProcessor` (64-d), `AudioProcessor` (32-d), `IotProcessor` (16-d) with deterministic character-bucket embedding + L2 normalization; `fuse()` concatenative combinator.
- **Adaptive routing** (`src/features/ai/completion.zig`): `completeAdaptive()` loads persisted `AdaptiveModulator` weights from WDBX, updates via EMA based on sentiment, selects best profile; `completeWithStoreAdaptive()` wraps with persistence. SEA learn loop uses this path.
- Live connectors: require explicit credentials + live transport selection. Local helpers and `connector_test` must stay offline.
- Connector validation:
  - Discord: validates printable non-whitespace credentials, numeric snowflake-like IDs, author IDs, message size.
  - Twilio: validates `AC` + 32-hex account SIDs, 32-hex auth tokens, base URL, timeout, explicit `.live` transport, XML/form escaping, ConversationRelay aliases.

## Claims & Docs
- Do not claim unproven capabilities (distributed sharding, full production FHE, AES/RBAC, K8s/H100, Swift/Python/TF stacks, QPS/latency/accuracy, energy, non-loopback hardening, regulatory certs) unless source/tests prove it.
- Public wording: consult `docs/contracts/external-claims-audit.mdx`.
- Security surface: `abi-threat-model.md` (MCP/WDBX listeners, creds at rest).

## OpenCode Setup
- Project config: `opencode.json` (root, loaded automatically; schema `https://opencode.ai/config.json`).
- Instruction files: `AGENTS.md`, `tasks/lessons.md`, `tasks/todo.md`.
- `.opencode/skills/` is a symlink to `.agents/skills/` — the canonical skill set.
- MCP servers: `abi-mcp` (`mcp/launcher.sh`), `skill-loop` (`@stylusnexus/skill-loop-cli@0.3.3`).
- OpenCode MCP entries use `type: "local"`, `enabled: true`, and a single `command` array; do not copy `.mcp.json`'s `command` + `args` shape into `opencode.json`.
- Sync canonical skills to other CLI targets: `.agents/skills/sync-clis/launch.sh`.

## Keep in Sync
- Root siblings `CLAUDE.md` and `GEMINI.md` restate the same durable conventions. When commands, contracts, feature flags, or Zig patterns change, update all three.
