# GEMINI.md - ABI Framework

Quick reference for Google Gemini and compatible agents working on this Zig 0.17 ABI framework. Prefer executable config and source over prose when they disagree.

## First Commands

```bash
./build.sh check              # primary gate on macOS/Darwin
./build.sh full-check         # check + integration tests + benchmarks + dashboard/agent TUI smoke
./build.sh cli                # build zig-out/bin/abi
./build.sh mcp                # build zig-out/bin/abi-mcp
zig build test-integration    # explicit integration suite
zig build benchmarks          # explicit benchmark suite
npx mint@latest validate   # optional Mintlify docs site (docs/docs.json); not in CI/check
```

Zig is pinned by `.zigversion` to `0.17.0-dev.978+a078d55a2`; `build.zig.zon` keeps `0.17.0-dev.978+a078d55a2` as the package minimum. Plain `zig build` may work with a compatible local toolchain, but use `./build.sh ...` on macOS for the documented Darwin workflow. Note: `build.sh`/`tools/build.sh` do not switch or enforce the pin — they run whatever `zig` is on `PATH` (Zig `0.16.0` fails to compile, since the WDBX/MCP network listeners use the 0.17 `std.Io.net.Stream.read(io, …)` API).

`zig build cross-smoke` is opt-in locally (Linux/Windows/macOS CLI compile/link smoke; deliberately outside `check`), but CI runs both `zig build check` and `zig build cross-smoke` on `macos-latest` with the pinned Zig.

Local Codex mega-plugin handoff: `ABI-MEGA-PLUGIN.md` points to the personal `abi-mega` plugin that consolidates TODO/roadmap/spec/skill inventory and focused validation workflows.

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

Supported top-level commands are `help`, `complete`, `train`, `agent`, `backends`, `plugin`, `auth`, `twilio`, `tui`, `dashboard`, `wdbx`, `scheduler`, and `nn`. The top-level `abi --tui` shortcut also renders the dashboard and accepts the same flags. `help --json [command] [subcommand]` emits typed command/subcommand plus shortcut and completion-shell metadata for automation; `help --completion <bash|zsh|fish>` emits metadata-driven shell completions; `tui` / `dashboard` accept `--pane <pane>`, `--plain`/`--no-color`, `--compact`, `--once`, `--interval <ms>` (100-60000), `--json`, and `--list-panes`; dashboard JSON includes layout metadata (`compact`, color, visible panes, pane titles/hotkeys). (`nn` is a miniature char-level demo trainer behind `feat-nn`: `nn train "<text>" | train --jsonl <path> | sample …` — not a production/LLM trainer.)

Subcommand grammar mirrors `src/cli/`: `complete [--live] [--model <id>] [--confirm] [--learn] <input>` (the model id alias-resolves through the catalog, `--live` serves anthropic models over the live transport, an unrecognized id prints a stderr warning before passing through, `--confirm` is required for on-device `apple-fm`, and `--learn` routes through the SEA self-learning loop; `agent tui` is now an interactive REPL with `/help /model /status /history /reset /quit`, and `/model` accepts printable non-whitespace ids only); `agent <plan | train <profile|all> | tui | os <dry-run|execute --confirm>>`; and `wdbx <db <init|verify|compact> | block <insert|get> | query | benchmark | cluster <status|demo|serve <port> [node] [host]> | compute info | secure demo | gpu info | api serve [port]>`. Malformed numeric arguments return usage with exit code 2 rather than silently using a default.

Do not assume old command names exist: `version`, `doctor`, `features`, `platform`, `connectors`, `search`, `info`, `chat`, `db`, and `serve` are not currently dispatched.

## Project Map

| Path | Purpose |
| --- | --- |
| `src/root.zig` | Public `abi` module root |
| `src/main.zig` | CLI entry |
| `src/cli/` | CLI usage, dispatch, handlers |
| `src/mcp/main.zig` | MCP entry point (spawns HTTP, runs stdio loop) |
| `src/features/mod.zig` | Feature flag mod/stub selection |
| `src/features/ai/` | AI profiles, router, constitution, training, local streaming helpers; SEA scorer (8-signal), IoT monitor (z-score anomaly), multimodal fusion (vision/audio/IoT vectors), adaptive routing |
| `src/features/wdbx/` | In-process store, segment/WAL persistence, hybrid retrieval, real TCP cluster RPC; `runAuthenticatedLoopbackRound()` multi-node helper |
| `src/features/gpu/` | GPU status, Metal attempt on macOS, CPU fallback |
| `src/features/tui/` | Operational diagnostics dashboard renderer |
| `src/connectors/` | OpenAI, Anthropic, Discord, Grok, Twilio connector surfaces |
| `src/foundation/` | Time, sync, logger, utils, errors, OS, IO, credentials |
| `src/core/registry.zig` | Generated plugin registry loading and metadata accessors |
| `src/plugins/plugin_manager.zig` | Required manifest validation and local plugin manager API |
| `tools/` | Build helpers, plugin registry generation, parity checker |

## Feature Flags

Default enabled (all `-Dfeat-*` flags): `feat-ai`, `feat-gpu`, `feat-tui`, `feat-accelerator`, `feat-shader`, `feat-mlir`, `feat-wdbx`, `feat-os-control`, `feat-hash`, `feat-telemetry`, `feat-nn`, `feat-mobile`, `feat-metrics`, `feat-sea` (includes 8-signal weighted scorer, `MemoryKind`/`Authority` types, budgeted greedy selection, context packing), `feat-foundationmodels`.

Default disabled: none — turn any feature off with `-Dfeat-<name>=false`. Special notes: `feat-sea` (`src/features/sea/`, Sparse Evidence Attention self-learning loop, requires `feat-wdbx` at runtime) and `feat-foundationmodels` (`src/connectors/fm.zig`, Apple on-device FoundationModels — defaults on but the `FoundationModels.framework` + `swiftc`-built `libabi_fm_shim.dylib` link is comptime-gated on an arm64 macOS target (`os.tag == .macos and cpu.arch == .aarch64`), so non-macOS and x86_64-macOS builds compile it out and `apple-fm` reports `FMUnavailable`. CAVEAT: the default build on an arm64 macOS host still runs `xcrun swiftc -target arm64-apple-macosx26.0`, requiring the Xcode/Swift toolchain + macOS 26 SDK; lacking those, build `-Dfeat-foundationmodels=false`. On-device generation is wired through a Swift `@c` shim (SE-0495) and requires Apple-Intelligence hardware at runtime).

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
- Tools (12, count asserted in `tests/contracts/surface.zig`): `ai_run`, `ai_complete`, `ai_train`, `ai_learn`, `wdbx_query`, `scheduler_stats`, `scheduler_info`, `connector_test`, `gpu_status`, `plugin_list`, `wdbx_stats`, `plugin_run`. (`ai_learn` runs the SEA loop; degrades to a stored completion when `feat-sea` is off.) Completion output includes `audit_escore=` (weighted constitutional E-score 0-1) and `audit_vetoed=` (hard veto flag) from the six-principle constitution audit.

## Development Rules

- Read `tasks/lessons.md` and `tasks/todo.md` before substantial work.
- Public feature API changes require matching `mod.zig` and `stub.zig` updates.
- Run `zig build check-parity` after public API changes. It only matches column-0 `pub const `/`pub fn ` names (not `pub var`/`pub threadlocal`/nested decls), and builds just the std-only host checker, so it runs even when the feature graph won't compile under a mismatched Zig.
- Do not edit generated `src/plugin_registry.zig`; change plugin source/manifests or `tools/generate_plugin_registry.zig`.
- Plugin manifests require `name`, `version`, `description`, `target_feature`, and a safe relative `.zig` `entry_point` whose file exists under the plugin directory; `targetFeature` / `entryPoint` aliases are accepted. Generated multi-plugin registry metadata is covered by `tests/contracts/plugin_registry.zig`.
- Discord connector IDs are validated as numeric snowflake-like IDs, and local/live paths enforce printable non-whitespace credentials, author ID validation, and message-size checks.
- Twilio validates account SIDs as `AC` + 32 hex characters, auth tokens as 32 hex characters, non-empty base URL, non-zero timeout, explicit `.live` transport selection, XML/form escaping, and ConversationRelay aliases before local/live dispatch.
- SEA scorer details: `src/features/sea/scorer.zig` implements 8-signal weighted scoring (`SeaSignals`), `selectSeaCandidates()` for budgeted greedy selection, and `contextPack()` for evidence rendering. `src/features/sea/types.zig` defines `MemoryKind` (9 variants) and `Authority` (5 rungs) with parse/text/score round-trips. Adaptive routing uses `completeAdaptive()`/`completeWithStoreAdaptive()` in `src/features/ai/completion.zig` with persisted `AdaptiveModulator` weights.
- IoT monitor (`src/features/ai/iot_monitor.zig`): `IotMonitor` with z-score anomaly detection (Welford's online mean/variance, configurable threshold). Multimodal fusion (`src/features/ai/multimodal_fusion.zig`): `VisionProcessor` (64-d), `AudioProcessor` (32-d), `IotProcessor` (16-d) with deterministic character-bucket embedding + L2 normalization; `fuse()` concatenative combinator.
- Only the MCP executable + handler module graph (`src/mcp/main.zig` plus the `handlers.zig` group: `handlers.zig`, `ai_tools.zig`, `connector_tools.zig`, `plugin_tools.zig`, `state.zig`) may import `@import("abi")` from inside `src/` — never modules re-exported by `src/root.zig`; other `src` imports should be relative `.zig` imports.
- WDBX `cluster serve` is real TCP RequestVote/AppendEntries; `ABI_WDBX_CLUSTER_TOKEN` is required for non-loopback binds and `ABI_WDBX_CLUSTER_PEERS` can allowlist node ids. `runAuthenticatedLoopbackRound()` provides a deterministic multi-node vote+append helper with quorum verification (tested with 4-node round). This is not production multi-host deployment or sharding.
- Do not claim unproven capabilities (distributed sharding, full production FHE, AES/RBAC, non-loopback hardening, K8s/H100, Swift/Python/TF stacks, certifications, QPS/latency/accuracy/energy numbers) unless source/tests prove them; use `docs/contracts/external-claims-audit.mdx` for public wording.
- Do not use plain `rm`; use safe alternatives.

## Zig 0.17 Notes

- Entry point: `pub fn main(init: std.process.Init) !void`.
- Use `ArrayListUnmanaged(T).empty`.
- Use `std.mem.trimEnd`.
- Use `std.mem.splitScalar`, `splitAny`, or `splitSequence`.
- Use `foundation.time.unixMs()` for timestamps.
- `@panic` only for unrecoverable invariants, `unreachable` only for provably impossible branches.
- Tests are inline `test {}`; each module ends with `std.testing.refAllDecls(@This())`.
- Naming: `camelCase` for fns/vars, `PascalCase` for types, `SCREAMING_SNAKE_CASE` for constants, `snake_case` for enum variants.

## Verification

Before finishing code changes, run:

```bash
./build.sh check
```

`./build.sh check` includes feature-off compile smoke tests, focused `test-feature-contracts` behavior coverage, feature-aware `test-contracts` coverage for every `-Dfeat-*` flag, and `-Dfeat-mobile=true` coverage through `tools/check_feature_stubs.sh`.

Use `./build.sh full-check` when you need the full local gate, including integration tests, benchmarks, dashboard smoke, and `agent tui` line-mode smoke.

Run these explicitly when touched:

```bash
zig build test-integration
zig build benchmarks
zig build test-mcp-server     # MCP server transport tests (stdio + HTTP/SSE)
zig build test-plugins        # Bundled plugin refAllDecls coverage
```

`tasks/todo.md` is the current source of truth for known failures. At the current modernization point, no known test failures are reproduced locally.

## Project Skills

`.claude/skills/` ships task-specific skills that build a real binary and exercise one surface — prefer them over ad-hoc commands when they match. Each is scoped to files under `abi/`: `run-abi` (build/launch CLI + `abi-mcp`), `backend-diagnostics` (GPU/accelerator/shader/MLIR + compute matrix), `wdbx-bench` (insert/search timing), `cluster-demo-guide` (Raft consensus/failover), `secure-demo` (compression + homomorphic encryption), `sea-learn-loop` (`complete --learn` SEA path), `connector-localcheck` (Twilio/auth, no network), `os-control-dryrun` (policy dry-run, never executes), `plugin-runtime-tester` (registry + `plugin run` dispatch), `cross-compile-check` (`zig build cross-smoke`), `zig-pin` / `zig-newest-skills` (toolchain pin vs. master-nightly forward-compat), `nn-demo` (char-LM demo train/sample), `agent-plan-train` (`agent plan` + `agent train`), `wdbx-roundtrip` (db init→insert→query→verify), `auth-localcheck` (auth status/signin wiring, no creds touched), `complete-base` (base local `complete`, no `--live`/`--learn`), `wdbx-api-serve` (WDBX REST server + bearer-auth smoke), `wdbx-cluster-serve` (networked WDBX consensus node), `run-tui` (interactive diagnostics dashboard via a tmux pty), `mcp-smoke` (abi-mcp JSON-RPC `tools/list` 12-tool contract smoke), `scheduler-status` (one-shot `abi scheduler status` probe-task + telemetry smoke), `dashboard-smoke` (non-interactive one-shot `abi dashboard` 5-panel render), `docs-validate` (user-only `npx mint validate` pre-push gate for the Mintlify docs site).

## OpenCode Setup

- Project config: `opencode.json` (root, loaded automatically; schema `https://opencode.ai/config.json`).
- Instruction files: `AGENTS.md`, `tasks/lessons.md`, `tasks/todo.md`.
- `.opencode/skills/` is a symlink to `.agents/skills/` — the canonical skill set.
- MCP servers: `abi-mcp` (`mcp/launcher.sh`), `skill-loop` (`@stylusnexus/skill-loop-cli@0.3.3`).
- OpenCode MCP entries use `type: "local"`, `enabled: true`, and a single `command` array; do not copy `.mcp.json`'s `command` + `args` shape into `opencode.json`.
- Sync canonical skills to other CLI targets: `.agents/skills/sync-clis/launch.sh`.
