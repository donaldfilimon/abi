# ABI Framework

ABI is a **Zig 0.17.0-dev** framework for local AI service orchestration, semantic vector storage, GPU capability reporting, and runtime primitives.

## Quick Start
```bash
zig version             # Confirm the pinned Zig 0.17.0-dev.1398+cb5635714 toolchain (see .zigversion)
./build.sh check        # Primary validation gate on macOS/Darwin
./build.sh full-check   # Check + integration tests + benchmarks + dashboard/agent TUI smoke
./build.sh cli          # Build zig-out/bin/abi
./build.sh mcp          # Build zig-out/bin/abi-mcp
```

Plain `zig build` and `zig build check` are expected to work with the pinned toolchain. On macOS/Darwin, keep using `./build.sh ...` for the documented project workflow.

## Local Walkthrough

Build the CLI, then exercise the local surfaces without live network credentials:

```bash
./build.sh cli
./zig-out/bin/abi backends
./zig-out/bin/abi scheduler status
./zig-out/bin/abi dashboard < /dev/null   # one-shot operational snapshot: health, plugins, WDBX, scheduler, memory
./zig-out/bin/abi complete "summarize ABI scheduler status"
./zig-out/bin/abi complete --model fable-5 "summarize ABI scheduler status"   # alias resolves to claude-fable-5; unknown ids warn and pass through
./zig-out/bin/abi agent plan "stage a safe WDBX refactor"
./zig-out/bin/abi agent train all
./zig-out/bin/abi wdbx db init zig-out/local-memory.jsonl
./zig-out/bin/abi wdbx block insert zig-out/local-memory.jsonl abi "{\"note\":\"local memory checkpoint\"}"
./zig-out/bin/abi wdbx query zig-out/local-memory.jsonl
./zig-out/bin/abi wdbx db compact zig-out/local-memory.jsonl 2
./zig-out/bin/abi wdbx benchmark 256          # local insert/query timing with P50/P95/P99
./zig-out/bin/abi wdbx compute info           # CPU/GPU/NPU/TPU backends, ANE detection, remote-dispatch endpoint
./zig-out/bin/abi wdbx secure demo            # int8 + autoencoder compression; additive HE + DGHV add/multiply SHE demos
./zig-out/bin/abi wdbx cluster serve 8090     # networked consensus RPC node endpoint (loopback; set ABI_WDBX_CLUSTER_TOKEN for non-loopback)
```

`abi scheduler status` runs a one-shot scheduler probe and reports task/memory counters. `abi help --json [command] [subcommand]` emits typed command/subcommand plus shortcut and completion-shell metadata for automation, and `abi help --completion <bash|zsh|fish>` emits metadata-driven shell completions. `abi dashboard` / `abi tui` renders the operational diagnostics snapshot; `abi --tui` is the same dashboard shortcut. With non-TTY stdin it exits after one frame, and on a real TTY it auto-refreshes until `q`/Esc. Use `--pane <pane>`, `--plain`/`--no-color`, `--compact`, `--once`, `--interval <ms>` (100-60000), `--json`, and `--list-panes` to choose the initial pane, log-safe styling, selected-pane-only rendering, forced one-shot rendering, refresh cadence, a machine-readable snapshot, or pane metadata; JSON snapshots include layout metadata (`compact`, color, visible panes, pane titles/hotkeys). `abi agent plan`, `train`, `multi`, and `spawn` use the local scheduler-backed AI helper surface; `abi agent tui` is the interactive REPL with `/status` session telemetry and validated printable model ids. `abi agent browser` emits a reviewed local plan and local planner output only—ABI does not embed or launch a browser, and real navigation remains an external MCP integration step. Training writes local WDBX records when `feat-ai` and `feat-wdbx` are enabled; feature-disabled builds return explicit degraded responses instead of fabricating work. `wdbx query <path> [text] [persona]` does store-stats, hybrid semantic search, or persona-isolated retrieval; `cluster serve`/`compute info`/`secure demo` expose the networked consensus RPC, accelerator selection, and security demos honestly as single-host / reference-scoped surfaces (see `docs/spec/wdbx-north-star.mdx` for the Current/Partial/Proposed mapping). Cluster RPC supports shared-secret frames via `ABI_WDBX_CLUSTER_TOKEN` and an optional node allowlist via `ABI_WDBX_CLUSTER_PEERS`; non-loopback binds refuse to start without the token. `complete --model <id>` records the catalog-canonical id (aliases such as `fable-5` -> `claude-fable-5`; unrecognized ids print a stderr warning and pass through); `complete --live` serves anthropic-provider models over the explicit live transport and therefore needs stored credentials (`abi auth signin anthropic`).

For MCP smoke testing, build the server and call the same contract tools through an MCP client:

```bash
./build.sh mcp
./zig-out/bin/abi-mcp stdio
```

Contract-covered MCP tool names are `ai_run`, `ai_complete`, `ai_train`, `ai_learn`, `wdbx_query`, `scheduler_stats`, `scheduler_info`, `connector_test`, `gpu_status`, `plugin_list`, `wdbx_stats`, and `plugin_run`. `wdbx_query` returns a local hybrid-ranked match when WDBX is enabled; `connector_test` uses deterministic local connector paths and does not perform live network dispatch.

## Current Status

- ABI currently targets the pinned Zig 0.17.0 development toolchain and should be validated from the live checkout before release or external reuse.
- Core feature modules and MCP transport have contract coverage; MCP HTTP can use `ABI_MCP_HTTP_PORT` when `127.0.0.1:8080` is occupied.
- Documentation: `CLAUDE.md`, `GEMINI.md`, and `AGENTS.md` describe the 0.17+ development lifecycle.
- Build: `./build.sh check` builds CLI/MCP, runs module tests, connector tests, contract tests, focused feature-off contracts, feature-aware public contracts for every `-Dfeat-*` stub, linting, and validates mod/stub parity.
- Full validation: `./build.sh full-check` executes all integration tests, benchmarks, dashboard smoke, and `agent tui` line-mode smoke.
- GPU/modules: all feature modules keep real/stub parity; GPU status and vector operations fall back deterministically to CPU when native kernels are unavailable.
- Plugins: `tools/generate_plugin_registry.zig` automatically maintains `src/plugin_registry.zig` from bundled `abi-plugin.json` manifests, with multi-plugin metadata coverage for `name`, `version`, `description`, `target_feature`, and safe relative `.zig` `entry_point` paths. The generator and plugin manager accept `targetFeature` / `entryPoint` aliases and require entry files to exist under the plugin directory.
- AI/WDBX: API callers opt into persistence with `CompletionRequest.store_result=true`; CLI/MCP completion uses WDBX stores for query/response vectors, metadata, and block-chain entries when WDBX is enabled. Scheduler-backed completion, training, and agent helpers expose live task/memory observability.
- WDBX: contract coverage verifies ordered vector search results, block metadata round-tripping, segment/WAL recovery and compaction, temporal graph snapshot restore, and MCP hybrid ranking while disabled builds return explicit `error.FeatureDisabled` operations for key-value/vector/search/block/spatial/temporal writes.
- Connectors: Discord validates printable non-whitespace credentials, numeric snowflake-like IDs, and message size; Twilio validates account SID/auth-token shape, base URL, timeout, explicit `.live` transport, TwiML/form escaping, and ConversationRelay payload aliases before local/live dispatch.
- External collateral should not cite distributed sharding, AES/RBAC, Swift, Python/TensorFlow stacks, Kubernetes/H100 deployments, regulatory certifications, QPS/latency/accuracy, energy-efficiency, or model-benchmark claims unless a repo test, benchmark artifact, or documented source file proves them; see [docs/contracts/external-claims-audit.mdx](docs/contracts/external-claims-audit.mdx).

See [docs/index.mdx](docs/index.mdx) for architecture, public API contracts, onboarding, and development guides, and [CHANGELOG.md](CHANGELOG.md) for release-note style modernization highlights.
