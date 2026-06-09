# Repository Guidelines

## First Reads
- Trust executable config over prose: `build.zig`, `tools/build.sh`, and source win over docs when they conflict.
- Read `tasks/lessons.md` and `tasks/todo.md` before substantial work; they track current ABI-specific conventions and known status.
- The worktree may be dirty from another agent/user. Inspect `git status --short --branch`; never revert unrelated changes.

## Commands That Matter
- Toolchain is pinned by `.zigversion` to Zig `0.17.0-dev.813+2153f8143`; `build.zig.zon` only sets the package minimum.
- On macOS/Darwin prefer `./build.sh ...`; it delegates to `tools/build.sh` and keeps the documented Metal/link workflow.
- Primary gate: `./build.sh check` builds CLI/MCP, runs module + connector + contract tests, CLI contract smoke, feature-off stub contracts, `zig fmt --check`, and parity.
- Full local gate: `./build.sh full-check` adds integration tests, benchmarks, and TUI smoke.
- Focused commands: `zig build test -Dtest-filter="<pattern>"`, `zig build test-integration`, `zig build test-mcp-contracts`, `zig build check-parity`, `zig build lint`, `zig build fix`.
- Build binaries with `./build.sh cli` (`zig-out/bin/abi`) and `./build.sh mcp` (`zig-out/bin/abi-mcp`).

## Architecture Anchors
- Public API root is `src/root.zig`; CLI entry is `src/main.zig` with dispatch under `src/abi_cli/`; MCP server code is under `src/mcp/`.
- Repo-root `mcp/` is launcher/config glue (`.mcp.json` calls `mcp/launcher.sh`), not the Zig MCP implementation.
- Feature selection happens in `src/features/mod.zig`; each feature uses a real `mod.zig` and disabled `stub.zig` selected by `-Dfeat-*`.
- Feature defaults from `build.zig`: enabled `ai`, `wdbx`, `gpu`, `accelerator`, `shader`, `mlir`, `os-control`, `tui`, `hash`, `telemetry`; disabled `mobile`, `metrics`.
- Generated plugin metadata is `src/plugin_registry.zig`; do not hand-edit it. Build regeneration comes from `tools/generate_plugin_registry.zig` and `src/plugins/*/abi-plugin.json`.

## API And Contract Gotchas
- Public feature API changes usually require matching real/stub declarations and disabled behavior returning `error.FeatureDisabled`; run `zig build check-parity`.
- Frozen CLI surface is contract-tested. Top-level commands are `help`, `complete`, `train`, `agent`, `backends`, `plugin`, `auth`, `twilio`, `tui`, `dashboard`, `wdbx`, `scheduler`; `abi --tui` is handled separately in `src/main.zig`.
- Do not resurrect legacy CLI names such as `version`, `doctor`, `features`, `platform`, `connectors`, `search`, `info`, `chat`, `db`, or `serve`.
- MCP tools are contract-tested: `ai_run`, `ai_complete`, `ai_train`, `wdbx_query`, `scheduler_stats`, `scheduler_info`, `connector_test`, `gpu_status`, `plugin_list`, `wdbx_stats`, `plugin_run`.
- MCP stdio has a 64 KB request cap; optional HTTP/SSE binds loopback `127.0.0.1:8080` unless `ABI_MCP_HTTP_PORT` selects another valid port.

## Zig Conventions Agents Miss
- Inside `src/`, use relative `.zig` imports. Only `src/mcp/main.zig` and `src/mcp/handlers.zig` should import `@import("abi")`.
- Use Zig 0.17 idioms already present here: `pub fn main(init: std.process.Init) !void`, `ArrayListUnmanaged(T).empty`, `std.mem.trimEnd`, `splitScalar`/`splitAny`/`splitSequence`, and `foundation.time.unixMs()`.
- Inline tests are the norm; modules should end with `std.testing.refAllDecls(@This())` coverage unless there is a clear reason not to.
- Avoid silent empty `catch {}` in persistence, inference, connector, or data-access paths; propagate or log errors.

## WDBX, GPU, Connectors
- WDBX runtime uses WAL/segment checkpoint paths and hybrid temporal/causal retrieval, but in-process demos are not production distributed storage.
- GPU/vector code reports runtime capability and falls back deterministically to CPU; there is no `-Dgpu-backend` build option and no native accelerator dispatch claim unless source proves it.
- Live connectors must require explicit credentials and live transport selection. Deterministic local helpers and `connector_test` should not hit the network.

## Claims And Docs
- Do not claim distributed sharding, AES/RBAC, regulatory certifications, Kubernetes/H100 deployments, Swift/Python/TensorFlow stacks, QPS/latency/accuracy, energy, learned compression, full FHE, or non-loopback REST hardening unless source/tests/artifacts prove it.
- For public-facing capability wording, check `docs/contracts/external-claims-audit.md` and keep unimplemented north-star items framed as targets/proposed work.
