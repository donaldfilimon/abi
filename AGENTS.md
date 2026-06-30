# Repository Guidelines

## First Reads
- Trust executable config over prose: `build.zig`, `tools/build.sh`, and source win over docs when they conflict.
- Read `tasks/lessons.md` and `tasks/todo.md` before substantial work; they track current ABI-specific conventions and known status.
- The worktree may be dirty from another agent/user. Inspect `git status --short --branch`; never revert unrelated changes.

## Commands That Matter
- Toolchain is pinned by `.zigversion` to Zig `0.17.0-dev.978+a078d55a2`; `build.zig.zon` only sets the package minimum. `build.sh`/`tools/build.sh` do **not** switch or enforce the pin — they run whatever `zig` is on `PATH`. The system `zig` must already be that dev build (Zig `0.16.0` fails: the WDBX/MCP listeners use the 0.17 `std.Io.net.Stream.read(io, …)` API).
- On macOS/Darwin prefer `./build.sh ...`; it delegates to `tools/build.sh` and keeps the documented Metal/link workflow.
- Primary gate: `./build.sh check` builds CLI/MCP, runs module + connector + contract tests, CLI contract smoke, feature-off stub contracts, `zig fmt --check`, and parity.
- Full local gate: `./build.sh full-check` adds integration tests, benchmarks, and TUI smoke.
- Focused commands: `zig build test -Dtest-filter="<pattern>"`, `zig build test-integration`, `zig build test-mcp-contracts`, `zig build test-mcp-server` (MCP transport: stdio + HTTP/SSE), `zig build check-parity`, `zig build lint`, `zig build fix`.
- Build binaries with `./build.sh cli` (`zig-out/bin/abi`) and `./build.sh mcp` (`zig-out/bin/abi-mcp`).

## Architecture Anchors
- Public API root is `src/root.zig`; CLI entry is `src/main.zig` with dispatch under `src/cli/`; MCP server code is under `src/mcp/`.
- Repo-root `mcp/` is launcher/config glue (`.mcp.json` calls `mcp/launcher.sh`), not the Zig MCP implementation.
- Feature selection happens in `src/features/mod.zig`; each feature uses a real `mod.zig` and disabled `stub.zig` selected by `-Dfeat-*`.
- Feature defaults from `build.zig`: enabled `ai`, `wdbx`, `gpu`, `accelerator`, `shader`, `mlir`, `os-control`, `tui`, `hash`, `telemetry`, `nn`; disabled `mobile`, `metrics`.
- Generated plugin metadata is `src/plugin_registry.zig`; do not hand-edit it. Build regeneration comes from `tools/generate_plugin_registry.zig` and `src/plugins/*/abi-plugin.json`.

## API And Contract Gotchas
- Public feature API changes usually require matching real/stub declarations and disabled behavior returning `error.FeatureDisabled`; run `zig build check-parity`.
- Frozen CLI surface is contract-tested. Top-level commands are `help`, `complete`, `train`, `agent`, `backends`, `plugin`, `auth`, `twilio`, `tui`, `dashboard`, `wdbx`, `scheduler`, `nn`; `abi --tui` is handled separately in `src/main.zig`. (`nn` is a miniature char-level demo trainer behind `feat-nn` — `nn train "<text>" | train --jsonl <path> | sample …`.)
- Subcommand grammar (do not drift from `src/cli/`): `complete [--live] [--model <id>] [--confirm] [--learn] <input>` (alias-resolves via the model catalog; `--live` serves anthropic models over the live transport; `--confirm` is required for on-device `apple-fm`; `--learn` routes through the SEA self-learning loop; `agent tui` is now an interactive REPL); `agent <plan | train <profile|all> | tui | os <dry-run|execute --confirm>>`; `wdbx <db <init|verify> | block <insert|get> | query | benchmark | cluster <status|demo|serve> | compute info | secure demo | gpu info | api serve>`. Malformed numeric args (counts/ports/node ids) return usage (exit 2), not a silent default.
- Do not resurrect legacy CLI names such as `version`, `doctor`, `features`, `platform`, `connectors`, `search`, `info`, `chat`, `db`, or `serve`.
- MCP tools are contract-tested (12, count asserted in `tests/contracts/surface.zig`): `ai_run`, `ai_complete`, `ai_train`, `ai_learn`, `wdbx_query`, `scheduler_stats`, `scheduler_info`, `connector_test`, `gpu_status`, `plugin_list`, `wdbx_stats`, `plugin_run`. (`ai_learn` runs the SEA loop; degrades to a stored completion when `feat-sea` is off.)
- New default-off feature flags: `feat-sea` (`src/features/sea/`, Sparse Evidence Attention self-learning) and `feat-foundationmodels` (`src/connectors/fm.zig`, Apple on-device FoundationModels — macOS-only, links `FoundationModels.framework` + a `swiftc`-built `libabi_fm_shim.dylib`; on-device generation is wired through a Swift `@c` shim (SE-0495) and requires Apple-Intelligence hardware at runtime). Model id `apple-fm` (`fm-local`/`fm`) routes to provider `fm`.
- MCP stdio has a 64 KB request cap; optional HTTP/SSE binds loopback `127.0.0.1:8080` unless `ABI_MCP_HTTP_PORT` selects another valid port. `ABI_MCP_HTTP_TOKEN` requires `Authorization: Bearer <token>` on the HTTP/SSE transport (stdio stays tokenless); the WDBX REST listener (`abi wdbx api serve`) uses `ABI_WDBX_REST_TOKEN` for the same scheme. Both are loopback-only hardening, not a TLS substitute.

## Zig Conventions Agents Miss
- Inside `src/`, use relative `.zig` imports. Only the MCP executable + handler module graph (`src/mcp/main.zig` plus the `handlers.zig` group: `handlers.zig`, `ai_tools.zig`, `connector_tools.zig`, `plugin_tools.zig`, `state.zig`) should import `@import("abi")`; never modules re-exported by `src/root.zig`.
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
- `abi-threat-model.md` (repo root) is the AppSec-grade, repo-path-anchored threat model (MCP/WDBX loopback listeners, credentials at rest); consult it before changing listener, auth, credential, or OS-control surfaces.
