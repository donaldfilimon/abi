# GEMINI.md — abi

Quick reference for Google Gemini and compatible agents. Canonical instruction file is `AGENTS.md` — if this conflicts with it, `AGENTS.md` wins. If any of these conflict with `build.zig`, `tools/build.sh`, or source, trust the executable source. `tasks/lessons.md` is the session-start checklist; `tasks/todo.md` tracks active work.

## Toolchain
- Pinned to `0.17.0-dev.1398+cb5635714` (`.zigversion`). Use zvm/zigup to select it; the wrapper does **not** switch. Zig 0.16 fails on WDBX/MCP listeners (`std.Io.net.Stream`).
- On macOS: use `./build.sh ...` (Darwin Metal-linking entrypoint). `feat-foundationmodels` needs arm64 macOS + Xcode + macOS 26 SDK (`xcrun swiftc`) — disable with `-Dfeat-foundationmodels=false`.

## Commands
| Command | What it does |
|---------|-------------|
| `./build.sh check` | Primary gate: build, tests, lint, parity, feature-off stubs, CLI smoke |
| `./build.sh full-check` | check + integration + benchmarks + dashboard/agent TUI smoke |
| `./build.sh cli` / `mcp` | Build `zig-out/bin/abi` / `zig-out/bin/abi-mcp` |
| `./build.sh test -Dtest-filter="<pattern>"` | Single test (the `zig build test -- --test-filter` form is silently ignored — use the `-Dtest-filter=` build option) |
| `./build.sh test-cli`/`test-plugins`/`test-contracts`/`test-mcp-contracts`/`test-mcp-server`/`test-integration`/`test-feature-contracts` | Focused suites |
| `./build.sh check-parity` | Verify mod/stub public-decl parity (run after any public API change) |
| `./build.sh lint`/`fix` | Check/apply formatting |
| `npx mint@latest validate` | Docs validation (not in CI) |

## Feature flags
15 flags, all default `true` (see `build.zig` lines 9–23): `feat-ai`, `feat-gpu`, `feat-tui`, `feat-accelerator`, `feat-shader`, `feat-mlir`, `feat-mobile`, `feat-wdbx`, `feat-os-control`, `feat-hash`, `feat-metrics`, `feat-telemetry`, `feat-nn`, `feat-sea`, `feat-foundationmodels`. Each selects a real `mod.zig` vs a `stub.zig` (declared in `src/features/mod.zig`). **Honest stubs** (`available=false`/`native_dispatch=false` in source): `accelerator`, `shaders`, `mlir`, `mobile` — selection/report only, no linked native toolchain.

## Architecture
- Entrypoints: `src/main.zig` (CLI), `src/mcp/main.zig` (MCP server). Public API: `src/root.zig` (`@import("abi")`).
- **MCP module-root isolation**: only the `src/mcp/` handler group (`main.zig`, `handlers.zig`, `ai_tools.zig`, `connector_tools.zig`, `plugin_tools.zig`, `state.zig`) may `@import("abi")`; everything else under `src/` uses relative `.zig` imports only. Don't unify MCP HTTP with WDBX REST — duplication is intentional.
- **Generated**: `src/plugin_registry.zig` is regenerated from `src/plugins/*/abi-plugin.json` at build time — never hand-edit.
- Repo-root `mcp/` is launcher scripts, **not** the Zig implementation.

## Frozen surfaces (contract-tested — don't break)
- **CLI**: 13 commands (`help`, `complete`, `train`, `agent`, `backends`, `plugin`, `auth`, `twilio`, `tui`, `dashboard`, `wdbx`, `scheduler`, `nn`). Do **not** resurrect legacy names (`version`, `doctor`, `features`, `platform`, `connectors`, `search`, `info`, `chat`, `db`, `serve`).
- **MCP**: 12 frozen tools (`ai_run`, `ai_complete`, `ai_learn`, `ai_train`, `wdbx_query`, `scheduler_stats`, `scheduler_info`, `connector_test`, `gpu_status`, `plugin_list`, `wdbx_stats`, `plugin_run`). Stdio JSON-RPC, 64 KB cap, JSON depth 32. Loopback-only HTTP/SSE.

## API & parity rules
- Any public API change → update **both** `mod.zig` and `stub.zig`, then `./build.sh check-parity`. Parity scans column-0 `pub const`/`pub fn` only; struct methods are invisible — reach a tracker via a method (e.g. `Store.getTracker()`), not a signature change.
- 5 contract suites: `surface.zig`, `feature_modules.zig`, `mcp_tools.zig`, `plugin_registry.zig`, `public_docs.zig`.

## Zig 0.17 patterns
- `pub fn main(init: std.process.Init) !void`; `ArrayListUnmanaged(T).empty`; `std.mem.trimEnd`/`splitScalar`/`splitAny`; `foundation.time.unixMs()`.
- Tests: inline `test {}`, end modules with `std.testing.refAllDecls(@This())`.
- No silent `catch {}` in data/inference/persistence paths. `build_options.feat_*` for conditional compilation. Explicit `std.mem.Allocator` (no global).
- **MemoryTracker**: track owned-and-freed scratch as `trackAllocNoTag`/`trackFreeNoTag` pairs; never track escaping buffers (search `results`, completion `response`) at the alloc site. In tests, `getTotalFreed() > 0` proves a balanced transient pair fired.

## Claims discipline
No unproven claims (production FHE/AES/RBAC, multi-host sharding, QPS/latency/accuracy, K8s/H100, external stacks). WDBX demo modules (`compression.zig`, `entropy.zig`, `neural_compress.zig`, `crypto_he.zig`, `fhe.zig`) are reference-grade, not production. Audit: `docs/contracts/external-claims-audit.mdx`; `docs/spec/wdbx-north-star.mdx` §2.

## Commits & CI
- Conventional Commits. Never force-push `main`.
- CI: `zig build check` + `cross-smoke` on macOS. Keep `.zigversion` and `.github/workflows/ci.yml` ZIG_VERSION in sync.

## Linux / non-macOS note
Cross-compiles link cleanly for `x86_64-linux-gnu` / `aarch64-linux-gnu` / `windows-gnu` (exe + all test modules set `link_libc=true`; `metal_shared.zig` gates objc externs to macOS via `comptime`). Ambient WDBX persist EBADF is **fixed** (`ensureOwnerOnlyDir` opens with `iterate=true` so Linux `fchmod` works). Execution of cross binaries still needs a Linux/Windows host (CI `cross-smoke`); this macOS host cannot run them. Green native suites on macOS: full `./build.sh check`. Feature-stub smoke in `check` overwrites `zig-out/bin/abi` — re-run `./build.sh cli` to restore it.
