# AGENTS.md — abi

Canonical instruction file. If this conflicts with `build.zig`, `tools/build.sh`, or source, trust the executable source. Sibling files `CLAUDE.md`/`GEMINI.md` are thin redirects that each point here — update only this file on command/flag/pattern changes. Session-start checklist: `tasks/lessons.md`; active board: `tasks/todo.md`.

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
No unproven claims (production FHE/AES/RBAC, multi-host sharding, QPS/latency/accuracy, K8s/H100, external stacks). WDBX demo modules (`compression.zig`, `entropy.zig`, `ans.zig`, `neural_compress.zig`, `crypto_he.zig`, `fhe.zig`) are reference-grade, not production. Audit: `docs/contracts/external-claims-audit.mdx`; `docs/spec/wdbx-north-star.mdx` §2.

## Commits & CI
- Conventional Commits. Never force-push `main`.
- CI: `zig build check` + `cross-smoke` on macOS. Keep `.zigversion` and `.github/workflows/ci.yml` ZIG_VERSION in sync.

## Linux / non-macOS note
Cross-compiles link cleanly for `x86_64-linux-gnu` / `aarch64-linux-gnu` / `windows-gnu` (exe + all test modules set `link_libc=true`; `metal_shared.zig` gates objc externs to macOS via `comptime`). Ambient WDBX persist EBADF is **fixed** (`ensureOwnerOnlyDir` opens with `iterate=true` so Linux `fchmod` works). Execution of cross binaries still needs a Linux/Windows host (CI `cross-smoke`); this macOS host cannot run them. Green native suites on macOS: full `./build.sh check`. Feature-stub smoke in `check` overwrites `zig-out/bin/abi` — re-run `./build.sh cli` to restore it.

## Learned User Preferences
- Prefer feature branches named with the `cursor/` prefix from `origin/main`; do not commit or push directly to `main`; never force-push `main`.
- Recurring ask: land finished work onto `main` via PR/merge rather than leaving stranded feature branches.
- For AGENTS.md Learned-section-only updates, append prefs/facts onto `origin/main` rather than overwriting the compact toolchain/commands body.
- Prefer draft PRs when the create-pull-request flow requests draft.
- Verify interactive dashboard/TUI with `.agents/skills/run-tui/tui.sh` (tmux pty); never prepend Homebrew `/opt/homebrew/bin` ahead of the pinned Zig on PATH.
- Prefer honest status digests and labeled demos over fake live bridges when IPC or production capability is absent.
- For refactor/organization work, prefer scoped tracks (module extraction vs north-star features vs docs/claims) over open-ended clean-slate rewrites; confirm scope before planning.
- When reducing Cursor context budget, disable unused `alwaysApply` plugin rules and unrelated MCP servers; do not re-inflate `CLAUDE.md`/`GEMINI.md` (they are thin redirects to `AGENTS.md`).
- When the user invokes `/abi`, route ABI implementation through the `abi` subagent.

## Learned Workspace Facts
- Org/extraction waves are largely done; TUI hub is `repl.zig` (~564) with leaves `repl_io`, `repl_complete`, `repl_pane`, `repl_commands`, `repl_git_*`, `repl_session`, `repl_types`; `dashboard.zig` + `dashboard_render.zig` are ~399 each.
- Interactive `abi dashboard` / `abi tui` / `abi --tui` use a split layout (diagnostics + Agent Output); one-shot `--once` stays stacked — layouts diverge by design.
- Dashboard Agent Output is a status digest, not live `agent tui` traffic; dashboard WDBX is an ephemeral CLI probe (labeled), not the durable agent store.
- Plugin-declared slash-commands dispatch via `__cmd__:<name>` (parallel to `__context__:<name>` for context providers).
- REPL `/pane` split landed in `repl_pane.zig`; in-process persona streaming is iterative word/token emission (`stream=incremental` via `incremental.zig`), not a neural LM sampler.
- `tasks/goals.md` is gitignored (`/tasks/*` + root `*.md`); treat committed `tasks/todo.md` as the active board.
- Canonical refactor layout/status: `docs/spec/abi-refactor-design.mdx`; Approach-1 waves A–C complete; `modern-refactor/examples/` is historical, not the active board.
- `modernized/` holds Phase D–approved package-layout pointers under `packages/`; live code remains `src/` until cutover.
- Ambient WDBX Linux `EBADF` owner-only repair is fixed (`iterate=true`); `ABI_WDBX_PERSIST=0` is no longer required to avoid ambient-open panics on Linux.
