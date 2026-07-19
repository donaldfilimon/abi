# AGENTS.md — abi

Canonical instruction file. If this conflicts with `build.zig`, `tools/build.sh`, or source, trust the executable source. Sibling files `CLAUDE.md`/`GEMINI.md` are thin redirects that each point here. Session-start checklist: `tasks/lessons.md`; active board: `tasks/todo.md`.

## Toolchain
- Pinned to `0.17.0-dev.1398+cb5635714` (`.zigversion`). Use zvm/zigup to select it; the wrapper does **not** switch.
- On macOS: use `./build.sh ...` (Darwin Metal-linking entrypoint). `feat-foundationmodels` needs arm64 macOS + Xcode + macOS 26 SDK.

## Commands
| Command | What it does |
|---------|-------------|
| `./build.sh check` | Primary gate: build, tests, lint, parity, feature-off stubs, CLI smoke |
| `./build.sh full-check` | check + integration + benchmarks + dashboard/agent TUI smoke |
| `./build.sh cli` / `mcp` | Build `zig-out/bin/abi` / `zig-out/bin/abi-mcp` |
| `./build.sh test -Dtest-filter="<pattern>"` | Single test (use build option form; `-- --test-filter` is silently ignored) |
| `./build.sh test-{cli,plugins,contracts,mcp-contracts,mcp-server,integration,feature-contracts}` | Focused suites |
| `./build.sh check-parity` | Verify mod/stub public-decl parity (run after any public API change) |
| `./build.sh lint`/`fix` | Check/apply formatting |
| `.agents/skills/docs-validate/validate.sh` | Docs validation (CI `docs (mint validate)`) |

## Architecture
- Entrypoints: `src/main.zig` (CLI), `src/mcp/main.zig` (MCP server). Public API: `src/root.zig` (`@import("abi")`).
- **MCP module-root isolation**: only `src/mcp/` handler group (`main.zig`, `handlers.zig`, `ai_tools.zig`, `connector_tools.zig`, `plugin_tools.zig`, `state.zig`) may `@import("abi")`; everything else under `src/` uses relative `.zig` imports. Don't unify MCP HTTP with WDBX REST — duplication is intentional.
- **Generated**: `src/plugin_registry.zig` is regenerated from `src/plugins/*/abi-plugin.json` at build time — never hand-edit.
- Repo-root `mcp/` is launcher scripts, **not** the Zig implementation.

## Frozen surfaces (contract-tested — don't break)
- **CLI** (13): `help`, `complete`, `train`, `agent`, `backends`, `plugin`, `auth`, `twilio`, `tui`, `dashboard`, `wdbx`, `scheduler`, `nn`. Do **not** resurrect legacy names (`version`, `doctor`, `features`, `platform`, `connectors`, `search`, `info`, `chat`, `db`, `serve`).
- **MCP** (12): `ai_run`, `ai_complete`, `ai_learn`, `ai_train`, `wdbx_query`, `scheduler_stats`, `scheduler_info`, `connector_test`, `gpu_status`, `plugin_list`, `wdbx_stats`, `plugin_run`. Stdio JSON-RPC, 64 KB cap, JSON depth 32. Loopback-only HTTP/SSE.

## API & parity rules
- Any public API change → update **both** `mod.zig` and `stub.zig`, then `./build.sh check-parity`. Parity scans column-0 `pub const`/`pub fn` only; struct methods are invisible.
- 5 contract test files: `tests/contracts/{surface,feature_modules,mcp_tools,plugin_registry,public_docs}.zig`.

## Zig 0.17 essentials
- `pub fn main(init: std.process.Init) !void`; `ArrayListUnmanaged(T).empty`; `std.mem.trimEnd`/`splitScalar`/`splitAny`/`splitSequence`; `foundation.time.unixMs()`.
- Tests: inline `test {}`, end modules with `std.testing.refAllDecls(@This())`.
- No silent `catch {}` in data/inference/persistence paths. `build_options.feat_*` for conditional compilation. Explicit `std.mem.Allocator` (no global).
- **MemoryTracker**: track owned-and-freed scratch as `trackAllocNoTag`/`trackFreeNoTag` pairs; never track escaping buffers at the alloc site. `getTotalFreed() > 0` proves a balanced transient pair fired.

## Claims discipline
No unproven claims (production FHE/AES/RBAC, multi-host sharding, QPS/latency/accuracy, K8s/H100, external stacks). WDBX demo modules (`compression`, `entropy`, `ans`, `neural_compress`, `crypto_he`, `fhe`) are reference-grade, not production. Audit: `docs/contracts/external-claims-audit.mdx`.

## CI & commits
- Conventional Commits. Never force-push `main`. Keep `.zigversion` and `.github/workflows/ci.yml` `ZIG_VERSION` in sync.
- Self-hosted macOS ARM64 runner for same-repo events; fork PRs use `macos-latest`. Same-repo jobs gated by `github.repository == 'donaldfilimon/abi'` check.

- **Ambient WDBX persistence panics on Linux.** `abi complete` (no flags) and `abi-mcp` default to the ambient durable store, whose `durable_store.ensureOwnerOnlyDir` calls `dir.setPermissions` → `fchmod` on an `O_PATH` directory fd, which returns `EBADF` and panics (`programmer bug caused syscall error: BADF`). This is a Zig-std/Linux interaction, not a repo bug (macOS has no `O_PATH`). Run these with `ABI_WDBX_PERSIST=0` (in-memory store) to exercise the full completion/MCP pipeline. The explicit `abi wdbx ...` subcommands use a different open path and work without the flag.
- **`./build.sh check` is not fully green on Linux.** Libc-linked test targets (`test-integration`, `test-cli`, `test-mcp-server`, and the root test) fail to compile with `dependency on libc must be explicitly specified` / `undefined symbol: getsockname`/`getpid` because `build.zig` only implicitly links libc on macOS. Green suites on Linux: `zig build lint`, `check-parity`, `test-plugins`, `test-mcp-contracts`, `test-feature-contracts`, `test-contracts` (public_docs pin assertions), plus the 100+ inline unit tests that don't need libc. Prefer these for verification; run the full `check` gate on macOS.
- **`abi` binary can get overwritten by feature-stub smoke.** `./build.sh check` runs `tools/check_feature_stubs.sh`, which builds `abi` with feature flags disabled and installs it over `zig-out/bin/abi`. Re-run `zig build cli` (or `./build.sh cli`) afterward to restore the full-featured binary (otherwise e.g. `wdbx` shows disabled in `abi backends`).
## Linux / non-macOS note
Cross-compiles link cleanly for `x86_64-linux-gnu` / `aarch64-linux-gnu` / `windows-gnu` (exe + all test modules set `link_libc=true`; `metal_shared.zig` gates objc externs to macOS via `comptime`). Ambient WDBX persist EBADF is **fixed** (`ensureOwnerOnlyDir` opens with `iterate=true` so Linux `fchmod` works). Execution of cross binaries still needs a Linux/Windows host (CI `cross-smoke`); this macOS host cannot run them. Green native suites on macOS: full `./build.sh check`. Feature-stub smoke in `check` overwrites `zig-out/bin/abi` — re-run `./build.sh cli` to restore it.

## Cross-platform
- macOS is primary (full `./build.sh check` green). Linux: `test-integration`, `test-cli`, `test-mcp-server` need libc linking; prefer unit tests, `test-plugins`, `test-{contracts,mcp-contracts,feature-contracts}`.
- Feature-stub smoke in `check` overwrites `zig-out/bin/abi` — re-run `./build.sh cli` to restore full-featured binary.

## Learned preferences
- Feature branches: `cursor/` prefix from `origin/main`. Land via PR merge (`gh pr merge --squash`). Delete merged local `cursor/*` branches. Prefer draft PRs.
- For AGENTS.md Learned-section-only updates, append to `origin/main`.
- Verify interactive TUI with `.agents/skills/run-tui/tui.sh` (tmux pty).
- Prefer honest status digests and labeled demos over fake live bridges.
- For refactor/organization work, prefer scoped tracks over open-ended clean-slate rewrites.
