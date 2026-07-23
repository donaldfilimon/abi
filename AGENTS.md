# AGENTS.md — abi

Canonical instruction file. If this conflicts with `build.zig`, `tools/build.sh`, or source, trust the executable source. Sibling files `CLAUDE.md`/`GEMINI.md` are thin redirects here. Session-start checklist: `tasks/lessons.md`; active board: `tasks/todo.md`.

## Toolchain
- Pinned to `0.17.0-dev.1442+972627084` (`.zigversion`). Use zvm/zigup; the wrapper does **not** switch. `tools/check_zigversion.sh` (wired into `check`) fails on drift vs `.github/workflows/ci.yml`.
- On macOS: **use `./build.sh ...`** (Darwin Metal-linking entrypoint). `feat-foundationmodels` needs arm64 macOS + Xcode + macOS 26 SDK.

## Commands
| Command | What it does |
|---------|-------------|
| `./build.sh check` | Primary gate: build, tests, lint, parity, feature-off stubs (overwrites `zig-out/bin/abi`), CLI smoke. Re-run `./build.sh cli` afterward to restore full binary. |
| `./build.sh full-check` | check + integration + benchmarks + dashboard/agent TUI smoke |
| `./build.sh cli` / `mcp` | Build `zig-out/bin/abi` / `zig-out/bin/abi-mcp` |
| `./build.sh test -Dtest-filter="<pattern>"` | Single test (use build option; `-- --test-filter` is silently ignored) |
| `./build.sh test-{cli,plugins,contracts,mcp-*,integration,feature-contracts}` | Focused suites |
| `./build.sh check-parity` | Verify mod/stub public-decl parity (run after any public API change) |
| `./build.sh lint`/`fix` | Check/apply formatting |
| `.agents/skills/docs-validate/validate.sh` | Docs validation (CI `docs (mint validate)`) |

## Architecture (non-obvious from names)
- Entrypoints: `src/main.zig` (CLI), `src/mcp/main.zig` (MCP server). Public API: `src/root.zig` (`@import("abi")`).
- **MCP module-root isolation**: `src/mcp/` is its own Zig module (depends on `abi` only via named import). mcp files reach `foundation.*` (http/env/json) via `@import("abi").foundation.*` only. No `src/mcp/*.zig` reaches `abi.features`/`ai`/`wdbx` internals; nothing outside imports MCP internals. MCP↔WDBX REST duplication is intentional.
- **Generated**: `src/plugin_registry.zig` is regenerated from `src/plugins/*/abi-plugin.json` at build time — never hand-edit. (See `tools/generate_plugin_registry.zig` and build.zig.)
- Repo-root `mcp/` holds only launcher scripts (`launcher.sh` wired by `opencode.json`).

## Frozen surfaces (contract-tested — don't break)
- **CLI (13)**: `help`, `complete`, `train`, `agent`, `backends`, `plugin`, `auth`, `twilio`, `tui`, `dashboard`, `wdbx`, `scheduler`, `nn`. Do **not** resurrect legacy names.
- **MCP (12)**: `ai_run`, `ai_complete`, `ai_learn`, `ai_train`, `wdbx_query`, `scheduler_stats`, `scheduler_info`, `connector_test`, `gpu_status`, `plugin_list`, `wdbx_stats`, `plugin_run`. Stdio JSON-RPC (64 KB cap, JSON depth 32). Loopback-only HTTP/SSE.

## API & parity rules
- Any public API change → update **both** `mod.zig` and `stub.zig`, then `./build.sh check-parity`. Parity scans column-0 `pub const`/`pub fn` only; struct methods are invisible.
- 5 contract test files: `tests/contracts/{surface,feature_modules,mcp_tools,plugin_registry,public_docs}.zig`.

## Zig 0.17 essentials (easy to miss)
- `pub fn main(init: std.process.Init) !void`; `ArrayListUnmanaged(T).empty`; `std.mem.trimEnd`/`splitScalar`/`splitAny`/`splitSequence`; `foundation.time.unixMs()`.
- Tests: inline `test {}`, end modules with `std.testing.refAllDecls(@This())`.
- No silent `catch {}` in data/inference/persistence paths. `build_options.feat_*` for conditional compilation. Explicit `std.mem.Allocator` (no global).
- **MemoryTracker**: track owned-and-freed scratch as `trackAllocNoTag`/`trackFreeNoTag` pairs; never track escaping buffers at the alloc site. `getTotalFreed() > 0` proves a balanced transient pair fired.

## Claims discipline
No unproven claims (production FHE/AES/RBAC, multi-host sharding, QPS/latency/accuracy, K8s/H100, external stacks). WDBX demo modules are reference-grade, not production. Audit: `docs/contracts/external-claims-audit.mdx`.

## CI & commits
- Conventional Commits. **Never force-push `main`**. Keep `.zigversion` and CI `ZIG_VERSION` in sync.
- Self-hosted macOS ARM64 runner for same-repo events; fork PRs use `macos-latest`. Same-repo jobs gated by `github.repository == 'donaldfilimon/abi'`.
- Feature-stub smoke in `check` overwrites the `abi` binary — re-run `./build.sh cli` to restore.

## Linux / non-macOS note
Cross-compiles link cleanly (use `tools/cross_smoke.sh`). Execution of cross binaries needs a Linux/Windows host. Green native suites on macOS: full `./build.sh check`.

Gitignore gotcha: global `*.md` allowlist in effect — every new tracked markdown (e.g. `examples/*/README.md`) needs an explicit `!` entry or it is silently untracked.

## Learned User Preferences
- Prefer feature branches `cursor/*` from `origin/main`; do not commit/push directly to `main`; never force-push.
- Land finished work via PR/merge (prefer `gh pr merge --squash`); return to main; remove merged `cursor/*` branches after.
- AGENTS.md learned-only updates: append prefs onto `origin/main`.
- Prefer draft PRs when flow requests.
- Verify interactive TUI with `.agents/skills/run-tui/tui.sh` (tmux pty); never prepend Homebrew zig ahead of pinned.
- Honest status/demos only — never fake-complete honest stubs, ANE, audited FHE, SOTA compression, or prod multi-host sharding.
- Refactors: prefer scoped tracks over open-ended rewrites.
- Skills/docs: full repo-relative paths in `.agents/` etc.; sync `.agents` → `.claude` via `.agents/skills/sync-clis/launch.sh`.
- `/abi` → route implementation through the `abi` subagent.

## Learned Workspace Facts
- Live code in `src/`; `modernized/` is scaffold only — `tools/check_modernized_refs.sh` (in full-check) rejects stale `src/...` pointers inside it.
- `tasks/goals.md` gitignored; use committed `tasks/todo.md` as active board.
- Interactive `abi tui|dashboard|--tui` uses split layout; `--once` is stacked (different by design). Dashboard is digest only.
- Plugin slash-commands dispatch via `__cmd__:<name>` (parallel to `__context__:<name>`).
- WDBX borrowed vectors (SearchResult/RankedNode) are zero-copy; lifetime ends on next mutation.
- `sl` skill-loop is external npm `@stylusnexus/skill-loop-cli@0.3.3` (npx); use manual + sync-clis if missing.
- Metal details, shared foundation/http+json, and similar are reference only in Metal/CUDA stubs.

## Common Pitfalls
1. Circular imports: `@import("abi")` **only** from MCP executable + handler group (`src/mcp/main.zig` + handlers group). Use relative everywhere else in `src/`.
2. Path imports must include `.zig` extension.
3. Empty `catch {}` forbidden in data/inference/persistence.
4. `ArrayListUnmanaged(T).empty` (not `.init(allocator)`).
5. Deprecated: `trimRight`→`trimEnd`; `split`→`splitScalar`/`splitAny`/`splitSequence`.
6. Timestamps: `foundation.time.unixMs()`, never `std.time.milliTimestamp`.
7. Public API change: always touch **both** mod.zig + stub.zig.
8. macOS builds: prefer `./build.sh` (wrapper + Metal).
9. Every module test block: `std.testing.refAllDecls(@This())`.
10. Feature flags: `build_options.feat_*` at compile time (not runtime checks).

After any edit: `./build.sh check`.
