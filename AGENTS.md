# AGENTS.md — abi

Canonical instruction file. If this conflicts with `build.zig`, `tools/build.sh`, or source, trust the executable source. Sibling files `CLAUDE.md`/`GEMINI.md` are thin redirects that each point here. Session-start checklist: `tasks/lessons.md`; active board: `tasks/todo.md`.

## Toolchain
- Pinned to `0.17.0-dev.1442+972627084` (`.zigversion`). Use zvm/zigup to select it; the wrapper does **not** switch.
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
- **MCP module-root isolation**: `src/mcp/` is its own Zig module (`build.zig` compiles it as a separate `Module`, depending on `abi_mod` only via the named `"abi"` import) — `@import("../foundation/...")`-style relative imports can't reach across that module boundary, so any `src/mcp/*.zig` file that needs a `foundation.*` leaf (`http`, `env`, `json` — currently `protocol.zig`, `http_parse.zig`, `http_transport.zig`, `json_helpers.zig`, plus the core handler group `main.zig`/`handlers.zig`/`ai_tools.zig`/`connector_tools.zig`/`plugin_tools.zig`/`state.zig`) reaches it via `@import("abi").foundation.*`. The isolation property that matters is narrower than "which files import abi": no `src/mcp/*.zig` file reaches into `abi.features`/`ai`/`wdbx` internals, and nothing outside `src/mcp/` imports MCP internals. Everything else under `src/` (non-mcp) uses relative `.zig` imports. Don't unify MCP HTTP with WDBX REST — duplication is intentional.
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

- **`abi` binary can get overwritten by feature-stub smoke.** `./build.sh check` runs `tools/check_feature_stubs.sh`, which builds `abi` with feature flags disabled and installs it over `zig-out/bin/abi`. Re-run `zig build cli` (or `./build.sh cli`) afterward to restore the full-featured binary (otherwise e.g. `wdbx` shows disabled in `abi backends`).

## Linux / non-macOS note
Cross-compiles link cleanly for `x86_64-linux-gnu` / `aarch64-linux-gnu` / `windows-gnu` (exe + all test modules set `link_libc=true`; `metal_shared.zig` gates objc externs to macOS via `comptime`). Ambient WDBX persist EBADF is **fixed** (`ensureOwnerOnlyDir` opens with `iterate=true` so Linux `fchmod` works). Execution of cross binaries still needs a Linux/Windows host (CI `cross-smoke`); this macOS host cannot run them. Green native suites on macOS: full `./build.sh check`. Feature-stub smoke in `check` overwrites `zig-out/bin/abi` — re-run `./build.sh cli` to restore it. `tools/cross_smoke.sh` covers both the CLI and the WDBX 3D-hybrid example (`example-3d-hybrid` compile-only step) per target.

Gitignore gotcha: a global `*.md` markdown allowlist is in effect — every new `examples/*/README.md` (and any other tracked-markdown location) needs an explicit `!` entry in `.gitignore` or it is silently untracked.

## Learned User Preferences
- Prefer feature branches named with the `cursor/` prefix from `origin/main`; do not commit or push directly to `main`; never force-push `main`.
- Recurring ask: land finished work onto `main` via PR/merge rather than leaving stranded feature branches; when asked to "stay on main" / "merge all into main", finish via PR merge (prefer `gh pr merge --squash` when a PR exists) and return checkout to `main`; when asked to delete other branches after landing, remove merged local `cursor/*` branches (and remotes when safe).
- For AGENTS.md Learned-section-only updates, append prefs/facts onto `origin/main` rather than overwriting the compact toolchain/commands body.
- Prefer draft PRs when the create-pull-request flow requests draft.
- Verify interactive dashboard/TUI with `.agents/skills/run-tui/tui.sh` (tmux pty); never prepend Homebrew `/opt/homebrew/bin` ahead of the pinned Zig on PATH.
- Prefer honest status digests and labeled demos over fake live bridges; when asked to "do all" on deferred/non-goal tracks, ship maximum claim-honest scope only — never fake-complete honest stubs, ANE dispatch, audited FHE, SOTA compression, or production multi-host sharding.
- For refactor/organization work, prefer scoped tracks (module extraction vs north-star features vs docs/claims) over open-ended clean-slate rewrites; confirm scope before planning.
- When reducing Cursor context budget, disable unused `alwaysApply` plugin rules and unrelated MCP servers; do not re-inflate `CLAUDE.md`/`GEMINI.md` (they are thin redirects to `AGENTS.md`).
- When the user invokes `/abi`, route ABI implementation through the `abi` subagent.
- Prefer full repo-relative paths in in-repo skill/agent docs (`.agents/`, `.claude/agents/`) so skill-loop inspect false positives stay low; sync `.agents` → `.claude` via `.agents/skills/sync-clis/launch.sh` after path fixes.

## Learned Workspace Facts
- Org/extraction waves are largely done; TUI hub is `repl.zig` (~564) with leaves `repl_io`, `repl_complete`, `repl_pane`, `repl_commands`, `repl_git_*`, `repl_session`, `repl_types`; `dashboard.zig` + `dashboard_render.zig` are ~399 each.
- Interactive `abi dashboard` / `abi tui` / `abi --tui` use a split layout (diagnostics + Agent Output); one-shot `--once` stays stacked — layouts diverge by design. Dashboard Agent Output is a status digest, not live `agent tui` traffic; dashboard WDBX is an ephemeral CLI probe (labeled), not the durable agent store.
- Plugin-declared slash-commands dispatch via `__cmd__:<name>` (parallel to `__context__:<name>` for context providers).
- REPL `/pane` split landed in `repl_pane.zig`; in-process persona streaming is iterative word/token emission (`stream=incremental` via `incremental.zig`), not a neural LM/ggml sampler.
- WDBX `SearchResult`/`RankedNode` can attach borrowed vector dims for zero-copy search/CLI use; mutation lifetime remains a documented residual.
- Metal `vectorOps` on macOS: fused cosine/dot/L2, multi-pass `reduce_sum_kernel`, and demo-grade two-pass softmax (`softmax_kernel` + `softmax_norm_kernel`, host-side max/partition-sum — not a perf claim); CUDA/Vulkan stay disclosed stubs (no native dispatch / ANE claimed).
- Same-repo CI `docs-validate` (mint) runs on the self-hosted macOS ARM64 runner; fork PRs use hosted `docs-validate-hosted` on `ubuntu-latest` (untrusted heads never touch self-hosted; hosted runners can be blocked by GitHub billing lock). Windows credential writes can apply owner-only DACL (SDDL `OW`); OS keychain / Windows ACL verification still need a Windows host/CI.
- `tasks/goals.md` is gitignored (`/tasks/*` + root `*.md`); treat committed `tasks/todo.md` as the active board (includes A–G claim-honest scoreboard).
- Canonical refactor layout/status: `docs/spec/abi-refactor-design.mdx`; Approach-1 waves A–C complete; `modern-refactor/examples/` is historical, not the active board. `modernized/` holds Phase D–approved package-layout pointers under `packages/`; live code remains `src/` until cutover. `tools/check_modernized_refs.sh` (bash 3.2 portable) scans `modernized/` Markdown for stale fenced `src/...` paths and is wired into `zig build full-check` / `./build.sh full-check` (no-op if scaffold absent; not a second build root). Optional host override template: `modern-refactor/.claude/modern-refactor.local.md.example` → copy to repo-root `.claude/modern-refactor.local.md` (not auto-loaded from the plugin package).
- `tools/check_zigversion.sh` runs as part of `zig build check` / `./build.sh check` and fails when `.zigversion` and `.github/workflows/ci.yml` `ZIG_VERSION` diverge (also warns if active `zig version` ≠ pin).
- `foundation/http.zig` holds shared HTTP helpers (read/write/find body, Content-Length, header parse, bearer auth, readHttpRequest/HttpReadResult) used by both MCP and WDBX REST; `foundation/json.zig` has `appendJsonString`/`escapeJsonString` used by MCP. Keep connector/WDBX inline JSON copies: `connector_test_mod` is an isolated test root (no `abi`/foundation import), so sharing needs a `foundation_json` leaf module in `build.zig` — prefer keep-copies over unfinished dedup.
- abi-skills/`sl` `skill-loop` is the external npm CLI `@stylusnexus/skill-loop-cli` (pin `@0.3.3` via `npx`), not an in-repo binary; useful cmds: `init`/`status`/`inspect`/`log` (no `scan`). When absent, use manual skill scan + `.agents/skills/sync-clis/launch.sh` (propagates SKILL.md and references/examples to Claude/grok targets).
