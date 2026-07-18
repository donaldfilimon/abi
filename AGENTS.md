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
| `.agents/skills/docs-validate/validate.sh` | Docs validation (CI job `docs (mint validate)`) |

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

- Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `chore(build):`, …).
- Local `main` and `origin/main` share history (reconciled at `848ec2c8`; the old no-common-ancestor state is resolved). Never force-push `main`.
- CI (`.github/workflows/ci.yml`) runs `zig build check` + `cross-smoke` on macOS. Same-repo push/PR/workflow_dispatch use the labeled self-hosted runner (`self-hosted,macOS,ARM64,abi`); fork PRs stay on GitHub-hosted `macos-latest` only. Never add self-hosted jobs without the same-repo `if:` gate — see `.github/self-hosted-runner.md`. Keep workflow `ZIG_VERSION` in sync with `.zigversion` when either moves.

## Zig 0.17 Patterns

- `pub fn main(init: std.process.Init) !void`
- `ArrayListUnmanaged(T).empty` (not `.init(allocator)`)
- `std.mem.trimEnd` (not `trimRight`); `splitScalar`/`splitAny`/`splitSequence`
- `foundation.time.unixMs()` for ms timestamps
- Tests: inline `test {}`; end modules with `std.testing.refAllDecls(@This())`
- No silent `catch {}` in persistence/inference/connector/data paths — propagate or log
- Conditional compilation: `build_options.feat_*`
- Pass an explicit `std.mem.Allocator`; no global/hidden allocator
- Naming: `camelCase` functions/vars, `PascalCase` types, `SCREAMING_SNAKE_CASE` constants, `snake_case` enum variants
- **MemoryTracker wiring**: balance transient tracking — track owned-and-freed scratch as `trackAllocNoTag`/`trackFreeNoTag` pairs; never track escaping buffers (search `results`, completion `response`) at the alloc site. Isolate transient from persistent in tests: `getTotalFreed() > 0` proves a balanced transient pair fired (persistent allocs never free until `deinit`). KV `store.store()` is intentionally NOT tracked (AI paths add their own transient pair on top).

## WDBX / GPU / Connectors

- **WDBX**: in-process store + segment/WAL persistence; durable parent dirs `0700` on POSIX. Cluster RPC is real TCP RequestVote/AppendEntries (`ABI_WDBX_CLUSTER_TOKEN` for non-loopback; `ABI_WDBX_CLUSTER_PEERS` allowlist), with same-host loopback/routable-bind coverage — **not** proven production multi-host or sharding. `ABI_REMOTE_COMPUTE_ENDPOINT` is currently reported as configuration metadata; `remote_compute.zig` separately provides a loopback-tested DOT transport, but no production caller wires the env value into dispatch/fallback. Loopback REST supports optional bearer auth (`ABI_WDBX_REST_TOKEN`), token-bucket rate limiting (`ABI_WDBX_RATE_LIMIT_CAPACITY`/`ABI_WDBX_RATE_LIMIT_REFILL`), and TLS env config (`ABI_WDBX_TLS_CERT`/`ABI_WDBX_TLS_KEY` — validated but native TLS not linked; deploy behind a TLS-terminating proxy).
- **WDBX demo/reference modules** in `src/features/wdbx/`: `compression.zig` (lossy int8 quantization), `entropy.zig` (lossless order-0 Huffman — NOT ANS/arithmetic), `neural_compress.zig` (in-process autoencoder — NOT SOTA), `crypto_he.zig` (additive single-key HE — NOT multi-key/FHE), `fhe.zig` (DGHV somewhat-homomorphic, reference parameters — NOT security-audited). All demo/reference-grade. Exercised via `abi wdbx secure demo`.
- **GPU**: Metal framework linked at build time on macOS; `accelerated=false` is the normal state until `g_metal_context.init()` succeeds at runtime. Mid-run GPU failure gracefully degrades to CPU. ANE execution is a disclosed non-goal (100% Zig constraint; requires CoreML/ObjC).
- **Live remote connectors** require explicit credentials + `.live` transport + `https://` base URL. POSIX `auth signin` no-echo; credentials stored as plaintext JSON under `~/.abi/` (or `ABI_CREDENTIALS_PATH`/`XDG_CONFIG_HOME`), dir `0700` + file `0600` owner-only. Windows ACL/keychain + zeroing remain disclosed gaps.
- **Local inference bridge** (`local_bridge`): uses `.live` transport to a loopback `http://127.0.0.1` server with no credentials (exempted from `https://` by loopback carve-out). `endpointFor` accepts optional override; `handleLocalBridgeComplete` (in `src/cli/handlers/train.zig`) checks `ABI_LLAMA_CPP_ENDPOINT` / `ABI_MLX_ENDPOINT` env vars (defaults `http://127.0.0.1:8080` / `http://127.0.0.1:8081`). Falls back to in-process persona router if the local server is unreachable.
- Discord/Twilio logs are redacted (metadata/byte counts only, never message/response bodies).

## AI Subsystem

- **SEA loop**: evidence-augmented self-learning completion with 8-signal scorer + budgeted greedy selection. Persists `AdaptiveModulator` weights (EMA, `alpha=0.3`, key `modulator:weights`) in WDBX. EMA weights load+save **only** on the `--learn`/SEA path; plain `complete` re-runs sentiment analysis each turn with no EMA persistence. Query planning classifies 7 task types; `code_repair`, `project_recall`, and `benchmark_review` apply explicit weight deltas, while the other four currently retain baseline weights.
- **Constitution audit** (6 principles: truthfulness, safety, helpfulness, fairness, privacy, transparency): **observability-only, not a gate** — sets `audit_passed`/`audit_vetoed`/`escore` in metadata and `std.log.warn`s on violation, but `complete`/`run` still return the response. Safety+privacy form a safety class with a hard veto if either scores < 0.5. Checks use case-insensitive **substring** (infix) matching — "harm" fires on "harmless"; cannot detect novel harm patterns, only the 7 hardcoded negative substrings.
- **Router**: `analyzeSentiment` uses **prefix-only** single-token keyword matching (`startsWithIgnoreCase`); `selectBestProfile` ties resolve `abbey > aviva > abi`, while neutral input is not tied (`0.33/0.33/0.34`) and routes to `abi`. The former `routeInputAdaptive` wrapper is absent; the live EMA path is `completeAdaptive`/`completeWithStoreAdaptive` via `runLearnLoop`.
- **Connector validation**: Discord validates numeric snowflake IDs + non-empty printable credentials + ≤2000-byte messages; Twilio validates `AC`+32-hex SIDs, 32-hex tokens, non-empty base URL, non-zero timeout, explicit `.live` transport.
- **Streaming + file context**: `CompletionRequest`/`LearnLoopConfig` accept `stream_callback`/`stream_ctx` for post-hoc chunked output (~4-byte splits; true per-token streaming requires a chunked model backend). `file_context.zig` resolves `@file` mentions (sandboxed to cwd, rejects `..`/absolute/symlink escape) with an 8 KB budget; wired into `agent plan`, `agent multi`, and the `agent tui` REPL.

## Claims & Docs

No unproven claims (distributed sharding, production FHE/AES/RBAC, non-loopback hardening, K8s/H100, Swift/Python/TF stacks, QPS/latency/accuracy/energy/certifications). Public wording: `docs/contracts/external-claims-audit.mdx`. Per-module demo-vs-production boundary: `docs/spec/wdbx-north-star.mdx` §2. Security: `abi-threat-model.md`.

## OpenCode Setup

`opencode.json` auto-loaded (instructions: `AGENTS.md`, `tasks/lessons.md`, `tasks/todo.md`). `.opencode/skills/` is a symlink to `.agents/skills/`. MCP servers: `abi-mcp`, `skill-loop`. Sync canonical skills: `.agents/skills/sync-clis/launch.sh`. Modern-refactor skills (codebase-analysis etc.) in `.agents/skills/` — use for refactors; follow with `./build.sh check`. Superpower + operational skills are enumerated in the system prompt's `available_skills`; don't maintain a parallel list here.

## Cursor Cloud specific instructions

Linux x86_64 VM with the pinned Zig (`.zigversion`) already installed at `/opt/zig` and symlinked to `/usr/local/bin/zig` (the startup update script re-installs it only if missing/mismatched). Build/run/test commands are the standard ones documented under **Commands** above (`./build.sh cli`, `./build.sh mcp`, `zig build lint`, focused `zig build test-*` suites). This repo is macOS-first (CI is macOS-only; GPU is Metal), so a few surfaces behave differently on this Linux VM:

- **Ambient WDBX persistence panics on Linux.** `abi complete` (no flags) and `abi-mcp` default to the ambient durable store, whose `durable_store.ensureOwnerOnlyDir` calls `dir.setPermissions` → `fchmod` on an `O_PATH` directory fd, which returns `EBADF` and panics (`programmer bug caused syscall error: BADF`). This is a Zig-std/Linux interaction, not a repo bug (macOS has no `O_PATH`). Run these with `ABI_WDBX_PERSIST=0` (in-memory store) to exercise the full completion/MCP pipeline. The explicit `abi wdbx ...` subcommands use a different open path and work without the flag.
- **`./build.sh check` is not fully green on Linux.** Libc-linked test targets (`test-integration`, `test-cli`, `test-mcp-server`, and the root test) fail to compile with `dependency on libc must be explicitly specified` / `undefined symbol: getsockname`/`getpid` because `build.zig` only implicitly links libc on macOS. Green suites on Linux: `zig build lint`, `check-parity`, `test-plugins`, `test-mcp-contracts`, `test-feature-contracts`, `test-contracts` (public_docs pin assertions), plus the 100+ inline unit tests that don't need libc. Prefer these for verification; run the full `check` gate on macOS.
- **`abi` binary can get overwritten by feature-stub smoke.** `./build.sh check` runs `tools/check_feature_stubs.sh`, which builds `abi` with feature flags disabled and installs it over `zig-out/bin/abi`. Re-run `zig build cli` (or `./build.sh cli`) afterward to restore the full-featured binary (otherwise e.g. `wdbx` shows disabled in `abi backends`).
## Linux / non-macOS note
Cross-compiles link cleanly for `x86_64-linux-gnu` / `aarch64-linux-gnu` / `windows-gnu` (exe + all test modules set `link_libc=true`; `metal_shared.zig` gates objc externs to macOS via `comptime`). Ambient WDBX persist EBADF is **fixed** (`ensureOwnerOnlyDir` opens with `iterate=true` so Linux `fchmod` works). Execution of cross binaries still needs a Linux/Windows host (CI `cross-smoke`); this macOS host cannot run them. Green native suites on macOS: full `./build.sh check`. Feature-stub smoke in `check` overwrites `zig-out/bin/abi` — re-run `./build.sh cli` to restore it.

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

## Learned Workspace Facts
- Org/extraction waves are largely done; TUI hub is `repl.zig` (~564) with leaves `repl_io`, `repl_complete`, `repl_pane`, `repl_commands`, `repl_git_*`, `repl_session`, `repl_types`; `dashboard.zig` + `dashboard_render.zig` are ~399 each.
- Interactive `abi dashboard` / `abi tui` / `abi --tui` use a split layout (diagnostics + Agent Output); one-shot `--once` stays stacked — layouts diverge by design. Dashboard Agent Output is a status digest, not live `agent tui` traffic; dashboard WDBX is an ephemeral CLI probe (labeled), not the durable agent store.
- Plugin-declared slash-commands dispatch via `__cmd__:<name>` (parallel to `__context__:<name>` for context providers).
- REPL `/pane` split landed in `repl_pane.zig`; in-process persona streaming is iterative word/token emission (`stream=incremental` via `incremental.zig`), not a neural LM/ggml sampler.
- WDBX `SearchResult`/`RankedNode` can attach borrowed vector dims for zero-copy search/CLI use; mutation lifetime remains a documented residual.
- Metal `vectorOps` includes a fused cosine kernel on macOS; CUDA/Vulkan stay disclosed stubs (no native dispatch claimed).
- Windows credential writes can apply owner-only DACL (SDDL `OW`); OS keychain and Windows runtime ACL verification still need a Windows host/CI.
- `tasks/goals.md` is gitignored (`/tasks/*` + root `*.md`); treat committed `tasks/todo.md` as the active board (includes A–G claim-honest scoreboard).
- Canonical refactor layout/status: `docs/spec/abi-refactor-design.mdx`; Approach-1 waves A–C complete; `modern-refactor/examples/` is historical, not the active board. `modernized/` holds Phase D–approved package-layout pointers under `packages/`; live code remains `src/` until cutover. Optional host override template: `modern-refactor/.claude/modern-refactor.local.md.example` → copy to repo-root `.claude/modern-refactor.local.md` (not auto-loaded from the plugin package).
- `tools/check_zigversion.sh` runs as part of `zig build check` / `./build.sh check` and fails when `.zigversion` and `.github/workflows/ci.yml` `ZIG_VERSION` diverge (also warns if active `zig version` ≠ pin).
- `foundation/http.zig` holds shared HTTP helpers (read/write/find body, Content-Length, header parse, bearer auth, readHttpRequest/HttpReadResult) used by both MCP and WDBX REST; `foundation/json.zig` has `appendJsonString`/`escapeJsonString` used by MCP. Keep connector/WDBX inline JSON copies: `connector_test_mod` is an isolated test root (no `abi`/foundation import), so sharing needs a `foundation_json` leaf module in `build.zig` — prefer keep-copies over unfinished dedup.
- abi-skills/`sl` `skill-loop` is the external npm CLI `@stylusnexus/skill-loop-cli` (pin `@0.3.3` via `npx`), not an in-repo binary; useful cmds: `init`/`status`/`inspect`/`log` (no `scan`). When absent, use manual skill scan + `.agents/skills/sync-clis/launch.sh` (propagates SKILL.md and references/examples to Claude/grok targets).
