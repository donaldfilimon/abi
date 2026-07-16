# AGENTS.md — abi

Canonical instruction file for this repo. Trust executable config over prose: when this file conflicts with `build.zig`, `tools/build.sh`, or source, trust the executable source. `tasks/lessons.md` is the session-start checklist; `tasks/todo.md` tracks active work; `tasks/goals.md` holds long-horizon direction.

Three sibling instruction files share repo conventions — `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`. When commands, contracts, feature flags, or Zig patterns change, update all three. Use the `instruction-sync` agent (read-only) to verify a convention change propagated to all three before landing.

## Toolchain

Pinned by `.zigversion` to `0.17.0-dev.1398+cb5635714`. `build.zig.zon` `minimum_zig_version` is `0.17.0-dev.1252+e4b325c19` (a separate lower bound — the older PATH zig still compiles). `build.sh`/`tools/build.sh` invoke whatever `zig` is on PATH — they do **not** switch. Zig 0.16 fails on WDBX/MCP listeners (uses `std.Io.net.Stream`). Use zvm/zigup to select the pin before building (old nightlies may need `zvm install`, not just `zvm use`).

On macOS: `./build.sh ...` for the documented Metal-linking workflow.

## Commands

| Command | What it does |
|---------|-------------|
| `./build.sh check` | Primary gate: build, tests, lint, parity, feature-off stubs, CLI smoke |
| `./build.sh full-check` | check + integration + benchmarks + dashboard/agent TUI smoke |
| `./build.sh cli` | Build `zig-out/bin/abi` |
| `./build.sh mcp` | Build `zig-out/bin/abi-mcp` |
| `./build.sh test -Dtest-filter="<pattern>"` | Single test on macOS (wrapper passes args through) |
| `zig build test -Dtest-filter="<pattern>"` | Single test without wrapper (post-`--` form silently ignored) |
| `zig build test-cli` / `test-plugins` / `test-contracts` / `test-mcp-contracts` / `test-mcp-server` / `test-integration` / `test-feature-contracts` | Focused test suites |
| `zig build benchmarks` | Benchmark suite |
| `zig build lint` / `fix` | Check/apply formatting |
| `zig build check-parity` | Verify mod/stub public declaration-name parity |
| `zig build cross-smoke` | Opt-in cross-compile (Linux/Windows/macOS; slow) |
| `npx mint@latest validate` | Docs site validation (not in CI) |

## Feature Flags

15 flags, all default `true`: `feat-ai`, `feat-gpu`, `feat-tui`, `feat-accelerator`, `feat-shader`, `feat-mlir`, `feat-mobile`, `feat-wdbx`, `feat-os-control`, `feat-hash`, `feat-metrics`, `feat-telemetry`, `feat-nn`, `feat-sea`, `feat-foundationmodels`. Each selects between a real `mod.zig` and a disabled `stub.zig` (declared in `src/features/mod.zig`).

- **`feat-foundationmodels`**: comptime-gated to arm64 macOS. Requires Xcode + macOS 26 SDK for `xcrun swiftc`. Use `-Dfeat-foundationmodels=false` to skip.
- **`feat-os-control`**: OS command policy gate; `os_control/stub.zig` hard-denies execution when off.
- **Honest stubs** (all `available=false` / `native_dispatch=false` in source): `accelerator` (selection report + CPU SIMD fallback only), `shaders` (validate + checksum only, no compiler), `mlir` (textual lower only, no LLVM toolchain), `mobile` (profile report only, no runtime). Trust the `available`/`native_dispatch` flags in each `src/features/*/mod.zig` over any prose; absent flags mean not proven.

## Architecture

Layered modular codebase. The executable config (`build.zig`, `tools/build.sh`) owns linking and feature selection; trust it over prose.

| Layer | Path | Role |
|-------|------|------|
| Public API | `src/root.zig` | Exposes the `abi` module to consumers (`@import("abi")`). |
| CLI | `src/main.zig`, `src/cli/` | Arg parsing, sub-command dispatch, and handlers. Entry: `pub fn main(init: std.process.Init) !void`. |
| MCP server | `src/mcp/main.zig` + `handlers.zig` group (`handlers.zig`, `ai_tools.zig`, `connector_tools.zig`, `plugin_tools.zig`, `state.zig`) | JSON-RPC 2.0 over stdio + optional loopback HTTP/SSE. `src/mcp/middleware.zig` runs declarative argument validation on every `tools/call` before dispatch. `src/mcp/protocol.zig` holds `MAX_REQUEST_SIZE` (64 KB) + `MAX_JSON_DEPTH` (32). |
| Feature selection | `src/features/mod.zig` | Each `-Dfeat-*` flag selects between a real `mod.zig` and a disabled `stub.zig`. |
| AI | `src/features/ai/` | Profiles, router, constitution, training, and model catalog (`models.zig`). Default model: `claude-fable-5`. |
| Vector store | `src/features/wdbx/` | In-memory KV + vector storage, HNSW index, MVCC snapshots, WAL/segment checkpoints. Demo/reference compression + crypto modules (see WDBX section). |
| GPU | `src/features/gpu/` | Runtime capability report; Metal dispatch on macOS (pure-Zig objc FFI; activates on init) with deterministic CPU SIMD fallback until native kernels initialize. No `-Dgpu-backend` option. |
| Connectors | `src/connectors/` | Local/live adapters: openai, anthropic, grok, discord, twilio, fm, http, json, local_bridge. |
| OS control | `src/features/os_control/` | Safe OS command policy (dry-run/execute gate, allow-list, workspace containment). |
| Plugins | `src/plugins/`, `src/plugin_registry.zig` | Manifest validation + generated metadata registry. Plugin manifests (`abi-plugin.json`) can declare a `commands` array (`name`, `summary`, `aliases`) that registers slash-commands in `agent tui`, and a `context_providers` array (`name`, `summary`) whose snippets are injected into the REPL prompt context via `__context__:<name>` dispatch. |
| Core/Foundation | `src/core/`, `src/foundation/` | Scheduler, registry, memory, time, sync, logger, IO, credentials, OS abstractions. |

Repo-root `mcp/` holds launcher scripts and `.mcp.json` host wiring — it is **not** the Zig MCP implementation.

- **Generated**: `src/plugin_registry.zig` — never hand-edit. Regenerated from `src/plugins/*/abi-plugin.json` at build time.

## Import Rules

Inside `src/`: relative `.zig` imports only. **Only** the MCP handler group (`src/mcp/main.zig`, `handlers.zig`, `ai_tools.zig`, `connector_tools.zig`, `plugin_tools.zig`, `state.zig`) may `@import("abi")`.

## CLI Surface (frozen, contract-tested)

13 commands: `help`, `complete`, `train`, `agent`, `backends`, `plugin`, `auth`, `twilio`, `tui`, `dashboard`, `wdbx`, `scheduler`, `nn`. Full specs in `src/cli/usage.zig`.
- `help --json` / `--completion <bash|zsh|fish>`
- `complete` supports `--live`, `--model`, `--confirm` (apple-fm), `--learn` (SEA), `--stream`, `--soul <file.json>` + `--soul-alpha <0.0-1.0>` (SoulLayout neural-routing blend)
- `agent` supports `plan`, `train`, `tui`, `multi`, `spawn`, `browser`, `os`:
  - `browser` is reviewed local planning only and never embeds or launches a browser.
  - `os` runs commands through the `os_control` policy gate: `agent os dry-run <cmd>` only renders a plan (no side effects); `agent os execute --confirm <cmd>` is the only execution path, requires the literal `--confirm` token, and is restricted to a read-only allow-list (`true`/`pwd`/`ls`/`whoami`/`date`) plus workspace path-containment. Shells (`sh`/`bash`/`zsh`/`fish`) and a destructive deny-list are blocked.
  - `agent tui` REPL slash-commands include `/open <path>` (load file into context), `/diff` (git diff), `/commit` (git commit), `/context` (show context state), `/features` (show build-time feature flags), `/learn` (toggle SEA self-learning mode), `/save <name>` / `/load <name>` (session save/restore), `/sessions` (list saved sessions), `/clear` (clear screen), and plugin-provided commands declared via `abi-plugin.json` `commands` field.
- `tui`/`dashboard` flags: `--help` documents them. `abi --tui` is a shortcut.
- `scheduler status` runs a one-shot self-terminating probe task and reports counters + attached MemoryTracker stats + always-on telemetry block. Probe is a no-op, so memory counters read 0 by design.
- Malformed numeric args → usage + exit 2
- **Do not** resurrect legacy names: `version`, `doctor`, `features`, `platform`, `connectors`, `search`, `info`, `chat`, `db`, `serve`

## MCP Surface (12 tools)

`ai_run`, `ai_complete`, `ai_learn`, `ai_train`, `wdbx_query`, `scheduler_stats`, `scheduler_info`, `connector_test`, `gpu_status`, `plugin_list`, `wdbx_stats`, `plugin_run` (order matches source `handlers.zig`). JSON-RPC 2.0 over stdio (64 KB cap via `protocol.MAX_REQUEST_SIZE` + `protocol.MAX_JSON_DEPTH=32` nesting bound; per-field 16 KB cap in middleware). Optional HTTP/SSE on `127.0.0.1:8080` (`ABI_MCP_HTTP_PORT`, `ABI_MCP_HTTP_TOKEN`); loopback-only. `ai_train` paths confined under cwd or `ABI_TRAIN_DATA_ROOT`.

- Stdio exits on stdin EOF/read failure (not a long-lived daemon). `shutdown` RPC only signals; scheduler/store teardown deferred to `main` after HTTP thread joins (avoids use-after-free during in-flight calls).
- `handlers.errorMessage` normalizes every `anyerror` to a stable non-leaking string; raw `@errorName` never leaks on either transport.
- Frozen enums: `connector_test.service` ∈ {openai, anthropic, discord, twilio, grok}; `ai_train.format` ∈ {jsonl, csv, text} (default jsonl).

## API & Contract Rules

- Public API change → update both `mod.zig` + `stub.zig`; run `zig build check-parity`.
- Parity tool: scans column-0 `pub const`/`pub fn` (not `pub var`, threadlocal, nested). Struct methods are invisible to parity — reach a tracker via a method (e.g., `Store.getTracker()`), not a signature change.
- 5 contract test suites: `surface.zig`, `feature_modules.zig`, `mcp_tools.zig`, `plugin_registry.zig`, `public_docs.zig`.

## Commits & CI

- Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `chore(build):`, …).
- Local `main` and `origin/main` share history (reconciled at `848ec2c8`; the old no-common-ancestor state is resolved). Never force-push `main`.
- CI (`.github/workflows/ci.yml`) runs `zig build check` + `cross-smoke` on macOS; keep its Zig version in sync with `.zigversion` when either moves.

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

- **WDBX**: in-process store + segment/WAL persistence; durable parent dirs `0700` on POSIX. Cluster RPC is real TCP RequestVote/AppendEntries (`ABI_WDBX_CLUSTER_TOKEN` for non-loopback; `ABI_WDBX_CLUSTER_PEERS` allowlist). **Not** production multi-host or sharding. `ABI_REMOTE_COMPUTE_ENDPOINT` points at the operator's own TPU/GPU inference service (no accelerator bundled). Loopback REST supports optional bearer auth (`ABI_WDBX_REST_TOKEN`), token-bucket rate limiting (`ABI_WDBX_RATE_LIMIT_CAPACITY`/`ABI_WDBX_RATE_LIMIT_REFILL`), and TLS env config (`ABI_WDBX_TLS_CERT`/`ABI_WDBX_TLS_KEY` — validated but native TLS not linked; deploy behind a TLS-terminating proxy).
- **WDBX demo/reference modules** in `src/features/wdbx/`: `compression.zig` (lossy int8 quantization), `entropy.zig` (lossless order-0 Huffman — NOT ANS/arithmetic), `neural_compress.zig` (in-process autoencoder — NOT SOTA), `crypto_he.zig` (additive single-key HE — NOT multi-key/FHE), `fhe.zig` (DGHV somewhat-homomorphic, reference parameters — NOT security-audited). All demo/reference-grade. Exercised via `abi wdbx secure demo`.
- **GPU**: Metal framework linked at build time on macOS; `accelerated=false` is the normal state until `g_metal_context.init()` succeeds at runtime. Mid-run GPU failure gracefully degrades to CPU. ANE execution is a disclosed non-goal (100% Zig constraint; requires CoreML/ObjC).
- **Live remote connectors** require explicit credentials + `.live` transport + `https://` base URL. POSIX `auth signin` no-echo; credentials stored as plaintext JSON under `~/.abi/` (or `ABI_CREDENTIALS_PATH`/`XDG_CONFIG_HOME`), dir `0700` + file `0600` owner-only. Windows ACL/keychain + zeroing remain disclosed gaps.
- **Local inference bridge** (`local_bridge`): uses `.live` transport to a loopback `http://127.0.0.1` server with no credentials (exempted from `https://` by loopback carve-out). `endpointFor` accepts optional override; `handleLocalBridgeComplete` (in `src/cli/handlers/train.zig`) checks `ABI_LLAMA_CPP_ENDPOINT` / `ABI_MLX_ENDPOINT` env vars (defaults `http://127.0.0.1:8080` / `http://127.0.0.1:8081`). Falls back to in-process persona router if the local server is unreachable.
- Discord/Twilio logs are redacted (metadata/byte counts only, never message/response bodies).

## AI Subsystem

- **SEA loop**: evidence-augmented self-learning completion with 8-signal scorer + budgeted greedy selection. Persists `AdaptiveModulator` weights (EMA, `alpha=0.3`, key `modulator:weights`) in WDBX. EMA weights load+save **only** on the `--learn`/SEA path; plain `complete` re-runs sentiment analysis each turn with no EMA persistence. SEA is task-aware (7 task types shift signal weights).
- **Constitution audit** (6 principles: truthfulness, safety, helpfulness, fairness, privacy, transparency): **observability-only, not a gate** — sets `audit_passed`/`audit_vetoed`/`escore` in metadata and `std.log.warn`s on violation, but `complete`/`run` still return the response. Safety+privacy form a safety class with a hard veto if either scores < 0.5. Checks use case-insensitive **substring** (infix) matching — "harm" fires on "harmless"; cannot detect novel harm patterns, only the 7 hardcoded negative substrings.
- **Router**: `analyzeSentiment` uses **prefix-only** single-token keyword matching (`startsWithIgnoreCase`); `selectBestProfile` ties resolve `abbey > aviva > abi`, so neutral input (no keyword matches) routes to `abi`. `routeInputAdaptive` in `router.zig` is unreferenced; the live EMA path is `completeAdaptive`/`completeWithStoreAdaptive` via `runLearnLoop` only.
- **Connector validation**: Discord validates numeric snowflake IDs + non-empty printable credentials + ≤2000-byte messages; Twilio validates `AC`+32-hex SIDs, 32-hex tokens, non-empty base URL, non-zero timeout, explicit `.live` transport.
- **Streaming + file context**: `CompletionRequest`/`LearnLoopConfig` accept `stream_callback`/`stream_ctx` for post-hoc chunked output (~4-byte splits; true per-token streaming requires a chunked model backend). `file_context.zig` resolves `@file` mentions (sandboxed to cwd, rejects `..`/absolute/symlink escape) with an 8 KB budget; wired into `agent plan`, `agent multi`, and the `agent tui` REPL.

## Claims & Docs

No unproven claims (distributed sharding, production FHE/AES/RBAC, non-loopback hardening, K8s/H100, Swift/Python/TF stacks, QPS/latency/accuracy/energy/certifications). Public wording: `docs/contracts/external-claims-audit.mdx`. Per-module demo-vs-production boundary: `docs/spec/wdbx-north-star.mdx` §2. Security: `abi-threat-model.md`.

## OpenCode Setup

`opencode.json` auto-loaded (instructions: `AGENTS.md`, `tasks/lessons.md`, `tasks/todo.md`). `.opencode/skills/` is a symlink to `.agents/skills/`. MCP servers: `abi-mcp`, `skill-loop`. Sync canonical skills: `.agents/skills/sync-clis/launch.sh`. Modern-refactor skills (codebase-analysis etc.) in `.agents/skills/` — use for refactors; follow with `./build.sh check`. Superpower + operational skills are enumerated in the system prompt's `available_skills`; don't maintain a parallel list here.

## Learned User Preferences

- Prefer feature branches named with the `cursor/` prefix; when creating a PR from the default branch, branch first and do not commit or push directly to `main`.
- Prefer draft PRs when the create-pull-request flow requests draft.
- Verify interactive dashboard/TUI with `.agents/skills/run-tui/tui.sh` (tmux pty); never prepend Homebrew `/opt/homebrew/bin` ahead of the pinned Zig on PATH.
- Prefer honest status digests and labeled demos over fake live bridges when IPC or production capability is absent.
- For refactor/organization work, prefer scoped tracks (module extraction vs north-star features vs docs/claims) over open-ended clean-slate rewrites; confirm scope before planning.

## Learned Workspace Facts

- Modern-refactor Phases 2–4 and major module-extraction waves are done; remaining high-value org hotspots are mainly `src/features/tui/repl.zig` and `src/features/tui/dashboard.zig` (next safe extracts often `repl_git.zig` / `dashboard_render.zig`).
- Interactive `abi dashboard` / `abi tui` / `abi --tui` use a split layout (diagnostics + Agent Output); one-shot `--once` stays stacked panes — layouts diverge by design.
- Dashboard Agent Output is a status digest, not live `agent tui` traffic; dashboard WDBX is an ephemeral CLI probe (labeled), not the durable agent store.
- Plugin-declared slash-commands dispatch via `__cmd__:<name>` (parallel to `__context__:<name>` for context providers).
- Open product goal for TUI/CLI north-star (streaming, pane-split, richer `@file`) is in `tasks/goals.md`; Partial north-star / demo modules must not be promoted to Current without source and tests.
- MCP HTTP vs WDBX REST framing duplication is intentional (MCP module-root isolation); do not unify them as an organization refactor.
- Canonical refactor layout/status: `docs/spec/abi-refactor-design.mdx`; Approach-1 waves: `docs/superpowers/plans/2026-07-15-approach1-waves-a-b-c.md`; `modern-refactor/examples/` is historical, not the active board.

