# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Note: `AGENTS.md` (and its twin `GEMINI.md`) is this repo's normal canonical
> instruction file, kept deliberately thin to avoid drift across the three. This
> file is an expanded version for Claude Code. If anything here disagrees with
> `AGENTS.md`, `AGENTS.md` wins; if either disagrees with executable source
> (`Cargo.toml`, `tools/check.sh`, `crates/`), trust the source.

ABI is a **nightly Rust** framework for local AI service orchestration, semantic
vector storage (WDBX), claim-honest GPU capability reporting, and an MCP server.
**Intelligence Without Limits.** IWL is Abbey/ABI only; Quesar never carries this
tagline. See `docs/brand.md`.
The Zig implementation that used to live under `src/` has been fully replaced;
see `RUST-REWRITE-PLAN.md` for the port history if you find stale Zig references
in older docs. All live code is under `crates/`; the retired root `src/` tree is
fully absent.
`.github/copilot-instructions.md` was rewritten for the Rust workspace in
PR #777 and now defers to `AGENTS.md`.

## Toolchain — read this before running anything

- **Nightly Rust** pinned via `rust-toolchain.toml`.
- Homebrew installs a stable `cargo`/`rustc` that shadows rustup on `PATH`.
  **Never invoke bare `cargo`** — always use `./tools/cargo.sh`, which resolves
  the rustup nightly toolchain bin dir and prepends it to `PATH`. It also pins
  `cc` to `/usr/bin/cc` ahead of Swiftly's shim, which otherwise refuses to
  link because of an unrelated `.swift-version` pin.

## Commands

| Command | What it does |
|---|---|
| `./tools/check.sh` | **Primary gate.** Run this before considering any change done. `AGENTS.md` carries the authoritative step order; the steps that surprise people are the ones that run *before* fmt — repository policy tests, `./tools/cargo.sh xtask ci verify` (judo #817), the Abbey contract corpus (Python oracle + `./tools/cargo.sh xtask abbey verify`; Python authoritative until byte-identical), and `tools/check_rust_sizes.sh` — plus `tools/bench_regress.sh` (same-system benchmark-regression guard), which runs after the workspace tests and the platform-feature step (Darwin-only `abi-model-runtime --features metal` test/doc, `nvcc`-gated CUDA check) and before the doc build. `./build.sh full-check` is an alias. |
| `./tools/cargo.sh xtask ci verify` | Rust port of `tools/ci_contract.py` (judo #817). `check.sh` runs this; Python tests under `tools/tests/test_ci_contract.py` remain the CI-contract oracle. |
| `./tools/cargo.sh xtask abbey verify contracts/abbey` | Rust port of Abbey corpus verify. `check.sh` runs Python then xtask. Vendor: `./tools/cargo.sh xtask abbey vendor --source … --destination … --source-revision …`. |
| `./build.sh check` | Thin compat wrapper → `./tools/check.sh` |
| `./tools/cargo.sh build -p abi-cli` | Build `target/debug/abi` |
| `./tools/cargo.sh build -p abi-mcp` | Build `target/debug/abi-mcp` |
| `./tools/cargo.sh test -p <crate> --lib -- <filter>` | Focused **unit** tests (in-module `#[cfg(test)]`), e.g. `./tools/cargo.sh test -p abi-contracts --lib -- manifest` |
| `./tools/cargo.sh test -p <crate> --test <name>` | A single **integration** test target under a package's `tests/`, e.g. `./tools/cargo.sh test -p abi-cli --test golden`. `--lib` cannot reach these. |
| `./tools/cargo.sh test --workspace` | Full test suite (also run by `check.sh`) |
| `./tools/cargo.sh fmt --all` | Apply rustfmt |
| `python3 -m unittest discover -s tools/tests -p 'test_*.py'` | The repository policy tests that open `check.sh` (six suites: abbey_contracts, ci_contract, docs_policy, docs_templates, pages_contract, site_contract) |
| `python3 tools/abbey_contracts.py verify contracts/abbey` | The authoritative Python oracle for the Abbey corpus; `xtask abbey verify` must match it |
| `bash ./tools/check_rust_sizes.sh` | File-size gate on its own |
| `RUSTDOCFLAGS="-D warnings" ./tools/cargo.sh doc --workspace --no-deps` | The doc gate on its own — a common late-stage failure |
| `./build.sh cli` / `mcp` / `test` / `fmt` | Remaining `build.sh` aliases besides `check`/`full-check` |
| `./tools/cargo.sh clippy --workspace --all-targets -- -D warnings` | Lint, matching the gate exactly |
| `./mcp/launcher.sh` | Launch the MCP server; prefers `target/release/abi-mcp` then `target/debug/abi-mcp`; run from repo root (or via the launcher) so `@loader_path` resolves `libabi_fm_shim.dylib` on arm64 macOS; set `ABI_MCP_AUTO_BUILD=1` to build on demand |

**Run any `cargo test` here with `< /dev/null`.** `abi auth signin` reads a
secret from stdin when `ABI_AUTH_TOKEN` is unset and stdin is not a TTY, and
`app::tests::auth_signin_without_token_fails_honestly` exercises that no-token
path in-process — so it blocks forever on an inherited stdin that stays open
(a pipe, or a terminal). `tools/check.sh` redirects for you; a bare
`./tools/cargo.sh test --workspace` typed by hand does not.

There is no separate lint-only or build-only CI — `.github/workflows/ci.yml`
runs `./tools/check.sh` on a **self-hosted macOS ARM64 runner** for trusted
same-repo pushes/PRs, and that job does execute (see
`.github/self-hosted-runner.md`). The GitHub-hosted `windows credential ACL`
job executes too: PR #794 ran the Windows Server 2025 ACL and credential-file
tests successfully on 2026-08-19. The hosted macOS `check-hosted` fallback is
restricted to fork PRs and is skipped on same-repo branches by design. Treat a
red executed self-hosted or Windows job as blocking; do not treat a
conditionally skipped fallback as a code failure.

### Local smoke walkthrough

After building `abi-cli`, exercise real command surfaces without live network
credentials — useful for verifying a change beyond what unit tests cover:

```bash
ABI=./target/debug/abi
$ABI backends
$ABI scheduler status
$ABI dashboard --once --plain            # --json / --list-panes / --pane <n> are the scriptable forms
$ABI complete "summarize ABI scheduler status"
$ABI agent plan "stage a safe WDBX refactor"
$ABI wdbx query "$(mktemp -d)/store"   # store stats as JSON; `wdbx stats` is not a command (README's walkthrough still says it is; README is the stale one)
```

The `complete` / `agent` lines above touch the store, so prefix them with
`ABI_WDBX_PATH=:memory:` unless you actually mean to write to `~/.abi/`.

## Architecture

The ABI workspace has **17 local crates** under `crates/*`, verified 2026-09-03
with `cargo metadata` (the 17th is `xtask`, judo #817). Five more packages are sibling path dependencies under
`../wdbx/crates/`: `abi-compute`, `abi-core`, `abi-foundation`,
`abi-telemetry`, and `abi-wdbx`. They are not ABI-local workspace members.
Keep `abi` and `wdbx` adjacent and verify the live metadata rather than trusting
dated prose. Without `../wdbx`, cargo fails at manifest resolution for the whole
workspace (12 of 17 members need a sibling; only `abi-agent-runtime`,
`abi-capability`, `abi-agent-host`, `abi-contracts`, and `xtask` build alone), so
`./tools/check.sh` cannot pass even its early steps because `xtask ci verify`
goes through cargo. CI pins the sibling to an exact SHA (`WDBX_REVISION` in
`.github/workflows/ci.yml`, checked out in all three jobs); a local `../wdbx`
is not pinned, so local green and CI green can diverge on substrate changes.

Workspace policy from `Cargo.toml`: edition 2024, resolver 3,
`rust-version = "1.99"` (the floor `tools/cargo.sh` exists to satisfy),
`unsafe_code = "deny"`, `missing_docs = "warn"`, `clippy::all = "deny"` with
`clippy::pedantic = "warn"`, and a release profile with `panic = "abort"` +
thin LTO. Every crate inherits these with `[lints] workspace = true`.

| Crate | Role |
|---|---|
| `abi-nn` | Tiny character-level neural-net demo trainer — explicitly **not** a production LLM, not distributed. No ABI dependencies. |
| `abi-agent-runtime` | Provider-neutral agent runtime contracts plus deterministic test providers. No ABI dependencies. |
| `abi-capability` | Deny-by-default capability authorization against bounded recording adapters. Depends on `abi-agent-runtime`; it contains no production actuator. |
| `abi-models` | Hash-verified model manifest registry, license-acceptance ledger, and resumable download plumbing. Depends on `abi-foundation`. |
| `abi-contracts` | Independent, bounded verifier for the language-neutral Abbey contract corpus. External schema resolution is disabled. |
| `abi-connectors` | External-service connectors (OpenAI, Anthropic, Grok, Discord, Twilio) built around a `Transport` trait. Every connector has a local and a live transport — see "The local/live split" below. |
| `abi-ai` | Persona identity, routing (Abbey/Aviva/Abi), generation, governance/constitution, and the model catalog (`models.rs`, default `claude-fable-5`). **Pure**: no WDBX dependency, no I/O, fully deterministic — this is what makes `ai_run` byte-reproducible and golden-testable. |
| `abi-plugins` | The 16 bundled plugins plus the plugin manager. Each plugin ships as a compiled-in `mod.rs`/`stub.rs` pair under `crates/abi-plugins/plugins/`, checked with `assert_plugin_parity!`. `abi plugin run` and the MCP `plugin_run` tool dispatch through the same `PluginManager` over the same `BUNDLED` table. |
| `abi-agent-host` | Bounded, policy-authorized tool orchestration for model providers. Depends on `abi-agent-runtime`. This is the crate closest to constitutional invariant A3: authorization is not a generative decision. |
| `abi-wdbx-gateway` | Authenticated bounded gRPC and WebSocket gateway for WDBX v2. Depends on `abi-wdbx`. Its RPC surface (`PutVector`/`Search`/`PutKv`/`GetKv`/`ResolveConflict`/`Stats`/`MembershipChange`/`WatchMutations`) is **not** the CSAPS `MemoryService` surface: there is no `ProposeWrite` write gate and no `Verify`. |
| `abi-model-runtime` | Explicit local model loading and evidenced Candle execution. Depends on `abi-agent-runtime`, `abi-compute`, `abi-models`. |
| `abi-gpu` | Claim-honest GPU/accelerator backend detection. Depends on `abi-compute` and `abi-foundation` only (not `abi-wdbx`); default features are `metal-kernels` + `coreml-ane`, with `cuda-adapter`/`vulkan-adapter` as opt-in capability-only rows. Metal preferred on macOS; the `metal-kernels` feature is **on by default**, but `accelerated=true` additionally requires the Metal DOT pipeline to actually link and initialize at runtime — otherwise deterministic CPU SIMD fallback with `accelerated=false`. Also hosts claim-honest shaders/MLIR/mobile report surfaces. |
| `abi-sea` | SEA (Sparse Evidence Attention) self-learning loop: recalls prior WDBX records relevant to an input, prepends them as context, runs adaptive completion, updates persona-router weights. |
| `abi-worker` | Authenticated, bounded worker-control contracts and admission. Depends on `abi-agent-runtime`, `abi-wdbx-gateway`. |
| `abi-cli` (bin `abi`) | Command metadata, help rendering, process dispatch. Depends on the local workspace and sibling substrate packages. The help surface is a stable, golden-tested contract boundary. |
| `abi-mcp` (bin `abi-mcp`) | JSON-RPC MCP server: the frozen 12-tool stdio surface (primary, 64 KB frame cap); startup also attempts a custom loopback HTTP listener, and bind failure leaves stdio running. It is not persistent MCP HTTP+SSE. |
| `xtask` | In-repo task runner (judo #817). Ports `tools/ci_contract.py` and Abbey corpus/vendor checks. Not a published product crate. Invoke via `./tools/cargo.sh xtask …` (alias in `.cargo/config.toml`). |

The sibling substrate packages supply shared primitives (`abi-foundation`),
telemetry, compute selection, core scheduling/configuration, and the durable
`abi-wdbx` implementation. Under the Abbey System Constitution, `abi-wdbx` is
the provenance-aware episodic substrate. See
`docs/superpowers/specs/2026-08-22-wdbx-conformance-gap-analysis.md` for the
measured implementation gap.

### The local/live split (connectors)

Every external connector distinguishes a **local** transport (deterministic,
no network) from an explicit **live** transport (real network dispatch, needs
stored credentials). This is a safety property, not an implementation detail —
MCP `connector_test` always uses the local path; `complete --live` requires
`abi auth signin <provider>` first and is Anthropic-only for HTTP providers.
Discord validates printable non-whitespace credentials, numeric snowflake-like
IDs, and message size; Twilio validates account SID/auth-token shape, base URL,
timeout, and escapes TwiML/form payloads before dispatch either way.

### Frozen surfaces — contract-tested, don't break casually

- **CLI (13 commands)**: `help`, `complete`, `train`, `agent`, `backends`,
  `plugin`, `auth`, `twilio`, `tui`, `dashboard`, `wdbx`, `scheduler`, `nn`.
- **MCP (12 tools)**: `ai_run`, `ai_complete`, `ai_learn`, `ai_train`,
  `wdbx_query`, `scheduler_stats`, `scheduler_info`, `connector_test`,
  `gpu_status`, `plugin_list`, `wdbx_stats`, `plugin_run`.
- Golden fixtures pinning these live under `tests/golden/` (help text, MCP
  JSON-RPC call/response pairs, WDBX sample segments/manifest, shell completion
  scripts for bash/zsh/fish — `powershell` is explicitly rejected as a
  malformed shell argument, see `app.rs`). Changing frozen-surface output means
  updating the corresponding golden file deliberately, not incidentally.
- The fixtures are pulled in with `include_str!`/`include_bytes!`, so editing a
  file under `tests/golden/` requires a rebuild, not just a re-run. The
  assertions live in:
  - `crates/abi-cli/tests/golden.rs`, `crates/abi-cli/tests/process.rs` — CLI
    help text/JSON, `backends`, completion scripts.
  - `crates/abi-mcp/src/rpc.rs` — `initialize`, `tools/list` (tool order is
    contract order, **not** alphabetical — see `handlers.rs`), `tools/call`.
  - `crates/abi-cli/tests/golden_scheduler.rs` — scheduler status; moved here
    from `abi-core` when WDBX was extracted. Its fixtures were captured from the
    retired Zig binaries and are the only record of the frozen format, so never
    regenerate them casually.
  - `crates/abi-plugins/tests/golden_plugins.rs` — the 16-plugin listing. This is
    a three-way parity check (compiled-in `BUNDLED` table, on-disk
    `abi-plugin.json` manifests, golden fixtures), and the MCP `plugin_list`
    formatter is deliberately duplicated rather than shared so the two surfaces
    fail instead of sharing a bug. `abi plugin list` is alphabetical; MCP
    `plugin_list` is declaration order.
  - CLI help goldens are one file per command (`tests/golden/help-*.txt`), so a
    14th command means a new fixture, not just an edit to `usage.rs`.

## Claims discipline

Do not add or let stand unproven capability claims: production FHE/AES/RBAC,
multi-host sharding, QPS/latency/accuracy numbers, K8s/H100 deployment claims,
native CUDA/ANE kernels. Concretely:
- GPU reports `accelerated=false` whenever native kernels aren't linked — never
  paper over that.
- WDBX "secure" and "cluster" demos are reference-grade / single-host; describe
  them that way.
- `complete --live` is Anthropic-only for HTTP providers; `apple-fm --confirm`
  uses the FoundationModels Swift shim on arm64 macOS only when Apple
  Intelligence is actually ready, and otherwise discloses unavailability rather
  than fabricating a reply.
- See `docs/contracts/external-claims-audit.mdx` for the full policy before
  writing docs/README/CHANGELOG copy that describes capabilities.

## Environment variables

`../wdbx/crates/abi-foundation/src/env.rs` is the single registry for every `ABI_*` var —
constants plus the `get`/`get_or`/`get_bool`/`get_parsed` accessors and the
`set_override`/`reset_overrides`/`lock_for_test` hooks tests use instead of
mutating the real process environment. Add new vars there; don't scatter raw
`std::env::var` calls.

| Var | Effect |
|---|---|
| `ABI_WDBX_PATH` | Store path; `:memory:` for a non-persisting store |
| `ABI_WDBX_PERSIST` | `0` disables persistence |
| `ABI_WDBX_ALLOW_MEMORY_FALLBACK` | Permit falling back to memory when the path is unusable |
| `ABI_WDBX_REST_PORT` / `ABI_WDBX_REST_TOKEN` | Loopback REST listener port / bearer token |
| `ABI_WDBX_RATE_LIMIT_CAPACITY` / `ABI_WDBX_RATE_LIMIT_REFILL` | REST token-bucket tuning |
| `ABI_WDBX_TLS_CERT` / `ABI_WDBX_TLS_KEY` | REST TLS material |
| `ABI_WDBX_CLUSTER_PEERS` / `ABI_WDBX_CLUSTER_TOKEN` | Reference cluster peer list / shared token |
| `ABI_MCP_HTTP_PORT` / `ABI_MCP_HTTP_TOKEN` | Custom loopback MCP compatibility port / optional bearer token (stdio stays tokenless) |
| `ABI_LLAMA_CPP_ENDPOINT` / `ABI_MLX_ENDPOINT` | Local inference endpoints |
| `ABI_OS_POLICY` | Override the `abi agent os` policy file (default `~/.abi/os-policy.toml`); tests set it so they never read the user's real policy |
| `ABI_MCP_AUTO_BUILD` | `mcp/launcher.sh` only — build the server on demand |
| `ABI_WDBX_ENCRYPTION_KEY_FILE` / `ABI_WDBX_SIGNING_KEY_FILE` / `ABI_WDBX_VERIFY_KEY_FILE` | WDBX v2 key material consumed by `abi wdbx db keygen` / `rekey` (`crates/abi-cli/src/wdbx/db.rs`) |
| `ABI_AUTH_TOKEN` | Non-interactive credential for `abi auth signin`; unset + non-TTY stdin reads the secret from stdin (the reason tests need `< /dev/null`) |
| `ABI_BROWSER_STUDIO_TOKEN` | Bearer for the loopback browser studio (`abi agent browser --studio`) |

Only vars referenced from local crates are listed here; the WDBX REST/rate-limit
rows above are consumed in the sibling `abi-foundation` registry and cannot be
verified without `../wdbx` present. Other locally used vars (`ABI_CREDENTIALS_*`,
`ABI_MODELS_DIR`, `ABI_FM_SHIM_FORCE`, `ABI_METAL_KERNELS_FORCE`,
`ABI_CUDA_TOOLCHAIN_DETECTED`, `ABI_VULKAN_TOOLCHAIN_DETECTED`) are documented at
their use sites.

Bearer tokens here are loopback-only hardening, not a TLS substitute.

## Store safety

`~/.abi/` is the **user's live WDBX store** — real data, not a fixture. Tests
must never open that path. Use a scratch `DurableStore` path, or set
`ABI_WDBX_PATH=:memory:` / `ABI_WDBX_PERSIST=0`. Before committing any change
that touches store I/O, re-verify content digests rather than assuming the
existing golden fixtures still cover the new path.

## Conventions

- Conventional Commits. Never force-push `main`.
- Naming: `snake_case` for functions/variables/modules, `PascalCase` for
  types/traits, `SCREAMING_SNAKE_CASE` for constants.
- No silent error swallowing on persistence, inference, or connector paths —
  prefer typed `Result`/domain errors, log or propagate.
- Prefer feature branches `cursor/*` off `origin/main`; land via draft PR then
  `gh pr merge --squash`; delete merged `cursor/*` branches after.
- Session-start reading order for agents: `tasks/lessons.md`, then
  `tasks/todo.md` for current priorities, then `tasks/goals.md` (the largest
  ledger; fed by `tools/goal_capture.sh` and the `goal-ledger` skill).

## Code quality hotspots

Re-measure before trusting this table — it goes stale as work lands
(`find crates -name '*.rs' -exec wc -l {} + | sort -rn | head`).

`tools/check_rust_sizes.sh` rejects Rust files (tracked or untracked, tests
included) over 1,000 lines and rejects `crates/abi-cli/src/main.rs` over 200.
Local watch list (900–1000 lines, measured 2026-09-04):
`crates/abi-cli/src/dashboard.rs` (932),
`crates/abi-contracts/src/lib.rs` (929), and
`crates/abi-cli/src/complete.rs` (904); next up is
`crates/abi-cli/src/wdbx_simulate.rs` (827, the bounded multiway rewriting
engine behind `abi wdbx simulate`). The sibling `../wdbx` tree has its own
band-dwellers (`abi-wdbx/src/{hnsw,store,multiway,v2/lifecycle}.rs` were all
920–960 when last measured) but that checkout is not always present, so
measure there separately. Re-measure before trusting these numbers.

Already split — don't recreate the flat versions: `crates/abi-cli/src/wdbx.rs` is
now the `wdbx/` module directory, `../wdbx/crates/abi-wdbx/src/format.rs` is down to
~444 lines, and `crates/abi-cli/src/agent.rs` is down to ~550 (the line-mode
REPL moved to `repl.rs`; its raw editor/TTY transport is isolated in
`repl_editor.rs`; `os.rs` owns command execution with `os/policy.rs` +
`os/audit.rs` beside it). Scheduler tests live under `scheduler/tests.rs`;
multiway export/resume logic and tests live under `multiway/`; WAL tests live
under `wal/tests.rs`.

<!-- machine-git-policy -->
## Git workflow (machine policy, 2026-08-27)

Work on the default branch in this canonical checkout. Do not create
branches or worktrees by default; they are for tasks that genuinely need
isolation, or when Donald asks. Any worktree or topic branch created here
must be merged back into this checkout's default branch, the worktree
removed, and the branch deleted, before pushing and before the task is
called done. Full policy: `~/.claude/CLAUDE.md` (*Git discipline*).
<!-- /machine-git-policy -->
