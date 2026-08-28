# AGENTS.md — abi

Canonical instruction file for the **Rust (nightly)** ABI framework. Sibling
files `CLAUDE.md` / `GEMINI.md` are thin redirects here. If anything conflicts
with executable source (`Cargo.toml`, `tools/check.sh`, `crates/`), trust the
source. Session-start checklist: `tasks/lessons.md`; active board: `tasks/todo.md`
(OpenCode loads all three through `opencode.json`'s `instructions`).

## Toolchain

- **Nightly Rust** via `rust-toolchain.toml`. Homebrew stable `cargo` shadows
  rustup on this machine — **always** use `./tools/cargo.sh` (never bare
  `cargo`). That wrapper also keeps Swiftly's `cc` from breaking the link step.
- Primary gate: `./tools/check.sh`, in step order — repository policy tests,
  Abbey contract corpus, Rust source size limits, fmt, clippy `-D warnings`,
  build, workspace tests, local model device features (Metal on macOS; CUDA
  compile only when `nvcc` exists), the same-system benchmark regression guard,
  and docs. A step skipped for a missing platform prints why and is not a failure.
- `./tools/check.sh` also runs `tools/check_rust_sizes.sh`, which caps every Rust
  file at **1,000 lines** and `crates/abi-cli/src/main.rs` at **200**. A refactor
  that pushes a file over either limit fails the gate before clippy runs.

## Commands

| Command | What it does |
|---------|-------------|
| `./tools/check.sh` | Primary gate: format, clippy, build, workspace tests, docs |
| `./tools/cargo.sh build -p abi-cli` | Build `target/debug/abi` |
| `./tools/cargo.sh build -p abi-mcp` | Build `target/debug/abi-mcp` |
| `./tools/cargo.sh test -p <crate> --lib -- <filter>` | Focused unit tests |
| `./tools/cargo.sh fmt --all` | Apply rustfmt |
| `./tools/cargo.sh clippy --workspace --all-targets -- -D warnings` | Lint |

Thin compatibility entrypoint: `./build.sh check` → `./tools/check.sh`.

**Redirect stdin on every hand-typed `cargo test` here:**
`./tools/cargo.sh test --workspace < /dev/null`. `abi auth signin` reads a
secret from stdin when `ABI_AUTH_TOKEN` is unset and stdin is not a TTY, and
`app::tests::auth_signin_without_token_fails_honestly` exercises that path
in-process — on an inherited open stdin it blocks in `read_line` forever and
reads as a hung suite rather than a failing test. `tools/check.sh` redirects
for you (added 2026-08-28); a direct `cargo.sh test` does not.

## Architecture

- Local workspace crates under `crates/*` — **16**, verified 2026-08-23 from
  `cargo metadata`: `abi-agent-host`, `abi-agent-runtime`, `abi-ai`,
  `abi-capability`, `abi-cli`, `abi-connectors`, `abi-contracts`, `abi-gpu`,
  `abi-mcp`, `abi-model-runtime`, `abi-models`, `abi-nn`, `abi-plugins`,
  `abi-sea`, `abi-wdbx-gateway`, and `abi-worker`.
- Five additional workspace dependencies are sibling packages in
  `../wdbx/crates/`, not ABI-local crates: `abi-compute`, `abi-core`,
  `abi-foundation`, `abi-telemetry`, and `abi-wdbx`. The sibling directories
  must stay adjacent because `Cargo.toml` resolves them through relative path
  dependencies; never recreate stale `crates/abi-*` copies here.
- `abi-ai` owns deterministic persona routing and depends only on
  `abi-foundation`, `abi-telemetry`, and serde — no WDBX, no I/O. Keep
  retrieval and persistence in `abi-sea`, the CLI/MCP layer, or the substrate;
  adding a store dependency here breaks the routing crate's determinism.
- The sibling `abi-wdbx` package is the **provenance-aware episodic substrate**
  under the Abbey System Constitution, not merely a vector store. Conformance
  against its specification is measured in
  `docs/superpowers/specs/2026-08-22-wdbx-conformance-gap-analysis.md`; the
  evidence half of that specification is largely unimplemented today.
- **MCP launcher** (`mcp/launcher.sh`): prefers `target/release/abi-mcp` then
  `target/debug/abi-mcp`; optional `ABI_MCP_AUTO_BUILD=1`.
- Golden fixtures under `tests/golden/` pin frozen CLI/MCP surfaces.
- Canonical `ABI_*` environment names and all environment access live in
  `../wdbx/crates/abi-foundation/src/env.rs`. Use its override and locking
  hooks in tests rather than mutating the process environment ad hoc.

## Frozen surfaces (contract-tested — don't break)

- **CLI (13)**: `help`, `complete`, `train`, `agent`, `backends`, `plugin`,
  `auth`, `twilio`, `tui`, `dashboard`, `wdbx`, `scheduler`, `nn`.
- **MCP (12)**: `ai_run`, `ai_complete`, `ai_learn`, `ai_train`, `wdbx_query`,
  `scheduler_stats`, `scheduler_info`, `connector_test`, `gpu_status`,
  `plugin_list`, `wdbx_stats`, `plugin_run`. Stdio JSON-RPC (64 KB cap).
  A loopback-only custom HTTP compatibility listener is attempted at startup;
  bind failure leaves stdio running. Its one-event `/sse` discovery response
  is not persistent MCP HTTP+SSE.

## Claims discipline

No unproven claims (production FHE/AES/RBAC, production multi-host deployment,
QPS/latency/accuracy, K8s/H100, CUDA/Vulkan runtime, or ANE residency). Metal
execution is local-runtime scoped. CoreML has output-checked tiny-model inference
under a `.cpuAndNeuralEngine` request, but no placement/residency proof. WDBX secure
demos are reference-grade. GPU reports `accelerated=false` when kernels are not linked.
`complete --live` is Anthropic-only for HTTP providers; `apple-fm --confirm`
uses the FoundationModels Swift shim on arm64 macOS when Apple Intelligence is
ready, otherwise discloses unavailability (never fabricates a reply).

## Store safety

`~/.abi/` is the user's live WDBX store. Tests must not open the real path —
use scratch `DurableStore` paths, `ABI_WDBX_PATH=:memory:`, or
`ABI_WDBX_PERSIST=0`. Recheck content digests before commits that touch store I/O.

## CI & commits

- Conventional Commits. **Never force-push `main`**.
- CI runs `./tools/check.sh` on nightly Rust on the **self-hosted macOS ARM64
  runner** for trusted same-repo pushes/PRs. The GitHub-hosted Windows credential
  ACL job also executes: PR #794 ran the Windows Server 2025 tests successfully
  on 2026-08-19. The hosted macOS fallback is restricted to fork PRs and is
  therefore skipped on trusted same-repo work. Treat either an executed
  self-hosted gate or Windows ACL failure as blocking; a conditionally skipped
  fallback is not a code failure.
- Every CI job checks out `donaldfilimon/wdbx` into `../wdbx` at a pinned
  revision, because ABI's manifests resolve the five substrate crates through
  that relative path. A CI failure naming `abi-compute`/`abi-wdbx` as a missing
  package is a checkout-pin problem, not a broken manifest.

## Learned User Preferences

- Prefer feature branches `cursor/*` from `origin/main`; do not commit/push
  directly to `main`; never force-push.
- Land finished work via draft PR then merge (prefer `gh pr merge --squash`);
  return to main; remove merged `cursor/*` branches after.
- "continue", "continue with all", "do all", "finalize", or "merge all into main"
  means broaden and keep going; a green gate alone is not a stop unless a stop
  was named.
- Honest status/demos only — never fake-complete honest stubs, ANE, audited FHE,
  SOTA compression, or prod multi-host sharding.
- For ABI "test all features" / live verify, smoke `target/debug/abi` (e.g.
  `backends`, representative commands) in addition to `./tools/check.sh`.

## Learned Workspace Facts

- Live code is Rust under `crates/`. The Zig tree, `modernized/`, and
  `modern-refactor/` scaffolds were removed; historical archive docs may still
  mention Zig.
- Interactive `abi dashboard|tui` enters raw-mode on a TTY (one-shot for
  `--once`/`--json`/non-TTY); `agent tui` uses a bounded raw-mode editor only
  when stdin and stdout are TTYs, with the deterministic legacy line mode for
  redirected input.
- Plugin slash-commands dispatch via `__cmd__:<name>` (parallel to
  `__context__:<name>`).
- Legacy v1 WDBX borrowed vectors are mutation-scoped. V2/versioned search
  views retain immutable `Arc` segment/index snapshots and remain valid across
  later journal or segment publication.

After any edit: `./tools/check.sh`.

<!-- machine-git-policy -->
## Git workflow (machine policy, 2026-08-27)

Work on the default branch in this canonical checkout. Do not create
branches or worktrees by default; they are for tasks that genuinely need
isolation, or when Donald asks. Any worktree or topic branch created here
must be merged back into this checkout's default branch, the worktree
removed, and the branch deleted, before pushing and before the task is
called done. Full policy: `~/.claude/CLAUDE.md` (*Git discipline*).
<!-- /machine-git-policy -->
