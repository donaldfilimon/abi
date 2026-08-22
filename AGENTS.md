# AGENTS.md — abi

Canonical instruction file for the **Rust (nightly)** ABI framework. Sibling
files `CLAUDE.md` / `GEMINI.md` are thin redirects here. If anything conflicts
with executable source (`Cargo.toml`, `tools/check.sh`, `crates/`), trust the
source. Session-start checklist: `tasks/lessons.md`; active board: `tasks/todo.md`.

## Toolchain

- **Nightly Rust** via `rust-toolchain.toml`. Homebrew stable `cargo` shadows
  rustup on this machine — **always** use `./tools/cargo.sh` (never bare
  `cargo`). That wrapper also keeps Swiftly's `cc` from breaking the link step.
- Primary gate: `./tools/check.sh` (fmt, clippy `-D warnings`, build, test, docs).

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

## Architecture

- Workspace crates under `crates/*` — **19**, verified 2026-08-22 (this list
  previously named 13 and omitted six): `abi-foundation`, `abi-telemetry`,
  `abi-nn`, `abi-compute` (cycle-free compute contracts/CPU SIMD),
  `abi-agent-runtime` (provider-neutral agent contracts), `abi-core`,
  `abi-models` (hash-verified model manifests + license ledger),
  `abi-connectors`, `abi-ai`, `abi-plugins`,
  `abi-agent-host` (policy-authorized tool orchestration), `abi-wdbx`,
  `abi-wdbx-gateway` (authenticated gRPC/WebSocket gateway),
  `abi-model-runtime` (local model loading + evidenced Candle execution),
  `abi-gpu`, `abi-sea`, `abi-worker` (worker-control contracts + admission),
  `abi-cli` (binary `abi`), `abi-mcp` (binary `abi-mcp`).
- `abi-wdbx` is the **provenance-aware episodic substrate** under the Abbey
  System Constitution, not merely a vector store. Conformance against its
  specification is measured in
  `docs/superpowers/specs/2026-08-22-wdbx-conformance-gap-analysis.md`; the
  evidence half of that specification is largely unimplemented today.
- **MCP launcher** (`mcp/launcher.sh`): prefers `target/release/abi-mcp` then
  `target/debug/abi-mcp`; optional `ABI_MCP_AUTO_BUILD=1`.
- Golden fixtures under `tests/golden/` pin frozen CLI/MCP surfaces.

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
