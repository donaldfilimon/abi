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

- Workspace crates under `crates/*`: `abi-foundation`, `abi-core`, `abi-ai`,
  `abi-sea`, `abi-nn`, `abi-gpu`, `abi-wdbx`, `abi-connectors`, `abi-plugins`,
  `abi-telemetry`, `abi-cli` (binary `abi`), `abi-mcp` (binary `abi-mcp`).
- **MCP launcher** (`mcp/launcher.sh`): prefers `target/release/abi-mcp` then
  `target/debug/abi-mcp`; optional `ABI_MCP_AUTO_BUILD=1`.
- Golden fixtures under `tests/golden/` pin frozen CLI/MCP surfaces.

## Frozen surfaces (contract-tested — don't break)

- **CLI (13)**: `help`, `complete`, `train`, `agent`, `backends`, `plugin`,
  `auth`, `twilio`, `tui`, `dashboard`, `wdbx`, `scheduler`, `nn`.
- **MCP (12)**: `ai_run`, `ai_complete`, `ai_learn`, `ai_train`, `wdbx_query`,
  `scheduler_stats`, `scheduler_info`, `connector_test`, `gpu_status`,
  `plugin_list`, `wdbx_stats`, `plugin_run`. Stdio JSON-RPC (64 KB cap).
  Loopback-only HTTP/SSE when enabled.

## Claims discipline

No unproven claims (production FHE/AES/RBAC, multi-host sharding, QPS/latency/
accuracy, K8s/H100, native CUDA/ANE kernels). WDBX secure demos are
reference-grade. GPU reports `accelerated=false` when kernels are not linked.
`complete --live` is Anthropic-only for HTTP providers; `apple-fm --confirm`
uses the FoundationModels Swift shim on arm64 macOS when Apple Intelligence is
ready, otherwise discloses unavailability (never fabricates a reply).

## Store safety

`~/.abi/` is the user's live WDBX store. Tests must not open the real path —
use scratch `DurableStore` paths, `ABI_WDBX_PATH=:memory:`, or
`ABI_WDBX_PERSIST=0`. Recheck content digests before commits that touch store I/O.

## CI & commits

- Conventional Commits. **Never force-push `main`**.
- CI runs `./tools/check.sh` on nightly Rust (self-hosted macOS ARM64 for
  same-repo events; hosted runners for fork PRs).

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

- Live code is Rust under `crates/`. The Zig tree was removed in the rewrite
  teardown; historical references in docs/skills may still say Zig.
- Interactive `abi tui|dashboard` is one-shot digest by default; `agent tui` is
  line-mode (raw-mode not linked). Dashboard is digest only.
- Plugin slash-commands dispatch via `__cmd__:<name>` (parallel to
  `__context__:<name>`).
- WDBX borrowed vectors are zero-copy; lifetime ends on next mutation.

After any edit: `./tools/check.sh`.
