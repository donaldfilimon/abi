# AGENTS.md - abi

Canonical agent instructions. Executable source wins over prose (`Cargo.toml`, `tools/check.sh`, `crates/`). `GEMINI.md` redirects here; `CLAUDE.md` is the expanded redirect; `.github/copilot-instructions.md` is the editor reminder. Deeper rules, crate roles, and the env table: `CLAUDE.md`. `opencode.json` loads `tasks/lessons.md`, `tasks/todo.md`, and `tasks/goals.md` alongside this file.

## Toolchain and gate

- Stable Rust **1.99.1** (`rust-toolchain.toml`); declared floor `rust-version = "1.99"`; edition 2024; resolver 3.
- Never bare `cargo`/`rustup run` (Homebrew shadows; Swiftly `cc` shim breaks links). Always `./tools/cargo.sh` (toolchain bin + `/usr/bin/cc` first).
- Gate Python needs 3.11+ (`tomllib`); `check.sh`/`cargo.sh` resolve it (override `ABI_PYTHON`); the probe uses a distinguishing exit status (3).
- Required dependency scanner (`cargo-audit`/`cargo-deny`) in `~/.cargo/bin`; the scan must RUN; SKIP is not evidence.
- Primary gate: `./tools/check.sh` (`./build.sh check` alias). Steps:
  1. rustc/cargo versions
  2. `python -m unittest discover -s tools/tests -p 'test_*.py'` (policy; needs 3.11+)
  3. `./tools/cargo.sh xtask ci verify` (locked)
  4. `python tools/abbey_contracts.py verify contracts/abbey` then `./tools/cargo.sh xtask abbey verify contracts/abbey` (Python oracle authoritative until byte-identical)
  5. `ABI_DEP_SCAN_REQUIRE=1 ./tools/security/run-dep-scan.sh`
  6. `bash ./tools/check_rust_sizes.sh` (`*.rs` <= 1000 lines; `crates/abi-cli/src/main.rs` <= 200)
  7. `./tools/cargo.sh fmt --all -- --check`
  8. `./tools/cargo.sh clippy --locked --workspace --all-targets -- -D warnings`
  9. `./tools/cargo.sh build --locked --workspace --all-targets`
  10. `./tools/cargo.sh test --locked --workspace < /dev/null`
  11. sibling WDBX: `./tools/cargo.sh test --locked --manifest-path ../wdbx/Cargo.toml -p abi-wdbx --test abbey_contracts --test v3_cross_language_commitment --test v3_cross_language_episode < /dev/null`
  12. Darwin: `./tools/cargo.sh test --locked -p abi-model-runtime --features metal < /dev/null` + `RUSTDOCFLAGS="-D warnings" ./tools/cargo.sh doc --locked -p abi-model-runtime --features metal --no-deps --document-private-items`; if `nvcc`: `./tools/cargo.sh check --locked -p abi-model-runtime --features cuda`
  13. `./tools/bench_regress.sh`
  14. `RUSTDOCFLAGS="-D warnings" ./tools/cargo.sh doc --locked --workspace --no-deps --quiet`
  15. `./tools/cargo.sh build --locked --release -p abi-cli -p abi-mcp`
- Dependency-resolving Cargo stages use `--locked`. Set `CARGO_TARGET_DIR` for an isolated build; `bench_regress.sh` uses the same target.
- Doc-only edits: `python3 -m unittest discover -s tools/tests -p 'test_docs*.py'` (plus policy suites) + `git diff --check`; for `docs/` also run `.agents/skills/docs-validate/validate.sh` after loading `abi-doc-claims-sync`.
- Size gate: `tools/check_rust_sizes.sh` (git-tracked + untracked, non-ignored `.rs`). Near-limit files (e.g. `complete.rs` ~945 lines) must be split into modules; do not trim the gate.
- Smoke: `./.agents/skills/run-abi/smoke.sh` (builds both binaries, exercises CLI + MCP stdio, writes `target/debug/run-abi-smoke.txt`).
- xtask: `./tools/cargo.sh xtask ci verify`; `./tools/cargo.sh xtask abbey verify contracts/abbey`; vendor via `./tools/cargo.sh xtask abbey vendor --source ... --destination ... --source-revision ...`.
- Focused tests: unit `./tools/cargo.sh test -p <crate> --lib -- <filter> < /dev/null`; integ `./tools/cargo.sh test -p <crate> --test <name> [<filter>] < /dev/null` (e.g. `-p abi-cli --test golden`).
- WDBX substrate: `./tools/cargo.sh test --manifest-path ../wdbx/Cargo.toml -p abi-wdbx --lib -- <filter> < /dev/null`.
- Feature flags: `abi-model-runtime --features metal|cuda`; `abi-gpu` (default `metal-kernels`+`coreml-ane`; opt-in `cuda-adapter`,`vulkan-adapter`); `abi-connectors` (default `live`+`foundationmodels`); `abi-cli --features full-fhe|experimental-dghv-bootstrap`.
- Always run cargo test with `< /dev/null` (auth signin reads stdin and hangs on an open pipe/TTY).
- Store-safe smoke: `ABI_WDBX_PATH=:memory: ./tools/cargo.sh run -p abi-cli -- complete "..."`.
- Launch MCP only via `./mcp/launcher.sh [stdio]` (repo root or script); `ABI_MCP_AUTO_BUILD=1` builds if absent. Binaries: `target/{debug,release}/abi` and `target/{debug,release}/abi-mcp`.
- Rebuild the embedding test target after editing frozen surfaces or goldens (e.g. `-p abi-cli --test golden`).
- dylib rewrite hazard: never `cp`/`fs::copy` over an existing `target/<profile>/*.dylib` in place; repair with `cp X X.new && mv -f X.new X`.
- Direct toolchain `cargo` drops `DYLD_FALLBACK_LIBRARY_PATH` (rust-objcopy SIGABRT); use the wrapper.

## Workspace and boundaries

- All live code under `crates/*` (17 members): `abi-agent-host`, `abi-agent-runtime`, `abi-ai`, `abi-capability`, `abi-cli`, `abi-connectors`, `abi-contracts`, `abi-gpu`, `abi-mcp`, `abi-model-runtime`, `abi-models`, `abi-nn`, `abi-plugins`, `abi-sea`, `abi-wdbx-gateway`, `abi-worker`, `xtask` (unpublished runner). No Zig (retired; see `RUST-REWRITE-PLAN.md` history only).
- Sibling substrate (required, adjacent `../wdbx/crates/`): `abi-compute`, `abi-core`, `abi-foundation`, `abi-telemetry`, `abi-wdbx`. Cargo resolution/identity require the layout; do not vendor or git-dep. CI pins `WDBX_REVISION` in `.github/workflows/ci.yml` (`2969d71a2c3a81a060dcfc6cf53b4617f13c8b94`); update the pin only after that exact rev passes the gate.
- No separate daemon binary; the scheduler runs in-process (`abi scheduler ...`).
- `abi-ai` core is store-agnostic/deterministic (I/O only in `file_context` etc.); WDBX bits belong in sea/CLI/MCP/wdbx crates.
- Launch MCP only via the launcher.
- `ABI_*` env vars + test hooks live in `../wdbx/crates/abi-foundation/src/env.rs`; use provided overrides/locks in tests, never raw env mutation.
- Frozen surfaces (change needs coordinated source + golden + tests):
  - CLI 14 (`crates/abi-cli/src/usage.rs`): help, complete, train, agent, backends, plugin, auth, twilio, tui, dashboard, wdbx, scheduler, nn, improve.
  - MCP 12 (`crates/abi-mcp/src/handlers.rs`): ai_run, ai_complete, ai_learn, ai_train, wdbx_query, scheduler_stats, scheduler_info, connector_test, gpu_status, plugin_list, wdbx_stats, plugin_run.
- Golden contracts under `tests/golden/` (help, mcp json, completions bash/zsh/fish, wdbx samples). Pulled via `include_str/bytes` — rebuild the test binary after edits.
- MCP transport: stdio is the contract (newline JSON-RPC, 64 KiB frame cap). Loopback HTTP (`ABI_MCP_HTTP_*`): GET `/sse` opens a persistent MCP 2024-11-05 HTTP+SSE session (POST `/message?sessionId=<id>` -> 202, response arrives as an SSE `message` event; unknown session 404; at most 16 sessions, then 503); POST `/message` without a session keeps the one-shot direct-reply compat mode. Not Streamable HTTP (2025-03-26).
- `connector_test` (MCP/CLI) always uses deterministic local transport, never live net.
- Codegen/contracts: `abi-wdbx-gateway` compiles `proto/gateway.proto` via vendored protoc; `abi-gpu` builds `libabi_metal_dot.dylib` on macOS; `abi-connectors` builds `libabi_fm_shim.dylib` on arm64 macOS (`foundationmodels` feature). ABI plugins ship as compiled-in `mod.rs`/`stub.rs` pairs checked with `assert_plugin_parity!`. Golden fixtures are contract truth — update deliberately.
- Instruction consistency: no automated checker. `.claude/agents/instruction-sync.md` is a read-only reviewer across AGENTS/CLAUDE/GEMINI. Policy tests: `tools/tests/test_*.py`.

## Workflow gotchas

- At start of work: read `tasks/lessons.md` + `tasks/todo.md` + `tasks/goals.md`; run `./tools/check.sh`; `git status --short --branch`.
- `opencode.json` and `.mcp.json` wire MCP to `./mcp/launcher.sh stdio` — use that, never a raw binary path.
- Store-safe smoke: `ABI_WDBX_PATH=:memory: ./tools/cargo.sh run -p abi-cli -- complete "..."` (and similar for agent etc.).
- After editing frozen surfaces or goldens, rebuild the specific test target that embeds them.
- Rewriting a `target/<profile>/*.dylib` in place truncates the inode; a long-lived `abi-mcp stdio` keeps old code and later execs die `SIGKILL (Code Signature Invalid)`. Build scripts copy to `.tmp` and `rename`; repair with `cp X X.new && mv -f X.new X`.
- Exec'ing the toolchain's `cargo` directly drops rustup's `DYLD_FALLBACK_LIBRARY_PATH`; rustc's release strip then SIGABRTs. `tools/cargo.sh` exports the fallback list; the rustc warning line is the evidence.

## Data and claims safety

- Never open the live user store (`~/.abi/`) from tests/smokes. Use `ABI_WDBX_PATH=:memory:`, `ABI_WDBX_PERSIST=0`, or foundation temp paths / scratch `DurableStore`.
- Read `docs/contracts/external-claims-audit.mdx` before docs touching capabilities. No claims of prod FHE/AES/RBAC, multi-host sharding, K8s/H100, verified CUDA/Vulkan/ANE kernels, or benchmark numbers without source+test+artifact evidence. WDBX secure/cluster are ref-grade/single-host demos. Compiled feature != executed path; always preserve `accelerated=false` fallback reporting.

## Environment

- Registry: `../wdbx/crates/abi-foundation/src/env.rs`. Core vars: `ABI_WDBX_PATH`, `ABI_WDBX_PERSIST`, `ABI_WDBX_ALLOW_MEMORY_FALLBACK`, `ABI_WDBX_REST_PORT`/`ABI_WDBX_REST_TOKEN`, rate-limit/tls/cluster vars, `ABI_MCP_HTTP_PORT`/`ABI_MCP_HTTP_TOKEN`, `ABI_LLAMA_CPP_ENDPOINT`/`ABI_MLX_ENDPOINT`, `ABI_OS_POLICY`, `ABI_MCP_AUTO_BUILD`, `ABI_WDBX_ENCRYPTION_KEY_FILE`/`_SIGNING_KEY_FILE`/`_VERIFY_KEY_FILE`, `ABI_AUTH_TOKEN`, `ABI_BROWSER_STUDIO_TOKEN`. Full table and use-site-only vars: `CLAUDE.md` section Environment variables.

## CI

- `.github/workflows/ci.yml`: one job, `check`, on the self-hosted macOS ARM64 runner for trusted same-repo events; fork PRs get no job. WDBX checked out as sibling at exact `WDBX_REVISION`. Hosted `check-hosted` fallback, `windows-acl` job, and `dependency-scan.yml` removed 2026-09-28 (billing lock; no Windows/Linux runner), so Windows credential ACL tests run nowhere in CI.
- Billing-locked hosted jobs may refuse (steps=[], no logs, annotation about account lock) — treat self-hosted green as the signal; do not alter code for refused jobs.
- Local tests authorized for the 2026-10-20 completion train; CI remains paused until Donald resumes it.

## References

- `CLAUDE.md`: crate roles/ownership, env table, code-quality hotspots, macOS hazards.
- `.github/copilot-instructions.md`: editor-facing reminder.
- `docs/contracts/external-claims-audit.mdx`: claims policy.

<!-- machine-git-policy -->
## Git Workflow

Work only in the canonical checkout on default branch (`main`). Create branch/worktree only for required isolation (parallel risky work, or when asked); before done: merge back, rm worktree, delete branch label. Inspect `git status --short --branch` at start of changes and preserve unrelated dirty files. Conventional commits; no force-push on main. See `~/.claude/CLAUDE.md` for full.
<!-- /machine-git-policy -->
