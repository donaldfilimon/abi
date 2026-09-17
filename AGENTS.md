# AGENTS.md - abi

Canonical instructions; executable source wins over prose. `opencode.json` loads
`tasks/lessons.md` and `tasks/todo.md` alongside this file.

## Toolchain And Gates

- **Never bare `cargo`** (or plain rustup run). Homebrew installs real cargo/rustc that ignores rust-toolchain.toml and Swiftly cc shim breaks links. Always `./tools/cargo.sh` (it puts nightly bin + /usr/bin cc first).
- Primary gate (run before commit or handoff): `./tools/check.sh` (build.sh check is compat alias).
  Exact steps (trust script): versions; py policy tests (tools/tests/test_*.py); xtask ci verify;
  py abbey_contracts.py verify + xtask abbey verify contracts/abbey (py oracle authoritative until byte-id);
  sizes; fmt --check; clippy -D warnings; build --workspace --all-targets; test --workspace < /dev/null;
   Darwin: abi-model-runtime --features metal test+doc (RUSTDOCFLAGS=-D); if nvcc: check --features cuda;
  bench_regress.sh; RUSTDOCFLAGS=-D cargo doc --workspace --no-deps.
  Skips are not evidence.
- Doc-only edits: py unittest (test_docs*.py or policy) + `git diff --check`; for docs/ also run (after loading abi-doc-claims-sync skill) `.agents/skills/docs-validate/validate.sh`.
- Size gate: `tools/check_rust_sizes.sh` (and in check.sh): *.rs <=1000 lines via git ls-files; crates/abi-cli/src/main.rs <=200.
- Build: `./tools/cargo.sh build -p abi-cli` (→ target/debug/abi), `-p abi-mcp` (→ abi-mcp).
- xtask: `./tools/cargo.sh xtask ci verify` (alias defined in `.cargo/config.toml` as `run --manifest-path crates/xtask/Cargo.toml --`); same for `xtask abbey verify contracts/abbey`.
- Focused: unit `./tools/cargo.sh test -p <crate> --lib -- <filter> < /dev/null`; integ `./tools/cargo.sh test -p <crate> --test <name> < /dev/null`.
- WDBX substrate tests (from abi checkout): `./tools/cargo.sh test --manifest-path ../wdbx/Cargo.toml -p abi-wdbx --lib -- <filter> < /dev/null`.

Always run hand-invoked `cargo test` (or via tools/cargo.sh) with `< /dev/null`. Auth signin tests read secret on non-TTY stdin and hang on inherited open pipe/TTY; check.sh enforces it.

## Workspace & Boundaries

- All live code under `crates/*` (17 members incl. xtask unpublished runner). No Zig (retired; see RUST-REWRITE-PLAN.md only for history).
- **Sibling substrate (required)**: `abi-compute|core|foundation|telemetry|wdbx` live in adjacent `../wdbx/crates/`. Cargo resolution and crate identity require the layout; do not vendor or git-dep them. CI pins `WDBX_REVISION` in .github/workflows/ci.yml — update pin only after the exact rev passes `./tools/check.sh` with local sibling at that sha.
- `abi-ai` core is store-agnostic/deterministic (I/O only in file_context etc.); WDBX bits belong in sea/CLI/MCP/wdbx crates.
- Launch MCP **only** via `./mcp/launcher.sh [stdio]` (from repo root or the script) so release/debug pick + @loader_path for libabi_fm_shim.dylib succeeds on arm64. `ABI_MCP_AUTO_BUILD=1` builds if absent (not on source change). Direct target/.../abi-mcp can hide stale build or break dylib.
- ABI_* env vars + test hooks are in `../wdbx/crates/abi-foundation/src/env.rs`; use the provided overrides/locks in tests, never raw env mutation.
- Frozen surfaces (change requires coordinated source + golden + test updates):
  - 14 CLI commands (usage.rs in abi-cli): help, complete, train, agent, backends, plugin, auth, twilio, tui, dashboard, wdbx, scheduler, nn, improve.
  - 12 MCP tools (handlers.rs in abi-mcp): ai_run, ai_complete, ai_learn, ai_train, wdbx_query, scheduler_stats, scheduler_info, connector_test, gpu_status, plugin_list, wdbx_stats, plugin_run.
- Golden contracts under `tests/golden/` (help, mcp json, completions bash/zsh/fish, wdbx samples). Pulled via include_str/bytes — must rebuild the test binary (e.g. -p abi-cli --test golden) after edits.
- MCP: stdio is the contract (newline JSON-RPC, 64 KiB frame cap). Loopback HTTP (ABI_MCP_HTTP_*) is one-shot compat only; GET /sse just advertises POST /message, not persistent SSE.
- `connector_test` (MCP/CLI) always uses deterministic local transport, never live net.

## Workflow Gotchas

- At start of work: read `tasks/lessons.md` + `tasks/todo.md` (opencode loads them); run `./tools/check.sh`; `git status --short --branch`.
- opencode.json (and .mcp.json) wire MCP to `./mcp/launcher.sh stdio` — use that, never raw binary path.
- Store-safe smoke: `ABI_WDBX_PATH=:memory: ./tools/cargo.sh run -p abi-cli -- complete "..."` (and similar for agent etc.).
- After editing frozen surfaces or goldens, rebuild the specific test target that embeds them.
- **Rewriting a `target/<profile>/*.dylib` in place** — a build that `fs::copy`s or `cp`s over an existing dylib truncates the same inode; if a long-lived `abi-mcp stdio` still has it mapped, the kernel keeps the old code directory and every later exec of `abi`/`abi-mcp` dies `SIGKILL (Code Signature Invalid)` while `codesign -vv` says valid (2026-09-16). Build scripts copy to `.tmp` and `rename`; repair a broken tree with `cp X X.new && mv -f X.new X`, not a rebuild.
- **Exec'ing the toolchain's `cargo` directly drops rustup's `DYLD_FALLBACK_LIBRARY_PATH`** — rustc on Apple targets strips release binaries with `lib/rustlib/<host>/bin/rust-objcopy`, whose `@rpath/libLLVM.dylib` only resolves through the fallback path the rustup proxy injects (`<toolchain>/lib` first). Without it every release link prints `warning: stripping debug info with rust-objcopy failed: signal: 6 (SIGABRT)`, ships the binary unstripped, and writes a dyld crash report, while cargo still exits 0 (2026-09-16). `tools/cargo.sh` now exports the same list rustup uses; the rustc warning line is the evidence, because macOS throttles crash reports and their count under-reports aborts.

## Data And Claims Safety

- Never open live user store (`~/.abi/`) from tests/smokes. Use `ABI_WDBX_PATH=:memory:`, `ABI_WDBX_PERSIST=0`, or foundation temp paths / scratch DurableStore.
- Read `docs/contracts/external-claims-audit.mdx` before docs touching capabilities. No claims of prod FHE/AES/RBAC, multi-host sharding, K8s/H100, verified CUDA/Vulkan/ANE kernels, or benchmark numbers without source+test+artifact evidence here. WDBX secure/cluster are ref-grade/single-host demos. Compiled feature != executed path; always preserve `accelerated=false` fallback reporting.

## CI

- `.github/workflows/ci.yml`: self-hosted macOS ARM64 for same-repo trusted (push/PR from this repo); hosted macOS fallback only for fork PRs. Separate windows-latest for credential ACL tests (abi-foundation windows_acl/file).
- WDBX always checked out as sibling at exact `WDBX_REVISION` (update the const + verify full gate on the rev before PR).
- Billing-locked hosted jobs may refuse (steps=[], no logs, annotation about account lock) — treat self-hosted green as the signal; do not alter code for refused jobs.

<!-- machine-git-policy -->
## Git Workflow

Work only in the canonical checkout on default branch (`main`). Create branch/worktree only for required isolation (parallel risky work, or when asked); before done: merge back, rm worktree, delete branch label. Inspect `git status --short --branch` at start of changes and preserve unrelated dirty files. Conventional commits; no force-push on main. See `~/.claude/CLAUDE.md` for full.
<!-- /machine-git-policy -->
