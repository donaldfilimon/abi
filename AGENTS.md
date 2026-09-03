# AGENTS.md - abi

Canonical instructions for this nightly-Rust workspace. `opencode.json` also
loads `tasks/lessons.md` and `tasks/todo.md`; read those for the session checklist
and active work. Executable source (`Cargo.toml`, `tools/check.sh`, `crates/`)
wins over prose. `CLAUDE.md` is an expanded companion; `GEMINI.md` redirects here.

Brand: **Intelligence Without Limits.** IWL is Abbey/ABI only; Quesar (private AI ops)
never carries this tagline. See `docs/brand.md`.

## Toolchain And Gates

- Never run bare `cargo`. Homebrew's stable Cargo ignores
  `rust-toolchain.toml`, and Swiftly's `cc` shim can break linking.
  `./tools/cargo.sh` selects the pinned nightly toolchain and system compiler.
- Run `./tools/check.sh` after every edit. It executes policy tests, `./tools/cargo.sh xtask ci verify`
  (Rust port of `tools/ci_contract.py`; judo #817), Abbey-contract checks (Python
  oracle plus `./tools/cargo.sh xtask abbey verify`; Python remains authoritative until
  byte-identical), Rust size limits, fmt check, warning-denied clippy, build,
  workspace tests, platform feature checks, the local benchmark guard, and rustdoc
  in that order. Missing platform tooling is reported as a skip.
- `tools/check_rust_sizes.sh` rejects Rust files over 1,000 lines and
  `crates/abi-cli/src/main.rs` over 200 lines.
- `./build.sh check` and `./build.sh full-check` are compatibility aliases for
  `./tools/check.sh`.

## Focused Commands

- Build CLI: `./tools/cargo.sh build -p abi-cli` (`target/debug/abi`).
- CI/Abbey contract (xtask, judo #817): `./tools/cargo.sh xtask ci verify` ·
  `./tools/cargo.sh xtask abbey verify contracts/abbey`. Alias is in `.cargo/config.toml`.
- Build MCP: `./tools/cargo.sh build -p abi-mcp` (`target/debug/abi-mcp`).
- Unit test/filter: `./tools/cargo.sh test -p <crate> --lib -- <filter> < /dev/null`.
- Integration target: `./tools/cargo.sh test -p <crate> --test <name> < /dev/null`.
- Workspace tests: `./tools/cargo.sh test --workspace < /dev/null`.
- Format: `./tools/cargo.sh fmt --all`.
- Gate-equivalent lint:
  `./tools/cargo.sh clippy --workspace --all-targets -- -D warnings`.

Always redirect stdin from `/dev/null` for hand-run tests. The CLI auth tests
exercise a non-TTY secret read and can block indefinitely on inherited open
stdin. `tools/check.sh` already redirects its test invocations.

## Workspace Boundaries

- Live code is under `crates/*` (17 workspace members, including `xtask`).
  `crates/xtask` is the in-repo task runner, not a published product crate; it
  ports CI-contract and Abbey corpus/vendor checks from Python (judo #817).
  Removed Zig and rewrite scaffold trees are historical only.
  `crates/abi-cli/src/main.rs`, `crates/abi-mcp/src/main.rs`,
  and `crates/abi-wdbx-gateway/src/main.rs` are the executable entrypoints.
- `abi-compute`, `abi-core`, `abi-foundation`, `abi-telemetry`, and `abi-wdbx`
  are sibling path dependencies under `../wdbx/crates/`, not local workspace
  crates. Keep the repositories adjacent; do not recreate stale local copies.
- `abi-ai` owns deterministic persona identity/routing and must stay free of
  WDBX and I/O. Evidence retrieval and persistence belong in `abi-sea`,
  CLI/MCP integration, or the WDBX substrate.
- `abi-agent-runtime` defines provider-neutral contracts;
  `abi-agent-host` owns bounded tool orchestration. `abi-models` owns registry,
  artifact, and license contracts; `abi-model-runtime` owns model execution.
- `abi-cli` owns command metadata and process dispatch. `abi-mcp` owns JSON-RPC
  stdio and its custom loopback HTTP compatibility listener. Launch MCP through
  `mcp/launcher.sh`; it resolves the built binary and arm64 macOS shim, and
  `ABI_MCP_AUTO_BUILD=1` permits an on-demand build.
- Canonical `ABI_*` names and environment access live in
  `../wdbx/crates/abi-foundation/src/env.rs`. Tests must use its override and
  locking hooks instead of mutating process environment directly.

## Contract Surfaces

- The 13 top-level CLI commands are defined by
  `crates/abi-cli/src/usage.rs`; the 12 MCP tools are defined by
  `crates/abi-mcp/src/handlers.rs`. Treat both catalogs and their ordering as
  frozen unless a deliberate contract change updates source, tests, and fixtures.
- `tests/golden/` pins CLI/MCP output, completions, and persisted samples.
  Fixtures included with `include_str!` or `include_bytes!` require rebuilding
  the affected test target after edits.
- MCP stdio uses newline-delimited JSON-RPC with a 64 KiB physical-frame cap.
  The loopback `/sse` endpoint only advertises the message endpoint; it is not a
  persistent spec-conforming MCP HTTP+SSE response channel.

## Data And Claims Safety

- `~/.abi/` is the user's live store. Tests and smoke runs must use scratch
  `DurableStore` paths, `ABI_WDBX_PATH=:memory:`, or `ABI_WDBX_PERSIST=0`.
- Do not claim production FHE/AES/RBAC, production multi-host deployment,
  QPS/latency/accuracy, Kubernetes/H100, CUDA/Vulkan runtime execution, or ANE
  residency without current evidence. WDBX secure/cluster demos are
  reference-grade or single-host. See
  `docs/contracts/external-claims-audit.mdx` before capability documentation.
- A compiled backend is not runtime evidence. Preserve explicit fallback and
  `accelerated=false` reporting when native execution is not verified.

## CI And Git

- `.github/workflows/ci.yml` checks out a pinned `donaldfilimon/wdbx` revision
  beside ABI before running `./tools/check.sh`. Missing sibling `abi-compute` or
  `abi-wdbx` packages in CI usually indicate checkout/pin drift, not a manifest
  that should be redirected to local copies.
- Trusted same-repository events use the self-hosted macOS ARM64 gate; fork PRs
  use the hosted macOS fallback. A separate Windows job covers credential ACLs.
- Use Conventional Commits. Never force-push `main`.

<!-- machine-git-policy -->
## Git Workflow

Work in this canonical checkout on the default branch. Create a branch or
worktree only when isolation is genuinely required or explicitly requested.
Before completion, merge any such work back, remove the worktree, and delete the
branch. Preserve unrelated dirty work. Full policy: `~/.claude/CLAUDE.md`.
<!-- /machine-git-policy -->
