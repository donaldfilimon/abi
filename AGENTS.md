# AGENTS.md - abi

Canonical instructions; executable source wins over prose. `CLAUDE.md` is an
expanded companion; `GEMINI.md` redirects here. `opencode.json` also loads
`tasks/lessons.md` and `tasks/todo.md` for the session checklist and active work.

Brand: **Intelligence Without Limits.** IWL is Abbey/ABI only; Quesar (private AI ops)
never carries this tagline. See `docs/brand.md`.

## Toolchain And Gates

- Never run bare `cargo`: Homebrew binaries can bypass `rust-toolchain.toml`,
  and Swiftly's `cc` shim can break linking.
  `./tools/cargo.sh` selects the pinned nightly toolchain and system compiler.
- Code gate: `./tools/check.sh`. Order: Python policy tests, xtask CI verify,
  Abbey corpus (Python oracle then Rust verifier), size limits, fmt, clippy
  (`-D warnings`), build/tests, platform features, local benchmark guard, rustdoc.
  Metal tests/docs run on Darwin; CUDA compilation requires `nvcc`. Skips are
  not runtime evidence. Python remains the Abbey oracle until byte-identical.
- Documentation-only: `python3 -m unittest discover -s tools/tests -p 'test_docs*.py'`
  and `git diff --check`, plus source comparison. For `docs/` changes also run
  `.agents/skills/docs-validate/validate.sh`; load `abi-doc-claims-sync` first.
- `tools/check_rust_sizes.sh` rejects Rust files over 1,000 lines and
  `crates/abi-cli/src/main.rs` over 200 lines.
- `./build.sh check` / `full-check` alias `./tools/check.sh`.
- Build: `./tools/cargo.sh build -p abi-cli` or `-p abi-mcp`;
  binaries are `target/debug/abi` and `target/debug/abi-mcp`.
- CI/Abbey contract: `./tools/cargo.sh xtask ci verify` and
  `./tools/cargo.sh xtask abbey verify contracts/abbey`. Alias is in `.cargo/config.toml`.
- Unit test/filter: `./tools/cargo.sh test -p <crate> --lib -- <filter> < /dev/null`.
- Integration target: `./tools/cargo.sh test -p <crate> --test <name> < /dev/null`.

Always redirect stdin from `/dev/null` for hand-run tests. The CLI auth tests
exercise a non-TTY secret read and can block indefinitely on inherited open
stdin. `tools/check.sh` already redirects its test invocations.

## Workspace Boundaries

- Live Rust is under `crates/*`; `xtask` is the unpublished CI/corpus task runner.
  Removed Zig and rewrite scaffold trees are historical only.
- `abi-compute`, `abi-core`, `abi-foundation`, `abi-telemetry`, and `abi-wdbx`
  are sibling path dependencies under `../wdbx/crates/`, not local workspace
  crates. Keep repositories adjacent; do not recreate local copies or replace
  path dependencies with git sources (that splits Cargo crate identity).
- `abi-ai` has no WDBX dependency. Its routing/completion core is deterministic,
  but `file_context` and dataset inspection perform I/O. Store retrieval
  and persistence belong in `abi-sea`, CLI/MCP integration, or WDBX.
- `abi-agent-runtime` defines provider-neutral contracts;
  `abi-agent-host` owns bounded tool orchestration. `abi-models` owns registry,
  artifact, and license contracts; `abi-model-runtime` owns model execution.
- Launch MCP through `mcp/launcher.sh`: release binary wins over debug, so a
  stale release can hide a debug rebuild. `ABI_MCP_AUTO_BUILD=1` builds only when
  neither binary exists, not when sources change.
- Canonical `ABI_*` names and environment access live in
  `../wdbx/crates/abi-foundation/src/env.rs`. Tests must use its override and
  locking hooks instead of mutating process environment directly.
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
- MCP `connector_test` is deterministic/local, not live network dispatch.

## Data And Claims Safety

- `~/.abi/` is the user's live store. Tests and smoke runs must use scratch
  `DurableStore` paths, `ABI_WDBX_PATH=:memory:`, or `ABI_WDBX_PERSIST=0`.
- Before capability docs, read `docs/contracts/external-claims-audit.mdx`.
  Production FHE/AES/RBAC, multi-host deployment, benchmarks, Kubernetes/H100,
  CUDA/Vulkan execution, and ANE residency require current evidence.
  WDBX secure/cluster demos are reference-grade or single-host.
- A compiled backend is not runtime evidence. Preserve explicit fallback and
  `accelerated=false` reporting when native execution is not verified.

## CI

- `.github/workflows/ci.yml` checks out WDBX beside ABI at `WDBX_REVISION`.
  Missing substrate packages call for checking that checkout/pin, not new copies.
- Trusted same-repository events use the self-hosted macOS ARM64 gate; fork PRs
  use the hosted macOS fallback. A separate Windows job covers credential ACLs.

<!-- machine-git-policy -->
## Git Workflow

Work in this canonical checkout on the default branch. Create a branch or
worktree only when isolation is genuinely required or explicitly requested.
Before completion, merge any such work back, remove the worktree, and delete the
branch. Conventional Commits; never force-push `main`. Full policy: `~/.claude/CLAUDE.md`.
<!-- /machine-git-policy -->
