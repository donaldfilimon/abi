# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Note: `AGENTS.md` (and its twin `GEMINI.md`) is this repo's normal canonical
> instruction file, kept deliberately thin to avoid drift across the three. This
> file is an expanded version for Claude Code. If anything here disagrees with
> `AGENTS.md`, `AGENTS.md` wins; if either disagrees with executable source
> (`Cargo.toml`, `tools/check.sh`, `crates/`), trust the source.

ABI is a **nightly Rust** framework for local AI service orchestration, semantic
vector storage (WDBX), claim-honest GPU capability reporting, and an MCP server.
The Zig implementation that used to live under `src/` has been fully replaced;
see `RUST-REWRITE-PLAN.md` for the port history if you find stale Zig references
in older docs.

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
| `./tools/check.sh` | **Primary gate.** fmt check, clippy `-D warnings`, workspace build + tests, doc build. Run this before considering any change done. |
| `./build.sh check` | Thin compat wrapper → `./tools/check.sh` |
| `./tools/cargo.sh build -p abi-cli` | Build `target/debug/abi` |
| `./tools/cargo.sh build -p abi-mcp` | Build `target/debug/abi-mcp` |
| `./tools/cargo.sh test -p <crate> --lib -- <filter>` | Focused unit tests, e.g. `./tools/cargo.sh test -p abi-wdbx --lib -- wal::` |
| `./tools/cargo.sh test --workspace` | Full test suite (also run by `check.sh`) |
| `./tools/cargo.sh fmt --all` | Apply rustfmt |
| `./tools/cargo.sh clippy --workspace --all-targets -- -D warnings` | Lint, matching the gate exactly |
| `./mcp/launcher.sh` | Launch the MCP server; prefers `target/release/abi-mcp` then `target/debug/abi-mcp`; run from repo root (or via the launcher) so `@loader_path` resolves `libabi_fm_shim.dylib` on arm64 macOS; set `ABI_MCP_AUTO_BUILD=1` to build on demand |

There is no separate lint-only or build-only CI — `./tools/check.sh` **is** CI
(GitHub Actions is billing-locked on this repo, so it's also the only gate that
actually runs). Treat a red `check.sh` as blocking.

### Local smoke walkthrough

After building `abi-cli`, exercise real command surfaces without live network
credentials — useful for verifying a change beyond what unit tests cover:

```bash
ABI=./target/debug/abi
$ABI backends
$ABI scheduler status
$ABI dashboard --once --plain
$ABI complete "summarize ABI scheduler status"
$ABI agent plan "stage a safe WDBX refactor"
$ABI wdbx stats
```

## Architecture

Cargo workspace, one crate per concern, under `crates/*`. Dependency direction
runs roughly top-to-bottom (later crates depend on earlier ones):

| Crate | Role |
|---|---|
| `abi-foundation` | Shared primitives (errors, env, time, validation, JSON, logging). No dependency on any other ABI crate — everything builds on this. |
| `abi-core` | Config, task scheduler, memory accounting, plugin registry. Depends only on `abi-foundation`. |
| `abi-telemetry` | Bounded, process-wide counters; insertion order preserved because CLI Prometheus exposition is a captured compatibility surface. |
| `abi-ai` | Persona identity, routing (Abbey/Aviva/Abi), generation, governance/constitution. **Pure**: no WDBX dependency, no I/O, fully deterministic — this is what makes `ai_run` byte-reproducible and golden-testable. |
| `abi-sea` | SEA (Sparse Evidence Attention) self-learning loop: recalls prior WDBX records relevant to an input, prepends them as context, runs adaptive completion, updates persona-router weights. |
| `abi-nn` | Tiny character-level neural-net demo trainer — explicitly **not** a production LLM, not distributed. |
| `abi-gpu` | Claim-honest GPU/accelerator backend detection. Metal preferred on macOS; `accelerated=true` only when the optional `metal-kernels` feature actually links and initializes a Metal DOT pipeline, otherwise deterministic CPU SIMD fallback with `accelerated=false`. Also hosts claim-honest shaders/MLIR/mobile report surfaces. |
| `abi-wdbx` | The vector store: on-disk format, checkpoint publication/salvage, CRC-framed WAL recovery, exact-search + layered HNSW index, durable store integration, loopback REST, reference-scoped cluster protocol, 3-D spatial index, and reference quantization/Huffman/rANS/autoencoder codecs. This is the largest and most contract-sensitive crate (~14k LOC). |
| `abi-connectors` | External-service connectors (OpenAI, Anthropic, Grok, Discord, Twilio) built around a `Transport` trait. Every connector has a local and a live transport — see "The local/live split" below. |
| `abi-plugins` | The 16 bundled plugins plus the plugin manager. Each plugin ships as a compiled-in `mod.rs`/`stub.rs` pair under `crates/abi-plugins/plugins/`, checked with `assert_plugin_parity!`. `abi plugin run` and the MCP `plugin_run` tool dispatch through the same `PluginManager` over the same `BUNDLED` table. |
| `abi-cli` (bin `abi`) | Command metadata, help rendering, process dispatch. The help surface is a stable, golden-tested contract boundary. |
| `abi-mcp` (bin `abi-mcp`) | JSON-RPC MCP server: the frozen 12-tool surface over stdio (primary, 64 KB frame cap) plus optional loopback-only HTTP/SSE. |

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
  scripts for bash/zsh/fish/PowerShell). Changing frozen-surface output means
  updating the corresponding golden file deliberately, not incidentally.

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
  `tasks/todo.md` for current priorities.

## Code quality hotspots

These files exceed 1000 lines and warrant decomposing before large edits
rather than growing further in place:

| File | Lines | Notes |
|---|---|---|
| `crates/abi-wdbx/src/multiway.rs` | ~1738 | `evolve()` is a 52-line loop with 13 exits; manual JSON serialization throughout |
| `crates/abi-cli/src/wdbx.rs` | ~1476 | 41 free functions in one flat file; natural split is `db.rs`/`block.rs`/`query.rs`/`cluster.rs` |
| `crates/abi-wdbx/src/format.rs` | ~1248 | Four domain types (Hash, Record, Segment, Manifest) coiled into one file |
| `crates/abi-wdbx/src/wal.rs` | ~1102 | Seven `append_*` functions each redefine an identical `#[derive(Serialize)] struct Mutation` |
| `crates/abi-cli/src/agent.rs` | ~1052 | `open_store()` already extracted to `crate::util` (was duplicated with `complete.rs`) |
| `crates/abi-cli/src/complete.rs` | ~856 | 154-line `run()` mixes arg-parse, validation, and dispatch |
