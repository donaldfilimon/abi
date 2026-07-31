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
in older docs. All live code is under `crates/` — the surviving root `src/`
holds only a `plugins/.central-synced` marker, not source. Note that
`.github/copilot-instructions.md` has **not** been ported and still describes the
Zig build (`zig build`, `-Dfeat-*`, `src/features/`); ignore it.

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
| `./tools/cargo.sh test -p <crate> --lib -- <filter>` | Focused **unit** tests (in-module `#[cfg(test)]`), e.g. `./tools/cargo.sh test -p abi-wdbx --lib -- wal::` |
| `./tools/cargo.sh test -p <crate> --test <name>` | A single **integration** test target under `crates/<crate>/tests/`, e.g. `--test golden`. `--lib` cannot reach these. |
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

Cargo workspace, one crate per concern, under `crates/*`, listed in dependency
order — every crate depends only on crates above it. (Confirm with
`grep -oE '^abi-[a-z]+' crates/<crate>/Cargo.toml` rather than trusting prose.)

| Crate | Role |
|---|---|
| `abi-foundation` | Shared primitives (errors, env, time, validation, JSON, logging). No dependency on any other ABI crate — everything builds on this. |
| `abi-telemetry` | Bounded, process-wide counters; insertion order preserved because CLI Prometheus exposition is a captured compatibility surface. No ABI dependencies. |
| `abi-nn` | Tiny character-level neural-net demo trainer — explicitly **not** a production LLM, not distributed. No ABI dependencies. |
| `abi-core` | Config, task scheduler, memory accounting, plugin registry. Depends on `abi-foundation` + `abi-telemetry`. |
| `abi-connectors` | External-service connectors (OpenAI, Anthropic, Grok, Discord, Twilio) built around a `Transport` trait. Every connector has a local and a live transport — see "The local/live split" below. |
| `abi-ai` | Persona identity, routing (Abbey/Aviva/Abi), generation, governance/constitution, and the model catalog (`models.rs`, default `claude-fable-5`). **Pure**: no WDBX dependency, no I/O, fully deterministic — this is what makes `ai_run` byte-reproducible and golden-testable. |
| `abi-plugins` | The 16 bundled plugins plus the plugin manager. Each plugin ships as a compiled-in `mod.rs`/`stub.rs` pair under `crates/abi-plugins/plugins/`, checked with `assert_plugin_parity!`. `abi plugin run` and the MCP `plugin_run` tool dispatch through the same `PluginManager` over the same `BUNDLED` table. |
| `abi-wdbx` | The vector store: on-disk format, checkpoint publication/salvage, CRC-framed WAL recovery, exact-search + layered HNSW index, durable store integration, loopback REST, reference-scoped cluster protocol, 3-D spatial index, and reference quantization/Huffman/rANS/autoencoder codecs. This is the largest and most contract-sensitive crate (~14k LOC). |
| `abi-gpu` | Claim-honest GPU/accelerator backend detection — note it depends on `abi-wdbx`, not the reverse. Metal preferred on macOS; the `metal-kernels` feature is **on by default**, but `accelerated=true` additionally requires the Metal DOT pipeline to actually link and initialize at runtime — otherwise deterministic CPU SIMD fallback with `accelerated=false`. Also hosts claim-honest shaders/MLIR/mobile report surfaces. |
| `abi-sea` | SEA (Sparse Evidence Attention) self-learning loop: recalls prior WDBX records relevant to an input, prepends them as context, runs adaptive completion, updates persona-router weights. |
| `abi-cli` (bin `abi`) | Command metadata, help rendering, process dispatch. Depends on every crate above. The help surface is a stable, golden-tested contract boundary. |
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
  - `crates/abi-core/tests/golden_scheduler.rs`, `crates/abi-core/src/registry.rs`
    — scheduler status and the 16-plugin listing.

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

`crates/abi-foundation/src/env.rs` is the single registry for every `ABI_*` var —
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
| `ABI_MCP_HTTP_PORT` / `ABI_MCP_HTTP_TOKEN` | Loopback MCP HTTP/SSE port / bearer token (stdio stays tokenless) |
| `ABI_LLAMA_CPP_ENDPOINT` / `ABI_MLX_ENDPOINT` | Local inference endpoints |
| `ABI_MCP_AUTO_BUILD` | `mcp/launcher.sh` only — build the server on demand |

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
  `tasks/todo.md` for current priorities.

## Code quality hotspots

Re-measure before trusting this table — it goes stale as work lands
(`find crates -name '*.rs' -exec wc -l {} + | sort -rn | head`).

Over 1000 lines; decompose before adding to them:

| File | Lines | Notes |
|---|---|---|
| `crates/abi-wdbx/src/multiway.rs` | ~1743 | Manual JSON serialization throughout; the `evolve()` frontier loop has been extracted to a helper |
| `crates/abi-wdbx/src/wal.rs` | ~1057 | Seven `append_*` functions each redefine an identical `#[derive(Serialize)] struct Mutation` |

Watch list (850–1000 lines — not yet over the line, but the next candidates):
`crates/abi-cli/src/agent.rs` (~951), `crates/abi-wdbx/src/hnsw.rs` (~941),
`crates/abi-wdbx/src/store.rs` (~933), `crates/abi-cli/src/complete.rs` (~896),
`crates/abi-core/src/scheduler.rs` (~883).

Already split — don't recreate the flat versions: `crates/abi-cli/src/wdbx.rs` is
now the `wdbx/` module directory, and `crates/abi-wdbx/src/format.rs` is down to
~444 lines.
