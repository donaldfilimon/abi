# ABI Rust Rewrite — Port Plan

Goal: replace the entire Zig implementation of `abi` with Rust (nightly), and
delete every Zig source file, build script, and toolchain pin from the repo.

Branch: `rust-rewrite`.

## Scope (measured, not estimated)

Only git-tracked Zig counts. The ~2,100 `.zig` files under `.claude/worktrees/`
are git-ignored agent scratch worktrees and are not project source.

```
git ls-files '*.zig' | wc -l        # 302 files
git ls-files '*.zig' | xargs wc -l  # 61,500 LOC
```

| Area | Zig LOC | Notes |
|---|---:|---|
| `src/features/wdbx/` | 12,841 | vector store: HNSW, MVCC, WAL, REST, cluster |
| `src/features/ai/` | 7,764 | router, personas, constitution |
| `src/cli/` | 7,488 | registry + 25 handlers; **frozen surface** |
| `src/features/tui/` | 5,461 | dashboard / diagnostics render loop |
| `src/connectors/` | 4,633 | OpenAI, Anthropic, Grok, Discord, Twilio, HTTP |
| `src/foundation/` | 3,741 | env, creds, io, http, logger, json |
| `src/features/gpu/` | 3,437 | Metal/CUDA/Vulkan/WebGPU detect + CPU fallback |
| `src/mcp/` | 2,507 | JSON-RPC server; **frozen 12-tool contract** |
| `src/features/sea/` | 1,608 | sparse evidence attention learn loop |
| `src/core/` | 1,548 | config, scheduler, memory, registry, task |
| `src/plugins/` | 1,264 | 14 plugins, each `mod.zig` + `stub.zig` |
| `src/features/nn/` | 1,205 | small neural net |
| other `src/features/*` | ~2,990 | os_control, mobile, telemetry, shaders, accelerator, mlir, metrics, hash |
| root/tests/tools/examples | ~4,000 | `main.zig`, `root.zig`, contracts, benchmarks |

## Toolchain (a trap — read this)

`/opt/homebrew/bin/cargo` is Homebrew **stable** 1.97.1 and shadows rustup on
PATH. Because Homebrew ships real binaries rather than rustup shims,
`rust-toolchain.toml` is **silently ignored** — a nightly-only feature would
compile for whoever has the right PATH and fail everywhere else. This is the
same failure shape as the existing `brew-zig-shadows-zvm` note.

Every build must go through rustup explicitly:

```
rustup run nightly cargo …    # cargo 1.99.0-nightly / rustc 1.99.0-nightly
```

`tools/cargo.sh` wraps this; `tools/check.sh` is the replacement gate. Do not
call bare `cargo`.

## Sequencing: port additively, delete once at the end

**This reverses an earlier plan to delete each surface's Zig as it was ported.
Do not re-adopt that; it does not work here.** `build.zig` wires the whole
module graph through `src/root.zig`, and the test steps (`test-contracts`,
`test-feature-contracts`, `check-parity`) walk all of it, so deleting any one
directory breaks `zig build check` — and it stays broken until the last
directory goes. From that first deletion onward there would be no running Zig
implementation to compare against.

That oracle is worth more than a monotonically decreasing Zig count. So:

- **Steps 2–10 are purely additive.** The Zig tree stays intact and
  `./build.sh check` stays green, so behaviour can be diffed against a working
  implementation at any point and every commit leaves a shippable repo.
- **Step 11 deletes all Zig in one commit**, once `tools/check.sh` covers
  everything the Zig gate did — including the golden fixtures below.

### Golden fixtures — capture before deleting anything

Captured at commit `919dad8` while the Zig gate was green; see
`tests/golden/README.md`. They convert "the Rust CLI has the right command
names" into "the Rust CLI emits byte-identical output".

- `tests/golden/help.json` — the full 18 KB frozen CLI surface
- `tests/golden/help.txt`, `help-<command>.txt` × 13
- `tests/golden/completion.{bash,zsh,fish}`
- `tests/golden/mcp-initialize.json`, `mcp-tools-list.json` — the 12 tools with
  full input schemas, **in emitted order** (which is not declaration order:
  `wdbx_stats` comes after `plugin_list`)
- `tests/golden/wdbx-format.md` + synthetic `wdbx-sample.*` fixtures
- `tests/golden/mcp-tool-calls.jsonl` — all 12 tools with empty arguments; the
  nine validation error strings are contract too
- `tests/golden/mcp-tool-calls-args.jsonl` — the success paths
- `tests/golden/wdbx-db-verify.txt`, `wdbx-stats.txt`, `backends.txt`,
  `plugin-list.txt`, `scheduler-status.txt`, `mcp-scheduler-calls.jsonl`

- [x] **0. Workspace** — cargo workspace, nightly pin, wrapper scripts, gate.
- [x] **1. `abi-foundation`** — errors, env, time, temp_path, json, validation, logger, io, text, credentials (+file/keychain/secret/Windows ACL), http, system, plugin_manifest. 155 tests.
- [x] **1b. Golden fixtures** — captured while the Zig gate was green.
- [x] **2. `abi-core`** — config, registry, task, scheduler, memory. Concurrency decided: **one-shot**, tasks run synchronously on the caller's thread (`mode=one-shot`, `running=0` at rest). Golden-tested against the captured `scheduler_stats` output. 79 tests.
- [x] **3a. `abi-connectors` core** — connector types, URL/auth (HTTPS enforcement + host-boundary check), payload builders + byte-exact local synthesis, SSE parsing (both dialects), `Transport` trait with `ureq` live impl + `RecordingTransport`, clients for OpenAI/Anthropic/Grok/Discord/Twilio. 75 tests.
- [ ] **3b. `abi-connectors` remainder** — Discord gateway + WS client (`discord_gateway.zig` 483, `discord_ws_client.zig` 228, `discord_routing.zig` 126), Twilio relay (`twilio_relay.zig` 554), the local bridge (`local_bridge.zig` 236) and the Apple `FoundationModels` shim (`fm.zig` 217). These need a WebSocket client and a Swift FFI shim; ~1.8k Zig LOC.
- [x] **4a. `abi-wdbx` on-disk format** — records (all 6 types), both hash encodings, manifest, checkpoint load, chain verification. **Verified against the user's real 301-epoch store**: 327 blocks, chain verifies from genesis, 32-dim vectors. 56 tests.
- [x] **4b. WDBX checkpoint salvage** — descending newest-valid recovery
  skips missing/corrupt active epochs without merging full checkpoints or
  duplicating the block chain; total corruption remains a loud error.
- [x] **4c. WDBX checkpoint writer** — all six record types, invariant
  validation, SHA-256 trailer verification, atomic segment/manifest publication,
  and multi-epoch round trips.
- [x] **4d. WDBX WAL/recovery core** — CRC32-framed append, bounded verify,
  torn-tail handling, absolute vector continuity, deterministic block replay,
  epoch-gated checkpoint merge/stale-WAL discard, and checkpoint reset.
- [ ] **4e. WDBX algorithms/services** — HNSW index + storage, richer MVCC,
  cluster surfaces, compression/entropy/neural-compress,
  FHE + crypto_he demos, spatial 3-D octree, temporal graph, multiway engine,
  ANS, retrieval, remote compute. A deterministic exact cosine index provides
  the correctness oracle, and the layered HNSW graph/storage/search core with
  rollback journaling is now ported. A durable store facade now joins recovery,
  WAL-backed mutations, checkpoints and HNSW search. Manifest-authoritative
  retain-latest compaction and reset are also ported. The deterministic
  in-process Raft-style election/replication/failover core is ported; its TCP
  RPC transport remains open. The loopback-only REST
  surface is ported with its five routes, hybrid temporal/causal/persona
  re-ranking, optional fixed-work bearer authentication, 64 KiB request bound,
  failed-auth-aware token bucket, and real-TCP tests. Full MVCC remains outside
  the current Zig claim boundary; the remaining listed services stay open.
- [ ] **5. `abi-ai` + `abi-sea` + `abi-nn`**
- [ ] **6. `abi-gpu` + small features** — gpu, accelerator, shaders, mlir, hash, metrics, telemetry, mobile, os_control.
- [ ] **7. `abi-tui`**
- [x] **8a. `abi-cli` contract model** — frozen 13-command metadata,
  top-level help, shortcut resolution, and argument-free command help are
  golden-tested. This does **not** claim handler or full typed-help parity.
- [ ] **8b. `abi-cli` executable** — typed/raw dispatch, all command handlers,
  full `help.json` / `help-*.txt`, and `completion.*` parity.
- [ ] **9. `abi-mcp`** — golden-tested against `mcp-tools-list.json`, order included.
- [ ] **10. `abi-plugins`**
- [ ] **11. Zig teardown, in one commit** — `src/**/*.zig`, `build.zig`, `build.zig.zon`, `build.sh`, `.zigversion`, `zig-out/`, `zig-cache/`, `.zig-cache/`, `tools/*.zig`, `tests/**/*.zig`, `examples/**`, `.gitattributes` Zig rules, `.github/workflows` calling `./build.sh`.
- [ ] **12. Docs + memory** — `CLAUDE.md`, `AGENTS.md`, `GEMINI.md` together (they must not drift); `README.md`, `CHANGELOG.md`, `docs/**`; the `abi/` row in `~/CLAUDE.md`; delete the now-false `zig-pin-path` and `brew-zig-shadows-zvm` memories.

## Frozen contracts the Rust side must satisfy

These separate "done" from "it compiles".

**MCP: exactly 12 tools, no more, no fewer** (`src/mcp/handlers.zig`):
`ai_run`, `ai_complete`, `ai_learn`, `ai_train`, `wdbx_query`, `wdbx_stats`,
`scheduler_stats`, `scheduler_info`, `connector_test`, `gpu_status`,
`plugin_list`, `plugin_run`. The contract test ports with the server.

**CLI: 13 top-level commands, metadata verbatim** (`src/cli/usage.zig`):
`help`, `complete`, `train`, `agent`, `backends`, `plugin`, `auth`, `twilio`,
`tui`, `dashboard`, `wdbx`, `scheduler`, `nn`. Plus: the `--tui` →
`tui` shortcut rewrite in `main.zig`, `help-json` output, and bash/zsh/fish
completion output. Name/usage/summary are deliberately isolated in `usage.zig`
so help renders without the rest of the graph — keep that separation.

**WDBX on-disk format.** `~/.abi/` holds live `wdbx.seg.*.jsonl`, a manifest,
and an index. The Rust store must read existing segments, or ship a documented
migration. Silently orphaning the user's local state is not acceptable.

**`std.Io` has no Rust analog.** Zig 0.17 threads an explicit `std.Io` through
`main` → dispatch → every handler. This is the one part that is not mechanical
translation: each module needs a decision. Default is blocking `std` I/O with
an injected writer/reader pair for testability; async only where a listener
genuinely needs it (WDBX REST, MCP HTTP).

**mod/stub parity** (`tools/check_parity.zig`, 14 plugins × `mod.zig` +
`stub.zig`) is a Zig-shaped invariant: a feature and its no-op stub must expose
identical APIs. In Rust the same guarantee comes from a trait plus a
compile-time impl check, so the parity *script* is dropped and the invariant is
preserved as a test. Recorded here so the drop is deliberate, not silent.

## Gate

`./build.sh check` (build + tests + lint + parity) is replaced by
`tools/check.sh`: `cargo fmt --check`, `cargo clippy -D warnings`,
`cargo build --workspace`, `cargo test --workspace`, contract tests. GitHub
Actions on this repo is billing-locked, so the gate is local-only and must be
run before every commit.
