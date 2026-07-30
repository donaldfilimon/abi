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

## Sequencing: delete as you go

Each vertical is ported *and* its Zig deleted in the same commit, so
"remaining Zig" only ever decreases and any interruption leaves a coherent
tree. Zig `foundation`/`core` files delete last, when their final Zig consumer
does.

- [ ] **0. Workspace** — cargo workspace, nightly pin, wrapper scripts, gate. Additive.
- [ ] **1. `abi-foundation`** — env, errors, logger, json, io, http, credentials, keychain, validation, sync, temp_path, time, os, pool_allocator.
- [ ] **2. `abi-core`** — config, registry, task, scheduler, memory.
- [ ] **3. `abi-connectors`** — delete `src/connectors/`.
- [ ] **4. `abi-wdbx`** — delete `src/features/wdbx/`.
- [ ] **5. `abi-ai` + `abi-sea` + `abi-nn`** — delete those three feature dirs.
- [ ] **6. `abi-gpu` + small features** — gpu, accelerator, shaders, mlir, hash, metrics, telemetry, mobile, os_control.
- [ ] **7. `abi-tui`** — delete `src/features/tui/`.
- [ ] **8. `abi-cli`** — delete `src/cli/`, `src/main.zig`, `src/root.zig`.
- [ ] **9. `abi-mcp`** — delete `src/mcp/`.
- [ ] **10. `abi-plugins`** — delete `src/plugins/`, `src/plugin_registry.zig`.
- [ ] **11. Zig teardown** — `build.zig`, `build.zig.zon`, `build.sh`, `.zigversion`, `zig-out/`, `zig-cache/`, `.zig-cache/`, `tools/*.zig`, `tests/**`, `examples/**`, `.gitattributes` Zig rules, `.github/workflows` calling `./build.sh`.
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
