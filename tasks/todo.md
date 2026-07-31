# TODO — ABI Framework (Rust nightly)

Forward-looking tracker for **incomplete and in-flight** work after the Zig →
Rust rewrite. Completed rewrite history: `RUST-REWRITE-PLAN.md`, `git log`,
`CHANGELOG.md`. Claims gate: `docs/contracts/external-claims-audit.mdx`.

Status legend: `✅ Done` · `🟡 In progress` · `⚪ Not started` · `🔴 Blocked` · `◑ Partial / disclosed`

> Discipline: no Session Summary narratives here. When an item closes, delete
> its row. Source and tests override prose. Gate: `./tools/check.sh`.

---

## Rewrite closeout

| Item | Status | Notes |
| ---- | ------ | ----- |
| Zig teardown | ✅ | 0 tracked `*.zig` / `build.zig*` |
| Frozen CLI (13) + MCP (12) | ✅ | Golden + unit coverage |
| FoundationModels shim | ✅ | `libabi_fm_shim.dylib` on arm64 macOS; honest offline |
| Local OpenAI bridge + MCP HTTP/SSE | ✅ | Loopback; fallback when bridge unusable |
| Land `rust-rewrite` on `main` | ✅ | Squash-merged [#756](https://github.com/donaldfilimon/abi/pull/756) as `34c35d5` |

---

## Cleanup / refactor backlog (goal: "Cleanup and refactor Rust 2024 nightly abi codebase")

| Item | Status | Notes |
| ---- | ------ | ----- |
| `crates/abi-wdbx/src/multiway.rs` decomposition | ✅ | `evolve()`'s frontier-draining loop extracted to `process_frontier()`; elapsed-time bookkeeping now happens once instead of at 4 duplicated early-return sites. 12/12 multiway tests + full gate green. |
| `crates/abi-cli/src/wdbx.rs` split | ⚪ | ~1476 lines / 41 free fns → `db.rs`, `block.rs`, `query.rs`, `cluster.rs` |
| `crates/abi-wdbx/src/format.rs` split | ⚪ | ~1248 lines; separate Hash/Record/Segment/Manifest into own modules |
| `crates/abi-wdbx/src/wal.rs` dedup | ⚪ | ~1102 lines; 7 `append_*` fns each redefine an identical `Mutation` struct — factor to one shared type |
| `crates/abi-cli/src/agent.rs` decomposition | ⚪ | ~1052 lines |
| `crates/abi-cli/src/complete.rs` split | ⚪ | ~856 lines; 154-line `run()` mixes arg-parse, validation, dispatch — split into stages |

> Each row: pure refactor, no behavior change. `./tools/check.sh` green + golden fixtures byte-identical before marking ✅.

---

## Disclosed residuals (do NOT fake-complete)

| Item | Status | Constraint |
| ---- | ------ | ---------- |
| Native GPU kernels (CUDA/Vulkan) | ◑ | Not linked; Metal DOT is optional |
| External shader / MLIR toolchains | ◑ | Validation / textual IR only |
| Mobile `native_dispatch` | ◑ | Simulated desktop profile |
| Production FHE / multi-host sharding | ◑ | Reference demos / ops guidance only |
| Full ggml/llama.cpp | ◑ | Demo GGUF container only (char-LM payload) |

---

## Recently closed product slices

| Item | Status | Notes |
| ---- | ------ | ----- |
| Live Discord/Twilio TLS clients | ✅ | rustls `wss://` via `abi-connectors::tls_ws`; offline process-local TLS peer tests |
| Metal DOT kernels | ✅ | `metal-kernels` feature; `accelerated=true` when init succeeds; CPU oracle test |
| Windows credential ACL CI | ✅ | `cfg(windows)` tests + `windows-acl` job on `windows-latest` |
| True incremental NN sampler | ✅ | `SampleState::step` + demo GGUF load/sample |

---

## Recently landed

- Product residuals: Discord/Twilio TLS WS, Metal DOT kernels, Windows ACL CI, incremental NN + demo GGUF
- Aggressive residual teardown: deleted `modernized/`, `modern-refactor/`, `zig-pin`, `zig-newest-skills`, `zig-build-doctor`, Zig-only cross-compile skill; rewrote `abi` agents + lessons + docs hub for nightly Rust; freed local `zig-out`/`.zig-cache`
- Agent skill drivers ported off `zig-out`/`./build.sh cli` → `./tools/cargo.sh` + `target/debug/abi` (dashboard, backends, plugins, SEA scratch store, WDBX roundtrip, etc.)
- nn demo JSON checkpoint (`--out` / `--checkpoint`); Rust smoke scripts + goals/run-abi/mcp-smoke skills
- **Rust rewrite on `main`** via [#756](https://github.com/donaldfilimon/abi/pull/756) (`34c35d5`)
- FoundationModels Swift `@c` shim + `complete --live --model apple-fm --confirm`
- shaders / mlir / hash / metrics / mobile report surfaces
- local_bridge + MCP HTTP/SSE; wdbx_stats open-failure disclosure for CI
- Discord gateway/routing/WS framing (offline); Twilio ConversationRelay local path
- Zig one-shot teardown
