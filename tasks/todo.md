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
| `crates/abi-cli/src/wdbx.rs` split | ✅ | Thin `wdbx/mod.rs` + `db`/`block`/`query`/`cluster`/`api`/`benchmark`/`compute`/`secure`/`gpu`; goldens byte-identical. |
| `crates/abi-wdbx/src/format.rs` split | ✅ | Facade + `hash`/`record`/`segment`/`manifest`; 34 format tests green. |
| `crates/abi-wdbx/src/wal.rs` dedup | ✅ | Shared `Mutation` serde enum for all `append_*` frames. |
| `crates/abi-cli/src/agent.rs` decomposition | ✅ | `util::open_store` + `os` module extracted, then the line-mode REPL moved to `repl.rs`. agent.rs 942 → 520 lines. |
| `crates/abi-cli/src/complete.rs` split | ✅ | `parse_complete_args` + `CompleteRequest` separated from dispatch. |

> Each row: pure refactor, no behavior change. `./tools/check.sh` green + golden fixtures byte-identical before marking ✅.

---

## #647 Rust-rescoped (optional hardening)

See `docs/superpowers/plans/2026-07-31-rust-647-followups.md`.

| Item | Status | Notes |
| ---- | ------ | ----- |
| Lock-across-I/O audit (REST/cluster/MCP) | ✅ | No lock held across TCP I/O; rate-limiter mutex is math-only |
| MCP malformed/empty bearer contracts | ✅ | `abi-mcp` HTTP tests cover empty/Basic/wrong-case/no-space |
| DurableStore concurrency regression test | ✅ | `DurableStore` holds a lifetime-scoped advisory writer lock; 50 concurrent opens return `WriterBusy`, drop releases the lock, and 50 real REST query → joined teardown → reopen/search lifecycles stay green. `open` also waits out a transient `WouldBlock` (50 ms budget, 1 ms steps) so the fork/exec window that duplicates the lock fd into a child cannot masquerade as contention; a genuinely held lock still reports `WriterBusy`. |
| Bench regression gate in `check.sh` | ✅ | `tools/bench_regress.sh`: live Rust HNSW insert/search workload, best p50 of 5, 25% local debug threshold; same OS/arch baseline required (other host classes disclose `SKIP`). Deterministic pass/fail hooks prove the comparator. |

---

## OS control (goal: "Improve OS control safety and flexibility")

| Item | Status | Notes |
| ---- | ------ | ----- |
| 30s execute timeout | ✅ | `exec_command_with_timeout`; stdout/stderr drained on threads so a chatty command cannot deadlock the pipe before the timeout |
| Env filtering | ✅ | `env_clear()` then re-add only non-`ABI_*` / `*SECRET*` / `*TOKEN*` / `*KEY*` / `*PASSWORD*` / `*CREDENTIAL*` |
| Dry-run accepts any command | ✅ | Read-only by design; emits `policy=allowed\|denied` + allowlist note. Execute stays gated |
| Configurable timeout | ✅ | `timeout_secs` in the policy file, bounded 1..=3600; default stays 30s |
| WDBX audit block for executed commands | ✅ | `os/audit.rs` appends vector + `os-cmd:<id>` KV + audit block per **executed** command (never dry-run), including killed timeouts (`timed_out=true`, exit 124). Store injected so tests use a scratch path. Intentional no-store and open/write failures are disclosed distinctly on the `[os-cmd]` line. |
| `~/.abi/os-policy.toml` | ✅ | `os/policy.rs`, strict TOML subset. **Narrow-only**: `allow` is intersected with the compiled `CEILING`, so the file can never grant a command the binary does not already permit. Unknown, duplicate, and malformed keys fail closed. Path overridable via `ABI_OS_POLICY`. |

---

## Docs hygiene

| Item | Status | Notes |
| ---- | ------ | ----- |
| Archive completed Zig-era `docs/superpowers/plans/*` | ✅ | Moved under `docs/superpowers/archive/plans/` with Archived banners |

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
| Windows credential ACL CI | ◑ | `cfg(windows)` tests are written and the `windows-acl` job is configured on `windows-latest`, but it has **never executed**: every GitHub-hosted job is refused at dispatch with *"The job was not started because your account is locked due to a billing issue."* (~3s, zero steps). The ACL behavior is therefore unproven at runtime on any host. Re-verify and restore ✅ only after a hosted run actually reports steps. |
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
