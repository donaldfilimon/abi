# TODO — ABI Framework (Rust nightly)

Current status ledger for the Rust rewrite closeout, completed hardening slices,
and explicitly disclosed residuals. Detailed history also lives in
`RUST-REWRITE-PLAN.md`, `git log`, and `CHANGELOG.md`. Claims gate:
`docs/contracts/external-claims-audit.mdx`.

Status legend: `✅ Done` · `🟡 In progress` · `⚪ Not started` · `🔴 Blocked` · `◑ Partial / disclosed`

> Discipline: keep completion evidence compact and keep residual constraints
> explicit. Source and tests override prose. Gate: `./tools/check.sh`.

---

## Rewrite closeout

| Item | Status | Notes |
| ---- | ------ | ----- |
| Zig teardown | ✅ | 0 tracked `*.zig` / `build.zig*` |
| Frozen CLI (13) + MCP (12) | ✅ | Golden + unit coverage |
| FoundationModels shim | ✅ | `libabi_fm_shim.dylib` on arm64 macOS; honest offline |
| Local OpenAI bridge + MCP HTTP compatibility transport | ◑ | Loopback one-shot HTTP + endpoint-advertising SSE; not a persistent spec-conforming MCP 2024-11-05 SSE response channel |
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

## Active finish-all wave

| Item | Status | Acceptance boundary |
| ---- | ------ | ------------------- |
| Generate Bash/Zsh/Fish completions from live metadata | ✅ | Production generator uses `usage::COMMANDS`/`SHORTCUTS`; captured scripts remain independent byte-exact oracles and live byte comparisons pass. PowerShell and a new top-level flag remain frozen-surface changes, not implicit cleanup. |
| Wire the SEA eight-signal scorer into evidence recall | ✅ | Stable-ID deduplication, all current signals, task weights, deterministic budgets, 100-candidate public/defense-in-depth cap, indexed timestamps, provenance, raw routing input, prompt-byte cap, and scratch-store persona regression are covered. |
| Add TTY line editing and dashboard navigation | ✅ | Bounded Unicode-column-aware editor, history, Tab completion, single-stream output, Ctrl-C/D restoration, session-local SEA state, keyboard cycling, and bounded SGR mouse pane selection are covered. Capture enable/disable is guard-scoped; unit tests plus the dashboard and `tui` PTY drivers prove selection, exit, and cleanup. |
| Harden MCP transports found in code review | ✅ | 64 KiB physical-frame discard/recovery, absent-ID notification semantics, explicit-null validation, port-zero shutdown, hostile-Origin rejection, and exact HTTP 202/no-body behavior are covered without changing the frozen 12-tool catalog. |

---

## #647 Rust-rescoped (optional hardening)

See `docs/superpowers/archive/plans/2026-07-31-rust-647-followups.md`.

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
| Scratch-path hardening for the os-control audit tests | ✅ | 2026-08-01: the two tests built their scratch dir from `{pid}-{thread_id:?}`; both now use `abi_foundation::temp_path::temp_file_path()` (PID + per-process counter), which cannot collide with anything this process made earlier. Hygiene, kept on its own merits. |
| CI flake: `os::audit`/`os` scratch-store `WriterBusy` | ✅ | Fixed by the writer-lock retry in [#772](https://github.com/donaldfilimon/abi/pull/772), **not** by the row above. Measured head-to-head, same harness (`os::` at `--test-threads 8`, 40 runs each): scratch-path hardening alone still failed **3/40**; the writer-lock retry alone and both together failed **0/40**. The "leftover dir from a panicked run / `ThreadId` reuse" theory is not reachable — a dead process cannot hold an `flock` (the kernel releases it at exit) and `scratch_store` already called `remove_dir_all` first. The real holder is live and transient: a `fork` duplicates the lock's fd into the child until `exec` closes it via O_CLOEXEC, which is why renaming the path cannot help. |

**Disclosed, not fixed:** two `nn.rs` fixtures still use PID-only scratch
directory names. The earlier five-file inventory is no longer current:
`wdbx_simulate.rs`, `wdbx/mod.rs`, and `abi-wdbx/src/retrieval.rs` use
per-process counters, while `complete.rs` includes a wall-clock component. None
has been observed to flake; the two NN fixtures remain a bounded hygiene slice,
not evidence of the fixed writer-lock race.

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
| WDBX v2 causal multi-writer program | ◑ | `abi-wdbx::v2` now has legacy/UUID `RecordId`, per-writer append-only journals, hash-covered commit frames, causal conflict retention/explicit resolution, immutable `Arc` snapshots, 4,096-dimension validation/HNSW dense-slot mapping, and a 50-writer stress test. The versioned lifecycle provides non-mutating v1 reads plus lock-guarded, record-complete v1 migration into a verified sibling generation, byte-exact permanent backup, and atomic activation pointer; clean, WAL-ahead/torn-tail, legacy-snapshot, corrupt-source, and restart cases are covered. Transactions and compaction snapshots are independently authenticated objects with plaintext compatibility plus opt-in XChaCha20-Poly1305 encryption, Ed25519 signatures, pre-parse authentication, owner-only key files, and `wdbx db keygen`; tampering, wrong/missing/mismatched keys, nonce uniqueness, torn tails, causal segment coverage, and incomparable compactions are tested. Compaction retains all old objects; per-writer advisory leases keep ordinary writers lock-free while confirmation-gated rekey atomically verifies/activates a sibling generation and retains its predecessor. Confirmed generation-local GC publishes/re-verifies one covering segment, refuses live writers or uncovered published heads, removes only recognized dominated objects, and leaves prior generations/backups intact. `migration-status`, `migration-dry-run`, and `verify --require-signature` are read-only operator controls. Existing CLI/MCP data callers still use `DurableStore`, so product-wide cutover, platform credential adapters, codecs, FHE/accelerator wiring, gateway, and authenticated cluster proof remain open. |

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
