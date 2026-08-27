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
| `../wdbx/crates/abi-wdbx/src/multiway.rs` decomposition | ✅ | `evolve()`'s frontier-draining loop extracted to `process_frontier()`; elapsed-time bookkeeping now happens once instead of at 4 duplicated early-return sites. 12/12 multiway tests + full gate green. |
| `crates/abi-cli/src/wdbx.rs` split | ✅ | Thin `wdbx/mod.rs` + `db`/`block`/`query`/`cluster`/`api`/`benchmark`/`compute`/`secure`/`gpu`; goldens byte-identical. |
| `../wdbx/crates/abi-wdbx/src/format.rs` split | ✅ | Facade + `hash`/`record`/`segment`/`manifest`; 34 format tests green. |
| `../wdbx/crates/abi-wdbx/src/wal.rs` dedup | ✅ | Shared `Mutation` serde enum for all `append_*` frames. |
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

## Abbey runtime foundation train

| Item | Status | Acceptance boundary |
| ---- | ------ | ------------------- |
| Provider-neutral agent runtime contracts | ✅ | `abi-agent-runtime` owns model requests/events, tool descriptions, policy, audit, usage, budgets, cancellation, bounded capture, and deterministic providers. `RunContext::checkpoint` gives blocking adapters a read-only cancellation/budget polling boundary without synthetic events or access to cancellation authority. Evidence: 52 unit + 11 external contract tests and warning-denied private-item rustdoc. It deliberately performs **no tool execution** and **no model inference**. |
| Hash-verified model registry contracts | ✅ | `abi-models` owns validated immutable manifests, artifact hashing, external storage policy, resumable-download state, license acceptance, and registry resolution. Its original unsigned loaders remain for explicit local/test provenance; authenticated delivery uses the signed-envelope APIs below. |
| Agent host orchestration | ✅ | `abi-agent-host` composes the existing provider, registry, policy, and audit contracts with an object-safe `ToolExecutor`. Startup compiles full JSON Schemas with external resolution disabled; calls are resolved and validated before audited authorization. The continuation loop rejects duplicate IDs, unknown tools, malformed/schema-invalid JSON, provider-fabricated results, post-terminal events, oversized events/results, and recursive calls beyond finite event/output/tool/deadline/round/run budgets. Evidence: 5 unit + 11 external contract tests, warning-denied clippy, and warning-denied private-item rustdoc. |
| Registry delivery security | ✅ | `abi-models` ships a blocking Rustls HTTPS range transport with redirects disabled, exact `206 Content-Range` validation, body/artifact size ceilings, immutable-revision URLs, hash verification, fsynced partials, and atomic no-clobber publication. Ed25519 envelopes verify exact manifest bytes against an explicit publisher trust store before parsing; content hashes remain mandatory. License records bind accepting principal, license digest, model/revision, and the ordered artifact path/hash set. Evidence: 71 unit + 4 external API tests, warning-denied clippy, and warning-denied private-item rustdoc. No weights, datasets, or adapters are stored here. |
| Local model execution | ✅ | `abi-model-runtime` connects exact principal-bound `abi-models` resolution to `abi-agent-runtime::ModelProvider` through Tokenizers, Safetensors, Candle, and `abi-compute` evidence. Callers select an exact model and device; no default, model/device substitution, or fallback exists. Generated scratch fixtures prove deterministic CPU inference and locally exercised Metal initialization plus native tensor operations. Model, tokenizer, raw-prompt, context, generation, runtime-event, and cancellation bounds apply before their respective allocations or emissions. Load and inference reports keep requested, initialized, executed, fallback, mixed-path, and runtime-verification evidence separate. The first architecture is only the tiny `abi-bigram-v1` offline fixture; Gemma, tool-call generation, model data, CUDA runtime execution, placement profiling, and performance remain unclaimed. Evidence: 14 default external contracts, 14 Metal-feature contracts on macOS, warning-denied clippy, and warning-denied private-item rustdoc. CUDA source is feature-gated through Candle but local compilation is unavailable without `nvcc`. |
| Authenticated worker plane | ✅ | `abi-worker` adds typed worker identity/capability, signed audience-bound job and cancellation envelopes, manager-derived principal-scoped idempotency digests, finite leases/deadlines, cooperative cancellation, bounded ordered result chunks, independently verifiable result digests, quotas, replay resistance, health, and append-only job-control reconstruction. Transport configuration reuses the gateway's owner-protected mandatory-mTLS loader while job-control storage remains independent of WDBX data-plane authority. Evidence: 1 unit + 25 external contract tests, warning-denied clippy, warning-denied private-item rustdoc, and the complete workspace gate. This foundation does **not** provide a network listener, scheduler, model executor, production certificate lifecycle, production cluster, or separate-host proof. |

> This train is additive. It must not change ABI's frozen 13-command CLI or
> 12-tool MCP catalogs. Model weights, datasets, and generated adapters stay
> outside this repository.

---

## Abbey System federation

| Item | Status | Acceptance boundary |
| ---- | ------ | ------------------- |
| Federation reconciliation and P1 Abbey Contracts design | ✅ | Donald approved the C0 design on 2026-08-22. The constitutional P0-P7 slug map is fixed without Program 8; no production, deployment, Discord, or home-directory mutation is authorized by the document alone. |
| Program 1 Abbey contract source corpus | ✅ | ABI-local C1 source/contract evidence: 81 manifest-bound artifacts (86,945 bytes), aggregate digest `43d606a06d4bd9de08a651a984a61c611f9ffe0c8150b105b0cbf50c801f0fa7`, 73 Python repository/corpus behavior tests, independent Rust verification, four fail-closed temporary mutations, and the complete strict gate. Stable-Rust/Swift consumers, installed artifacts, hosted CI, runtime federation, durable WDBX behavior, authorization, production, and live Discord remain unperformed. |

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

**Disclosed, not fixed:** the same ad hoc `temp_dir().join(format!("..{pid}-{thread:?}"))` scratch-path pattern (not `temp_file_path`) also appears in `nn.rs`, `complete.rs`, `wdbx_simulate.rs`, `wdbx/mod.rs`, and `abi-wdbx/src/retrieval.rs`. None have been observed to flake; left untouched per "smallest verified slice" rather than swept into a broad rename.

---

## Docs hygiene

| Item | Status | Notes |
| ---- | ------ | ----- |
| Archive completed Zig-era `docs/superpowers/plans/*` | ✅ | Moved under `docs/superpowers/archive/plans/` with Archived banners |
| Executable dependency-security scan helper | ✅ | `tools/security/run-dep-scan.sh` runs installed `cargo-audit` or `cargo-deny`, otherwise emits an explicit SKIP; `ABI_DEP_SCAN_REQUIRE=1` makes missing tooling fail closed. The local RustSec scan has no vulnerability advisories after replacing the unmaintained PEM parser/server edge; two explicitly disclosed unmaintained transitive crates remain isolated behind optional TFHE-rs. |

---

## Disclosed residuals (do NOT fake-complete)

| Item | Status | Constraint |
| ---- | ------ | ---------- |
| Native accelerator execution | ◑ | Metal dot/cosine/norm/batch-cosine paths are locally scoped and only count after initialization plus CPU-oracle verification. CoreML loads and executes an output-checked tiny model under a `.cpuAndNeuralEngine` request; placement and ANE residency remain unverified. CUDA/Vulkan adapters compile/report capability but have no verified runtime execution. |
| External shader / MLIR toolchains | ◑ | Validation / textual IR only |
| Mobile `native_dispatch` | ◑ | Simulated desktop profile |
| Production FHE / multi-host sharding | ◑ | DGHV educational refresh and optional TFHE-rs demos are reference-scoped and have no independent cryptographic audit. Exact replication/read repair/rebalance and `cluster local-demo` are single-host multi-process proof, not production multi-host deployment. |
| Full ggml/llama.cpp | ◑ | Demo GGUF container only (char-LM payload) |
| WDBX v2 causal multi-writer program | ◑ | The local product path now includes causal per-writer journals, verified migration with retained backup, authenticated transaction/segment objects, retained-generation rekey, confirmation-gated GC, exact committed-transaction export/import, accelerator-backed batch search with deterministic CPU parity, and exact single-host replica/read-repair/rebalance proof. A 50-process crash/compaction stress test recovers every reported commit and surfaces all recovered conflicts. Deterministic versioned PQ and persisted-autoencoder artifacts are integrated into segment codecs with validation and quality metrics. `abi-compute` supplies cycle-free accelerator contracts and five-state evidence. Additive credential-provider evidence covers the existing macOS Keychain and Windows protected-file paths plus a target-gated, in-process Linux Secret Service implementation. Linux source and tests cross-compile, but no Linux daemon round trip is claimed and the default auth backend is unchanged. The integrated gateway supplies bounded authenticated gRPC plus metadata-only WebSocket events with local TLS/mTLS runtime tests. Remaining boundaries: production separate-host deployment, hosted/Windows/Linux runtime proof, DAST, and independent crypto/security review. |

---

## Recently closed product slices

| Item | Status | Notes |
| ---- | ------ | ----- |
| Live Discord/Twilio TLS clients | ✅ | rustls `wss://` via `abi-connectors::tls_ws`; offline process-local TLS peer tests |
| Metal DOT kernels | ✅ | `metal-kernels` feature; `accelerated=true` when init succeeds; CPU oracle test |
| Windows credential ACL CI | ✅ | PR #794 ran on Windows Server 2025 on 2026-08-19: 2 ACL tests and 7 credential-file tests passed. The test found and pinned the missing protected-DACL flag plus Windows' `AI` bookkeeping form; caller docs now point to that runtime proof. |
| True incremental NN sampler | ✅ | `SampleState::step` + demo GGUF load/sample |

---

## Recently landed

- Loopback browser studio: `abi agent browser --studio` plus `abi_ai::multimodal` image/video/audio/voice/fuse processors. Claim-honest local analysis only; Web Speech STT/TTS stay in the page. Browser plan output now matches `embedded_browser=false`.
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

---

## Residual reclassification against CSAPS (2026-08-22)

The seven `◑ Partial / disclosed` rows above were the only available definition
of "abi is not complete". Read against `CSAPS_WDBX_Revised_2026.pdf` — a
**proposed architecture**, not an obligation — several stop being gaps and
become the paper's own recommended posture. Section 9.1 states the design
principle plainly: *prove the architecture digitally first*, and treat
Loihi-class processors, physical reservoirs, and memristor arrays as **optional
research backends that must outperform matched digital emulation on an
end-to-end Pareto metric before adoption**. Section 2.5 says the same for
memristive substrates; section 2.3 says a physical reservoir "is an optimization
experiment, not an architectural dependency"; section 2.4 says "the baseline
architecture therefore requires no neuromorphic processor".

| Residual | Reclassified | Why |
| --- | --- | --- |
| Native accelerator execution (Metal verified against CPU oracle; CUDA/Vulkan compile-only; ANE residency unverified) | **Correct by design** | Section 9.1's stance exactly. Deterministic CPU parity is the acceptance test; unverified exotic placement is the honest state, and claiming it would violate R11's complete-system accounting. |
| External shader / MLIR toolchains, validation only | **Correct by design** | Same principle. No CSAPS requirement depends on textual IR being executed. |
| Mobile `native_dispatch` simulated on a desktop profile | **Correct by design, disclosed** | CSAPS-1 is explicitly a nonhazardous digital testbed on conventional programmable hardware. Mobile is not on its critical path. |
| Production FHE / multi-host sharding (DGHV educational, TFHE-rs demo, single-host multi-process cluster proof) | **Correct by design, with one real gap** | Reference-scoped crypto matches the paper's posture. The genuine gap is **independent cryptographic and security review**, which section 6.6's threat model needs and which no amount of local work supplies. |
| Full ggml/llama.cpp (demo GGUF container, char-LM payload) | **Out of scope, not a gap** | CSAPS never requires a production LLM. The deliberative path may call one; hosting it is not the substrate's job. |
| Local OpenAI bridge + MCP HTTP compatibility transport (not a persistent spec-conforming MCP 2024-11-05 SSE channel) | **Genuine gap, small** | Bounded, well understood, and belongs to Program 6 (Abbey API and Application Federation) rather than to the substrate. |
| WDBX v2 causal multi-writer program | **Genuine gap, large — and now measured** | See `docs/superpowers/specs/2026-08-22-wdbx-conformance-gap-analysis.md`. The row's own listed boundaries (separate-host deployment, hosted/Windows/Linux runtime proof, DAST, independent crypto review) remain accurate, and the analysis adds the substrate-contract gaps: canonical CBOR/COSE, sorted parent hashes, episode-level signing, regime and version fields, contradiction and quarantine edges, evidence-weighted retrieval, the selective write gate, and section 6.9 retention semantics. |

**Net effect.** Of seven disclosed residuals, four are the paper's recommended
engineering posture rather than shortfalls, one is out of scope, one is a small
gap owned by Program 6, and one is the large gap that Program 3 exists to close.
"Complete abi" is therefore a much smaller target than the row count suggested —
but it is not zero, and the two survivors are correctly disclosed rather than
quietly overstated.

**Not reclassified as done, and deliberately so:** independent cryptographic
review, independent security review, and DAST. Those cannot be self-certified.
They are `🔴 Blocked` on an external party, not `◑`.

### Aviva persona surface

Reviewed as part of the same pass. Aviva is one of the three routed personas in
`crates/abi-ai` (`soul.rs`, `identity.rs`, `router.rs`, `modulator.rs`,
`keywords.rs`), and `abi-ai` is **pure**: no WDBX dependency, no I/O, fully
deterministic, which is what makes `ai_run` byte-reproducible and
golden-testable. The extraction did not touch it, and it must stay that way —
under constitutional invariant A1 abi-ai owns persona routing and modulation
while Abbey owns persona definition, so introducing I/O or a substrate
dependency here would both break determinism and blur an authority boundary.
No change was required and none was made.
