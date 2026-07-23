# TODO — ABI Framework

Forward-looking tracker for **incomplete and in-flight** work. Completed history lives in `git log` and `CHANGELOG.md`. Long-horizon map: `docs/spec/wdbx-north-star.mdx` (§2/§8 Current/Partial/Proposed). Claims gate: `docs/contracts/external-claims-audit.mdx`.

Status legend: `✅ Done` · `🟡 In progress` · `⚪ Not started` · `🔴 Blocked` · `◑ Partial / disclosed`

> Discipline: no Session Summary narratives here. When an item closes, delete its row (or one line under Recently landed). Source and tests override prose.

---

## Tracks A–G scoreboard (claim-honest)

Landed on `main` via [#676](https://github.com/donaldfilimon/abi/pull/676) (`cursor/tracks-a-g-claim-honest`). This table is the canonical post-merge board for those asks.

| Track | Ask | Status | Source of truth | Residual (do NOT fake-complete) |
| ----- | --- | ------ | --------------- | ------------------------------- |
| **A** | In-process streaming — real incremental local generator | ◑ Done for template path | `src/features/ai/incremental.zig`; TUI `stream=incremental` | Neural LM / ggml in-process sampler (needs embedded runtime). Post-hoc path remains labeled where used. |
| **B** | Metal native kernels for `vectorOps` hot paths (Zig+Metal only) | ◑ Partial | Metal fused cosine/dot/L2 + elementwise add/sub/max/min/div (`compute_api`) + unary scale/relu + multi-pass `reduce_sum_kernel` / `reduce_max_kernel` + demo-grade softmax (on-GPU max when available) | Broader kernels; CUDA/Vulkan. **ANE = non-goal**. |
| **C** | Windows credential ACL + secret zeroing; runtime verify CI-blocked | ◑ Partial | Win32 SDDL owner-only DACL on credential write; POSIX `secureZero` / `secureWipe`; macOS login keychain via `ABI_CREDENTIALS_BACKEND=keychain` (`src/foundation/keychain.zig`, Security.framework `SecItem*`) | Windows/Linux keychain still Proposed. **Windows runtime verification** needs a Windows host/runner (cross-smoke = compile-only). |
| **D** | Lossless ANS/order-1 demo next to Huffman — not SOTA | ✅ Demo landed | `src/features/wdbx/ans.zig` + `abi wdbx secure demo` | Production/SOTA learned codec (ANS/arithmetic/context-model at scale). |
| **E** | FHE deepen reference tests/docs — not audited | ✅ Reference deepened | `fhe.zig` (`REF_P_BITS` / `REF_NOISE_BITS` / `VERIFIED_MUL_DEPTH`) | External security audit + bootstrapped FHE. |
| **F** | Cluster TLS-fronting story + ops docs — not sharding | ✅ Ops honesty landed | `cluster_rpc` TOKEN/PEERS tests; non-loopback TLS-fronting note | Production multi-host, mTLS, dynamic membership, **sharding**. |
| **G** | Phase D first approved scaffold under `modernized/` | ✅ Minimal scaffold | `modernized/README.md` + `modernized/packages/*`; `tools/check_modernized_refs.sh` in `full-check` | Cutover / second build root. **Live code remains `src/`**. |

---

## Open work

### Honest stubs — keep disclosed, do NOT fake-complete

| Item | Status | Constraint |
| ---- | ------ | ---------- |
| `accelerator` backend dispatch | ◑ Selection report only | Native CUDA/NPU/TPU not linked; CPU SIMD + partial Metal path is real. |
| `shaders` validation | ◑ Validate + checksum only | No shader compiler/toolchain linked. |
| `mlir` lowering | ◑ Textual analyze/lower only | No external MLIR/LLVM toolchain linked. |
| `mobile` runtime profile | ◑ Profile reporting only | `native_dispatch=false`; no platform runtime. |

### Still Proposed (demos/partials exist; production forms do not)

| Item | Status | Gap to production |
| ---- | ------ | ----------------- |
| Native compute beyond Metal cosine | ◑ Partial | Metal fused cosine/dot/L2 + elementwise add/sub/max/min/div + unary scale/relu + multi-pass threadgroup reduce_sum/reduce_max + demo-grade softmax for HNSW/`vectorOps`/`compute_api`; CUDA/Vulkan/broader kernels not linked. ANE **out of scope**. |
| Production/SOTA learned compression | ◑ Partial / disclosed | Huffman (`entropy.zig`) + demo rANS/order-1 (`ans.zig`) + int8 + reference autoencoder — **not** SOTA. |
| Security-audited FHE | ⚪ Not started (reference only) | DGHV reference params deepened; **not** audited. |
| Non-loopback REST hardening | ◑ Partial / disclosed | Loopback + bearer + rate-limit + TLS env validation; native TLS not linked; needs threat review for external expose. |
| Multi-host cluster / sharding | ◑ Partial / disclosed | Real TCP Raft RPC + token/peers + TLS-fronting **ops guidance**; no sharding, no prod multi-host. |

### Feature-parity north-star (pinned to source)

Frozen surfaces: **13 CLI commands**, **12 MCP tools** — extend via flags/subcommands/plugins/slash-commands only.

| Item | Status | Gap / Notes |
| ---- | ------ | ------------ |
| Local llama-cpp / OpenAI-compat bridge | ◑ Landed (HTTP) | `local_bridge.zig`; env `ABI_LLAMA_CPP_ENDPOINT`. Not embedded ggml. SSE when server streams. |
| MLX bridge / on-device FM | ◑ Partial | HTTP `ABI_MLX_ENDPOINT`; Apple FM separate (`feat-foundationmodels`). ANE non-goal. |
| Codex/claude-code TUI parity | ◑ Partial | Line editor, slash-cmds (`/open` `/diff` `/commit` `/context` `/features` `/learn` `/live` `/pane` / sessions…), plugins, multi-turn, `/pane`, live Anthropic SSE, `stream=incremental` footers. Residual: neural LM sampler. |
| Streaming completion | ◑ Partial (improved) | Template: **during-generation** emit via `incremental.zig`. Bridge + Anthropic: SSE. Residual: ggml/neural sampler. |
| File-aware agent context | ✅ Landed (v2) | `@file` + fair-share budget + workspace tree + plan/multi git diff. |
| Zero-copy RankedNode / SearchResult views | ✅ Done | Borrowed `vector` views; lifetime ends at next store mutation. |
| Trainable PointNeuralNetwork + SoulLayout | ✅ Done | Wired through train + `--soul` / `--soul-alpha`. |
| PipelineTelemetry per-turn | ✅ Done | Ethics + neural + summaries. |
| Discord gateway loop | ✅ Done | WS Hello→Identify→heartbeat; TLS not linked. |

### Candidate next slices (real remaining work)

Prioritized after A–G. Do not promote to Done without source + tests + honest claims.

| Priority | Item | Status | Notes |
| -------- | ---- | ------ | ----- |
| 1 | Neural / ggml in-process sampler (or keep Partial forever) | ⚪ / disclosed | Only if embedding a real chunked local backend; otherwise leave A residual labeled. |
| 2 | Broader Metal GPU path (more kernels / reduce) | ◑ Improved | Fused cosine/dot/L2 + elementwise add/sub/max/min/div + unary scale/relu + multi-pass `reduce_sum_kernel` / `reduce_max_kernel` + demo-grade softmax (on-GPU max). Residual: more kernels / CUDA/ANE disclosed. |
| 3 | Windows **runtime** CI/job | 🔴 Blocked (no Windows runner) | ACL code exists for windows-gnu; execution verify blocked on host. |
| 4 | OS keychain credential storage | ◑ Partial | macOS login keychain via Security.framework SecItem (opt-in `ABI_CREDENTIALS_BACKEND=keychain`); off-macOS env request is disclosed and load/save fall back to file (Windows/Linux Proposed); default remains file-based; OS-provided at-rest protection only — not hardware-backed, not audited; runtime-verified on macOS host only. |
| 5 | Phase D cutover plan (HITL) | ◑ Plan landed | `docs/spec/phase-d-cutover-plan.mdx` — checklist only; cutover still needs explicit HITL + gates. Scaffold honesty gate: `tools/check_modernized_refs.sh` in `full-check` (stale `` `src/...` `` refs fail). |
| 6 | Non-loopback REST threat review + native TLS link decision | ◑ Docs landed | `docs/spec/non-loopback-rest-threat-review.mdx` — proxy TLS preferred; native TLS deferred; not a hardened-expose claim. |
| 7 | Cluster mTLS / membership (still not sharding) | ◑ Ops docs landed | `docs/spec/cluster-mtls-ops.mdx` — proxy mTLS preferred; dynamic membership + sharding stay Proposed. |
| 8 | Mark `check-parity` as a required status check (branch protection) | 🔴 Blocked (Actions billing lock) | Explicit `check-parity` / `check-parity-hosted` jobs added to `.github/workflows/ci.yml`; jobs currently die in ~2s under the account billing lock — do not trigger CI; flip branch protection once billing is unblocked. |

---

## Constraints & intentional non-goals

Do not schedule these as “complete”:

- **ANE execution** — CoreML/ObjC; excluded by 100% Zig. Detection only.
- **MCP HTTP ↔ WDBX REST framing duplication** — intentional (`src/mcp/` module-root isolation).
- **Never force-push `main`.**
- **No unbacked claims** — sharding / AES / RBAC / cert / QPS / latency / accuracy / audited FHE / SOTA compression.

---

## Known test failures

- None currently reproduced. Pin: `0.17.0-dev.1442+972627084`. Prefer `./build.sh check` (macOS) as the primary gate.

---

## Recently landed (short digest)

Full detail: `git log` + `CHANGELOG.md`. Keep this list short.

- **Production hardening wave** — incomplete HTTP reject; constant-time bearer; WAL fsync + torn-tail skip + parent-dir sync; putVector HNSW rollback + id-burn on WAL fail; REST rate-limit before auth; MCP durable fail-closed (`ABI_WDBX_ALLOW_MEMORY_FALLBACK`); scheduler OOM-safe error_msg.
- **Metal div/scale/relu + reduce_max** — `compute_api.div`, `vectorOps.scale`/`relu`, softmax prefers on-GPU max; claims/docs keychain honesty sync (macOS Partial / Win+Linux Proposed).
- **#734** — Metal `sub_kernel` + macOS-gated keychain file fallback off-macOS.
- **#733** — Metal `add_kernel` + honest non-macOS keychain status label; stub `.add` parity; Backend before load.
- **#732** — Parallel strangler extracts: AI router leaves, CLI agent hub+dispatch leaves, Metal `metal_kernels.zig`, credentials file/keychain leaves.
- **#731** — `metal_objc.zig` extract; `wdbx_simulate` options/format leaves + hub.
- **#730** — Split oversized `multiway.zig` into typed leaf modules; `wdbx.multiway.*` API unchanged.
- **modernized-refs gate** — portable `tools/check_modernized_refs.sh` (bash 3.2) wired into `zig build full-check`; fails on stale fenced `` `src/...` `` pointers under `modernized/` (not a second build root).
- **#728** — `abi auth status` surfaces active credential backend (`keychain` / `file`).
- **#727** — P0-1 lock-across-I/O audit table, bench regression gate, multiway/octtree docs refresh.
- **#726** — Zig pin `0.17.0-dev.1442+972627084` (+ CI `ZIG_VERSION` sync).
- **#712 Metal softmax** — demo-grade `softmax_kernel` + `softmax_norm_kernel` with CPU/GPU parity; claims sync. Broader kernels / CUDA / ANE remain Proposed.
- **#684 Metal multi-pass reduce** — `runReduceSum` loops 256-wide threadgroup partials until one scalar.
- **Approach-1 A–C + modern-refactor** — complete; product reimagine stays Phase D scaffold until cutover.

---

## P0-1 lock-across-I/O audit (2026-07-23)

Scope: every lock site in WDBX network-facing paths (`rest.zig`, `rest_handlers.zig`, `cluster.zig`, `cluster_rpc.zig`, `runtime.zig`, `durable_store.zig`, `remote_compute.zig`) and MCP HTTP (`src/mcp/http_transport.zig`). Method: `grep -rn "Mutex|\.lock\(\)|\.lockShared|\.tryLock"` over `src/`, then trace each critical section for `std.Io.net` reads/writes under the lock.

Result: exactly **one** mutex exists in all of `src/` — the token-bucket spin-lock in `rate_limiter.zig`. All other scoped files hold zero locks; their concurrency model is a single-threaded accept→handle loop with per-connection arenas, and `Store` has no internal locking, so "no lock held across network I/O" holds vacuously there (a site-by-site row would be fabricating lock sites that don't exist).

| Site | Lock | Critical section | Verdict |
|------|------|------------------|---------|
| `src/features/wdbx/rate_limiter.zig:65` `acquire()` | spin-lock (`std.atomic.Mutex`) | `refillLocked()` + token/counter arithmetic only; the 429 response write in `rest.zig:99-111` happens after `acquire()` returns (lock released by `defer`). | SAFE |
| `src/features/wdbx/rate_limiter.zig:92` `statsJson()` | same spin-lock | one `std.fmt.allocPrint` (allocation, no socket ops); response write happens after return (`rest.zig:121-125`). | SAFE |
| `rest.zig` / `rest_handlers.zig` / `cluster.zig` / `cluster_rpc.zig` / `runtime.zig` / `durable_store.zig` / `remote_compute.zig` / `mcp/http_transport.zig` | none | no `Mutex`/`lock()` in any of these files (grep-verified 2026-07-23). | SAFE (vacuous) |

Residual risk: if any of these servers ever moves to a threaded accept model, `Store` needs a locking design first — nothing guards concurrent mutation today; that is a known single-threaded-by-design boundary, not a latent race in current code.

## References

- `docs/spec/wdbx-north-star.mdx` — Current/Partial/Proposed
- `docs/contracts/external-claims-audit.mdx` — claim boundaries
- `docs/spec/non-loopback-rest-threat-review.mdx` — REST/MCP HTTP expose ops (not hardened)
- `docs/spec/cluster-mtls-ops.mdx` — cluster RPC mTLS/membership ops (not sharding)
- `docs/spec/phase-d-cutover-plan.mdx` — HITL cutover checklist
- `modernized/README.md` — Phase D scaffold rules
- `ABI-MEGA-PLUGIN.md` — local Codex `~/plugins/abi-mega` operator plugin (inventory/gates/audit)
- `CHANGELOG.md` — release notes
