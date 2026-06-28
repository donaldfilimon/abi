# TODO — ABI Framework

Forward-looking tracker for **incomplete and in-flight** work. Completed history is **not** kept here — it lives in `git log` and `CHANGELOG.md`. The lightweight active board is `TASKS.md`; long-horizon direction is `docs/spec/wdbx-north-star.md` (§2/§8 Current/Partial/Proposed mapping) and `tasks/roadmap-next.md`.

Status legend: `✅ Done` · `🟡 In progress` · `⚪ Not started` · `🔴 Blocked` · `◑ Partial / disclosed`

> Discipline: do **not** add "Session Summary" logs here — that is what git history and the CHANGELOG are for. When an item closes, delete its row (or move a one-line note to "Recently landed"), don't append a narrative.

---

## Open work

### Honest stubs — keep disclosed, do NOT fake-complete

These ship real local artifacts but truthfully disclose that native/external dispatch is not linked. "Completing" them with simulated capability would violate `docs/contracts/external-claims-audit.md`. Leave as-is unless wiring genuine native dispatch/toolchains.

| Item | Status | Constraint |
| ---- | ------ | ---------- |
| `accelerator` backend dispatch | ◑ Selection report only | Native CUDA/Metal/NPU/TPU dispatch not linked; CPU SIMD fallback is the real path. |
| `shaders` validation | ◑ Validate + checksum only | No real shader compiler/toolchain linked. |
| `mlir` lowering | ◑ Textual analyze/lower only | No external MLIR/LLVM toolchain linked. |
| `mobile` runtime profile | ◑ Profile reporting only | `native_dispatch=false` reported explicitly; no platform runtime. |

### Still Proposed (in-process demos exist; production forms do not)

| Item | Status | Gap to production |
| ---- | ------ | ----------------- |
| Native compute dispatch | ⚪ Not started | ANE/TPU/CUDA/Metal-kernel execution (vs the current local SIMD/Metal-fallback path). ANE execution is **out of scope** under the 100% Zig constraint — see Non-goals. |
| Production/SOTA learned compression codec | ◑ Partial / disclosed | Exact order-0 Huffman entropy (`entropy.zig`), int8 embedding quantization, and the reference `neural_compress.zig` autoencoder exist; no ANS/arithmetic/context-model or production-scale learned codec is proven. |
| Security-audited FHE | ⚪ Not started | `fhe.zig` (DGHV, depth-2) is reference-parameter, bounded-depth, **not** audited. |
| Non-loopback REST hardening | ◑ Partial / disclosed | `rest.zig` remains 127.0.0.1-only and can require `Authorization: Bearer` via `ABI_WDBX_REST_TOKEN`; external exposure still needs TLS, rate limiting, authz, and threat review. |
| Multi-host cluster | ◑ Loopback-tested | `cluster_rpc.zig` runs real TCP Raft over 127.0.0.1; multi-host needs a routable bind + ops story. |

### Candidate next slices (real remaining work)

| Item | Status | Notes |
| ---- | ------ | ----- |
| Broader native/batched GPU acceleration | 🟡 In progress | HNSW pairwise + neighbor-expansion batch scoring route through `gpu.vectorOps()` with SIMD fallback. AI completion/SEA paths delegate similarity to `store.search` (already GPU-routed), so the remaining expansion is native kernel dispatch — the deferred 100%-Zig-constraint item, not a completable gap. |
| Cross-compilation CI | ✅ Matrix added | `.github/workflows/ci.yml` runs `zig build check` + `zig build cross-smoke` (linux-gnu/windows-gnu/aarch64-macos). Remaining (out of scope from a macOS host): Windows runtime verification + Windows test-only helpers (`/tmp`, `std.c.getpid`). |

---

## Constraints & intentional non-goals

These are decisions, not unfinished work — do not "fix" them.

- **ANE execution** requires CoreML/ObjC + on-device profiling; excluded by the 100% Zig constraint (user-accepted). Detection (`compute.aneHardwarePresent()`) is truthful; dispatch is not linked.
- **`rest.zig` ↔ `src/mcp/server.zig` HTTP-framing duplication is intentional.** `src/mcp/` is its own module root and cannot import a shared `src/foundation/` leaf (confirmed by compile error). See memory `mcp-module-root-isolation`.
- **`origin/main` shares no common ancestor** with local `main` (different roots). Never force-push to reconcile. See memory `origin-main-unrelated-history`.
- **External-claims policy** (`docs/contracts/external-claims-audit.md`): no unbacked sharding/AES/RBAC/cert/QPS/latency/accuracy claims; frame unproven metrics as targets.

---

## Known test failures

- None currently reproduced. `./build.sh check` (36/36, ~1042 tests) and `./build.sh full-check` green at HEAD.

---

## Recently landed (digest — full detail in git + CHANGELOG)

One-line pointers only; the authoritative record is `git log` and `CHANGELOG.md`.

- **MCP concurrency hardening** — shutdown use-after-free closed (teardown deferred to `main` after the HTTP thread joins); TOCTOU lazy-init race in shared scheduler/store closed (double-checked locking, release/acquire ordering).
- **Credential-file hardening** — `abi auth` now creates/repairs `~/.abi` as owner-only (`0700`) and opens/truncates `credentials.json` as owner-only (`0600`) before writing secrets on POSIX-capable targets; still plaintext, with keychain/Windows ACL/zeroing left as disclosed future work.
- **Connector log redaction** — Discord local send/receive logs and Twilio live response logs now emit metadata/byte counts instead of message or provider-response bodies.
- **MCP/REST loopback auth hardening** — optional bearer-token enforcement added for MCP HTTP/SSE (`ABI_MCP_HTTP_TOKEN`) and WDBX REST (`ABI_WDBX_REST_TOKEN`); still not a production non-loopback exposure claim without TLS/authz/rate-limit review.
- **WDBX/SEA correctness** — WAL double-free guards on `putVector`/`store`; `remote_compute` overflow guard; corrupt-manifest rejection; SEA persist→recall round-trip + evidence-recall coverage.
- **WDBX perf** — redundant work removed from HNSW/WAL/block-chain hot paths.
- **WDBX compression** — exact order-0 Huffman entropy codec added beside int8 embedding quantization and the reference autoencoder; still no SOTA/production learned-compression claim.
- **Build/parity** — `check-parity` now fails on a `mod.zig` leaf missing its `stub.zig`.
- **AI training observability** — `training_support.inspectDatasetTracked` routes dataset path/read/JSONL parse allocations through `MemoryTracker`, and `trainWithStore` now falls back to the attached store tracker for the initial training phase.
- **WDBX north-star Phase 1 + V18 cognitive runtime** — WAL+recovery, multi-segment checkpoints, temporal/causal hybrid ranker, persona-scoped retrieval, P50/P95/P99 benchmarks, loopback REST, in-process consensus/compression/FHE demos. (10/11 V18 criteria; ANE execution is the disclosed non-goal.)

---

## References

- `docs/spec/wdbx-north-star.md` — Current/Partial/Proposed capability mapping
- `tasks/roadmap-next.md` — full refreshed roadmap view
- `tasks/scheduler-memory-wireup.md` — scheduler/memory integration detail
- `docs/contracts/external-claims-audit.md` — what public docs may and may not claim
- `CHANGELOG.md` — release-note record of landed changes
