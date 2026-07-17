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
| **B** | Metal native kernels for `vectorOps` hot paths (Zig+Metal only) | ◑ Partial | Metal fused `cosine_parts_kernel`; wider SIMD fallback | Full GPU reduce; CUDA/Vulkan kernels. **ANE dispatch = non-goal** (100% Zig). |
| **C** | Windows credential ACL + secret zeroing; runtime verify CI-blocked | ◑ Partial | Win32 SDDL owner-only DACL on credential write; POSIX `secureZero` / `secureWipe` | OS keychain. **Windows runtime verification** needs a Windows host/runner (cross-smoke = compile-only). |
| **D** | Lossless ANS/order-1 demo next to Huffman — not SOTA | ✅ Demo landed | `src/features/wdbx/ans.zig` + `abi wdbx secure demo` | Production/SOTA learned codec (ANS/arithmetic/context-model at scale). |
| **E** | FHE deepen reference tests/docs — not audited | ✅ Reference deepened | `fhe.zig` (`REF_P_BITS` / `REF_NOISE_BITS` / `VERIFIED_MUL_DEPTH`) | External security audit + bootstrapped FHE. |
| **F** | Cluster TLS-fronting story + ops docs — not sharding | ✅ Ops honesty landed | `cluster_rpc` TOKEN/PEERS tests; non-loopback TLS-fronting note | Production multi-host, mTLS, dynamic membership, **sharding**. |
| **G** | Phase D first approved scaffold under `modernized/` | ✅ Minimal scaffold | `modernized/README.md` + `modernized/packages/*` | Cutover / second build root. **Live code remains `src/`**. |

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
| Native compute beyond Metal cosine | ◑ Partial | Metal fused cosine + SIMD for HNSW/`vectorOps`; CUDA/Vulkan/full reduce not linked. ANE execution **out of scope**. |
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
| 2 | Broader Metal GPU path (more kernels / reduce) | 🟡 | Extend beyond fused cosine; keep CUDA/ANE disclosed. |
| 3 | Windows **runtime** CI/job | 🔴 Blocked (no Windows runner) | ACL code exists for windows-gnu; execution verify blocked on host. |
| 4 | OS keychain credential storage | ⚪ | Disclosed gap after Win32 ACL + POSIX zeroing. |
| 5 | Phase D cutover plan (HITL) | ⚪ | Scaffold exists; no second `build.zig`; cutover needs explicit approval + package ports. |
| 6 | Non-loopback REST threat review + native TLS link decision | ⚪ | Docs/ops first; do not claim hardened expose. |
| 7 | Cluster mTLS / membership (still not sharding) | ⚪ | Only with real TLS story; sharding stays Proposed. |

---

## Constraints & intentional non-goals

Do not schedule these as “complete”:

- **ANE execution** — CoreML/ObjC; excluded by 100% Zig. Detection only.
- **MCP HTTP ↔ WDBX REST framing duplication** — intentional (`src/mcp/` module-root isolation).
- **Never force-push `main`.**
- **No unbacked claims** — sharding / AES / RBAC / cert / QPS / latency / accuracy / audited FHE / SOTA compression.

---

## Known test failures

- None currently reproduced. Pin: `0.17.0-dev.1398+cb5635714`. Prefer `./build.sh check` (macOS) as the primary gate.

---

## Recently landed (short digest)

Full detail: `git log` + `CHANGELOG.md`. Keep this list short.

- **#676 Tracks A–G** — `incremental.zig`; Metal fused cosine; Win32 credential DACL; `ans.zig` demo; FHE param honesty; cluster TLS-fronting ops; `modernized/packages/*` Phase D scaffold.
- **#674 Feature polish** — RankedNode/SearchResult borrowed vectors; stream footers; Ctrl-R redraw; constitution `/status`; POSIX `secureZero`.
- **#673** — Restored AGENTS.md Learned prefs/facts (continual-learning).
- **#672 / #671 / #670 / #669** — Board sync; CLAUDE/GEMINI → AGENTS redirects; wdbx query UX; remaining-improvements (0700 Linux, live SSE, file-context v2, leaf extracts).
- **#668** — OS-control trusted command hardening.
- **Approach-1 A–C + modern-refactor 2–4** — complete; product reimagine = Phase D scaffold only until cutover.

---

## References

- `docs/spec/wdbx-north-star.mdx` — Current/Partial/Proposed
- `docs/contracts/external-claims-audit.mdx` — claim boundaries
- `modernized/README.md` — Phase D scaffold rules
- `CHANGELOG.md` — release notes
