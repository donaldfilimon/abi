# WDBX Compute/Cluster Layer — Modernization Strategy

**Scope:** `src/features/wdbx/{compute,cluster,cluster_rpc,remote_compute}.zig` (+ shared helper `net_line.zig`).
**Kind:** Planning only. No Zig code changed by this document.
**Skill applied:** `refactor-strategy` (`.claude/skills/refactor-strategy/SKILL.md`).
**Inputs re-verified this session:** `tasks/todo.md` Track F (✅ Ops honesty landed), `docs/contracts/external-claims-audit.mdx`, `docs/spec/wdbx-north-star.mdx` §2/§3.5, `docs/spec/cluster-mtls-ops.mdx`.

## Top-line recommendation

**This subsystem is functionally complete and claim-honest for its declared single-host / reference scope.** No rewrite, no strangler-fig migration, and no phased "harden toward production multi-host" program is justified — that would contradict the explicit non-goals in the claims audit and Track F. The only work worth doing is a short list of **internal quality/testability cleanups** (converging duplicated logic, reducing test flakiness, small dedup) that do not change external behavior, do not touch the CLI/MCP frozen surfaces, and do not license any new capability wording. Several items are legitimately "leave as-is."

---

## 1. Current behavior + success criteria

### What exists (verified by reading source, not summaries)

| Module | Role | Native tests (inline) |
|---|---|---|
| `cluster.zig` | In-process Raft-style consensus core: `Cluster.init/deinit`, `quorum`, `startElection`, `replicate`, `failNode`/`reviveNode`, `statusLine`; free functions `applyVote`/`applyAppend` for the networked path. Deterministic, caller-driven step model — no timers/randomness. | 4 tests: single-leader election, replication+commit, leader failover at higher term, quorum-loss unavailability. |
| `cluster_rpc.zig` | Real TCP transport over `std.Io.net` for RequestVote/AppendEntries: `listen`/`listenAddr`, `serveOnce(Auth)`/`serveLoop(Auth)`, `dialVote(Addr)(Auth)`/`dialAppend(Addr)(Auth)`, `readVoteReply`/`readAppendReply`. Newline-framed text wire protocol (`VOTE`/`APPEND`/`AUTH <token> ...`). `ClusterPolicy{ auth, peers }` gates on shared secret (constant-time compare `fixedWorkEql`) and an optional node-id allowlist. | 10 tests: loopback election, downed-peer election, loopback replication, routable `0.0.0.0` bind, authenticated vote+append, missing/wrong-secret rejection (state untouched), peer-allowlist rejection, null-allowlist-permits-any, auth-append-requires-leader-id, multi-node authenticated loopback round (4 nodes, votes+appends+log verification). |
| `compute.zig` | Backend enum (`cpu_scalar/avx2/avx512/neon`, `gpu_cuda/metal/vulkan`, `npu_ane`, `tpu_remote`) + `capabilities()` report + `select()` (always falls back to best CPU SIMD for any accelerator) + `dot()` (host-width `@Vector` kernel, same result on every backend by construction) + `aneHardwarePresent()` (arm64+macOS detection only). | 5 tests: accelerator fallback, ANE detection matches target, capability table shape, CPU/GPU/NPU dot-product parity, ragged-tail-length correctness. |
| `remote_compute.zig` | TCP dispatch transport for the "remote TPU" backend: `localDot` (reference/fallback), `endpoint()` (reads `ABI_REMOTE_COMPUTE_ENDPOINT`), `serveOnce` (reference/mock remote server), `dialDot`/`readDotReply`. | 3 tests: loopback dispatch matches local reference, malformed request surfaced as error (not silently mis-served), unreachable endpoint → `null` → caller CPU fallback. |
| `net_line.zig` | Shared newline-framing helper (`writeLine`, `resolveHost`, `dialAddr`/`dial`, `readLine` with `LineTooLong` on overflow). Used by both `cluster_rpc` and `remote_compute`. | 3 tests: clean frame, partial frame on early peer close, `LineTooLong` on buffer-filling frame with no newline. |

### What the contract suites assert today

- `tests/contracts/feature_modules.zig` pins **module presence and a handful of unrelated `wdbx` decls** (`persistence.CHECKSUM_PREFIX`, `Store.restoreBlock`, `storage.BlockChain.appendAt`, `spatial_3d.SpatialIndex3D.initWithPool`) plus the feature-flag-on/off serialize round-trip and `error.FeatureDisabled` degrade check. It does **not** assert any cluster/compute-specific behavioral invariant (no RPC frame shape check, no election/replication assertion, no compute-selection assertion) at the contract-suite level.
- The real behavioral invariants — RPC frame format, token/allowlist enforcement, leader-election and replication correctness, CPU/GPU parity, ANE detection matching build target, dispatch fallback on unreachable endpoint — are pinned entirely by the **inline `test { ... }` blocks inside each module** listed above (25 tests total across the four files + `net_line.zig`), not by the shared contract-test tree. This is fine (Zig-idiomatic, module-local), but it means "regression protection" for this subsystem lives with the module, not with `tests/contracts/`. A refactor plan must preserve those inline tests as the acceptance bar, since there is no separate contract file to lean on.
- `mod.zig`/`stub.zig` parity (`zig build check-parity`) covers only public top-level decl names — struct methods (`Cluster.startElection`, etc.) are invisible to it per repo convention; that's expected, not a gap to "fix."

### Real invariants to preserve (do not regress in any cleanup)

1. **RPC wire format** is exactly the four frame shapes in `cluster_rpc.zig`'s doc comment (`VOTE`, `APPEND`, `AUTH <token> VOTE`, `AUTH <token> APPEND`) and the `remote_compute.zig` `DOT <n> ...` frame. Any internal cleanup must keep these byte-compatible (nothing today depends on cross-version compat, but two internal modules and their tests both encode/decode the same frames — changing the format non-atomically breaks the pair).
2. **Auth/allowlist semantics**: constant-time secret compare (`fixedWorkEql`), reject-without-mutating-state on auth/peer failure, non-loopback bind refusal without `ABI_WDBX_CLUSTER_TOKEN` (enforced at the CLI layer in `wdbx_runtime.zig`, not in `cluster_rpc.zig` itself — worth noting as a layering fact, not a bug).
3. **Raft correctness properties**: single leader per term, quorum-gated election/replication, failover elects a strictly higher term, no quorum ⇒ no leader ⇒ `replicate` returns `error.NoLeader`.
4. **CPU/GPU parity-by-construction** in `compute.dot` (all backends literally call the same kernel) — must remain true after any restructuring, not just re-tested.
5. **Deterministic dispatch fallback**: unreachable remote endpoint ⇒ `null` ⇒ caller uses `localDot`/CPU path; malformed request ⇒ surfaced error, never silently mis-served.
6. **Claims boundary**: nothing here may start describing itself as distributed, sharded, or production multi-host. This is a stronger invariant than any test — it's a documentation/wording contract (`external-claims-audit.mdx`), and it binds every phase below.

---

## 2. Ideal modern design (clean-slate sketch)

If this subsystem were designed fresh today, for the same explicitly-scoped goal (single-host reference implementation, with a transport that happens to be routable), the shape would be:

- **Consensus state machine** (`cluster.zig`'s `applyVote`/`applyAppend` free functions) as the *single* source of truth for the Raft grant/append rules — both the in-process demo path (`Cluster.startElection`/`replicate`) and the networked path (`cluster_rpc.serveOnceAuth`) call into it, instead of the demo path re-deriving its own inline grant condition.
- **Transport layer** (`cluster_rpc.zig` minus the state-machine calls) reduced to pure framing + dial/serve/policy-check, already close to this shape.
- **Policy layer** (`ClusterAuth`/`ClusterPolicy`) already cleanly separated from both state machine and transport — this is a good existing boundary, worth preserving as-is.
- **Compute selection vs. dispatch** kept as two distinct concerns as they already are: `compute.zig` answers "which backend, and what does the CPU-equivalent kernel compute" (a pure, synchronous decision), while `remote_compute.zig` answers "how do I ship one op to an out-of-process endpoint" (an I/O concern). This separation is already correct and should not be merged.
- **Shared error taxonomy** for the two TCP transports (`cluster_rpc.RpcError`, `remote_compute.RemoteError`) instead of two independently-declared sets with the same member names.
- **Deterministic test harness ergonomics**: bind ephemeral ports (`:0`) and read back the assigned port, rather than hardcoded literal ports per test, removing a class of CI flakiness risk (port already in use) with zero behavior change.

This sketch is explicitly **not** a step toward multi-host/production/sharding — it's the same single-host reference design, just with duplicated logic consolidated and test infrastructure hardened. State this framing in any PR description to keep review honest.

---

## 3. Gap analysis: current vs. ideal

### Already solid — leave as-is

- `net_line.zig` framing (`readLine`/`writeLine`/`resolveHost`/`dialAddr`) — clean, well-tested (3 dedicated frame-boundary tests), shared correctly by both transports already.
- `ClusterAuth`/`ClusterPolicy` separation from transport and state machine — good boundary, no change needed.
- `compute.zig` vs `remote_compute.zig` separation (selection/kernel vs. dispatch transport) — correct as designed.
- Non-loopback bind refusal logic in `wdbx_runtime.zig` (CLI layer) — correctly kept out of `cluster_rpc.zig` itself, since the transport is deliberately host-agnostic and the policy decision belongs to the caller.
- Doc-comment honesty in all four files — already precise about `native=false`, "not production multi-host/sharding," "operator-supplied remote endpoint," etc. No wording changes needed.

### Genuinely missing — explicitly out of scope (per claims audit + Track F), NOT milestones

- Dynamic cluster membership (join/leave protocol).
- Native TLS/mTLS linked in-process (proxy-fronted mTLS is the standing decision per `docs/spec/cluster-mtls-ops.mdx`).
- Data sharding / partitioning across nodes.
- Production multi-host deployment/ops automation beyond the existing ops doc.
- Native ANE/CUDA/Vulkan dispatch (disclosed non-goals elsewhere in the codebase).

These stay in `docs/spec/wdbx-north-star.mdx` §2/§8 as "Proposed" and in `tasks/todo.md` Track F residual column. This plan does not schedule any of them, and no milestone below should be read as progress toward them.

### Worthwhile internal cleanup (implies no new claims)

1. **Converge Raft vote-granting logic** (highest value). `Cluster.startElection` inlines its own grant condition (`peer.term < new_term or peer.voted_for == null`) rather than calling the free function `applyVote` that the networked path (`cluster_rpc.serveOnceAuth`) already uses. Similarly `Cluster.replicate`'s per-node `appendEntry` helper duplicates (a simplified form of) what `applyAppend` does for the networked path. The two implementations currently agree on all tested scenarios (verified by tracing both against the existing 4 `cluster.zig` tests + 10 `cluster_rpc.zig` tests), but having two independently-maintained encodings of "when does a peer grant a vote / accept an append" is a correctness-drift risk, not a proven bug today. Converging the demo path to call `applyVote`/`applyAppend` directly (or extracting one shared helper both call) removes the risk and is a pure internal simplification.
2. **Duplicated error sets.** `cluster_rpc.RpcError` and `remote_compute.RemoteError` both declare `MalformedRequest`/`MalformedResponse` independently. A shared transport error set (e.g. in `net_line.zig` or a small new `transport_errors.zig`) removes the duplication. Low risk, low value, but cheap.
3. **Hardcoded test ports.** Every RPC test across `cluster_rpc.zig` and `remote_compute.zig` binds a literal port (39101, 39102, ... 39312). This is a latent CI-flakiness source (port collision under parallel test runs or repeated local `zig build test` invocations that don't fully release sockets). Switching to bind-`:0` + read-back-assigned-port removes this class of flake with no behavior change. Medium value, low risk.
4. **Frame-buffer size asymmetry.** Vote dial buffer is `[64]u8`, append dial buffer is `[4096]u8`, response buffer in `serveOnceAuth` is `[64]u8`. Not a bug (append genuinely needs more room for `data`), but worth one explicit boundary test (large-but-valid append payload near the 4096 limit, and confirmation that an oversized payload fails predictably via `net_line.readLine`'s `LineTooLong`) rather than leaving the size choice implicitly load-bearing.
5. **Minor tokenizing duplication.** `cluster_rpc.zig`'s `parseVote`/`parseAppend`/`parseRequest` and `remote_compute.zig`'s inline parsing in `serveOnce` both hand-roll `indexOfScalar`/`splitScalar` token parsing. A small shared "split first token, parse rest" helper in `net_line.zig` would reduce duplication, but the two protocols are different enough (vote/append/auth prefixes vs. a flat `DOT <n> <floats>` list) that this is optional polish, not a correctness item.

---

## 4. Strategy

**Incremental cleanup only — no phased rewrite, no strangler fig, no parallel implementation.** Per the skill's strategy-selection matrix, this subsystem is "small isolated module[s], good tests" (25 inline tests across 4 files, each module single-responsibility already), which calls for direct, small, in-place edits — not a parallel-build-and-cutover process. Given the explicit non-goal of expanding toward multi-host/sharding, there is also no "ideal design" gap large enough to justify a phased migration: the gap is entirely internal-quality-sized.

Recommended execution shape: **up to 3 independent, separately-reviewable slices**, each optional and each a strict subset of "internal cleanup, no behavior change, no new claims":

- **Slice 1 (highest value):** Converge `Cluster.startElection`/`replicate` onto `applyVote`/`applyAppend` (item 1). This is the only item with real correctness-risk payoff; it should land first and alone so any regression is easy to bisect.
- **Slice 2 (test hygiene):** Ephemeral test ports for `cluster_rpc.zig` + `remote_compute.zig` (item 3), plus the frame-boundary test addition (item 4). Pure test-file changes, zero production-code risk.
- **Slice 3 (polish, optional/low-priority):** Shared transport error set (item 2) and/or shared tokenizing helper (item 5). Do only if slices 1–2 land clean and there's appetite; skip without loss if not — these are the lowest-value items on the list.

If the honest call after slice 1 is "not worth the diff churn," it is legitimate to stop there or even do nothing at all — re-affirming: **most of this subsystem is fine as a single-host reference implementation**, and slices 2–3 exist only to remove small, real risks (flaky ports, drifted duplicate logic), not to chase a larger redesign.

---

## 5. Validation criteria per phase

Applies uniformly to every slice above:

1. **Existing inline tests must all still pass unmodified in intent** (the 25 tests enumerated in §1), specifically:
   - `cluster.zig`'s 4 tests (single-leader election, replication+commit, failover-at-higher-term, quorum-loss).
   - `cluster_rpc.zig`'s 10 tests (loopback election, downed-peer election, loopback replication, routable bind, authenticated vote/append, missing/wrong-secret rejection, peer-allowlist rejection, null-allowlist, auth-requires-leader, multi-node authenticated round).
   - `compute.zig`'s 5 tests (fallback, ANE detection, capability table, parity, ragged tail).
   - `remote_compute.zig`'s 3 tests (loopback dispatch, malformed-request, unreachable-endpoint).
   - `net_line.zig`'s 3 framing tests.
2. **New tests proposed per slice:**
   - Slice 1: a property-style test asserting `Cluster.startElection`/`replicate`'s outcome is identical to driving the same sequence of `applyVote`/`applyAppend` calls directly (regression guard for the convergence, and a permanent guard against future re-divergence).
   - Slice 2: the large-append-near-buffer-limit boundary test (item 4) plus confirmation the ephemeral-port tests still assert quorum/vote/ack counts identically to today.
   - Slice 3 (if done): no new behavioral tests required beyond keeping existing ones green (pure structural dedup).
3. **Gates, run after each slice:**
   - `./build.sh check` (primary gate — build, tests, lint, parity, feature-off stubs, CLI smoke).
   - `zig build test -Dtest-filter="cluster"` and `-Dtest-filter="compute"` / `-Dtest-filter="remote_compute"` for fast focused iteration before the full gate.
   - `zig build check-parity` only if any slice touches a public top-level decl name in `mod.zig`/`stub.zig` (none of the proposed items should — all are internal to the real modules, and `stub.zig` already has empty stand-ins for all four).
4. **Claims-audit discipline re-check** after each slice: re-diff `docs/contracts/external-claims-audit.mdx`, `docs/spec/wdbx-north-star.mdx` §2/§3.5, and `docs/spec/cluster-mtls-ops.mdx` against the change. Expected outcome for every slice in this plan: **no wording change required**, since none of the proposed work alters what the subsystem is capable of — only how the existing capability is implemented/tested. If any slice's diff would require a claims-doc wording change, that is a signal the slice scope crept beyond "internal cleanup" and should be re-scoped or dropped.

---

## 6. Milestones with Definition of Done

| # | Milestone | Definition of Done |
|---|---|---|
| M0 | Plan approved | This document reviewed; no code changed (current state). |
| M1 | Slice 1 — converge vote/append grant logic | `Cluster.startElection`/`replicate` call `applyVote`/`applyAppend` (or a single shared helper); new equivalence test added; all 25 existing inline tests pass unmodified in assertion; `./build.sh check` green; claims-audit re-check shows no wording delta. |
| M2 | Slice 2 — test-port hardening | `cluster_rpc.zig` + `remote_compute.zig` tests bind ephemeral (`:0`) ports and read back the assigned port instead of hardcoded literals; new frame-boundary test for near-limit append payload added; all existing assertions (vote/ack counts, quorum booleans) unchanged; `./build.sh check` green; claims-audit re-check shows no wording delta. |
| M3 (optional) | Slice 3 — shared error set / tokenizing dedup | `RpcError`/`RemoteError` consolidated (or explicitly left separate with a one-line rationale comment if consolidation turns out awkward given differing error semantics); optional shared token-split helper in `net_line.zig`; no behavior change; `./build.sh check` green. |
| M-final | Re-affirm scope boundary | `tasks/todo.md` Track F row and `docs/spec/wdbx-north-star.mdx` §2/§8 remain unchanged in substance (still "Partial/disclosed," still explicitly not sharding/production-multi-host) — confirming the cleanup did not drift into a claims expansion. |

**If none of M1–M3 is picked up:** that is an acceptable outcome. The default state — this document existing as a recorded decision that "compute/cluster is single-host-reference-complete, only optional internal cleanup remains, sharding/mTLS/dynamic-membership stay Proposed" — is itself the deliverable of this planning pass.
