# Issue #647 — Refactor/Completion Wave Plan (verified 2026-07-22)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. One task per `cursor/` branch, squash-merge to `main`, `./build.sh check` green before every PR.

**Goal:** Run the next ABI refactor/completion wave as a verified, low-risk sequence instead of a broad rewrite.

## Context — corrections from live-repo verification

- **DONE, excluded from plan:** silent-catch scanner (`tools/check_silent_catch.sh`, wired at `build.zig:491` into `zig build check`); CI `check-parity` jobs (commit `f4eda43`). Only the *required-status-check* setting remains.
- **Spec drift:** `mcp/src/features/core/database/engine.zig`, `db_lock`, `AIClient.snapshot()`, `connectAI()` do **not exist** in this repo. The lock-audit item is reframed as P0-1/P0-2 against the real surfaces: `src/features/wdbx/rest.zig`, `cluster_rpc.zig`, `runtime.zig`, and the MCP HTTP server.
- **Blocking hazard:** working tree has 9 modified files replacing `@intFromEnum` with `@backingInt` (e.g. `src/connectors/http.zig`) — `@backingInt` **does not compile on the pinned** `0.17.0-dev.1398+cb5635714`. Resolve first (W0).
- Bench harness exists but untracked (`tools/bench_regress.sh`, `tools/bench_baseline.json`). No `feat_ha` flag, no ACP code, no MCP subscriptions/prompt templates, no expired/malformed-token auth tests.

## Global constraints (apply to every task)

- Zig pin `0.17.0-dev.1398+cb5635714`; verify against `~/.zvm/$(cat .zigversion)/zig build test`, not PATH `zig`.
- Zig 0.17 patterns only: `std.process.Init` main, `ArrayListUnmanaged(T).empty`, `splitScalar`/`splitAny`/`splitSequence`, `foundation.time.unixMs()`, explicit allocators, `std.Io.net.Stream`, `std.testing.refAllDecls(@This())` in every module.
- No silent `catch {}` in data/inference/persistence paths (CI now enforces).
- Public API changes touch `mod.zig` **and** `stub.zig`; `./build.sh check-parity` green.
- Disabled flags compile and return `error.FeatureDisabled`. Keep `AGENTS.md`/`CLAUDE.md`/`GEMINI.md` in sync.
- No unproven claims (FHE/RBAC/sharding/QPS/K8s/H100). GitHub Actions is billing-locked → run `./build.sh check` locally as the gate.

---

## P0 — Safety/stability (Week 1)

| ID | Task | Owner | Effort | Deps | Acceptance |
|----|------|-------|--------|------|------------|
| W0 | Resolve `@backingInt` working-tree drift: revert the 9-file diff **or** bump `.zigversion`+CI `ZIG_VERSION` together (one PR, not both directions). | maintainer + zig-build-doctor | S (½d) | — | `~/.zvm/$(cat .zigversion)/zig build test` green; `git status` clean or committed |
| P0-1 | Lock-across-I/O audit of real surfaces: grep every `Mutex`/lock in `src/features/wdbx/{rest,cluster_rpc,runtime,durable_store}.zig` + MCP HTTP server; confirm no lock held across `std.Io.net.Stream` read/write; where found, copy-out-then-unlock (value snapshot owning its memory). Write findings to `tasks/todo.md`. | maintainer + wdbx-explorer agent | M (1–2d) | W0 | Audit note lists every lock site with verdict; any hold-across-I/O fixed in same PR |
| P0-2 | Concurrency regression test: threaded test — concurrent WDBX `search()`/REST query vs. store `deinit()`/reconfig on the runtime; run under the pinned toolchain. | maintainer | M (1d) | P0-1 | New test in `src/tests/`; passes 50 iterations locally; no TSan/leak findings from `zig build test` |
| P0-3 | Commit bench harness: track `tools/bench_regress.sh` + `bench_baseline.json`; add `bench-regress` step to `./build.sh full-check` (>5% slowdown fails); add CI job (activates when billing unlocks). | maintainer | S (½d) | W0 | `./build.sh full-check` runs regression gate; deliberate 10% slowdown in a scratch branch fails it |
| P0-4 | Mark `check-parity` a required status check on `main` (GitHub branch protection). | maintainer | S (5min) | billing unlock | Setting visible in repo settings; documented in `AGENTS.md` |

## P1 — Feature completion (Weeks 2–4)

| ID | Task | Owner | Effort | Deps | Acceptance |
|----|------|-------|--------|------|------------|
| P1-1 | Offline-first E2E inference suite: CLI → router → inference engine → local connector shim (in-process `std.Io.net` loopback server speaking OpenAI-compatible JSON) → response cleanup. Deterministic skip when no endpoint configured (`error.SkipZigTest`). Asserts response body is owned and freed (allocator leak check). | maintainer | L (3d) | W0 | Suite green with and without a local endpoint; leak-checked; covers cleanup on error paths |
| P1-2 | Provider smoke lanes: parameterize P1-1 shim for Ollama / LM Studio / llama.cpp / MLX / vLLM request shapes (all `.local`-classified prefixes). No cloud credentials anywhere. | maintainer | M (1–2d) | P1-1 | 5 lanes deterministic-green offline; catalog classifies each prefix `.local` (asserted) |
| P1-3 | MCP auth-failure contract tests: expired, malformed, wrong-scheme, empty bearer vs `ABI_MCP_HTTP_TOKEN`; plus subscriptions/prompt-template seams as stubs returning proper JSON-RPC errors (12-tool surface frozen — additive only). | maintainer + mcp-contract-auditor | M (1–2d) | W0 | Contract tests enumerate 12 tools unchanged; each bad-token case → 401 w/o body leak |
| P1-4 | ACP streaming + push-notification seams (greenfield): define minimal seam types behind existing feature gates; stub transport returns `error.FeatureDisabled` until validated; contract test for the seam shape only. | maintainer | M (2d) | P1-3 | Compiles with flag off/on; disabled path returns `error.FeatureDisabled`; no capability claims in docs |
| P1-5 | TUI completion: full chat panel + database panel behind `feat_tui` (metrics/log/diagnostics already exist). Reuse `src/features/tui/` render helpers. | maintainer + tui-navigation-guide | M (2d) | W0 | Panels render in `abi tui`; `feat_tui=off` build compiles; snapshot/smoke test via `tools/run_tui_smoke.sh` |
| P1-6 | Add `feat_ha` build flag (default **off**) + leader-election/WAL-shipping stub modules with acceptance tests asserting stubs return `error.FeatureDisabled`; parity pair (`mod.zig`/`stub.zig`). | maintainer | M (1–2d) | P0-1 | Both flag states compile; `check-parity` green; README/AGENTS wording stays claim-honest ("skeleton, not validated") |

## P2 — Decomposition slices (Weeks 4–6, opportunistic)

One subsystem per slice, only where test seams exist; parity + focused tests + `tasks/todo.md` (validation & residual risk) each PR.

| ID | Slice | Owner | Effort | Deps | Acceptance |
|----|-------|-------|--------|------|------------|
| P2-1 | Discord REST split | maintainer + connector-validator | S–M | P0 done | `./build.sh check` green; redaction tests still pass |
| P2-2 | GPU device/fusion split | maintainer + gpu-backend-analyzer | M | P0 done | Metal demo + CPU fallback parity tests green |
| P2-3 | AI streaming split | maintainer | M | P1-1 | E2E suite (P1-1) still green — it is the seam |
| P2-4 | Training module split | maintainer | M | P0 done | Existing nn tests green |
| P2-5 | Network raft + security password-helper splits | maintainer | S–M | P1-6 | Flag-off builds compile; no new claims |

## Milestones

- **M1 (end Week 1):** W0 + P0-1..P0-3 merged; P0-4 queued on billing unlock. Wave unblocked.
- **M2 (end Week 3):** P1-1..P1-3 merged — local-first inference proven offline; MCP auth contract complete.
- **M3 (end Week 4):** P1-4..P1-6 merged — seams + `feat_ha` skeleton, claim-honest.
- **M4 (end Week 6):** ≥3 P2 slices merged; `./build.sh full-check` (incl. bench gate) green; #647 closed.

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Zig master drift re-introduces pin-breaking edits (`@backingInt` pattern) | High | Build red on pin | W0 policy: all verification via `~/.zvm/$(cat .zigversion)`; 30m pin-drift cron already watches |
| CI billing lock persists → no remote gates | High | Regressions merge unguarded | Local `./build.sh check` mandatory pre-merge; P0-4 deferred, not dropped |
| Local-provider smoke lanes flake on real endpoints | Medium | CI noise, eroded trust | Lanes hit the in-process shim only; real endpoints strictly opt-in via env var |
| `feat_ha`/ACP stubs read as capability claims | Medium | Violates claims policy | external-claims-auditor pass on every doc touch; stubs return `error.FeatureDisabled` |
| P2 splits cause parity/API drift | Medium | check-parity red, stub rot | One-subsystem slices; parity gate in `check`; mod/stub edited in same commit |

## Final acceptance checklist (Issue #647)

- [ ] 1. No `mod.zig`/`stub.zig` parity drift → `check-parity` in CI (done) + P0-4 required check + green through M4.
- [ ] 2. No silent `catch {}` in persistence/inference/connector/DB paths → scanner in `check` (done); stays green through all slices.
- [ ] 3. Disabled flags compile and return `error.FeatureDisabled` → P1-4, P1-6 acceptance + existing `tools/check_feature_stubs.sh`.
- [ ] 4. Local-first AI works via Ollama/LM Studio/llama.cpp-compatible endpoints w/o cloud credentials → P1-1 + P1-2.
- [ ] 5. Connector-backed inference owns/frees response memory correctly with regression coverage → P1-1 leak-checked cleanup assertions.
- [ ] 6. DB engine public methods race-safe, no global locks held across network I/O → P0-1 audit + P0-2 regression test.
