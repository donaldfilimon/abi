# Implementation Playbook

Safe execution steps for clean-slate modernization. Pair with **refactor-validation** after each significant change.

## Preconditions

1. Analysis checklist complete (targets + evidence).
2. Strategy chosen (direct / phased strangler / parallel modern impl).
3. Baseline gate green: `./build.sh check` (or the narrowest gate that covers the blast radius).
4. Frozen surfaces listed — do **not** resurrect legacy CLI names or invent MCP tools.

## Execution loop

### 1. Characterize before rewrite

- Prefer existing contract/unit tests that drive the **shipped** path.
- If missing, add a characterizing test that fails on wrong behavior without re-implementing the code under test.
- Capture baseline output when behavior is observational (CLI smoke, docs structure).

### 2. One semantic change at a time

- Prefer parallel files (`foo_v2.zig` / extracted helper) over in-place rewrites of hot contracts.
- Keep pure helpers extracted first; push IO/effects to edges.
- For Zig 0.17: `ArrayListUnmanaged(T).empty`, `trimEnd`, `splitScalar`/`splitAny`/`splitSequence`, explicit allocators, no silent empty `catch {}` on data paths.

### 3. Gate after each batch

| Change kind | Minimum gate |
| ----------- | ------------ |
| Docs prose only | Spot-check claims vs `docs/contracts/external-claims-audit.mdx`; `npx mint@latest validate` if nav/content structure changes |
| Tools scripts (no assertion rewrite) | Re-run the wired step or `./build.sh check` |
| Public feature API | Update `mod.zig` + `stub.zig`; `zig build check-parity`; `./build.sh check` |
| CLI/MCP handler | Contract suites + `./build.sh check` (surface must stay frozen unless intentional) |

### 4. Cutover rules

- Do not delete the old path until parity is proven.
- Prefer expand → migrate callers → contract → delete.
- Leave the tree buildable and claim-honest mid-refactor.

## ABI-specific guardrails

- Source wins over prose (`AGENTS.md` / `CLAUDE.md` / `GEMINI.md` stay siblings when conventions change).
- Claims: no unproven sharding, production FHE, native GPU dispatch, non-loopback hardening, QPS/latency/accuracy figures.
- Inside `src/`: relative `.zig` imports only (MCP handler group may `@import("abi")`).
- Prefer configurable temp dirs (`TMPDIR` / `SCRATCH` env) over hardcoded session paths in tools.

## Done criteria (before handoff)

- [ ] Chosen validation checklist (behavioral + modern + structural) checked
- [ ] Relevant gate log captured
- [ ] Active board (`tasks/todo.md`) updated if work was tracked
- [ ] No new unproven capability claims in docs
