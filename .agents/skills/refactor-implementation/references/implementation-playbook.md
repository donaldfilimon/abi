# Implementation Playbook

Safe execution steps for clean-slate modernization. Pair with **refactor-validation** after each significant change.

## Preconditions

1. Analysis checklist complete (targets + evidence).
2. Strategy chosen (direct / phased strangler / parallel modern impl).
3. Baseline gate green: `./tools/check.sh` (or the narrowest gate that covers the blast radius).
4. Frozen surfaces listed — do **not** resurrect legacy CLI names or invent MCP tools.

## Execution loop

### 1. Characterize before rewrite

- Prefer existing contract/unit tests that drive the **shipped** path.
- If missing, add a characterizing test that fails on wrong behavior without re-implementing the code under test.
- Capture baseline output when behavior is observational (CLI smoke, docs structure).

### 2. One semantic change at a time

- Prefer parallel files (`foo_v2.rs` / extracted helper) over in-place rewrites of hot contracts.
- Keep pure helpers extracted first; push IO/effects to edges.
- For Rust nightly: prefer explicit `Result`, avoid silent `unwrap`/`expect` on persistence and inference paths, keep clippy `-D warnings` clean.

### 3. Gate after each batch

| Change kind | Minimum gate |
| ----------- | ------------ |
| Docs prose only | Spot-check claims vs `docs/contracts/external-claims-audit.mdx`; `npx mint@latest validate` if nav/content structure changes |
| Tools scripts (no assertion rewrite) | Re-run the wired step or `./tools/check.sh` |
| Public feature API | Update `mod.rs` + `stub.rs`; `./tools/check.sh`; `./tools/check.sh` |
| CLI/MCP handler | Contract suites + `./tools/check.sh` (surface must stay frozen unless intentional) |

### 4. Cutover rules

- Do not delete the old path until parity is proven.
- Prefer expand → migrate callers → contract → delete.
- Leave the tree buildable and claim-honest mid-refactor.

## ABI-specific guardrails

- Source wins over prose (`AGENTS.md` / `CLAUDE.md` / `GEMINI.md` stay siblings when conventions change).
- Claims: no unproven sharding, production FHE, native GPU dispatch, non-loopback hardening, QPS/latency/accuracy figures.
- Inside `src/`: relative `.rs` imports only (MCP handler group may crate root / `abi_*` deps).
- Prefer configurable temp dirs (`TMPDIR` / `SCRATCH` env) over hardcoded session paths in tools.

## Done criteria (before handoff)

- [ ] Chosen validation checklist (behavioral + modern + structural) checked
- [ ] Relevant gate log captured
- [ ] Active board (`tasks/todo.md`) updated if work was tracked
- [ ] No new unproven capability claims in docs
