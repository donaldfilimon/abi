# A2 exit receipt — 2026-10-27

Plan: `docs/plans/2026-10-26-a1-remainder-a2-relay-requalification.md`, Task 8.
Coverage map and per-task results:
[`2026-10-26-a2-relay-requalification.md`](2026-10-26-a2-relay-requalification.md).

## Gate

No source changed after Task 7b (`1dd7f0b`). The A1 bound gate therefore also
serves as A2's exit gate: [`2026-10-27-a1-gate-receipt.md`](2026-10-27-a1-gate-receipt.md).

- Source: ABI `1dd7f0b`, WDBX `9d386dd`, both from clean shared clones.
  Pre- and post-gate captures are identical.
- Result: exit 0, `check: all green`. Rust ran 1,248 tests: 0 failed,
  0 ignored. `dependency-scan: RUN cargo-audit`, with no `SKIP`.
  `benchmark regression gate: PASS`.
- WDBX `HEAD` differs from the 2026-10-26 plan's observation (`952bb66`). The
  pinned revision is the one above.

## A2 commits

`5490a96` (coverage map), `abc1dbb` (Task 4 fix), `37f840a` (Task 5),
`e172ef5` and `80ae036` (Task 6), `59c29e1` (Task 7), `b07edd9`, `d76f4a9` and
`1dd7f0b` (Task 7b and review M1).

| Task | Result |
| --- | --- |
| 4 — relay certificate recheck after authority callback | defect, RED→GREEN |
| 5 — worker authority refusal / cancellation | characterization PASS |
| 6 — governed cancellation in final authority refresh | characterization PASS after a plan-calibration ruling (not a defect) |
| 7 — outbox terminal-persistence failure | characterization PASS |
| 7b — governed tool path certificate recheck; dispatch-time recheck (M1) | defects, RED→GREEN |

Every test the coverage map names (54 test functions) shows `... ok` in the
gate log, and none shows `FAILED`. The load-sensitive tests the A2 record
flagged also passed without reruns, at a host load of about 80–110:
`twenty_retained_manifests_do_not_starve_fresh_poll_and_epoch_reset_is_bounded`,
`native_five_second_proofs_expire_before_the_existing_live_proof_budget_deadlocks`,
`node_certificate_expiring_after_approval_never_reaches_tool_adapter` and
`governed_cancellation_during_final_authority_refresh_never_reaches_adapter`.

## Status change

Task 8's conditions hold: every mapped test passed, Tasks 5–7 recorded no
defect, and the gate is valid. C05 and C06 move from `U` to `Q local Mac`.
C03 stays `Q local Unix` and C04 stays `Q local Mac`.

Limits on all four rows: there is no proof for power loss, separate hosts,
public deployment or Windows/Linux runtime. The process kills and TCP loss are
local loopback fixtures on one Mac.
