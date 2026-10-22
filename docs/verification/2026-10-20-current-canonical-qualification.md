# Current canonical ABI and Abbey qualification — 2026-10-20

Local continuation of the earlier stable-source checkpoint. CI remains paused and
publication separate. Historical receipts retain their original source scope.

## Completed local repairs

- ABI commit `7b9519b6cfb25cc9bc9db2982e0a901318ba039b` adds owner-controlled
  durable replay refusal before the host emits a zero-binding stage record.
  Abbey records the refusal atomically using the original stored binding. It
  grants no execution handle, preserves original outcomes and consumed approvals,
  and reports an infrastructure failure if the refusal cannot be persisted.
- The exact replay diff passed independent review, 22 host tests, 19 owner tests,
  eight feature-enabled real host/store integrations and focused warning-denied
  clippy. The old behavior was reproduced before the repair.
- Abbey's child cancellation fixture now publishes its PID file by same-directory
  rename. The existing cancellation/reaping assertions passed ten consecutive
  runs; independent review approved the test-only repair.
- The existing Abbey stable migration separately fixes prediction-worker receipt
  readiness. A frozen candidate reproduced its missing repair: its failing library suite
  reported 841 passed and one failed. That existing owner implementation is now on canonical main; this
  lane did not create a second fix or competing compiler wrapper.

## Qualification evidence

Evidence directory:
`/Users/donald/.codex/verification/abi-abbey-canonical-20261020/`.

| Check | Actual result and scope |
| --- | --- |
| Current canonical baseline | Complete ABI gate exit 0 on Rust 1.99.1; 1,055 Rust test executions; ABI/WDBX/Abbey file manifests unchanged |
| Reviewed ABI with declared WDBX | Complete gate exit 0, 150.79 seconds, 1,055 Rust executions, zero failures/ignored/strip warnings; exact WDBX `2969d71a2c3a81a060dcfc6cf53b4617f13c8b94` |
| Copied ABI runtime artifacts | Help golden, nine additional local CLI/WDBX scenarios, seven MCP responses and exact 12-tool list passed; native signatures and loader inspection passed |
| Abbey desktop before replay/migration | Exit 0; 18 Python and 49 Rust test executions, frontend/bundle security, scratch-daemon reads and native link checks. This predates subsequent source changes |
| Live WDBX pairing attempts | Two final attempts failed formatting while the sibling's prototype files changed; no passing current-sibling claim derives from them |
| Abbey frozen candidates | PID readiness failures preserved; no pin-promotion claim derives from a failed or moving-source candidate |

The immutable ABI/Abbey/declared-WDBX snapshot is identified by
`candidate/source-manifest.json`, SHA-256
`8995d79815048cca35c4e5b7950fcf5365d6097260c6a968e0dc8374cd9ac848`.
Its source bytes and modes remained unchanged after the ABI gate. The canonical
sibling's concurrent changes are recorded separately in the receipt; they were
not inputs to that frozen gate. Test totals count executions, not unique cases.

## Coordinated migration checkpoint

`reviewed-owner-checkpoint.json` records the exact three ABI and four Abbey files.
The other Abbey owner's `combined-owner-source.json` matches all seven hashes.
Its combined producer gate passed with unchanged fingerprints. The first
combined root gate failed nested desktop IPC codegen with an older cached
lifecycle trait. Rebuilding that artifact without source-byte changes fixed it.
The complete combined root rerun passed: 3,831 Rust test executions, zero
failures/ignored and zero stripping warnings, with all three fingerprints
unchanged. Receipt: `abbey-combined-fresh-result.json` in the other owner evidence
directory `/Users/donald/.codex/verification/abbey-rust-1991-20261020/`; an exact
copy is retained in this evidence directory under `coordinated-receipts/`.
The separate combined desktop gate also passed, exit 0 with all three
fingerprints unchanged (49 Rust executions and 18 Python tests, frontend/build
security, scratch-daemon reads and native linking). Canonical Abbey main now
contains migration commit `30144342dd7e58935a2f818ffd3ff32a77c90162` and pins
Rust 1.99.1. Our reviewed four-file Abbey delta was preserved during integration.
The compiler is selected through the repository wrapper; no global default was
changed. The CI workflow remains unexecuted under the pause.

## Current dependency refresh

The authorized ABI root lock regeneration preserved byte-exact copies of both
working and HEAD locks under `dependency-refresh/`. Rust 1.99.1 generated the
same bytes as the pre-refresh working lock: SHA-256
`1dbbba40d78c3a321953b6fb3380dd2cbc5ef98c3ac87b50bc39915015c0e35f`.
Independent review found 51 version updates against HEAD, with existing major
versions and source identities preserved. No manifest was changed. The full
canonical gate passed in 322.35 seconds: 1,055 Rust test executions, zero failures,
zero ignored Rust tests, and zero stripping warnings. Python ran 173 policy
cases with eight unavailable central-sync cases skipped. Dependency security,
clippy, workspace build/tests, sibling conformance, Metal, benchmarks, rustdoc
and release builds passed. Receipt: `abi-dependency-refresh-receipt.json`.

Code and lockfiles did not change during that gate. Its source delta contains
only this report and other owners' documentation/metadata. The captured WDBX
HEAD was `1af409f10020eb359c4ea31f1dbd3d28722fc800`, with lock SHA-256
`e2fac4f3f125cecc771a0dc7d6f33eb1629972b3979ac0771849fdf61368680e`.
The before-manifest hash is
`eca86affdb7f8fc5ca3bdeaed4d60103d5ed6a3fc635f76b32c9471252d77b09`.
Copied release artifacts passed CLI help, nine local scenarios, seven MCP
responses with the exact 12-tool golden, native signatures and loader checks
(`dependency-refresh/artifact-smoke.json`). Ten hook tests, three docs policy
tests and shell syntax checks also passed.

Canonical Abbey `2ee9f7964428314c558bd43d70170eaecddb73c5` includes the
reviewed owner changes. `integrated-replay-equality.json` verifies all seven
ABI/Abbey implementation hashes still match the reviewed checkpoint. Its newer
lock is covered by the separate exact-pin consumer qualification below; ABI's
gate includes only the documented sibling WDBX conformance targets.

The separate complete-episode dossier review found no evidenced task-02
approval. `ratification-review.md` records owner decisions needed for identity,
time, authenticated scope, terminal receipts, COSE and signer history. It grants
no schema approval and changes no contract or consumer fixture.

## Later owner changes outside the passing cohort

`abi-final-handoff.json` detects subsequent WDBX complete-episode decoder edits
and Abbey route-log/workflow changes. They are preserved and are not inputs to
the passing ABI receipt above. The Abbey owner qualified exact proposed CI
pins ABI `7b9519b6cfb25cc9bc9db2982e0a901318ba039b` and WDBX
`5182bc3a60300e1dd152611025e774c4b906638b`. Both gates passed with unchanged candidate sources:
root 3,831 Rust executions and 98 Python tests; desktop 49 Rust executions and
18 Python tests. Installer acceptance and rollback checks passed. Receipt:
`coordinated-receipts/exact-ci-result.json`. Windows/Linux cross-check targets
were not installed, so those checks were skipped. The initial broader candidate failed the
personal-edition metadata-overflow cancellation assertion; a passing focused
rerun does not erase that failure. No current-whole-tree acceptance follows
from these narrower results. No commits or pushes were made after the user's
latest prohibition; CI remains paused.

## Limits retained

The replay precheck handles existing identities; concurrent admission races remain
fail-closed through existing strict binding/prepare checks and can have different
diagnostics. No new concurrency proof is claimed by the focused replay tests.
Eight personal central-sync policy cases were unavailable; CUDA compilation lacks
nvcc. Linux/Windows runtime and native GUI acceptance are separate. Missing WDBX
member-erasure, retention and key-custody APIs, real model training, live providers,
production deployment and independent audits remain outside this local receipt.
