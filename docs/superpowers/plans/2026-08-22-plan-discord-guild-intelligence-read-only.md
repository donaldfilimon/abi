# Program 3 read-only guild intelligence implementation plan

> Execute only inside isolated worktrees created from fetched `origin/main`.
> Preserve concurrent/user work, do not push or merge, and report local evidence
> separately from any later live Discord validation.

**Goal:** Establish the first C1 Program 3 contract slice, including a
deterministic local replay mechanism that can be extended to a complete C2
gate, as a presentation-neutral pure Abbey library over closed synthetic
recordings.

**Architecture:** ABI owns this focused normative design and plan. Abbey
`app_core` owns a pure model/analyzer/replay boundary. A recording source
implements only typed reads over an in-memory fixture. Deterministic functions
produce the five-view twin, alternatives, selected desired state, rollback
preview, and redacted status. A static guard rejects forbidden dependency,
write, storage, runtime-tool, and private-content surfaces.

**Technology:** Rust/Serde/SHA-256 in Abbey; Python standard library for the
static guard; Markdown contracts in ABI.

## Implementation status — 2026-08-23

Abbey's closed synthetic foundation was implemented at
`68a5c8530767883a14111954dbbb6b3bf6835414`, merged by
[`donaldfilimon/abbey#92`](https://github.com/donaldfilimon/abbey/pull/92) as
`edf0b029f23d38eea8a7076e8dcb17b9d77a2551`, and passed the exact-head and
post-merge macOS production gates. It establishes the bounded C1 data-only
claim described below; it does not complete Program 3.

The merged foundation covers closed synthetic replay, deterministic ordering,
five views, watermarks and coverage, lower-authority/schema/reference failure,
explicit alternative selection, rollback metadata, redacted status, and the
static read-only exclusion guard. Remaining acceptance work includes explicit
administrator-acceptance and hard-limit vectors, exhaustive permission and
unknown-target vectors, plan precondition/postcondition fields, and the later
complete C2 replay gate. No live Discord, provider, production, durable-store,
participant-consent, or Program 5 write evidence was produced.

## Task 1: Reconcile and focus the approved design

**Files:**
- Modify `docs/superpowers/specs/2026-08-22-spec-discord-guild-intelligence-and-execution.md`
- Create `docs/superpowers/specs/2026-08-22-spec-discord-guild-intelligence-read-only.md`
- Create this plan

Update the stale proposal status to approved-but-not-evidence. Extract only P3
read behavior, its C1/C2 evidence table, and explicit P5 exclusions. Review the
diff against the constitution and reconciliation, then run `./tools/check.sh`.

## Task 2: Drive the public contract from synthetic replay tests

**Files:**
- Create `tests/guild_intelligence_replay.rs`
- Create `tests/fixtures/guild_intelligence/community-risk.json`
- Modify `src/app_core/mod.rs`
- Create `src/app_core/guild_intelligence/{mod,model,permissions,analysis,replay}.rs`

Write an external integration test that loads only a fixture marked synthetic.
Assert synthetic owner/admin claims, owner-reference consistency, closed schema failure, coverage and per-object
watermarks, all five views, stable finding order, two substantive alternatives
plus do-nothing, explicit selection, non-executable desired-state and rollback
preview, fixed redacted status, and byte-identical replay. Run the test first
and observe the missing API failure. Implement the smallest closed public API,
then rerun until green.

## Task 3: Verify permission and negative boundaries

**Files:**
- Add focused unit tests beside the Program 3 modules
- Extend `tests/guild_intelligence_replay.rs`

Cover owner/admin override, role union, everyone/role/member overwrite ordering,
unknown overwrite targets, stale observations, missing references, invalid
selection, limits, and lower-authority refusal. Confirm normalization removes
input-order nondeterminism.

## Task 4: Add and gate the static exclusion check

**Files:**
- Create `tools/check_p3_readonly.py`
- Create `tools/tests/test_check_p3_readonly.py`
- Modify `check.sh`

Write a failing Python test for a synthetic violating tree and a passing test
for the real Program 3 module. Implement a lexical/manifest guard that scans
only the Program 3 production surface for forbidden network, Discord runtime,
durable-store, dynamic-tool, executor/actuator, write-operation, and private-
content vocabulary. Add it to the repository's strict gate.

## Task 5: Verify, inspect, and commit local evidence

Run focused Rust and Python tests, formatting, and static guard. Inspect `git
diff --check`, changed-file scope, and the absence of generated artifacts.
Run exact final gates:

```sh
# ABI worktree
./tools/check.sh

# Abbey worktree
./check.sh
```

Commit each clean isolated branch. Report exact base/commit,
focused tests, strict-gate results, limitations, and the evidence boundary:
local C1 contract evidence only; no complete P3 C2, live Discord, production, provider, participant-consent,
or Program 5 write evidence.
