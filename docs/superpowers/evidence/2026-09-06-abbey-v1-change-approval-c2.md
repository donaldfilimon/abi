# Abbey v1 change-set and approval replay evidence

Date: 2026-09-06  
Evidence ceiling: **local C2 for the PR #814 change-set/approval kernel slice**  
Predecessor: [PR #814](https://github.com/donaldfilimon/abi/pull/814) and
[C1 evidence](2026-08-24-abbey-v1-contract-kernel-c1.md)  
Contract major/revision: 2/2  
Corpus digest: `3ffd487bdc497b7ce54b8c29978a3686dcbffdb66a85957a0ee4f99ba576cdfd`

## Scope and implementation

`crates/abi-capability/src/change.rs` now recomputes a proposal's commitment
and rechecks the existing ChangeSet construction rules before issuing an
approval. The new `ChangeApproval::validate_for` checks an existing decision's
proposal binding, live window, approval state, role floor, identity separation,
and dual-control requirements. It is pure and does not consume, revive, or
execute a decision.

The original tampering regression was observed failing before the fix:
`approving_a_tampered_proposal_fails_closed` accepted a snapshot edit that
retained the original commitment. It passes with commitment verification.

The replay driver and its frozen JSON recording live under
`crates/abi-capability/tests/`, outside `contracts/abbey`. The corpus remains
data-only, with its 113 artifacts, schemas, manifest, revision, and digest
unchanged. The 13-command CLI, 12-tool MCP catalogs, and surface goldens are
unchanged.

## Replay witness

`tests/fixtures/change_set_replay.json` pins the corpus revision/digest, kernel
package version, exact capability/version and package commitment, synthetic
scope and identities, generator commitment, proposal fields, and injected
millisecond timestamps. The policy is the compiled approval implementation;
no model, provider, random source, wall clock, actuator, or store participates.
The test checks the fixture's contract pins against the current manifest and
its kernel package version against the compiled crate.

The 17 recorded events cover issuance, validation immediately before and at
approval expiry, cancellation-state invalidation, unchanged cancelled-state
rejection, a new decision, snapshot tampering, refusal to approve the tampered
proposal, restoration, regeneration with a new commitment, refusal to reuse
the old decision, explicit new approval, and final expiry refusals.

Every event is checked against a reviewed literal acceptance/refusal vector.
The full serialized transcript, including every emitted approval field, must
also match a frozen SHA-256 commitment. The golden commitment was calculated
independently with Python's standard-library JSON encoder and SHA-256 from the
reviewed decisions; the test has no golden-update path. Two fresh OS processes
decode the same recording, run the production approval APIs, validate the
golden result, and emit byte-identical transcripts.

- Initial proposal: `sha256:5f17724aed56d466994cf1bfffdb30f3b6bae78554c0dab800423cf06d1d54b9`
- Replay transcript: `sha256:4eed0c7e51945ada2dbd7e17af9d57ead9dcb84005e45274453a902bbb28d3c7`

These are local test transcripts of approval values and refusals, not wire
receipts or signatures. Restart evidence is fresh test-process reconstruction
from the frozen input, not daemon recovery or durable lifecycle recovery.

## Focused coverage

`crates/abi-capability/tests/change_replay.rs` contains nine tests covering:

- Golden decisions/transcript and byte-identical replay in fresh processes.
- 27 independent field mutations, covering every serialized ChangeSet field
  and every nested principal/scope field; new and reused approvals fail closed.
- Regeneration invalidates the previous approval and permits an explicit new
  decision bound to the regenerated commitment.
- Ten resealed malformed-proposal cases cannot bypass the existing construction
  rules, including author identity, capability syntax, lifetime, prepared TTL,
  and required rollback commitment.
- Proposal creation and exclusive expiry, approval exclusive expiry, and
  `u64::MAX` refusal boundaries using injected timestamps.
- `Denied`, `Cancelled`, `Expired`, and `Consumed` states remain invalid after
  serialization/reload; validation leaves the decision unchanged.
- Reloaded approval binding, role, identity separation, expiry, and A5 dual
  control, including rejected service, weak, duplicate, requester, and proposer
  coapprovers.

## Verification observed

- `./tools/cargo.sh test -p abi-capability --test change_set --test change_replay < /dev/null`:
  **14 passed, 0 failed, 0 ignored** (five existing/extended change-set tests and
  nine replay tests).
- `./tools/check.sh < /dev/null`: baseline, implementation, and documentation runs exited
  **0**, including policy and CI-contract checks, Python/Rust Abbey corpus
  verification, Rust size limits, formatting, warning-denied clippy, build,
  workspace tests, available platform feature checks, local benchmark guard,
  and rustdoc. CUDA compilation was explicitly unavailable because `nvcc` is
  absent; this is not CUDA verification.
- `.agents/skills/docs-validate/validate.sh < /dev/null`: **exit 0**, Mintlify
  build/configuration/page validation passed. The driver used its bundled
  Node 24.19.0 fallback because the default Node 26.8.1 is unsupported.

The initial rebase onto fetched `origin/main` was already current at
`6321a47bf4c48a5f58caf2df0eb631b5a2ecdcee`. During this work a concurrent commit,
`8d4e9da6f88e21fe9420a3255808018adf671a85`, included the tracked kernel fix and
regression alongside unrelated skill work. It was preserved. The replay tests
and fixture were subsequently included in shared commit
`92e3bb14`; this evidence records the verified combined slice.

Local verification uses `nightly-2026-09-01` on macOS ARM64 and the existing
sibling WDBX checkout at `b82a7d3e26db9a00ce57b98efd24719683466eb1`. That differs
from ABI's CI pin `14cb1341cb454bd3f887c4e54a83f8c42775a91d`; the sibling was not
changed. The focused `abi-capability` tests do not depend on WDBX. Local gate
success is not hosted-CI or pinned-sibling CI evidence.

## Evidence boundary

This qualifies only the local immutable proposal/approval slice. Full
federation C2 remains unproven: this does not replay wire receipts, negotiation,
transport, policy/model version changes, or a daemon's persistent state. The
recorded cancellation event supplies `Cancelled` state to the pure validator;
cancellation delivery, trusted state provenance, durable single-use decision
IDs, and execution authorization remain caller responsibilities. A digest is
a content commitment, not proof of who authored an approval.

No stable-Rust or Swift consumer, daemon deployment, WDBX write, provider call,
Discord effect, live consent, installed artifact, production security,
cross-platform replay, C3-C7 promotion, or full Program 7 qualification is
claimed. The full federation design remains C0; Program 1 corpus evidence
remains C1.
