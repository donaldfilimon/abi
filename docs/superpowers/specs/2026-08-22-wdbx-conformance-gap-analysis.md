# WDBX conformance gap analysis

Status: **observation.** Measured against `dev/active/abi` at `origin/main`
(`0278a2f` lineage) on 2026-08-22. Input to the canonical WDBX episodes spec
(`2026-08-22-spec-canonical-wdbx-episodes.md`), which the ratified
constitution section 13 calls Program 4.

Reference: `CSAPS_WDBX_Revised_2026.pdf` revision 2.0, sections 6.1 through 6.9
and normative requirements R1 through R12. The paper is a **proposed
architecture**; its requirements are proposed criteria, not obligations this
codebase previously accepted. Nothing below is a defect report against past work.
It is a distance measurement between what exists and what the vision specifies.

## Method and a warning about it

Every row was checked by reading the implementation, not by counting keywords.
That distinction is load-bearing: an initial keyword sweep scored four
requirements as present that are not present at all. Recording the false
positives, because the same trap will catch the next reader:

| Keyword hit | What it actually is | Spec item it does *not* satisfy |
| --- | --- | --- |
| `redact` (5 files) | Masking key material in `Debug` output (`v2/security.rs:45`) | 6.9 redacted derivative blocks |
| `tombstone` (`cluster/membership.rs`) | Permanently removed cluster node identities | 6.9 payload tombstones after cryptographic erasure |
| `retention` (`segments.rs:67`) | How many segments to keep on disk | 6.9 retention classes and legal holds |
| `supersede` (`wal.rs:481`) | A WAL from an older epoch being replaced | 6.5 supersession edges between episodes |

Section 6.9 is therefore **absent in full**, not partially implemented.

## Summary

`abi-wdbx` is 25,171 lines and implements most of the **structural** half of the
substrate and almost none of the **evidence** half.

Present and genuinely strong: a multi-parent causal audit DAG, content addressing
by SHA-256, Ed25519 signing machinery, MVCC with conflict sets, WAL plus segment
durability with CRC framing, cluster replication and read repair, a pluggable
retrieval scoring seam, and a bounded authenticated gRPC gateway.

Absent: everything that lets the system decide whether a retrieved episode
*applies*. There is no regime, no policy or model version, no schema version, no
signer identity on an episode, no contradiction or quarantine edge, and no
evidence-weighted retrieval. The store can prove a record was not modified. It
cannot yet answer why the record should be trusted, which is the substrate's
stated purpose (section 6.1).

## 6.2 Logical block schema

The specified `WDBXBlock` carries roughly 28 fields. The implemented
`V2AuditBlock` (`crates/abi-wdbx/src/v2/types.rs:116`) carries 8.

| Spec field group | Implemented? | Note |
| --- | --- | --- |
| `block_id` | Yes | `hash`, lowercase SHA-256 |
| `parent_hashes[]` | Yes | `parents: Vec<String>`, all observed causal heads |
| `created_at_utc` | Partial | `timestamp_ms: i64`; no separate acquisition versus receipt time (see 5.5) |
| `monotonic_time_range` | No | |
| `schema_version` | No | Zero occurrences crate-wide |
| `source_node`, `actor_or_agent` | Partial | `profile: String` is the only actor-ish field |
| `task_id`, `task_regime` | No | |
| `state_summary` (`reservoir_embedding`, `habituation_summary`, `regime_posterior`, `resource_state`) | No | |
| `observation_refs[]` | No | Vectors are referenced by `query_id`/`response_id` only |
| `retrieved_parent_blocks[]` | No | Distinct from causal `parents`; what the decision actually consulted |
| `action_sequence[]` | No | |
| `predicted_outcome`, `observed_outcome` | No | The prediction/outcome pair is the core of section 6.1's five questions |
| `utility`, `uncertainty`, `risk`, `novelty` | No | |
| `provenance` | No | `metadata: String` is opaque |
| `model_versions[]`, `policy_version`, `calibration_versions[]`, `safety_policy_hash` | No | |
| `payload_hash` | Yes | `hash` doubles as it |
| `signer_key_id`, `signature` | **No** | See 6.4 below |

Implication: the block records *that a query produced a response*, which is a
retrieval log. It does not record what was predicted, what was done, what
followed, or under which policy, which is what makes an episode an episode.

## 6.4 Serialization, content addressing, and signing

**Two independent failures against equation (30).**

The spec requires `c_t = CanonicalCBOR(header, payload, sort(parent_hashes))`,
then `d_t = SHA256(c_t)`, then `sigma_t = Sign(d_t)`.

`audit_hash` (`crates/abi-wdbx/src/versioned.rs:499`) computes:

```rust
let bytes = serde_json::to_vec(&input).expect("audit hash input serializes");
format!("{:x}", Sha256::digest(bytes))
```

1. **Encoding.** The digest is taken over `serde_json` output. Section 6.4
   explicitly rejects deterministic Protocol Buffers as a durable canonical
   representation across schemas, builds, languages, and library versions;
   `serde_json` is weaker still, since field order follows struct declaration
   order and there is no canonical number or string form. CBOR and COSE have
   **zero occurrences** in the crate.
2. **Ordering.** `AuditHashInput.parents` is `&'a [String]` passed through in
   insertion order. The spec's `sort(parent_hashes)` is not applied, so two
   writers observing the same head set in different orders produce different
   digests for the same logical block.

**Signing is at the wrong granularity.** Ed25519 exists and is used, but only in
`v2/security.rs` over transaction and segment objects. `V2AuditBlock` has no
`signature` and no `signer_key_id` field, so equation (32) is not applied to
episodes. Segment-level authentication proves a segment was written by a holder
of the key. It does not attribute an individual episode to a signer, which is
what 6.5's "source identity, role, and reputation" dimension needs.

Assessment: the DAG shape is correct and reusable. The commitment function needs
replacement, not adjustment.

## 6.5 Trust and semantic validity

Specified: eight evidence dimensions, exposed separately, never collapsed.

Implemented: `ScoreComponents` (`crates/abi-wdbx/src/temporal.rs:15`) with four
dimensions, collapsed multiplicatively.

```rust
pub fn combined(self) -> f32 {
    self.semantic * self.temporal * self.causal * self.persona
}
```

| Spec dimension | Implemented |
| --- | --- |
| Cryptographic validity and key status | No (verification exists at segment level, not surfaced as a retrieval dimension) |
| Source identity, role, reputation | No |
| Calibration and hardware health | No |
| Outcome validation and later contradiction | No |
| Model, policy, schema compatibility | No |
| Task-regime and safety-constraint compatibility | No |
| Age, supersession, retention status | Partial (`temporal` is exponential recency, which is not supersession) |
| Confidence interval or recorded uncertainty | No |

Two structural notes beyond the count. First, multiplicative combination is
precisely the opaque collapse invariant I3 of the constitution forbids, and it has
a sharp behavioral consequence: any single zero factor zeroes the result, so one
dimension can silently veto retrieval with no way for a caller to see which.
Second, the spec's handling of an invalid record differs by kind. Cryptographically
invalid means reject. Validly signed but semantically uncertain means retain,
inspectable, at reduced weight, with quarantine status or a contradiction edge.
`quarantine` and `contradiction` have zero occurrences, so the second path does
not exist; today a record is either stored or not.

The good news is that `HybridScorer` is already a seam. Extending it is an
additive change to an existing extension point, not a redesign.

## 6.6 Threat model

| Threat | Required control | State |
| --- | --- | --- |
| Block tampering | Canonical digest, signature verification, parent-hash verification | Partial. Digest and parent verification exist (`v2.rs:319` rejects invalid parent hashes, `v2.rs:326` rejects self-parenting); the digest is not canonical |
| Key compromise | Rotation, revocation, hardware-backed keys, incident quarantine, temporal key validity | Partial. Retained-generation rekey exists per `todo.md`; no quarantine, no temporal validity window |
| Semantic poisoning | Outcome validation, contradiction graph, source reputation, risk filter, retrieval explanations, human review | **Absent.** This is the largest single gap and the one section 8.8 benchmarks |
| Replay or stale policy | Policy and model version compatibility, staleness penalty, supersession edges, active constraint comparison | **Absent.** No version fields exist to compare |
| Cross-regime confusion | Regime posterior, constraint compatibility, hard negatives in retrieval training | **Absent.** No regime concept |
| Privacy leakage | Data minimization, content references, retention policy, access control, encryption, redaction and deletion | Partial. ChaCha20-Poly1305 encryption and owner-only file modes exist; retention, redaction, and deletion semantics do not |
| Provenance truncation | Required ancestry fields, completeness indicators, retrieval penalty for incomplete lineage | Partial. Ancestry is required and validated; no completeness indicator, no retrieval penalty |
| Availability loss | Fast-policy fallback, local cache, timeout budgets, replicated metadata, safety independence | Partial. Replication and read repair exist; the fallback and budget concepts belong to Program 4 |
| Index poisoning | Rebuildable indexes, block verification on materialization, index version and root hash | Partial. Indexes are rebuildable; no index version or root hash |

## 6.7 Service boundaries

Specified `MemoryService`: `Retrieve`, `ProposeWrite` returning a `WriteDecision`,
and `Verify`.

Implemented `WdbxGateway`: `PutVector`, `Search`, `PutKv`, `GetKv`,
`ResolveConflict`, `Stats`, `MembershipChange`, `WatchMutations`.

- `Search` maps to `Retrieve`, ranked on the four dimensions above.
- **There is no `ProposeWrite` and no `WriteDecision`.** `PutVector` and `PutKv`
  are unconditional. This means R6 (selective memory writes: store high-value
  episodes rather than an unbounded copy of every observation, with mandatory
  safety and failure retention as exceptions) and section 4.9's selective write
  gate have no implementation surface at all. This is an architectural gap, not a
  missing method: today the caller decides what to store, so write policy lives in
  every adapter rather than in the substrate. That is precisely the "four
  competing meanings of memory" problem the constitution exists to end.
- **There is no `Verify`.** Verification happens internally on open and
  materialization; it is not callable by a consumer that wants to check a block
  reference before acting on it.

The gateway is otherwise a good foundation: bounded, authenticated, TLS and mTLS
tested, with a streaming mutation watch that the spec does not require but Program
7 will want for shadow evaluation.

## 6.9 Retention and deletion semantics

Absent in full, per the false-positive table above. Required and unimplemented:
retention classes and legal or operational holds; cryptographic erasure or
deletion of encrypted payloads while retaining a minimal tombstone commitment;
redacted derivative blocks that link to but do not overwrite the original;
explicit revocation, supersession, contradiction, and quarantine edges; and
auditable garbage collection for unreferenced high-rate traces.

One adjacent behavior does exist and is compatible: `abbey`'s WDBX memory backend
never deletes and marks records `obsolete` instead. That is a supersession-shaped
primitive at the consumer layer that should be lifted into the substrate rather
than reinvented.

## R1 through R12

| Req | Subject | State |
| --- | --- | --- |
| R1 | Explicit state separation | Out of scope for the substrate; Program 4 |
| R2 | Multiple update schedules | Out of scope; Program 4 |
| R3 | Calibrated uncertainty and surprise | Absent from the substrate; no uncertainty field |
| R4 | Reversible habituation | Out of scope; Program 4 |
| R5 | Nonlearned safety bypass | Out of scope for the substrate; constitutional invariant A4 |
| R6 | Selective memory writes | **Absent.** No `ProposeWrite`, no write gate |
| R7 | Evidence-weighted retrieval | **Absent.** Four dimensions, none of them evidentiary |
| R8 | Origin is not truth | **Absent as a mechanism.** Nothing distinguishes cryptographic validity from semantic validity, because there is no semantic validity representation |
| R9 | Planner authority is bounded | Out of scope; constitutional invariant A3 |
| R10 | Deterministic replay | Partial. Deterministic migration and byte-identical golden fixtures exist; replay of a recorded stream to equivalent internal trajectories is not implemented |
| R11 | Complete-system accounting | Partial. `abi-telemetry` exists (396 lines) but does not account sensor, conversion, communication, host, storage, or control overhead |
| R12 | Reconstructible experiments | Partial. `abi` has a claims registry and golden fixtures; there is no manifest binding a result to code, model, schema, firmware, seed, calibration, hardware, and corpus hashes |

## What this implies for extraction

The extraction moves `abi-wdbx`, `abi-compute`, `abi-foundation`, `abi-core`, and
`abi-telemetry` into a substrate repository beneath ABI. Three consequences follow
from the analysis:

1. **Extract before contract work, not after.** Every gap above is additive to the
   crate. None requires reorganizing what moves. Doing the move first means the
   contract work happens once, in its final home.
2. **`abi-core` is dead weight in the move.** `abi-wdbx` has zero references to it;
   `cargo check -p abi-wdbx --all-targets` exits 0 with the dependency deleted.
   It still travels with the closure because four other crates in it use it, but
   the `abi-wdbx` dependency line should be removed.
3. **Do not rename toward CSAPS service names during the move.** `abi-sea`'s
   eight-signal scorer resembles the escalation arbiter and `abi-wdbx-gateway`
   resembles `MemoryService`, but renaming during an extraction converts a
   resemblance into an architectural commitment without a spec. Recorded here as
   analysis; deferred to Programs 4 and 6.

## Honest residual

This analysis covers sections 6.1 through 6.9 and R1 through R12. It does not
cover sections 4, 5, 7, 8, 9, or 11, which specify the CSAPS control hierarchy,
training curriculum, benchmark protocol, and hardware platform. Those are out of
scope for the substrate and belong to Programs 4 and 7. No claim is made here
about whether the CSAPS architecture works; the paper says that is unvalidated,
and nothing in this document tests it.
