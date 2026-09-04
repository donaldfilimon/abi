# Canonical WDBX Episodic Contract

> **Constitutional mapping.** This file was written against the seven-program
> list Donald gave in conversation. The ratified constitution's section 13
> "Delivery programs" uses a different numbering, and the constitution wins.
> In section 13's terms this document specifies **Program 4, Canonical WDBX episodes and claims.**
>
> The filename is therefore name-based rather than numbered, so no numbering is
> asserted. Nothing in section 13 was renumbered: section 15 reserves amendment
> to Donald, and the collision is raised as one request covering the whole set
> rather than five independent ones.

**Reconciled ownership.** This is constitutional Program 4,
`canonical-wdbx-episodes-claims`. Program 1, `abbey-contracts`, owns the
language-neutral episode, evidence, claim, and tombstone wire schemas and
fixtures. Program 4 owns their durable WDBX behavior; transport JSON is never a
canonical WDBX commitment input.


Status: **proposed.** Written 2026-08-22. See the constitutional mapping below for its section 13 program number.

Depends on the ratified `2026-08-22-abbey-system-constitution.md` and the
measured `2026-08-22-wdbx-conformance-gap-analysis.md`. Where those disagree
with this document, they win.

**Reference correction, 2026-08-22.** This spec was drafted against the earlier
199-line constitution, whose lettered invariants (A2, A3, I3, I5) and L0-L8
ladder were superseded when Donald expanded the document to 907 lines at
`c113aec`. The ratified equivalents are the numbered decision register in
section 14 and the **C0 through C7** ladder in section 11. Mappings used below:
I1 -> decision 26 (integrity never implies truth), I2 -> decision 27 (similarity
never implies applicability), retention and deletion -> decisions 28 through 30,
projections -> decision 25. There is no C8.

## The one-sentence problem

WDBX can prove a record was not modified. It cannot yet answer why the record
should be trusted, and those are different questions.

## Scope

This spec covers the **durable contract**: what an episode is, how it is
committed, how trust is represented, what may be written, and what deletion
means. It does **not** cover who decides to write (Program 6's arbiter), who is
allowed to act on a retrieval (Program 2), or how a capability earns promotion
(the cross-cutting evaluation discipline, with adaptive behavior owned by
Program 6).

## Current, verified

Read from `~/dev/active/wdbx` at `feb16fd`. Everything in this section was
confirmed by reading source, not by keyword search, because a keyword sweep
scored four requirements as present that are not.

- **Multi-parent causal audit DAG.** `versioned.rs:430` appends a block whose
  parents are all currently observed heads. `v2.rs:319` rejects an invalid
  parent hash and `v2.rs:326` rejects self-parenting.
- **`V2AuditBlock`** (`v2/types.rs:116`) carries 8 fields: `hash`, `parents`,
  `timestamp_ms`, `sequence`, `profile`, `query_id`, `response_id`, `metadata`.
- **Commitment** is `SHA256(serde_json::to_vec(AuditHashInput))`
  (`versioned.rs:499`), over parents in **insertion order**.
- **Signing** exists (Ed25519, `v2/security.rs`) but over transaction and
  segment objects, not over audit blocks. `V2AuditBlock` has no `signature` and
  no `signer_key_id`.
- **Retrieval scoring** is `ScoreComponents` (`temporal.rs:15`): `semantic`,
  `temporal`, `causal`, `persona`, combined **multiplicatively**.
- **Gateway RPCs** are `PutVector`, `Search`, `PutKv`, `GetKv`,
  `ResolveConflict`, `Stats`, `MembershipChange`, `WatchMutations`.
- **Absent entirely:** CBOR, COSE, `schema_version`, `policy_version`,
  `signer_key_id`, `task_regime`, `regime_posterior`, `quarantine`,
  `contradiction`, and all block-level retention or deletion semantics.

> **Updated 2026-09-04 against source. Three items on that "absent entirely"
> line are no longer absent.** WDBX v3 landed 2026-08-24 (PRs #2 `7ab61b5`,
> #3 `5b41eca`) in `../wdbx/crates/abi-wdbx/src/v3/`:
>
> - **CBOR is implemented.** `v3/commitment.rs` (328 lines) defines the
>   `abbey-cbor-episode-v1` deterministic profile (`PROFILE_NAME`, line 38) and
>   hashes SHA-256 over the canonical bytes, with **parents sorted**
>   (`parents.sort_unstable()`, line 144) — i.e. equation (30) as specified.
> - **`schema_version` and `policy_version` exist** on the v3 episode types
>   (`v3/episode/types.rs`), alongside `evidence_level` and `contract_digest`.
> - `v3/episode/store.rs` (835 lines) is a durable append-only
>   `episodes.v1.jsonl` ledger with `propose_write`, `preview_commitment` and
>   `retrieve`.
>
> **Still genuinely absent**, verified by repo-wide grep on the same date: COSE,
> `signer_key_id` on episodes, `task_regime`, `regime_posterior`, `quarantine`,
> and block-level retention/deletion semantics. Beware the near-miss:
> `signer_key_id` does occur in `abi-worker` for task and cancellation signing,
> and `contradiction` occurs in `abi-sea` as a retrieval tag. Neither satisfies a
> conformance row here.
- **Consumers today:** `abbey` writes memory records as JSON into the durable KV
  space under `mem/<id>`; `abbey-bot` writes a single-file `# ABI-WDBX v1`
  projection, proven loadable by `abi-wdbx` in
  `../wdbx/crates/abi-wdbx/tests/abbey_bot_projection_conformance.rs`.

## Proposed

### 1. The episode

An episode answers five questions: what happened, what was predicted, what was
done, what followed, and why this record is trusted. `V2AuditBlock` answers
roughly one of them (a query produced a response), which makes it a retrieval
log rather than an episode.

Proposed `EpisodeBlock`, superseding `V2AuditBlock` rather than extending it,
because every added field would otherwise be optional and therefore unenforced:

```
identity      block_id, schema_version, payload_hash
time          created_at_utc, monotonic_time_range
origin        source_node, actor_or_agent, signer_key_id, signature
task          task_id, task_regime
state         state_summary { habituation_summary, regime_posterior,
                              resource_state, reservoir_embedding? }
lineage       parent_hashes[], retrieved_parent_blocks[]
decision      action_sequence[], predicted_outcome, observed_outcome
judgment      utility, uncertainty, risk, novelty
versions      model_versions[], policy_version, calibration_versions[],
              safety_policy_hash
provenance    provenance
```

Two fields carry most of the value and neither exists today.
`retrieved_parent_blocks` is distinct from causal `parents`: it records what the
decision actually consulted, which is what makes "reused a right-looking answer
for the wrong reason" detectable after the fact. And the
`predicted_outcome`/`observed_outcome` pair is what turns a log into evidence,
because a divergence between them is the only signal that generalizes.

High-rate traces and large model artifacts are **referenced by digest**, never
embedded in the signed block. `observation_refs[]` holds those digests.

Missing fields and zero-valued fields must be distinguishable. A schema upgrade
must never reinterpret a previously signed payload without retaining the
original canonical bytes or their digest.

### 2. Commitment

Replace the commitment function. Both halves of the current one fail the
specification and each fails independently.

```
c = CanonicalCBOR(header, payload, sort(parent_hashes))
d = SHA256(c)
sig = Ed25519_sign(signing_key, d)
```

**Encoding.** `serde_json` output is not a canonical form: field order follows
struct declaration order, and there is no canonical number or string
representation. CSAPS section 6.4 explicitly rejects even deterministic Protocol
Buffers as a durable canonical representation across schemas, builds, languages,
and library versions. Canonical CBOR with a documented profile, wrapped in a
COSE-compatible structure, is the target.

**Ordering.** `AuditHashInput.parents` is passed through in insertion order, so
two writers observing the same head set in different orders produce different
digests for the same logical block. Sorting is not cosmetic; without it the DAG
cannot be deduplicated and cross-node agreement is accidental.

**Granularity.** Signing moves to the episode. Segment-level authentication
proves a segment was written by a key holder; it cannot attribute an individual
episode to a signer, which is what the source-identity evidence dimension needs.

**Migration.** `V2AuditBlock` records keep their original hashes and original
bytes. A v3 reader accepts both, reports `schema_version`, and never rewrites a
v2 digest, because rewriting it would destroy the only evidence that the old
commitment was ever valid.

### 3. Trust is eight dimensions, never one score

`ScoreComponents::combined()` multiplies four factors into one number. That is
the opaque collapse the constitution forbids (section 5, and decisions 26 and 27 read together), and it has a sharp
behavioral consequence: any single zero factor silently vetoes a result with no
way for the caller to see which one did it.

Proposed: retrieval returns each dimension separately, plus an explanation.

```
cryptographic     valid | invalid | unverifiable; key status
source            identity, role, reputation
calibration       sensor/model calibration and hardware health
outcome           validated | contradicted | unobserved
compatibility     model, policy, schema
regime            task-regime and safety-constraint compatibility
lifecycle         age, superseded_by, retention class, quarantine
confidence        recorded uncertainty or interval
```

Two disposition rules, and they differ by kind. A **cryptographically invalid**
record is rejected. A **validly signed but semantically uncertain** record is
retained, inspectable, at reduced retrieval weight, carrying a quarantine status
or a contradiction edge. The second path does not exist today, so a record is
currently either stored or absent, with nothing in between.

`HybridScorer` is already a pluggable seam, so this is an extension of an
existing extension point rather than a redesign. Ranking must be evaluated
against a plain vector database using equivalent embeddings, storage, and
candidate count, or the added machinery is unproven.

### 4. The selective write gate

`PutVector` and `PutKv` are unconditional, so write policy currently lives in
every adapter. That is the mechanism by which four, and in fact five, competing
meanings of memory arose.

Proposed `MemoryService`:

```
Retrieve(MemoryQuery)          -> MemoryResults      # evidence-weighted
ProposeWrite(EpisodeCandidate) -> WriteDecision      # the gate
Verify(BlockReference)         -> VerificationResult # callable before acting
```

`ProposeWrite` returns a decision with a reason, and the reason is recorded.
Mandatory retention classes bypass the gate entirely: **safety events and
failures are always retained**, because a system that learns to stop recording
its own failures is worse than one that records nothing.

`Verify` matters because verification currently happens internally on open and
materialization and is not callable by a consumer that wants to check a block
before acting on it. Under section 11's ladder, acting on an unverified
retrieval claims C6 evidence (an operator witnessed the exact outcome) on the
strength of C1 (source conforms under test).

### 5. Deletion without an undeletable record

Section 6.9 is absent in full. The keyword hits that suggested otherwise are
unrelated: `redact` is `Debug` masking of key material, `tombstone` is dead-node
cluster membership, `retention` is a segment count, `supersede` is WAL epoch
rotation.

Immutability of a commitment does not require indefinite retention of every
payload. Proposed:

- **Retention classes** with legal and operational holds.
- **Cryptographic erasure**: destroy the payload key, retain a minimal tombstone
  commitment, so the block's position in the DAG survives while its content does
  not.
- **Redacted derivative blocks** that link to, and never overwrite, the original.
- **Explicit edges**: revocation, supersession, contradiction, quarantine.
- **Auditable garbage collection** for unreferenced high-rate traces.

The design constraint is stated in one line: provenance must make deletion
attributable **without turning privacy-sensitive content into an undeletable
permanent record**. `abbey`'s existing behavior (never delete, mark `obsolete`)
is a supersession-shaped primitive at the consumer layer that should be lifted
into the substrate rather than reinvented.

### 6. The five meanings of memory, and which survive

| Implementation | Disposition |
| --- | --- |
| `abi-wdbx` v2 (canonical) | Becomes v3. The one authority. |
| `abbey` memory backend | Consumer of the substrate. Its `obsolete` semantics get lifted into section 5. |
| `abbey-bot` `# ABI-WDBX v1` projection | **Stays a projection** (decision 25), and stays independent. It pins stable Rust 1.97.1 while `abi-compute` needs nightly `portable_simd`, so no toolchain compiles both. Conformance stays fixture-based and gated on both sides. It must declare which episode fields it drops. |
| Swift `AbbeyBot` store at `~/.abbey/wdbx/` | **Currently unreconciled.** 862 records, 18-dimension vectors, camelCase JSON with UUID ids and a `manifest.json` plus `segments/seg-0000.jsonl` layout. `abi wdbx` cannot read it. Program 3 must either define a projection contract it satisfies or migrate it. |
| `abbey` WDBX bridge naming | Cosmetic. `abbey_store_base` already documents that Abbey opens a directory while `abi wdbx` splits parent plus base name. |

The Swift store is the largest unresolved item and it holds real data.

## Sequencing

1. Sorted parents plus canonical CBOR encoding, v2 digests preserved unchanged.
2. `EpisodeBlock` schema and episode-level signing.
3. Evidence dimensions exposed separately, ranking still backward compatible.
4. `ProposeWrite` and `Verify` on the gateway; adapters migrate off unconditional
   writes.
5. Retention, redaction, and erasure.
6. Reconcile the Swift store.

Steps 1 and 2 are prerequisites for everything else, because a trust dimension
computed over a non-canonical digest cannot be reproduced by a second reader.

## Falsification

This program is wrong if evidence-weighted retrieval does not beat a plain
vector database with equivalent embeddings, storage, and candidate count, on
repeated-failure rate and stale-policy reuse. That comparison is Program 7's to
run, and it must be able to return "no benefit."

## Honest residual

**Corrected 2026-09-04: the sentence "Nothing in this document is implemented"
is no longer true, and the correction must not be over-read either.** WDBX v3
implements the commitment and the episode store (see the update box in the
Baseline section). What it does **not** have is a single consumer: `grep "v3::"`
outside `src/v3/` matches only its own two test files
(`tests/v3_canonical_commitment.rs`, `tests/v3_episode_store.rs`), and nothing in
`abi/`, `abbey/` or `abbey-bot/` references it. The **live** write path is still
unchanged from this document's baseline: `V2AuditBlock` with 8 fields and
`audit_hash = SHA256(serde_json::to_vec(...))` over **insertion-ordered**
parents.

So the accurate status is: **implemented and tested, shipped to no one.** Treat
v3 as Proposed for any claim about system behaviour, and as Current only for
claims about what code exists. Note also that `episodes.v1.jsonl` is a further
on-disk memory format that nothing writes today; it becomes real the moment a
consumer is wired.

The remainder of this residual still stands as written. The CSAPS paper it derives from is a
proposed architecture whose own status box states the integrated system has not
been empirically validated, and its quantitative thresholds are acceptance
targets rather than results. Independent cryptographic review of the commitment
and erasure design is required before any production claim and cannot be
self-certified.
