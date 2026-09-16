# Memory-edge episodes: quarantine, contradiction, and resolution on the WDBX gate

Status: **approved by Donald 2026-09-16 05:3x, with the recommended answers (see §9).** Written 2026-09-16. This is a written
revision under constitution section 15. Nothing in it is authorized for
implementation until Donald's approval is recorded in §9 with a date; the plan
in §7 is what would follow, not what exists.

Extends `2026-09-06-spec-memory-candidate-episodes.md` (approved 2026-09-06,
implemented in wdbx `14cb134`) and proposes two decision-register entries (86
and 87) for constitution section 14. Where the ratified constitution disagrees
with this document, the constitution wins.

## 1. The problem, in one paragraph

The constitution requires "contradiction, supersession, revocation,
resolution, and quarantine edges" in the canonical record (section on the
episode record, and item 30: they "remain visible"), and says correction
"never silently overwrites history". The v3 store implements two of these
edges, `supersedes` and `forgets`, as fields of `MemoryCandidate`. It has no
way to say "this memory is suspect" or "these two memories disagree", and no
way to lift either. The 2026-08-22 gap analysis calls semantic poisoning the
largest single gap. Per-episode signing landed first (wdbx `56767f7`), as
Donald decided on 2026-09-16 04:1x, so edges can now be attributed to a
signing writer. This document adds the edges that were deferred until then.

## 2. What the constitution already decides (not reopened here)

- **Visibility.** A quarantined or contradicted memory stays in the ledger and
  in the adapter's projection. The store never filters on these edges.
  Retrieval weighs them, together with the other factors the constitution
  lists (relevance, regime, provenance, versions, staleness, reuse risk,
  contradiction, constraints). A quarantine does not delete anything.
- **Reversibility.** A quarantine or a contradiction is lifted by a new
  `resolves` edge that names it. The original edge is never changed or
  removed.
- **Committed, not detached.** Edges are part of the canonical record, so
  they cannot be stored beside the record the way signatures are.
- **Guild isolation.** An edge may name only digests admitted in its own
  guild.

## 3. The amendment

### 3.1 Envelope route (measured 2026-09-16, not assumed)

`canonical_memory_candidate` (`wdbx/crates/abi-wdbx/src/v3/episode/store.rs:898`)
writes every `MemoryCandidate` field into the canonical map, using `Null`
when a field is absent. `tests/v3_memory_candidate.rs:18` pins
`GOLDEN_DIGEST = 3c19a479…c768bc`, and abbey-bot's
`src/episode_gate/tests.rs:541` reproduces that digest byte for byte. Adding
fields to `MemoryCandidate` would therefore change the digest of **every**
existing memory candidate, including already-signed ones. Under "no dual
canonical writers" that is a schema migration, not an extension.

Adding a new `EpisodeEvent` variant only adds a match arm in `canonical_event`
(`store.rs:854`). Every existing envelope, the golden digest, and abbey-bot's
copy stay byte-identical. **This document takes that route.**
`MemoryCandidate`, its encoding, and `supersedes`/`forgets` do not change.

### 3.2 The event

```
EpisodeEvent::MemoryEdge {
    recorded_by: ActorRef,       // see §3.4 for who may record each kind
    edge:        MemoryEdge,
}

MemoryEdge {
    kind:        MemoryEdgeKind, // quarantines | contradicts | resolves
    target:      [u8; 32],       // episode digest this edge names
    counterpart: Option<[u8; 32]>, // required for contradicts, absent otherwise
    reason:      EdgeReason,     // closed, content-free (§3.5)
}
```

The canonical map uses the keys `kind`, `target`, `counterpart` (`Null` when
absent), and `reason`, following the existing deterministic-CBOR profile. The
event's kind label is `memory_edge`.

### 3.3 Transition and graph rules

A memory edge is a **single-event operation**, like a memory candidate. It is
accepted only as the first event of a fresh `operation_id`, with
`previous_digest` absent. The store records it as terminal (`completed`) in
the same append, and any later event on that operation is
`InvalidTransition`. Edge episodes carry no payload and add nothing to
`payload_bytes`, but the record's own serialized size still counts against
the storage budget. `source = discord_voice` is rejected, as it is for
candidates. `learning_enabled`, `quiet`, replay, binding, and budget rules all
apply unchanged.

For each kind, with "live candidate" meaning a memory-candidate digest this
guild has admitted and not forgotten:

- **`quarantines`**: `target` must be a live candidate, and `counterpart` must
  be absent. Quarantining an already-quarantined candidate is
  `InvalidTransition`, so one open quarantine exists per target at a time.
- **`contradicts`**: `target` and `counterpart` must be two distinct live
  candidates with equal `member_scoped`. They are stored with
  `target < counterpart` (lexicographic byte order, the same rule as sorted
  `parent_digests`), so one disagreement has exactly one encoding. A second
  open contradiction on the same pair is `InvalidTransition`.
- **`resolves`**: `target` must be the digest of an open `quarantines` or
  `contradicts` edge episode in this guild, and `counterpart` must be absent.
  Resolving closes that edge. A `resolves` edge cannot itself be resolved,
  and resolving an already-closed edge is `InvalidTransition`.

Interaction with the existing edges:

- **`forgets` stays unconditional.** Deletion must never be blocked by
  quarantine, because a quarantined memory is the one most likely to need
  erasing. Forgetting a candidate leaves any edges that name it in place
  (they are content-free). No new edge may name the forgotten digest, but an
  open edge that already names it can still be resolved.
- **`supersedes` still works on a quarantined candidate.** A correction is
  how a suspect memory usually gets fixed. The new candidate starts
  unquarantined, and the old digest's quarantine stays open until it is
  resolved.

The store's rebuilt `GuildMemories` gains the open quarantine set, the open
contradiction pairs, and the edge-episode index needed for `resolves`. All of
them are rebuilt from the ledger on open, so the live path and the replay path
check the same rules, as they already do for `admitted`/`forgotten`.

### 3.4 Authorization (decided 2026-09-16: the asymmetric rule)

The store must decide which `ActorKind` may record each kind of edge. The
proposal is the asymmetric rule:

- `quarantines` and `contradicts` may be recorded by a `Service` actor (the
  adapter or Abbey's own checks). Quarantine is the protective direction and
  has to be fast when poisoning is suspected.
- `resolves` requires a human governance actor (`GuildOwner`,
  `GuildAdministrator`, `GuildManager`, or `OrganizationOwner`), never
  `Service` and never `HumanSubject`. Lifting a quarantine increases what
  retrieval may trust, which makes it a consequential action.

Alternatives: (b) any service actor may record all three kinds, which is
simplest but lets an automated path clear its own flags; (c) all three kinds
require a human, which is safest but slows down the stopgap the gap analysis
asks for.

### 3.5 Reasons

`EdgeReason` is a closed, content-free enum:
`source_untrusted | signature_invalid | policy_violation | operator_report |
conflicting_observation | superseded_evidence | reviewed_valid |
reviewed_invalid`. The store checks it against the edge kind:
`reviewed_*` only on `resolves`, `conflicting_observation` only on
`contradicts`, and the rest only on `quarantines`. Free text is never
admitted, so the ledger stays content-free.

### 3.6 Contradiction scope (decided 2026-09-16: candidates only)

The proposal lets `contradicts` name only two memory candidates, since
candidates are the only memories in the ledger. The alternative would also
allow operation episodes (a proposal or execution contradicting a memory).
That would require a regime/outcome vocabulary the gap analysis assigns to
Program 4, so the proposal defers it.

## 4. The adapter contract

The adapter keeps retrieval. When it reads its projection, it must be able to
tell, for each record, whether the record is under open quarantine or has an
open contradiction, and it must surface that state rather than drop the
record. The store exposes a read over the rebuilt state (the digest's open
quarantine, open contradiction counterparts, and the closing `resolves`
digest, if any). The gateway's `VerifyEpisode` gains matching additive fields,
the same way it gained `signature_status`. How strongly retrieval
down-weights such a record is a Program 4 scoring question, not part of this
revision.

## 5. Decision-register entries (proposed)

> 86. Quarantine, contradiction, and resolution are single-event memory-edge
>     episodes. They never alter the records they name, never hide them, and
>     never block `forgets`; a quarantine or contradiction ends only by a
>     `resolves` edge naming it.
>
> 87. New memory vocabulary enters the canonical record as new event
>     variants, not as new fields on an existing committed type, so that
>     existing digests and their signatures stay valid without a migration.

## 6. What this does not do

- No retrieval scoring, regime compatibility, R3 uncertainty, or staleness
  penalty. Those belong to Program 4.
- No `revokes` edge. Revocation of an authority or key is a separate concern
  from memory truth, and signing explicitly left key revocation out of scope.
- No legal hold, cryptographic erasure, or redacted derivative blocks. These
  are the rest of §6.9 and have their own forcing functions.
- No change to `MemoryCandidate`, its encoding, the golden digest, or any
  existing record.
- No automatic quarantine policy. This revision only lets a quarantine be
  recorded; deciding when to record one is the adapter's job.

## 7. Implementation plan, contingent on approval

1. **wdbx** (`crates/abi-wdbx/src/v3/episode/`): `MemoryEdgeKind`,
   `EdgeReason`, `MemoryEdge`, the `MemoryEdge` event variant and its
   canonical arm, shape validation (§3.5 and `counterpart` presence), the
   §3.3 graph rules on the live and replay paths, the §3.4 actor rule, and the
   §4 read. Tests: one per rejection, open/close/re-open cycles,
   forget-while-quarantined, supersede-while-quarantined, contradiction pair
   ordering, and a replay of a mutated edge line (`Corrupt`). A **new** golden
   fixture pins an edge episode's digest. **The existing golden test must
   pass unchanged; that is the compatibility proof for decision 87.** Gate:
   `cargo fmt --all --check`, `cargo clippy --workspace --all-targets`,
   `cargo test --workspace`.
2. **abi**: CI `WDBX_REVISION` pin bump, `VerifyEpisode` additive fields,
   `abi wdbx episode verify` output, and the register entries in the
   constitution. This is also where the unpushed gateway commits `b8b8d0c6`
   and `0cd6eb41` (currently pinned to `56767f7`) get re-gated against the new
   pin. Gate: `./tools/check.sh < /dev/null`.
3. **abbey-bot**: the transcription of the event encoding gains the new arm,
   with the new golden copied byte for byte. Its `deny_unknown_fields`
   consumers (`episode_gate/`, `memory_gate.rs`) only accept what abbey-bot
   itself writes, so nothing breaks until abbey-bot starts emitting edges.
   Gate: `./check.sh`.
4. Push order: wdbx, then abi, then abbey-bot, each on an explicit yes.

## 8. Falsification

This amendment is wrong if, after implementation, any of the following holds:

- any pre-existing memory-candidate digest or the pinned `3c19a479…` golden
  changes;
- an edge names a digest from another guild, or a forgotten digest, and is
  accepted;
- a quarantine hides a record from the ledger or blocks a `forgets`;
- one contradiction has two accepted encodings;
- a `resolves` edge from an unauthorized actor is accepted;
- the live path and the replay path disagree on any edge rule;
- free text reaches the ledger through an edge.

## 9. Approval record

- **2026-09-16 05:3x EDT, approved by Donald** in an AskUserQuestion from the session that wrote this document: question A = the asymmetric rule (§3.4), question B = two memory candidates only (§3.6), and "Approve and build wdbx". Implementation may start with §7 step 1. Each push still needs its own yes. The register entries 86 and 87 are only quoted here; they go into the constitution with the abi slice (§7 step 2).
- **2026-09-16 05:5x EDT, §7 steps 1 and 2 implemented.** wdbx `9788ae7` (store, pushed on
  Donald's yes; 620 tests, the pinned memory-candidate golden unchanged, edge golden
  `dfc839a6…`); abi gateway/CLI edge fields, register entries 86 and 87, and this document,
  gated with `./tools/check.sh` green (854 tests) against the sibling at `9788ae7`, followed by
  the `WDBX_REVISION` pin commit. Step 3 (abbey-bot emitting edges) is not started.
