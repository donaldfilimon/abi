# Memory-candidate episodes: routing adapter memory writes through the WDBX gate

Status: **proposed, awaiting Donald's explicit approval.** Written 2026-09-06.
This is a written revision under constitution section 15. Nothing in it is
authorized for implementation until that approval is recorded with a date; the
implementation plan at the end is what would follow, not what exists.

Amends `2026-08-22-spec-canonical-wdbx-episodes.md` (Program 4) section 4, and
proposes two decision-register entries for constitution section 14. Where the
ratified constitution disagrees with this document, it wins.

## 1. The problem, in one paragraph

The v3 episode gate that exists today (`abi_wdbx::v3::episode`, exposed by
`abi-wdbx-gateway` as `ProposeEpisodeWrite`/`VerifyEpisode`, driven by
`abi wdbx episode propose|verify`, and called by abbey-bot's default-off
`episode_gate.rs`) speaks an **operation lifecycle**: `proposal` → `approval`
→ `execution` → `compensation` → `terminal`. That vocabulary fits a guild
operation an administrator asks for. It does not fit the writes that make the
bots adaptive: abbey-bot's memory bank (`mem:{scoped_guild}:{vector_id}` KV
plus a 32-dimension vector, written by `put_kv`/`insert_vector`), its DQN
replay `Experience`s, and `abbey`'s memory backend. Those remain unconditional
local writes, so section 4's `ProposeWrite(EpisodeCandidate) -> WriteDecision`
is unreachable for exactly the writes it was written for, and decisions 28
through 31 (explicit classed retention; opt-in, default-off learning) are
enforced by each adapter separately or not at all. Donald chose, on
2026-09-06, to amend the contract rather than bend the lifecycle vocabulary
(shape B), which is what this document does.

## 2. What the ledger must never hold

Section 5 of the constitution and its invariants fix the shape before any
field is named:

- The ledger stays **content-free**. No vector, no fact text, no prompt, no
  transcript, no summary crosses into an episode. Decisions 21 and 25 read
  together: the payload is a disposable projection held by the adapter; the
  ledger holds a commitment over it and the decision about it.
- **Guild isolation is the correctness boundary** (decision 23). A candidate
  is bound to one `guild_ref` and, when member-scoped, carries only the
  member's keyed principal (the same `admin-{wyhash}` shape abbey-bot already
  uses), never a platform user id.
- **Retention is explicit and classed** (decision 28). Every candidate names
  one class; the gate refuses a class the guild policy does not admit.
- **Deletion is attributable without becoming an undeletable record**
  (spec section 5). Forgetting is a new content-free candidate that references
  the superseded commitment; the adapter destroys the payload, the ledger
  keeps the edge.
- **Safety events and failures are always retained** (spec section 4) and
  therefore do **not** travel through this class: the mandatory-incident path
  bypasses the gate by design, and a memory candidate cannot claim that class.

## 3. The amendment to section 4

Section 4's `ProposeWrite(EpisodeCandidate)` is realised as one new
append-only event kind on the existing `EpisodeWrite` envelope. Nothing about
the envelope, commitment, policy binding, budgets, or `quiet` changes; the
event enum gains a variant and the transition table gains one rule.

### 3.1 The event

```
EpisodeEvent::MemoryCandidate {
    recorded_by: ActorRef,          // kind must be Service (the adapter)
    candidate:   MemoryCandidate,
}

MemoryCandidate {
    class:               MemoryClass,       // fact | experience | embedding | summary
    retention:           RetentionClass,    // session | operational | durable
    payload_commitment:  [u8; 32],          // SHA-256 over the adapter's canonical payload bytes
    payload_bytes:       u64,               // size of those bytes, for storage accounting
    dimension:           Option<u16>,       // embedding width, when class carries a vector
    embedding_version:   Option<String>,    // bounded identifier of the embedding model/transcription
    member_scoped:       bool,              // true when the record is guild-plus-user isolated
    supersedes:          Option<[u8; 32]>,  // episode digest of the candidate this replaces
    forgets:             Option<[u8; 32]>,  // episode digest of the candidate this erases
}
```

Identifiers obey the existing bounded rule (`[a-z0-9_.-]`, 64 characters).
`supersedes` and `forgets` are mutually exclusive; a candidate with `forgets`
set carries `payload_bytes = 0` and an all-zero `payload_commitment`, which
is the one place an all-zero commitment is legal, and only there.

### 3.2 Transition rule

A memory candidate is a **single-event operation**. It is accepted only as the
first event of a fresh `operation_id`, with `previous_digest` absent, from a
service actor, and the store records the operation as terminal
(`TerminalStatus::Completed`) in the same append. Any later event on that
`operation_id` is `InvalidTransition`. Correction never overwrites: a changed
memory is a new candidate whose `supersedes` names the old digest.

Everything else the store already enforces applies unchanged and is not
restated: replayed `request_id`/`operation_id` → `Replay`; stale contract
revision, digest, guild, policy version, or consent epoch → `StaleBinding`;
`learning_enabled = false` → `LearningDisabled`; `quiet` → `Quiet`; token and
storage budgets → their existing rejections. `payload_bytes` is charged
against `storage_budget_bytes` in addition to the record's own serialized
size, so a guild's memory footprint is bounded by the same policy that bounds
its ledger.

### 3.3 Source and consent

`source_type` is `discord_guild` or `local_runtime`. A memory candidate with
`source_type = discord_voice` is `InvalidInput`: raw audio and transcripts are
ephemeral (decision 19; abbey-bot's own rule), and there is no content-free
memory of a voice epoch that this class should carry. Voice counters continue
to ride `Execution.voice` on operation episodes, as today.

### 3.4 Receipt

The receipt is the existing `EpisodeReceipt` with `event_kind =
"memory_candidate"` and `terminal_status = Some(completed)`. It stays
content-free by construction. `VerifyEpisode` answers for it exactly as for
any other event.

### 3.5 Retention classes and the gate

| Class | Admitted when | Adapter obligation |
| --- | --- | --- |
| `session` | policy `learning_enabled` | payload lives only for the bounded session; adapter never persists it to disk |
| `operational` | policy `learning_enabled` | adapter persists under a TTL from its own configuration (the store carries no TTL field in this revision); expiry emits a `forgets` candidate |
| `durable` | policy `learning_enabled` | adapter persists; correction and deletion go through `supersedes`/`forgets` |

`ephemeral` is never proposed (there is nothing to record) and
`mandatory_incident` is not a memory class (section 2). A future policy field
`memory_classes: [..]` may narrow the admitted set per guild; until it exists,
`learning_enabled` is the whole switch, which matches decision 31.

## 4. The adapter contract

This is the half that turns a gate into a boundary, and it is stated for every
adapter, abbey-bot first because its caller already exists.

1. **Propose before writing.** The adapter computes the canonical payload
   bytes, their commitment, and the candidate; it proposes; it writes locally
   **only** on `appended`. On any rejection it writes nothing and records a
   content-free counter of the rejection label. Byte-identical behaviour when
   the gate is unconfigured stays the rule (default-off).
2. **Key by receipt.** The local record carries the receipt's episode digest
   (abbey-bot: a sibling KV `mem-receipt:{scoped_guild}:{vector_id}` holding
   the hex digest), so a later audit can join a projection row to the ledger
   without the ledger ever learning the row.
3. **Fail closed, visibly.** When the gate is configured but unreachable
   (`Unavailable`), the adapter does not write and increments a visible
   counter (`inspect_status` surfaces it in abbey-bot). Decision 78: failure
   degrades visibly and cannot weaken safety; a memory that was never
   admitted is the safe direction.
4. **Rate, not step.** DQN `Experience`s are per decision and would exhaust
   any sane token budget. The `experience` class is proposed once per
   persisted replay checkpoint (the existing `abbey-state.json` write), with
   `payload_commitment` over the checkpoint's canonical bytes and
   `payload_bytes` its size. A checkpoint the gate refuses is not persisted.
5. **The local store stays canonical for the bot** until migration parity
   (constitution section 5, last paragraph). This amendment changes who
   *decides* a write, not who *holds* it.
6. **Transcribe, never depend.** abbey-bot keeps its stable toolchain and its
   subprocess path through the `abi` binary. The JSON form of the new event is
   `{"kind":"memory_candidate","recorded_by":{…},"candidate":{…}}` with the
   field order above, and a fixture generated from `abi-wdbx`'s types
   (`tests/fixtures/episode_write_memory_candidate.json`) pins the
   transcription, exactly as the proposal fixture does today.

## 5. Proposed decision-register entries (text only; not edited into the constitution)

> 84. Adapter memory writes are proposals. An adapter persists a memory
>     record locally only after the WDBX gate appends a content-free
>     memory-candidate episode for it; the ledger holds the commitment and the
>     decision, never the payload, and the local store remains the projection.
>
> 85. Memory candidates are single-event operations. Correction and
>     forgetting are new candidates that reference the superseded commitment;
>     nothing in the ledger is rewritten, and a forgotten payload is destroyed
>     by the adapter while its tombstone edge remains.

## 6. What this does not do

- It does not migrate any existing memory. Records written before the gate was
  wired have no receipt and are reported as such; a backfill is a separate,
  later decision.
- It does not make the ledger a memory store. Retrieval stays with the
  adapter's projection; the ledger can prove a record was admitted, when, and
  under which policy, and nothing more.
- It does not touch the Swift `AbbeyBot` store, which spec section 6 already
  marks unreconciled.
- It does not change `abi-wdbx-gateway`'s proto: the event travels inside the
  existing `EpisodeWrite` JSON.

## 7. Implementation plan, contingent on approval

Only after Donald's approval is recorded here with a date:

1. **wdbx** (`crates/abi-wdbx/src/v3/episode/`): `MemoryClass`,
   `RetentionClass`, `MemoryCandidate`, the `MemoryCandidate` event variant,
   the transition rule in `validate_transition`, storage accounting for
   `payload_bytes`, the all-zero-commitment rule for `forgets`, the voice
   rejection; tests for each rejection label, for single-event terminality,
   for supersede/forget chains, and a canonical-CBOR golden so the digest is
   pinned. Gate: the three wdbx commands.
2. **abi**: `abi-wdbx-gateway` needs no code change beyond the receipt label
   passthrough; `abi-cli`'s help gains the event kind; the CI `WDBX_REVISION`
   pin moves. Gate: `./tools/check.sh`.
3. **abbey-bot**: `episode_gate.rs` gains `record_memory_candidate`; `memory.rs`
   and the replay checkpoint call it per section 4; `inspect_status` shows the
   counters; the fixture is regenerated from the canonical crate. Gate:
   `./check.sh`, on all three hosted platforms.
4. Push order wdbx → abi → abbey-bot, each on an explicit yes, as before.

## 8. Falsification

The amendment is wrong if any of these holds after implementation: a memory
payload byte can be recovered from the ledger alone; a candidate is appended
for a guild whose policy has `learning_enabled = false`; a second event on a
memory operation is accepted; the adapter writes locally on a non-`appended`
decision; two adapters produce different digests for the same canonical
candidate; or the DQN path proposes per step rather than per checkpoint.

## 9. Approval record

_Empty until Donald records approval or requested changes here, with a date._
