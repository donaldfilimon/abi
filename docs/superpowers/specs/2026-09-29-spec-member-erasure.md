# Provable member erasure: encrypted member memory in WDBX, retention holds, and tombstones

Status: **Draft, revision 3, not approved, not implemented.** Written 2026-09-29 from
Donald's decisions of 2026-09-29 (§1.2, §1.3, §1.4). Every Donald decision and its source is
listed in §12. This is a written revision under constitution section 15. Nothing in it is
implemented, and nothing is authorized for implementation until Donald's approval is recorded in
§12 with a date. The build order in §10 is future work, written out in
`../plans/2026-09-29-plan-member-erasure.md`, not work that exists. Every claim about existing
code cites a `file:line` read on 2026-09-29 at wdbx `e454103`, abi `eac7e7fb0`, abbey-bot
`48af4bc`. Rules of this draft that Donald did not separately decide are §11.2. Items left out of
this revision are the single deferral list in §11.3.

Extends `2026-09-06-spec-memory-candidate-episodes.md` (memory candidates) and
`2026-09-16-spec-memory-edge-episodes.md` (memory edges, register entries 86 and 87). It closes
the holds, cryptographic erasure, and content-free tombstone parts of gap-analysis §6.9
(`2026-08-22-wdbx-conformance-gap-analysis.md:203-210`). Redacted derivative blocks and auditable
garbage collection of high-rate traces stay deferred (§11.3). It also performs the fact-memory
migration the
constitution defers: "The Rust Discord bot's JSON facts remain canonical until WDBX migration
parity, replay, recovery, deletion, and rollback pass" (`2026-08-22-abbey-system-constitution.md:318-320`).
It proposes register entries 88 to 93 (§9) and amendments to entries 84 and 85. Where the
ratified constitution disagrees with this document, the constitution wins.

## 1. The problem, and what Donald decided

### 1.1 The problem in one paragraph

A member has no way to say "forget everything about me" and get a result anyone can check.
`/forget` removes one fact by wording (`abbey-bot/src/commands_brain/memory_commands.rs:162-200`).
Nothing removes a member's transcript turns, reputation, projection rows, receipts, or pending
replacements together, and nothing proves afterwards what was removed. The constitution requires
that "Deletion removes payload keys and every derived projection. A content-free tombstone
prevents accidental resurrection. A mandatory hold retains only what the governing policy and
applicable obligation require" (`constitution:308-311`), and lists "Cryptographic erasure leaves
only the permitted content-free tombstone" as a privacy test (`:674`). None of that exists today:
fact text lives in plaintext in abbey-bot's `abbey-state.json`, its projection, and their
backups; WDBX holds unsalted hashes of it; there are no member keys, no holds, no erasure event.

### 1.2 Donald's decisions (binding, 2026-09-29)

1. A member may erase their own memories. A guild admin (owner, administrator, manager) may
   erase any member's memories in that guild.
2. Payloads are encrypted per member. The abi WDBX gateway holds one data key per member,
   wrapped by a master key kept in the macOS Keychain. Erasure deletes the wrapped key and
   appends a signed tombstone, so digests, signatures and replay stay verifiable. abbey-bot
   never sees raw keys.
3. Scope is everything about the member: WDBX memory candidates and edges (crypto-erased), plus
   abbey-bot's local copies (facts and transcript turns in `abbey-state.json`, the
   `wdbx.seg.0.jsonl` projection, and style-addenda observations). One command, one receipt
   listing what was removed and what was retained and why.
4. Existing plaintext member records get a one-time migration to encrypted records that can be
   verified by replay, with a signed migration receipt. Design exactly how plaintext bytes leave
   disk without breaking chain or digest verification.
5. Holds: the host operator (via the abi CLI) and a guild owner (for their own guild only) may
   place and release holds. Holds are signed records with a reason. An active hold blocks
   erasure until it is released, and the erasure request reports the hold.

### 1.3 Donald's answers to revision 1's open questions (2026-09-29)

- **Q1, key model: "Move payloads into WDBX, encrypted."** Keyed commitments are rejected as the
  end state. WDBX becomes the canonical store for member memory payloads (fact text), encrypted
  per member with the gateway-held data key wrapped by the Keychain master. Erasure destroys the
  key.
- **Q3: yes.** The host operator can erase any member in any scope directly, with a signed
  receipt.
- **Q9: rotate the master key after erasure,** automatically, at most hourly, re-wrapping the
  remaining member keys. The Keychain item is this-device-only; the exact attribute is verified
  on macOS 27.2 in the build step.
- **Q11: clear all channel summaries in the scope** when a member there is erased.

The other revision-1 questions are either decided rules of this draft (§11.2) or named deferrals
(§11.3). They are not open product questions.

### 1.4 Donald's answers to revision 2's new questions (2026-09-29)

- **Per-fact keys:** each fact has its own key, wrapped by the member key; forgetting one fact
  destroys that fact's key (§4.1).
- **Keychain access:** the master is stored with after-first-unlock, this-device-only access; the
  exact attribute is verified on macOS 27.2 in the build step (§5.1, §6.2).
- **Ledger capacity:** the new byte fields are base64 in ledger lines, and the ledger cap rises to
  256 MiB (§4.2).
- **Switchover is automatic:** the bot switches a scope after 7 days of shadow reads with zero
  mismatches; the switch is a signed event; rollback is operator-only through the abi CLI; an
  environment kill switch disables auto-switchover (§7.2).

## 2. Premise check (measured 2026-09-29)

**2.1 What WDBX holds today.** `MemoryCandidate` carries `payload_commitment` ("SHA-256 over the
adapter's canonical payload bytes") and `payload_bytes` (`wdbx/crates/abi-wdbx/src/v3/episode/types.rs:342-345`)
and a bare `member_scoped: bool` (`:351`). No payload, no member reference. abbey-bot's gate
states it: "The ledger holds the SHA-256 of the payload and its size, never the payload"
(`abbey-bot/src/episode_gate.rs:16-17`), and a candidate's `recorded_by` is always the service
(`episode_gate.rs:444-447`). Nothing reads a payload back from the gateway: the only readable
content RPC is the v2 `GetKv` (`abi/crates/abi-wdbx-gateway/proto/gateway.proto:9`), and
abbey-bot's `get_kv` calls hit its local projection (`abbey-bot/src/wdbx.rs:330`, `:501`). The
2026-09-06 spec said a member-scoped candidate "carries only the member's keyed principal"
(`2026-09-06-spec-memory-candidate-episodes.md:39-42`); the code carries only the bool. So moving
payloads into WDBX is new capability, not a re-encryption of something already there, and it
changes constitution entry 84 ("the ledger holds the commitment and the decision, never the
payload", `constitution:927-931`), amended in §9.

**2.2 Member-derived bytes already on the gateway's disk (the legacy residue).**

- (a) `payload_commitment = Sha256::digest(&request.payload)`, unsalted (`episode_gate.rs:415`),
  over facts capped at 300 characters (`memory_commands.rs:38-40`). Anyone with the ledger
  confirms a guessed fact with one hash.
- (b) `payload_bytes`, the fact's exact length (`episode_gate.rs:416-419`).
- (c) the episode digest: a flat SHA-256 over the canonical envelope
  (`wdbx/crates/abi-wdbx/src/v3/commitment.rs:20-32`, `:170-173`), not a Merkle structure. Every
  other field is public or low-entropy: `request_id` and `operation_id` are `wyhash` of
  `guild_ref`, unix seconds, and a per-process counter under seeds compiled into abbey-bot's
  public source (`episode_gate.rs:65-69`, `:307-314`). So the digest also confirms a guessed fact
  once `(now, nonce)` is searched.

Memory edges carry nothing member-derived: `kind`, `target`, `counterpart`, `reason`
(`store/canonical.rs:107-119`). A `resolves` edge names the reviewer's `admin-{wyhash}` principal
(`episode_gate.rs:299-305`), not the subject's.

**2.3 Where member plaintext lives in abbey-bot** (`ABBEY_DATA_DIR`):

| Store | Location | Member linkage |
|---|---|---|
| Facts, reputation, counters, pending supersessions | `MemoryBank.users` keyed guild U+001F user (`src/memory.rs:99-114`, `:479-491`); at most `MAX_FACTS = 100` facts per member (`src/memory.rs:24`) | exact |
| Transcript turns | `ChannelContext.recent: RecentMessage { author, text, at }` (`src/memory.rs:131-150`) | **display name only**: `record_message(&scoped_channel, &event.user_display_name, …)` (`src/pipeline.rs:299`) |
| Channel summaries | `ChannelContext.summary`, free text (`src/memory.rs:140-142`) | unattributable |
| Fact receipts | `Stores.memory_receipts`, key `"{scoped_guild}\u{1f}{scoped_user}\u{1f}{fact}"` (`src/persist.rs:483-489`), so the key **is** fact text | exact |
| Reputation | `Stores.reputations`, `Stores.events: ReputationEvent { user_id, … }` (`src/persist.rs:462-468`, `src/brain/social.rs:36-43`) | exact |
| Projection | `wdbx.seg.0.jsonl` rows `mem:{scoped_guild}:{id}` = `{user, text, at}` plus a vector from `text_embedding(text)` (`src/wdbx.rs:485-496`) | exact |
| Style-addenda observations | specified, not yet in code: `AddendaLedger::observe(user, signal, now)`, persisted in `abbey-state.json` (`abbey-bot/docs/superpowers/specs/2026-09-29-fm-primary-addenda-design.md:81-90`) | hashed user id |
| Queued model-tool writes | `memory_queue: Vec<QueuedFact>`, RAM (`src/memory_gate.rs:19-29`) | exact |

Only the first row's **facts** move into WDBX (§4). Everything else stays local and is deleted on
erasure (§6.7).

## 3. Goals and non-goals

### 3.1 Goals

- G1. WDBX becomes the canonical store for member facts in gated scopes, encrypted per member
  inside the signed record. abbey-bot holds fact text only in RAM.
- G2. One command per actor (member self, guild governance, host operator) erases everything
  Abbey holds about one member in one scope, and returns one receipt: removed, retained, why.
- G3. After erasure and the following master-key rotation, no party holding the gateway's live
  disk, its backups, abbey-bot's disk and backups (for facts written after cutover), and the
  Keychain can recover a member's fact from any encrypted record (conditions in §5.1).
- G4. Every digest, signature, and replay check that passed before an erasure passes after it.
  Erasure appends; it never rewrites an encrypted record.
- G5. A one-time migration moves existing local facts into encrypted records, then removes the
  legacy plaintext-equivalent bytes (2.2a, 2.2b) from the live ledger under a signed compaction
  receipt, with the loss of verifiability stated exactly (§7.4).
- G6. Signed, reasoned retention holds that block erasure and are reported by it.
- G7. Idempotent, crash-safe ordering across the gateway and abbey-bot (§8).

### 3.2 Non-goals

- Erasing Discord's own message history, or copies third parties made (other members,
  screenshots, other bots, model providers that already received a prompt).
- Erasing APFS local snapshots or Time Machine backups. §5 states what they keep.
- Moving transcripts, reputation, summaries, or style signals into WDBX. They stay local.
- Opting the member out of future memory (Q7).
- Cross-scope "erase me everywhere" for members (Q8). The host operator's erasure also names one
  scope per call.
- Retrieval weighting, TTL-expiry automation, garbage collection of high-rate traces. Any future
  TTL-expiry path must check holds (§4.7).
- Ungated scopes. A scope the episode gate does not cover (`abbey-bot/src/episode_gate/config.rs:142-146`)
  keeps local plaintext facts; erasure there is local-only and the receipt says so.

## 4. The amendment: vocabulary

All new vocabulary enters as new `EpisodeEvent` variants (entry 87). `MemoryCandidate`,
`MemoryEdge`, their encodings, and the pinned goldens `3c19a479…c768bc`
(`wdbx/crates/abi-wdbx/tests/v3_memory_candidate.rs:18`) and `dfc839a6…47d47f`
(`tests/v3_memory_edge.rs:20`) do not change. Each variant adds one arm to `canonical_event`
(`store/canonical.rs:59-105`) and one label to `EpisodeEvent::label` (`types.rs:534-546`). All are
single-event operations like `memory_candidate`: accepted only as the first event of a fresh
`operation_id` with `previous_digest` absent, recorded terminal (`completed`) in the same append,
and rejected for `source_type = discord_voice` (mirroring `validate.rs:84-88`). `quiet`, binding,
and budget rules apply unchanged; the `learning_enabled` exception is in §4.4.

### 4.1 Keys

Each `(guild_ref, member)` pair has one 32-byte **member key** `MK`, held only by the gateway and
wrapped by the Keychain master `KEK` (§6.2). **Decided rule of this draft (§11.2): granularity is per
`(guild_ref, member)`**, not per member globally: principals are guild-scoped and unlinkable
across guilds (`episode_gate.rs:296-305`), guild isolation is the correctness boundary
(`constitution:290-291`), and an admin's authority covers one guild. A DM scope is its own
`guild_ref` and gets its own key.

**Decided (§1.4): per-record keys under the member key.** Each encrypted record
has its own random 32-byte record key `RK`, wrapped by `MK` and stored in the gateway's key
registry, never in the ledger. Reason: with WDBX canonical, a per-fact `/forget` appends a
`forgets` tombstone but the fact's ciphertext stays in the signed ledger, decryptable for as long
as `MK` lives, whereas today `/forget` destroys the only copy (`memory.rs:559-573`). Per-record
keys keep that property: `/forget` deletes `RK`. There is still exactly one member key, and
member erasure destroys `MK` and every `RK` under it in one registry rewrite. Forgetting one fact
deletes its `RK`, so its ciphertext is unreadable from then on, while every other fact of the
member stays readable.

### 4.2 `encrypted_memory_record`

```
EpisodeEvent::EncryptedMemoryRecord {
    recorded_by: ActorRef,             // service only (§4.6)
    record: EncryptedMemoryRecord,
}

EncryptedMemoryRecord {
    class:      MemoryClass,           // fact | experience | summary; embedding refused
    retention:  RetentionClass,
    member_key: [u8; 16],              // key id; random, never derived from identity
    record_key: [u8; 16],              // key id of RK (§4.1); never all-zero
    nonce:      [u8; 12],              // AES-256-GCM nonce, random per record
    ciphertext: Vec<u8>,               // AES-256-GCM(RK, nonce, padded plaintext, aad), incl. 16-byte tag
    supersedes: Option<[u8; 32]>,      // live encrypted record or legacy candidate in this guild
    migrates:   Option<[u8; 32]>,      // live legacy candidate; migration only (§7.1)
}
```

Plaintext is the fact's UTF-8 bytes, prefixed with its length (u16 big-endian) and zero-padded to
the next multiple of 64 bytes, so ciphertext length reveals only the 64-byte bucket (revision 1's
Q6, now built in). A 300-character fact is at most 1,200 UTF-8 bytes, so the largest ciphertext
is 1,216 + 16 bytes.

AAD binds the ciphertext to its record and scope:

```
aad = "abbey-member-record-v1" || guild_ref || 0x1f || member_key || record_key
      || class_label || 0x1f || retention_label
```

Canonical map (label `encrypted_memory_record`): `recorded_by` (the actor map,
`store/canonical.rs:166-174`) and `record` with `class` (text), `retention` (text), `member_key`
(bytes, 16), `record_key` (bytes, 16), `nonce` (bytes, 12), `ciphertext` (bytes), `supersedes`
(bytes or `Null`), `migrates` (bytes or `Null`). `member_scoped` is implied `true` and not encoded.

**What the digest covers, and why erasure leaves it verifiable.** The episode digest is SHA-256
over the canonical envelope (`commitment.rs:170-173`), which here contains the nonce and the
ciphertext bytes themselves. Nothing in the envelope depends on the plaintext except through the
cipher. After `RK`/`MK` are destroyed the record is byte-identical, so replay recomputes its
digest exactly, and its detached signature still verifies over that digest (`signing.rs:12-14`).
Erasure changes whether the ciphertext can be read, never its bytes.

**No keyed commitment is kept.** Revision 1's keyed tag is dropped because dedup does not need
it. abbey-bot dedups against its RAM cache before proposing (`remember_blocked`,
`memory_gate.rs:50-55`), and the gateway re-checks under its mutex by decrypting the member's live
records in this guild and comparing plaintext: at most `MAX_FACTS = 100` AES-GCM decryptions of
about 1.2 KB each, sub-millisecond. A deterministic tag would also be an oracle: anything able to
call the gateway could test "is this exact fact stored?" without reading. AES-GCM's own tag
authenticates the ciphertext, and the envelope digest and signature cover it.

Shape rules (write: `InvalidInput`; replay: `Corrupt`; as `validate_event_shape`,
`validate.rs:77-93`): nonzero `member_key`; nonzero `record_key`; `ciphertext.len() >= 16 + 64`, `(ciphertext.len() - 16) % 64 == 0`, and
`<= 4096`; `class != embedding`; `supersedes` and `migrates` mutually exclusive. Storage
accounting charges the ciphertext length as the record's payload bytes, as `payload_bytes` does
for candidates (`validate.rs:134-140`).

Per-fact forget keeps using the existing `memory_candidate` `forgets` form (zero commitment, zero
bytes, `validate.rs:109-132`), which may now also name an encrypted record. The gateway
deletes that record's `RK` after the tombstone appends (ordering as §8.1), which is what makes
per-fact forgetting a crypto-erasure.

### 4.3 `member_erasure` (the tombstone)

```
EpisodeEvent::MemberErasure {
    recorded_by: ActorRef,             // §4.6
    erasure: MemberErasure,
}

MemberErasure {
    member_key:        [u8; 16],
    basis:             ErasureBasis,   // self_request | governance_request | operator_request
    erased_records:    u64,            // derived by the store, checked on replay
    erased_set_digest: [u8; 32],       // SHA-256 over the sorted 32-byte digests erased
}
```

Canonical map (label `member_erasure`): `recorded_by`, and `erasure` with `member_key`, `basis`,
`erased_records`, `erased_set_digest`.

The erased set is derived, never supplied: every `encrypted_memory_record` in this guild with this
`member_key`, plus every legacy candidate one of them `migrates`. The gateway fills the two
derived fields from a store read; the store recomputes both on write and replay and rejects a
mismatch. A verifier can check from the ledger alone which N records a tombstone covers. `basis`
must agree with the actor kind: `self_request` with `human_subject`, `governance_request` with
the three governance kinds, `operator_request` with `host_operator`.

After the append the store treats every digest in the erased set as not live (§4.8) and the
`member_key` as dead: a later record naming it is `InvalidTransition`. A member who talks to Abbey
again after erasure gets a fresh key.

**Grouping linkage, stated and accepted.** Every encrypted record names its `member_key`, so the
ledger groups one member's records under an opaque, guild-scoped id. That is the direct cost of
decision 2 (one key per member) and of store-derived erasure sets; the tombstone adds no further
linkage.

### 4.4 `retention_hold` and `hold_release`

```
EpisodeEvent::RetentionHold { recorded_by: ActorRef, hold: RetentionHold }
RetentionHold {
    scope:      HoldScope,             // member | guild
    member_key: Option<[u8; 16]>,      // present exactly for scope = member
    reason:     HoldReason,
    reference:  Option<String>,        // bounded_identifier, <= 64: an opaque case id
}

EpisodeEvent::HoldRelease { recorded_by: ActorRef, release: HoldRelease }
HoldRelease {
    hold:   [u8; 32],                  // an open retention_hold in this guild
    reason: ReleaseReason,
}
```

Canonical maps (labels `retention_hold`, `hold_release`) with those fields; absent options encode
as `Null`. `HoldReason` is closed and content-free: `legal_obligation | law_enforcement_request |
security_investigation | abuse_report | dispute`. `ReleaseReason`: `obligation_ended |
placed_in_error | replaced`. Free text is never admitted; `reference` must pass the store's
`bounded_identifier` (`store.rs:718-724`), so a ticket id such as `case-2026-114` and nothing
more.

A hold is open from its append until a `hold_release` names it. **Holds and erasures are recorded
even when `learning_enabled` is false:** a guild that turns learning off must still be able to
erase what was learned, and to hold it. This is the one exception to `validate_new_write`'s
`LearningDisabled` rule (`validate.rs:59-61`), for `member_erasure`, `retention_hold`, and
`hold_release` only. Writing records stays subject to it.

### 4.5 `ledger_compaction` (the migration receipt)

```
EpisodeEvent::LedgerCompaction { recorded_by: ActorRef, compaction: LedgerCompaction }
LedgerCompaction {
    profile:             CompactionProfile,  // compacted_v1
    prior_ledger_sha256: [u8; 32],           // SHA-256 of the full pre-compaction file
    prior_ledger_bytes:  u64,
    first_sequence:      u64,
    last_sequence:       u64,
    stubbed_records:     u64,
    stub_set_digest:     [u8; 32],           // SHA-256 over sorted digests of stubbed records
}
```

Canonical map (label `ledger_compaction`) with those keys. Its `guild_ref` is a reserved operator
scope `host` that the policy file must list. §7.4 defines what it attests.

### 4.6 Authority

`ActorKind` has no operator class (`types.rs:12-25`). This revision adds
`ActorKind::HostOperator` (label `host_operator`); a new variant changes no existing digest,
because labels enter the envelope only for variants a record uses (`types.rs:27-38`).

The gateway authenticates with one bearer token (`abi-wdbx-gateway/src/auth.rs:11-50`), so an
`ActorKind` is asserted by the caller. The gateway gains a second, owner-only operator token
(`--operator-token-file`); it refuses `host_operator` on a service-token request and refuses
operator-only RPCs without the operator token (Q2, decided rule §11.2). abbey-bot's config names
only the service token.

The store enforces kinds, mirroring `edge_author_allowed` (`validate.rs:341-355`):

| Event | Allowed `recorded_by.kind` | Store rule | Gateway rule |
|---|---|---|---|
| `encrypted_memory_record` | `service` | key known-or-new and not dead; `supersedes`/`migrates` target live in this guild | only `WriteMemberRecord` produces it; `ProposeEpisodeWrite` refuses the kind |
| `member_erasure` | `human_subject` (self); `guild_owner`, `guild_administrator`, `guild_manager` (governance); `host_operator` (any scope, Donald's Q3 answer) | key known and not dead; no open covering hold (§4.7) | only `EraseMember`; a `human_subject` principal must equal the key's bound principal (§6.2); `host_operator` needs the operator token |
| `retention_hold` | `host_operator`; `guild_owner` | member scope: key known and not dead | `host_operator` needs the operator token |
| `hold_release` | the placing class: `host_operator` releases operator holds; `guild_owner` or `host_operator` releases owner holds | target is an open hold here | as above |
| `ledger_compaction` | `host_operator` | §7.3 | offline CLI only |
| `memory_authority` | `service` (to `wdbx_canonical`, `auto_shadow_clean`); `host_operator` (to `shadow`, `operator_rollback`/`operator_resume`) | §4.9 | `host_operator` needs the operator token |

"A guild owner, for their own guild only" is scoped by the write's `guild_ref`; the ownership
claim is asserted by abbey-bot after a fresh REST check (`abbey-bot/src/commands_brain/memory_review.rs:40-53`).
The ledger cannot verify Discord roles; it records the claim under the gateway's signature.

The erasure commands reuse `reviewer_for` (owner, Discord Administrator, Manage Server → the
three governance kinds, `abbey-bot/src/memory_review.rs:23-33`). The existing `/forget`
cross-user rule is wider (it also admits Manage Messages, `commands_brain.rs:53`) and is not
reused for erasure.

### 4.7 Hold semantics

- A `member` hold covers one `member_key` in one guild; a `guild` hold covers every member of that
  guild.
- An open covering hold makes `member_erasure` `InvalidTransition`, including the host operator's.
  The operator can release its own and owners' holds and then erase; both steps are signed
  records.
- A member hold may name a subject with no key yet: the gateway creates the key so the hold has
  something to name. The key is random and unbound to identity in the ledger.
- An open hold also blocks per-fact `forgets` of a covered member's encrypted records, and
  abbey-bot's `/forget` checks the hold first; it does not block new writes or supersession (Q4,
  decided rule §11.2). A hold differs from quarantine, which never blocks `forgets` (edge spec
  §3.3), because a hold exists to prevent deletion.

### 4.8 Liveness and memory edges

`GuildMemories::is_live` is `admitted && !tombstone && !forgotten` (`store.rs:146-152`). This
revision admits encrypted records into the same `admitted` map (member-scoped) and adds an
`erased` set: a digest in any tombstone's erased set is not live. With the edge rules unchanged
(`validate.rs:357-382`):

- New `quarantines`/`contradicts` may name encrypted records and may not name erased ones (live
  targets required, `validate.rs:368-378`).
- An **open** edge naming an erased record stays open and can still be resolved (`resolves`
  checks only `open_edges`, `validate.rs:379`), exactly as for forgotten candidates today.
  Erasure never auto-closes an edge (entry 86).
- Edges are content-free, so nothing in them is erased. They remain as references to a
  tombstoned digest.
- From the live side of a contradiction, `memory_edge_state` (`store.rs:437-476`) plus the new
  `memory_erased` flag (§6.1) show that the counterpart is gone; abbey-bot renders "the other
  fact was erased" and suggests `/admin resolve … valid`.
- `supersedes` naming an erased record is refused.

Rebuilt state gains, per guild: `member_keys: BTreeMap<[u8;16], Live | Dead(tombstone)>`,
`records_by_key: BTreeMap<[u8;16], BTreeSet<[u8;32]>>`, `migrated: BTreeMap<[u8;32], [u8;32]>`,
`erased: BTreeSet<[u8;32]>`, and `open_holds`. All are rebuilt from the ledger on open, as
`GuildMemories` already is (`store.rs:121-138`).

### 4.9 `memory_authority` (the switchover record)

```
EpisodeEvent::MemoryAuthority { recorded_by: ActorRef, change: MemoryAuthority }
MemoryAuthority {
    authority:       FactAuthority,   // wdbx_canonical | shadow
    basis:           AuthorityBasis,  // auto_shadow_clean | operator_rollback | operator_resume
    clean_window_secs: u64,           // auto_shadow_clean only; >= 604800; 0 otherwise
    clean_compares:  u64,             // auto_shadow_clean only; >= 7; 0 otherwise
    compared_records: u64,
    compared_set_digest: [u8; 32],    // SHA-256 over sorted digests compared; zero for operator bases
}
```

Canonical map (label `memory_authority`) with those six keys. Content-free: counts, a window
length, and a digest over record digests already in the ledger. Rules: `wdbx_canonical` only with
`auto_shadow_clean` from `service`, and only when the guild's current authority is `shadow`;
`shadow` with `operator_rollback` or `operator_resume` only from `host_operator`; a no-op change
(same authority, same basis) is `InvalidTransition`. The rebuilt state keeps the current authority
per guild, and `ReadMemberRecords` returns it. Recorded even when `learning_enabled` is false, like
holds, so a guild can be rolled back after turning learning off.

## 5. Threat model

### 5.1 What "unrecoverable" means here

For an **encrypted record**, erasure is complete when:

1. `MK`, and every `RK` under it, is gone from the gateway's live key registry
   (§6.2).
2. The `KEK` that wrapped it has been rotated away and its Keychain item deleted (§6.3, Donald's
   Q9 answer), so a copy of the old registry in an APFS snapshot or Time Machine backup is
   ciphertext under a key that no longer exists.
3. The `KEK` was never in any backup. Donald decided the item is this-device-only. The candidate
   Donald also decided after-first-unlock access (§1.4), so the gateway, a LaunchAgent, can read
   it with the screen locked. The intended implementation is the data-protection Keychain with the
   after-first-unlock, this-device-only accessibility class. **Whether that exact attribute gives
   device binding, backup and migration exclusion, and locked-screen readability on macOS 27.2 is
   not verified here**; it is verified and recorded in phase 2 (§10) before any receipt mentions
   backups.
4. abbey-bot has purged the record from its RAM cache (§6.5) and, after cutover, never wrote the
   fact to disk (§6.5 rules).

Under 1 to 4, every copy of the record anywhere is AES-256-GCM ciphertext under a destroyed key.

For **pre-cutover local plaintext** (facts, receipts, projection rows written before the §7.2
cutover): removed from the live files by the cutover and by erasure, but present in
`abbey-state.json` and `wdbx.seg.0.jsonl` copies in APFS local snapshots and Time Machine until
they age out. **Removed from live state, not unrecoverable.**

For **abbey-bot's other local data** (transcripts, reputation, summaries, style signals): deleted
from live files by atomic temp-file-plus-rename (`abbey-bot/src/persist.rs:13-14`); earlier copies
remain in snapshots and backups. Removed from live state only.

### 5.2 The legacy unsalted hashes, restated precisely

Before this revision every gated fact produced a candidate committing `SHA-256(fact)` (2.2a).
After migration (§7):

- **Live ledger, after compaction:** no legacy commitment or length remains. Each legacy digest
  remains, because later records name it and it is what was signed. To confirm a guessed fact
  for one stubbed digest, an attacker must search `(now, nonce)` jointly with the guess: the
  seeds are public, and `request_id`/`operation_id` are dropped rather than hashed (§7.3). For a
  one-year window (≈3.2·10^7 s) and a nonce below 10^3 that is ≈3·10^10 SHA-256 evaluations per
  guessed fact, on the order of half a minute on one current GPU (estimate). Severity: medium.
  It confirms an exactly guessed fact; it never recovers one.
- **Pre-compaction ledger copies** in APFS local snapshots and Time Machine still hold the
  unsalted commitment, one SHA-256 per guess, until those copies expire. Severity: medium for
  anyone with backup access.
- **Logs and receipts** quoting a legacy digest (gateway mutation notices, `episodes.rs:138-148`;
  abbey-bot's content-free logs) are the same oracle as a live stub digest.
- **Older abbey-bot backups** already hold the fact text itself, so for an attacker with backups
  the hash adds nothing.
- **Encrypted records** add no plaintext hash anywhere: there is no keyed or unkeyed commitment,
  and length is bucketed.

None of this can be removed without giving up "verifiable by replay" (decision 4), because the
digest is named by later records and is what the signature covers.

### 5.3 Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Legacy digest oracle (5.2) | Medium | Drop, never hash, the ids that would let `(now, nonce)` be solved first; the receipt says "older ledger entries keep a fingerprint". |
| Plaintext in transit to the gateway on write and back on read. Today writes go through a 0600 `TMPDIR` file (`episode_gate.rs:691-724`) that the abi CLI reads (`abi-cli/src/wdbx/episode.rs:122-130`), capturable by an hourly APFS local snapshot. | High if unchanged | Every call carrying fact text or a subject id uses **stdin/stdout pipes only** (§6.4); neither side writes a member-derived byte to a file. |
| The gateway holds every decryptable fact while keys live; a compromised gateway process reads them all. | High, accepted | Direct consequence of the Q1 answer. Keys are unwrapped only in gateway memory and zeroized on drop; registry and `KEK` are owner-only; the gateway stays loopback-bound by default (`config.rs:136-151`). |
| Memory availability now depends on the gateway and the Keychain. With the Keychain unavailable, gated-scope facts cannot be read. | Medium | The RAM cache survives gateway restarts while abbey-bot runs; honest degraded reply when facts are unavailable (`AGENTS.md`: "Missing backends must render an honest degraded reply"); after-first-unlock access keeps the Keychain readable with the screen locked (after the first unlock since boot). |
| Ledger capacity: the store refuses past `MAX_LEDGER_BYTES = 64 MiB` (`store.rs:28`, `:329-341`), and `serde_json` writes `Vec<u8>` as a number array (≈3.5 bytes per byte), so at ≈2 to 5 KB per record line only ≈13k to 30k facts would fit, shared with every other episode. | Medium | Decided (§1.4): the new variants' byte fields serialize as base64 strings in ledger lines (canonical CBOR is unchanged, so digests are unaffected), cutting a largest-bucket line to ≈2 KB, and `MAX_LEDGER_BYTES` rises to 256 MiB, ≈100k facts. Segmentation is a later revision if needed. |
| The gateway cannot verify Discord roles; a stolen service token can claim `guild_owner`. | Medium | Unchanged gate trust boundary; the operator token separates operator powers. |
| DM `guild_ref` is `discord-dm-<snowflake>` (`episode_gate.rs:291-294`) if an operator lists DM scopes in the policy (the store refuses unlisted ones, `validate.rs:55-58`); encrypted records in a DM scope then carry a member id in clear in every header. | Low (latent) | Q10. |
| A self-erasure tombstone carries the requester's keyed pseudonym, linkable by the holder of `K_index`. | Low | Intended: deletion is attributable, content is not (`2026-08-22-spec-canonical-wdbx-episodes.md:257-260`). |

## 6. Interfaces

### 6.1 Gateway gRPC (`abi/crates/abi-wdbx-gateway/proto/gateway.proto`)

Existing RPCs are unchanged (`gateway.proto:5-17`). `ProposeEpisodeWrite` refuses the six new
kinds with `INVALID_ARGUMENT` ("event kind requires its dedicated RPC"). Proposed:

```proto
service WdbxGateway {
  // ... existing ...
  rpc WriteMemberRecord(WriteMemberRecordRequest) returns (WriteMemberRecordResponse);
  rpc ReadMemberRecords(ReadMemberRecordsRequest) returns (ReadMemberRecordsResponse);
  rpc ForgetMemberRecord(ForgetMemberRecordRequest) returns (ProposeEpisodeWriteResponse);
  rpc EraseMember(EraseMemberRequest) returns (EraseMemberResponse);
  rpc ErasureFeed(ErasureFeedRequest) returns (ErasureFeedResponse);
  rpc AckErasure(AckErasureRequest) returns (AckErasureResponse);
  rpc PlaceHold(HoldRequest) returns (ProposeEpisodeWriteResponse);
  rpc ReleaseHold(HoldRequest) returns (ProposeEpisodeWriteResponse);
  rpc ListHolds(ListHoldsRequest) returns (ListHoldsResponse);
  rpc SetFactAuthority(SetFactAuthorityRequest) returns (ProposeEpisodeWriteResponse);
  rpc MemberKeyStatus(MemberKeyStatusRequest) returns (MemberKeyStatusResponse); // operator
  rpc RotateMasterKey(RotateMasterKeyRequest) returns (RotateMasterKeyResponse); // operator
}

// subject_ref is the adapter's scoped user id (`discord:<snowflake>`). The
// gateway HMACs it into the key index, seals it in the registry (§6.2), and
// never logs it or writes it to the ledger.
message WriteMemberRecordRequest {
  // EpisodeWrite whose event is encrypted_memory_record with member_key,
  // record_key, nonce and ciphertext zero or empty; the gateway fills them.
  // Zero or empty values fail shape validation, so a template cannot be appended as-is.
  bytes episode_write_template_json = 1;
  string subject_ref = 2;
  bytes plaintext = 3;                  // the fact, UTF-8, <= 1200 bytes; never persisted in clear
  bool preview_only = 4;
}
message WriteMemberRecordResponse {
  string decision = 1;                  // "appended" | "duplicate" | "preview"
  bytes episode_digest = 2;
  EpisodeReceipt receipt = 3;
  bytes member_key = 4;                 // key id only, never key material
}

message ReadMemberRecordsRequest {
  string guild_ref = 1;
  uint64 after_sequence = 2;            // 0 for a full load
  uint32 limit = 3;                     // <= 2048
}
message MemberRecord {
  bytes episode_digest = 1;
  uint64 sequence = 2;
  bytes member_key = 3;
  string subject_ref = 4;               // decrypted from the registry
  string class = 5;
  bytes plaintext = 6;                  // unpadded fact
  bytes supersedes = 7;
}
message ReadMemberRecordsResponse {
  repeated MemberRecord records = 1;    // live records only
  repeated bytes erased_keys = 2;       // keys erased after after_sequence
  repeated bytes forgotten = 3;         // digests forgotten after after_sequence
  uint64 high_sequence = 4;
  string authority = 5;                 // "shadow" | "wdbx_canonical" | "" (never switched)
  bool auto_resume_allowed = 6;         // false after an operator rollback until operator_resume
}

message ForgetMemberRecordRequest {
  bytes episode_write_json = 1;         // a memory_candidate `forgets` naming an encrypted record
}

message EraseMemberRequest {
  // EpisodeWrite with a member_erasure event whose member_key, erased_records
  // and erased_set_digest are zero; the gateway fills them.
  bytes episode_write_template_json = 1;
  string subject_ref = 2;
  bool preview_only = 3;
}
message HoldRef {
  bytes hold_digest = 1;
  string scope = 2;                     // "member" | "guild"
  string placed_by_kind = 3;            // "host_operator" | "guild_owner"
  string reason = 4;
  string reference = 5;
}
message EraseMemberResponse {
  // "erased", "already_erased" (same request_id seen), "held", "no_key", "preview"
  string decision = 1;
  EpisodeReceipt tombstone = 2;
  uint64 erased_records = 3;
  uint64 migrated_legacy = 4;           // legacy candidates erased via `migrates`
  bool key_destroyed = 5;               // false only while §8.1 G4 is pending
  bool kek_rotation_pending = 6;
  repeated HoldRef holds = 7;           // non-empty exactly when decision = "held"
}

// Erasures abbey-bot did not start (operator erasures, or ones whose reply
// abbey-bot lost). Each notice carries the subject so abbey-bot can delete the
// member's local data; the gateway deletes the notice on ack.
message ErasureFeedRequest {}
message ErasureNotice {
  bytes tombstone_digest = 1;
  string guild_ref = 2;
  string subject_ref = 3;
  string basis = 4;
}
message ErasureFeedResponse { repeated ErasureNotice notices = 1; }
message AckErasureRequest { bytes tombstone_digest = 1; }
message AckErasureResponse {}

message HoldRequest { bytes episode_write_json = 1; string subject_ref = 2; }
// A memory_authority write. For wdbx_canonical (service token) the gateway
// recomputes compared_records and compared_set_digest over the scope's live
// records and refuses a mismatch; shadow requires the operator token.
message SetFactAuthorityRequest { bytes episode_write_json = 1; }
message ListHoldsRequest { string guild_ref = 1; string subject_ref = 2; }
message ListHoldsResponse { repeated HoldRef holds = 1; }
```

`VerifyEpisodeResponse` (`gateway.proto:128-149`) gains additive fields:

```proto
  bool encrypted = 9;             // record is an encrypted_memory_record
  bool readable = 10;             // its keys still exist
  bool memory_erased = 11;
  bytes erasure_tombstone = 12;
  bool compacted = 13;            // compacted_v1 stub (§7.3)
  bytes compaction_receipt = 14;
  string hold_status = 15;        // hold episodes: "open" | "released"
```

Errors follow the existing mapping (`abi-wdbx-gateway/src/episodes.rs:241-258`). New labels:
`episode_member_key_dead`, `episode_hold_active`, `episode_keychain_unavailable` (`UNAVAILABLE`),
`episode_operator_required` (`PERMISSION_DENIED`).

`EraseMember` idempotency: a repeated `request_id` returns `already_erased` with the original
tombstone, found by a new store read `find_receipt_by_request_id` (the store keeps `request_ids`,
`store.rs:113-119`). Without it, a retry after a lost reply would find no key and answer
`no_key`.

The erasure-notice outbox is a gateway file `<store>/episodes/erasure-notices.v1.json` (0600,
rewritten whole). It is the only place the gateway stores a subject id in clear; it holds only
unacknowledged notices, and abbey-bot acknowledges after its local deletion commits. Notices for
erasures abbey-bot itself started are acked in the same flow (§8.2 A5).

### 6.2 Key registry (abi gateway, new module `member_keys.rs`)

One owner-only file `<store>/episodes/member-keys.v1.json`, beside the ledger directory
(`executor.rs:22-23`), rewritten whole by temp file, `fsync`, rename, directory `fsync`. Per
member key:

```json
{"key_id":"<32 hex>","guild_ref":"discord-123","index":"<64 hex>",
 "principal":"member-<16 hex>","subject_sealed":"<base64>","kek_id":"<16 hex>",
 "wrapped":"<base64 nonce||ct||tag>","created_sequence":42,
 "records":{"<record key id>":"<base64 RK wrapped by MK>"}}
```

- `key_id` and record key ids: 16 random bytes from the OS CSPRNG.
- `index = HMAC-SHA256(K_index, "abbey-member-index-v1" || guild_ref || 0x1f || subject_ref)`.
  `K_index` is a second Keychain item, not rotated with the `KEK`.
- `principal = "member-" || hex(HMAC-SHA256(K_index, "abbey-member-principal-v1" || guild_ref ||
  0x1f || subject_ref))[..16]`: the `principal_id` a `human_subject` erasure must carry.
- `subject_sealed = AES-256-GCM(MK, subject_ref)`, so reads can return the subject and it dies
  with `MK`.
- `wrapped = AES-256-GCM(KEK, nonce96, MK, aad = "abbey-member-key-v1" || key_id || guild_ref ||
  kek_id)`; `records[*] = AES-256-GCM(MK, nonce96, RK, aad = "abbey-record-key-v1" ||
  record_key_id)`.

Keychain use follows the existing pattern: `security_framework::passwords` get/set/delete under a
named service (`wdbx/crates/abi-foundation/src/credentials/keychain.rs:34`, `:85-120`), never
shelling out to `/usr/bin/security` (`keychain.rs:11`). Proposed service `abi-wdbx-member-keys`,
accounts `kek:<kek_id>` and `index`. The decided Keychain constraints are D8 and D11 in §12:
this-device-only, excluded from backup and migration, and readable by the LaunchAgent while the
screen is locked. Phase 2 measures which macOS 27.2 API meets those three constraints and records
the attribute. The gateway
loads the current `KEK` once and keeps it in zeroizing memory. Off macOS, or when the Keychain
cannot be read, writes and reads answer `episode_keychain_unavailable`. **Erasure needs no
`KEK`**: it deletes entries, so it works when the Keychain is unavailable.

Reconcile on open: after replay, delete every registry entry whose key is `Dead` in the ledger,
every record key whose record is forgotten or erased, then every `kek:*` item that is neither
current nor referenced. The ledger is authoritative over the key file.

The episode signing key stays a raw owner-only file (`config.rs:127-131`); out of scope.

### 6.3 Master key rotation (Donald's Q9 answer)

After any successful erasure or per-fact forget the gateway schedules a rotation,
batched to at most once per hour: generate a new `KEK`, store it as a this-device-only Keychain
item, re-wrap every live `MK`, rewrite the registry atomically, `fsync`, and only then delete the
old `KEK` item. `RotateMasterKey` (operator token) forces one. `EraseMember` reports
`kek_rotation_pending = true` until the rotation covering it completes, and abbey-bot's receipt
says so while pending. Cost: one unwrap and wrap per live member key; for hundreds of keys,
milliseconds (estimate). Record keys are wrapped by `MK`, not the `KEK`, so they need no re-wrap.

### 6.4 abi CLI

Following `abi wdbx episode` (`abi-cli/src/wdbx/episode.rs:22`, `:105-120`). Every command that
carries fact text or a subject id reads one bounded JSON object from **stdin** and writes its
result to **stdout**; nothing is written to a file.

```
abi wdbx member write   --stdin [--preview]
abi wdbx member read    --stdin                      (guild_ref, after_sequence, limit)
abi wdbx member forget  --stdin
abi wdbx member erase   --stdin [--preview]          (service token: abbey-bot)
abi wdbx member erase   <guild_ref> --subject <subject_ref> --operator-token-file <path> [--preview]
                                                     (host operator, any scope)
abi wdbx member feed    --stdin
abi wdbx member ack     --stdin
abi wdbx hold place   <guild_ref> (--member <subject_ref> | --guild) --reason <reason> [--reference <id>] --operator-token-file <path>
abi wdbx hold release <guild_ref> <hold-digest> --reason <reason> --operator-token-file <path>
abi wdbx hold list    <guild_ref> [--member <subject_ref>]
abi wdbx member rollback <guild_ref> --operator-token-file <path>
                                                     (host operator; §7.2; no resume flag, §11.2)
abi wdbx keys status | rotate --operator-token-file <path>
abi wdbx episode compact --store <path> --signing-key <path> [--dry-run]   (offline, §7.3)
```

The operator's `member erase` prints the signed tombstone receipt: digest, sequence, signature
status, counts, `kek_rotation_pending`. Exit codes as today: 0 success, preview, or found;
1 refusal (including `held`); 2 usage.

### 6.5 abbey-bot: source of truth, write and read paths, cache

**Source of truth.** In a gated scope that has been cut over (§7.2), WDBX is canonical for facts.
`MemoryBank.users[…].facts` for that scope is no longer persisted: `abbey-state.json` keeps
reputation, counters, pending supersessions, and a per-scope `fact_authority: wdbx` marker, but no
fact text, and `memory_receipts` entries for the scope are dropped (the digest lives in the RAM
cache). The projection's `mem:*` rows for the scope are no longer written to `wdbx.seg.0.jsonl`;
recall runs on an in-memory index rebuilt from the cache. The local copy is a RAM cache, never a
second canonical store.

**Cache rules.**

1. RAM only. Fact text from a cut-over scope is never written to any file, temp file, log, or
   projection. The inventory test (§6.7) asserts it.
2. Keyed `(scoped_guild, scoped_user)` → `Vec<CachedFact { digest, member_key, text, sequence }>`,
   plus `member_key → (scoped_guild, scoped_user)`.
3. Filled at startup by `ReadMemberRecords(after_sequence = 0)` per covered guild, paged.
   Refreshed incrementally with `after_sequence = high_sequence` on the learn tick and after every
   local write; each refresh applies `erased_keys` and `forgotten`.
4. Written through: `/remember` calls `WriteMemberRecord` and updates the cache only on
   `appended`, the existing "propose first, write on appended" rule (`memory_gate.rs:1-4`).
5. **Purged on erasure** before the erasure reply is sent: every entry for the member, the
   `member_key` mapping, and the in-memory recall rows. Also purged when a refresh reports the
   key in `erased_keys`, or an `ErasureFeed` notice arrives (§8.2).
6. Unavailable is honest: if a covered scope's cache could not be filled (gateway down, Keychain
   unavailable), memory reads for that scope answer "memory is temporarily unavailable", never an
   empty or stale on-disk result. Writes fail closed as today (`memory_gate.rs:249-257`).

**Latency (estimates, to measure in phase 3).** The per-reply path reads only the RAM cache, so
reply latency is unchanged. A write costs one `abi` subprocess plus one gRPC round trip, about the
cost of today's gated `propose` (the subprocess dominates; tens of milliseconds). A cold load of
10,000 facts is about 10,000 AES-GCM decryptions (well under a second) plus five pages of 2,048
records through the CLI pipe; target under 2 s per guild.

**Transport.** A new stdin/stdout transport beside `WriteFile` (`episode_gate.rs:691-724`), used
for every member call; the temp-file path stays only for content-free proposals (learning
toggles, edges).

### 6.6 abbey-bot commands and exact text

Buttons, not Components V2 (Serenity blocks V2, `AGENTS.md`); confirmation follows the pending
component pattern (`memory_commands.rs:217-219`, five minutes). Replies are ephemeral, clamped,
and send no mentions (`commands_brain.rs:96-105`).

- **`/forget-everything [status]`** (member, self; the guild, or the DM scope in a DM).
- **`/admin erase member:<User>`** (guild only; `reviewer_for`).
- **`/admin hold place member:<User>? reason:<choice> reference:<text>?`**,
  **`/admin hold release hold:<digest> reason:<choice>`**, **`/admin hold list`** (place and
  release: guild owner only, REST-checked as in `memory_review.rs:40-53`; list: any reviewer).
- Host operator: `abi wdbx member erase … --operator-token-file` (§6.4). abbey-bot finishes the
  local side from the erasure feed and posts nothing in Discord.

Preview:

```
This will erase everything Abbey holds about {you | <@ID>} in {this server | your DMs with Abbey}:
- remembered facts: {F} (and {P} pending replacements)
- messages in recent channel context: {M} ({L} of them matched by display name only)
- reputation score and {R} history entries
- style signals: {S}
All channel summaries in {this server | your DMs} will also be cleared.
This cannot be undone. Discord's own message history is not affected.
[Erase everything]  [Cancel]
```

Receipt:

```
Erased everything Abbey held about {you | <@ID>} in {this server | your DMs}.
Receipt: erasure:{tombstone_hex}

Removed
- Remembered facts: {F}. Their {N} encrypted ledger records can no longer be read: the key is destroyed{, and the master key rotates within the hour (while pending)}.
- Pending replacements: {P}
- Your messages in channel context: {M} ({L} matched by display name, older records)
- Reputation: score and {R} history entries
- Style signals: {S}{; {A} style adjustment(s) lapsed without your signals}
- Channel summaries cleared: {C}

Kept, and why
- The ledger tombstone and {N} encrypted records: they prove what was erased. Nothing in them can be read.
{- {K} older ledger entries keep a fingerprint that could confirm an exactly guessed fact at high computing cost. (K > 0 only)}
- Voice consent history: evidence of what you agreed to; it holds no words or audio.
- Server work items and approvals you took part in: server records, not memories of you.
- Host backups may hold copies from before {this server moved to encrypted memory | today} until they expire.
- Discord's own message history and copies others made: Abbey cannot delete these.
```

Refusals and hold replies:

| Case | Text |
|---|---|
| Not authorized | `Only the member themselves, the server owner, or a member Discord currently grants Administrator or Manage Server can erase a member's memory. Nothing was erased.` |
| Held | `Not erased: a retention hold is active for {you \| <@ID>} in this server (hold {first 12 hex}, placed by the {host operator \| server owner}, reason: {reason words}{, reference {ref}}). Nothing was deleted. The hold has to be released first.` One line per covering hold. |
| Gate did not answer | `Not erased: the memory ledger did not answer, so it is unknown whether anything was erased there. Nothing was deleted here yet; Abbey finishes the erasure automatically when the ledger answers. /forget-everything status shows the receipt.` |
| Gate refused | `Not erased: the memory ledger refused the request ({reason label}). Nothing was deleted.` |
| Local cleanup failed after the ledger erased | `The ledger erasure is done (erasure:{hex}) and the member key is destroyed, but saving the local cleanup failed ({persist category}). It is retried automatically; run /forget-everything status for the final receipt.` |
| Already erasing | `An erasure for {you \| <@ID>} is already in progress. Nothing new was started.` |
| Ungated scope | ledger lines replaced by `This server is not covered by the memory ledger, so there were no ledger records to erase.` |
| Hold placed | `Hold placed on {<@ID> \| this whole server} (hold {hex}). Erasure is blocked until it is released. Reason: {reason words}.` |
| Hold released | `Hold {first 12 hex} released. Erasure is possible again unless another hold applies.` |
| Not owner (hold) | `Only the server owner can place or release a retention hold in this server. Holds placed by the host operator can be released only by the host operator.` |
| Operator hold | `That hold was placed by the host operator, so only the host operator can release it.` |
| `/forget` under hold | `Not forgotten: a retention hold is active for {you \| <@ID>} in this server. Nothing was deleted.` |
| Memory unavailable | `Memory is temporarily unavailable in this server, so I can't see or change remembered facts right now. Nothing was changed.` |

### 6.7 abbey-bot local erasure plan (pure module `member_erasure.rs`)

Pure: takes `&Stores`, the cache, `&Recall`, `scoped_guild`, `scoped_user`, and the member's
current display names; returns an `ErasurePlan` (counts, exact keys and ids, receipt model).
Applying it is a separate step under the documented lock order, never held across a network
await (`AGENTS.md`).

| Item | Rule |
|---|---|
| Cache entries, `member_key` mapping, in-memory recall rows | purge (§6.5 rule 5) |
| `MemoryBank.users[guild U+001F user]` and its legacy colon key (`memory.rs:493-499`) | remove (any pre-cutover facts, reputation, counters, pending supersessions) |
| `memory_receipts` with prefix `guild U+001F user U+001F` | remove |
| `Stores.reputations[…]`, `Stores.events` for that `user_id` and `guild_id` | remove |
| `ChannelContext.recent` in this scope's channels | remove rows whose new `author_ref` is the member; for rows without it, rows whose `author` equals a current display name of the member, counted separately |
| `ChannelContext.summary` in this scope's channels | **clear all** (Donald's Q11 answer) |
| Projection `mem:{guild}:{id}` rows for the member and their vectors (`wdbx.rs:485-496`, `:577`) | remove both (pre-cutover scopes only; after cutover none exist on disk) |
| Style-addenda observations for the member's hashed id | remove; recompute active addenda and drop any that no longer meet "5 signals from at least 3 distinct users" (addenda spec `:83`) |
| `memory_queue` items for the member | drop, completing each as cancelled (`memory_gate.rs:113-139`) |
| In-memory engine sessions | clear the DM session and this guild's channel sessions (RAM only; `Stores` has no engine field, `persist.rs:454-490`) |
| `ConsentStore` members, `WorkStore`, action approvals, DQN brains, `pending_rewards`, `InteractionLog` | **retain**, listed with reasons (Q12) |

New additive field `RecentMessage.author_ref: Option<String>` (`#[serde(default)]`): a keyed hash
of the scoped user, set by `record_message`, so attribution stops depending on display names from
the day it lands.

Inventory test: populate `Stores`, the cache, and `Recall` with a sentinel member and a bystander
in every user-keyed structure; erase; serialize both files **and** capture every byte written by
the persistence sink and the gate transport. Assert that no occurrence of the sentinel's scoped
id, keyed hashes, display name, or fact text remains outside the retained allow-list, and that no
fact text from a cut-over scope was written to any file at any point. A later user-keyed field
without an erasure rule fails it, which covers the concurrent style-addenda work.

## 7. Migration

The constitution's rule applies: "Migrations shadow-read, replay, compare, cut over one writer,
and retain rollback evidence" (`constitution:293-294`). Migration runs per covered scope in three
stages, then one store-wide compaction.

### 7.1 Shadow (dual storage, local stays canonical)

With `"member_records": "shadow"` for a scope (new optional key in `RawConfig`,
`episode_gate.rs:216-238`), abbey-bot keeps its local fact store canonical and additionally:

- **Backfill.** `abbey-bot --migrate-facts [--dry-run] [--guild <id>]`, a non-gateway mode like
  `--server-plan`, writes one `encrypted_memory_record` per local fact. If the fact has a legacy
  receipt the record sets `migrates = <legacy digest>`; the store checks the target is a live
  legacy candidate in the same guild and not already migrated. Facts without receipts (stored
  before the gate) get plain encrypted records. Resumable: `duplicate` or "already migrated" count
  as success. Bounded by the guild's token budget.
- **Mirror.** Every new `/remember`, supersession, and `/forget` in the scope goes to both stores
  (local after WDBX `appended`, as today).
- **Shadow-read compare.** On each learn tick, `ReadMemberRecords` for the scope and compare with
  the local facts per member (set equality of text, supersession targets). Mismatches are counted
  in `inspect_status` with content-free labels, never logged with text.

Legacy candidates whose local fact is gone ("orphans") cannot be migrated because nobody holds the
plaintext; compaction still removes their hash.

### 7.2 Cutover (one writer) and rollback

**Decided (§1.4): switchover is automatic.** abbey-bot switches a scope by itself once the scope
has had 7 days of shadow reads with zero mismatches. The operator starts migration (by putting the
scope in `shadow`) and alone can roll it back.

**What counts as a mismatch.** A shadow compare runs on each learn tick for each shadowed scope:
it pages `ReadMemberRecords` for the whole scope and compares it member by member with the local
facts. Any one of these is a mismatch:

- a local fact with no live encrypted record whose decrypted text is byte-equal (after the
  bot's `validated_fact` normalization, `memory_gate.rs:49`) for the same member;
- a live encrypted record with no equal local fact for that member;
- a record attributed to a different member than the local fact (subject from the registry vs.
  the local key);
- a supersession link present on one side and not the other, or pointing at different facts;
- a local fact carrying a legacy receipt whose legacy digest no encrypted record `migrates`;
- a record that fails to decrypt (AAD or tag failure).

A compare that could not complete (gateway or Keychain unavailable, a page failed, the tick was
cancelled) is **not** a mismatch and **not** a clean read: it neither advances nor resets the
clock by itself.

**When the clock resets.** abbey-bot keeps, per scope in `abbey-state.json`, `clean_since` (unix
seconds of the first clean compare in the current run), `last_clean_compare`, and
`clean_compares` (content-free). `clean_since` is set to the current time, and `clean_compares`
to zero, when any of these happens:

- any compare finds a mismatch (the next clean compare starts a new window);
- more than 24 hours pass without a completed clean compare, so 7 days means 7 days of observed
  agreement, not 7 days of silence;
- backfill is incomplete: a local fact exists with no encrypted record yet (so the window only
  starts after `--migrate-facts` has finished for the scope);
- the scope's gate binding changes (`policy_version`, `contract_revision`, coverage), or the
  scope's shadow mode is re-entered after an operator rollback.

A write that is mirrored successfully does not reset the clock; a mirrored write that fails on
one side is a mismatch at the next compare unless repaired first.

**The switch.** When `now - clean_since >= 7 days`, `clean_compares >= 7`, the latest compare is
clean, and the kill switch is not set, abbey-bot:

1. runs one final full compare; a mismatch resets the clock and stops here;
2. appends a signed `memory_authority` event (§4.9) through the gateway with
   `authority = wdbx_canonical`, `basis = auto_shadow_clean`, the window length, the compare
   count, and a digest over the sorted record digests it compared; the gateway signs it like any
   episode;
3. on `appended` only, removes fact text and receipts for the scope from `abbey-state.json` and
   its `mem:*` rows from the projection, sets `fact_authority: wdbx`, and persists atomically;
4. logs one content-free line (`tracing::info!`, scope `guild_ref`, event digest, counts) and one
   closed managed observability event, and `inspect_status` shows the scope as canonical with the
   event digest.

No local plaintext rollback snapshot is kept; it would defeat the purpose. Rollback evidence is the
WDBX records themselves (decryptable while keys live) and the signed event.

Crash between steps 2 and 3: the ledger is authoritative. At startup abbey-bot reads each covered
scope's authority from the gateway (`ReadMemberRecords` returns it, §6.1) and, if it is
`wdbx_canonical` while local still holds fact text, finishes step 3. A refused or unavailable
append in step 2 leaves the scope in shadow, and the next tick retries.

**Kill switch.** `ABBEY_MEMORY_AUTO_SWITCHOVER=off` in abbey-bot's environment (read once at
startup, alongside `ABBEY_EPISODE_GATE_CONFIG`, `episode_gate.rs:49-50`) disables step 2 for every
scope. Compares and the clock keep running, so the operator can see which scopes are eligible;
`inspect_status` says "eligible; auto-switchover disabled by ABBEY_MEMORY_AUTO_SWITCHOVER". Any
other value, or unset, leaves auto-switchover on. The switch never happens while the variable is
`off`, including for a scope already past 7 days.

**Rollback (operator-only, abi CLI).** `abi wdbx member rollback <guild_ref>
--operator-token-file <path>` appends a signed `memory_authority` event with
`authority = shadow`, `basis = operator_rollback`. The store accepts `authority = shadow` only from
`host_operator` (§4.6), so abbey-bot cannot roll a scope back. On its next refresh abbey-bot sees
the authority change, rebuilds local facts from `ReadMemberRecords`, persists them, returns the
scope to shadow, and resets its clock. **Decided rule of this draft (§11.2):** after
`operator_rollback`, this revision does not auto-switch that scope again. A resume command is
deferred (§11.3). Erased members cannot be restored by rollback, by design.

### 7.3 Compaction (offline; removes the legacy hashes)

Run after every covered scope is cut over, with the gateway stopped (it holds the writer lock,
`store.rs:227-232`): `abi wdbx episode compact --store <path> --signing-key <path>`.

1. Take the lock; open with the signer and fully replay-verify (`store.rs:207-213`). Any failure
   aborts with nothing written.
2. Compute `prior_ledger_sha256` and `prior_ledger_bytes`.
3. Build the new ledger: each legacy `memory_candidate` without `forgets` becomes a stub line;
   every other record is copied byte for byte; append one signed `ledger_compaction`.
4. Write `episodes.v1.jsonl.compacting` (0600), `fsync`, rename over `episodes.v1.jsonl`, `fsync`
   the directory. Before the rename the old ledger is intact and a stale `.compacting` file is
   deleted on the next attempt; after it, the new ledger is complete.
5. Re-open and replay-verify; print the migration receipt (compaction digest, counts,
   `prior_ledger_sha256`).

The stub is a new line form that `StoredRecord` cannot parse (`deny_unknown_fields`,
`store.rs:89-90`):

```json
{"form":"compacted_v1","sequence":17,"guild_ref":"discord-123","event_kind":"memory_candidate",
 "token_cost":1,"member_scoped":true,"supersedes":null,"episode_digest":"…","signature":{…}}
```

Kept: `sequence`; `guild_ref` (isolation); `event_kind`; `member_scoped` (contradiction rule,
`validate.rs:374`); `supersedes` (liveness, `validate.rs:327-339`); `token_cost` (budget);
`episode_digest` (named by later records, and what was signed); `signature` (verifies over the
digest alone, `signing.rs:12-14`). `previous_digest` is always absent for a candidate
(`validate.rs:258`). Dropped: `payload_commitment`, `payload_bytes`, `class`, `retention`,
`dimension`, `embedding_version`, `request_id`, `operation_id`, `contract_revision`,
`contract_digest`, `consent_epoch`, `source_type`, `policy_version`, `evidence_level`.
`request_id` and `operation_id` are dropped rather than hashed: a hash of either would let an
attacker solve `(now, nonce)` once, after which each fact guess costs one SHA-256 (§5.2).

Replay changes: a stub is checked structurally (it cannot pass `validate.rs:200`); the sequence
counter becomes a record count instead of `request_ids.len()` (`validate.rs:188-192`); every stub
must be covered by the next `ledger_compaction`, whose `stub_set_digest` and `stubbed_records`
replay recomputes; request-id replay protection for stubs becomes **digest uniqueness** (a new
record equal to any existing digest is `Replay`, and an exact legacy replay reproduces its
digest). After a `ledger_compaction`, a plain `memory_candidate` without `forgets` is
`InvalidTransition`: no new unencrypted fact commitments. A pre-revision build cannot open a
compacted ledger, the same caveat signing introduced (`store.rs:204-206`).

### 7.4 Invariants, and what a verifier can still prove

**I1 (digest stability).** Every record's `episode_digest` is identical before and after
compaction; every copied record is byte-identical; no digest named in a later record changes.
Erasure and rotation never touch the ledger's existing bytes.

**I2 (verification classes).** A record is *full* (digest recomputed from its fields on every
open; this includes every encrypted record, readable or erased) or *stub* (digest not
recomputable; attested by its own signature when present and by membership in the signed
`ledger_compaction` whose `stub_set_digest` covers it and whose `prior_ledger_sha256` commits to
the prior bytes).

**I3 (state equivalence).** Replaying after compaction yields the same liveness, supersession,
forgetting, migration, and edge state, and the same accept or reject decision for every future
write, except that storage usage falls by the removed bytes and each stub's former
`payload_bytes`, id replay detection for stubs becomes digest uniqueness, and plain non-`forgets`
candidates are refused.

**I4 (no plaintext-equivalent on live disk).** After cutover and compaction, the gateway's and
abbey-bot's live files hold nothing derived from a fact except AES-GCM ciphertext (under a live or
destroyed key) and each legacy stub's digest.

A verifier can prove afterwards: every encrypted record's digest from its bytes, and its
signature; which records each tombstone erased (the derived set matches `erased_set_digest`); that
each stub digest existed at its sequence, and was signed if it was; that the gateway's key signed
a compaction covering exactly those stubs, from a prior ledger of stated size and hash; and, while
keys live, that a record decrypts under its AAD. A verifier can no longer prove a stub's former
contents unless it kept a copy of the prior ledger. Legacy records appended unsigned
(`store.rs:186-195`) are attested only by the compaction receipt.

"Plaintext bytes leave disk" means the **live** filesystem. Earlier copies stay in APFS local
snapshots and Time Machine until they age out (§5). Deleting local snapshots after cutover and
compaction is the owner's call; the runbook states the option and does not take it.

## 8. Failure modes and ordering

### 8.1 Gateway `EraseMember` (one call under the executor mutex, `executor.rs:25-30`)

| Step | Action | If it fails or crashes after this step |
|---|---|---|
| G1 | Authenticate; resolve `index → key_id`; `no_key` if none | nothing changed |
| G2 | Covering holds → `held` with `HoldRef`s | nothing changed |
| G3 | Append `member_erasure` (`fsync`, `store.rs:343-346`); for an operator erasure, add a notice to the outbox | the registry still holds `MK`; reconcile on open deletes it; the store already refuses the dead key; a retry with the same `request_id` answers `already_erased`; a missing notice is regenerated on open from tombstones newer than the last ack |
| G4 | Rewrite the registry without the member entry and its record keys; zeroize | on rewrite failure answer `erased` with `key_destroyed = false`, keep the key dead in memory, retry on each RPC and on open |
| G5 | Schedule rotation (§6.3); answer | `kek_rotation_pending` until done |

A tombstone without a deleted key is repaired automatically; a deleted key without a tombstone
cannot happen, because G4 follows G3. `ForgetMemberRecord` has the same shape:
tombstone first, then delete `RK`.

### 8.2 abbey-bot `/forget-everything` and `/admin erase`

Ledger first, local second, bookkeeping last, as `/forget` already does
(`memory_commands.rs:185-192`), plus a durable intent.

| Step | Action | If it fails or crashes after |
|---|---|---|
| A0 | Authorize from fresh Discord facts; refuse if an intent for this member is open | nothing changed |
| A1 | Plan (§6.7), gateway preview, show preview, wait for the button | nothing changed |
| A2 | Mark `(scope, member)` erasing in memory (member writes refuse); drop queued writes; persist `pending_erasures: [{request_id, scoped_guild, scoped_user, started_at}]` atomically | on restart the intent resumes at A3 |
| A3 | `abi wdbx member erase --stdin` (gated scope) | `held` or refused: clear intent, reply. Unavailable: keep intent, reply "unknown", retry with the **same** `request_id` on the learn tick and at startup (bounded backoff, at most hourly). `already_erased` counts as success. |
| A4 | Purge the cache; apply the plan in memory; persist canonical, then projection | canonical persist failed: memory is erased, the intent is still on disk, the persistence loop retries; a crash reloads the old document **and** the intent, and A4 re-runs (removals are idempotent). Projection failed: startup reconcile rebuilds `mem:*` from canonical (`wdbx.rs:620-625`); cut-over scopes have none. |
| A5 | Replace the intent with a content-free receipt record (tombstone digest, counts, time); persist; ack any feed notice for this tombstone | crash: the intent resumes; A3 answers `already_erased`; A4 is a no-op |
| A6 | Reply with the receipt | a lost reply is recovered by `/forget-everything status` |

**Operator erasures** reach abbey-bot through `ErasureFeed`, polled on the learn tick and at
startup. For each notice abbey-bot runs A2 (intent) through A5 with the notice's subject, skipping
A3, then acks. The gateway keeps the notice until then.

Partial erasure always resolves toward completion: once a tombstone exists, every path finishes
A4 and A5. Before A3 answers nothing local is deleted, so a local-first deletion can never bypass
a hold. The intent holds the scoped user id until A5 removes it.

### 8.3 Concurrency

- Gateway: the executor mutex serializes writes, reads, holds, and erasures. A hold that lands
  first refuses the erasure; an erasure that lands first makes the key dead, and a later member
  hold refuses with `episode_member_key_dead`.
- abbey-bot: a write in flight when A2 starts may reach the gateway after the tombstone; it fails
  with `episode_member_key_dead` and nothing is cached, because writes cache only on `appended`. A
  `/remember` after A5 creates a fresh key: new memory, not resurrection.
- Cache and erasure: rule 5 purges before the reply, and each incremental refresh applies
  `erased_keys` for erasures started elsewhere, so a cached fact never outlives the next refresh
  after its tombstone.

## 9. Decision-register entries (proposed)

> 88. Member memory payloads in gated scopes are canonical in WDBX, encrypted inside the signed
>     record under per-member keys held only by the gateway and wrapped by a device-bound master
>     key. Adapters hold them only in memory. Erasure destroys the member key and appends a
>     tombstone that names it; encrypted records are never rewritten, and their digests,
>     signatures and replay remain valid.
>
> 89. A retention hold is a signed, reasoned, content-free episode. While one is open, the member
>     it covers cannot be erased or have records forgotten; only the class that placed it may
>     release it.
>
> 90. Erasure is ledger-first and adapter-second. An adapter deletes its local copies only after
>     the ledger has recorded the erasure, and finishes its deletion whenever a tombstone exists.
>
> 91. A one-time compaction may replace legacy memory-candidate records by stubs that keep their
>     digest, signature, and the structural fields replay needs, under a signed compaction
>     receipt that commits to the prior ledger. No other rewrite of the ledger is permitted.
>
> 92. The master key rotates after erasure, at most hourly, so that copies of wrapped keys in
>     backups die with the key that wrapped them.
>
> 93. An adapter switches a scope's fact authority to WDBX by itself only after 7 days of clean
>     shadow reads, records the switch as a signed episode, and honours an environment kill
>     switch. Only the host operator rolls a scope back.

Amendments. Entry 84 (`constitution:927-931`): replace "the ledger holds the commitment and the
decision, never the payload" with "the ledger holds the commitment and the decision and, for member
memory in gated scopes, the payload encrypted under entry 88; never a plaintext payload". Entry 85
("nothing in the ledger is rewritten"): append "except by the one compaction entry 91 permits".
The canonical-facts sentence (`constitution:318-320`) is satisfied scope by scope by §7.2.

## 10. Build order, contingent on approval

Future work for this sub-project is one plan,
`../plans/2026-09-29-plan-member-erasure.md`. That plan is not implemented. Inside it the work is
sequenced as the four phases below; one commit per step, each gated in its repo; push order wdbx,
abi, abbey-bot, each on an explicit yes. A phase starts only after the previous phase's pushes.
This section does not authorize those commits.

**Phase 1: substrate (wdbx).**

1. `crates/abi-wdbx/src/v3/episode/`: `ActorKind::HostOperator`; `EncryptedMemoryRecord`,
   `MemberErasure`, `ErasureBasis`, `RetentionHold`, `HoldScope`, `HoldReason`, `HoldRelease`,
   `ReleaseReason`, `LedgerCompaction`, `CompactionProfile`, `MemoryAuthority`, `FactAuthority`,
   `AuthorityBasis`; variants, labels, canonical arms;
   shape rules; authority (§4.6); transitions and state (§4.8, `apply_record`,
   `store.rs:544-632`); the learning-disabled exception; digest uniqueness; ciphertext
   accounting; reads `find_receipt_by_request_id`, `erasure_preview`, `open_holds`,
   `member_records(guild, after_sequence)` (ciphertext, never plaintext); edge state gains
   `erased`. Python witness: six arms (`tools/abbey_cbor_episode_v1.py:408-431`) and six
   `EPISODE_GOLDENS` entries (`:301-309`); six new goldens. **The pinned `3c19a479…` and
   `dfc839a6…` pass unchanged (entry 87's proof).** Base64 ledger-line encoding for the new byte
   fields and `MAX_LEDGER_BYTES = 256 MiB` (`store.rs:28`) land here. Gate: `cargo fmt --all --check`, `cargo clippy --workspace
   --all-targets`, `cargo test --workspace`.
2. The `compacted_v1` stub form, `EpisodeStore::compact`, stub replay, `ledger_compaction` checks,
   post-compaction refusal of plain candidates, compacted-ledger golden. A separate commit because
   it is the only rewrite path. Same gate.

**Phase 2: gateway and CLI (abi).**

3. `member_keys.rs` (registry, per-fact record keys, `KEK` and `K_index` in the Keychain,
   reconcile, hourly rotation), AES-256-GCM and AAD, operator token, all §6.1 RPCs, notice outbox,
   `VerifyEpisode` fields, `ProposeEpisodeWrite` refusals, `WDBX_REVISION` pin bump, register
   entries 88 to 93 and the amendments. **Verify and record the exact Keychain attribute and
   accessibility class that make the `KEK` this-device-only and excluded from backup and
   migration on macOS 27.2, and that it stays readable to a LaunchAgent with the screen locked
   (§1.4).** Until that is recorded, no receipt mentions backups. Re-grep abi and abbey for
   other `memory_candidate` emitters before the post-compaction rule matters. Gate:
   `./tools/check.sh < /dev/null`.
4. CLI (§6.4) with the stdin/stdout transport, the operator erasure receipt, `member rollback`,
   `EPISODE_HELP`. Same
   gate.

**Phase 3: abbey-bot shadow, cache, cutover.**

5. Wire transcription of the six variants with byte-for-byte fixtures of those it reads or
   writes; stdin/stdout transport; gateway member client; `RecentMessage.author_ref`. Gate:
   `./check.sh`, never piped.
6. RAM cache and in-memory recall index; `"member_records": "shadow"`; `--migrate-facts`;
   mirrored writes; shadow-read compare counters in `inspect_status`. Same gate.
7. Automatic switchover (§7.2): mismatch rules, the persisted clean-window clock, the
   `memory_authority` append, `ABBEY_MEMORY_AUTO_SWITCHOVER`, startup completion from the
   ledger's authority, and rollback handling; stop persisting fact text and `mem:*` rows for
   switched scopes; the honest "memory unavailable" path; the "never written to a
   file" half of the inventory test. Same gate.

**Phase 4: erasure and holds (abbey-bot), then the operator run.**

8. `member_erasure.rs`, `pending_erasures` and receipt records (additive), apply and resume, the
   erasure-feed poller and ack, the hold check in `/forget`, the full inventory test (it covers
   style addenda if they have landed). Same gate.
9. `/forget-everything [status]`, `/admin erase`, `/admin hold …`, catalog entries, and the §6.6
   texts reviewed as rendered text (`AGENTS.md`). Same gate.
10. **Operator run (not a commit):** deploy; per guild, `--migrate-facts --dry-run`, then for
    real; the bot switches each scope itself after 7 clean days (or the operator leaves
    `ABBEY_MEMORY_AUTO_SWITCHOVER=off` and watches `inspect_status`); after every scope has switched, stop the gateway, run
    `abi wdbx episode compact --dry-run`, then for real; restart; verify a stub, an encrypted
    record, an erased record, and the compaction record with `abi wdbx episode verify`. Record the
    receipts in §12.

### 10.1 Tests and acceptance

**wdbx:** one test per rejection (shape, authority row, dead key, open hold, wrong release class,
learning-disabled exception); the erasure set is derived identically live and on replay; a
mutated `erased_set_digest` or `erased_records` on replay is `Corrupt`; a write under a dead key
is refused; an open edge on an erased record is resolvable, a new quarantine is refused;
`migrates` of a non-live, already-migrated, or cross-guild target is refused; ciphertext length
rules; compaction I1, I3 (a scripted write corpus replayed against both ledgers), I4 (no legacy
commitment bytes remain), crash before and after rename, an uncovered stub is `Corrupt`, an exact
legacy replay is `Replay`; the two existing goldens are unchanged; the Python witness reproduces
all eight goldens.

**abi:** a write encrypts and a read round-trips; a wrong AAD (guild, key, class) fails to
decrypt; duplicate detection by decrypt-compare; same subject gets the same key, another guild a
different key; erase preview, erased, already_erased, held, no_key; operator erasure works only
with the operator token and prints a signed receipt; a crash between G3 and G4 then reopen
deletes the key and keeps the notice; with the Keychain unavailable, write and read answer
`UNAVAILABLE` while erase succeeds; rotation re-wraps every key and deletes the old `KEK` only
after rename; forget deletes `RK`; no subject id or plaintext appears in any file except
the outbox's subject, and that only until ack.

**abbey-bot:** cache load, incremental refresh, purge on erasure and on `erased_keys`; memory
unavailable is reported, never empty; shadow compare detects an injected mismatch; cutover aborts
on a mismatch; each mismatch kind resets the clock; an incomplete compare neither advances nor
resets it; a 24-hour gap resets it; the switch happens at 7 days and 7 clean compares and not
before; the kill switch blocks it; a crash after the append completes the switch at startup; the
service cannot append `shadow`; operator rollback rebuilds local facts and does not
auto-switch that scope again in this revision; the inventory test; plan counts; display-name
matching only without `author_ref`; all summaries cleared; an addendum lapses below threshold;
resume from each §8.2 crash point with the fake gate pattern (`memory_gate.rs:401-424`); gate
unavailable deletes nothing; held deletes nothing; `/forget` refuses under a hold; a feed notice
drives local deletion and ack; the receipt fits the 2,000-character clamp.

**Live acceptance (a separate C6 claim):** on a scratch guild and scratch gateway: facts migrate,
shadow agrees, the scope cuts over; a member runs `/forget-everything`; an owner's hold blocks
`/admin erase` until released; the operator erases another member from the CLI and abbey-bot
finishes from the feed; `abi wdbx episode verify` reports the tombstones and `readable = false`.

### 10.2 Falsification

This revision is wrong if, after implementation, any of the following holds:

- either pinned golden digest, or any pre-existing record's digest, changes;
- an encrypted record can be decrypted after its tombstone and the covering rotation, using the
  live disk, backups, and the Keychain;
- fact text from a cut-over scope is found in any file abbey-bot or the gateway wrote;
- a cached fact is served after the refresh following its tombstone;
- a `member_erasure` is accepted under an open covering hold, or from an unauthorized kind;
- abbey-bot deletes local member data before the ledger answers `erased` or `already_erased` in
  a gated scope, or never finishes after a tombstone exists;
- the live path and the replay path disagree on any rule in §4 or §7;
- free text reaches the ledger through a hold, release, erasure, or compaction;
- the inventory test's sentinel survives outside the retained allow-list.

## 11. Decisions and deferrals

### 11.1 Decided by Donald (2026-09-29)

The source table is §12. These are binding, including the four end-state answers: encrypted
member payloads in WDBX (keyed commitments rejected), direct host-operator erasure with a signed
receipt, automatic master-key rotation at most hourly on a this-device-only Keychain item, and
clearing every channel summary in the scope. Per-fact keys, after-first-unlock Keychain access,
base64 ledger fields with a 256 MiB cap, and automatic switchover after 7 clean shadow days are
also binding. None of these is an open question.

### 11.2 Decided rules of this draft

These rules are how this revision makes §11.1 implementable. Donald did not separately decide
them. They are not open, and they do not authorize implementation (§12).

- **Q2.** The host operator is `ActorKind::HostOperator` plus a second owner-only operator token.
- **Q4.** An open hold blocks per-fact `/forget` of a covered member and does not block new writes.
- **Q5.** Transcripts, reputation, and style signals stay local and are deleted on erasure. This
  revision does not encrypt them at rest.
- **Q6.** Plaintext is padded to 64-byte buckets (§4.2).
- **Q8.** Self-erasure names one scope per call.
- **Q12.** Voice consent records and work items are retained, with reasons on the receipt (§6.7).
- **Q13.** abbey-bot keeps the content-free erasure receipt record for 90 days.
- **Q14.** Only the class that placed a hold may release it. A guild owner does not release a
  host-operator hold.
- **Key granularity.** One member key per `(guild_ref, member)`. A DM scope is its own `guild_ref`.
- **After operator rollback.** The scope returns to shadow and this revision does not auto-switch
  it again. There is no resume flag.

### 11.3 Explicit deferrals

This is the only deferral list in this spec.

- **Q7.** A future-learning opt-out. This revision does not add one. Erasure does not by itself
  stop later learning about the member.
- **Q10.** A dedicated scheme for assigning DM `guild_ref` values beyond treating a DM as its own
  `guild_ref`.
- **Q15.** A command that lets a rolled-back scope auto-switch again. This revision does not add
  one (§11.2).
- **Redacted derivative blocks** that link to an original without overwriting it (gap analysis §6.9).
- **Auditable garbage collection** of unreferenced high-rate traces (gap analysis §6.9).
- **Later evidence-half work**, ordered in `2026-09-29-wdbx-completion-design.md`: retrieval-time
  staleness weighting, evidence-weighted retrieval, `task_regime` and `regime_posterior`, COSE
  envelopes, claim levels C3 through C7, and hosted federation. None of those is Current, and this
  spec does not raise them.

## 12. Approval record

Status: **not approved and not implemented.** No implementation is authorized until Donald records
approval of this document here, with a date. §11.2 is already the draft's rule set. §11.3 stays
deferred. Each push still needs its own yes.

Decisions Donald made on 2026-09-29. Source for every row: AskUserQuestion, 2026-09-29, as relayed
to this session by the coordinating session.

| # | Decision | Where it lands |
|---|---|---|
| D1 | A member may erase their own memories; a guild owner, administrator, or manager may erase any member's memories in that guild. | §4.6, §6.6 |
| D2 | Payloads are encrypted per member; the gateway holds one data key per member, wrapped by a Keychain master; erasure deletes the wrapped key and appends a signed tombstone; abbey-bot never sees raw keys. | §4.1, §4.3, §6.2 |
| D3 | Scope is everything about the member: WDBX records and edges plus abbey-bot's local copies (facts, transcript turns, projection, style observations); one command, one receipt of what was removed and retained, and why. | §6.6, §6.7 |
| D4 | Existing plaintext member records get a one-time migration to encrypted records verifiable by replay, with a signed migration receipt. | §7 |
| D5 | The host operator (abi CLI) and a guild owner (own guild only) may place and release signed, reasoned holds; an active hold blocks erasure, and the erasure request reports it. | §4.4, §4.7 |
| D6 (Q1) | Move payloads into WDBX, encrypted per member; WDBX becomes canonical for member facts; erasure destroys the key. Keyed commitments rejected as the end state. | §4.2, §6.5, §7 |
| D7 (Q3) | The host operator can erase any member in any scope directly, with a signed receipt. | §4.6, §6.4, §8.2 |
| D8 (Q9) | Rotate the master key after erasure, automatically, at most hourly, re-wrapping remaining member keys; the Keychain item is this-device-only, exact attribute verified on macOS 27.2 in the build step. | §6.3, §10 phase 2 |
| D9 (Q11) | Clear all channel summaries in the scope when a member there is erased. | §6.7 |
| D10 | A key per fact, wrapped by the member key; forgetting one fact destroys that fact's key. | §4.1, §4.2, §8.1 |
| D11 | The Keychain master uses after-first-unlock, this-device-only access; the exact attribute is verified on macOS 27.2. | §5.1, §6.2, §10 phase 2 |
| D12 | The new byte fields are base64 in ledger lines, and the ledger cap rises to 256 MiB. | §5.3, §10 phase 1 |
| D13 | Switchover is automatic after 7 days of shadow reads with zero mismatches, recorded as a signed event and logged; rollback is operator-only through the abi CLI; an environment kill switch disables auto-switchover. Chosen over operator-driven switchover. | §4.9, §7.2 |

Approval of this document for implementation: **not yet given.**
