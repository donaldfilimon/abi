# Program 5: Discord organization vertical slice

> **Constitutional mapping.** This file was written against the seven-program
> list Donald gave in conversation. The ratified constitution's section 13
> "Delivery programs" uses a different numbering, and the constitution wins.
> In section 13's terms this document specifies **Program 3, Read-only Discord guild intelligence** and **Program 5, Approved reversible guild execution**. It spans both, because a plan that cannot be applied and verified is not a slice.
>
> The filename is therefore name-based rather than numbered, so no numbering is
> asserted. Nothing in section 13 was renumbered: section 15 reserves amendment
> to Donald, and the collision is raised as one request covering the whole set
> rather than five independent ones.


Status: **proposed design. No implementation authority.** Nothing in this
document is evidence that any mechanism below exists.

Date: 2026-08-22. Author: design pass for Donald J. Filimon.

Governing document: `docs/superpowers/specs/2026-08-22-abbey-system-constitution.md`.
Where this spec and the constitution disagree, the constitution wins and this
spec is wrong. Constitution section 15 requires an amendment before any program
contradicts it; this spec proposes no amendment and claims no exemption.

Scope, in Donald's words:

> Audit a guild, infer its structure, propose an improvement, calculate
> permissions, generate a before/after diff, receive approval, apply reversible
> changes, verify Discord state, and record the outcome.

## 0. Relationship to the constitution

### 0.1 Evidence ladder

The ratified ladder is constitution section 11, levels **C0 through C7**. The
task framing for this program refers to an "L0 to L8" ladder with "L5 =
restricted canary succeeds under real authorization" and "L6 = postconditions
and rollback are verified". Those two intents map cleanly and this spec adopts
the ratified names rather than inventing a parallel taxonomy:

| Task framing | Ratified level | Ratified definition |
| --- | --- | --- |
| L5, restricted canary under real authorization | **C5** | Bounded canary with fixed scope, budget, monitoring, and rollback. Permits the claim "works under restricted live authority". |
| L6, postconditions and rollback verified | **C6** | An authorized operator witnesses the exact end to end outcome. Permits the claim "live qualified for that environment and version". |

No ratified level is *defined* as "postconditions and rollback are verified".
This program therefore treats postcondition verification and a witnessed
rollback as the **content** of its C6 gate, not as a level of its own. A C5
canary that applies a change and reports success without an operator witnessing
the re-read and the revert is a C5 claim, never a C6 claim. There is no C8.

### 0.2 Program 3 is a prerequisite, not part of this slice

Constitution section 13 assigns read-only guild intelligence, the metadata-only
guild twin, and `/abbey audit` / `/abbey plan` / `/abbey status` to **Program
3**. It assigns "preview, approval, staged execution, verification,
compensation, and per-guild Skill View manifests" to **Program 5**.

Donald's scope sentence spans both. This spec resolves that by consuming
Program 3 by reference. It specifies only the **delta** Program 5 needs from the
twin, because a change set cannot be computed from a presentation-shaped twin:

1. Raw integer `position` for every role, not a rendered ordering.
2. Raw `allow` and `deny` permission bitfields for every channel overwrite, not
   the human readable name lists that `src/perms.rs` uses today.
3. A per-object **read watermark** (the observation timestamp plus a digest of
   the object's mutable fields) so drift between plan time and apply time is
   detectable rather than inferred.
4. The channel-to-category link plus a computed `synced` flag per child channel
   (see section 6.4).
5. The bot's own member record, role set, guild-level effective permissions,
   and top role position, observed at the same watermark as everything else.

If Program 3 has not landed when this program is planned, Program 5 must
implement the read path against the same contract and hand ownership back, not
fork a second twin. Constitution decision 73: one canonical owner writes each
domain.

### 0.3 What this slice deliberately excludes

- **Per-guild Skill View manifests.** Constitution section 7 and decisions 48
  through 50 make manifest compilation, preview, hashing, and rollback a
  distinct mechanism with its own compiler and its own failure modes. It is the
  second half of Program 5 and is out of scope here. This slice being complete
  does not make Program 5 complete.
- **Destructive moderation.** Constitution section 13 keeps it proposal only
  until separately specified. Nothing here changes that.
- **Channel or role deletion, guild-level settings, integrations, webhooks, and
  emoji.** See section 5.3 for the closed action set.
- **Any use of message content.** Constitution section 8 and decision 20 make
  message content non-telemetry regardless of intent availability. This is a
  policy floor, not a capability limit.

## 1. Current state, verified by reading source

Every claim in this section was verified by reading the file cited on
2026-08-22. Nothing here is proposed.

### 1.1 Gateway intents

`src/main.rs:213` through `:224`. The client is built with
`GatewayIntents::non_privileged() | GatewayIntents::GUILD_VOICE_STATES`, and
adds `GatewayIntents::MESSAGE_CONTENT` only when `ABBEY_MESSAGE_CONTENT` is
set. `GUILD_MEMBERS` and `GUILD_PRESENCES` are absent in both branches.
`README.md` "Design notes" records the deliberate consequence: `/whois` reports
no presence, and member data is fetched over REST rather than read from the
gateway cache.

### 1.2 Permission explanation

`src/perms.rs` is a pure module with no serenity dependency. It models
`Scope::{Everyone, Role{id,name}, Member{id,name}, Unrecognized}` and
`Overwrite { scope, allow: Vec<String>, deny: Vec<String> }`.

Two properties matter for this program:

- **It explains evaluation order; it does not compute an effective result.**
  `applicable()` returns the overwrite chain in Discord's evaluation position
  (`@everyone`, then role overwrites, then the member overwrite, with
  `Unrecognized` appended last so an unmodelled kind never impersonates
  `@everyone`). `explain()` renders that chain as prose. Neither function
  performs a union, a deny mask, or a negation.
- **Permissions are carried as display name strings, not bits.** The `allow`
  and `deny` fields are `Vec<String>` populated by
  `commands.rs:129 permission_names()`, which calls serenity's
  `get_permission_names()`. Strings cannot be intersected, subtracted, or
  masked. Any arithmetic in this program is a **new typed layer over serenity
  `Permissions` bitfields**, not an extension of `perms.rs`.

`perms.rs` also already encodes three facts this program depends on: owner and
Administrator short-circuit before any overwrite is consulted; role position is
irrelevant when unioning role overwrites; and an overwrite with no bits set is
"cosmetic", which is not the same thing as the overwrite being absent.

### 1.3 Guild reads in the command shell

`src/commands.rs:148 fetch_member_and_guild()` fetches the member and the
`PartialGuild` concurrently **over HTTP rather than from the cache**, with the
stated reason that the cache is only as complete as the held intents and a
partial cache would silently produce a thinner answer rather than an error.
`PartialGuild` carries `roles` and `owner_id`.

`src/commands.rs:161 top_role_position()` returns the member's highest role
`position` as a `u16`, or 0 when the member holds only `@everyone`.

`/perms` (`src/commands.rs:391`) maps live channel overwrites into
`perms::Overwrite`, identifies `@everyone` as the role whose id equals the guild
id, tolerates a stale overwrite naming a deleted role, and refuses threads
outright because threads carry no overwrites of their own.

### 1.4 Hierarchy checking

`src/moderation.rs:136 hierarchy_blocker(actor_is_owner, actor_top_position,
target_is_owner, target_is_admin, target_top_position, is_timeout)` encodes
Discord's refusal rules for a human moderator. Its tests
(`moderation.rs:463`) pin that equal positions are refused and that strictly
greater is required. `/modcall` calls it after checking the permission bit via
`guild.member_permissions(&moderator)`, which is the canonical calculation that
includes `@everyone` grants and returns `all()` for the owner and for
Administrator.

`hierarchy_blocker` answers "can the invoking human do this". Program 5 needs
the same question answered about **the bot**, which is a different subject and a
separate call site.

### 1.5 Structural writes today

There are none. `/server` (`src/commands.rs:586`) renders a text blueprint from
`src/server.rs` and creates nothing; the README states the reason as structural
change belonging to a human who can see what already exists. `/webhook` is
likewise emit only, explicitly so the bot never learns a credential it minted.
`/modcall` recommends and never acts.

`src/server.rs` already property-tests three things this program reuses:
text channel names must already be in the form Discord stores (lowercased,
whitespace hyphenated) while voice names are exempt; every gated channel must
name a role the same blueprint creates; and `@everyone` must never carry a
dangerous permission.

### 1.6 Guild configuration and isolation

`src/guild.rs` keys everything by `scoped_guild_id("discord", Some(id))`, which
formats as `"discord:123"`. `GuildSettings` defaults are
`learning_enabled: true`, `unsolicited: false`, `reply_cooldown_seconds: 20`,
`unsolicited_per_hour: 6`. `GuildConfigStore` is a trait; the only
implementation in tree, `InMemoryGuildConfigStore`, is `#[cfg(test)]`.
`GuildRegistry` is a write-through cache with lazy hydrate and an `evict`.

The `learning_enabled: true` default is the current-source mismatch the
constitution already names in section 8. Program 5 does not depend on it and
does not fix it.

### 1.7 Interaction and message constraints

`README.md` "Design notes": every command defers unconditionally before
touching the network, because Discord invalidates an interaction token 3 seconds
after issuing it. `commands.rs:177 clamp_message()` truncates every reply at
2,000 codepoints with a truncation marker; its own comment records that a
channel with two overwrites carrying dozens of flags already exceeds that
limit. All replies send an empty allowed-mentions policy so generated or
guild-derived text can never notify anyone.

### 1.8 Memory substrate

`src/wdbx.rs` provides `WdbxStore` (key/value plus vectors, serialized as a
`# ABI-WDBX v1` JSONL segment) and `Recall` for user-scoped and guild-scoped
facts. Per the constitution section 5, the bot's atomic JSON state document is
canonical and the WDBX rows are a semantic projection until migration parity,
replay, recovery, deletion, and rollback evidence exist. This program writes its
episode through whichever writer is canonical at implementation time and never
through both. Constitution decision 77: no dual canonical writers.

## 2. The slice end to end

The stages map one to one onto constitution section 8, "Server change
workflow". Names in bold are the artifacts each stage produces.

| # | Stage | Artifact | Section |
| --- | --- | --- | --- |
| 1 | Observe current state | **GuildObservation** with per-object watermarks | 3 |
| 2 | Diagnose facts and inferences | **StructureModel** plus **FindingSet** | 4 |
| 3 | Generate alternatives | **ProposalSet**, at least two options plus "do nothing" | 4.4 |
| 4 | Simulate permissions and dependencies | **EffectPrediction** per affected principal and object | 6 |
| 5 | Render the exact proposed change set | **ChangeSet** plus **Diff** plus **RollbackPlan** | 5, 7, 9 |
| 6 | Obtain scoped approval | **Approval** bound to the ChangeSet digest | 8 |
| 7 | Stage one bounded step | **StepAttempt** written durably before dispatch | 10 |
| 8 | Revalidate current state and postconditions | **StepVerification** from a fresh read | 10, 11 |
| 9 | Continue, compensate, or stop safely | **RunOutcome** | 10.4, 12 |
| 10 | Issue a redacted receipt, update desired state | **Receipt** plus **WDBX episode** | 11 |

Constitution section 3 requires three visible interaction stages. They land as:
**Recommend** = stages 1 through 3, **Propose** = stages 4 and 5, **Execute** =
stages 6 through 10. A model may participate only in stages 2 through 4, and
only as a proposer. Constitution invariant A3 and decision 9: authorization is
never a generative decision.

### 2.1 Where a model is and is not in the loop

Proposed. The generator may rank findings, draft option prose, name a role, and
write the human explanation. Everything it emits is untyped text or a candidate
that must survive:

1. deterministic schema validation into the typed `ChangeSet` (section 5);
2. the permission simulator, which recomputes every predicted effect from
   observed bits rather than from model assertions (section 6);
3. the invariant checker (section 6.6);
4. ABI authorization (section 8.1).

The per-guild DQN is architecturally excluded from this path. Constitution
decision 36: reinforcement learning never directly changes roles, permissions,
channels, moderation, integrations, or commands. The DQN's action space in
`src/brain` is `stay / reply / react`, and this program adds nothing to it.

## 3. The guild audit: what is read and what that costs

### 3.1 What a metadata-only audit reads

Proposed read set, all metadata, no message content:

| Object | Source | Fields the change set needs |
| --- | --- | --- |
| Guild | REST `GET /guilds/{id}` (serenity `to_partial_guild`) | `owner_id`, `roles`, features, verification and content-filter levels, `system_channel_id`, `rules_channel_id`, `public_updates_channel_id` |
| Roles | Same response | `id`, `name`, `position`, `permissions` bitfield, `managed`, `hoist`, `mentionable`, `tags` (premium, integration, bot) |
| Channels and categories | REST `GET /guilds/{id}/channels` | `id`, `type`, `name`, `parent_id`, `position`, full `permission_overwrites` with raw `allow` and `deny` bitfields |
| Threads | REST active-threads listing | Existence and `parent_id` only. Threads carry no overwrites of their own; `src/commands.rs:391` already refuses to reason about them and this program inherits that refusal |
| The bot itself | REST `GET /guilds/{id}/members/@me` plus the role table | Role set, computed guild-level permissions, top role position |
| Named members | REST single-member fetch, only on operator request | Role set, for a spot check |
| Audit log | REST `GET /guilds/{id}/audit-logs`, optional | Recent structural changes, to date the last human edit of an object |

Explicitly not read: message history, message content, reactions, presence,
voice state beyond what the voice subsystem already owns, invites, integrations,
and member enumeration by default.

### 3.2 Intents: what `non_privileged()` does and does not give

Current: `src/main.rs:224` uses `non_privileged() | GUILD_VOICE_STATES`.

Visible with that set:

- `GUILDS`, which delivers `GUILD_CREATE` with the full role and channel tables
  and streams `GUILD_ROLE_CREATE/UPDATE/DELETE` and `CHANNEL_CREATE/UPDATE/DELETE`.
  This is the entire structural surface this program mutates, and it is
  available without any privileged intent.
- `GUILD_MESSAGES` and `GUILD_MESSAGE_REACTIONS` as message and reaction
  metadata, which this program does not use.

Not visible with that set:

- **Member content of the gateway member list.** `GUILD_MEMBERS` is absent, so
  there is no member chunking, and `GUILD_MEMBER_UPDATE` does not arrive. A
  member's role set changing is therefore **not observable as an event**; it is
  only observable by re-reading that member.
- **Presence.** `GUILD_PRESENCES` is absent. Already documented in `README.md`
  as the reason `/whois` reports no status.
- **Message content**, unless `ABBEY_MESSAGE_CONTENT` is set and the Dev Portal
  toggle is on. Irrelevant here by policy.

Proposed consequence, and it shapes the whole design: **the audit and the
effect prediction are role-shaped, not member-shaped.** Predictions are stated
over `@everyone`, over each named role, and over the bot. Per-member statements
are produced only for members the operator names explicitly, using the same
single-member REST fetch `/perms` and `/modcall` already use
(`src/commands.rs:148`).

This is deliberate insurance. Whether `GET /guilds/{id}/members` (the list
endpoint) requires the `GUILD_MEMBERS` privileged intent to be enabled for the
application must be **verified against current Discord documentation at
implementation time and recorded as an observation**, not assumed from this
document. Single-member fetch demonstrably works today, because `/perms` and
`/modcall` both depend on it. A role-shaped design is correct under either
answer; a member-enumeration-shaped design would have to be rewritten if the
gate exists. Independently of the API answer, constitution section 8 requires
metadata minimization and prefers aggregate measures, so enumeration would need
its own justification even if it were free.

### 3.3 Discord permissions the bot must hold

Proposed. Read side, per audit scope:

| Read | Bot permission required |
| --- | --- |
| Guild, role table, channel list with overwrites | None beyond guild membership |
| A channel the bot cannot see | `VIEW_CHANNEL` on that channel. A channel denied to the bot is **absent from what it can meaningfully audit**, and the audit must say so rather than silently under-report |
| Audit log | `VIEW_AUDIT_LOG` |

Write side:

| Action | Bot permission required |
| --- | --- |
| Create, edit, reposition, or delete a role | `MANAGE_ROLES` at guild level, plus hierarchy (section 6.5) |
| Assign or remove a role from a member | `MANAGE_ROLES` plus hierarchy over that role |
| Create, rename, or reparent a channel | `MANAGE_CHANNELS` |
| Edit a channel's permission overwrites | `MANAGE_ROLES` **on that channel**, which is itself deniable by a channel overwrite. `MANAGE_CHANNELS` does not imply it |

The distinction in the last row is a real trap: a bot with guild-level
`MANAGE_ROLES` can still be denied it on one channel by an overwrite, and the
edit will fail on exactly that channel. The simulator computes the bot's
**channel-effective** permission at every target, not its guild-level one.

### 3.4 Audit output

Proposed. `/abbey audit` returns a **GuildObservation** plus a **FindingSet**.
It performs no writes and requires no approval. It is owner or administrator
only, consistent with the treatment of `/voice verify` in `README.md`
"Verification and acceptance layers".

Every observation carries, per constitution section 8: source, observation
time, confidence basis, staleness policy, contradiction state, privacy class,
and schema version. Platform facts, Abbey inferences, and human-approved goals
are separate types and are never merged into one list.

## 4. Structure inference

### 4.1 Facts versus inferences

Proposed. The **StructureModel** holds only typed platform facts, restated:
role graph with positions and permission bits; channel forest with categories,
parents, positions, and overwrites; the `@everyone` role identified as the role
whose id equals the guild id (as `src/commands.rs:445` already does); managed
and integration roles marked as such.

The **FindingSet** holds inferences, each with the facts it rests on. Every
finding names its evidence so a human can disagree with the reasoning without
disputing the data.

### 4.2 Findings this slice computes

Proposed, deterministic, no model required:

| Finding | Rule |
| --- | --- |
| `EveryoneCarriesDangerousPermission` | `@everyone`'s guild-level bits intersect a configured dangerous set (`ADMINISTRATOR`, `MANAGE_GUILD`, `MANAGE_ROLES`, `MANAGE_CHANNELS`, `MANAGE_WEBHOOKS`, `MENTION_EVERYONE`, `BAN_MEMBERS`, `KICK_MEMBERS`, `MODERATE_MEMBERS`). `src/server.rs` already treats this as a blueprint invariant |
| `RedundantAdministrator` | A non-managed role holds `ADMINISTRATOR`, which makes every other bit on that role and every channel overwrite touching it inert |
| `DesyncedChannel` | A child channel's overwrites differ from its category's (section 6.4) |
| `OrphanOverwrite` | An overwrite targets a role id absent from the role table. `src/commands.rs:453` already renders these as "deleted role {id}" |
| `CosmeticOverwrite` | An overwrite with empty `allow` and `deny`. `perms.rs` already labels these "cosmetic" |
| `UnreachableChannel` | No role, including `@everyone`, resolves to `VIEW_CHANNEL` on a channel |
| `RolelessJoinerHasNoChannel` | A member holding only `@everyone` resolves to `VIEW_CHANNEL` nowhere. `src/server.rs` pins the inverse of this as a blueprint property |
| `GatedChannelWithNoGrantingRole` | `@everyone` is denied `VIEW_CHANNEL` and no role is allowed it. The runtime form of a `server.rs` blueprint property |
| `BotAtRiskOfLockout` | The bot's top role sits at or below a role the operator commonly edits, or the bot lacks `MANAGE_ROLES` on channels it is expected to manage |
| `UnrecognizedOverwriteKind` | A `PermissionOverwriteType` this build does not model. `perms.rs::Scope::Unrecognized` already carries these rather than folding them into `Everyone` |

The last row is load bearing. An unmodelled overwrite kind makes any prediction
about that channel **incomplete**, and section 6.6 turns that into a hard
refusal to propose a change touching that channel, not a caveat.

### 4.3 Staleness

Proposed. Every fact carries the watermark from section 0.2. A StructureModel
older than a configured freshness bound cannot back a ChangeSet; the plan is
recomputed rather than reused. Constitution section 5: similarity is not
applicability, and staleness is one of the retrieval dimensions that must be
considered rather than collapsed away.

### 4.4 Alternatives

Proposed. Constitution section 8 stage 3 requires alternatives, not one answer.
A ProposalSet contains at least two substantive options plus an explicit
"change nothing" option carrying the cost of inaction. Each option states what
it fixes, what it does not fix, what it makes harder to reverse, and its step
count. The operator picks one; the system does not pre-select a default, because
a pre-selected default is the beginning of silence-as-consent, which
constitution section 1 forbids Abbey to learn.

## 5. The typed change set

### 5.1 Shape

Proposed.

```
ChangeSet {
  schema_version: u16,
  plan_id: Uuid,
  scoped_guild_id: String,          // "discord:{guild_id}", per src/guild.rs
  proposed_by: PrincipalRef,        // the human who asked
  generator: GeneratorRef,          // model + version, or "deterministic"
  observation_watermark: Watermark, // the GuildObservation this rests on
  capability: CapabilityRef,        // id + semver, per constitution section 6
  steps: Vec<Step>,                 // ordered, serial, each individually reversible
  predicted_effects: Vec<EffectPrediction>,
  invariants: Vec<Invariant>,       // must hold at every intermediate state
  rollback: RollbackPlan,           // computed before approval, section 9
  budget: StepBudget,
  digest: Digest,                   // canonical encoding of everything above
}

Step {
  index: u16,
  action: Action,                   // closed enum, section 5.3
  target: TargetRef,                // object id + object watermark digest
  preconditions: Vec<Precondition>, // revalidated immediately before dispatch
  postconditions: Vec<Postcondition>, // verified by re-read, section 11
  inverse: Action,                  // the exact compensating action
  risk: RiskClass,
}
```

The `digest` is over the **full** ChangeSet, canonically encoded. Section 8.2
binds approval to this digest and not to the summary a human happened to see.

### 5.2 Reversibility is a per-step property, not a plan property

Proposed. A step is admissible only if its `inverse` is a well-formed `Action`
that restores the exact prior value, and the prior value is captured in the
RollbackPlan **before** the step is dispatched. A step whose inverse is not
expressible is not a Program 5 step. This is why deletion is out of scope: the
inverse of deleting a channel is not creating a channel with the same name,
because the id changes and every overwrite, webhook, pin, and thread that
referenced it does not come back.

### 5.3 The closed action set for this slice

Proposed. Deny by default: anything not on this list is not an `Action`.

| Action | Inverse | Notes |
| --- | --- | --- |
| `CreateRole { name, permissions, hoist, mentionable, position }` | `DeleteRole { id }` | The only creation whose inverse is genuinely clean, because a freshly created role has no history to lose. Still refused if anything has been granted to it, which serial ordering prevents within a run |
| `EditRolePermissions { role_id, from_bits, to_bits }` | `EditRolePermissions { role_id, from: to_bits, to: from_bits }` | `from_bits` is a precondition, not decoration |
| `RepositionRoles { ordering_before: Vec<(RoleId, u16)>, ordering_after: Vec<(RoleId, u16)> }` | Restore `ordering_before` | One step covering the whole affected set, see section 6.5. `@everyone` is position 0 and cannot be moved, so it is rejected from both orderings at validation |
| `SetChannelOverwrite { channel_id, target, allow, deny, prior: Option<Overwrite> }` | Restore `prior`, or `RemoveChannelOverwrite` when `prior` was `None` | See section 6.3 on the empty-versus-absent distinction |
| `RemoveChannelOverwrite { channel_id, target, prior: Overwrite }` | `SetChannelOverwrite` restoring `prior` | |
| `CreateChannel { name, kind, parent_id, position, overwrites }` | `DeleteChannel { id }` | Same clean-inverse argument as `CreateRole`. Deleting a channel that has since received messages is refused at rollback time and reported as unresolved rather than performed |
| `RenameChannel { channel_id, from, to }` | Rename back | Text names normalized per `src/server.rs::normalize_text_name`; voice and category names exempt |
| `ReparentChannel { channel_id, from_parent, to_parent, sync: bool }` | Reparent back with the prior sync state and prior overwrites restored | Reparenting with sync overwrites the child's overwrites, so the prior set is captured in full |
| `AddRoleToMember` / `RemoveRoleFromMember` | Each other | Only for members the operator named. Never applied in bulk in this slice. `@everyone` is not assignable or removable and is rejected at validation |

Deliberately absent: `DeleteRole` and `DeleteChannel` as *forward* actions,
every guild-settings edit, every integration, webhook, emoji, sticker, and
automod action, and every moderation action.

## 6. Discord permission arithmetic

This is a **new typed layer** over serenity `Permissions` bitfields. It is not
an extension of `src/perms.rs`, which carries permission names as strings and
cannot express union, mask, or negation (section 1.2). `perms.rs` remains the
human explanation surface and should keep its role; the simulator is the
computation surface, and the two must agree on ordering.

### 6.1 Guild-level base

```
base(member) =
  if member.id == guild.owner_id      -> ALL
  else
    bits = @everyone.permissions
    for role in member.roles: bits |= role.permissions
    if bits & ADMINISTRATOR           -> ALL
    else                              -> bits
```

Two facts already relied on in tree: `@everyone` is the role whose id equals the
guild id (`src/commands.rs:445`), and serenity's
`guild.member_permissions(&member)` is the canonical implementation, which
`/perms` and `/modcall` both use because hand-rolling it is how the
`@everyone` bug happened. **The simulator must use serenity's implementation for
the observed present state** and reserve its own arithmetic for the predicted
future state, where no member object exists to pass in. Divergence between the
two on the present state is a bug in the simulator, and a differential test
between them (section 13.2) is the cheapest way to catch it.

### 6.2 Channel-level resolution

Applied only when `base` did not short-circuit to `ALL`:

```
1. bits = base(member)
2. everyone overwrite on this channel: bits &= !deny; bits |= allow
3. role_deny = union of deny over every applicable role overwrite
   role_allow = union of allow over every applicable role overwrite
   bits &= !role_deny; bits |= role_allow
4. member overwrite on this channel: bits &= !deny; bits |= allow
```

Step 3 is the one people get wrong, and `src/perms.rs`'s module doc already says
so: **role position is irrelevant here.** All applicable role denies union
before all applicable role allows. Step 4 wins outright.

Additional rules the simulator must encode:

- Owner and `ADMINISTRATOR` bypass steps 2 through 4 entirely.
- `VIEW_CHANNEL` denied implies everything else on that channel is unreachable
  in practice, so a prediction that grants `SEND_MESSAGES` without
  `VIEW_CHANNEL` is reported as ineffective rather than as a grant.
- Voice channels resolve the same way; text-specific bits on a voice channel are
  inert and are reported as inert.
- Threads have no overwrites of their own and inherit the parent's. Consistent
  with `src/commands.rs:427`, this program refuses to predict at thread
  granularity and points at the parent.
- An `Unrecognized` overwrite kind makes the channel's resolution
  **incomplete**. See section 6.6.

### 6.3 Empty overwrite is not absent overwrite

Proposed, and it is the difference between a working rollback and a silent
mutation. An overwrite with `allow = 0, deny = 0` exists, occupies a slot,
and is what `perms.rs` labels "cosmetic". An absent overwrite does not exist.
They resolve identically today and diverge the moment anything else is edited.

Therefore: **the inverse of "created an overwrite" is `RemoveChannelOverwrite`,
never `SetChannelOverwrite` with zeroed bits.** The `prior: Option<Overwrite>`
field in `SetChannelOverwrite` carries `None` precisely to make this
expressible.

### 6.4 Category inheritance is a copy, not a live lookup

Proposed, and this is the single item most likely to be gotten wrong.

Discord does not resolve a channel's permissions by walking up to its category
at evaluation time. When a channel is "synced" to its category, Discord
**copied** the category's overwrites onto the channel at sync time. Editing the
category afterwards does not retroactively change a child that has since
diverged, and there is no inheritance link to consult.

Consequences for every step touching a category:

1. Each child channel is classified at observation time as **synced** (its
   overwrite set is identical to the parent's) or **desynced** (it is not).
   The classification is a computed fact carried in the twin, per section 0.2.
2. A predicted effect on a category is stated **per child**, using each child's
   actual classification. A blanket "and this cascades to the channels below"
   is a false prediction for every desynced child.
3. `ReparentChannel { sync: true }` replaces the child's overwrites wholesale.
   The prior set is captured in full in the RollbackPlan, because the inverse is
   restoring a set, not toggling a flag.
4. If a child is desynced and the operator's stated intent implies they think it
   is synced, that is a finding to surface before approval, not a surprise to
   discover in the diff.

### 6.5 Role hierarchy

Proposed. Two separate subjects, both of which must pass.

**The bot.** The bot can only manage a role whose `position` is **strictly
below** its own highest role's position. Equal position is refused by Discord.
`src/moderation.rs:463` already pins the strictly-greater rule for the human
case; the bot case is a distinct call site with the bot as subject. The bot also
cannot edit a `managed` role (an integration or bot role), regardless of
position.

**The invoking human.** The same rule, already implemented as
`moderation::hierarchy_blocker`. An approval from someone who could not perform
the change themselves is an authority laundering path and is refused at the
approval gate (section 8.1), not merely warned about.

**`@everyone` is structurally fixed.** It sits at position 0, cannot be
repositioned, cannot be deleted, and cannot be added to or removed from a
member. Any step attempting one of those is refused at validation, before
dispatch, in the same class as the hierarchy refusals above. Its permission
bits and its channel overwrites remain editable, subject to the
`EveryoneStaysSafe` invariant in section 6.6.

**A bot cannot grant a permission it does not hold.** Before dispatch, the
actuator computes `requested_bits & !bot_effective_bits`. Non-empty means the
step is refused, named bit by bit, before any network call. `ADMINISTRATOR` is
the common case, and it is a real refusal rather than an attempted call that
Discord rejects, because a rejected call still consumes rate budget and still
leaves the run in an in-flight-unknown state (section 10.2).

**Repositioning renumbers others.** Discord's role positions are dense and
contiguous; moving one role shifts every role between the old and new slot. A
`RepositionRoles` step therefore captures the **complete prior ordering of the
whole affected range**, not just the moved role, and its inverse restores that
whole ordering. Capturing only the moved role produces a rollback that leaves
neighbours displaced.

### 6.6 Invariants checked at every intermediate state

Proposed. These are checked against the simulator's predicted state after
**each** step, not only against the final target, and against **each rollback
state** as well, because a rollback runs from an intermediate state and can pass
through configurations the forward plan never visited.

| Invariant | Rule |
| --- | --- |
| `BotRetainsManagement` | After every step, and after every rollback step, the bot still holds `VIEW_CHANNEL` and `MANAGE_ROLES` on every channel the remaining plan touches, still holds guild `MANAGE_ROLES` and `MANAGE_CHANNELS` where the remaining plan needs them, and its top role still sits strictly above every role the remaining plan touches |
| `OwnerUnaffected` | No step changes the owner's effective anything. The owner bypasses everything anyway, so a step that appears to is a modelling error |
| `EveryoneStaysSafe` | `@everyone` never gains a bit from the dangerous set. Runtime form of the `src/server.rs` blueprint property |
| `NoAdministratorGrant` | No step grants `ADMINISTRATOR` to anything, ever, in this slice |
| `RolelessJoinerKeepsAChannel` | At least one channel remains `VIEW_CHANNEL`-visible to a member holding only `@everyone`. Runtime form of the second `src/server.rs` usability property |
| `NoHumanLosesAccessSilently` | Any predicted loss of `VIEW_CHANNEL` for any role is surfaced in the diff as a loss, in its own section, never folded into a bit-count delta |
| `CompletePrediction` | No step targets a channel carrying an `Unrecognized` overwrite kind. Prediction there is incomplete, and constitution section 3 requires deny or pause on a failed invariant rather than a caveated proceed |

`BotRetainsManagement` is the lockout guard. The failure it prevents is
concrete: step 3 removes the bot's `MANAGE_ROLES` on a channel, step 4 needs it,
and the rollback of step 3 also needs it. The check must run forward and
backward or it does not prevent the failure it exists for.

## 7. The before and after diff

### 7.1 Content

Proposed. Four sections, in this order, because the order encodes priority:

1. **Access losses.** Every principal-object pair that loses `VIEW_CHANNEL`,
   `CONNECT`, or `SEND_MESSAGES`. Stated as "role @X loses View Channel in
   #Y", never as a bitfield delta. Empty is stated as empty, not omitted.
2. **Structural changes.** Roles created, permission bits changed with the
   named bits on both sides, positions moved with before and after numbers,
   channels created, renamed, or reparented, overwrites set or removed with the
   distinction from section 6.3 made visible.
3. **Access gains.** Same shape as section 1.
4. **Unchanged but at risk.** Findings the plan does not address, and any
   desynced children touched by a category step (section 6.4).

Every line cites the step index that causes it. A diff line with no step is a
bug in the renderer.

### 7.2 Delivery, given a 2,000 character ceiling

Current: `commands.rs:177 clamp_message()` truncates at 2,000 codepoints, and
its own comment records that two overwrites with dozens of flags already exceed
that. A realistic ChangeSet diff will not fit.

Proposed. The interaction reply carries a **summary**: plan id, step count,
counts per diff section, the highest risk class present, the invariant results,
and the expiry. The **full diff** is delivered as a file attachment or through
explicit pagination controls.

The binding rule: **approval is bound to the digest of the full ChangeSet**
(section 5.1), never to the summary. A human who approves without opening the
full diff has still approved the full plan, and the receipt records which
artifacts were rendered so a later review can see what was actually in front of
them.

### 7.3 The interaction window bounds the UI, not the job

Current: every command defers unconditionally (`README.md` "Design notes")
because the interaction token dies after 3 seconds. Deferral extends the
followup window to roughly 15 minutes, which is an upper bound on the
conversation, not on the work.

Proposed. Planning is synchronous and must fit comfortably inside the deferred
window. **Application is a background job keyed by `plan_id`** that reports
through followups while the window is alive and through a fresh message keyed to
the plan id afterwards. A run must never be structured such that an expired
interaction token leaves the guild mid-change with no way to report. Progress
state lives in the durable step ledger (section 10.2), not in the interaction.

## 8. The approval gate

### 8.1 What is checked

Proposed. Constitution section 3 requires every consequential request to carry
the authenticated principal and delegation chain, capability id and version,
scope, validated parameters, current platform permission and hierarchy facts,
policy versions, approval references, deadline, cancellation, idempotency key,
rate class, and correlation id. The ChangeSet plus Approval envelope carries all
of these.

ABI returns Allow, Approval required, or Deny/pause. For this slice, **every**
ChangeSet with a non-empty step list is Approval required. There is no
auto-allow tier in this program. Constitution decision 13: high-consequence
execution requires explicit human approval, and structural guild change is
high-consequence by classification, not by size.

The gate additionally refuses when:

- the approver could not perform the change themselves (section 6.5, human
  hierarchy). This closes the authority laundering path.
- the approver is not the guild owner or an administrator.
- any invariant in section 6.6 fails on any predicted intermediate state.
- the observation watermark is staler than the freshness bound.
- the capability version in the ChangeSet is not the currently promoted version.
  Constitution decision 46: schema drift disables the affected version.

### 8.2 Re-authorization at the click

Proposed, and it is not a formality. A button press is a **fresh interaction
with a fresh member object**. Between render and click the approver may have
lost a role, the bot may have been demoted, and the guild may have changed.

On click, before anything is dispatched:

1. Recompute the approver's permissions and hierarchy from a fresh fetch.
2. Recompute the bot's effective permissions and top role position.
3. Re-read the observation watermark for every object the plan targets, and
   compare digests. Any mismatch stops the run before step 1 and reports drift
   with the specific object named.
4. Verify the plan digest matches the one the approval references.

An Approval carries a short expiry and applies to exactly one `plan_id`.
Constitution decision 16: repeated approval does not become standing authority.
Constitution decision 17: revocation takes effect before new work begins, so a
revocation arriving mid-run stops the run at the next step boundary and enters
the compensation path.

### 8.3 Cancellation

Proposed. A run is cancellable at any step boundary by the approver, by an
administrator, or by the safety path without model consultation (constitution
section 3). Cancellation stops forward progress and enters compensation; it does
not abandon the run silently. `ABBEY_QUIET` is the higher global override
(constitution decision 33) and blocks a run from starting.

## 9. The rollback plan, captured before any mutation

### 9.1 Captured before approval, not before dispatch

Proposed. The RollbackPlan is computed and stored **as part of the ChangeSet**,
before the human sees the diff, so the approver is approving a plan whose
reversal is already known. It is not derived at failure time from whatever the
guild happens to look like then.

For each step it holds the exact prior value: prior permission bits, prior
overwrite or its absence, prior complete role ordering for the affected range,
prior parent and prior sync state and prior overwrite set. Not a description of
the prior value: the value.

### 9.2 What it cannot restore, stated up front

Proposed. The plan names its own limits in the diff:

- A created role or channel is deleted on rollback. If it received members,
  messages, or grants in the interim, deletion loses that. Rollback of a
  `CreateChannel` whose channel has received messages is **refused** and
  reported as unresolved rather than performed.
- Discord does not offer a transaction across structural calls (constitution
  section 8 says this explicitly). Rollback is compensation, not a transaction
  abort, and partial rollback is a real outcome with its own receipt shape.
- Anything a third party changed between apply and rollback is not restored.
  Rollback restores the value this run set, and only if the current value still
  matches what this run set. If it does not, the step is left alone and reported
  as `SupersededByThirdParty`.

### 9.3 Rollback ordering

Proposed. Reverse step order, serial, with the section 6.6 invariants checked
before each compensating step. Reverse order matters: if step 2 created a role
and step 5 granted it a channel overwrite, removing the overwrite must precede
deleting the role.

## 10. Reversible application

### 10.1 Serial, bounded, revalidated

Proposed.

- **Serial.** One step in flight at a time. Concurrency makes drift detection
  and rollback ordering unsound, and it multiplies rate pressure at the moment
  the run is least able to absorb it.
- **Bounded.** A `StepBudget` caps step count, wall clock, and retry count.
  Constitution section 8 requires separate per-guild budgets for structural
  change, distinct from speech and observation budgets.
- **Revalidated immediately before each step**, per constitution section 4 stage
  8 and the degraded-operation clause in section 10: "If Discord state changes
  during a plan, the adapter revalidates and stops rather than applying a stale
  operation." Preconditions include the target object's watermark digest **and**
  the bot's own effective permissions and top role position, because the bot can
  be demoted mid-run by someone else.

### 10.2 Four step states, not three

Proposed. The durable step ledger records, per step:

| State | Meaning | How it is reached |
| --- | --- | --- |
| `NotStarted` | No call dispatched | Initial |
| `InFlightUnknown` | Intent written durably, outcome unknown | Written **before** the call. A crash, timeout, or ambiguous error leaves the step here |
| `Applied` | Verified by re-read (section 11) | Never set from a response body alone |
| `RolledBack` | Compensating action verified by re-read | |

The intent must be durably written **before** the call so that a crash-restart
can classify the step at all. `InFlightUnknown` resolves **only by re-reading
Discord**, never by assuming either outcome.

This ledger is also how the idempotency key required by constitution section 3
is honored. Discord's structural endpoints are not idempotent, so idempotency is
enforced adapter-side: a step already `Applied` is skipped on resume, and a step
in `InFlightUnknown` is resolved by observation before anything else happens.

### 10.3 Rate limiting

Proposed. Discord's per-route and per-guild limits on structural edits are not
stated here as numbers, because a number written into a design document goes
stale and then gets trusted. The rules:

- Honor `Retry-After` on every 429, including global 429s.
- A 429 is **pause and revalidate**, never retry in place. The wait window is
  precisely when a third party is most likely to mutate the same objects, so the
  step's preconditions are re-checked after the wait, before the retry.
- Repeated 429s against the step budget stop the run and enter the receipt path
  rather than grinding.
- If the request deadline expires mid-plan, stop and receipt. A deadline is a
  constraint, not a suggestion.

### 10.4 Continue, compensate, or stop

Proposed decision rule after each step:

| Situation | Action |
| --- | --- |
| Step verified, invariants hold, budget remains | Continue |
| Step verified, an invariant now fails | Stop, compensate from here, receipt |
| Step failed cleanly (4xx, no mutation) | Stop, compensate prior steps, receipt |
| Step `InFlightUnknown` | Re-read. Resolve to `Applied` or `NotStarted`, then apply the matching rule. If the re-read is itself inconclusive, stop and receipt as unresolved, and **do not compensate a step whose state is unknown** |
| Drift detected at revalidation | Stop, compensate prior steps, receipt naming the drifted object |
| Cancellation or revocation | Stop at the boundary, compensate, receipt |
| Budget or deadline exhausted | Stop, compensate, receipt |

Compensation is itself verified by re-read and can itself fail. A failed
compensation produces a receipt naming completed, reverted, and unresolved steps
separately, per constitution section 10.

## 11. Verification by re-reading Discord

### 11.1 The rule

Proposed. **A successful Discord API call does not prove a good outcome.**
Verification re-reads the specific object over REST and recomputes effective
permissions with the **same simulator used at plan time**, then compares the
result against the step's `Postcondition`.

The response body of the mutating call is not verification. It is an echo, and
it cannot show what the change did to anyone's effective permissions.

Gateway events (`GUILD_ROLE_UPDATE`, `CHANNEL_UPDATE`) arrive on the
`non_privileged()` intent set and are useful corroboration and useful for drift
detection. They are not the verification, because their absence is not evidence
of anything.

### 11.2 Unverified is a first-class outcome

Proposed. Reads can lag writes. The verifier performs a bounded read-after-write
retry with a bounded delay, and if the postcondition still does not hold it
classifies the step as **`Unverified`**, distinct from both `Applied` and
`Failed`.

`Unverified` never rounds to success. It stops the run, and the receipt says
"applied, not verified" for that step, which is exactly the honest statement.
Constitution section 11 and decision 63: evidence never auto-promotes a higher
claim.

### 11.3 What the postcondition checks

Proposed, per step:

1. The object's own fields match the intended values.
2. The recomputed effective permissions for every principal named in that step's
   `EffectPrediction` match the prediction.
3. Every section 6.6 invariant still holds.

Point 2 is where prediction meets observation, and its result is what the
episode records as predicted-versus-observed (section 12). A prediction that was
wrong is a calibration signal and must be recorded as such even when the
operator is happy with the result. Constitution decision 66: negative and
rollback evidence remains publishable.

## 12. The WDBX episode

### 12.1 Envelope

Proposed. The episode answers constitution section 5's five questions and uses
the normative envelope. Fields specific to this program:

| Field | Content |
| --- | --- |
| Scope | Pseudonymous `scoped_guild_id` per `src/guild.rs`. Guild isolation is the correctness boundary; nothing here is retrievable from another guild |
| Operating context | Guild archetype if known, role and channel counts, findings count, regime, and the resource state at plan time |
| **Prediction** | The full `EffectPrediction` set, plus the uncertainty basis (which facts were stale, which overwrite kinds were unmodelled, which channels were invisible to the bot) |
| Action | The redacted step sequence: action kinds, target **classes** (role, channel, category), permission bit **names**, step count. Object ids are pseudonymized within guild scope |
| **Observation** | The verification result per step: `Applied`, `Unverified`, `Failed`, `RolledBack`, `SupersededByThirdParty` |
| Predicted versus observed | The per-principal, per-object diff between prediction and re-read, which is the calibration record |
| Outcome | Utility signal (operator accepted, operator reverted, operator amended), safety effect, cost, rollback state |
| Authority | Approver principal reference, approval id, plan digest, capability id and version, policy version, hierarchy facts at approval and at each dispatch |
| Evidence dimensions | Kept individually inspectable, never collapsed to one scalar. Constitution section 5 |
| Retention | See 12.2 |
| Edges | Supersession when a later plan changes the same objects; contradiction when observation contradicted prediction; revocation when an approval was withdrawn mid-run |

### 12.2 Retention class

Proposed, per constitution section 5:

- The **ChangeSet, Approval, and Receipt** are **Durable**: they have defined
  utility (a future plan must know what the last plan did), a defined authority
  basis (an approval record), and a privacy basis (metadata only, no message
  content, no member identities beyond the approver and any member the operator
  explicitly named).
- The **generator's draft prose and any model prompt or response** are
  **Ephemeral** and are never written. Constitution decision 21: prompts and
  generated responses are not durable operational evidence by default.
- A **failed or rolled-back run** is retained at the same class as a successful
  one. Preserving failures is one of the criteria the constitution imports in
  section 0.
- A run that trips a safety invariant is **Mandatory incident** and stores the
  smallest evidence required.

### 12.3 The write gate

Proposed. The episode passes through the selective write gate rather than being
written unconditionally. Section 1.8 applies: the bot's atomic JSON state
document is canonical until WDBX migration parity exists, and this program
writes through exactly one canonical writer.

Constitution section 5: integrity is not truth. A signed, digest-matching
episode establishes that this run produced this record. It does not establish
that the change was a good idea, and nothing in the episode may be phrased as
though it does.

## 13. Failure modes

### 13.1 The four named in scope

**Partial application.** Steps 1 through 3 applied, step 4 failed. Handled by:
serial execution so the boundary is unambiguous; the four-state ledger so every
step has a known or explicitly unknown state; compensation in reverse order; and
a receipt that names completed, reverted, and unresolved steps separately rather
than reporting one aggregate verdict. Constitution section 10: if a rollback is
incomplete, the receipt identifies completed, reverted, and unresolved steps
without exposing private content.

**Rate limiting mid-change.** Handled by section 10.3. The specific danger is
not the delay, it is that the delay is a window for third-party mutation, which
is why a 429 forces revalidation rather than a bare retry. A run that exhausts
its retry budget stops with a receipt; it does not silently continue later.

**A role moved by someone else between plan and apply.** The most likely real
failure, and the reason positions are captured as raw integers with watermarks
(section 0.2). Detected at three points: the approval click (section 8.2), the
per-step precondition check (section 10.1), and the post-step verification
(section 11). Any of the three detecting it stops the run. The receipt names the
object and states that the plan was computed against a state that no longer
exists. The plan is **recomputed from a fresh observation**, never patched.
A repositioning by a third party can also silently invalidate the bot's own
hierarchy over a target, which is why the bot's top role position is a
precondition on every step and not a once-per-run check.

**Permission changes that lock the bot out of its own management ability.** The
`BotRetainsManagement` invariant in section 6.6, checked after every predicted
intermediate state and every predicted rollback state, in both directions. The
concrete failure it prevents: a step removes the bot's channel-level
`MANAGE_ROLES`, a later step needs it, and the rollback of the first step also
needs it, so the guild is left mid-change with no automated way back. Because
the check runs on rollback states too, a plan whose *reversal* would lock the
bot out is refused at planning time, before approval.

### 13.2 Additional failure modes this design must handle

| Mode | Handling |
| --- | --- |
| Bot demoted mid-run by a human | Bot effective permissions and top role position are per-step preconditions. Detected at the next boundary; run stops and compensates while it still can |
| Approver loses authority between render and click | Re-authorization at the click, section 8.2 |
| Channel invisible to the bot | Absent from the audit with an explicit statement that it is absent. Never silently under-reported, and never a target |
| Unrecognized overwrite kind | `CompletePrediction` invariant refuses to plan against that channel. `perms.rs::Scope::Unrecognized` already prevents the worse failure of misattributing it to `@everyone` |
| Orphan overwrite naming a deleted role | Surfaced as a finding; a step may remove it, and its inverse restores it exactly |
| Two runs against one guild | A per-guild run lock. A second run is refused, not queued, because its observation is already stale |
| Simulator disagrees with serenity on present state | A differential test comparing the simulator against `guild.member_permissions()` over generated guild fixtures is part of the C1 gate. Divergence is a simulator bug, and shipping with it means every prediction is suspect |
| Diff exceeds 2,000 characters | Section 7.2. Approval binds to the full digest, so truncation cannot narrow what was approved |
| WDBX or ABI unavailable | Constitution section 10 degraded operation: with ABI authorization unavailable, consequential execution is denied. A run in progress stops and compensates rather than proceeding unauthorized |
| Process restart mid-run | The durable ledger classifies every step. `InFlightUnknown` steps are resolved by re-read before anything else. Constitution: no silent fresh start after durable-state corruption |

## 14. Acceptance and evidence placement

Preregistered, per constitution section 11. Thresholds are set before results
are inspected.

| Level | Evidence required for this slice | Permitted claim |
| --- | --- | --- |
| **C0** | This document plus the typed contracts, invariants, and falsification criteria | Specified |
| **C1** | Unit and property tests: simulator differential against `guild.member_permissions()`; overwrite ordering matching `perms.rs::applicable()`; empty-versus-absent overwrite; category sync classification; reposition renumbering; hierarchy strictness for bot and human; every section 6.6 invariant including on rollback states; privacy tests proving no message content and no member identity beyond the approver reaches a receipt or a log | Source conforms under test |
| **C2** | Deterministic replay: a recorded guild observation plus a recorded operator decision sequence reproduces byte-identical ChangeSet, diff, and RollbackPlan | Replay qualified |
| **C3** | Offline evaluation against recorded guild fixtures including adversarial ones (drift injected between plan and apply, 429 injected mid-run, bot demoted mid-run, unrecognized overwrite kind present). Baseline: the deterministic no-model path | Adds measured offline value |
| **C4** | Shadow: `/abbey plan` produces ChangeSets against a real guild with **no write path compiled in**, and predictions are compared against what the operator does by hand | Predicts acceptably in the target environment |
| **C5** | Bounded canary in one operator-owned test guild: fixed step budget, one action kind at a time starting with `SetChannelOverwrite`, monitoring, and rollback available | Works under restricted live authority |
| **C6** | An authorized operator witnesses one complete run: audit, plan, diff, approve, apply, **postcondition verification by re-read**, and a **witnessed rollback restoring the observed prior state** | Live qualified for that environment and version |
| **C7** | Repeated runs across guilds and versions establishing reliability and drift bounds | Sustained operational evidence |

C6 is the level at which this program's characteristic claim becomes
permissible, and its content is exactly the postcondition and rollback
verification the task framing called L6. A run that applies successfully but
whose rollback is not witnessed is a C5 result.

The scorecard (constitution section 11) for this program: authorization
false-allows and false-denies; prediction-versus-observation divergence rate per
principal-object pair; rollback success rate; unresolved-step rate; run
cancellation success; latency and step count; and evidence completeness.

Per constitution decision 67, this program cannot be its sole evaluator. The
simulator's correctness is judged by the differential against serenity, and the
live outcome is judged by the operator, not by the run's own success flag.

## 15. Canary boundary and program rollback

Proposed, per constitution decision 80.

**Canary boundary.** One guild, owned by Donald, with the bot's top role placed
deliberately. One action kind enabled at a time. Step budget in single digits.
No `CreateChannel` or `CreateRole` until `SetChannelOverwrite` and
`EditRolePermissions` have C6 evidence, because the former two have the messier
rollback stories.

**Program rollback.** The write path is behind a capability that can be revoked
without redeploying, and revocation takes effect before new work begins
(constitution decision 17). With the capability revoked, `/abbey plan` continues
to produce a ChangeSet and a diff and the apply path refuses. That degraded mode
is the Program 3 behavior, so the rollback target is a state that already exists
rather than a new one.

**Falsification criteria.** This design is wrong, and must be revised before
promotion, if any of these is observed:

1. The simulator diverges from `guild.member_permissions()` on any generated
   fixture.
2. A category step's predicted effect is wrong for a desynced child.
3. A rollback leaves any object in a state the RollbackPlan did not name.
4. Any step reaches `Applied` without a re-read.
5. The bot loses management ability during any run or any rollback.
6. Any receipt or log contains message content, or a member identity other than
   the approver and members the operator explicitly named.

## 16. Conflicts found, and open questions

### 16.1 Conflicts with the constitution

**None that this spec asserts.** Three tensions were resolved by deferring to
the constitution rather than by contradicting it:

1. **Ladder naming.** The task framing's "L0 to L8" is not the ratified ladder.
   Resolved by mapping in section 0.1 and using C0 through C7 throughout.
2. **Program boundary.** Donald's scope sentence spans constitution Program 3
   (audit, infer) and Program 5 (propose, approve, apply, verify, record).
   Resolved by consuming Program 3 by reference (section 0.2) rather than
   re-owning the twin.
3. **Program 5 completeness.** The constitution's Program 5 also includes
   per-guild Skill View manifests, which this slice excludes. Stated explicitly
   in section 0.3 so nobody reads a completed slice as a completed program.

### 16.2 Conflicts with existing code

1. **`src/perms.rs` cannot do the arithmetic this program needs.** Permissions
   are `Vec<String>` display names and the module computes ordering, not
   effective results. This is not a defect in `perms.rs`, whose job is
   explanation, but it does mean the simulator is a new module and the two must
   be kept in agreement by test rather than by shared code.
2. **`GuildSettings.learning_enabled` defaults to `true`** (`src/guild.rs`),
   against constitution decision 31. Already recorded in constitution section 8
   as a migration this program does not perform and must not depend on.
3. **`GuildConfigStore` has no non-test implementation.** `InMemoryGuildConfigStore`
   is `#[cfg(test)]` only. A durable per-guild store for the run ledger and the
   structural budget is a prerequisite this program cannot assume exists.
4. **`clamp_message()` at 2,000 characters cannot carry a real diff.**
   Addressed in section 7.2, but it is a genuine constraint on the operator
   experience and not merely a rendering detail.

### 16.3 Open questions requiring observation, not assumption

1. Whether `GET /guilds/{id}/members` requires the `GUILD_MEMBERS` privileged
   intent to be enabled for the application. Must be verified against current
   Discord documentation and recorded as an observation. The role-shaped design
   in section 3.2 is correct under either answer, which is why it was chosen.
2. Current per-route and per-guild rate limits for role and channel edits. To be
   measured, not quoted. Section 10.3 states rules rather than numbers for this
   reason.
3. Whether the canonical episode writer at implementation time is the atomic
   JSON state document or WDBX. Determined by Program 4's migration state, not
   by this program.
4. Whether the Discord audit log is reliable enough to attribute third-party
   drift to a principal, or only to detect that drift happened. This slice
   assumes only detection.
