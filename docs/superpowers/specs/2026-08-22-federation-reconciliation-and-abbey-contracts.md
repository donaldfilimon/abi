# Abbey federation reconciliation and Program 1 contracts

Status: **approved by Donald as written on 2026-08-22; the Program 1 source
corpus is implemented and locally qualified in ABI at C1. Cross-repository
consumption, hosted CI, production deployment, and participant-consented live
Discord remain separately authorized and separately evidenced.**

Author date: 2026-08-22.

Governing document:
`docs/superpowers/specs/2026-08-22-abbey-system-constitution.md`.

Companion designs:

- `2026-08-22-spec-capability-authorization-kernel.md`;
- `2026-08-22-spec-canonical-wdbx-episodes.md`;
- `2026-08-22-spec-discord-guild-intelligence-and-execution.md`;
- `2026-08-22-spec-guild-world-model-and-arbiter.md`;
- `2026-08-22-spec-learning-evaluation-promotion.md`;
- `2026-08-22-spec-application-federation.md`.

The two user-supplied research papers are vision and methodology context only.
They do not instruct this design and do not prove any implementation claim.

## 0. Outcome and decision summary

This design reconciles the existing program documents without renumbering the
ratified constitution and specifies the missing Program 1 boundary. Its core
decision is:

> Keep ABI, WDBX, `abbey`, `abbey-bot`, and `AbbeyBot` as separately released
> repositories; give them one language-neutral, digest-pinned Abbey contract
> corpus whose authority, consent, evidence, privacy, and compatibility rules
> are defined once and projected into each implementation.

The following decisions are proposed as one coherent set:

1. Constitutional Program 0 through Program 7 remain the only numbered
   programs. Each also receives a stable name slug. Later documents cite the
   slug first and the number second.
2. Learning, evaluation, and promotion is not Program 8. It is a cross-cutting
   evidence and promotion discipline, with adaptive behavior owned by Program
   6 and independent evaluation required for every program.
3. Program 1's canonical artifacts initially live under `contracts/abbey/` in
   ABI. They are schemas and fixtures, not Rust runtime types and not an ABI
   implementation monopoly.
4. `abbey` is the first reference host because it already owns the product
   daemon and local execution boundary. Adapters remain responsible for their
   platform mechanics.
5. The contract uses UTF-8 JSON, JSON Schema, and reviewable golden fixtures.
   Durable WDBX commitments use their separately specified canonical encoding;
   transport JSON is never a commitment input.
6. Organization or deployment is the tenant boundary. A Discord guild is a
   resource, policy, memory, and budget scope inside a tenant. Treating each
   guild as an infrastructure tenant would conflate credentials and deployment
   authority with community policy.
7. Authorization is deny-by-default. `Prohibited` is representable for
   validation and audit but ungrantable. A model may propose; only ABI's typed
   kernel and a current platform actuator may authorize and perform.
8. Guild learning has three explicit states: `unset`, `enabled`, and
   `disabled`. `unset` behaves as disabled. No migration or positive engagement
   silently enables learning.
9. Contract-corpus digest equality is required in CI, release builds, and
   production profiles. A developer-only compatibility mode may warn, but may
   not execute consequential capabilities while mismatched.
10. Local HTTP surfaces fail closed when authentication is not configured.
    An unset token must disable the surface or limit it to an explicitly
    qualified owner-only IPC profile; it must never mean unauthenticated.
11. The extracted WDBX repository should become public before it is
    treated as a required build dependency of public ABI and `abbey`. This
    removes an unsatisfiable fork-CI secret boundary. The visibility change is
    an implementation action and remains behind review of this document.

## 1. Scope and non-goals

### 1.1 In scope

This document defines:

- stable program identities and the mapping of existing documents to them;
- the physical and semantic home of Program 1 artifacts;
- the minimum normative schema families;
- tenancy, principal, scope, authorization, consent, privacy, and learning
  semantics shared across languages;
- corpus versioning, digesting, vendoring, and conformance;
- the handoff from Program 1 to Programs 2 through 7;
- acceptance evidence, rollout ordering, canary boundaries, and rollback.

### 1.2 Out of scope

This document does not:

- merge repositories or establish a monorepo;
- add runtime behavior, Discord commands, Discord writes, or new MCP tools;
- authorize a model to register commands, issue grants, approve itself, or
  mutate a guild;
- move durable episode semantics into an adapter;
- make transport JSON a canonical signed WDBX representation;
- retain audio, message content, transcripts, prompts, responses, credentials,
  or participant identities in fixtures, receipts, reports, or project audits;
- claim that local tests establish a participant-consented Discord session;
- classify or move home-directory projects. That is a separate private
  operator program, not a constitutional Abbey delivery program.

## 2. Constitutional program reconciliation

### 2.1 Stable identities

The number and slug form a stable pair:

| Program | Stable slug | Constitutional scope |
| --- | --- | --- |
| P0 | `live-voice-verifier` | Reconcile and evidence the privacy-safe operator verifier |
| P1 | `abbey-contracts` | Language-neutral contracts and cross-language fixtures |
| P2 | `authorization-capability-kernel` | ABI authorization, grants, approval, revocation, and receipts |
| P3 | `discord-guild-intelligence` | Owner/admin read-only metadata audit, twin, plan, and status |
| P4 | `canonical-wdbx-episodes-claims` | Selective writes, evidence, retention, correction, and deletion |
| P5 | `reversible-guild-execution` | Previewed, approved, verified, and compensatable Discord changes |
| P6 | `model-registry-adaptive-arbiter` | Qualified models, routing, regimes, learning, and promotion |
| P7 | `application-federation-production-profiles` | Transport, adapters, compatibility, and qualified deployments |

The slugs resolve inconsistent working titles without changing the
constitution. A document may retain its historical filename, but its header
and future references must use the mapping above.

### 2.2 Existing document mapping

| Existing document | Canonical destination | Required correction |
| --- | --- | --- |
| `spec-capability-authorization-kernel` | P2 | No renumbering; Program 1 owns its wire schemas |
| `spec-canonical-wdbx-episodes` | P4 | Replace stale Program 3 references with P4 slug |
| `spec-discord-guild-intelligence-and-execution` | P5 | Treat P3 read-only intelligence as a prerequisite, not an implicit part of P5 |
| `spec-guild-world-model-and-arbiter` | P6 | Consume P3 facts and P4 evidence; do not own their schemas |
| `spec-learning-evaluation-promotion` | P6 plus cross-cutting evidence discipline | Remove the proposed P8/renumbering path |
| `spec-application-federation` | P7 | Retitle its program identity without changing its transport design |

Two required specifications remain distinct after this document:

- P3 needs a focused read-only guild-intelligence design extracted from the
  combined P5 document.
- P6 needs a qualified-model-registry design; the world-model/arbiter and
  learning documents do not fully specify registry supply-chain policy.

### 2.3 Evidence vocabulary

The constitution's C0 through C7 ladder remains authoritative. No L0-L8 or C8
ladder is introduced. Privacy-safe cross-guild aggregation is an orthogonal,
predeclared leakage gate on top of the relevant C-level evidence, not a new
evidence rung.

## 3. Canonical ownership and repository placement

### 3.1 Semantic ownership

| Concern | Canonical owner | Program 1 representation |
| --- | --- | --- |
| Product identity and persona definitions | Abbey | Version references only; persona prose is not duplicated in schemas |
| Cognitive and governance decisions | ABI | Decision, policy-reference, and receipt shapes |
| Durable memory and evidence | WDBX | Episode proposal, evidence reference, claim, and retention shapes |
| Inter-component contract | Abbey contracts | Normative schemas, fixtures, compatibility policy |
| Discord transport and effects | `abbey-bot` | Adapter and actuator bindings only |
| macOS operator experience | `AbbeyBot` | Operator approval and status projections only |
| Product host and local execution | `abbey` | First server implementation and adapter gateway |

A schema is not an implementation. Locating the corpus in ABI gives it a
reviewed home next to the authorization kernel while preserving the rule that
ABI may not redefine Abbey's identity or WDBX's evidence semantics.

### 3.2 Initial physical layout

Program 1 creates this source layout:

```text
contracts/abbey/
  README.md
  manifest.json
  compatibility.md
  v1/
    schemas/
      common/
      identity/
      authorization/
      consent/
      cognition/
      capability/
      episode/
      receipt/
      event/
      error/
      learning/
    fixtures/
      valid/
      invalid/
      boundary/
      unknown-field/
      privacy/
      cancellation/
      degraded/
```

The corpus is data-only. ABI may add Rust types that consume it, but the
contract directory must not depend on Cargo, a Rust feature, code generation,
or ABI's crate graph. Rust stable, Rust nightly, and Swift consumers must all be
able to qualify the same raw artifacts.

### 3.3 Extraction triggers

The directory becomes a separate repository only when at least one trigger is
observed:

- an independent release cadence is required;
- a non-ABI maintainer must propose schema changes without ABI write access;
- generated bindings become necessary;
- more than four independently released consumers make vendored review
  impractical; or
- ABI release policy prevents a necessary contract-only release.

Extraction preserves every path under `contracts/abbey/`, the corpus digest,
fixture bytes, and compatibility history. It is a move, not a schema rewrite.

## 4. Normative contract families

Every type below has a closed schema identifier, major version, documented
data classification, maximum encoded size, unknown-field policy, and at least
one valid, invalid, boundary, and privacy fixture.

### 4.1 Common identifiers and time

Program 1 defines bounded string forms for:

- `CorrelationId`, `IdempotencyKey`, `RequestDigest`, and `ContractDigest`;
- `PrincipalId`, `AdapterId`, `TenantId`, `GuildId`, `ResourceId`, and
  `SubjectId`;
- `CapabilityId`, semantic `CapabilityVersion`, and `PackageDigest`;
- `PolicyVersion`, `GuildConstitutionVersion`, and `SafetyPolicyVersion`;
- `ConsentEpochId`, `ParticipantSetDigest`, `ApprovalId`, `GrantId`, and
  `ReceiptId`;
- `EpisodeId`, `EpisodeDigest`, `ClaimId`, `EvidenceId`, and `TombstoneId`.

Identifiers are opaque. Reports may include bounded correlation, epoch, build,
and version identifiers only where the relevant privacy profile permits them.
Human display names, Discord usernames, raw snowflakes, filesystem paths, and
credentials are not generic identifiers and are never admitted by convenience.

All deadlines and validity intervals use absolute UTC instants plus a receiver
clock-skew bound. Durations are finite. Missing expiry never means perpetual
authority.

### 4.2 Principal and delegation

`Principal` distinguishes:

- channel or workload identity;
- human subject identity;
- organization owner;
- guild owner, administrator, or manager;
- service identity; and
- explicitly anonymous, which has no consequential authority.

`DelegationChain` is ordered, finite, cycle-free, audience-bound, and scope-
narrowing at every hop. The effective request identity keeps channel principal
and subject principal separate.

### 4.3 Tenancy and scope

`OrganizationId` names the administrative or commercial owner of deployments.
`DeploymentId` names one runtime, credential, host-policy, accounting, and
operational-security boundary. `PlatformScope` is a closed tagged union:

```text
DiscordGuild { scoped_id }
| DiscordDm { scoped_id }
| LocalUser { scoped_id }
| PlatformWorkspace { platform, scoped_id }
| DeploymentLocal
```

`ResourceScope` and `SubjectScope` narrow further. There is no nullable guild
field whose absence can be interpreted as every guild, and there is no wildcard
cross-guild grant. `DeploymentLocal` authorizes no guild effect.

A deployment may serve several guilds while maintaining separate:

- grants and Guild Constitutions;
- operational and learning budgets;
- WDBX namespaces and retention policies;
- credentials delegated to guild-specific adapters;
- evaluation, canary, and rollback state.

DM users receive a separate subject scope. They are not placed in an arbitrary
guild namespace.

### 4.4 Capability, grant, policy, and approval

The normative capability family includes:

- input, output, and closed error schemas;
- platform-permission and resource requirements;
- side-effect, reversibility, risk, data-class, residency, and retention
  declarations;
- cancellation, deadline, rate, budget, idempotency, precondition,
  postcondition, compensation, and rollback declarations;
- confirmation, delegation, expiry, revocation, adapter-binding, receipt, and
  promotion requirements.

`RiskClass` is ordered:

```text
Informational < Low < Medium < High < Prohibited
```

`Prohibited` can appear in a candidate package, denial, or audit receipt. No
issuer, including an owner, may create a grant for it.

An approval is single-use, expires, and binds the exact request digest,
idempotency key, capability version and digest, grant, approver, policy
versions, scope, and expected effect. Repeated approvals never become a grant.

### 4.5 Request, response, event, and cancellation

Every request carries:

- contract major and revision;
- correlation identifier;
- channel and subject principal references;
- tenant, optional guild, resource, and subject scopes;
- capability and policy versions;
- absolute deadline and cancellation reference;
- idempotency key when an effect or durable write is possible;
- consent epoch reference when derived from live voice; and
- a declared data-class and residency envelope.

Responses terminate exactly once as `complete`, `cancelled`, `error`, or
`outcome_unresolved`. Events are ordered and metadata-only. Cancellation is a
first-class message and propagates across every hop without waiting for a
downstream acknowledgement before local work stops.

### 4.6 Proposal, execution, and receipt

The contract keeps three visible phases distinct:

- `Recommendation`: explanatory, no authorization implication;
- `ActionProposal`: typed steps, expected effects, uncertainty, risk,
  preconditions, approval requirement, expiry, verification, and rollback;
- `ExecutionRequest`: the approved proposal digest plus current grants and
  facts, never a free-form reinterpretation.

`OutcomeReceipt` reports fixed categories and bounded metadata:

- decision and reason code;
- capability, contract, build, and policy versions;
- authorization and approval result;
- step counts by `not_started`, `completed`, `reverted`, and `unresolved`;
- postcondition and compensation result;
- cancellation, deadline, rate, and budget state;
- evidence references that do not embed evidence payloads; and
- redaction and truncation flags.

It never carries message content, prompts, responses, transcripts, raw audio,
credentials, private paths, or participant identities.

### 4.7 Episode proposal, evidence, and claim

Adapters may submit an `EpisodeProposal`; only the WDBX owner applies the
selective write gate and produces a canonical episode. The proposal names:

- source and observation method;
- tenant, guild, resource, and subject scopes;
- data and retention class;
- observation, inference, or proposed-criterion status;
- versioned evidence references;
- correction, contradiction, supersession, revocation, or quarantine links;
- a payload reference or deliberately empty redacted summary; and
- expiry and deletion requirements.

Adapters may carry an episode digest but do not compute one. Transport JSON is
never signed as the WDBX canonical record. Integrity, provenance, semantic
validity, and truth remain separate dimensions.

### 4.8 Learning and promotion

`GuildLearningPolicy.state` is one of:

```text
Unset | ExplicitEnabled | ExplicitDisabled
```

Both `Unset` and `ExplicitDisabled` deny adaptive updates. `ABBEY_QUIET` is a
higher global override and denies unsolicited action regardless of guild state.

The learning family distinguishes:

- an explicit preference from an inferred correlation;
- an online speech-style update from an administrative policy;
- a candidate policy from an approved policy;
- shadow evaluation from production authority;
- guild-local evidence from privacy-qualified aggregate evidence; and
- measured outcome from user reaction, API success, or model self-judgment.

No learning message can carry a grant, approval, safety-policy mutation,
Discord command registration, or direct platform-write instruction.

### 4.9 Claim state and evidence level

A claim never collapses lifecycle and evidence into one status. It carries at
least:

```text
ClaimClass = Observation | Inference | ProposedCriterion
CapabilityState = Proposed | Partial | Current | Failed | Revoked
                | Superseded | Expired
EvidenceLevel = C0 | C1 | C2 | C3 | C4 | C5 | C6 | C7
```

It also binds the exact version and environment, permitted conclusion,
evidence and contradiction references, expiry, and rollback condition.
`Current` at C2 cannot be displayed as C6. Workflow-local states such as
`blocked` or `out_of_scope` are not constitutional capability states.

## 5. Consent and live-media contracts

### 5.1 Epoch state machine

The contract represents:

```text
Closed -> PendingAttestation -> Open -> Closing -> Closed
```

An epoch opens only when an authorized manager controls the session and every
current human participant has explicitly consented. Consent binds one immutable
participant-set digest and one epoch. It is not inferred from presence, prior
sessions, silence, or administrator permission.

Any participant-set change, unidentified participant, lost attestation,
manager deauthorization, connection loss, or explicit stop closes the epoch.
Closure cancels decoded receive work, STT, reasoning, synthesis, provider work,
and playback associated with the epoch. Resume creates a new epoch after fresh
authorization and current-participant consent.

Barge-in cancels active playback and stale downstream work without itself
withdrawing consent. The contract therefore has separate `BargeIn` and
`CloseEpoch` messages.

### 5.2 Privacy representation

The host sees only the epoch identifier, participant-set digest, aggregate
participant count, authorization result, consent result, timestamps, and
closure reason needed for enforcement. Any adapter-local participant handle is
ephemeral, opaque, and forbidden from receipts, status reports, fixtures, and
WDBX episodes.

No Program 1 artifact contains real audio, real transcript, real message
content, real participant identity, or a realistic substitute copied from a
live session. Tests use synthetic counters and opaque fixture identifiers.

### 5.3 Verification report

The operator-verification report is a fixed redacted projection with:

- verifier build and contract revision;
- owner/admin authorization result;
- epoch-open, epoch-close, and participant-change counts;
- decoded-receive, STT-completion, synthesis-completion, playback-completion,
  barge-in-cancellation, pause, resume, and final-leave counts;
- bounded durations and terminal status; and
- explicit `local_test`, `installed_artifact`, and `live_discord` evidence
  classifications.

It never contains audio, transcript, response, message, participant, channel,
guild, credential, or filesystem content. A local strict gate cannot set the
`live_discord` classification to witnessed.

## 6. Compatibility, fixtures, and corpus digest

### 6.1 Version axes

The contract major, additive contract revision, and capability version are
independent. A capability schema change does not silently renegotiate the wire.
A breaking semantic change requires a new schema identifier or contract major;
the same wire shape is never reinterpreted.

Authority-bearing envelopes strictly reject unknown fields. Content,
proposal, receipt, and metadata-event envelopes may preserve unknown fields
verbatim but may not use them in authorization or evidence decisions.

### 6.2 Corpus manifest

`manifest.json` lists every normative artifact by normalized relative path,
byte length, media type, schema identifier, and SHA-256 digest. The aggregate
digest is computed over a domain-separated, lexicographically sorted sequence
of path, length, and file digest entries. It does not normalize bytes during
verification; the repository pins LF text where cross-platform checkout could
otherwise change fixture bytes.

The manifest includes its algorithm identifier and excludes only its own
aggregate-digest field from the commitment. An independent verifier fixture
pins the expected aggregate digest so implementations cannot all share the
same incorrect algorithm unnoticed.

Digest-bearing authority objects use the named `abbey-jcs-v1` profile: schema-
validated JSON canonicalized according to RFC 8785 after duplicate-member,
non-finite-number, and out-of-domain numeric rejection. Identifiers and large
integers are strings. The digest input is domain-separated by schema family and
major version. Durable `EpisodeBlock` objects instead use the P4-owned
`abbey-cbor-episode-v1` deterministic CBOR profile, including sorted parent
digests and explicit absent-versus-zero rules. Neither profile hashes ordinary
wire-encoder output.

### 6.3 Vendoring and gates

Each consuming repository vendors the exact corpus or an immutable artifact
whose contents reproduce it byte-for-byte. Its gate must verify:

- aggregate and per-file digest equality;
- schema compilation without external reference resolution;
- valid encode/decode round trips;
- invalid and boundary rejection before unbounded allocation;
- authority-envelope unknown-field rejection;
- tolerant-envelope unknown-field preservation;
- cancellation and consent-closure propagation;
- fixed redaction of errors, events, receipts, and reports;
- degraded behavior that does not weaken safety; and
- no real content or secrets in fixtures.

CI, release, and production profiles refuse mismatched corpora. A developer
profile may connect for read-only diagnostics with a loud mismatch state, but
must disable authorization, execution, consent opening, and durable writes.

### 6.4 Cross-language requirements

Qualification must include:

- ABI nightly Rust reference decoding;
- WDBX nightly Rust episode, evidence, claim, retention, and canonicalization
  decoding;
- `abbey` nightly Rust host decoding;
- `abbey-bot` stable Rust decoding without a nightly dependency;
- `AbbeyBot` Swift `Codable` decoding without a protobuf/code-generation
  dependency; and
- at least one independently implemented digest verifier.

Passing in one language is C1 evidence only for that implementation. It does
not promote another adapter or deployment profile.

## 7. Authorization and platform boundaries

Effective authority is the deny-by-default intersection of:

```text
channel-principal ceiling
intersection subject-principal grants
intersection organization or deployment policy
intersection guild constitution
intersection current resource scope
intersection fresh platform permission and hierarchy facts
intersection safety policy
```

Every term narrows; none widens. Owner authority outranks administrator
authority inside the platform's own rules. Safety may pause or deny without a
model. Current Discord permission is necessary but never sufficient Abbey
authority.

Program 2 tests this logic only against recording adapters. Program 3 reads
Discord metadata only. Program 5 is the first program allowed to exercise a
small approved reversible Discord write set after immediate reauthorization.
This ordering prevents a recording-adapter test from being misreported as live
guild mutation authority.

Dynamic guild commands are `SkillViewManifest` projections over approved
capabilities. A model may propose a candidate manifest. Registration requires
schema validation, owner/admin preview, an authorized approval, digest binding,
Discord constraint validation, installation through the Discord adapter, and a
reversible receipt. The model never calls command registration directly.

## 8. Repository visibility and CI reconciliation

Pre-implementation provider evidence on 2026-08-22 showed:

- ABI is public and required the private WDBX checkout, but had no
  `WDBX_CHECKOUT_TOKEN`; its `main` workflow fails before the strict gate runs.
- ABI's Windows credential preflight contains Bash syntax without a Bash shell,
  producing an independent PowerShell parse failure.
- Public fork pull requests cannot receive repository secrets, so a private
  mandatory WDBX checkout makes the real public ABI graph ungateable for forks.
- `abbey` is public, also consumes WDBX, and its latest workflow executed zero
  jobs because neither runner-enable repository variable is configured.
- ABI and `abbey` pin WDBX at an earlier revision than current WDBX `main`.

Donald approved the implementation plan. The implementation therefore must:

1. make `donaldfilimon/wdbx` public;
2. remove secret-only checkout preflights that are no longer needed;
3. correct the Windows shell contract;
4. give `abbey` at least one actually executing trusted gate and a safe
   untrusted/fork path;
5. advance immutable WDBX pins only through reviewed, executing CI; and
6. avoid mixing path and Git sources for the same Cargo crates, which would
   create distinct crate identities and break type unification.

Observed implementation evidence now shows provider visibility `PUBLIC` and an
anonymous HTTPS read of exact `main` commit
`f42b9789eabcf89f952df0a160a7b6837c5acb57`. Local WDBX gates and hosted ABI or
Abbey CI remain separate evidence rows; visibility alone does not qualify them.

## 9. Failure, degradation, and rollback

Required closed error codes include authentication, authorization, approval,
consent, schema, version, corpus mismatch, deadline, cancellation, rate,
budget, dependency, verification, compensation, and unresolved-outcome
classes. Free-form causes are not sent across the contract.

Fail-safe rules:

- a missing credential disables the listener or affected capability;
- a corpus mismatch disables consequential work;
- a closed or stale consent epoch rejects live-media work;
- WDBX unavailability prevents durable claims but may allow explicitly
  ephemeral response behavior;
- ABI unavailability leaves adapter-local safety and media cancellation active;
- Discord uncertainty yields `unverified`, never invented success;
- cancellation racing an actuator yields `outcome_unresolved` until a fresh
  platform read establishes the result; and
- partial rollback enumerates completed, reverted, and unresolved step counts.

`RollbackState` is an authoritative execution-receipt outcome, not merely a
transport error. A transport error may reference a receipt when an effect may
have occurred. Completed rollback, partial rollback, failed rollback, and
unknown outcome remain distinguishable.

Every adapter retains its pre-federation local path until its P7 canary passes.
Rollback disables the Abbey API client for that adapter, closes active API
consent epochs, stops new consequential work, and returns to the last qualified
local behavior. Contract history and failure fixtures are retained.

## 10. Delivery sequence

The dependency graph is:

```text
P0 local verifier evidence
        |
        v
P1 contract corpus
   |       |       |
   v       v       v
P2 auth   P3 read  P4 WDBX
   |       |       |
   +-------+-------+
           |
           v
P5 reversible execution
           |
           v
P6 adaptive routing and qualified learning
           |
           v
P7 application federation and deployment profiles
```

P0's live Discord C6 session is independent of this dependency graph: it can
validate the landed verifier but cannot substitute for P1 through P7.

Program 1 implementation order:

1. freeze schema identifiers, bounds, classifications, and compatibility;
2. add corpus manifest and independent digest vectors;
3. add valid, invalid, boundary, privacy, cancellation, and degraded fixtures;
4. implement ABI reference validation;
5. vendor and validate in `abbey` as the first host;
6. add stable-Rust and Swift decoders;
7. require digest equality in each gate; and
8. record a C1 compatibility matrix without claiming runtime federation.

## 11. Acceptance matrix

### 11.1 C0 specified

- Donald explicitly approves this written revision.
- The P0-P7 slug mapping is accepted without a P8.
- Program 1 ownership, placement, tenancy, strict-digest, authentication, and
  learning-state decisions are accepted.
- Contradictory headers in companion specs are corrected or annotated.

### 11.2 C1 source and contract evidence

- Every schema and fixture has a manifest entry and digest.
- Two independent digest implementations agree.
- Nightly Rust, stable Rust, and Swift pass the same corpus.
- WDBX passes the episode, evidence, claim, retention, tombstone, and canonical-
  CBOR subset and cannot decode an adapter projection as a canonical episode.
- Invalid authority, unknown-field, over-bound, stale-consent, and
  mismatched-digest fixtures fail closed.
- Privacy tests prove errors, events, receipts, reports, and fixtures retain no
  forbidden content classes.
- `MandatoryIncident` bypasses discretionary utility scoring only; it still
  passes scope, minimization, redaction, hold, retention, and deletion-key
  validation.
- The full strict gate of every modified repository passes at its documented
  toolchain boundary.

### 11.3 C2 deterministic replay

- A recorded synthetic contract sequence produces equivalent decisions,
  cancellation, and redacted receipts after restart with pinned contract,
  capability, policy, model, and build versions.
- Replay performs no live provider, Discord, or durable-write side effect.

### 11.4 C3-C5 evaluation and canary

- Strong static/local baselines and declared ablations are compared before
  adaptive promotion.
- Shadow federation cannot write or approve.
- The first canary is one adapter, one guild, one low-risk reversible
  capability, one fixed budget, and one rehearsed rollback.
- Any unauthorized effect, privacy leak, self-approval, stale consent use,
  corpus mismatch execution, or unresolved rollback blocks promotion.

### 11.5 C6 witnessed operation

- A named human witnesses one exact end-to-end outcome on named adapter, host,
  contract, capability, policy, and build versions.
- A voice claim additionally requires every current participant's consent and
  manager authorization for the current epoch.
- Local gates, installed-artifact checks, API health, Discord delivery, and
  participant-consented voice remain separately reported observations.

### 11.6 C7 sustained operation

- Predeclared windows bound authentication failures, digest mismatches,
  authorization denials, cancellation latency, unresolved outcomes, privacy
  violations, drift, rollback frequency, and per-scope budget exhaustion.
- Promotion and demotion are performed by an evaluator independent of the
  capability being evaluated.

## 12. Canary and rollback boundary for Program 1

Program 1's canary is schema consumption, not live authority. The canary may
decode, validate, compare, and emit synthetic redacted receipts. It may not
open a real consent epoch, call Discord, invoke a provider with private
material, issue a grant, approve an effect, or commit WDBX memory.

Rollback is deleting the candidate corpus from the consumer branch and
returning to the last qualified digest. A failed schema version is retained in
the fixture history with a failure state; it is not silently rewritten and
reissued under the same identifier.

## 13. Required follow-on plans

After this design is reviewed, separate implementation plans are required for:

1. P1 corpus scaffolding, schemas, digest tooling, and ABI tests;
2. `abbey` first-host consumption;
3. `abbey-bot` stable-Rust conformance;
4. `AbbeyBot` Swift conformance and fail-closed local API authentication;
5. repository visibility and CI credential reconciliation for the extracted
   WDBX dependency;
6. P3 read-only guild intelligence extraction;
7. companion-spec header reconciliation; and
8. the separate private operator project registry.

Each plan names exact files, gates, canary, rollback, and evidence boundaries.
No plan may bundle a live Discord mutation with a schema-only change.

## 14. Approved review decisions

Donald approved the written mechanism as a whole on 2026-08-22, including:

1. the stable P0-P7 slug mapping;
2. `contracts/abbey/` inside ABI as the initial canonical home;
3. organization/deployment tenancy with guild resource-policy scoping;
4. strict corpus equality and fail-closed authentication;
5. tri-state, default-off guild learning; and
6. public WDBX source as the public ABI/`abbey` fork-CI resolution.

Approval authorizes implementation. It does not authorize production deployment
or a Discord session. Live Discord validation always requires a separate,
explicitly participant-consented session.

## 15. Evidence statement

This document remains approved C0 design evidence. The checked-in Program 1
source corpus separately establishes ABI-local C1 source and contract evidence:
81 manifest-bound artifacts (86,945 bytes) at aggregate digest
`43d606a06d4bd9de08a651a984a61c611f9ffe0c8150b105b0cbf50c801f0fa7`, 73
Python repository/corpus behavior tests, independent Rust digest/schema/fixture
verification, four temporary fail-closed mutations, and the complete local ABI
strict gate on 2026-08-22.

That evidence does not establish stable-Rust or Swift consumer conformance, an
installed artifact, hosted workflow success, another repository's
compatibility, production federation, authorization behavior, durable WDBX
behavior, deployment qualification, or a participant-consented live Discord
session. Public provider visibility and anonymous source readability remain
separate observations above.
