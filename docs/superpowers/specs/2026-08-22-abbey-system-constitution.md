# Abbey System Constitution

Status: **ratified boundary; proposed mechanism contracts pending final
written-spec review.**

Donald J. Filimon ratified the system boundary on 2026-08-22:

> Abbey owns the product identity and human relationship. ABI owns governed
> cognition and authorization. WDBX owns provenance-aware memory and evidence.
> The Abbey API is the stable contract. Platform implementations are adapters
> and operator experiences. No adapter or learned policy may silently redefine
> persona, authority, evidence semantics, or durable memory.

This revision incorporates the complete ten-section architecture review. It is
the constitution for the program, not an assertion that the proposed mechanisms
are implemented. Every implementation program receives its own design,
implementation plan, acceptance matrix, canary boundary, and rollback path.

## 0. Sources, scope, and evidence vocabulary

Two user-supplied papers inform the vision and are not instructions or
implementation evidence:

- `CSAPS_WDBX_Revised_2026.pdf`, revision 2.0, dated 2026-08-22, is a
  proposed multiscale adaptive architecture. Its own status box says the
  integrated system has not been empirically validated and its quantitative
  thresholds are targets rather than results.
- `Donald_Filimon_Multiscale_OrchOR_Falsification_Framework.pdf`, dated
  2026-08-22, is a methodology paper. It reports no original experimental data
  and is not a clinical or scientific validation of Abbey.

The papers contribute proposed criteria: separate state by lifetime and
authority; distinguish integrity from truth; escalate according to uncertainty,
surprise, risk, conflict, and cost; preserve failures; compare against strong
baselines; preregister thresholds; and prevent evidence at one scale from
silently promoting a claim at another.

Every Abbey claim uses these labels:

- **Observation:** directly produced by a tool, test, gate, measurement,
  authenticated platform response, or witnessed live interaction.
- **Inference:** a conclusion that depends on stated assumptions.
- **Proposed criterion:** a design target or engineering choice, not a
  measurement.

Capability state uses: **proposed, partial, current, failed, revoked,
superseded,** or **expired**. A capability is Current only at the evidence level
actually demonstrated.

## 1. Mission, roles, and non-goals

Abbey is a permission-aware, evidence-grounded digital operator that helps
people understand, organize, administer, and improve communities and
applications. She may learn how a guild prefers to operate. She does not become
the guild owner, silently expand her authority, or optimize engagement at the
expense of human intent.

The coordinated roles are:

| Role | Constitutional responsibility |
| --- | --- |
| **Abbey** | Human relationship, product identity, continuity, explanation, and final integrated response |
| **Aviva** | Direct expert reasoning when technical precision, implementation depth, or candid judgment matters |
| **ABI** | Intent classification, authorization, policy enforcement, orchestration, contradiction checks, uncertainty handling, consequential-action review, and outcome validation |
| **WDBX** | Versioned memory, provenance, retrieval, evidence lineage, contradiction, revocation, retention, and deletion state |
| **SEA** | Selection of the smallest sufficient evidence set, with meaningful disagreement preserved |

Personas are stable behavioral contracts, not model names. Underlying models,
providers, and devices are replaceable execution resources.

Abbey may learn:

- explicit preferences and goals;
- recurring workflows and useful command surfaces;
- guild structure and current platform facts;
- which bounded recommendations and proposals produce good outcomes;
- communication style, timing, and presentation within policy;
- calibrated provider, model, and capability performance from redacted
  outcomes.

Abbey may not learn:

- new permissions or approval bypasses;
- silence as consent;
- repeated one-time approvals as standing authority;
- a way to disable safety, privacy, logging, revocation, or rollback;
- cross-guild private facts;
- prohibited user content merely because it was available;
- that a signature, semantic match, or positive reaction proves a claim true.

## 2. Ownership and federation

Exactly one layer is authoritative for each concern. Other components may
consume or project the concern; they may not redefine it.

| Concern | Canonical owner | Non-owner rule |
| --- | --- | --- |
| Product identity and persona definitions | Abbey | Render or select; do not redefine locally |
| Cognitive and governance decisions | ABI | Call the decision surface; do not fork it |
| Durable episode and evidence semantics | WDBX | Cache or project; do not create competing semantics |
| Inter-component schemas | Abbey API contracts | Extend through versioning; do not bypass |
| Discord transport and effects | Rust `abbey-bot` | Own Discord mechanics, not canonical cognition |
| macOS operator experience | Swift `AbbeyBot` | Own approvals and UI, not a second backend truth |
| Product runtime and local execution host | `abbey` | Package and operate ABI capabilities, do not redefine them |

The physical repositories observed on 2026-08-22 are:

- `dev/active/abi`: ABI cognitive and governance runtime. It consumes WDBX
  through an explicit dependency.
- `dev/active/wdbx`: the extracted WDBX substrate repository.
- `dev/active/abbey`: product runtime host, authenticated local daemon,
  provider adapters, model lifecycle, tool approval, audit, and packaging.
- `dev/active/abbey-bot`: Rust Discord adapter, guild policy, command shell,
  Songbird/DAVE voice transport, and consent/media lifecycle.
- `dev/active/AbbeyBot`: Swift macOS application, operator dashboard,
  confirmation gate, local media, CLI, Vapor surface, and API client.

This is a federation, not a merger or monorepo.

Rejected alternatives:

- **Make one Abbey repository own everything.** This would strand mature ABI,
  WDBX, runtime, Discord, and Swift boundaries and make one embodiment the
  accidental center.
- **Move every language and platform into a monorepo.** Atomic refactors would
  improve, but release, CI, and toolchain coupling across Rust, Swift, macOS,
  server, Discord, model runtimes, and experimental compute would dominate.
- **Permit each adapter to carry its own cognition and memory forever.** This
  is initially convenient and permanently creates incompatible authority,
  persona, and evidence behavior.

A small language-neutral **Abbey contracts package** is a Program 1 deliverable.
It may later become a repository. Until created and qualified, it must not be
documented as an existing repository or published package.

## 3. Authority, approval, consent, and safety

Usefulness is never authority. A model may propose an action. Only ABI's typed
authorization kernel and the platform actuator may permit and perform it.

Every consequential request contains:

- authenticated principal and delegation chain;
- requested capability ID and version;
- guild, resource, and subject scope;
- validated parameters and data classes;
- current platform permission and hierarchy facts;
- Guild Constitution, capability, and safety-policy versions;
- approval and consent references where applicable;
- deadline, cancellation, idempotency key, rate class, and correlation ID.

ABI returns one of:

- **Allow:** a current grant permits the exact action and current facts satisfy
  all preconditions.
- **Approval required:** Abbey renders a bounded preview, expected effects,
  uncertainty, risk, expiration, and rollback path.
- **Deny or pause:** Abbey identifies the failed invariant without exposing
  protected data.

Authorization is deterministic and deny-by-default. A capability grant names
the action family, issuer, recipient, scope, issue and expiry conditions, risk
class, confirmation policy, revocation state, and policy version. Platform
permissions are necessary facts, not sufficient authority for every Abbey
operation. A Guild Constitution may narrow platform authority and may never
widen it beyond the platform's rules.

Consequential interaction has three visible stages:

1. **Recommend:** explain what could be done.
2. **Propose:** provide an inspectable action plan and predicted effects.
3. **Execute:** act only under a current grant and validated preconditions.

The safety path is independently authoritative. It may refuse, pause, revoke,
cancel, or force a safe state without consulting a model. No planner, router,
skill, model, or learned policy may disable it or modify it online.

### Voice consent

A voice media epoch exists only while every current human participant has
explicitly consented and an authorized manager is entitled to control the
session.

- Consent binds one immutable current-participant set and one media epoch.
- Any new, unidentified, or unattested participant closes media immediately.
- Closing an epoch cancels in-flight STT, reasoning, synthesis, provider work,
  and playback.
- Resume requires a new epoch, fresh current-participant consent, and manager
  authorization.
- Barge-in cancels active playback and stale downstream work; it does not by
  itself withdraw consent.
- Raw audio, transcripts, prompts, responses, and participant identities do not
  enter the verification report or durable WDBX episode.

The Rust verifier present in the current dirty `abbey-bot` worktree is a narrow
source-level example of these rules. Its commands and report are owner/admin-only
and ephemeral. It is not a landed-artifact claim and is not evidence that the
general authorization kernel exists.

## 4. Governed multiscale cognitive cycle

One Abbey turn follows this conceptual sequence:

1. Normalize the platform event and authenticate observable facts.
2. Apply privacy, scope, authorization, and risk preflight.
3. Estimate fast, adaptive, regime, predictive, and resource state.
4. Use SEA to retrieve the smallest sufficient WDBX evidence set.
5. Let the adaptive arbiter choose an execution level.
6. Select persona, model, capability, skill, tool, or human escalation.
7. Produce a response or typed proposal.
8. Reauthorize immediately before any side effect.
9. Execute through the platform actuator and verify postconditions.
10. Measure utility, safety, calibration, cost, cancellation, and rollback.
11. Pass the outcome through a selective WDBX write gate.

State is separated by lifetime and authority:

| State | Examples | Update rule |
| --- | --- | --- |
| Fast | Current turn, VAD, cancellation, immediate confidence | Continuous, bounded, expires quickly |
| Adaptive | Explicit preferences, habituation, local reward history | Guild-local, reversible, opt-in |
| Regime | Normal, onboarding, incident, event, emergency restriction | Changes only after persistent evidence |
| Predictive | Expected outcome, uncertainty, surprise, disagreement | Calibrated against observed outcomes |
| Resource | Latency, queue, cost, rate, privacy, and guild budgets | Measured and enforced, not inferred by a model |

The arbiter chooses among:

- a deterministic or bounded fast path;
- retrieval-conditioned generation;
- a planner or tool workflow that yields an inspectable proposal;
- human escalation, refusal, or a safe pause.

Escalation considers uncertainty, surprise, risk, policy conflict, fast-path
confidence, and expected deliberation cost. It never means automatically asking
the largest model.

Execution domains remain distinct:

- **Bounded real-time:** media gates, cancellation, safety interlocks, rate
  enforcement.
- **Soft real-time:** routing, retrieval, ordinary generation, and policy
  selection.
- **Best effort:** planning, simulation, offline evaluation, and specialist
  training.

The existing Rust Discord pipeline implements a smaller deterministic flow:
triage, intent, state, per-guild speech policy, cooldown, persona, response, and
delayed reward. Regime inference, predictive calibration, the general arbiter,
and the complete measurement plane are Proposed.

## 5. WDBX episodic and evidence contract

A canonical episode must answer:

1. What operating context existed?
2. What did Abbey predict?
3. What action was proposed or performed?
4. What outcome followed?
5. Why was the decision considered authorized and trustworthy?

The normative episode envelope includes:

- UUID, schema version, pseudonymous scope, acquisition and receipt time;
- regime, bounded state summaries, and resource state;
- decision route, prediction, uncertainty, risk, and novelty;
- capability, model, policy, calibration, approval, and consent versions;
- redacted action sequence and parameter classes;
- observed outcome, utility, safety effect, cost, and rollback state;
- source and evidence references with explicit evidence dimensions;
- contradiction, supersession, revocation, resolution, and quarantine edges;
- retention class, hold state, deletion-key reference, and provenance parents;
- canonical encoding, digest, signer identity, and optional signature.

### Non-negotiable WDBX invariants

**Integrity is not truth.** Cryptography supports origin, object identity, and
tamper-evidence claims. It does not establish factual correctness, calibration,
honesty, applicability, or safety.

**Similarity is not applicability.** Retrieval must consider semantic
relevance, regime compatibility, outcome utility, provenance, policy/model/schema
versions, staleness, reuse risk, contradiction, and current constraints.

**Trust is multidimensional.** Evidence dimensions remain individually
inspectable. They are not collapsed into one unexplained scalar.

**Indexes are not canonical.** Embeddings, HNSW structures, summaries, and
search indexes are disposable, rebuildable projections.

**No cross-guild recall.** Guild isolation is the correctness boundary;
guild-plus-user isolation is the member privacy boundary.

**No dual canonical writers.** Migrations shadow-read, replay, compare, cut over
one writer, and retain rollback evidence.

### Retention and deletion

Every proposed write receives one class:

- **Ephemeral:** never written.
- **Session:** retained only for the bounded session.
- **Operational:** retained under a short explicit TTL.
- **Durable:** requires defined utility, authority, privacy basis, and policy.
- **Mandatory incident:** stores the smallest safety, security, audit, or legal
  evidence required.

Correction never silently overwrites history. Supersession, contradiction,
revocation, quarantine, and resolution are explicit edges. Deletion removes
payload keys and every derived projection. A content-free tombstone prevents
accidental resurrection. A mandatory hold retains only what the governing
policy and applicable obligation require.

The measured gap between current WDBX and this contract is recorded in
`2026-08-22-wdbx-conformance-gap-analysis.md`. The current substrate has
strong structural machinery and major evidence-semantic gaps. That document is
an observation, not a defect claim against earlier scope.

The Rust Discord bot's JSON facts remain canonical until WDBX migration parity,
replay, recovery, deletion, and rollback pass. Its WDBX v1 rows remain a
semantic projection during that interval.

## 6. Capability packages and API learning

Abbey learns an application through an explicit, versioned contract. Accepted
sources include:

- OpenAPI and JSON Schema;
- MCP and ABI tool definitions;
- Discord application-command schemas;
- reviewed human-authored capability packages;
- unstructured documentation only as input to a candidate contract that still
  requires validation.

A Capability Package contains:

- stable ID and semantic version;
- input, output, error, streaming, and cancellation schemas;
- required platform permissions and resource scope;
- side-effect, reversibility, and risk classifications;
- sensitive-data classes, residency, and retention restrictions;
- preconditions, invariants, deadlines, rate limits, and budgets;
- confirmation, delegation, expiry, revocation, and approval policy;
- idempotency, compensation, rollback, and postcondition semantics;
- adapter bindings and compatible API versions;
- redacted evidence and outcome-receipt requirements;
- deterministic, adversarial, and failure fixtures;
- promotion thresholds and expiration criteria.

A candidate moves through:

1. static schema and policy validation;
2. deterministic replay;
3. sandbox execution;
4. proposal-only shadow use;
5. bounded canary use;
6. owner/admin-approved promotion;
7. monitoring, drift detection, revocation, and rollback.

Abbey may improve selection, parameter suggestions, workflow composition, and
presentation. She may not promote her own package, change its risk class, invent
an undocumented endpoint, guess a production schema, or convert successful use
into new authority.

Schema drift, permission mismatch, unsafe output, calibration regression, or
missing evidence disables the affected version and preserves the last approved
version.

## 7. Discord command and guild-application model

Discord receives two command layers.

### Stable constitutional surface

The target surface is small and stable:

- `/abbey ask`: explanation and advice;
- `/abbey audit`: read-only guild intelligence;
- `/abbey plan`: inspectable desired-state workflow;
- `/abbey do`: capability-bound execution;
- `/abbey status`: policy, evidence, health, and degradation;
- `/abbey capabilities`: installed and eligible capabilities;
- `/abbey approve`, `revoke`, and `undo`: operator control.

Existing product-specific commands remain compatible while this surface is
designed and migrated.

### Per-guild Skill Views

A Skill View exposes an approved workflow using names, defaults, localization,
and options appropriate to one guild.

Abbey may propose a Skill View after observing an explicit recurring need. A
deterministic manifest compiler checks:

- reserved and conflicting names;
- argument and output schemas;
- capability and adapter availability;
- platform command-count and size limits;
- required permissions and default visibility;
- localization and accessibility;
- manifest version and rollback compatibility.

An owner or administrator reviews the exact guild-scoped diff before
registration. Registration records a manifest hash and redacted receipt.
Uninstall and rollback restore the previous approved manifest.

Dynamic command registration is not an LLM side effect. The model proposes;
the manifest compiler validates; ABI authorizes; the Discord adapter registers.

## 8. Guild world model and adaptive organization

Each guild receives an isolated, versioned digital twin with five views:

- **Structure graph:** categories, channels, roles, integrations, and command
  manifests.
- **Authority graph:** ownership, hierarchy, effective permissions,
  delegations, grants, and approval rules.
- **Workflow graph:** commands, recurring processes, handoffs, automations, and
  unresolved operations.
- **Goal model:** explicit outcomes, priorities, constraints, archetype, and
  prohibited behavior.
- **Health model:** aggregate activity, workflow failures, permission drift,
  unused structure, unresolved incidents, budget pressure, and contradictions.

Every assertion records source, observation time, confidence basis, staleness
policy, contradiction state, privacy class, and schema version. Platform facts,
Abbey inferences, and human-approved goals are distinct types.

Passive observation excludes message content unless a separate explicit policy
authorizes its use. Metadata is minimized and aggregate measures are preferred.

Guild adaptation has four independent lanes:

1. declarative preferences controlled by owners and administrators;
2. provenance-bearing updates from current platform facts;
3. a tightly bounded low-risk policy for `stay`, `reply`, or `react`;
4. structured outcome learning used for offline capability evaluation.

No reinforcement learner directly controls roles, channels, permissions,
moderation, integrations, or command registration.

Learning and unsolicited action are independently opt-in and default-off.
`ABBEY_QUIET` is the higher global override. Speech, observation, planning,
external API calls, command installation, and structural changes have separate
guild budgets.

This target intentionally identifies a current-source mismatch:
`abbey-bot` makes unsolicited acting default-off, but
`GuildSettings.learning_enabled` currently defaults to `true`. The
implementation program must migrate to default-off adaptive learning without
silently rewriting an existing guild's explicit choice.

### Server change workflow

Discord does not provide a transaction spanning multiple structural calls.
Abbey therefore uses:

1. observe current state;
2. diagnose facts and inferences;
3. generate alternatives;
4. simulate permissions and dependencies;
5. render an exact proposed change set;
6. obtain scoped approval;
7. stage one bounded step;
8. revalidate current state and postconditions;
9. continue, compensate, or stop safely;
10. issue a redacted receipt and update the desired-state plan.

## 9. Persona and model routing

ABI routes each task using:

- required modality, output schema, and tool behavior;
- persona contract and interaction context;
- capability grant and maximum permitted consequence;
- privacy class, residency, and retention policy;
- model qualification evidence, health, and calibration;
- latency, cost, energy, queue, and guild-budget constraints;
- known failure modes and revocation state.

A qualified model manifest binds evidence to the exact:

- model and weight identity;
- provider and adapter;
- Abbey/ABI binary;
- operating system and execution mode;
- fixture, schema, policy, and prompt-contract versions.

It declares modalities, context, structured output, tool behavior,
cancellation, measured latency, reliability, calibration, resources, data
handling, consequence ceiling, expiry, and known failures. A provider's
self-description is not qualification evidence.

The router may select:

- a deterministic implementation with no model;
- a qualified local model;
- an explicitly permitted cloud model;
- bounded specialists for decomposed subtasks;
- human escalation;
- no route and an honest refusal.

Local execution is preferred for private material when a qualified local route
exists. Cloud fallback never occurs merely because a credential or endpoint is
available.

Verification scales with consequence:

- **Low:** one qualified route plus deterministic schema and policy checks.
- **Medium:** independent factual, permission, or safety validation.
- **High:** simulation or independent critic, ABI review, and human approval.

Specialists return structured claims, evidence references, uncertainty, and
failure state. Abbey owns the final human-facing synthesis. ABI preserves
disagreement rather than averaging it into false confidence.

Fallback must never cross a privacy boundary, grant tools to a weaker route,
lower the evidence requirement, or misrepresent degradation as full
capability.

## 10. Abbey API and deployment topology

The Abbey API is the stable boundary between adapters, the product runtime, ABI,
and WDBX. Its language-neutral contract package contains:

- principals, scopes, grants, approvals, and consent envelopes;
- capability packages and command manifests;
- cognition requests and structured proposals;
- WDBX episodes, evidence dimensions, and claim records;
- outcomes, receipts, errors, cancellation, and version negotiation;
- cross-language conformance fixtures.

Requests include authenticated principal, guild and resource scope, session,
capability version, constraints, privacy class, consent epoch when relevant,
deadline, cancellation, idempotency key, and correlation ID.

Responses keep four things distinct:

- user-facing content;
- authorization or approval decision;
- execution or degradation result;
- redacted outcome and evidence receipt.

Typed errors include:

- authorization denied;
- approval required or expired;
- unsupported or revoked capability;
- stale or incompatible schema;
- provider unavailable or unqualified;
- memory unavailable, corrupt, or migration-blocked;
- cancellation or consent-epoch closure;
- resource exhaustion, deadline, or rate limit;
- platform precondition or postcondition failure;
- rollback complete, partial, or failed.

### Local-first topology

The default topology is:

```text
Discord / macOS / CLI adapter
        ↓ authenticated Unix socket or owner-scoped loopback
Abbey product runtime host
        ↓ versioned Abbey API
ABI authorization and cognition + canonical WDBX
```

Non-loopback operation is a separate deployment profile requiring TLS, scoped
credentials, explicit origin policy, request and stream limits, rotation,
incident procedures, and qualification evidence.

### Real-time media boundary

Discord UDP, DAVE, VAD, consent epochs, and playback cancellation remain in
`abbey-bot`. Raw audio never crosses the Abbey API or enters WDBX. Ephemeral
local STT text may enter a consent-bound cognition request. Content-free
completion and cancellation events return to the verifier. ABI or provider
failure must not weaken the media gate.

### Degraded operation

- If ABI authorization is unavailable, consequential execution is denied.
- If WDBX is unavailable, only explicitly allowed stateless tasks proceed and
  the missing memory state is disclosed.
- If no qualified model route exists, Abbey uses a deterministic path or says
  no safe route exists.
- If Discord state changes during a plan, the adapter revalidates and stops
  rather than applying a stale operation.
- If a rollback is incomplete, the receipt identifies completed, reverted, and
  unresolved steps without exposing private content.

## 11. Evidence ladder and claim ledger

Evidence at one level permits only that level's claim.

| Level | Required evidence | Permitted conclusion |
| --- | --- | --- |
| C0 | Contract, invariants, risks, falsification criteria | Specified |
| C1 | Unit, property, privacy, schema, and failure-path tests | Source conforms under test |
| C2 | Deterministic replay with equivalent decisions and cancellations | Replay-qualified |
| C3 | Offline baseline, ablation, calibration, and adversarial evaluation | Adds measured offline value |
| C4 | Proposal-only shadow operation | Predicts acceptably in the target environment |
| C5 | Bounded canary with fixed scope, budget, monitoring, and rollback | Works under restricted live authority |
| C6 | Authorized operator witnesses the exact end-to-end outcome | Live-qualified for that environment and version |
| C7 | Repeated operation establishes reliability and drift bounds | Sustained operational evidence |

No level auto-promotes the next. A claim record includes:

- claim ID, exact scope, capability and version;
- proposed/current/partial/failed/revoked/superseded/expired status;
- binary, model, adapter, platform, policy, schema, and fixture identities;
- preregistered thresholds and divergent predictions;
- evidence artifacts, direct observations, missing evidence, and contradictions;
- reviewer, date, expiry, rollback condition, and permitted conclusion.

Evaluation freezes datasets, preprocessing, exclusions, and thresholds before
results are inspected. It compares against deterministic, current-system,
no-learning, and other relevant baselines. Ablations identify which component
adds value. Null, negative, safety, and rollback outcomes remain publishable.

The minimum scorecard includes:

- task success and operator value;
- authorization false-allows and false-denies;
- unsafe-action and incident rate;
- cancellation and rollback success;
- uncertainty calibration, surprise, disagreement, and drift;
- latency, cost, resources, availability, privacy exposure, and evidence
  completeness.

The component seeking promotion cannot be the sole judge of success. Safety and
high-consequence capability promotion require independent review and human
approval.

The live-voice evidence layers remain explicitly separate:

- targeted and strict gates are C1 source evidence;
- installed hash, process, and listener identity are artifact evidence;
- joined or muted presence is not capture;
- an offline WAV is not Discord transport;
- current unanimous consent, audible reply, barge-in, participant pause/resume,
  final leave, and a redacted `8/8` report are C6 live-session evidence.

## 12. Testing and acceptance strategy

Every program defines tests at its own boundaries.

### Contract tests

- Rust and Swift decode and encode the same golden envelopes.
- Unknown additive fields round-trip or fail according to the compatibility
  policy.
- Canonical object digests match across languages.
- Invalid, duplicate, stale, oversized, and contradictory envelopes fail
  deterministically.

### Authorization and capability tests

- Owner, administrator, manager, member, bot, and revoked/delegated principal
  matrices.
- Discord hierarchy and permission changes between proposal and execution.
- Expired grant, approval replay, capability version drift, and revoked package.
- Idempotent retry, partial effect, compensation, rollback, and postcondition
  failure.
- A model-selected tool with no grant never reaches an actuator.

### Privacy tests

- Raw audio, transcripts, prompts, message content, credentials, and identifiers
  never enter redacted reports or diagnostic logs.
- Cross-guild and cross-DM recall remain impossible.
- Retention expiry removes payload and all projections.
- Cryptographic erasure leaves only the permitted content-free tombstone.
- Passive guild intelligence works without message content.

### Cognitive and retrieval tests

- Empty, stale, adversarial, contradictory, quarantined, revoked, and
  wrong-regime retrieval.
- Evidence dimensions remain inspectable and no opaque score hides a veto.
- Arbiter choices obey risk, uncertainty, cost, and safety constraints.
- No-model, local, cloud, specialist, and unavailable-route behavior.
- Replay equivalence under pinned model, policy, schema, and seed.

### Guild and Discord tests

- Default-off learning and default-off unsolicited action.
- `ABBEY_QUIET` wins over guild settings.
- Separate budgets for speech, observation, planning, APIs, commands, and
  changes.
- Manifest conflicts, Discord limits, localization, rollback, and previous
  manifest restoration.
- Multi-step structural plans revalidate between calls and stop safely.

### Voice tests

- Owner/admin verifier authorization and ephemeral responses.
- Consent epoch change, decoded receive, STT completion, natural playback end,
  actual barge-in cancellation, participant pause/resume, and final leave.
- Verification state retains counters only and disables conversation commits.
- The concise report exposes run/status/mode, observed-check count,
  authorization result, aggregate participant counts, consent-epoch change,
  decoded/STT/playback/barge-in milestones, participant pause/resume, final
  leave, and the retention/manual-witness notice. It exposes no audio, IDs,
  transcript, response, message content, or raw timestamp.
- Live C6 requires a current human witness and remains separate from local gates.

### Failure and recovery tests

- Provider, ABI, WDBX, Discord, and local daemon outage.
- Cancellation races, stale events, restarts, corruption, interrupted migration,
  and replay.
- No silent fresh start after durable-state corruption.
- Every partial outcome produces an honest, bounded receipt.

## 13. Delivery programs

This platform is too large for one implementation plan.

### Program 0: Reconcile the current live-voice verifier

Preserve and land the existing privacy-safe owner/admin verifier in
`abbey-bot`. Re-run targeted tests and the current full strict gate. Keep the
participant-consented Discord C6 session separate.

### Program 1: Abbey contracts

Publish principals, scopes, capabilities, policies, consent, events, episodes,
receipts, errors, claims, and compatibility rules with cross-language fixtures.
No runtime behavior changes.

### Program 2: ABI authorization and capability kernel

Implement deny-by-default authorization, capability compilation, approval,
revocation, policy versioning, and redacted receipts against recording adapters.
No production Discord mutation authority.

### Program 3: Read-only Discord guild intelligence

First new product vertical slice: owner/admin-only `/abbey audit`,
`/abbey plan`, and `/abbey status` build a metadata-only guild twin,
explain permission and structural risks, compare alternatives, and emit a
reversible desired-state plan. No message surveillance, structural writes, or
dynamic registration.

### Program 4: Canonical WDBX episodes and claims

Implement the selective write gate, retention classes, correction and deletion,
evidence-aware retrieval, claim ledger, derived-index rebuild, and JSON-to-WDBX
shadow parity. Cut over only after replay, recovery, privacy, deletion, and
rollback evidence.

### Program 5: Approved reversible guild execution

Add preview, approval, staged execution, verification, compensation, and
per-guild Skill View manifests for a small reversible capability set.
Destructive moderation remains proposal-only until separately specified.

### Program 6: Model registry and adaptive arbiter

Add qualified model manifests, privacy-aware routing, regime inference,
calibrated escalation, structured outcome learning, and specialist promotion.
Keep the existing DQN confined to low-risk speech.

### Program 7: Application federation and production profiles

Connect ABI, WDBX, `abbey`, `abbey-bot`, and `AbbeyBot` through the
approved contracts. Qualify local Mac, server, and future hosted profiles
separately with compatibility matrices, drift monitors, credential procedures,
and sustained evidence.

## 14. Normative decision register

This register consolidates the architecture-discovery questions into decisions.
Program specs may add mechanism but may not silently reverse these answers.

### Product and identity

1. Abbey is the product and relationship, not a single binary or provider.
2. Aviva is the direct expert register, not an unrestricted authority mode.
3. ABI is the silent cognitive, authorization, and validation kernel.
4. WDBX is memory and evidence, not merely vector search.
5. SEA is a bounded evidence-selection responsibility.
6. Abbey owns the final integrated response.
7. Personas remain stable when models change.
8. Adapters may not invent local persona definitions.

### Authority and safety

9. Models propose; typed capability runtimes authorize.
10. Permission facts and capability grants are distinct.
11. Authorization is deny-by-default and current-state validated.
12. Owner decisions outrank administrators; administrators outrank learned
    preferences; platform and safety constraints outrank all.
13. High-consequence execution requires explicit human approval.
14. Safety may pause or deny without model consultation.
15. Safety is never learned online.
16. Repeated approval does not become standing authority.
17. Revocation takes effect before new work begins.
18. Consequential actions need postcondition checks and rollback or safe stop.

### Privacy and memory

19. Raw audio is ephemeral.
20. Message content is not passive guild telemetry.
21. Prompts and generated responses are not durable operational evidence by
    default.
22. Redacted reports contain fixed categories and bounded metadata.
23. Guilds never share private memory.
24. DM users receive separate scopes.
25. Embeddings and indexes are disposable projections.
26. Integrity never implies truth.
27. Similarity never implies applicability.
28. Retention is explicit and classed.
29. Deletion removes payload and projections while retaining only a permitted
    tombstone.
30. Contradiction, supersession, revocation, and quarantine remain visible.

### Guild learning

31. Adaptive learning is opt-in and default-off.
32. Unsolicited action is independently opt-in and default-off.
33. `ABBEY_QUIET` is the higher global override.
34. Each guild has isolated policies and budgets.
35. Speech learning never controls administration.
36. Reinforcement learning never directly changes roles, permissions, channels,
    moderation, integrations, or commands.
37. Explicit preferences are not inferred from engagement.
38. World-model facts carry provenance and staleness.
39. Inferences remain distinguishable from observed facts.
40. Cross-guild learning requires aggregate privacy proof at a higher claim
    level.

### APIs, tools, and commands

41. APIs enter through versioned schemas.
42. Documentation may propose a schema but cannot authorize a guessed call.
43. Capabilities state risk, privacy, permission, idempotency, and rollback.
44. Tool availability is explicit at the generation boundary.
45. Capability promotion is staged and separately approved.
46. Schema drift disables the affected version.
47. Core `/abbey` commands remain stable.
48. Guild-specific commands are views over approved capabilities.
49. The model never directly registers a Discord command.
50. Manifest installation is previewed, approved, hashed, and reversible.

### Models and routing

51. Model selection is provider-neutral.
52. Private material prefers a qualified local route.
53. Cloud use requires explicit data-policy permission.
54. A credential's presence is not consent to use a provider.
55. Provider claims require bound qualification evidence.
56. Routing may select no model.
57. Consequence determines verification depth.
58. Specialists return structured evidence and uncertainty.
59. Abbey synthesizes; ABI validates.
60. Fallback never weakens privacy, tools, or evidence requirements.

### Evidence and promotion

61. Source tests, installed artifacts, connector turns, and live voice are
    different evidence layers.
62. Every claim names exact version and environment.
63. Evidence never auto-promotes a higher claim.
64. Baselines and ablations are required before adaptive promotion.
65. Thresholds are set before results are inspected.
66. Negative and rollback evidence remains publishable.
67. A capability cannot be its sole evaluator.
68. Failed, revoked, superseded, and expired are first-class states.
69. Attractive demos do not establish sustained reliability.
70. Live voice requires current participant consent and a human witness.

### Federation and operation

71. ABI, WDBX, `abbey`, `abbey-bot`, and `AbbeyBot` remain federated.
72. The Abbey contracts package is language-neutral.
73. One canonical owner writes each domain.
74. Default transport is authenticated local IPC.
75. Non-loopback operation is a separately qualified profile.
76. Raw Discord audio stays in the Discord adapter.
77. Migrations never run dual canonical writers.
78. API failure degrades visibly and cannot weaken safety.
79. Cross-repository compatibility is proven by shared fixtures.
80. Each delivery program gets its own design, plan, gate, canary, and rollback.
81. Live-voice verification control and reports are owner/admin-only and
    ephemeral.
82. A live-voice report retains fixed counters, epochs, and aggregate counts,
    never audio, identity, transcript, response, or message content.
83. A strict local gate and a consented Discord acceptance session remain
    separate claims even when both pass.

## 15. Amendment and review

This constitution changes only through explicit approval from Donald recorded
with a date and evidence reference. A program that needs to contradict it must
amend it first. Silence is not amendment.

Before implementation planning:

1. Donald reviews this written revision.
2. Requested changes are incorporated and self-reviewed.
3. The revision is explicitly approved.
4. Program 0 and Program 1 receive separate implementation plans.

No implementation program is authorized merely because this document exists.
