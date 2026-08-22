# Abbey System Constitution

Status: **ratified boundary, proposed mechanisms.** Approved by Donald J. Filimon
on 2026-08-22. Program 1 of seven.

This document fixes ownership and authority across the Abbey system. It is
deliberately short on mechanism and absolute on boundaries, because its purpose
is to stop four codebases from evolving four competing answers to the same
questions. Mechanism belongs to Programs 2 through 7, each of which gets its own
spec and may not contradict this one.

## 0. Status vocabulary

Every claim in this system carries one of three labels, and they never merge:

- **Observation** — something a tool, test, gate, or measurement directly produced.
- **Inference** — a conclusion that holds only if stated additional assumptions hold.
- **Proposed criterion** — a design threshold or engineering choice introduced by a
  document, not a measurement.

This vocabulary is taken from the Orch-OR falsification framework and applies to
engineering claims for the same reason it applies to scientific ones: the common
failure is promoting evidence from one scale to a claim at another.

Two source documents inform this constitution and are **not** authoritative over
it:

- `CSAPS_WDBX_Revised_2026.pdf` (revision 2.0, 2026-08-22) — proposed architecture.
  Its own status box states the integrated system has not been empirically
  validated and that quantitative thresholds are acceptance targets, not results.
- `Donald_Filimon_Multiscale_OrchOR_Falsification_Framework.pdf` (2026-08-22) —
  methodology. It reports no original experimental data.

Where this constitution cites them, it cites them as **proposed criteria**.

## 1. Components and ownership

Exactly one layer is authoritative for each concern. Any component may *read* a
concern it does not own; none may redefine it.

| Concern | Authoritative owner | Everyone else |
| --- | --- | --- |
| Product identity, persona, human relationship | **Abbey** | May render it, may not redefine it |
| Cognitive and governance runtime | **ABI** | May call it, may not fork its decisions |
| Provenance-aware memory and evidence substrate | **WDBX** | May project from it, may not invent a second semantics |
| Stable inter-component contract | **Abbey API** | May extend under versioning, may not bypass |
| Platform embodiment and operator experience | **Adapters** (Rust `abbey-bot`, Swift `AbbeyBot`, `abbey` runtime host) | Own their platform's safety boundary only |

The physical repositories as of 2026-08-22:

- `dev/active/abi` — ABI runtime. 19 crates (not the 12 its own docs claimed).
- `dev/active/abbey` — execution and control host: provider backends, daemon
  lifecycle, tool approval and audit, local MCP, memory, claim registry.
- `dev/active/abbey-bot` — Rust Discord adapter. serenity/poise/songbird, pure
  decision modules behind a thin shell, per-guild learning, consent-gated voice.
- `dev/active/AbbeyBot` — Swift companion application plus headless server:
  dashboard, API, sync, confirmation gate, local inference, operator UI.

This is a **federation**, not a merger and not a monorepo. The rejected
alternatives and why:

- *One Abbey repository owning everything, ABI reduced to a library.* Would
  strand ABI's WDBX, gateway, persona, and evidence work, and would make the
  Discord embodiment the accidental center of a broader platform.
- *A monorepo containing every application and runtime.* Would simplify atomic
  refactors at the cost of a release, platform, and toolchain surface spanning
  stable Rust, nightly Rust, Swift, macOS-only features, Linux services, Discord,
  web, WDBX, and experimental compute. A shared protocol and schema repository is
  reconsiderable later; a physical merger is not the first move.

## 2. Authority boundaries

**A1. Persona authority.** Abbey owns persona definition. ABI owns persona
*routing* and *modulation* (`abi-ai`). An adapter may select among defined
personas and may render them; it may not define a new one or alter a definition
locally.

**A2. Memory authority.** WDBX owns durable episodic semantics: what an episode
is, what makes it trusted, how it is superseded, contradicted, quarantined, or
deleted. An adapter may hold a *projection* of WDBX for local operation. A
projection is read-shaped and lossy by design, must declare which fields it
drops, and may never become a second definition of what an episode means.

**A3. Authorization authority.** Authorization is never a generative decision. A
model may *propose* an action. Only the typed actuator and capability runtime
(Program 2) may authorize one, and it does so by schema validation, grant
checking, rate and deadline limits, approval validation, postcondition
verification, and rollback. A model-selected tool is not an authorized tool.

**A4. Safety authority.** The policy and safety path is separately authoritative.
It may refuse, pause, revoke, or force a safe state without consulting any model.
No planner, router, or language model may disable it. This mirrors CSAPS R5 and
R9 (proposed criteria) and is adopted here as a binding invariant.

**A5. Evidence authority.** No component may promote a capability's claim level
without evidence at that level. The ladder is normative:

```
L0  Capability described
L1  Schema parsed and statically validated
L2  Deterministic local qualification passes
L3  Sandbox execution succeeds
L4  Shadow execution predicts the correct result
L5  Restricted canary succeeds under real authorization
L6  Postconditions and rollback are verified
L7  Repeated guild-local benefit is demonstrated
L8  Cross-guild aggregate benefit is demonstrated without privacy leakage
```

Evidence at level *n* may justify *attempting* level *n+1*. It never
auto-promotes a claim to level *n+1*. Concretely, and each of these is a real
failure mode already present in this system's history: a successful API call does
not prove a good outcome; a positive user reaction does not prove a good guild
policy; a model-selected tool does not prove authorization; semantic similarity
does not prove an old episode applies; a local test does not prove deployment;
and an attractive demonstration does not prove the adaptive architecture beats a
simpler baseline.

**A6. Documentation authority.** A capability is documented as **Current** only at
the evidence level that supports it, with the level named. Everything else is
**Proposed**. A stub is never Current, regardless of who asks.

## 3. Non-negotiable invariants

**I1. Integrity is not truth.** A valid signature supports claims about origin,
integrity, key identity, and non-modification. It cannot establish that a sensor
was calibrated, a model was correct, an operator was honest, or a statement was
semantically true. Cryptographically invalid records are rejected; validly signed
but semantically uncertain records remain inspectable at reduced retrieval weight
with quarantine status or a contradiction edge.

**I2. Similarity is not applicability.** Retrieval that ranks on semantic distance
alone is insufficient. An episode is applicable only when regime, policy version,
model version, permission scope, provenance confidence, and staleness are all
compatible. Retrieving a right-looking answer for the wrong reason is the
substrate's primary failure mode.

**I3. Trust is multidimensional.** Evidence dimensions are exposed separately and
never collapsed into one opaque score.

**I4. Deletion is bounded, not absent.** Immutability of a commitment does not
require indefinite retention of every payload. Retention classes, legal and
operational holds, cryptographic erasure with a minimal tombstone, redacted
derivative blocks that link to but never overwrite the original, and auditable
garbage collection are all required. Provenance must make deletion attributable
without turning privacy-sensitive content into an undeletable permanent record.

**I5. No silent redefinition.** An adapter that needs different semantics raises
it to the owning layer. It does not implement a local variant. Where a
transcription already exists for a deliberate, tested reason, it is pinned by
conformance tests and declared as a projection under A2.

**I6. Honest residuals.** Every component reports what is Current and what is
Proposed. Source and gates override prose.

## 4. Tenant model

The tenant is the **guild** (or equivalently, the platform-scoped organization).

- Guild isolation is the correctness boundary: no cross-guild read by default.
- Guild-plus-user isolation is the privacy boundary between members of one guild.
- Cross-guild learning is permitted only in aggregate, only at L8, and only with
  a demonstrated absence of privacy leakage. Absence of leakage is an
  **observation** requiring a test, never an inference from design intent.

The Rust adapter already implements guild and guild-plus-user scoping in its
memory layer. That is the reference behavior, not an adapter-local invention, and
it is hereby lifted to a constitutional requirement.

## 5. What this constitution does not decide

Deliberately deferred, each to its own spec:

- Program 2: typed capabilities, API-learning packages, credential isolation,
  guild grants, approval levels, actuator validation, postconditions, audit,
  rollback.
- Program 3: the canonical WDBX episodic contract, including block schema,
  canonicalization, signing, evidence dimensions, and retention semantics.
- Program 4: guild world model, regime inference, budgets, and the escalation
  arbiter across fast, retrieval-conditioned, and deliberative modes.
- Program 5: the Discord organization vertical slice.
- Program 6: the Abbey API surface and versioning policy.
- Program 7: deterministic replay, shadow evaluation, baselines, ablations,
  canaries, rollback, and falsification criteria.

Naming note: CSAPS section 6.7 proposes service names (`StateService`,
`PredictionService`, `ArbiterService`, `MemoryService`, `PlannerService`,
`SafetyService`). `abi-sea`'s eight-signal scorer plus `abi-ai`'s router and
modulator resemble a partial arbiter, and `abi-wdbx-gateway`'s gRPC surface
resembles a partial `MemoryService`. Those resemblances are recorded as analysis
only. Renaming is Program 4 and Program 6 work and is explicitly not performed
during substrate extraction, because a rename during a move is an accidental
architectural commitment.

## 6. Amendment

This document is amended by explicit approval from Donald, recorded in
`tasks/goals.md` with a date. A program spec that needs to contradict it amends
it first. Silence is not amendment.
