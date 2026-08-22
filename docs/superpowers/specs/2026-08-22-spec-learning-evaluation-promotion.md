# Program 6 and cross-cutting: Learning, evaluation, and promotion

> **Constitutional mapping.** This file was written against the seven-program
> list Donald gave in conversation. The ratified constitution's section 13
> "Delivery programs" uses a different numbering, and the constitution wins.
> In section 13's terms this document specifies the **learning and promotion half of Program 6**, and is cross-cutting: every program's claims move up the ladder through it.
>
> The filename is therefore name-based rather than numbered, so no numbering is
> asserted. Nothing in section 13 was renumbered: section 15 reserves amendment
> to Donald, and the collision is raised as one request covering the whole set
> rather than five independent ones.

**Reconciled ownership.** Learning and specialist promotion are part of
constitutional Program 6, `model-registry-adaptive-arbiter`; evaluation and
evidence promotion are a cross-cutting discipline for Programs 0 through 7.
This is not Program 8. Program 1, `abbey-contracts`, owns the learning-policy
and promotion-decision wire schemas and synthetic fixtures.


Status: **proposed design specification.** No implementation is authorized by
this document. Dated 2026-08-22.

Scope, in Donald's words: establish deterministic replay, shadow evaluation,
strong baselines, ablations, canaries, rollback, per-guild learning,
privacy-safe cross-guild aggregation, and falsification criteria.

Governing document: `2026-08-22-abbey-system-constitution.md`. Where this spec
and the constitution disagree, the constitution wins and this spec is wrong
until amended under section 15. Measured input:
`2026-08-22-wdbx-conformance-gap-analysis.md`, rows R10, R11, and R12.

Everything below is labeled **Current** (exists, verified by reading the named
source file on 2026-08-22) or **Proposed** (a design target, not a
measurement). Nothing here is an Observation about a system that runs.

---

## 0. Constitutional reconciliation

Three discrepancies existed between this document's assigned working number
and the ratified text. The approved federation reconciliation resolves them by
keeping Programs 0 through 7 unchanged, placing adaptive learning in Program 6,
and making evaluation and evidence promotion cross-cutting.

### 0.1 The program number is already occupied

Constitution section 13 assigns **Program 7 to "Application federation and
production profiles"** and **Program 6 to "Model registry and adaptive
arbiter"**. The reconciled title records that mapping while preserving this
filename as stable historical context.

No Program 8 is created. Program 7 remains application federation and
production profiles.

### 0.2 The content overlaps Program 6

Section 13's Program 6 already claims "structured outcome learning" and
"specialist promotion", and states "Keep the existing DQN confined to low-risk
speech." Per-guild learning therefore sits inside Program 6's declared scope
today.

The reconciled division is:

- **Program 6 retains** the qualified model manifest, privacy-aware routing,
  regime inference, and the adaptive arbiter. It remains the owner of *what
  learns and what routes*.
- **The cross-cutting evaluation discipline owns** the machinery that decides
  whether any of it is allowed to advance: replay, shadow, baselines,
  ablations, accounting, canary, rollback, the promotion rule, and the
  falsification register. It owns *how a claim moves*, for every program, not
  only for Program 6.

Under that division, Program 6 cannot promote its own arbiter, which is what
register #67 requires ("a capability cannot be its sole evaluator").

### 0.3 The evidence-ladder vocabulary in the assignment does not exist

The assignment refers to "A5 (the L0-L8 evidence ladder)" and to cross-guild
learning being permitted "only at L8". The ratified constitution has no section
A5 and no L-prefixed levels. It has **section 11, "Evidence ladder and claim
ledger", with rungs C0 through C7**, eight rungs, top rung C7.

This spec reads the assignment's L0-L8 as section 11's C0-C7 and uses C0-C7
exclusively. **There is no C8.** The consequence is load-bearing and is stated
here rather than buried in section 12 below: cross-guild aggregation is **not a
higher rung on the ladder**. It is an orthogonal gate applied on top of C7,
consisting of a distinct predeclared leakage test whose result is an
Observation. This matches register #40 ("cross-guild learning requires
aggregate privacy proof at a higher claim level") without inventing a rung the
constitution does not define.

### 0.4 A smaller inconsistency, recorded so it does not propagate

The gap analysis header says it is "Input to Program 3, the Canonical WDBX
Episodic Contract." Constitution section 13 assigns "Canonical WDBX episodes
and claims" to **Program 4**, and Program 3 to "Read-only Discord guild
intelligence". One of the two documents has a stale number. This spec assumes
section 13 is correct and that canonical WDBX episodes are Program 4.

---

## 1. Current state, verified by reading source

This section is the measurement this program builds on. Every line was checked
against the file named. Nothing here is inferred from a document.

### 1.1 The per-guild learning loop that already exists

**Current.** `dev/active/abbey-bot/src/brain/` is 3,743 lines across twelve
modules and implements a complete per-guild deep Q-learning loop.

| File | What it is |
| --- | --- |
| `brain/state.rs` | `BotAction::{Stay, Reply, React}` and an 18-dimension deterministic state encoder (`STATE_DIMENSIONS = 18`). The hour of day is injected, not read from a clock. |
| `brain/dqn.rs` | `DqnAgent` with `GAMMA = 0.99`, `EPSILON_INITIAL = 0.1`, `EPSILON_MIN = 0.01`, `EPSILON_DECAY = 0.995`, `BATCH_SIZE = 64`, `TARGET_SYNC_INTERVAL = 100`, `LEARNING_RATE = 0.001`. Output activation is `Linear`, deliberately. |
| `brain/nn.rs` | `NeuralNetwork` plus `Rng(u64)`, a splitmix64 generator seeded per agent. |
| `brain/replay.rs` | `Experience` and `ReplayBuffer`. |
| `brain/reward.rs` | `RewardCollector` with `SETTLEMENT_WINDOW_SECS = 150`, `ATTRIBUTION_TTL_SECS` bound to it, `REPLY_BASELINE = -0.2`, `MAX_POSITIVE_REACTIONS = 3`, `REWARD_CLAMP = 3.0`. Pure, clock injected. |
| `brain/outcome.rs` | `blend(immediate, delayed_sum, delayed_count)`, which returns `immediate` untouched when `delayed_count == 0`. |
| `brain/registry.rs` | `Brain` and `BrainStore` traits, one policy per scoped guild id, `DEFAULT_EVICT_AFTER_SECS = 6 * 3600`. |
| `brain/budget.rs` | Refilling token bucket, one per scoped guild id. |
| `brain/telemetry.rs` | `BrainStats` and `BrainView`, in-memory, `RECENT_REWARDS = 20`. |

`GuildSettings` (`src/guild.rs:72`) carries `learning_enabled`,
`epsilon_override: Option<f32>`, `unsolicited`, `unsolicited_per_hour`, and
`reply_cooldown_seconds`.

### 1.2 Two verified deterministic-replay defects in the snapshot path

**Current, and this is the single most important measured finding in this
document.** `BrainSnapshot` (`src/brain/dqn.rs:39`) carries exactly five
fields: `topology`, `layers`, `epsilon`, `step_count`, and `experiences`
(capped at `SNAPSHOT_EXPERIENCES = 1_000`).

1. **The RNG state is not in the snapshot.** `DqnAgent` holds `rng: Rng` at
   `dqn.rs:103`. `Rng` is a single `u64` of splitmix64 state (`nn.rs:16`). It
   drives the epsilon-greedy branch at `dqn.rs:176` and replay-buffer sampling
   at `dqn.rs:195`. `export_weights` (`dqn.rs:220`) does not write it and
   `import_weights` (`dqn.rs:246`) does not restore it. A restarted agent
   therefore resumes with a different exploration and sampling stream from the
   same weights. The module docstring's claim that "randomness comes from the
   agent's own seeded `Rng`, so every test is reproducible" is true within one
   process lifetime and false across a restart.
2. **The target network's phase is destroyed on import.** `import_weights`
   executes `self.target = self.online.clone()` at `dqn.rs:268`. Against
   `TARGET_SYNC_INTERVAL = 100`, a snapshot taken at step 47 of an interval
   restores as though a hard sync had just occurred. The Bellman target
   `reward + GAMMA * max Q_target(s', a')` is consequently computed against a
   different target network than the one the original run used.

Two smaller behaviors in the same function, recorded because they affect
replay fidelity: the replay buffer is **not cleared** before the snapshot's
experiences are pushed, and experiences whose `state`/`next_state` width does
not match are **silently skipped** with no counter.

Together these are the concrete content of gap-analysis row **R10: "Partial.
Deterministic migration and byte-identical golden fixtures exist; replay of a
recorded stream to equivalent internal trajectories is not implemented."**

### 1.3 The claims machinery this program must extend

**Current.** `dev/active/abbey/src/claims/` is the existing typed claim
registry: `mod.rs` (the `Status` enum, `validate_registry()`, `lookup`,
`by_status`, `CLAIMS_SCHEMA_VERSION: u16 = 1`) and `registry.rs` (767 lines,
the `Claim` struct, the `claim!` macro, and the `CLAIMS` table).

`ClaimEvidence` already separates four things that a weaker design would
collapse: `implementation_refs`, `automated_test_refs`, `local_live`, and
`external_required`. `EvidenceState::{Verified, Required, NotRequired}` already
makes "no proof" typed and distinct from "proof not applicable", which is
exactly the distinction section 11 needs. `validate_registry()` already fails
closed on empty ids, duplicate ids, malformed ids, and empty required text.

This is the right foundation. This program extends it. It does not replace it.

### 1.4 A verified conflict between the registry and the constitution

**Current.** `Status::ALL` (`abbey/src/claims/mod.rs`) is exactly five variants:
`Current`, `Partial`, `Proposed`, `Blocked`, `OutOfScope`.

Constitution section 0 defines seven capability states: **proposed, partial,
current, failed, revoked, superseded, expired.** Register #68 makes "failed,
revoked, superseded, and expired" first-class states.

The registry is therefore missing all four of the states this program most
needs, and carries two (`Blocked`, `OutOfScope`) that the constitution does not
name. A program whose entire purpose includes producing informative negative
results has, today, nowhere to record one. `Failed` does not exist as a value.

### 1.5 Substrate surfaces available and missing

**Current.** `abi-wdbx` now lives at `dev/active/wdbx/crates/abi-wdbx`
alongside `abi-foundation`, `abi-core`, `abi-compute`, and `abi-telemetry`. The
extraction described at the end of the gap analysis has landed in the working
tree; `dev/active/abi/Cargo.toml` points at `../wdbx/crates/*` for all five.

- `abi-wdbx-gateway` exposes `PutVector`, `Search`, `PutKv`, `GetKv`,
  `ResolveConflict`, `Stats`, `MembershipChange`, and **`WatchMutations`**.
- There is **no `ProposeWrite`** and **no `Verify`**. Shadow and canary writes
  therefore have no substrate-level gate and no consumer-callable verification
  today.
- `abi-sea/src/scorer.rs` exposes `SeaSignals` and `SeaWeights` with eight
  named signals (`semantic`, `keyword`, `metadata`, `recency`, `authority`,
  `graph`, `contradiction`, `task_fit`), `DEFAULT_SEA_WEIGHTS`, `sea_score`,
  and `adjust_weights_for_task`. These are individually addressable and are
  therefore usable ablation knobs.
- `abi/tests/golden/` holds the frozen CLI, MCP, and completion fixtures.

---

## 2. Deterministic replay

**Proposed.** A run is replayable when re-executing the recorded input stream
against the recorded configuration reproduces equivalent decisions. Section 11
rung C2 says "deterministic replay with equivalent decisions and cancellations",
and the word is *decisions*, not floats. Section 3 below defines equivalence.

### 2.1 What a run must record

A **run record** is the unit of replay. It is guild-scoped, retention-classed,
and contains three separable parts.

**Part A, the frozen configuration.** Recorded once at run start. Any change
during the run ends the run and starts a new one.

- ABI, WDBX, `abbey`, and `abbey-bot` binary digests, one each.
- Rust toolchain identity, target triple, and host OS build.
- Model identity and weight digest for every route the run may take.
- Policy version, safety-policy hash, schema version, and capability package
  versions and digests.
- The complete DQN hyperparameter block as literal values, not as a reference
  to constants that may move: `GAMMA`, `EPSILON_INITIAL`, `EPSILON_MIN`,
  `EPSILON_DECAY`, `BATCH_SIZE`, `TARGET_SYNC_INTERVAL`, `LEARNING_RATE`,
  `STATE_DIMENSIONS`, output activation, and topology.
- Reward constants as literal values: `SETTLEMENT_WINDOW_SECS`,
  `ATTRIBUTION_TTL_SECS`, `REPLY_BASELINE`, `MAX_POSITIVE_REACTIONS`,
  `REWARD_CLAMP`, and the delayed blend weight.
- `SeaWeights` in effect, and whether `adjust_weights_for_task` was applied.
- `GuildSettings` at run start, including `learning_enabled`,
  `epsilon_override`, `unsolicited`, `unsolicited_per_hour`, and
  `reply_cooldown_seconds`.
- The initial `BrainSnapshot` digest.

**Part B, the initial mutable state.** This is where the current gaps bite.

- The full `BrainSnapshot` as it exists today.
- **The `Rng` u64 state**, which the snapshot does not carry today. Closing
  this is prerequisite work, not optional polish.
- **The target network's parameters as a distinct object**, plus the step count
  modulo `TARGET_SYNC_INTERVAL`, so a mid-interval target is restored as a
  mid-interval target rather than force-synced.
- The `ReplayBuffer` contents in full, with an explicit statement of whether
  the buffer was cleared before loading.
- Budget bucket tokens and last-refill timestamp per scoped guild id.
- `RewardCollector` pending turns, with their open windows.

**Part C, the ordered event stream.** Every input that can change a decision.

- Each normalized platform event, content-classed per section 8 of the
  constitution and stored by reference and class rather than by raw content.
- Every injected clock value. The encoder, the reward collector, and the budget
  bucket all take time as a parameter today, which is what makes this feasible.
- Every retrieval result actually consulted, by episode reference, together
  with the `SeaSignals` computed for it. Recording the query without recording
  what came back makes the run unreplayable the moment the store changes.
- Every provider response, by digest, with the request digest that produced it.
- Every authorization decision from ABI, with the grant and policy versions.
- Every cancellation, consent-epoch change, budget refusal, and cooldown
  refusal.
- For each turn: the state vector, the full Q-vector, the chosen action, and
  whether the choice was the greedy branch or the exploration branch.

Recording the **full Q-vector**, not just the argmax, is what makes section 3's
tie rule checkable.

### 2.2 What replay must not do

A replay run never reaches an actuator, never performs a side effect, never
writes to a canonical WDBX store, and never emits a Discord call. Replay is a
best-effort execution domain per constitution section 4 and is not permitted to
influence the bounded real-time or soft real-time domains.

### 2.3 Retention

Run records are `Operational` class by default with an explicit TTL, and
`Mandatory incident` class when the run is attached to a safety or rollback
event. They are never `Durable` by default, because a run record is dense with
event references and would otherwise become a slow-motion content archive.

---

## 3. Equivalence tolerance

**Proposed.** Equivalence is defined at three layers, checked in order, and a
failure at any layer fails the replay.

**Layer 1, decision equivalence. Exact, zero tolerance.** For every turn, the
replayed `BotAction` must equal the recorded `BotAction`, the replayed
authorization outcome must equal the recorded outcome, and every recorded
cancellation must occur at the same position in the stream. There is no
tolerance here because a decision is discrete.

**Layer 2, the tie rule.** `argmax` in `dqn.rs` resolves ties to the lowest
index, and `BotAction::ALL` is ordered `[Stay, Reply, React]`, so `Stay` wins
every exact tie. A float perturbation of one part in `2^-23` at a tie therefore
flips a decision. Replay must distinguish that case from a real divergence:

- If the recorded Q-vector had a tie or a near-tie within the numeric tolerance
  of Layer 3 at the decision boundary, the turn is classified
  `TieSensitive` and reported separately.
- A replay is **not** declared equivalent merely because all divergences were
  tie-sensitive. The count of tie-sensitive turns is a reported metric with a
  preregistered ceiling. Exceeding the ceiling means the policy is operating on
  the margin and the run is not replay-qualified.

**Layer 3, numeric tolerance on continuous quantities.** Q-values, reward
values, and `SeaSignals` compare within a preregistered relative tolerance,
declared in the run record before replay is executed. The default proposal is
`1e-5` relative for `f32` quantities on an identical target triple, and
**replay across differing target triples is not claimed at all** until measured.
The tolerance is part of the frozen configuration, so it cannot be widened after
a failing replay is seen.

**What equivalence does not establish.** A replay-qualified run has demonstrated
that the recorded trajectory is reproducible. It has demonstrated nothing about
whether the trajectory was good. C2 permits the conclusion "replay-qualified"
and no other.

---

## 4. Shadow evaluation

**Proposed.** Shadow evaluation runs a candidate configuration alongside
production on the same live event stream, computing everything and acting on
nothing.

### 4.1 Construction

The candidate consumes the same normalized events as production, through the
same `brain/state.rs` encoder, and produces a decision that is recorded and
discarded. Concretely, the candidate runs with the actuator disconnected: no
Discord call, no structural change, no message, no reaction.

`WatchMutations` on `abi-wdbx-gateway` is the intended feed for the memory side
of a shadow run. The gap analysis explicitly nominates it: "a streaming
mutation watch that the spec does not require but Program 7 will want for
shadow evaluation." This spec takes that nomination.

### 4.2 The write problem, stated honestly

There is **no `ProposeWrite`** and **no `WriteDecision`** in
`abi-wdbx-gateway` today, and `PutVector`/`PutKv` are unconditional. A shadow
run therefore has no substrate-level mechanism preventing its own writes from
contaminating the store that production reads. Until Program 4 lands the
selective write gate, shadow isolation must be enforced by giving the candidate
a **separate store path** and treating any write from a shadow process to a
canonical path as a gate failure, not as a policy question. This is a weaker
guarantee than a substrate gate and is labeled as such.

### 4.3 What shadow evidence permits

Rung C4: "proposal-only shadow operation" permits the conclusion "predicts
acceptably in the target environment." It does not permit "works", because
nothing was done. In particular, a shadow run cannot measure the reward signal
honestly for actions production did not take: the guild reacted to what
production did, not to what the candidate proposed. Off-policy reward estimates
from a shadow run are **Inference**, must be labeled as such, and require a
stated estimator with stated assumptions. Presenting a shadow reward mean as if
it were an observed outcome is the specific failure this paragraph exists to
prevent.

---

## 5. Strong baselines

**Proposed.** Constitution section 11 already names the required comparison
set: "deterministic, current-system, no-learning, and other relevant baselines."
This section makes them concrete and says why a weak one is worthless.

### 5.1 Why a weak baseline invalidates the result

A comparison against a weak baseline measures the baseline's weakness, not the
candidate's strength. Three specific ways it fails here:

- **A baseline given less compute is not a baseline.** If the adaptive path
  runs a model and the baseline runs a coin flip, the experiment has measured
  that a model beats a coin flip. CSAPS names the negative result the program
  must be able to produce: "parity or inferiority after equivalent compute,
  memory, data, and instrumentation are accounted for." The phrase "after
  equivalent compute is accounted for" is the whole point, and it is why
  section 7 below is not optional.
- **A baseline that is not tuned is not a baseline.** An untuned heuristic
  loses to a tuned learner for reasons that have nothing to do with learning.
  Each baseline gets a declared tuning budget, and the budget is recorded.
- **A baseline evaluated on the learner's own metric is not a baseline.** See
  section 8.

### 5.2 The required baseline set

| Baseline | Construction | What beating it would establish |
| --- | --- | --- |
| **B0, deterministic** | The existing pre-DQN flow: triage, intent, per-guild speech policy, cooldown, persona, response. No policy network. | That the adaptive path adds anything at all. |
| **B1, current system** | Production exactly as configured today, including the DQN with its current hyperparameters and its current `learning_enabled: true` default. | That the change is an improvement over what ships. |
| **B2, no-learning** | The full pipeline with the policy frozen: weights loaded, `learn()` never called, epsilon pinned to `EPSILON_MIN`. | That online adaptation, not merely the network's presence, is doing the work. |
| **B3, no-delayed-channel** | Full pipeline, but reward settles on the immediate heuristic only. | That the delayed outcome channel earns its complexity. |
| **B4, tuned heuristic** | B0 with its cooldown, budget, and persona thresholds tuned under a declared budget against the same training window. | That the learner beats a competent hand-tuned rule, which is the honest bar. |

B3 has a gift in the source that makes it cheap and exact rather than a
reimplementation: `outcome::blend` already "returns the immediate value
untouched when no outcome ever arrived", so B3 is the documented degradation
path of the shipped code rather than a separate branch.

B2 requires the fix from section 1.2 before it is meaningful, because a frozen
policy whose RNG stream is not restored still explores differently after a
restart.

### 5.3 Evaluation hygiene

Per section 11, datasets, preprocessing, exclusions, and thresholds are frozen
before results are inspected. Concretely, this means a written, dated,
digest-stamped preregistration file exists in the repository before the first
comparison is computed, and the comparison tooling refuses to run if the
preregistration digest does not match.

---

## 6. Ablations

**Proposed.** An ablation removes one mechanism and re-runs the frozen
evaluation. A mechanism that does not measurably lose ground when removed has
not earned its cost, and the ablation table is the evidence that says so.

Register #64 requires baselines and ablations before adaptive promotion, so
this is not optional for any promotion above C3.

| Ablation | Knob that exists today | Question it answers |
| --- | --- | --- |
| Delayed reward channel | `brain/outcome.rs::blend`, degrade to `delayed_count == 0` | Does typed `ReplyOutcome` attribution beat the immediate heuristic? |
| Exploration | `GuildSettings.epsilon_override`, pin to `EPSILON_MIN` | Is online exploration adding value or only variance? |
| Experience replay | `BATCH_SIZE`, and a variant with a buffer of size 1 | Does replay beat a purely online update? |
| Target network | `TARGET_SYNC_INTERVAL`, and a variant syncing every step | Does the target network stabilize anything at this scale? |
| Unsolicited budget | `Budget` token bucket, `unsolicited_per_hour` | Does the budget cost utility, and how much? |
| Evidence selection | `SeaWeights`, zeroing one signal at a time across the eight named fields | Which of the eight SEA signals carries the retrieval quality? |
| Retrieval index | Layered HNSW versus exact search in `abi-wdbx` | Does the approximate index change decisions, not merely latency? |
| State features | Zero one of the 18 `STATE_DIMENSIONS` at a time | Which features the policy actually uses. |

Each row reports effect size with an interval, not a point estimate, and each
reports the compute delta from section 7. An ablation that improves the primary
metric is a finding to act on, not an anomaly to explain away.

---

## 7. Complete-system accounting and the experiment manifest

### 7.1 R11, complete-system accounting

**Current.** Gap-analysis row R11 reads: "Partial. `abi-telemetry` exists (396
lines) but does not account sensor, conversion, communication, host, storage, or
control overhead." `abi-telemetry` now lives at
`dev/active/wdbx/crates/abi-telemetry`.

**Proposed, and this is a hard prerequisite.** The CSAPS negative verdict the
program must be able to produce is "parity or inferiority **after equivalent
compute, memory, data, and instrumentation are accounted for**." Without R11
closed, neither the positive nor the negative verdict can be issued, because
neither arm's true cost is known. Stated plainly: **this program cannot deliver
its central verdict until R11 is closed.** Any comparison run before then is
labeled provisional and cannot support a promotion above C3.

Accounting must cover, per arm and per guild:

- Wall-clock and CPU time at each pipeline stage.
- Peak and steady resident memory, including replay buffers and loaded brains.
- Store reads, store writes, bytes moved, and index rebuild cost.
- Provider tokens, provider wall time, and provider cost where a price exists.
- Instrumentation overhead itself, measured, so the measurement is not free by
  assumption.
- Idle cost of eviction and reload, given `DEFAULT_EVICT_AFTER_SECS`.

### 7.2 R12, the experiment manifest

**Current.** Gap-analysis row R12 reads: "Partial. `abi` has a claims registry
and golden fixtures; there is no manifest binding a result to code, model,
schema, firmware, seed, calibration, hardware, and corpus hashes."

**Proposed.** An `ExperimentManifest` is emitted for every evaluation run and
is the only object a claim record may cite as a result. It binds:

- The run record digests for every arm.
- The preregistration digest from section 5.3.
- Code, model, schema, policy, and fixture digests.
- Seeds, including the `Rng` u64 seed and restored state per arm.
- Hardware identity and host OS build.
- Corpus identity and the exact exclusion list applied.
- The accounting record from 7.1.
- The computed metrics, with intervals, and the tie-sensitive turn count.

A claim citing a result without a manifest digest fails validation. That is the
mechanism by which register #62 ("every claim names exact version and
environment") stops being a norm and becomes a check.

---

## 8. The scorecard, and three things it must never accept as evidence

**Proposed.** Constitution section 11 fixes the minimum scorecard. This section
adds the exclusions that make it honest.

### 8.1 A positive user reaction does not prove a good guild policy

`RewardCollector` is engagement-shaped by construction. Its inputs are
reactions, human replies, and deletions, with `REPLY_BASELINE = -0.2` so that
engagement has to earn the reward back. That is a defensible design for a
learning signal. It is disqualified as an evaluation metric for two independent
reasons:

- Register #37: "Explicit preferences are not inferred from engagement."
- Register #67: "A capability cannot be its sole evaluator." A policy trained
  to maximize a signal cannot be promoted on the strength of that same signal.

**Therefore the primary evaluation metric is sourced outside the reward
channel.** The proposed primary is an operator-value measure: a
periodic, structured, owner-or-administrator judgment over a sampled and
redacted set of turns, collected blind to arm assignment. Reaction counts
remain a reported secondary metric and can never carry a promotion on their own.

### 8.2 A successful API call does not prove a good outcome

A `200` response, a delivered message, or a completed Discord mutation
establishes that the effect occurred. It establishes nothing about whether the
effect was wanted. The scorecard therefore separates:

- **Execution success**: the actuator completed and postconditions verified.
- **Outcome quality**: the operator-value measure from 8.1.
- **Authorization correctness**: false-allow and false-deny counts, which are
  independent of both.

A run may show 100 percent execution success and a negative outcome delta. That
combination is a normal, expected, publishable result, not an instrumentation
bug.

### 8.3 An attractive demonstration does not prove the architecture beats a simpler baseline

Register #69: "Attractive demos do not establish sustained reliability." A
demonstration is C6 evidence at most, and C6 permits only "live-qualified for
that environment and version." A demo is not a comparison, contains no
baseline, has no preregistered threshold, and has a selection effect built in
because it is shown when it works. No demonstration, of any quality, advances a
claim past C6 or substitutes for section 5.

### 8.4 The full scorecard

Task success and operator value; authorization false-allows and false-denies;
unsafe-action and incident rate; cancellation and rollback success; uncertainty
calibration, surprise, disagreement, and drift; latency, cost, resources,
availability, privacy exposure, and evidence completeness. Plus, added by this
program: tie-sensitive turn count, replay divergence count, accounting deltas
per arm, and shadow-to-live prediction error.

---

## 9. Canary and rollback

**Proposed.** A canary is the first point at which a candidate holds live
authority, and it is bounded before it starts, not after it misbehaves.

### 9.1 Canary boundary

Every canary declares, before it runs:

- **Scope**: an explicit allowlist of scoped guild ids, opted in by an owner or
  administrator, with a stated maximum count.
- **Authority ceiling**: the capability set and maximum consequence class. For
  the DQN specifically the ceiling is fixed and is not a per-canary parameter;
  see section 10.4.
- **Budget**: separate caps for speech, observation, planning, external API
  calls, command installation, and structural changes, per constitution section
  8. The existing `Budget` token bucket and `unsolicited_per_hour` are the
  enforcement point for the speech budget.
- **Duration and expiry**: a wall-clock end after which the canary reverts
  automatically without requiring a human to notice.
- **Halt conditions**: preregistered thresholds on the scorecard which, if
  crossed, trip the rollback without deliberation.
- **The rollback artifact**: the exact prior `BrainSnapshot`, prior
  `GuildSettings`, prior manifest hash, and prior capability version, staged and
  verified restorable **before** the canary starts.

`ABBEY_QUIET` remains the higher global override during any canary, per
register #33. A canary cannot narrow or disable it.

### 9.2 Rollback

Rollback restores the staged artifact and is verified, not assumed. The receipt
identifies completed, reverted, and unresolved steps without exposing private
content, per constitution section 10. A rollback that cannot be completed is a
**Mandatory incident** retention-class event.

Rollback of a *learned policy* has a property that rollback of a deployment does
not: the guild's subsequent experiences were generated under the rolled-back
policy. Restoring the prior snapshot therefore does not restore the prior
distribution. The receipt must state that the replay buffer contents from the
canary window were generated under the reverted policy, and the operator
chooses whether to discard or retain them. Silently retaining them is a way for
a reverted policy to keep influencing the guild, and is prohibited.

### 9.3 Safety independence

Per register #14 and constitution section 3, the safety path may pause, deny, or
revoke without consulting a model, and no canary, evaluation harness, or learned
policy may disable or modify it online. A canary that requires the safety path
to be relaxed is refused, not negotiated.

---

## 10. The promotion rule

**Proposed.** Promotion is the only mechanism by which a capability's permitted
conclusion changes. It is deliberate, evidenced, and reversible.

### 10.1 The rule

For every rung C0 through C7:

1. The required evidence for that rung, per constitution section 11, exists and
   is cited by digest through an `ExperimentManifest` where the rung requires a
   measurement.
2. The evidence was produced against the exact binary, model, adapter, platform,
   policy, schema, and fixture identities recorded in the claim.
3. The preregistered threshold for that rung was set before results were
   inspected, and was met.
4. A reviewer who is not the component under evaluation signs the promotion.
   For safety-relevant and high-consequence capabilities, a human approval from
   Donald or a designated owner is additionally required, per register #13 and
   section 11's closing paragraph.
5. **No rung is skipped, and no rung auto-promotes the next.** Evidence at C3
   permits exactly the C3 conclusion.

### 10.2 Evidence demanded at each rung, made concrete for this program

| Rung | Concretely, for a learning change |
| --- | --- |
| C0 | This spec's contract for the change, its invariants, its risks, and its entry in the falsification register of section 13. |
| C1 | Unit, property, privacy, schema, and failure-path tests pass under `./tools/check.sh` for `abi`, and the repository gate for `abbey-bot`. |
| C2 | A run record replays to equivalence per section 3, including the tie-sensitive ceiling. Requires the section 1.2 fixes. |
| C3 | Frozen offline evaluation against B0 through B4 with the ablation table of section 6, calibration, adversarial cases, and the accounting record of section 7. Provisional until R11 closes. |
| C4 | A shadow run per section 4, with shadow-to-live prediction error reported and off-policy estimates labeled Inference. |
| C5 | A canary per section 9, with the rollback artifact verified restorable before start. |
| C6 | An authorized operator witnesses the exact end-to-end outcome on the exact version. |
| C7 | Repeated operation over a preregistered window establishes reliability and drift bounds. |

### 10.3 Demotion is a promotion rule running backwards

Schema drift, permission mismatch, calibration regression, missing evidence, or
a crossed halt condition **disables the affected version and preserves the last
approved version**, per constitution section 6. Demotion needs no new evidence,
only the triggering observation. This asymmetry is deliberate: advancing
requires proof, retreating requires only a reason.

### 10.4 An invariant of the promotion rule that the ladder alone does not give

**The action space does not widen as a side effect of promotion.** Registers #35
and #36, plus section 13's "Keep the existing DQN confined to low-risk speech",
mean the ladder governs *confidence in the existing*
`BotAction::{Stay, Reply, React}` and never an extension of it. C7 on the speech
policy is C7 on speech. It does not become authority over roles, permissions,
channels, moderation, integrations, or command registration, and no accumulation
of evidence converts it into that. Widening the action space is a new capability
starting again at C0, with its own contract, its own falsification criteria, and
its own separate approval.

### 10.5 Extending the existing claims registry

**Proposed changes to `dev/active/abbey/src/claims/`**, additive, preserving the
existing `Claim` fields and the `claim!` macro shape:

- Bump `CLAIMS_SCHEMA_VERSION` from `1`. The existing constant exists precisely
  so this is a versioned change rather than a silent one.
- Add the four constitutional states missing from `Status`: `Failed`, `Revoked`,
  `Superseded`, `Expired`. Extend `Status::ALL`, `label()`, and `key()`
  accordingly. The existing doc comment on `Status::ALL` already warns that
  per-status count literals go stale across a merge, so tests must partition
  through `ALL` rather than restating counts.
- Decide explicitly, with Donald, what happens to `Blocked` and `OutOfScope`,
  which the constitution does not name. The proposal is to keep both, since
  `Blocked` carries `blocker_owner` and encodes something real that the
  constitution's vocabulary does not, and to document them as registry-local
  extensions rather than constitutional states.
- Add to `Claim`: `ladder_level: Level` (C0 through C7),
  `preregistration: Option<&'static str>` (a digest), `manifest:
  Option<&'static str>` (an `ExperimentManifest` digest), `expiry:
  Option<&'static str>`, and `rollback_condition: &'static str`.
- The new fields arrive through a second `claim!` macro arm, or with defaults,
  so the roughly thirty existing positional call sites in `registry.rs` migrate
  incrementally. Adding required positional fields to the single existing arm
  breaks every row at once and turns a versioned schema change into a
  mechanical rewrite of the whole table.
- Extend `validate_registry()` with fail-closed checks:
  - a claim's `ladder_level` may never exceed what its recorded evidence
    supports, checked structurally (for example, C3 or above requires a
    non-`None` `manifest`; C5 or above requires a non-empty
    `rollback_condition`);
  - a `Current` status requires a `ladder_level` and a permitted-conclusion
    string consistent with that level;
  - a `Failed` claim requires a manifest digest, because a negative result is a
    result and must cite its evidence exactly as a positive one does;
  - an `Expired` claim may not be cited by any other claim as supporting
    evidence.

The point of putting the check in `validate_registry()` rather than in a
document is that section 11's rules then fail a build instead of failing a
reader's attention.

---

## 11. Per-guild learning and its isolation

### 11.1 What isolation means here

Constitution section 5: "Guild isolation is the correctness boundary;
guild-plus-user isolation is the member privacy boundary." Register #23: "Guilds
never share private memory."

**Current.** `brain/registry.rs` keys both the brain and its persisted snapshot
by `scoped_guild_id`, and `brain/budget.rs` keys the token bucket the same way.
The structural separation exists.

**Proposed additions:**

- An isolation test that is an Observation rather than an inference from the
  key type: plant a distinguishing token in guild A's experience stream, then
  assert it is unrecoverable from guild B's brain, guild B's decisions, guild
  B's budget, and guild B's telemetry. Run it against the shipped path.
- Eviction and reload (`DEFAULT_EVICT_AFTER_SECS = 6 * 3600`) must not leak
  across guilds through a shared allocator, a shared buffer, or a reused
  `Rng`. Each guild's `Rng` is seeded from guild-scoped material and is never
  shared.
- `abbey-bot`'s `wyhash.rs` and `embedding.rs` are transcriptions pinned by
  golden tests specifically so WDBX stores stay bit-compatible, and the repo's
  own guidance says not to deduplicate them. Nothing in this program implies a
  shared embedding across guilds, and no aggregation design may quietly create
  one.

### 11.2 The default-off migration

**Current, and a verified conflict.** `GuildSettings::default()` at
`abbey-bot/src/guild.rs:104` sets `learning_enabled: true`. `unsolicited`
defaults to `false` (verified by the older-row test at `guild.rs:452`).

Register #31 says adaptive learning is opt-in and default-off. The constitution
already names this mismatch in section 8 and requires migrating to default-off
"without silently rewriting an existing guild's explicit choice."

**Proposed migration**, which belongs to this program because it changes what
every subsequent evaluation is measuring:

1. Distinguish a stored `true` that a human set from a stored `true` that came
   from the default. Today the serialized row cannot tell them apart, so a
   provenance field is required before the default flips.
2. Flip the default for guilds with no explicit setting.
3. Leave an explicit human `true` untouched, and surface the change in
   `/admin brain` so operators can see which state they are in.
4. Record, in the run record, which of the two conditions each guild is in.
   Otherwise a post-migration evaluation silently compares an opt-in population
   against a default-on one and attributes the population difference to the
   policy.

### 11.3 The learning signal is not the evaluation signal

Restating section 8.1 as an isolation property, because it is easy to lose:
`RewardCollector` output feeds `DqnAgent::remember`. It does not feed the
scorecard. The two paths never merge, and the evaluation harness has no read
access to the reward channel other than to report it as a labeled secondary.

---

## 12. Privacy-safe cross-guild aggregation

**Proposed.** Register #40: "Cross-guild learning requires aggregate privacy
proof at a higher claim level." Constitution section 5: "No cross-guild recall."

### 12.1 The three conditions, all required

Cross-guild learning is permitted only when all three hold:

1. **Aggregate only.** What crosses the boundary is a statistic over many
   guilds, never an episode, an embedding, a message, a member identity, or a
   per-guild parameter vector attributable to one guild.
2. **C7 plus the orthogonal gate.** As established in section 0.3, C7 is the top
   rung. Cross-guild aggregation additionally requires the leakage test of 12.3
   to have passed as an Observation, on the exact shipped aggregation path, at
   the exact version.
3. **Demonstrated absence of leakage.** Absence is an observation requiring a
   test. It is never an inference from design intent, from the key type, from a
   code review, or from the fact that the aggregation "obviously" cannot leak.
   A design argument for why leakage is impossible is an Inference and is
   labeled as one.

### 12.2 Minimum construction

- A **minimum contributing-guild floor**, preregistered, below which no
  aggregate is computed or published. An aggregate over two guilds is a
  cross-guild disclosure wearing a statistic's clothing.
- A **per-guild contribution bound**, so no single guild can dominate an
  aggregate and thereby be read out of it.
- **No per-guild identifiers** in the aggregate, including indirectly through
  contribution counts, timing, or ordering.
- The aggregate is a **projection, not a canonical write**, consistent with
  register #25 ("embeddings and indexes are disposable projections").

### 12.3 The leakage test that must pass before any cross-guild claim

The construction that earns the word Observation is a **canary token**:

1. Plant a unique, high-entropy, otherwise-unused token into guild A's
   experience stream, in a position where it would influence the aggregate if
   anything guild-specific survives aggregation.
2. Run the **shipped aggregation path**, not a model of it, not a simplified
   harness, and not a description of it.
3. Assert the token is unrecoverable from the published aggregate.
4. Assert the token is unrecoverable from **every other guild's behavior**
   after the aggregate is applied: decisions, Q-vectors, telemetry, and budget
   state.
5. Repeat with the token placed in the smallest contributing guild, which is
   the worst case for the contribution bound.
6. Repeat under an adversarial reader who has full knowledge of the aggregation
   algorithm and of all guilds except A.

A failure at any step is a **Failed** claim in the registry, with its manifest
digest, and cross-guild learning stays off. A pass at the current version does
not carry to the next version; the test is re-run per version, like any other
C-rung evidence.

---

## 13. Predeclared falsification conditions

**Proposed, and predeclared.** Per registers #65 and #66, thresholds are set
before results are inspected and negative results remain publishable. The
Multiscale Orch-OR Falsification Framework's discipline applies directly here:
these conditions are written to produce informative negative results, not to be
avoided.

Each condition names what stops, so that "the program is falsified" is an
action rather than a mood.

**F1, the CSAPS verdict.** F1 cannot be evaluated until two prerequisites hold:
R11 accounting is closed per section 7.1, and the operator-value instrument of
section 8.1 exists, has been piloted, and has a measured inter-rater agreement.
Until both hold, F1 is pending and no verdict may be issued in either direction.
With both complete, if the adaptive path does not beat the B2 no-learning baseline on the preregistered primary metric by
the preregistered margin, at the preregistered interval, then the adaptive
architecture has not earned its cost. The program stops advancing the adaptive
path, the policy stays at its current scope, and the result is recorded as a
`Failed` claim with its manifest. This is the "parity or inferiority after
equivalent compute, memory, data, and instrumentation are accounted for" verdict
that CSAPS itself names, and this program is built so it can be issued.

**F2, replay.** If a run cannot be replayed to the equivalence definition of
section 3 after the section 1.2 fixes land, nothing above C2 may be claimed for
anything that depends on that run. A system whose behavior cannot be reproduced
cannot be evaluated, and no amount of favorable aggregate statistics substitutes.

**F3, tie sensitivity.** If the tie-sensitive turn count exceeds its
preregistered ceiling, the policy's decisions are dominated by numerical margin
rather than by learned structure, and the promotion is refused regardless of the
outcome metric.

**F4, isolation.** If the guild isolation test of 11.1 or the leakage canary of
12.3 fails, the affected capability is revoked immediately, not scheduled for
remediation. Cross-guild learning stays off indefinitely until a passing
Observation exists at the current version.

**F5, safety.** Any unsafe action, false-allow, or safety-path bypass observed
during shadow or canary halts the program branch that produced it. Register #15:
safety is never learned online, so a safety failure is never a tuning problem.

**F6, calibration drift.** If predicted outcome distributions drift outside
their preregistered calibration bounds during C7 monitoring, the capability
demotes to its last approved version per section 10.3.

**F7, the operator-value metric goes the wrong way.** If engagement-derived
reward improves while the operator-value measure of section 8.1 declines, the
policy is optimizing engagement at the expense of human intent, which
constitution section 1 forbids by name. This is a halt condition, not a
trade-off to balance.

**F8, accounting never closes.** If R11 cannot be closed within the program's
declared window, the program reports that its central verdict is unreachable and
publishes that as its result. An unreachable verdict is an honest outcome and is
recorded as such rather than replaced with a weaker comparison that omits cost.

---

## 14. Acceptance matrix

**Proposed.** What must be true for this program itself to be considered
delivered, independent of whether the adaptive path wins or loses.

| Item | Acceptance |
| --- | --- |
| Constitutional reconciliation | Section 0's amendments approved or rejected by Donald in writing, dated. |
| Snapshot completeness | `BrainSnapshot` carries RNG state and target-network parameters; `import_weights` restores target phase; buffer-clear semantics and skipped-experience counts are explicit and tested. |
| Run record | A run record round-trips and replays to section 3 equivalence on a fixture. |
| Shadow harness | A candidate runs on a live stream with the actuator disconnected and a separate store path, verified by an assertion that no canonical write occurred. |
| Baselines | B0 through B4 implemented and runnable from one command against a frozen corpus. |
| Ablations | The eight-row table of section 6 produces effect sizes with intervals and compute deltas. |
| Accounting | R11 closed, or explicitly reported unclosed with F8 triggered. |
| Operator-value instrument | Designed, piloted on real turns, inter-rater agreement measured and reported, collected blind to arm assignment. F1 is not evaluable until this row passes. |
| Manifest | `ExperimentManifest` emitted and cited by every claim above C3. |
| Registry | `Status` carries the four constitutional states; `validate_registry()` fails closed on level-versus-evidence mismatch; `CLAIMS_SCHEMA_VERSION` bumped. |
| Canary | A canary starts only with a verified-restorable rollback artifact, and an induced halt condition trips an actual rollback with a receipt. |
| Isolation | The guild isolation test and the cross-guild leakage canary both run against shipped paths and both are Observations. |
| Falsification | F1 through F8 preregistered, dated, digest-stamped, before the first comparison is computed. |

---

## 15. Honest residual

What this spec does not do, stated so it is not assumed.

- It specifies no production code and authorizes no implementation. Per
  constitution section 15, no program is authorized merely because a document
  exists.
- It does not test the CSAPS architecture. CSAPS revision 2.0's own status box
  says the integrated system is not empirically validated. Nothing here changes
  that, and this program is designed to be capable of confirming it.
- It does not cover Program 6's arbiter design, Program 4's write gate, or
  Program 1's contracts, each of which this program depends on. In particular,
  shadow isolation is weaker than it should be until `ProposeWrite` exists, and
  section 4.2 says so rather than hiding it.
- It assumes the constitution's section 11 rung definitions are final. If
  Donald's L0-L8 vocabulary reflects an intended revision of section 11 rather
  than a slip, this spec's rung mapping needs redoing, and section 0.3 is the
  place to start.
- The operator-value measure of section 8.1 is specified as a requirement, not
  as an instrument. Designing a structured, blind, low-burden operator judgment
  that owners will actually complete is real work and is not solved here.
- No claim is made that the current `abbey-bot` DQN is good or bad. It has never
  been evaluated against a strong baseline, which is the observation that
  motivates this entire program.
