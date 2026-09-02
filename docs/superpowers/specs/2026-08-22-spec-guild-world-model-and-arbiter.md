# Program 6: Guild world model and adaptive arbiter

> **Constitutional mapping.** This file was written against the seven-program
> list Donald gave in conversation. The ratified constitution's section 13
> "Delivery programs" uses a different numbering, and the constitution wins.
> In section 13's terms this document specifies the **arbiter half of Program 6, Model registry and adaptive arbiter**, plus the world-model half of section 8. The model-registry half is a separate spec.
>
> The filename is therefore name-based rather than numbered, so no numbering is
> asserted. Nothing in section 13 was renumbered: section 15 reserves amendment
> to Donald, and the collision is raised as one request covering the whole set
> rather than five independent ones.

**Reconciled ownership.** This is the arbiter portion of constitutional Program
6, `model-registry-adaptive-arbiter`. It consumes Program 3 guild facts and
Program 4 evidence. Program 1, `abbey-contracts`, owns the shared wire schemas
and fixtures; this document owns proposed inference and routing behavior only.


Status: **proposed design.** No implementation is authorized by this document.
Written 2026-08-22 against `dev/active/abi`, `dev/active/wdbx`, and
`dev/active/abbey-bot` as they stood in the working tree on that date.

Scope, in Donald's words: model guild goals, organizational state, active
regime, uncertainty, surprise, risk, budgets, and escalation among fast,
retrieval-conditioned, and deliberative modes.

Governing documents, in precedence order: the Abbey System Constitution
(`2026-08-22-abbey-system-constitution.md`), the WDBX conformance gap analysis
(`2026-08-22-wdbx-conformance-gap-analysis.md`), and then this spec. Where this
spec appears to contradict the constitution, the constitution wins and this
spec is wrong.

`CSAPS_WDBX_Revised_2026.pdf` revision 2.0 is cited throughout as a **proposed
criterion** only. Its own status box says the integrated system has not been
empirically validated, so every threshold it supplies is an acceptance target
to be preregistered and tested, never evidence that a mechanism works.

Every claim below is labeled **Current** (verified by reading the named source
file on 2026-08-22) or **Proposed** (a design target). Nothing labeled Current
is a defect report against earlier scope.

---

## 0. Constitutional reconciliation

This spec was assigned the name "Program 4, Guild World Model and Adaptive
Arbiter" and the filename above. That name does not match the constitution's
delivery-program register, and the mismatch is three-way:

| Source | What it calls this number |
| --- | --- |
| Constitution section 13 | Program 4 is "Canonical WDBX episodes and claims"; Program 6 is "Model registry and adaptive arbiter", explicitly including regime inference, calibrated escalation, and structured outcome learning |
| WDBX gap analysis, header line | Describes itself as "Input to Program 3, the Canonical WDBX Episodic Contract" |
| This assignment | Program 4 is the guild world model and adaptive arbiter |

The mechanism content of this document maps onto **constitution section 13,
Program 6**, plus the arbiter-facing half of **constitution section 8**. It
depends on constitution-Program-3 (the read-only guild twin) and
constitution-Program-4 (canonical WDBX episodes and claims); see section 9.

The approved federation reconciliation keeps section 13 unchanged and maps
this document to Program 6, `model-registry-adaptive-arbiter`. The earlier
Program 4 label is historical assignment context, not a second program number.
New work cites the constitutional slug and number above.

### What this program does not own

Constitution section 2 gives every concern exactly one canonical owner, and
section 8 already specifies the guild twin's five views. This program
**consumes** the structure graph, authority graph, workflow graph, goal model,
and health model exactly as constitution-Program-3 defines them. It does not
re-specify them, does not extend their schemas, and does not write to them.

Its own state ownership is the last three rows of the constitution's section 4
state table, and only those:

| State class | Owner | This program |
| --- | --- | --- |
| Fast | the turn | Reads a bounded per-turn summary |
| Adaptive | guild lanes 1 to 4, section 8 | Reads declarative preferences only |
| Regime | **this program** | Owns representation, inference, and transitions |
| Predictive | **this program** | Owns uncertainty, surprise, calibration |
| Resource | measured by the host, **ledgered by this program** | Owns the composed ledger, never the measurement |

The arbiter runs in the **soft real-time** domain of constitution section 4.
It is never on the bounded-real-time path: it does not participate in media
gates, VAD, consent-epoch closure, cancellation, or safety interlocks, and no
later revision may wire regime inference into the voice interlock. Constitutional
invariant A4 stands above everything in this document.

---

## 1. Current state, measured

### 1.1 Repository shape

**Current.** In the `dev/active/abi` working tree on 2026-08-22, `Cargo.toml`
`[workspace.dependencies]` resolves `abi-foundation`, `abi-core`,
`abi-telemetry`, `abi-compute`, and `abi-wdbx` to `../wdbx/crates/`, and
`crates/` holds 14 crates. `Cargo.toml` is not among the roughly 18 modified
files in that tree, so the split is committed at this checkout. The gap
analysis measured against `origin/main` (`0278a2f` lineage); this document did
not verify branch or merge state, so treat the extraction as observed at this
checkout rather than as a statement about `origin/main`.

Consequence for this program: a new arbiter crate belongs in `dev/active/abi`,
above the substrate, because it depends on `abi-ai`, `abi-sea`, and
`abi-agent-host`, none of which moved.

### 1.2 The eight-signal scorer is a good seam and a partial precedent

**Current.** `crates/abi-sea/src/scorer.rs` defines `SeaSignals` with eight
orthogonal fields (`semantic`, `keyword`, `metadata`, `recency`, `authority`,
`graph`, `contradiction`, `task_fit`) and combines them in `sea_score` as a
**weighted sum** clamped to `[0,1]`, with `DEFAULT_SEA_WEIGHTS` summing to 1.0
and per-task additive adjustments in `adjust_weights_for_task`.

This is materially better than the WDBX substrate's
`ScoreComponents::combined()` (`temporal.rs:15`), which the gap analysis
records as a multiplicative collapse where any single zero factor silently
vetoes retrieval. A weighted sum has no silent-veto term.

**Current, and the reason this program must not copy the pattern at the
decision boundary.** `select_sea_candidates` returns a `SeaSelection` carrying
`selected_ids`, `rejected_ids`, `total_estimated_tokens`, and a single
`reason: &'static str` for the whole selection, valued either
`"all candidates selected"`, `"no candidates to select from"`, or
`"budget-limited"`. Three distinct constraints can reject a candidate (the
per-cluster diversity cap, `max_records`, and `max_tokens`), and the returned
selection cannot attribute any given rejected `record_id` to which one fired.
The cluster cap additionally carries an unexplained numeric override,
`count >= options.per_cluster_limit && c.final_score < 0.92`, so a candidate
above 0.92 bypasses the diversity cap for a reason the type does not record.
The per-signal `SeaSignals` are computed and are present on each
`SeaCandidate`, but they do not survive into `SeaSelection`.

This is an observation, not a defect claim: SEA was built as a retrieval
budgeter, and for that job one aggregate reason is adequate. This program does
not modify `abi-sea`. It records the pattern because constitution section 12
requires that "no opaque score hides a veto" at the cognitive-decision
boundary, and the arbiter is exactly that boundary. **Design rule A** in
section 6.4 below states the non-repetition requirement.

### 1.3 The adaptive modulator is style drift, not learning

**Current, and the strongest single finding in this survey.**
`crates/abi-ai/src/modulator.rs` implements `AdaptiveModulator` as an
exponential moving average over `ProfileWeights` with `DEFAULT_ALPHA = 0.3`,
a saturating `update_count`, and hardened `deserialize` validation that falls
back to the neutral prior on any malformed state.

The loop that drives it is `crates/abi-sea/src/learn_loop.rs`:

```rust
modulator.update(analyze_sentiment(input));
```

`analyze_sentiment` (`crates/abi-ai/src/router.rs`) scores the **input text**
against a keyword table. No outcome, reward, correction, acceptance signal, or
observed consequence appears anywhere in this update. The EMA therefore tracks
what the user has recently been talking about, and nothing else.

Three consequences bind this program:

1. `AdaptiveModulator` is **not** a confidence source and must never be read as
   one. Its `weights()` are a persona prior; its `update_count()` is a count of
   turns, not a count of confirmations.
2. It is not "structured outcome learning" in the sense of constitution section
   13 or section 8 lane 4, because there is no outcome in the loop. Any future
   claim that Abbey learns from outcomes must not cite this code as evidence.
3. It is a clean illustration of the failure this spec is required to prevent:
   a quantity that rises with repetition and looks like growing confidence
   while being uncorrelated with correctness. The arbiter's confidence
   estimator must be structurally incapable of the same thing; see section 7.

`crates/abi-ai` is otherwise pure, deterministic, and free of I/O and of any
WDBX dependency, which is what makes `ai_run` byte-reproducible and
golden-testable. The arbiter must preserve that property in whatever code it
adds to that crate, and should prefer not to add any.

### 1.4 The existing Discord decision flow

**Current.** `dev/active/abbey-bot` implements a smaller deterministic pipeline
that this program conditions on rather than replaces:

| File | What it establishes |
| --- | --- |
| `src/pipeline.rs` | Triage, intent, state encoding, per-guild policy, cooldown, persona routing, delayed reward, all behind an injected clock and a recording `Outbound` trait, so the whole decision path runs in tests |
| `src/brain/state.rs` | `STATE_DIMENSIONS = 18`, an explicitly deterministic encoder with the hour of day injected rather than read, and the action space `BotAction::{Stay, Reply, React}` |
| `src/brain/dqn.rs` | epsilon-greedy DQN, `GAMMA = 0.99`, `EPSILON_INITIAL = 0.1`, `EPSILON_MIN = 0.01`, `BATCH_SIZE = 64`, `TARGET_SYNC_INTERVAL = 100`, and `OutputActivation::Linear` chosen deliberately so Q-values keep magnitude and sign for the Bellman target |
| `src/brain/budget.rs` | Per-scoped-guild refilling token bucket for unsolicited actions, injected clock, and a documented "clock that went backwards adds nothing" rule |
| `src/guild.rs` | `DEFAULT_COOLDOWN_SECONDS = 20`, `MAX_COOLDOWN_SECONDS = 600`, `DEFAULT_BUDGET_PER_HOUR = 6`, `MAX_BUDGET_PER_HOUR = 60`, scoped ids of the form `"{platform}:{native}"` as the isolation invariant |
| `src/routing_signals.rs` | A signal layer that composes on top of the canonical router's *decision* and fires only when the canonical router returns the neutral prior, with an integer `FIRE = 2` threshold |

Two constraints follow. First, per constitution section 13 and decision 36,
the existing DQN **stays confined to low-risk speech**. The arbiter does not
subsume it, does not extend the `{Stay, Reply, React}` action space, and never
lets a Q-value reach a role, permission, channel, moderation, integration, or
command decision. Second, `OutputActivation::Linear` means Q-values are
unbounded reals by design. They are not probabilities, they are not calibrated,
and section 5 forbids comparing them to anything without normalization.

**Current, and a constitutional conflict already flagged in section 8 of the
constitution.** `GuildSettings::default()` in `src/guild.rs` sets
`learning_enabled: true`, against decision 31 (adaptive learning is opt-in and
default-off). The neighboring `unsolicited` field correctly defaults to
`false`. This program does not depend on `learning_enabled` and must not read
a `true` value as consent for anything it does; its own learning lanes carry
their own default-off flags.

### 1.5 Existing resource-limit machinery

**Current.** `crates/abi-agent-host/src/budget.rs` defines `HostBudget` with
nine finite ceilings for one complete agent run: `max_events` (1,024),
`max_event_bytes` (65,536), `max_output_tokens` (16,384), `max_output_bytes`
(1,048,576), `max_tool_calls` (32), `max_tool_rounds` (8),
`max_provider_runs` (9), `max_tool_result_bytes` (65,536), and `max_duration`
(300 s), with a matching `HostBudgetLimit` enum whose `as_str` gives each limit
a stable lowercase label. This is the correct shape: a run stops on a **named**
limit, not on an aggregate score.

**Current.** `crates/abi-sea/src/evidence.rs` sets `MAX_PROMPT_BYTES = 4096`
and `MAX_EVIDENCE_LIMIT = 100`, the latter applied before embedding search,
metadata cloning, and scoring so an untrusted caller limit cannot turn a
request into unbounded recall work.

**Absent.** There is no money axis, no rate-limit-availability axis, and no
composed per-guild ledger spanning speech, observation, planning, external API
calls, command installation, and structural changes as constitution section 8
requires as separate budgets.

---

## 2. The guild world model: contents and update cadence

The guild world model (GWM) is the versioned, guild-isolated digital twin of
constitution section 8, plus this program's three state classes. Guild
isolation is the correctness boundary and guild-plus-user is the privacy
boundary; a GWM instance is keyed by scoped guild id in the
`"{platform}:{native}"` form `abbey-bot/src/guild.rs` already establishes, and
no read path may span two keys.

### 2.1 Views and cadence

**Proposed.** Rows one through five are owned by constitution-Program-3 and
appear here only so the cadence table is complete and so this program's read
dependencies are explicit.

| View | Owner | Update trigger | Cadence | Durability |
| --- | --- | --- | --- | --- |
| Structure graph | Program-3 | Gateway structural events, plus a bounded periodic reconciliation sweep because gateway streams are lossy across reconnects | Event-driven; sweep on a budgeted schedule | Durable, per-assertion staleness stamp |
| Authority graph | Program-3 | Permission, role, hierarchy, grant, and revocation events | Event-driven; revalidated immediately before any side effect | Durable |
| Workflow graph | Program-3 | Derived from observed command and process completions | Recomputed on a best-effort schedule | Durable |
| Goal model | Program-3, human-approved only | Explicit human action only | On change | Durable |
| Health model | Program-3 | Rolling aggregates over fixed windows | Window close | Durable |
| **Fast summary** | the turn | Every inbound event | Per turn, TTL in seconds | Never durable |
| **Regime** | this program | A persistent shift (section 3) or a human declaration | Only on transition, subject to minimum dwell | Durable, as a supersession edge |
| **Predictive** | this program | Calibration refit, offline | Refit on a schedule and on regime change; **never fit in the request path** | Durable, versioned |
| **Resource** | this program | Continuous measurement by the host | Continuous; read, never predicted, by any model | Operational TTL |

Two cadence rules are load-bearing rather than incidental. **Safety is never
learned online** (decision 15), so no calibration map, threshold, or regime
transition may be fit inside a request. And the goal model changes only by
explicit human action; it is never inferred from engagement (decision 37),
which forbids a design where sustained activity in a channel is read as a goal.

### 2.2 Assertion envelope

**Proposed.** Every GWM assertion, in every view, carries source, observation
time, confidence basis, staleness policy, contradiction state, privacy class,
and schema version, and the three assertion types stay type-distinct:
**platform fact**, **Abbey inference**, and **human-approved goal**
(constitution section 8, decisions 38 and 39). An inference may never be
promoted to a platform fact by repetition or by an absence of contradiction.

### 2.3 Privacy floor for world-model features

**Proposed.** Passive guild intelligence works without message content
(decision 20, and the privacy test in constitution section 12). Every feature
the arbiter reads from the GWM must be derivable from metadata and aggregates
alone.

The current turn's own text is different: it is in scope for that turn, and
`abbey-bot/src/brain/state.rs` already derives turn-local scalars from it
(length capped at 400 characters, trailing question mark, a deterministic
lexicon sentiment). Those scalars remain permissible **as fast state**. They
may not accumulate into a durable per-user or per-guild profile, and prompts
and generated responses are not durable operational evidence by default
(decision 21). Anything that enters a durable episode is a bounded, redacted
state summary.

---

## 3. Regime: representation and change-point detection

### 3.1 Representation

**Proposed.** A regime is a discrete label carried with a posterior, never a
bare hard label, because the arbiter needs to distinguish "confidently Normal"
from "no idea, defaulting to Normal", and those two produce opposite correct
behavior.

| Regime | Meaning | Entered by |
| --- | --- | --- |
| `Unknown` | Insufficient evidence; the initial state of every new guild | Default at first observation |
| `Normal` | Steady-state operation | Inference, after sufficient evidence |
| `Onboarding` | New guild or major influx, structure still forming | Inference or declaration |
| `Event` | A planned, bounded, high-activity period | Declaration preferred; inference permitted |
| `Incident` | Active disruption: raid, outage, moderation surge | Inference **or** the safety path |
| `EmergencyRestriction` | Guild-wide restriction in force | Human declaration or the safety path only |
| `Maintenance` | Deliberate quiet or freeze | Declaration only |

Rules that make the representation safe rather than merely descriptive:

1. **A declared regime outranks an inferred one.** Owner beats administrator
   beats learned preference (decision 12).
2. **Entry into a restrictive regime is cheap; exit is expensive.** The safety
   path may enter `Incident` or `EmergencyRestriction` without consulting any
   model (invariant A4). Those two regimes are exited only by human action or
   by an explicit declared TTL, never by the inference deciding things look
   calm again. Asymmetric thresholds and asymmetric authority are deliberate.
3. **`Unknown` is not `Normal`.** A guild with no history is `Unknown`, and
   section 7.5 specifies what the arbiter does there. Defaulting a new guild to
   `Normal` is the single easiest way to ship a confidently wrong system.
4. A regime transition is an episodic write with provenance and a supersession
   edge, never an overwrite of the prior regime record (constitution section 5,
   decision 30).

### 3.2 Persistent shift versus transient anomaly

**Proposed.** These are different phenomena with different correct responses,
and collapsing them is a known way to make a system flap.

- A **transient anomaly** is a short excursion in one or more signals. Correct
  response: raise this turn's surprise, which raises the required verification
  depth for this turn only (section 6.5). The regime does not change.
- A **persistent shift** is a durable change in the generating distribution.
  Correct response: change the regime, invalidate the calibration that was fit
  under the old regime, and widen confidence until new outcomes accumulate.

Detection is two-timescale:

**Fast detector, per signal.** Maintain a short-window EWMA baseline and a
**robust** scale estimate (median absolute deviation, not standard deviation,
so a single spike does not inflate the scale and mask the next one). A
standardized residual above a preregistered threshold flags a transient
anomaly and contributes to surprise. This detector never changes the regime.

**Slow detector, on the normalized surprise stream.** A run-length change-point
posterior (Bayesian online change-point detection, with a CUSUM or
Page-Hinkley variant acceptable as a simpler first implementation) declares a
persistent shift only when **all three** hold:

1. the run-length posterior concentrates on a new segment above a preregistered
   probability;
2. the shift is sustained past a minimum dwell measured in **both** wall time
   and event count; and
3. the effect size clears a preregistered minimum.

The dual dwell requirement is not belt-and-braces. A quiet guild produces few
events, so an event-count-only window is arbitrarily long in wall time and a
real incident goes undetected for hours. A raid produces thousands of events in
seconds, so a wall-time-only window is arbitrarily many events and one burst
rewrites the regime. Requiring both bounds the error in both directions.

Hysteresis: the exit threshold for a regime is strictly stricter than its entry
threshold, and a minimum dwell applies to the new regime before any further
transition is considered. This is what stops flapping between `Normal` and
`Incident` at the boundary.

### 3.3 What a regime change does to confidence

**Proposed, and counterintuitive enough to state explicitly.** A regime change
**widens** uncertainty. The calibration map was fit under the old regime; WDBX
invariant "similarity is not applicability" says the same reasoning applies
that evidence retrieved under one regime does not license a conclusion under
another. Until a preregistered minimum number of outcomes accumulate in the new
regime, the arbiter falls back to the conservative prior and lowers its
escalation threshold, so it escalates **more** immediately after a regime
change, not less.

A design that grows more confident after a change point is the same failure
mode as section 1.3's modulator, wearing a different hat.

---

## 4. Uncertainty and surprise

**Proposed.** The two are distinct and must not be summed into one number.

- **Uncertainty** is the calibrated probability that a chosen mode produces an
  acceptable outcome for this turn: `P(success | mode, features, regime)`. It
  answers "how likely am I to be right."
- **Surprise** is the degree to which the current turn is off the distribution
  the calibration map was fit on. It answers "is my uncertainty estimate itself
  trustworthy." High surprise does not mean the answer is wrong; it means the
  confidence number should not be believed.

They act at different points in the decision rule. Uncertainty is compared to a
consequence-class floor. Surprise raises the **required verification depth**
and can force escalation regardless of a high confidence reading, because a
confident estimate from an off-distribution input is exactly the pathology.

Surprise inputs, all metadata-derivable: standardized residuals from the
section 3.2 fast detectors; retrieval-set novelty (how far the top SEA
candidates sit from the guild's historical retrieval distribution); regime
posterior entropy; disagreement between the deterministic router and the
retrieval-conditioned prediction; and the recency of the last regime change.

---

## 5. Normalization: why raw heterogeneous scores cannot be compared

**Current.** The scores this program can observe today live on incompatible
scales, and none of them is a probability:

| Signal | Source | Range and meaning |
| --- | --- | --- |
| `sea_score` | `abi-sea/src/scorer.rs` | `[0,1]`, a clamped weighted sum of eight sub-scores with hand-chosen weights |
| Q-values | `abbey-bot/src/brain/dqn.rs` | Unbounded reals; `OutputActivation::Linear` is deliberate so magnitude and sign survive for the Bellman target |
| Cosine similarity | WDBX retrieval | `[-1,1]`, with the useful mass concentrated in a narrow high band that varies by embedding and by corpus |
| `Authority::score()` | `abi-sea/src/types.rs` | A hand-assigned ladder: 0.30, 0.78, 0.86, 0.90, 1.00 |
| `routing_signals` score | `abbey-bot/src/routing_signals.rs` | Small integers with `FIRE = 2` |
| Budget tokens | `abbey-bot/src/brain/budget.rs` | A float count of hourly actions |
| `ProfileWeights` | `abi-ai/src/router.rs` | Normalized to sum to 1.0, but a persona prior, not a probability of anything |

**Proposed.** Comparing these directly is a category error with three separate
failure modes. They have different supports, so a threshold tuned on one is
meaningless on another. They have different distributions per guild and per
regime, so a fixed threshold drifts. And most of them are monotone in something
useful without being calibrated to anything, so a value of 0.9 carries no
frequency interpretation at all.

The normalization pipeline, in order:

1. **Rank-normalize within a per-guild, per-regime, per-signal reference
   distribution.** This removes scale and support differences and is robust to
   the heavy tails that cosine similarity and unbounded Q-values both produce.
   Reference distributions are maintained as bounded quantile sketches with an
   explicit staleness TTL, and are invalidated on regime change.
2. **Map to a calibrated probability with a monotone function** fit on held-out
   observed outcomes: isotonic regression where sample size permits, Platt
   scaling as the small-sample fallback.
3. **Compare only calibrated probabilities**, and only against
   consequence-class floors expressed in the same units.

The monotonicity requirement in step two is the property that makes this safe:
a monotone map cannot reorder candidates within a single signal, so calibration
can never hide or invert a signal's own ranking. It only makes cross-signal
comparison mean something. A non-monotone learned map would be able to
manufacture agreement between signals that disagree, which is precisely the
"averaging disagreement into false confidence" the constitution forbids in
section 9.

**Preserved, not collapsed.** Normalization produces a comparable scalar per
signal. It does not delete the per-signal values. Constitution section 5 keeps
evidence dimensions individually inspectable, and section 12 requires that no
opaque score hide a veto. The arbiter's decision record therefore carries the
raw value, the rank-normalized value, and the calibrated value for every signal
it consulted, plus the calibration map version each used.

---

## 6. The resource ledger

**Proposed.** A composed, per-guild, per-window ledger over four axes that are
**not** interchangeable and must not be summed into a single cost scalar.

| Axis | Unit | Measured from | Enforcement |
| --- | --- | --- | --- |
| Latency | Milliseconds against a per-request deadline; plus queue depth | Host measurement | Deadline is hard; a mode whose p95 exceeds the remaining deadline is inadmissible |
| Tokens | Input and output tokens, per turn and per guild-window | Provider-reported, host-counted | Composes `HostBudget::{max_output_tokens, max_output_bytes, max_provider_runs}` |
| Money | Currency, per guild-window | Versioned price table times measured tokens | Hard cap per window; see the unknown-price rule below |
| Rate-limit availability | Boolean plus retry-after | Platform response headers and 429s | An **availability** constraint, never a cost |

Four rules make this correct rather than merely tabular.

**6.1 Rate limits are availability, not expense.** A Discord 429 with a
retry-after means the action is impossible right now. Modeling it as a cost
term lets an arbiter with budget headroom "afford" to retry into a hard wall,
and lets a high-value turn justify hammering a limiter. Rate-limit state gates
admissibility before any cost comparison happens.

**6.2 Separate budget lines, per constitution section 8.** Speech, observation,
planning, external API calls, command installation, and structural changes each
have their own guild budget. They do not fund each other. `ABBEY_QUIET` is the
higher global override (decision 33) and is checked before any ledger read.

**6.3 An unknown price is `Unknown`, not zero.** Prices come from an
operator-supplied, versioned table with an effective date. When a route's price
is unknown, the money axis returns `Unknown`, and a mode whose money cost is
`Unknown` is treated as **exceeding** the remaining budget for any turn whose
consequence class requires a money check. Defaulting an unknown price to zero
makes the most expensive unmapped route look free, which is the mirror image of
the suppression failure and just as silent.

**6.4 Design rule A: the ledger names the binding constraint.** Following
`HostBudgetLimit::as_str`'s precedent and deliberately not following
`SeaSelection`'s single aggregate `reason`, every admissibility refusal names
which axis and which limit refused, per mode. An `ArbiterDecision` that says a
mode was inadmissible without naming the constraint is a defect, and section
10's C1 tests assert this.

**6.5 The audit sample gets its own preallocated budget line.** See section
7.3. It is not funded from the general planning budget, because the audit is
most needed exactly when the general budget is under pressure.

---

## 7. The arbiter

### 7.1 The three modes

**Proposed.** Three execution modes ordered by cost, plus two terminal
outcomes that are not modes.

| Mode | What it is | Existing code it composes | Domain |
| --- | --- | --- | --- |
| **M0 fast deterministic** | No model call at all | `abbey-bot`'s `platform::triage`, `brain::intent`, `persona::route` composed with `routing_signals::route`, cooldown, budget check, template and receipt rendering | Soft real-time, bounded |
| **M1 retrieval-conditioned** | One qualified generation call conditioned on SEA evidence, no tools, no side effects | `abi-sea::gather_evidence_with_plan`, `augment_prompt` under `MAX_PROMPT_BYTES`, then one generation | Soft real-time |
| **M2 deliberative** | A planner or tool workflow yielding an **inspectable typed proposal**, never a direct effect | `abi-agent-host` under a `HostBudget` | Best effort |

Terminal outcomes the arbiter must also be able to select, per constitution
section 4: **human escalation** and **refusal or safe pause**. These are
modeled as outcomes rather than as a fourth mode because they are authority
decisions, not compute levels. Escalating to a human is categorically different
from escalating to a larger model, and the constitution is explicit that
escalation "never means automatically asking the largest model."

**Invariant M1: modes are ordered in cost and are not ordered in authority.**
M2 does not receive a wider capability grant than M0. Escalation never widens
permission, never adds a tool that was not already granted, never lowers an
evidence requirement, and never crosses a privacy boundary (decision 60). The
naive design where the expensive path gets tools and the cheap path does not is
exactly the A3 and decision-9 violation this invariant forbids.

### 7.2 The decision rule

**Proposed.** Evaluated in this order, and the order is normative.

1. **Safety and authorization floor first.** ABI's authorization result and the
   safety path's verdict are computed before the arbiter runs and are inputs to
   it, not outputs of it. They may force refusal, force human approval, or
   raise the required verification depth. The arbiter can never lower any of
   them. Invariant A4.
2. **Admissibility.** For each mode, check privacy class (a private-material
   turn with no qualified local route makes cloud modes inadmissible, decisions
   52 to 54), rate-limit availability, deadline, and each budget line. Record
   the named binding constraint for every inadmissible mode.
3. **Consequence class fixes the floor.** The turn's consequence class `k`
   determines a required success floor `q_k` and a required verification depth,
   using the constitution's section 9 low / medium / high ladder.
4. **Choose the cheapest admissible mode whose calibrated
   `P(success | mode, features, regime) >= q_k`,** with surprise-adjusted
   verification depth applied.
5. **If that set is empty, escalate to a human or refuse.** Never silently
   select the best available mode below the floor.

Stated as a single sentence, because the shape is the whole safety argument:
**cost breaks ties among modes that already clear the floor; cost never lowers
the floor.**

### 7.3 Invariant C: cost never enters the confidence estimator

**Proposed, and this is the load-bearing invariant of the entire program.**

The confidence estimator has one training signal: observed outcome labels. Its
loss function contains no latency term, no token term, no money term, no rate
term, and no budget-pressure feature. Cost enters the system **only** in step 2
(admissibility) and step 4 (tie-breaking) of the decision rule, downstream of a
confidence number that was computed without any knowledge of what the answer
would cost.

The mechanism this prevents: if cost is in the confidence estimator's loss,
gradient descent discovers that predicting high confidence on expensive turns
reduces loss. The estimator then learns to be most confident exactly where
escalation is most expensive, escalation rate falls, the cheap metric improves,
and every dashboard looks better while quality silently degrades. This is not a
hypothetical failure mode; it is the direct consequence of putting cost in the
objective, and it is unobservable from aggregate metrics because the arbiter
stops generating the counterfactual that would reveal it.

Implementation consequence: the estimator and the mode selector are separate
components with separate training data, separate versioning, and separate
promotion gates. A single end-to-end policy trained on a reward that nets
quality against cost is **out of scope for this program** and would require its
own spec, its own claim record, and independent review, because it structurally
cannot satisfy this invariant.

### 7.4 Detecting the silently overconfident arbiter

**Proposed.** Invariant C removes the most direct route to the failure. It does
not prove the failure is absent, because confidence can also degrade through
distribution shift, a stale calibration map, or a correlated feature that
proxies for cost. The following are the detection mechanisms, and they are
required for promotion past C4.

**7.4.1 Stratified forced audit sampling.** A fixed fraction of eligible turns
is escalated to M2 regardless of the arbiter's decision. Three properties are
mandatory:

- The sampling decision is made by a **seeded PRNG independent of the
  arbiter**, and the arbiter never observes whether a turn was audit-sampled,
  before or after. An arbiter that can detect audit turns can behave
  differently on them, and the sample stops being informative.
- The sample is **stratified by claimed confidence, with a per-bucket floor**.
  Uniform sampling puts nearly every audit turn in the high-confidence bulk,
  which is the region where the estimator is already fine, and leaves the tail
  where the arbiter says 0.97 and is wrong with almost no observations.
  Per-bucket floors buy bounded variance in every bucket, which is what makes a
  per-bucket reliability curve readable instead of an aggregate ECE that
  averages the failure away.
- The audit draws on its **own preallocated ledger line** (section 6.5). If it
  is funded from the general planning budget, it is cut first under budget
  pressure, which is precisely when the suppression failure is occurring.

Without this, the system observes outcomes only on the arm it chose, and
confidence looks excellent by construction because it is never tested where it
matters.

**7.4.2 Divergence-versus-confidence curve.** On audit turns, measure how often
M2 materially changes the answer or the safety verdict relative to what the
arbiter chose. Plot divergence against claimed confidence, per bucket. An
honest estimator produces a curve that falls monotonically as confidence rises.
**A flat or non-monotone curve is the failure signature**, and it is visible in
the stratified sample long before it is visible in any aggregate.

**7.4.3 Paired guardrail metrics.** Preregistered pairs where the cheap metric
and the quality metric must move together. The failure is defined as one moving
without the other:

| Cheap metric | Paired quality metric | Failure signature |
| --- | --- | --- |
| Escalation rate falls | Audit divergence, human-correction rate | Escalation falls while either rises or stays flat |
| Mean claimed confidence rises | Per-bucket reliability | Confidence rises while reliability degrades |
| Cost per turn falls | Undo and rollback rate, task success | Cost falls while either worsens |
| Refusal rate falls | Authorization false-allow count | Any false-allow at all |

**7.4.4 Cost-blind shadow arbiter.** Run a second arbiter offline on the same
feature stream with the cost term zeroed. If the cost-aware arbiter's
escalation rate sits far below the cost-blind arbiter's **and** audit divergence
is elevated, budget pressure is buying itself confidence through some path
Invariant C did not close. This is the cross-check that catches a proxy feature.

**7.4.5 Budget-forced downgrades are recorded as a distinct outcome.** A turn
where the ledger forced M0 is recorded as `Forced { by: <named limit> }`, not
as `Chosen`. Three things depend on this: the response must disclose degraded
operation rather than presenting it as full capability (decision 60,
constitution section 10); the calibration analysis must exclude forced turns,
because the mode was not the estimator's choice and including them corrupts the
reliability estimate; and the audit stratum stays exchangeable with the
suppressed stratum. Budget-forced silence that is not labeled reads exactly
like confidence.

### 7.5 Cold start

**Proposed, and the likeliest way this ships wrong while looking fine.** A new
guild has no outcomes, therefore no calibration map, therefore **no basis on
which to certify that any mode clears the floor**.

The honest rule: uncalibrated, or regime `Unknown`, means the arbiter **cannot
certify the floor** and must escalate to M2, escalate to a human, or refuse,
according to consequence class. It must not default to the cheap path. A system
that treats "no evidence of a problem" as "evidence of no problem" is the same
error as reading `update_count` as confidence in section 1.3.

This is a first-class, preregistered, tested path, not a footnote of the
`Unknown` regime. Its C1 tests assert that a fresh guild with zero outcomes
never selects M0 for a medium or high consequence class, and its canary
boundary (section 10) includes at least one genuinely new guild.

**The global prior is a cross-guild back door.** Cross-guild pooling is out of
scope for this program (decision 40 requires an aggregate privacy proof at a
higher claim level, which this program does not attempt). That exclusion is
defeated if the "global prior" used at cold start is itself fit on pooled live
guild outcomes. The prior is therefore fit on **fixtures and the deterministic
baseline only**, never on live guild data from any guild, and the C1 privacy
tests assert its provenance.

### 7.6 Determinism and replay

**Proposed.** Given a pinned feature vector, regime posterior, calibration-map
version, policy version, price-table version, and seed, the arbiter's decision
is replayable byte for byte, which is what C2 in the evidence ladder means. No
wall-clock read, no filesystem read, and no network read occurs inside the
decision function; time, ledger state, and regime are injected. This follows
the convention already established across `abbey-bot/src/guild.rs`,
`brain/budget.rs`, and `brain/state.rs` (all inject the clock) and preserves
`abi-ai`'s purity property.

---

## 8. Naming: what this spec recommends and what it does not adopt

CSAPS section 6.7 proposes `ArbiterService`, `StateService`, and
`PredictionService`. The gap analysis deferred that renaming question to this
program and warned that renaming during extraction "converts a resemblance into
an architectural commitment without a spec."

**This spec recommends against adopting all three names, and adopts none of
them.** The constitution defers the decision to this program; a recommendation
is not a decision, and Donald's approval is required either way.

Reasons, in order of how verifiable they are:

1. **`abi-sea` is load-bearing on frozen surfaces.** The name appears in the
   frozen 12-tool MCP surface (`ai_learn`), in the `sea` and `sea-learn-loop`
   skills, and in golden fixtures under `tests/golden/` that are pulled in with
   `include_str!`. Renaming it toward `PredictionService` would break
   golden-tested contract boundaries for cosmetic alignment with a paper whose
   own status box says the integrated system is unvalidated. That is a bad
   trade in both directions.
2. **`abi-sea` is not the arbiter.** It is a retrieval-conditioning evidence
   selector, and section 1.2 shows its selection boundary deliberately makes a
   different tradeoff than the arbiter requires. Naming it as the prediction
   service would be a false claim about what the crate does.
3. **`StateService` would recreate the dual-owner problem the constitution
   exists to end.** The constitution's section 4 state table gives Fast,
   Adaptive, Regime, Predictive, and Resource state to different owners with
   different authority and different update rules. One `StateService` implies
   one owner for all five, which contradicts constitution section 2 and
   decision 73.
4. **ABI's unit of modularity is the crate and the trait, not the RPC
   service.** `HostBudget`, `PluginManager`, and `SeaSelection` are the house
   style; `-Service` suffixes are not.

**Recommended instead.** A new crate `abi-arbiter` in `dev/active/abi`, above
the substrate, depending on `abi-ai`, `abi-sea`, and `abi-agent-host`, with
concept names that say what they are:

| Concept | Recommended name | CSAPS 6.7 equivalent |
| --- | --- | --- |
| Guild digital twin, this program's slice | `GuildWorldModel` | part of `StateService` |
| Regime label plus posterior | `RegimeEstimate` | part of `StateService` |
| Composed four-axis budget | `ResourceLedger` | part of `StateService` |
| Calibrated uncertainty and surprise | `Calibrator`, `SurpriseEstimate` | `PredictionService` |
| Mode selection result | `ArbiterDecision`, `ExecutionMode` | `ArbiterService` |

The CSAPS names are kept **only** in this cross-reference table, so a reader of
the paper can map the vocabulary without the codebase inheriting a service
taxonomy it does not use.

---

## 9. Dependencies, and what blocks this program

**Current.** The WDBX gap analysis measured every field this program needs to
persist as absent from `V2AuditBlock` (`../wdbx/crates/abi-wdbx/src/v2/types.rs:116`,
8 fields against roughly 28 specified): no `schema_version`, no `task_regime`,
no `state_summary`, no `predicted_outcome` or `observed_outcome`, no
`uncertainty`, `risk`, or `novelty`, no `model_versions`, `policy_version`, or
`calibration_versions`, and no `signer_key_id`. It also measured
`ProposeWrite` and `WriteDecision` as absent from the gateway surface
(`PutVector` and `PutKv` are unconditional), and R6, R7, and R8 as absent.

**Consequence.** This program **cannot reach C5 or above** until
constitution-Program-4 lands the canonical episode envelope, the selective
write gate, and evidence-weighted retrieval. Regime-compatible retrieval in
particular is not a nicety here: without it, the arbiter conditions on evidence
gathered under a different regime and its calibration is meaningless.

**Proposed interim, and one requirement inside it that is easy to get wrong.**
Until the substrate carries these fields, the arbiter writes to a **single**
bounded local decision log, explicitly declared a projection rather than a
canonical store, with the Rust Discord bot's JSON facts remaining canonical as
constitution section 5 already provides. No dual canonical writers (decision
77).

That decision log must carry the full future envelope's fields, **with the same
names, from the first line written**. Calibration is a historical series;
retrofitting `regime`, `predicted_outcome`, `uncertainty`, and the version
triple later means every record written before the retrofit is uncalibratable
and the series restarts. The cost of carrying nullable fields early is small;
the cost of discovering you need them after six months of traffic is the six
months.

Other dependencies:

| Dependency | Provided by | Blocking level |
| --- | --- | --- |
| Structure, authority, workflow, goal, health views | constitution-Program-3 | Blocks C3; the arbiter can be specified and unit-tested without them |
| Typed authorization decisions and consequence class | constitution-Program-2 | Blocks C4; the safety floor must be a real input, not a stub |
| Qualified model manifests with measured latency and calibration | constitution-Program-6, model registry half | Blocks C5; without them the latency and money axes are estimates |
| Episode envelope, write gate, evidence-weighted retrieval | constitution-Program-4 | Blocks C5 |
| Language-neutral contracts | constitution-Program-1 | Blocks any Swift `AbbeyBot` participation |

---

## 10. Acceptance matrix, canary boundary, and rollback

Constitution section 13 requires every program to carry its own design, plan,
gate, canary, and rollback. Thresholds are preregistered before results are
inspected (decision 65), and the arbiter cannot be its sole evaluator
(decision 67): the evaluation harness is a separate component with its own
fixtures and its own review.

### 10.1 Evidence ladder

| Level | What must be demonstrated |
| --- | --- |
| **C0** | This document, plus the preregistered threshold set: `q_k` floors per consequence class, maximum acceptable ECE per bucket, maximum audit divergence per bucket, minimum audit sampling rate per bucket, regime dwell minima, and every revocation trigger in 10.3 |
| **C1** | Unit and property tests: the calibration map is monotone; a budget can never lower `q_k`; safety can only raise required verification depth; every inadmissible mode names its binding constraint (design rule A); the cold-start path never selects M0 at medium or high consequence; the global prior's provenance contains no live guild data; no cross-guild read is reachable; the arbiter is not reachable from the media-gate or cancellation path |
| **C2** | Deterministic replay: a recorded event stream with pinned calibration, policy, price-table, schema, and seed versions reproduces identical `ArbiterDecision` values, including identical named binding constraints |
| **C3** | Offline evaluation on frozen datasets against **all** of: always-M0, always-M1, always-M2, random-at-matched-cost, the current `abbey-bot` deterministic flow, and an offline oracle upper bound. Ablations removing regime, removing surprise, removing calibration (raw scores), and removing the cost term. Metrics: per-bucket reliability, ECE and MCE with intervals, Brier decomposed into reliability, resolution, and uncertainty, and AUC. The random-at-matched-cost baseline is the one that matters: beating always-M2 on cost and always-M0 on quality is not sufficient if spending the same budget at random does as well. Resolution is the term that shows the estimator is doing work at all, since a perfectly calibrated constant predictor has zero resolution and is useless |
| **C4** | Shadow: the arbiter decides and records; the existing deterministic flow acts. Uniquely valuable here because it yields **full counterfactual coverage before any suppression can occur** |
| **C5** | Bounded canary; see 10.2 |
| **C6** | An authorized operator witnesses exact end-to-end outcomes for the canary version |
| **C7** | Sustained operation with drift monitors on every 7.4.3 pair |

No level auto-promotes the next (decision 63). Safety-relevant and
high-consequence promotion requires independent review and human approval.

### 10.2 Canary boundary

- A small, fixed, named set of consenting guilds, including **at least one
  genuinely new guild** so the cold-start path is exercised rather than assumed.
- Low consequence class only. Medium and high stay proposal-only.
- Forced audit sampling **on**, stratified, with its own ledger line. The
  canary is not permitted to run with the audit disabled.
- Per-guild ledger caps set below normal operating limits.
- The existing DQN stays confined to low-risk speech throughout, unchanged
  action space (decision 36).
- No structural writes, no command registration, no moderation.

### 10.3 Revocation triggers

Automatic and non-discretionary. Any one fires a revert to the previous
approved version:

1. Per-bucket ECE above the preregistered threshold for two consecutive
   evaluation windows.
2. Audit divergence rising in any confidence bucket while escalation rate falls
   (the 7.4.3 signature).
3. Any authorization false-allow, at any consequence class.
4. Calibration map staleness past its TTL, or a regime change without
   recalibration.
5. The cost-blind shadow arbiter (7.4.4) diverging past its preregistered
   bound.
6. Any audit sample falling below its per-bucket floor, which means the
   detector itself has stopped working and no claim about calibration is
   supportable while it is down.

### 10.4 Rollback path

The arbiter is additive. Rollback disables `abi-arbiter` and returns every turn
to `abbey-bot`'s existing deterministic pipeline, which continues to work
unchanged because this program composes it rather than replacing it. The
decision log is retained as evidence, including for the failed version
(decision 66: negative and rollback evidence remains publishable). Failed,
revoked, superseded, and expired are first-class states (decision 68), and a
revoked arbiter version is recorded as revoked rather than deleted.

---

## 11. Open questions for Donald

1. **Program numbering.** Amend constitution section 13, or move to name-based
   program references? Section 0 cannot be resolved inside this spec.
2. **`GuildSettings::learning_enabled` migration.** Decision 31 requires
   default-off. Migrating without silently rewriting an existing guild's
   explicit choice needs a distinction between "explicitly set true" and
   "defaulted true", which the current `serde` shape does not preserve. That is
   a schema question for the guild-settings owner, not for the arbiter.
3. **Consequence-class taxonomy.** This spec assumes constitution section 9's
   low, medium, high ladder. The mapping from Discord actions to classes is
   constitution-Program-2's to define, and the `q_k` floors cannot be
   preregistered until it exists.
4. **Crate name.** `abi-arbiter` is a recommendation, not an adoption.
