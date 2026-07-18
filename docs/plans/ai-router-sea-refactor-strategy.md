# Modernization Plan: AI Router + Constitution + SEA Loop

**Scope**: `src/features/ai/router.zig`, `src/features/ai/constitution.zig`,
`src/features/sea/` (`mod.zig`, `learn_loop.zig`, `scorer.zig`, `evidence.zig`,
`query_plan.zig`, `types.zig`).

**Status**: Strategy plus one bounded correctness fix discovered during source
verification: `routeInputWithSoul` now preserves keyword routing when its
neural network is absent or its output shape/value is rejected, and SEA's public surface/degraded
behavior now has contract coverage. Weight-tuning milestones remain
unimplemented.

**Skill applied**: `refactor-strategy` (`.agents/skills/refactor-strategy/SKILL.md`),
Recommended Process steps 1-6.

**Headline finding**: the premise this plan was scoped against is partly
stale. `routeInputAdaptive` — flagged as unreferenced dead code — is **not
present in the current tree** (see §1.4 provenance). The "beyond prefix-only
matching" modern alternative already exists and ships (`routeInputWithSoul` +
`PointNeuralNetwork`, wired to CLI `--soul`). The prefix-only matcher itself is
a deliberate, test-locked fix, not a gap. Net effect: this is mostly a
**leave-alone** module with one bounded router fallback correction, two
independent SEA test/tuning candidates, and an explicitly-declined behavior
change (constitution gating). See §4 for the reasoning per sub-area.

---

## 1. Current behavior and success criteria

### 1.1 Router (`src/features/ai/router.zig`, ~510 lines incl. tests)

- `ProfileWeights{ w_abbey, w_aviva, w_abi }` + `SentimentKeyword` table (19
  single-word entries) drive `analyzeSentiment`.
- `analyzeSentiment(input)`: whitespace-splits `input`, trims trailing
  punctuation per token, and does a **prefix-only, case-insensitive,
  single-token** match (`startsWithIgnoreCase`) against the keyword table,
  accumulating `+kw.score * 0.1` per hit, then normalizes. This is a
  deliberate fix (test: `"analyzeSentiment ignores suffix false positives but
  keeps prefix stems"`) that intentionally keeps `quickly`->`quick` and
  `running`->`run` matching while rejecting `overrun`->`run`,
  `unsafe`->`safe`, `prefix`->`fix`, `redesign`->`design`. Any future change
  to the matching strategy must keep that test's four negative + two positive
  cases green.
- `selectBestProfile(weights)`: ties resolve `abbey > aviva > abi`
  (`>=`/`>=` comparison order), test-locked
  (`"selectBestProfile tie-break order is abbey then aviva then abi"`).
  Neutral input is not a tie: `analyzeSentiment` starts from
  `{abbey=0.33,aviva=0.33,abi=0.34}`, so no-match input routes to `abi`.
- `routeToProfile` / `routeInput`: **live**, not dead code. Reached via
  `mod.zig::run()` (`profile.routeInput`, used by the plain non-adaptive
  `complete`/`run` CLI path) and `mod.zig::runAgent()`
  (`profile.routeToProfile`, used by `agent plan`). Both call through to
  `incremental.generateProfileIncremental` so the one-shot and streaming
  paths render string-identical output.
- `routeInputWithSoul`: blends keyword weights with a 3-output
  `PointNeuralNetwork` softmax via `blend_alpha`; it retains keyword weights
  when the network is absent or its output shape/value is rejected, while
  `forward()` errors propagate. Wired to CLI `complete --soul <file> --soul-alpha <a>`
  (`src/cli/handlers/complete_handlers.zig`). Source verification found and
  corrected a mismatch in this fallback: the old code blended against neutral
  defaults when the network was absent, so `alpha=1` could change the selected
  profile. The implementation now initializes the neural side from the keyword
  decision and uses a finite, numerically stable softmax when logits are usable.
- `AdaptiveModulator`: EMA smoothing (`alpha`, default 0.3) over
  `ProfileWeights`, `init`/`initWithAlpha`/`update`/`weights`/`serialize`/
  `deserialize`/`loadWeights`/`saveWeights`. `deserialize` hardening landed via
  PR #704 and rejects non-finite/negative weights, non-positive or
  non-finite totals, malformed fields/counts, invalid `u32` update counts,
  and alpha outside `[0,1]`, falling back to
  `AdaptiveModulator.init()` on any parse failure — covered by 17
  malformed-input cases in
  `"AdaptiveModulator rejects invalid persisted state deterministically"`
  plus boundary/overflow-saturation tests. This hardening is **done**, not a
  gap this plan needs to propose.
- Live adaptive path: `completeAdaptive`/`completeWithStoreAdaptive`
  (`src/features/ai/completion.zig`) via `sea.runLearnLoop`
  (`src/features/sea/learn_loop.zig`), which calls
  `AdaptiveModulator.loadWeights` / `.update(router.analyzeSentiment(input))`
  / `.saveWeights` directly — not through any `router.zig`-level
  "adaptive routing" wrapper function.

### 1.2 Constitution (`src/features/ai/constitution.zig`, ~370 lines incl. tests)

- 6 `Principle`s (truthfulness, safety, helpfulness, fairness, privacy,
  transparency), each with a `specAlias()` mapping to the WDBX 4-pillar
  governance vocabulary (autonomy/non-maleficence/beneficence/justice).
- `PRINCIPLE_CHECKS`: 13 hardcoded case-insensitive **substring** patterns (7
  negative, 6 positive), e.g. `"harm"` (negative) fires on `"harmless"`;
  `"unsafe"`, `"password"`, `"personal data"`, `"discriminate"`,
  `"cannot help"`, `"definitely false"` are the other negatives. This is a
  crude, hardcoded-vocabulary matcher by design — it cannot detect novel harm
  patterns, only lexical presence of these 7 strings.
- `PRINCIPLE_WEIGHTS` (sums to 1.0): truthfulness 0.20, safety 0.175,
  helpfulness 0.125, fairness 0.20, privacy 0.175, transparency 0.125,
  mapped from the WDBX E = alpha*Autonomy + beta*Non-Maleficence +
  gamma*Beneficence + delta*Justice formula.
  `SAFETY_CLASS = {safety, privacy}` with `SAFETY_VETO_THRESHOLD = 0.5`: if
  either safety-class principle scores below 0.5, `AuditResult.vetoed = true`
  and `passed` is forced `false` regardless of the weighted `escore` — the
  "hard safety veto." Well covered by tests (clean-response escore=1.0,
  safety/privacy veto cases, fairness-violates-without-vetoing case,
  idempotent `finalize()`).
- **Two divergent scoring paths** exist side by side:
  - `Constitution.validate(response)` — the one actually used by
    `mod.zig::run()`/`runAgent()`/`completion.zig`. Binary per-check
    contribution: negative hit -> score forced to `0.0` + `passed=false`;
    positive hit -> `score += 0.2` (capped at 1.0), starting from `1.0`.
  - `Constitution.evaluateResponse(response, principles)` — an **alternate
    public diagnostic** scorer with a different formula (starts at `0.7`,
    `+0.15`/`-0.4` per hit, `<0.5` triggers a per-principle violation) and a
    caller-supplied principle subset. It is covered both by
    `constitution.zig` inline tests and `src/integration_tests.zig`; it is not
    the completion-delivery path, but it is a tested API contract and must not
    be treated as dead code.
- **Consumption today is intentionally two-tier, not purely inert**:
  - `mod.zig::run()`: `std.log.warn`s on `!audit.passed`, still returns the
    response unchanged. Pure observability.
  - `mod.zig::runAgent()` (backs `agent plan`): folds `!audit.passed` into
    `requires_review = !config.dry_run or !audit.passed` — i.e. the audit
    *does* influence a surfaced field (whether human review is flagged as
    required) even though it never blocks the response body from being
    returned. This nuance matters: "observability-only" is accurate for
    response delivery, but the audit result already participates in one
    decision surface (`review_required`).
  - `completion.zig`: persists `audit_passed`/`audit_vetoed`/`escore` into
    completion metadata and surfaces them via CLI/MCP fields
    (`audit_passed`, `audit_escore`, `audit_vetoed` per
    `docs/spec/abi-refactor-design.mdx` §5.3).
- **Product-decision signal check** (per skill step 2's requirement to check
  `docs/spec/`/`tasks/todo.md` before proposing a change): no entry in
  `tasks/todo.md` or `tasks/goals.md` requests promoting the audit to a hard
  gate. `docs/spec/abi-refactor-design.mdx` §5.3 documents current behavior
  ("Post-generation validation... produces `AuditResult`") without framing it
  as an interim state pending a gate. `AGENTS.md`'s AI Subsystem section
  states the observability-only design explicitly and gives the underlying
  reason (case-insensitive substring matching can't reliably block on).
  Conclusion: this is a **deliberate, documented product decision**, not an
  overlooked gap. See §4.2 for the explicit recommendation to preserve it.

### 1.3 SEA loop (`src/features/sea/`)

- `mod.zig` re-exports a stable public surface: evidence gathering
  (`EvidenceItem`/`EvidenceContext`/`gatherEvidence`/`gatherEvidenceWithPlan`/
  `augmentPrompt`), query planning (`QueryPlan`/`TaskType`/`inferQueryPlan`),
  8-signal scoring (`SeaSignals`/`SeaWeights`/`SeaCandidate`/`SeaOptions`/
  `SeaSelection`/`DEFAULT_SEA_WEIGHTS`/`seaScore`/`adjustWeightsForTask`/
  `selectSeaCandidates`/`contextPack`), and the orchestrating
  `LearnLoopConfig`/`LearnLoopResult`/`runLearnLoop`.
- `scorer.zig`: 8 orthogonal signals (semantic 0.30, keyword 0.15, metadata
  0.15, recency 0.10, authority 0.10, graph 0.10, contradiction 0.05,
  task_fit 0.05 — sums to 1.0). `seaScore` is a weighted sum clamped to
  `[0,1]`, with no learned transform or threshold gate, matching
  `docs/spec/sea-design-extract.mdx` §5.2. `adjustWeightsForTask` applies
  additive deltas for 3 of 7
  `TaskType`s (code_repair, project_recall, benchmark_review); the other 4
  (general, implementation_design, legal_review, research_synthesis) get no
  adjustment today — a real, disclosed gap, not a bug (see §3).
  `selectSeaCandidates` is budgeted greedy: sorts by `final_score` desc,
  admits until token/record/per-cluster caps, with an escape hatch
  (score >= 0.92) that bypasses the per-cluster cap.
- `learn_loop.zig::runLearnLoop`: infers a `QueryPlan` once, gathers evidence
  with that plan, augments the prompt (capped at `evidence.MAX_PROMPT_BYTES`),
  loads `AdaptiveModulator` weights from the store, calls
  `ai.completeWithStoreAdaptive`, then (if `config.adapt_router`) updates and
  saves the modulator weights — with a `TrackingAllocator`-routed transient
  buffer when a `MemoryTracker` is supplied, and a save failure downgraded to
  `adapted=false` + `std.log.warn` (never discards the caller-owned
  completion, never silent-`catch{}`s). This directly implements the
  MemoryTracker wiring convention from `AGENTS.md`'s Zig 0.17 Patterns
  section.
- Success criteria include module-local tests across `learn_loop.zig`,
  `scorer.zig`, `query_plan.zig`, `evidence.zig`, and `types.zig`, plus public
  contract coverage in `tests/contracts/feature_modules.zig`. The contract
  pins exported SEA declarations, enabled scoring, disabled zero-score
  degradation, and owned empty-selection behavior. Future changes must keep
  both this public contract and the module-local behavioral tests green.

### 1.4 Provenance check on `routeInputAdaptive` (dead-code claim)

`git log -S 'routeInputAdaptive' --oneline` shows it was introduced in
`f6285504` ("consolidate ABI Zig 0.17 modernization") and last touched in
`24e64c32` ("wave-5 file splits and dedup cleanup"), where it was removed. It
does not appear in the current `router.zig`, and `mod.zig` does not re-export
it. **Action for this plan: none.** The historical "should
`routeInputAdaptive` be removed or wired in" question is resolved — it is gone,
and the live
adaptive path (`completeAdaptive`/`completeWithStoreAdaptive` via
`runLearnLoop`) already covers the same need through `AdaptiveModulator`
directly. Do not reintroduce it.

---

## 2. Ideal modern design (sketch)

Written as "what would this look like designed fresh today," for gap
comparison only — not a rewrite proposal:

- **Router**: a single `SentimentAnalyzer` interface with two swappable
  strategies (keyword/prefix-match, neural/soul-blend) behind one call site,
  explicit `RoutingDecision{ profile, weights, source: enum{keyword,neural,blended} }`
  return type for observability, and tie-break behavior as an explicit,
  documented policy rather than an implicit `>=`/`>=` chain. **Reality
  check**: this is close to what already exists — `analyzeSentiment` +
  `routeInputWithSoul` + `blendWeights` already provide the two strategies
  and a blend knob; the only missing piece is a unified return type that
  states *which* strategy decided, which is a nice-to-have, not a
  functional gap.
- **Constitution**: preserve both public scoring contracts while documenting
  their distinct roles: `validate` audits a complete generated response,
  whereas `evaluateResponse` scores a caller-selected principle subset for
  diagnostics. A future internal extraction could share pattern-evaluation
  mechanics, but changing either formula or removing either API is a behavior
  change with inline and integration coverage. A `CheckResult{ principle,
  matched, is_negative }` output could improve explainability, and the
  observability-only delivery contract should remain explicit. No change to
  the hardcoded-substring approach without a real NLP/classifier upgrade,
  which is out of scope.
- **SEA**: task-aware weight adjustment covering all 7 `TaskType`s instead of
  3, and a `tests/contracts/sea_contract.zig` (or equivalent) that pins the
  public `sea.*` surface the way `surface.zig` does for CLI/MCP, so a future
  refactor has an explicit regression net beyond `learn_loop.zig`'s inline
  tests.

---

## 3. Gap analysis (current vs. ideal)

| Area | Gap | Size | Risk if changed |
|---|---|---|---|
| Constitution scoring modes | `validate` and `evaluateResponse` intentionally expose different inputs/formulas; their role distinction needs explicit documentation | Doc-only | Removing or numerically merging them would break tested behavior, including `src/integration_tests.zig` |
| Constitution "observability-only" | None — deliberate, documented, correctly implemented per AGENTS.md/docs | N/A | Promoting to a gate is a **product decision**, explicitly declined for this plan (see §4.2) |
| Router tie-break vs. neutral default | Equal weights resolve to `abbey`, while the non-equal neutral baseline routes to `abi` | None after this correction | Keep the true three-way tie test and neutral baseline distinct |
| SEA task-aware weights | `adjustWeightsForTask` only covers 3 of 7 `TaskType`s | Small-medium (extend one `if` chain + add test cases per new task) | Low-medium — weight tuning without a labeled eval set risks silently shifting retrieval quality; needs before/after `learn_loop.zig` test additions, not just code |
| SEA contract test coverage | Completed in `tests/contracts/feature_modules.zig` | None after this branch | Keep enabled/disabled behavior and owned-slice cleanup green |
| `routeInputAdaptive` | None — already removed | N/A | N/A |
| Prefix-only sentiment matching | None — deliberate, test-locked | N/A | Any change is optional and must keep the 6 existing test assertions green |

---

## 4. Strategy per sub-area

### 4.1 Router — **leave alone**

The prefix-only matcher, `abbey`-wins-equal-ties behavior, and
`abi`-wins-neutral-baseline behavior are test-locked deliberate choices.
`routeInputWithSoul` delivers the "beyond keyword matching" capability via
`--soul`; its null-network fallback is test-locked to the keyword decision,
while guarded output-shape/value rejection preserves the same fallback by
construction. Network `forward()` errors still propagate.

### 4.2 Constitution — **preserve observability-only delivery and both scoring APIs**

- **Preserve, do not gate.** Recommend explicitly declining to promote the
  audit from observability to a hard gate. Rationale to record in this plan
  (not just assumed): `PRINCIPLE_CHECKS` is case-insensitive **substring**
  matching against 7 hardcoded negative strings (`"harm"` matches
  `"harmless"`, `"unsafe"` matches nothing narrower, etc.) — it has a high
  false-positive rate and zero recall against any wording it wasn't
  hand-authored to catch. Turning that into a hard block on `complete`/`run`
  would (a) silently drop legitimate responses that happen to contain
  "harm" as a substring of a benign word, and (b) implicitly claim a safety
  guarantee the mechanism cannot back — which conflicts with this repo's
  external-claims discipline (`docs/contracts/external-claims-audit.mdx`).
  This is a **product decision**, not a code gap, and no `tasks/todo.md` /
  `tasks/goals.md` entry asks for it — so it is out of scope for any
  "safe next slice," now or later, without an explicit human product
  decision to accept that trade-off.
- **Preserve both scoring APIs.** `evaluateResponse` is an alternate diagnostic
  surface with caller-selected principles and integration coverage, not dead
  code. A future doc-comment pass may clarify how it differs from `validate`;
  formula convergence or API removal requires an explicit behavior decision
  and updates to both inline and integration tests.

### 4.3 SEA loop — **incremental, two independently-shippable slices**

1. Extend `adjustWeightsForTask` to cover the remaining 4 `TaskType`s
   (general, implementation_design, legal_review, research_synthesis) with
   additive deltas analogous to the existing 3, each backed by a new
   `learn_loop.zig` (or `scorer.zig`) test asserting the expected weight
   shift and, where feasible, an end-to-end assertion that the shift changes
   which evidence gets admitted for a representative input. Low risk,
   incremental — ship one `TaskType` at a time or all four together, either
   is safe since they are independent `if` branches.
2. **Completed:** `tests/contracts/feature_modules.zig` pins the public SEA
   declarations plus enabled/disabled `seaScore` and empty-selection ownership
   behavior. This landed without changing scoring policy.

### 4.4 `routeInputAdaptive` — **no action; already resolved**

Confirmed absent from the tree (§1.4). No milestone needed.

---

## 5. Validation criteria per phase

| Phase | Validation |
|---|---|
| Router null-network fallback (landed with this plan) | `./build.sh test -Dtest-filter="routeInputWithSoul preserves keyword routing"`; `./build.sh check`; the public `routeInputWithSoul` signature and mod/stub declaration set remain unchanged |
| Constitution scoring-mode documentation (optional) | Keep both inline `evaluateResponse` tests and the `src/integration_tests.zig` coverage; any formula change requires explicit behavioral assertions for both modes |
| SEA task-weight extension | `./build.sh check`; new tests per added `TaskType` branch in `scorer.zig`/`learn_loop.zig`; re-run existing `learn_loop.zig` 4 tests unchanged (no regression on code_repair/project_recall/benchmark_review paths); `zig build test-contracts` if a new contract file is added |
| SEA contract test addition (landed) | `./build.sh test-contracts`; `./build.sh test-contracts -Dfeat-sea=false`; `./build.sh check` |
| Router / Constitution-gating | No phase — explicitly declined (see §4.1, §4.2); if a human later reverses that decision, treat it as a new design doc under `docs/superpowers/specs/`, not an extension of this plan |

All phases assume `./build.sh check` as the primary gate per `AGENTS.md`.

---

## 6. Milestones and Definition of Done

- **M0 (landed with this plan)** — Plan plus bounded fallback correction. DoD:
  this plan is committed under `docs/plans/`; a null Soul network preserves keyword
  routing; guarded invalid output shape/values retain that fallback, while forward
  errors propagate; the true equal-weight tie and neutral baseline remain distinct and
  tested; findings are grounded in current source and history.
- **M1 (incremental)** — SEA task-aware weights for the remaining 4
  `TaskType`s. DoD: all 7 `TaskType`s have an explicit `adjustWeightsForTask`
  branch (or a documented decision that "general" intentionally stays at
  baseline); new tests per branch; existing 4 `learn_loop.zig` tests
  unchanged; `./build.sh check` green.
- **M2 (✅ landed, additive/test-only)** — SEA contract coverage now
  `@hasDecl`-checks the public surface and exercises enabled/disabled scoring
  plus empty-selection ownership. Both contract configurations and the full
  gate are required to remain green.
- **Explicitly not a milestone**: promoting the constitution audit from
  observability-only to a hard gate. Recorded here as a deliberately declined
  item so it is not silently re-proposed as a "gap" by a future planning
  pass without a human product decision first.

---

## Appendix: source-audit commands

```bash
git log -S 'routeInputAdaptive' --oneline -- src/features/ai/router.zig
rg -n "routeInputAdaptive" src tests
rg -n "routeInput|routeToProfile|generateProfileIncremental" src/features/ai
rg -n "Constitution\\.|audit_(passed|escore|vetoed)" src/features/ai
./build.sh test -Dtest-filter="routeInputWithSoul preserves keyword routing"
```
