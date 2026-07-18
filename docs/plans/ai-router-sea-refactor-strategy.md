# Modernization Plan: AI Router + Constitution + SEA Loop

**Scope**: `src/features/ai/router.zig`, `src/features/ai/constitution.zig`,
`src/features/sea/` (`mod.zig`, `learn_loop.zig`, `scorer.zig`, `evidence.zig`,
`query_plan.zig`, `types.zig`).

**Type**: Planning only. No Zig code changes in this pass — three other
in-flight agents are editing this exact area concurrently (SEA learn-loop
validation, a safety review of `router.zig`'s persisted-state hardening, and
an unrelated repo improvement explicitly steered away from `router.zig`).
This document does not touch any `.zig` file.

**Skill applied**: `refactor-strategy` (`.claude/skills/refactor-strategy/SKILL.md`),
Recommended Process steps 1-6.

**Headline finding**: the premise this plan was scoped against is partly
stale. `routeInputAdaptive` — flagged as unreferenced dead code — is **not
present in the current tree** (see §1.4 provenance). The "beyond prefix-only
matching" modern alternative already exists and ships (`routeInputWithSoul` +
`PointNeuralNetwork`, wired to CLI `--soul`). The prefix-only matcher itself is
a deliberate, test-locked fix, not a gap. Net effect: this is mostly a
**leave-alone** module with one small, genuinely low-risk consolidation item
and one explicitly-declined behavior change (constitution gating). See §4 for
the reasoning per sub-area.

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
  (`"selectBestProfile tie-break order is abbey then aviva then abi"`). This
  means genuinely neutral input (no keyword hits) always routes to `abbey`,
  not `abi` as the module-level doc comment in the system prompt background
  suggested — worth correcting in any future doc pass, but out of scope here
  since no source file states the opposite.
- `routeToProfile` / `routeInput`: **live**, not dead code. Reached via
  `mod.zig::run()` (`profile.routeInput`, used by the plain non-adaptive
  `complete`/`run` CLI path) and `mod.zig::runAgent()`
  (`profile.routeToProfile`, used by `agent plan`). Both call through to
  `incremental.generateProfileIncremental` so the one-shot and streaming
  paths render string-identical output.
- `routeInputWithSoul`: blends keyword weights with a 3-output
  `PointNeuralNetwork` softmax via `blend_alpha`, falls back to keyword-only
  if no usable net. Wired to CLI `complete --soul <file> --soul-alpha <a>`
  (`src/cli/handlers/complete_handlers.zig:97`). This is the "beyond
  prefix-only keyword matching" modern design already implemented — no gap
  here.
- `AdaptiveModulator`: EMA smoothing (`alpha`, default 0.3) over
  `ProfileWeights`, `init`/`initWithAlpha`/`update`/`weights`/`serialize`/
  `deserialize`/`loadWeights`/`saveWeights`. `deserialize` was hardened in
  commit `2863658e` ("harden persisted router state", landed via PR #704,
  already merged to this branch) to reject non-finite/negative/out-of-range
  floats, malformed field counts, and out-of-`[0,1]` alpha, falling back to
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
  - `Constitution.evaluateResponse(response, principles)` — a **second,
    unused-in-production** scorer with a different formula (starts at `0.7`,
    `+0.15`/`-0.4` per hit, `<0.5` triggers a per-principle violation) and a
    caller-supplied principle subset. Grep confirms no non-test call site
    currently invokes `evaluateResponse` outside `constitution.zig`'s own
    tests. This dual-scorer design is a genuine design smell (see §3) —
    distinct from anything this plan proposes changing behaviorally.
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
  as an interim state pending a gate. `CLAUDE.md`'s AI Subsystem section
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
  task_fit 0.05 — sums to 1.0). `seaScore` is a plain weighted sum, no
  gating/non-linearity, matching `docs/spec/sea-design-extract.mdx` §5.2 by
  design. `adjustWeightsForTask` applies additive deltas for 3 of 7
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
  MemoryTracker wiring convention from `CLAUDE.md`'s Zig 0.17 Patterns
  section.
- Success criteria today: 4 inline tests in `learn_loop.zig` (tracked
  persistence, evidence recall across turns, task-intent surfacing, saved
  weights bias later routing to a non-default profile), plus router/
  constitution's own inline tests. No dedicated `tests/contracts/*` file
  targets `sea.*` symbols directly — `tests/contracts/surface.zig` only
  touches `ai.constitution.{Principle,AuditResult,Constitution}` (lines
  194-196) and the WDBX store contract; SEA-specific contract coverage lives
  entirely in `learn_loop.zig`'s own inline tests plus (per the task's
  framing) a concurrently-running agent's SEA learn-loop skill validation.
  Any future SEA change should keep the 4 `learn_loop.zig` tests green as the
  closest thing to a contract, and should not assume a `tests/contracts/`
  entry exists to catch a regression.

### 1.4 Provenance check on `routeInputAdaptive` (dead-code claim)

`git log -S 'routeInputAdaptive' --oneline` shows it was introduced in
`f6285504` ("consolidate ABI Zig 0.17 modernization") and last touched in
`24e64c32` ("wave-5 file splits and dedup cleanup"); it does not appear in the
current `router.zig` at all (confirmed by direct read + `grep -rn
routeInputAdaptive src/ tests/` returning nothing) and `mod.zig` does not
re-export it. It appears to have been removed as part of, or before, the
concurrently-landed `2863658e` router-state-hardening commit. **Action for
this plan: none.** The "should `routeInputAdaptive` be removed or wired in"
question the task posed is already resolved — it is gone, and the live
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
- **Constitution**: a single scoring path (retire the redundant
  `evaluateResponse`/`scorePrinciple`), a `CheckResult{ principle, matched,
  is_negative }` output for explainability, and a clearly labeled
  "advisory only" contract at the type level (e.g. rename `passed` to
  something less gate-suggestive, or add a doc comment pinning the
  observability-only guarantee next to the field). No change to the
  hardcoded-substring approach without a real NLP/classifier upgrade, which
  is out of scope (no such capability exists in the repo and none is
  claimed).
- **SEA**: task-aware weight adjustment covering all 7 `TaskType`s instead of
  3, and a `tests/contracts/sea_contract.zig` (or equivalent) that pins the
  public `sea.*` surface the way `surface.zig` does for CLI/MCP, so a future
  refactor has an explicit regression net beyond `learn_loop.zig`'s inline
  tests.

---

## 3. Gap analysis (current vs. ideal)

| Area | Gap | Size | Risk if changed |
|---|---|---|---|
| Constitution dual scorer | `evaluateResponse`/`scorePrinciple` is unused in production, diverges numerically from `validate`/`scorePrinciple`(inline) | Small (delete or merge ~45 lines + its 2 tests, or leave as an explicitly-labeled alternate scorer) | Low — no call site depends on it; only risk is deleting tests that exercise a real (if unused) behavioral contract |
| Constitution "observability-only" | None — deliberate, documented, correctly implemented per CLAUDE.md/docs | N/A | Promoting to a gate is a **product decision**, explicitly declined for this plan (see §4.2) |
| Router tie-break docs | Module-level framing elsewhere (this task's own background context) implies neutral input routes to `abi`; source says `abbey` wins ties | Doc-only | Low — no source change; a docs sync pass should correct any external doc that says otherwise |
| SEA task-aware weights | `adjustWeightsForTask` only covers 3 of 7 `TaskType`s | Small-medium (extend one `if` chain + add test cases per new task) | Low-medium — weight tuning without a labeled eval set risks silently shifting retrieval quality; needs before/after `learn_loop.zig` test additions, not just code |
| SEA contract test coverage | No `tests/contracts/*` pins `sea.*` public surface | Medium (new test file) | Low to add, but scope creep risk if it tries to also pin behavior beyond signatures |
| `routeInputAdaptive` | None — already removed | N/A | N/A |
| Prefix-only sentiment matching | None — deliberate, test-locked | N/A | Any change is optional and must keep the 6 existing test assertions green |

---

## 4. Strategy per sub-area

### 4.1 Router — **leave alone**

No functional gap. The prefix-only matcher and `abbey`-wins-ties behavior are
both test-locked deliberate choices; `routeInputWithSoul` already delivers
the "beyond keyword matching" capability via `--soul`. The only
recommended action is a **doc-only** correction pass (not code) if any
checked-in doc claims ties resolve to `abi` — grep
`docs/spec/*.mdx` for that claim before editing; if none exists, there is
nothing to do here at all.

### 4.2 Constitution — **preserve observability-only status explicitly; small optional consolidation on the dead second scorer**

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
- **Optional, small, low-risk**: consider retiring
  `evaluateResponse`/`scorePrinciple` (or clearly re-labeling them as an
  "alternate/experimental scorer, not on the production path" in a doc
  comment) since they diverge from `validate` and are unused outside their
  own tests. This is a direct-rewrite-sized cleanup (skill's "small/low-risk
  -> direct rewrite" bucket), not a phased effort — but it is optional
  housekeeping, not a correctness fix, and should only be picked up once the
  concurrent router.zig safety-review agent's work has landed and the file
  is not mid-edit.

### 4.3 SEA loop — **incremental, two independently-shippable slices**

1. Extend `adjustWeightsForTask` to cover the remaining 4 `TaskType`s
   (general, implementation_design, legal_review, research_synthesis) with
   additive deltas analogous to the existing 3, each backed by a new
   `learn_loop.zig` (or `scorer.zig`) test asserting the expected weight
   shift and, where feasible, an end-to-end assertion that the shift changes
   which evidence gets admitted for a representative input. Low risk,
   incremental — ship one `TaskType` at a time or all four together, either
   is safe since they are independent `if` branches.
2. Add a `tests/contracts/` entry (or extend `feature_modules.zig`) pinning
   the public `sea.*` surface (`@hasDecl` checks + a couple of behavioral
   assertions on `runLearnLoop`/`seaScore`/`selectSeaCandidates`), mirroring
   the `surface.zig` pattern already used for CLI/MCP/constitution. This is
   additive test-only work with no behavior change, so it can land
   independently and first, ahead of slice 1, to give slice 1 a stronger
   regression net.

### 4.4 `routeInputAdaptive` — **no action; already resolved**

Confirmed absent from the tree (§1.4). No milestone needed.

---

## 5. Validation criteria per phase

| Phase | Validation |
|---|---|
| Constitution dual-scorer cleanup (if picked up) | `./build.sh check`; keep or consciously retire the 2 `evaluateResponse`-specific tests (`"constitution evaluateResponse scores principles"`, `"constitution evaluateResponse empty response fails all"`); `zig build check-parity` (constitution has no mod/stub split of its own, but confirm no `ai/stub.zig` reference to the removed decls) |
| SEA task-weight extension | `./build.sh check`; new tests per added `TaskType` branch in `scorer.zig`/`learn_loop.zig`; re-run existing `learn_loop.zig` 4 tests unchanged (no regression on code_repair/project_recall/benchmark_review paths); `zig build test-contracts` if a new contract file is added |
| SEA contract test addition | `./build.sh check`; `zig build test-contracts`; confirm the new file follows the `surface.zig` `@hasDecl`-first pattern before adding behavioral assertions |
| Router / Constitution-gating | No phase — explicitly declined (see §4.1, §4.2); if a human later reverses that decision, treat it as a new design doc under `docs/superpowers/specs/`, not an extension of this plan |

All phases assume `./build.sh check` as the primary gate per `CLAUDE.md`, run
only once any phase actually touches `.zig` files (this planning pass does
not).

---

## 6. Milestones and Definition of Done

- **M0 (this document)** — Read-only planning pass. DoD: this plan committed
  under `docs/plans/`, zero `.zig` file changes, findings grounded in current
  source (verified via direct reads + `git log -S` provenance, not assumed
  from the task's background context).
- **M1 (optional, small)** — Constitution dual-scorer consolidation. DoD:
  `evaluateResponse`/`scorePrinciple` either removed with its 2 tests
  consciously retired, or re-labeled via doc comment as non-production;
  `./build.sh check` green; no change to `validate`'s behavior or any of the
  escore/veto tests.
- **M2 (incremental)** — SEA task-aware weights for the remaining 4
  `TaskType`s. DoD: all 7 `TaskType`s have an explicit `adjustWeightsForTask`
  branch (or a documented decision that "general" intentionally stays at
  baseline); new tests per branch; existing 4 `learn_loop.zig` tests
  unchanged; `./build.sh check` green.
- **M3 (additive, test-only, can precede M2)** — SEA contract test coverage.
  DoD: a new (or extended) contract file exists that `@hasDecl`-checks the
  `sea.*` public surface (mirroring `surface.zig`'s pattern for
  `ai.constitution`); `zig build test-contracts` green; no behavior change.
- **Explicitly not a milestone**: promoting the constitution audit from
  observability-only to a hard gate. Recorded here as a deliberately declined
  item so it is not silently re-proposed as a "gap" by a future planning
  pass without a human product decision first.

---

## Appendix: verification commands run for this plan (read-only)

```
git log -S 'routeInputAdaptive' --oneline -- src/features/ai/router.zig
grep -rn "routeInputAdaptive" src/ tests/
grep -rln "routeInput\b|routeToProfile|abbey\.processInput|aviva\.processInput|abi_profile\.processInput" src/
grep -rln "generateProfileIncremental" src/
grep -rln "Constitution\." src/
grep -n "audit_passed|audit_escore|audit_vetoed|Constitution\." src/features/ai/completion.zig
grep -n "constitution|gate|veto|observability" tasks/todo.md tasks/goals.md docs/spec/*.mdx
```

No `.zig` file was modified to produce this plan.
