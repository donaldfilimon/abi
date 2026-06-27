# SEA Design Extract: Memory, QueryPlan, Authority, and Sparse Evidence Attention

**Date:** 2026-06-27
**Status:** **DESIGN REFERENCE / PROPOSED — not a statement of current ABI capabilities.**
**Scope:** A distilled, source-grounded record of the SEA (Sparse Evidence Attention),
`QueryPlan`, `Authority`, and `MemoryKind` designs as expressed in the
`abbey_wdbx_sea_zig` reference repository, captured so the clearest articulation of
those designs survives independently of that repo.

> **Claims discipline (read first).** This document describes *design intent* extracted
> from a separate reference implementation; it is **not** a description of what ABI's
> shipped code does today. The reference repo (`~/abbey_wdbx_sea_zig`) is a compact
> teaching implementation that **does not currently compile** on the pinned Zig
> toolchain — its source predates `std.Io` and uses removed `std.posix` symbols. Its
> value is the clarity of the type and algorithm designs, not runnable behavior. Nothing
> here asserts performance, accuracy, retrieval-quality, or production-readiness
> numbers. ABI's own SEA surface (`src/features/sea/`, the `feat-sea` flag, **disabled by
> default**) is a *different and simpler* Phase-1 design; the relationship is mapped in
> §7. Where this design and ABI source disagree, ABI source wins (`build.zig`, `src/`,
> `tests/contracts/`). See `docs/contracts/external-claims-audit.md` and the
> `tests/contracts/public_docs.zig` claim-boundary test.

---

## 1. Why capture this

`abbey_wdbx_sea_zig` is the reference surface for the Abbey / WDBX / SEA / ABI
architecture (`docs/spec/wdbx-north-star.md` describes the long-term ABI direction; this
file records the *retrieval-side evidence-selection* design from the reference repo). The
reference repo packs the whole loop into two files — `src/root.zig` (types + algorithms)
and `src/main.zig` (a small CLI) — which makes its enum and scoring designs unusually
legible. This extract preserves those designs in prose plus small quoted snippets so the
intent is recoverable after the reference repo is archived.

The four designs worth keeping:

1. **`MemoryKind`** — a typed taxonomy of durable memory records.
2. **`Authority`** — a provenance/trust ladder with a numeric trust mapping.
3. **`QueryPlan`** (+ `TaskType`) — an inferred, structured intent for a query.
4. **SEA** — Sparse Evidence Attention: a multi-signal scorer plus a budgeted greedy
   selection that picks a *sparse* high-value evidence set before a model call.

---

## 2. `MemoryKind` — typed memory taxonomy (Proposed design)

A record's *kind* classifies what sort of durable knowledge it carries. The reference
design enumerates nine kinds as a `u8`-backed enum:

```zig
pub const MemoryKind = enum(u8) {
    note,
    user_preference,
    project_decision,
    code_fact,
    tool_output,
    benchmark,
    constraint,
    contradiction,
    summary,
};
```

Design intent:

- The kind is *not* cosmetic — it feeds three later stages: task-fit scoring (§5.4),
  the cluster-diversity budget (the cluster id is just `@intFromEnum(kind)`, §6), and an
  importance default at insert time (`constraint` records, like `system_pinned`
  authority, default to importance `1.0` instead of `0.5`).
- `contradiction` is a first-class kind so conflicting evidence can be deliberately
  surfaced rather than silently outranked; it carries its own scoring signal (§5.4).
- Each enum exposes `parse(s)` (string → optional enum, by tag-name match) and
  `text()` (= `@tagName`), so the same names round-trip through the CLI and JSONL
  persistence.

## 3. `Authority` — provenance ladder with trust mapping (Proposed design)

`Authority` records *where a fact came from* and maps that provenance to a scalar trust
weight. Five rungs, ascending in trust:

```zig
pub const Authority = enum(u8) {
    inferred,        // 0.30
    user_stated,     // 0.78
    tool_verified,   // 0.86
    file_verified,   // 0.90
    system_pinned,   // 1.00
};
```

The numeric mapping is the `score()` method (the right column above). Design intent:

- `score()` is reused two ways: as the record's stored `trust` at insert time, and
  directly as the `authority_score` signal in SEA (§5.4).
- The ladder encodes a clear policy — model *inference* is the least trusted, an
  explicit *user statement* is trusted well above inference, *tool* and *file*
  verification rank higher still, and a *system-pinned* invariant is absolute (`1.00`).
- Like `MemoryKind`, it offers `parse` / `text` for CLI + persistence round-tripping.

## 4. `QueryPlan` + `TaskType` — inferred structured intent (Proposed design)

Before retrieval, a free-text query is turned into a structured plan. `TaskType`
enumerates the recognized retrieval intents:

```zig
pub const TaskType = enum(u8) {
    general,
    implementation_design,
    code_repair,
    legal_review,
    research_synthesis,
    project_recall,
    benchmark_review,
};
```

`QueryPlan` is the plan shape — a small struct of tuning knobs with sane defaults:

```zig
pub const QueryPlan = struct {
    task: TaskType = .general,
    project: ?[]const u8 = null,
    query: []const u8,
    entities: []const []const u8 = &.{},
    require_grounding: bool = true,
    exact_recall: bool = false,
    recency_bias: f32 = 0.40,
    risk: f32 = 0.50,
};
```

Design intent:

- **`infer(allocator, query)`** is a deterministic keyword heuristic (no model call): it
  lowercases the query and sets `task` by substring matches — e.g. `"zig"`, `"code"`,
  `"build.zig"` → `implementation_design`; `"bug"`, `"patch"`, `"compile"` →
  `code_repair`; `"contract"`, `"legal"` → `legal_review`; `"paper"`, `"benchmark"`,
  `"eval"` → `research_synthesis`; `"remember"`, `"prior"`, `"decision"` →
  `project_recall`. Later matches overwrite earlier ones (last-match-wins ordering).
- A `project_recall` task additionally sets `exact_recall = true`, which later re-weights
  SEA toward authority/keyword and away from semantic similarity (§5.3).
- `require_grounding` (default true) expresses the intent that answers must be backed by
  selected evidence; `recency_bias` and `risk` are reserved tuning scalars.

## 5. SEA — Sparse Evidence Attention scoring (Proposed design)

SEA's premise: instead of dumping all candidate memories into a model's context, *score*
each candidate across several orthogonal signals, combine them under task-aware weights,
and select a **sparse** high-value subset under explicit budgets. "Attention" here is
WDBX-side evidence routing over durable memory — distinct from a model's internal
attention (the reference repo is explicit that model-internal sparse attention, "SSA", is
a different thing).

### 5.1 The eight signals

Each candidate (`SeaCandidate`) carries eight independent `[0,1]`-style sub-scores plus
bookkeeping (`estimated_tokens`, `cluster_id`, the computed `final_score`):

| Signal | Source (§5.4) | Captures |
| --- | --- | --- |
| `semantic` | cosine of query vs. record embedding | meaning similarity |
| `keyword` | fraction of ≥3-char query tokens present in the record | lexical overlap |
| `metadata` | project match + kind bonus + importance | structural relevance |
| `recency` | age-decay over `updated_ns` | freshness |
| `authority` | `Authority.score()` | provenance/trust |
| `graph` | supersedes/not-superseded/source/tags presence | record "well-connectedness" |
| `contradiction` | `1.0` iff kind is `contradiction` | surfaces conflicts |
| `task_fit` | `(TaskType × MemoryKind)` table | intent alignment |

### 5.2 The weight vector and combination

Default signal weights (`SeaWeights`) sum to `1.0`:

```zig
pub const SeaWeights = struct {
    semantic: f32 = 0.30,
    keyword: f32 = 0.15,
    metadata: f32 = 0.15,
    recency: f32 = 0.10,
    authority: f32 = 0.10,
    graph: f32 = 0.10,
    contradiction: f32 = 0.05,
    task_fit: f32 = 0.05,
};
```

The combiner is a plain weighted sum, clamped to `[0,1]`:

```zig
pub fn seaScore(c: SeaCandidate, w: SeaWeights) f32 {
    const raw = c.semantic_score * w.semantic +
        c.keyword_score * w.keyword +
        c.metadata_score * w.metadata +
        c.recency_score * w.recency +
        c.authority_score * w.authority +
        c.graph_score * w.graph +
        c.contradiction_score * w.contradiction +
        c.task_fit_score * w.task_fit;
    return clamp01(raw);
}
```

So `final_score = clamp01(Σ signalᵢ · weightᵢ)`. Semantic similarity dominates (0.30),
but it can never solely decide selection — lexical, structural, trust, freshness,
graph-connectivity, contradiction, and task-fit signals together carry the other 0.70.

### 5.3 Task-aware weight adjustments

Weights are *per-candidate, per-query* adjusted before combination, encoding that
different intents value different evidence:

- **`code_repair`** nudges toward structural/freshness signals:
  `metadata += 0.05`, `task_fit += 0.05`, `recency += 0.05`, `semantic -= 0.05`.
- **`exact_recall`** (set by `project_recall`) trusts provenance and exact wording over
  fuzzy meaning: `authority += 0.10`, `keyword += 0.05`, `semantic -= 0.10`.

These are additive deltas on the default vector; they intentionally let the weights drift
off the 1.0 sum (the final clamp absorbs it).

### 5.4 Sub-scorer designs

Each signal has a small, deterministic scorer. The designs (paraphrased from
`buildCandidates` and helpers in `root.zig`):

- **semantic** — `cosine(query_embedding, record_embedding)`, where embeddings are a
  *deterministic* 64-dim feature hash (each token hashed via Wyhash into a dimension with
  a sign and magnitude derived from the hash bits, then L2-normalized). The cosine is
  remapped from `[-1,1]` to `[0,1]` via `(dot + 1) / 2`. This is a stand-in for a learned
  embedding, chosen so the reference repo is fully reproducible with no model dependency.
- **keyword** — tokenize the query, ignore tokens shorter than 3 chars, return
  `hits / total` (fraction of query tokens that appear, case-insensitively, in the
  record text).
- **metadata** — base `0.25`; `+0.40` if the plan's `project` matches the record's
  project; `+0.20` if the record kind is `constraint` or `project_decision`; `+
  importance·0.15`; clamped.
- **recency** — age-decay on `updated_ns`: `1 / (1 + age_days / 30)` (a ~30-day soft
  half-life; newer records score near 1).
- **authority** — directly `Authority.score()` (§3).
- **graph** — four `+0.25` flags for "well-connectedness": has `supersedes`, is *not*
  superseded, has a `source_uri`, has tags.
- **contradiction** — `1.0` iff kind is `contradiction`, else `0.0`.
- **task_fit** — a `switch (task) → switch (kind)` table giving `1.0`/high to the kinds
  that matter for each task and a low floor otherwise. E.g. `implementation_design`
  favors `project_decision` / `code_fact` / `constraint`; `legal_review` favors
  `constraint` / `contradiction` / `summary`; `benchmark_review` favors `benchmark` /
  `contradiction` / `summary`; `general` is a flat `0.50`.

### 5.5 The backing `MemoryRecord`

The signals read off a record whose design carries exactly the fields the scorers need —
identity + text + embedding, the `kind`/`authority` classifications, project/tags/source
provenance, lifecycle timestamps (`created_ns`/`updated_ns`/`expires_ns`), supersession
links (`supersedes`/`superseded_by`), and the derived `trust`/`importance`/`access_count`
scalars. The supersession links and `source_uri`/`tags` are what make the `graph` signal
possible; the timestamps drive `recency`.

## 6. SEA selection — budgeted greedy choice (Proposed design)

Scoring ranks candidates; *selection* turns the ranking into a sparse set under three
budgets. `SeaOptions` carries the budgets (defaults: `max_tokens = 4096`,
`max_records = 16`, `per_cluster_limit = 4`, plus the weight vector). The algorithm
(`selectSeaCandidates`):

1. **Sort** all candidates by `final_score` descending.
2. **Greedily walk** the sorted list, admitting a candidate unless it trips a budget:
   - **token budget** — `used_tokens + candidate.estimated_tokens > max_tokens`
     (token estimate is the cheap `(len + 3) / 4`, min 1);
   - **record budget** — already selected `>= max_records`;
   - **cluster budget** — this `cluster_id` (= the record's `MemoryKind`) already has
     `>= per_cluster_limit` selections **and** the candidate's `final_score < 0.92`.
3. Admitted candidates go to `selected_ids` (and bump the cluster counter + token total);
   rejected ones go to `rejected_ids`.

```zig
const too_many_cluster = count >= per_cluster_limit and c.final_score < 0.92;
const too_many_records = selected.items.len >= max_records;
const too_many_tokens = used_tokens + c.estimated_tokens > max_tokens;
if (too_many_cluster or too_many_records or too_many_tokens) {
    try rejected.append(allocator, c.record_id);
    continue;
}
```

Design intent worth preserving:

- **Sparsity is the point.** The output is deliberately a *small* set bounded by tokens
  and record count, not a ranked dump — this is the "sparse evidence" in SEA.
- **Cluster diversity** prevents one memory kind from monopolizing the context (e.g. four
  `code_fact`s crowding out the one `constraint`). The `< 0.92` override is the escape
  hatch: an *exceptionally* high-scoring record is admitted past the per-kind cap, so a
  near-perfect match is never dropped purely for diversity.
- The result (`SeaSelection`) keeps both `selected_ids` and `rejected_ids` plus
  `total_estimated_tokens` and a human-readable `reason`, so the selection is auditable.
- A `contextPack` step then renders the selected records into a labeled text block
  (`[id] kind=… authority=… trust=… importance=…` + project/source/text per record) —
  the actual evidence preamble handed to the model.

End-to-end, the proposed flow is:
`QueryPlan.infer(query)` → `buildCandidates` (score every record across the 8 signals
with task-adjusted weights) → `selectSeaCandidates` (budgeted greedy sparse pick) →
`contextPack` (render the chosen evidence).

---

## 7. Cross-reference: ABI's `feat-sea` today

ABI already has a SEA surface, but it is a **different, simpler Phase-1 design** — the
multi-signal scorer and budgeted selection above are *not* what `feat-sea` implements
today. Mapping them honestly:

| Aspect | `abbey_wdbx_sea_zig` (this extract — Proposed) | ABI `feat-sea` (`src/features/sea/`, disabled by default — Current) |
| --- | --- | --- |
| Status | Reference design; repo does not compile on current toolchain | Real code behind `-Dfeat-sea` (off by default); `mod.zig`/`evidence.zig`/`learn_loop.zig`; contract-tested |
| Retrieval scoring | 8 weighted signals, task-aware weight deltas, clamp+combine | Single semantic score — a WDBX vector `store.search` over a shared text embedding |
| Selection | Budgeted greedy pick with token / record / **cluster-diversity** budgets | Top-`limit` hits; prompt preamble capped at `MAX_PROMPT_BYTES` (4096) |
| Intent model | `QueryPlan` + `TaskType` inference re-weights scoring | No `QueryPlan`; persona routing via the existing AI router |
| Typed memory | `MemoryKind` (9) + `Authority` (5) drive scoring | Records are WDBX vectors + `completion:<id>` metadata; persona label parsed to a static `abbey`/`aviva`/`unknown` |
| Loop | Score → select → context-pack → answer | `gatherEvidence` → `augmentPrompt` → reuse `ai.completeWithStore` → adapt persona-router weights (`router.AdaptiveModulator`); "introduces no new ML" |

What ABI's `feat-sea` shares with the reference design is the *shape of the loop* —
recall durable evidence, prepend it as bounded prompt context, then complete — and a
bounded-preamble discipline (ABI's `MAX_PROMPT_BYTES = 4096` mirrors the reference's
`max_tokens` default).

What the reference design offers that ABI's `feat-sea` does **not** yet have, and could
inform a future `feat-sea` phase:

- A **typed memory taxonomy** (`MemoryKind`) and a **provenance/trust ladder**
  (`Authority`) with an explicit numeric trust mapping.
- A **multi-signal scorer** (semantic + keyword + metadata + recency + authority + graph
  + contradiction + task-fit) instead of a single semantic similarity.
- A **task-aware** re-weighting driven by an inferred `QueryPlan` / `TaskType`.
- **Cluster-diversity-aware sparse selection** (per-kind caps with a high-score override)
  rather than a flat top-`limit` cut.

Any such promotion would have to land as real ABI source + tests under the mod/stub
parity rules and pass `./build.sh check`, and would be tracked the usual way in
`tasks/`. Until then, the above stays **Proposed**.

---

## 8. Maintenance

This is a design extract from a now-archivable reference repo, not a description of ABI
behavior. Before reusing any line externally, reconcile against ABI source
(`src/features/sea/`, `src/features/wdbx/`, `tests/contracts/`) and downgrade anything
ABI source does not prove. Promote any item here into ABI's real SEA surface only when
source + a passing test back it (`docs/spec/wdbx-north-star.md` §9 describes the same
Proposed → Partial → Current discipline).
