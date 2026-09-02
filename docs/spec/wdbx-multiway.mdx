---
title: "WDBX Multiway Rewriting Simulator"
description: "Bounded, deterministic Wolfram-style string-rewriting over finite slices of computational rule space. Not a physics engine."
---

**Status:** **Current** — implemented in `../wdbx/crates/abi-wdbx/src/multiway.rs`, exercised by unit, regression, persistence, resume, and CLI tests.
**Scope:** A bounded multiway rewriting engine integrated into WDBX. It enumerates reachable computational states, preserves every valid rewriting branch, merges equivalent states deterministically, and persists experiments in WDBX.

> **Scientific-scope discipline (read first).** This module simulates *finite, explicitly bounded slices* of computational rule space. It **does not** simulate "the complete ruliad" and makes **no** claim about fundamental physics. Structural resemblance between a rule's output and physical behavior is **not** proof — or evidence — that the rule describes the physical universe. Every exported number is either an *observed simulation output* (states, events, transitions) or a *derived structural metric* with an explicitly defined denominator; nothing here is statistical inference or speculative interpretation. When this document and the source disagree, the source wins (`../wdbx/crates/abi-wdbx/src/multiway.rs` and its tests).

## What a multiway rewriting system is

A rewriting system has a set of **rules** `LHS -> RHS` and one or more **initial states**. To evolve a state, you find every place a rule's left-hand side matches and, for each match, produce a successor state with that occurrence replaced by the right-hand side. A *multiway* system keeps **every** successor of **every** match rather than picking one — so evolution is a branching graph, not a single line. When two different branches reach the same canonical content, the states are merged, but the distinct rule applications (events) that led there are all retained.

This implementation is **exact string rewriting**: states are byte strings, a rule matches a literal substring, and a successor replaces that substring. Overlapping matches are included (a rule `AA->B` matches `AAA` at offsets 0 and 1).

## What this implementation simulates — and what it does not

**Simulates:**

- Enumeration of reachable canonical states from bounded breadth-first evolution.
- Preservation of every valid rewriting branch as a distinct **event**.
- Deterministic merging of equal states by content hash (SHA-256 of the payload).
- A **state-transition graph** (canonical states → valid transitions) and an **event graph** (every rule application, with multiplicity and provenance).
- Measurable structural metrics for comparing rules.
- Content-addressed persistence, provenance, and full reproducibility in WDBX.

**Does not simulate:**

- "The complete ruliad." The ruliad is the (uncomputable, unbounded) limit of *all* possible rules applied in *all* possible ways. It cannot be directly enumerated: for any non-trivial rule set the reachable-state set is generally infinite (e.g. `A->AA` grows without bound), rule space itself is infinite, and even a single bounded system's growth can be super-exponential. This engine therefore explores only what fits inside caps you set (depth, states, events, payload, wall-clock, memory).
- Physics. No claim is made that any rule corresponds to space, time, particles, or any physical law.
- Rigorous causal structure for strings (see [Causal graph](#graph-products), below).

## Core concepts: states, transitions, events

These three are deliberately distinct and must not be conflated:

| Concept | Vertex/record | Identity | Multiplicity |
|---|---|---|---|
| **State** | Canonical byte payload | SHA-256 of payload (content, never address/order/branch) | Deduplicated — one record per distinct content |
| **Transition** | An ordered pair `(src, dst)` of states | The `(src_hash, dst_hash)` pair | Deduplicated — one per distinct pair |
| **Event** | One rule application | Commit index (stable id) | **Preserved** — two events may share `(src, dst)` if they are genuinely different applications (different rule and/or offset) |

Deduplicating states while preserving events is the central invariant. Two identical rules `A->B` applied to `A` produce **one** state `B`, **one** transition `A→B`, and **two** events.

## Canonical state merging

Every payload is hashed with SHA-256. A state's identity is that hash — not its discovery order, not the branch that reached it, not a memory address. When a newly generated successor's hash already exists, the engine reuses the existing state (recording the event) instead of storing a duplicate. In breadth-first traversal the depth at which a state is first discovered is also its minimum known depth.

## Bounded completeness

A result is reported **exhaustive within the requested domain** *only* when the frontier drains (`termination = frontier_exhausted`) — i.e. no successor of any discovered state was left unexpanded and **no** resource bound stopped exploration. If any cap is hit, the result is a **valid partial** result: every state/event/transition in it is real, but there may be more states beyond the boundary. The engine never marks a bounded run "complete" just because it hit a limit. Termination reasons:

`frontier_exhausted` · `max_depth` · `max_states` · `max_events` · `payload_limit` · `deadline` · `cancelled` · `allocation_failure` · `invalid_rule` · `invariant_failure`

Hard caps (`max_states`, `max_events`, `max_depth`, `max_payload`) are enforced *before* committing a state expansion, so they can never be exceeded — successors of one source state commit atomically or not at all. The payload cap is checked *before* allocating a replacement buffer, so an unbounded growth rule cannot allocate its way past the limit.

## Metrics (with denominators)

All metrics are derived from observed output. Denominators are explicit:

- **Unique states**, **event count**, **unique transitions** — direct counts.
- **States/events per depth**, **frontier width per depth** — per-depth counts; in breadth-first traversal frontier width equals states first discovered at that depth.
- **Mean out-degree** = unique transitions ÷ unique states (terminal states count as degree 0). This is a *state-graph branching* measure, **not** a "computational complexity" claim.
- **Max / median out-degree** — over the per-state distinct-destination out-degrees of *all* unique states.
- **Convergent states** — states with distinct-predecessor in-degree > 1.
- **Self-loops**, **cycle presence** (3-color DFS over the unique-transition graph), **weakly connected components** (union-find).
- **Max / mean payload bytes** — over all unique states.
- **Growth rate by depth** — states at depth *d+1* ÷ states at depth *d*, defined only while the earlier depth is non-empty.
- **Runtime**, **states/s**, **events/s** — wall-clock; excluded from the canonical export so identical runs export identical bytes.
- **Termination reason** and **exhaustive-in-domain** flag.

No metric is labeled "computational complexity"; names describe exactly what is measured (state-growth rate, out-degree, graph components, …).

## Graph products

- **State evolution graph** — vertices are canonical states, edges are valid transitions. Exported to Graphviz DOT (`--format dot`).
- **Event graph** — vertices are events; multiplicity and provenance (source, destination, rule id, match offset, depth, local index) are retained. Every event is a distinct DOT edge, so multiplicity is visible.
- **Causal graph** — **token-lineage** (Current). Export derives event→event edges from reconstructed per-byte producer lineage: a later event is a child of every distinct prior event that produced a byte in its matched LHS. Hypergraph backends remain Proposed.

## Reproducibility guarantees

With the same initial states, rules, configuration, seed, and worker count, the simulator produces **byte-for-byte identical** canonical JSON exports. Determinism comes from: fixed rule order, ascending match-offset scanning (overlapping matches included), single-threaded commit of state/event identifiers (scheduler order can never influence exported ids), and exclusion of wall-clock timing from the canonical document. The full configuration is serializable and hashable; its SHA-256 is the experiment's identity. A regression test hashes the complete export across repeated runs and asserts equality.

## Resource limits

Configurable per experiment: `max_depth`, `max_states`, `max_events`, `max_payload` (bytes), `max_duration_ms` (deadline; 0 = off), `max_memory_bytes` (approximate engine-memory budget; 0 = off). Cancellation is polled once per state expansion (Ctrl-C from the CLI). All test and benchmark configurations are bounded so they cannot consume exponential resources.

## WDBX persistence layout

`persist_multiway` writes experiments through the v2 `VersionedStore` facade
and publishes a covering immutable segment; it does not introduce a second
storage architecture:

- **Canonical states** — each payload stored content-addressed under `multiway:state:<sha256-hex>`. Identical payloads across experiments share one entry (no duplicate payloads).
- **Experiment export** — the full canonical JSON (config, config hash, rules,
  initial states, states, events, transitions-by-events, metrics, termination,
  export hash, and resume cursor) stored under
  `multiway:experiment:<config-hash-hex>` plus a
  `multiway:experiment:latest` alias. The format retains a frozen legacy
  `zig_version` field for compatibility; it is not the current Rust compiler
  version.
- **Provenance block** — one SHA-linked block records config/export hashes,
  counts, termination, completeness, and the frozen compatibility field.

The key/value writes, block append, and checkpoint are separate durable
mutations. Callers must treat an error as a potentially partial persistence
attempt rather than an atomic transaction.

**Resume:** a bounded run persists its frontier cursor inside the export. `resume_` (CLI `--resume <file>` or `--resume-wdbx <path>`) rebuilds states, events, and the frontier from the export and continues under new effective limits. A resumed deterministic run commits identifiers in exactly the order an uninterrupted run under the same effective limits would, so the final exports match byte-for-byte (verified by resume-equivalence tests, including a mid-depth event-cap interruption).

## CLI examples

```bash
# Reference experiment: canonical JSON to a file, with per-depth table
abi wdbx simulate \
  --initial A --rule 'A->AB' --rule 'A->BA' --rule 'BB->A' \
  --depth 5 --max-states 500 --max-events 5000 \
  --format json --output experiment.json --verbose

# Graphviz DOT of the state-transition multigraph
abi wdbx simulate --initial A --rule 'A->AB' --rule 'BB->A' --depth 3 --format dot --output graph.dot

# Rules from a file; persist into a WDBX checkpoint
abi wdbx simulate --initial A --rules-file rules.txt --depth 4 --store experiments.wdbx.jsonl

# JSON experiment config; resume it deeper from the persisted checkpoint
abi wdbx simulate --config examples/multiway/branching.json --store experiments.wdbx.jsonl
abi wdbx simulate --config examples/multiway/branching.json --depth 7 --resume-wdbx experiments.wdbx.jsonl

# Validate only (no run); nonzero exit + actionable diagnostic on bad input
abi wdbx simulate --initial A --rule 'A->AB' --dry-run
abi wdbx simulate --initial A --rule 'AB'          # exit 2: expected 'LHS->RHS'
```

Invalid rules (missing `->`, empty LHS) and impossible bounds (zero cap, initial state over `--max-payload`) produce actionable diagnostics and a nonzero exit code.

## Regression evidence

For the depth-5 reference case, the Rust CLI regression asserts 31 unique
states and 62 unique transitions. Unit tests in
`../wdbx/crates/abi-wdbx/src/multiway.rs` separately cover deterministic canonical
exports, hard-cap behavior, resume equivalence, and WDBX persistence.

An opt-in compatibility test can compare Rust output with an operator-supplied
historical Zig binary via `ABI_ZIG_BIN`. The default fresh-clone test run does
not provide that cross-language proof.

## Next steps: hypergraph backends

Token lineage for the causal graph is **Current** (export-time derivation in `multiway_export.buildCausalEdges`). Remaining:

- **Hypergraph backend** — replace payloads with hypergraph encodings and rule matching with subgraph isomorphism, keeping the same bounded BFS, canonical hashing (of a canonical hypergraph serialization), event multiplicity, metrics, and WDBX layout.
