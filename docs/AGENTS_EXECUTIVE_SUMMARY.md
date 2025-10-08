# ABI Zig 0.16-dev — Executive Summary

This one-page brief distills the authoritative `AGENTS.md` so new collaborators can orient quickly, uphold the guardrails, and start contributing with confidence.

## Mission & Non-Goals
- **Mission:** Deliver a production-grade, allocator-disciplined, high-performance ABI framework on Zig 0.16-dev while preserving public APIs and improving reliability, modularity, and developer ergonomics.
- **Non-Goals:** Mandating a single ML architecture, locking the project to one GPU backend, or distributing proprietary model weights.

## Audience
- Systems/ML engineers shaping agents, database, and GPU components.
- Contributors expanding features, tests, and documentation.
- Maintainers enforcing performance, compatibility, and release gates.

## Architecture at a Glance
- **Framework runtime:** lifecycle management, feature toggles, plugin discovery.
- **Feature families:** `ai`, `database` (WDBX), `gpu`, `web`, `monitoring`, `connectors`.
- **I/O contract:** stdout is structured JSON for machines; stderr carries human-readable logs.
- **Testing & CI:** comprehensive unit, property, fuzz, and soak coverage backed by a GitHub Actions matrix.

## Hard Guardrails (Do Not Break)
1. **Public API stability:** keep `@import("abi").ai.agent.Agent` valid; ship shims for any surface changes.
2. **Allocator discipline:** no hidden heap usage—every allocation is explicit and caller-owned.
3. **Performance parity or better:** WDBX, agent runtime, SIMD, and GPU paths must not regress.
4. **Deterministic builds:** respect the pinned `.zigversion`; ensure reproducible `zig build` invocations.
5. **Typed errors & structured logging:** cleanly separated channels with explicit error sets.

## Authoritative Repository Layout
```
src/
  mod.zig, comprehensive_cli.zig
  framework/{mod,config,runtime}
  features/{ai,database,gpu,web,monitoring,connectors}
shared/, core/, examples/, benchmarks/, docs/, tests/
```

## Phase Plan (A → H)
A) Build & layout → B) Framework & toggles → C) I/O & logging → D) CLI →
E) WDBX → F) GPU/CPU SIMD → G) Agents → H) Tests & CI/CD.

**Exit criteria:** Every phase merges with tests, docs, and—where relevant—benchmarks.

## Initial Performance Budgets
- WDBX insert ≥ 50k ops/s @ D=128 on an 8-core desktop.
- Vector search P95 ≤ 10 ms for k=10 @ N=100k (IVF-Flat baseline).
- Agent process latency ≤ 1 ms (echo) / ≤ 5 ms (with middleware).
- GPU dense forward ≥ 5× CPU throughput or a clean CPU fallback.

## Contributor Quick Start
```sh
# Build, test, fmt
zig build -Doptimize=ReleaseSafe
zig build test
zig fmt --check .

# Run CLI
zig build run -- features list --json
zig build run -- agent run --name QuickStart

# Enable GPU path (if supported)
zig build -Denable-gpu run -- gpu bench
```

## PR Hygiene & Quality Gates
- Conventional commits; small, rationale-backed PRs.
- Couple code with tests/docs and JSON golden outputs where applicable.
- Benchmark thresholds enforced via JSON configs; CI must be green before merge.

## Tool-Calling Workflow (LLM Agents)
- Available tools: `create_file`, `modify_file`, `run_tests`, `run_benchmarks`.
- Every change set should bundle code, tests, docs, and benchmarks for hot paths.
- Public API shifts require shims plus migration guidance.

## Ownership & Communication
- Assign an owner per phase and track issues via phase labels (`phase:A-build`, …).
- Treat `AGENTS.md` as the single source of truth; this page is the high-level map.

