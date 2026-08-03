# Design: Mintlify docs hub redesign

**Date:** 2026-07-08

**Status: Completed** (hub Cards + nav landed; index hygiene closed `da2221cc` on `cursor/agent-orch-skill-docs-hygiene`). Implementation plan: `docs/superpowers/plans/2026-07-08-mintlify-docs-hub.md`.

**Owner slice:** modern-refactor Phase 2 @docs (IMPROVEMENT_PLAN Slice 1)

**Pipeline:** agent-development (`abi`) → abi-goal-orchestrator → writing-plans → SDD

## Problem

`docs/index.mdx` is a flat bullet list. Nav in `docs/docs.json` is sparse relative to hub links. Agents re-plan work that is already landed (MCP bearer auth, WDBX compact) because the hub does not surface "source wins" / contracts as first-class cards. `tasks/todo.md` still lists modern-refactor Phase 2–4 as deferred.

## Non-goals

- No new CLI commands or MCP tools.
- No capability claims beyond `docs/contracts/external-claims-audit.mdx` and executable sources.
- No production multi-host, FHE, native kernel, or QPS/latency wording.
- No rewrite of archived superpowers content; archive stays out of active nav.
- Do not touch unrelated dirty tree files (skill rephrases, `mcp/launcher.sh`, etc.) unless this slice needs them.

## Goals

1. Redesign `docs/index.mdx` as a Mintlify-native hub: hero blurb, source-wins callout, Card groups for Architecture / Contracts / Guides.
2. Align `docs/docs.json` groups with the hub (Overview, Architecture, Specs, Contracts) without adding unproven pages.
3. Keep all claim language honest; prefer links to `contracts/*` over re-listing 13 CLI / 12 MCP items on the index.
4. Gates: `npx mint@latest validate` when Node available; `./build.sh check` if any shared claim prose moves; no `src/` required.

## Already done (do not re-implement)

| Area | Evidence |
|------|----------|
| MCP loopback bearer | `ABI_MCP_HTTP_TOKEN` + tests; CHANGELOG |
| WDBX REST bearer | `ABI_WDBX_REST_TOKEN` + tests |
| `wdbx db compact` | handlers + recovery test |
| modern-refactor Phase 1 | skill `references/` filled |
| Mintlify Card hub + nav | `3b340b8b`, `92e07827`; hygiene `da2221cc` |

## Architecture (docs-only)

```
docs/docs.json  ──navigation groups──► Mintlify site
docs/index.mdx  ──hub cards──────────► contracts/* + spec/* + root guides
tests/contracts/public_docs.zig       (only if claim wording changes)
```

## Components

| Unit | Responsibility |
|------|----------------|
| `docs/index.mdx` | Hub UI: Cards + Note; single source-wins callout |
| `docs/docs.json` | Nav groups/pages matching hub destinations |
| Optional thin CONTRIBUTING note | Only if missing; pointer to source-wins + `./build.sh check` |

## Success criteria

- [x] Index uses Mintlify `Card`/`CardGroup` (or equivalent) instead of a single flat list for primary destinations.
- [x] Active nav does not promote `superpowers/archive` as contracts.
- [x] No new unproven capability sentences.
- [x] `npx mint@latest validate` passes (or document Node unavailability and still produce valid MDX). *Optional; not in CI; validate when Node available* — attempted with host Node 26.5.0; mint@latest requires LTS Node ≤24. MDX paths and `docs.json` verified manually (`PATHS_OK` / `JSON_OK`).
- [x] Unrelated dirty files remain untouched.

## Recommended execution order after this design

1. ~~This hub slice (low risk).~~ **Done.**
2. IMPROVEMENT_PLAN Slice 2 — factor `tools/run_contract_cli.sh`.
3. Slice 3 — data-driven feature-flag matrix.
4. Slice 4 — MCP contract depth (existing 12 tools only).
