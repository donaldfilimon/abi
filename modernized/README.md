# modernized/ — pointer only (no scaffold tree)

This directory is **not** a second production tree and does **not** contain
Zig packages. Production code lives under repo-root `src/` and is gated by
`./build.sh check`, frozen CLI (13 cmds), frozen MCP (12 tools), and
`docs/contracts/external-claims-audit.mdx`.

## What this folder is

A local placeholder left from reimagine / modern-refactor discussion. An older
draft claimed greenfield packages at `src-reimagined/{ai,mcp,wdbx}/` (also
referred to elsewhere as `modernized/abi-reimagined/`). Those paths:

- are **absent** on disk,
- never appear in `git` history or on any branch,
- were **not** supposed to be scaffolded yet — see Phase D in the local
  (markdown-ignored) design note `analysis/src/REIMAGINED_ARCHITECTURE.md`
  (“do not scaffold until explicitly approved”).

Approach-1 waves explicitly left untracked `modernized/` out of scope
(`docs/superpowers/plans/2026-07-15-approach1-waves-a-b-c.md`).

## Live ownership (use these)

| Domain | Production path |
|--------|-----------------|
| AI / SEA | `src/features/ai/`, `src/features/sea/` |
| MCP | `src/mcp/` |
| WDBX | `src/features/wdbx/` |

Honest adoption is **incremental change in `src/`**, not a directory replace.
Tracked planning sources: `docs/spec/abi-refactor-design.mdx`,
`docs/superpowers/plans/`, `tasks/todo.md`, `tasks/goals.md`.

## Do not

- Treat this directory as package sources or run a separate `build.zig` here.
- Invent `src-reimagined/` Zig to “match” the old claim without Phase D HITL.
- Merge any future scaffold wholesale over live `src/`.
