# modernized/ — Phase D approved minimal scaffold

**Live production code remains under repo-root `src/`** until an explicit
cutover. This tree is a **pointer + first package layout** for the reimagined
module DAG — not a second build root and not a replacement for `src/`.

## Status

| Item | State |
|------|-------|
| Phase D HITL | **Approved** (user `/abi do all`, 2026-07-16) |
| Scaffold | Minimal package READMEs under `packages/` |
| Build integration | **None** — do not add a separate `build.zig` here yet |
| Cutover | **Not started** — `./build.sh check` still gates `src/` |

## Package layout (planned DAG)

```
modernized/
  README.md
  packages/
    foundation/ README.md
    connectors/ README.md
    wdbx/       README.md
    ai/         README.md
    p1-thin/    README.md
    cli/        README.md
    mcp/        README.md
```

Each package README documents ownership and the **current** production path
under `src/`. Zig sources are not duplicated here.

## Do not

- Treat this directory as the active build input.
- Claim feature parity or cutover complete.
- Merge this tree wholesale over `src/`.
- Expand frozen CLI/MCP surfaces from this scaffold.

## Live ownership (use these today)

| Domain | Production path |
|--------|-----------------|
| AI / SEA | `src/features/ai/`, `src/features/sea/` |
| MCP | `src/mcp/` |
| WDBX | `src/features/wdbx/` |
| CLI | `src/cli/`, `src/main.zig` |

Tracked design: `analysis/src/REIMAGINED_ARCHITECTURE.md`,
`docs/spec/abi-refactor-design.mdx`, `docs/spec/phase-d-cutover-plan.mdx`,
`tasks/todo.md`.
