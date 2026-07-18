# Superpowers materials (not Mintlify nav)

Working and historical planning docs for agent workflows. **Not** listed in
[`docs/docs.json`](../docs.json). Do not treat these as public contracts.

| Path | Role |
| ---- | ---- |
| [`plans/`](plans/) | Dated implementation plans (may be completed or in-flight) |
| [`specs/`](specs/) | Design drafts pending review or landing |
| [`archive/`](archive/) | Superseded plans/specs — historical only |

Published layout and claim boundaries: [Docs layout](../README.md). Active board:
`tasks/todo.md`.

## Adding a plan or design draft

1. Prefer `YYYY-MM-DD-<slug>.md` under `plans/` or `specs/`.
2. Lead with **Status** (`Draft` / `Completed` / `Superseded`) and link the
   related `tasks/todo.md` row or PR.
3. Keep claim-honest wording — no fake-complete of honest stubs or non-goals.
4. Do **not** add the path to Mintlify navigation until it is promoted to a
   reviewed `.mdx` under `docs/spec/` or `docs/contracts/`.
