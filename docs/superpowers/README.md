# Superpowers materials (not Mintlify nav)

Working and historical planning docs for agent workflows. **Not** listed in
[`docs/docs.json`](../docs.json). Do not treat these as public contracts.

| Path | Role |
| ---- | ---- |
| `plans/` | Active / Rust-era implementation plans (created on demand; currently empty) |
| `specs/` | Design drafts pending review or landing (created on demand; currently empty) |
| [`archive/`](archive/) | Superseded plans/specs — historical only (includes completed Zig-era waves and the reimagine-era architecture spec) |

Published layout and claim boundaries: [Docs layout](../README.md). Active board:
`tasks/todo.md`.

## Adding a plan or design draft

1. Prefer `YYYY-MM-DD-<slug>.md` under `plans/` or `specs/`.
2. Lead with **Status** (`Draft` / `Completed` / `Superseded`) and link the
   related `tasks/todo.md` row or PR.
3. Keep claim-honest wording — no fake-complete of honest stubs or non-goals.
4. Do **not** add the path to Mintlify navigation until it is promoted to a
   reviewed `.mdx` under `docs/spec/` or `docs/contracts/`.
