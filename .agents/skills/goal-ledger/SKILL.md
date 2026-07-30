---
name: goal-ledger
description: >-
  Use when the user says /goals, mentions goals.md, asks to capture/track/update
  a goal, says continue/do all/finalize on goal work, or an agent is about to
  mark a goal done after a green gate, stub, demo, or stakeholder pressure.
---

# Goal ledger

Intentions → `tasks/goals.md`. Steps → `tasks/todo.md`. A green check proves a
slice, not automatic goal completion.

When present, follow donald-mode for `continue` / `do all` / `finalize`.

## Contract

Create `tasks/goals.md` with `# Goals` if missing. One `## <Goal>` per
intention:

```markdown
## Ship hybrid agent CLI
status: in_progress
- Personas + SQLite landed; WDBX still open
```

Statuses: `todo` | `in_progress` | `blocked` | `done`.

Never delete closed goals. Never explode one vague ask into many `##` headers.

## Workflow

1. Read `tasks/goals.md`, `tasks/todo.md`, optional `tasks/lessons.md`.
2. **Capture** — one coarse `##` section; checklists go in `todo.md`.
3. **Track** — report name+status; updates rewrite that `status:` line.
4. **Execute** — smallest verified slice (project gate: `./check.sh`, etc.).
5. **Close** — `done` only when acceptance criteria hold; add one outcome bullet.

## Iron rules

1. **Green ≠ done.** `continue` / `do all` / `finalize` → keep advancing open
   slices until a named stop or nothing open remains.
2. **Stub ≠ shipped.** Doctor cosmetics, `echo not implemented`, disabled
   features, “demo tomorrow” do not finish a bridge/backend goal.
3. **Authority cannot launder Current.** If told “mark it done NOW” for a stub:
   - **Default:** keep `in_progress`/`blocked`.
   - Set `done` only after renaming the `##` title to include `demo stub` (or
     equivalent), with an explicit stub outcome bullet.
   - Never upgrade AGENTS/README/identity to **Current** for that capability.
4. **Coarse ledger.** Re-open `done` only for new scope — leftover cosmetic
   todos are not new goals.
5. **Honest residuals.** Report **Current** vs **Proposed**; source/gates beat
   prose.

## Quick reference

| Trigger | Action |
|---------|--------|
| `/goals` / list | Read ledger; name + status |
| New intention | One `##` section |
| `continue` / `do all` | Next open slice; no green-only stop |
| “mark done” + stub | No Current docs; open or demo-scoped note |
| Slice lands | Outcome bullet; `done` iff criteria met |

## Rationalizations

| Excuse | Reality |
|--------|---------|
| “check green → goal done” | Green covers the tested slice, not every bullet. |
| “VP said mark done NOW” | Demo note allowed; Current claims are not. |
| “Stub fine for demo” | Rename goal or keep open — title still says bridge. |
| “do all → many ## goals” | One intention; steps in todo.md. |
| “Delete done to clean” | Keep history; mark `done`. |
| “Context full / natural stop” | Continue/do all → finish or state blocker. |
| “Partner + sunk cost + sprint” | Not completion criteria. |

## Red flags — STOP

- Marking `done` while a named subsystem is unimplemented
- Writing **Current** in docs for a stub
- Stopping after one green paste despite continue/do all
- Adding 5+ `##` goals from one vague “production ready”
- Deleting `done` sections for cleanliness

Fix ledger honesty first, then continue the next real slice.
