---
name: goal-ledger
description: >-
  Use when the user runs /goal (or says goals), mentions goals.md, asks to
  capture/track/update a goal, says continue/do all/finalize on goal work, or
  an agent is about to mark a goal done after a green gate, stub, demo, or
  stakeholder pressure.
---

# Goal ledger

Intentions live in `tasks/goals.md`, one coarse `##` section each. Steps live in
`tasks/todo.md` beside it. A green gate proves a slice; a goal is `done` only
when its acceptance criteria hold.

When `donald-mode` is available, follow its autonomy rules for `continue`,
`do all` and `finalize`.

## Section format

```markdown
## Ship hybrid agent CLI
status: in_progress
- **2026-09-21 23:1x EDT, `abc1234`:** WDBX bridge landed; gate clean, 701 tests. Next: persona routing.
- **2026-09-20 ...:** older bullet
```

- Statuses: `todo` | `in_progress` | `blocked` | `done`.
- **New bullets go directly under `status:`, newest first,** each opening with a
  bold date and time (plus the commit when there is one) and ending with
  `Next:` or the named stop. The top bullet is then the current state.
  Older sections were appended in no fixed order; treat their undated or lower
  bullets as history (see *Reading a section*).
- One `##` per intention. Checklists go in `todo.md` under a `##` of the same
  title. Closed goals stay in the file.

## Resolve the ledgers

A request resolves to a **write ledger** and a **read set**. Name both in the
reply.

1. **Write ledger:** the nearest `tasks/goals.md` walking up to the enclosing
   git repository root; otherwise `~/tasks/goals.md`. Container directories
   (`~/dev/active`) and home use `~/tasks/goals.md`.
2. **Read set:** the write ledger, plus, inside a repository, every open
   section of `~/tasks/goals.md` whose header names this repository (its
   directory name, its remote repo name, or the project's name, e.g. "Cell"
   for `cell-lang`). Goals for a repository are often tracked in the machine
   ledger while the repository's own ledger holds only closed slices. A
   repository ledger with nothing open is the moment to read the machine
   ledger, not a reason to stop.
3. **Record outcomes in the section that holds the goal.** If the goal lives in
   the machine ledger, its bullet goes there; a slice with no existing goal
   gets one new `##` in the write ledger.
4. **A repository's own goal skill layers on this one** (abi's
   `.agents/skills/goals` adds abi gates and claim rules): inside that
   repository follow it too; this skill stays the contract.
5. Create a `tasks/goals.md` (header `# Goals`) only inside a git repository,
   and only for `capture` or `execute`. Home root, `~/dev/active`, `~/Archive`
   and iCloud paths never get one.

Open sections with line numbers, in any ledger size (header line, then its
status line):

```bash
grep -n '^## \|^status:' <ledger> | grep -B1 '^[0-9]*:status: *\(todo\|in_progress\|blocked\)' | grep -v '^--$'
```

Keep `$` followed by a digit out of this file: the skill loader substitutes
positional arguments into the body, so awk's dollar-zero arrives as the user's first
argument.

## Reading a section

1. **Size it first:** `awk 'NR>=A && NR<B' <ledger> | wc -l`, where A is its
   header line and B the next header's. Over ~100 lines, read only the
   `status:` line, the dated bullets from the last three days
   (`grep -n '^- \*\*20'`), and any bullet naming a stop, blocker or pending
   decision. The rest is history.
2. **Verify the artefact, not the prose.** `git log`, `grep` the source, read
   the project's status document (a feature matrix such as `docs/FEATURES.md`
   beats any ledger bullet), run the gate. A bullet saying "next slice is X"
   is stale the moment the repository shows X landed.
3. **Re-measure counts before acting on them.** Ahead/behind needs a `git fetch`
   first; `@{u}` reads a cached tracking ref.
4. **Correct stale text with a new dated bullet** stating what was stale, the
   measurement, and how you measured it. Earlier bullets stay as written.

## Before taking a slice

The slice is clear to start when all three hold:

- **Owner:** no live session owns the files. `ListAgents` shows presence, not
  location (a peer listed as an idle Desktop session can be mid-task in this
  repository), so message every listed peer that could hold it, naming the
  files you will touch, and wait for the answers. An empty `cwd` sweep and a
  stale `.git/index` do not prove absence either.
- **Baseline:** `git --no-optional-locks status` and HEAD recorded; the gate's
  current verdict known (run it, or cite a run on this exact HEAD).
- **Decision:** the slice needs no call that belongs to the user. A section or
  brief saying a row "awaits Donald's decision" is a named stop for that row.

## Commands

| Args | Action |
|------|--------|
| empty, `list` | Report every open section in the read set with its `status:` and top bullet's `Next:`, then the closed count. Invent nothing; create no file. |
| `capture <text>` | Reuse a matching open section, else add one `## <text>` with `status: todo` (`in_progress` if work starts now). |
| `update <goal> <status>` | Only when the last word is a status and the rest names a section. Rewrite that `status:` line; on `done`, add the outcome bullet. |
| `execute <goal>` | Set `in_progress`, run *Before taking a slice*, land the smallest verified slice, run the project gate, add the outcome bullet. |
| `continue`, `do all`, `finalize` | The loop below. `do all` covers every open goal in the read set. |
| anything else | Names an open goal: `execute` it. Otherwise `capture` the whole string. Say which rule applied. |

## The continue loop

Repeat until a named stop or until nothing in the read set is open:

1. Pick the next slice: the top bullet's `Next:`, confirmed against the
   project's status document; otherwise the lowest-risk open row that needs no
   user decision.
2. Run *Before taking a slice*.
3. Land it: change, project gate green (read the verdict from the gate's own
   output and its exit code), commit when the repository's rules allow.
4. Add the dated outcome bullet to the goal's section, with the gate evidence
   (verdict, test count, log path) and `Next:`.

A green gate ends step 3, never the loop.

**Named stops**, each reported with its evidence:

- a decision that belongs to the user (cite the brief or bullet asking for it);
- a live peer owns the files the next slice needs;
- an outward or hard-to-reverse action without approval this session (push,
  deletion, a move out of a project, shared config);
- a red gate the current slice cannot fix;
- an external block (billing lock, missing credential, unavailable service).

When every open goal is at a named stop, report the stops; that is a complete
result.

## Closing a goal

`done` requires every acceptance criterion in the title and bullets to hold on
the current HEAD. Then:

- add the outcome bullet naming what shipped and what is left out of scope;
- reopen a `done` goal only for new scope; leftover cosmetics are todo items.

**Stubs.** A stub, a disabled feature, `echo not implemented` or a demo path
leaves a real-capability goal open, whoever asks for `done`. The honest
alternatives: keep it `in_progress`/`blocked`, or rename the `##` to include
`demo stub` before marking it `done` with a stub outcome bullet. Docs
(AGENTS/README/identity) describe that capability as **Proposed** until the
real one lands.

| Pressure | Answer |
|----------|--------|
| "gate is green, so it's done" | Green covers the slice tested, not every criterion. |
| "mark it done now, for the demo" | Rename to `demo stub`, or keep it open. |
| "do all, so split it into goals" | One intention, many `todo.md` items. |
| "delete done sections to tidy" | Closed sections stay; history is evidence. |
| "context is long, natural stop" | Only a named stop ends the loop. |

## Reply shape

Name the write ledger and read set, what landed (commits, gate verdict and
count), and end with a table: goal | status | next slice or named stop.
