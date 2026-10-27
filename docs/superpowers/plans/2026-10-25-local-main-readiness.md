# abi: local main integration readiness

## Goal and authority

User-approved documentation task, 2026-10-25. Prepare owned work for local
`main` only after ownership release, preservation and the repository gates.
Status: **pending owner confirmation and fresh checks**, not merge-ready.
Existing product plans and qualification contracts remain authoritative.

## Observed Git state

Inventory recorded on 2026-10-25; re-read Git and writer state before acting.
Canonical branch: `main`; HEAD: `69910bf61205458dec27dd4b616201e0a5f3898a`.
Remote comparisons below use existing local tracking refs; no fetch was run.

| Local branch | Pinned tip | Commits only on main / only on branch | Cached upstream state |
| --- | --- | --- | --- |
| `main` | `69910bf61205` | 0 / 0 | origin/main  |
| `task/modernization-abbey-gtk-20261025` | `4b0821660387` | 0 / 8 | none  |
| `task/modernization-baseline-20261025` | `3fea36fd9ae5` | 0 / 1 | none  |
| `task/modernization-consumer-dependencies-20261025` | `4b0821660387` | 0 / 8 | none  |
| `task/modernization-dependency-cohort-20261025` | `3f1d4c65de16` | 0 / 5 | none  |
| `task/modernization-updater-zip9-20261025` | `4b0821660387` | 0 / 8 | none  |
| `task/modernization-worker-lock-20261025` | `4b0821660387` | 0 / 8 | none  |

| Existing worktree (relative to DEV_HOME) | Branch | Pinned HEAD | Tracked dirty / untracked files |
| --- | --- | --- | --- |
| `projects/company/legacy/abi` | `main` | `69910bf61205` | 27 / 90 |
| `worktrees/local/the-donald-company/products-modernization-20261025/cohorts/000-baseline/abi` | `task/modernization-baseline-20261025` | `3fea36fd9ae5` | 1 / 0 |
| `worktrees/local/the-donald-company/products-modernization-20261025/tasks/003-dependency-cohort/abi` | `task/modernization-dependency-cohort-20261025` | `3f1d4c65de16` | 1 / 0 |
| `worktrees/local/the-donald-company/products-modernization-20261025/tasks/004-worker-lock/abi` | `task/modernization-worker-lock-20261025` | `4b0821660387` | 1 / 0 |
| `worktrees/local/the-donald-company/products-modernization-20261025/tasks/005-consumer-dependencies/abi` | `task/modernization-consumer-dependencies-20261025` | `4b0821660387` | 1 / 0 |
| `worktrees/local/the-donald-company/products-modernization-20261025/tasks/007-updater-zip9/abi` | `task/modernization-updater-zip9-20261025` | `4b0821660387` | 1 / 0 |
| `worktrees/local/the-donald-company/products-modernization-20261025/tasks/009-abbey-gtk/abi` | `task/modernization-abbey-gtk-20261025` | `4b0821660387` | 0 / 0 |

Worktree locations describe custody, not runtime dependencies. Clean status does
not prove a writer has released a worktree. Existing ignored assets remain in
their original custody; this task neither inventories their contents nor removes
them. App-managed worktrees retain their application-managed lifecycle.

Canonical dirty-file state (grouped to keep this plan readable):

| State | File or subtree | Files |
| --- | --- | --- |
| tracked dirty | `.claude/ralph-goal.verify.sh` | 1 |
| tracked dirty | `.codex/agents` | 1 |
| tracked dirty | `.gitignore` | 1 |
| tracked dirty | `AGENTS.md` | 1 |
| tracked dirty | `CLAUDE.md` | 1 |
| tracked dirty | `Cargo.lock` | 1 |
| tracked dirty | `GEMINI.md` | 1 |
| tracked dirty | `README.md` | 1 |
| tracked dirty | `crates/abi-agent-host` | 5 |
| tracked dirty | `crates/abi-capability` | 2 |
| tracked dirty | `crates/abi-cli` | 2 |
| tracked dirty | `crates/abi-worker` | 5 |
| tracked dirty | `docs/contracts` | 2 |
| tracked dirty | `docs/spec` | 1 |
| tracked dirty | `tasks/goals.md` | 1 |
| tracked dirty | `tasks/todo.md` | 1 |
| untracked | `crates/abi-agent-host` | 13 |
| untracked | `crates/abi-control-plane` | 59 |
| untracked | `crates/abi-worker` | 8 |
| untracked | `docs/spec` | 2 |
| untracked | `docs/superpowers` | 6 |
| untracked | `docs/verification` | 2 |

Counts cover every path from the captured short-status inventory. The separate
local execution inventory retains the complete path list. Re-enumerate paths
with `git status --short --untracked-files=all` before any integration.

This documentation task uses `task/agent-main-plans-20261025`, based on
`69910bf61205458dec27dd4b616201e0a5f3898a`. Its worktree is separate from the
observed checkouts above. Existing uncommitted guidance was not copied into this
branch or claimed as this task's work; reconcile it with its owner at integration.

## Ownership and repository blockers

Use the mandatory tools/cargo.sh wrapper and declared adjacent WDBX layout. Coordinate modernization cohorts before integration; retain the authorization, lease and cancellation contracts and all desktop-platform qualification gaps.

Each existing worktree remains with its current writer until release is
confirmed. The inventory establishes branch custody, not writer identity or
authorization to merge. An unpublished commit is retained local work, even when
its branch is already reachable from main. Do not discard it or its branch label
as cleanup.

## Integration checklist

- [ ] Re-read AGENTS.md, this plan, active product plans, refs and all worktree
  statuses. Confirm each selected writer has released an exact commit/file set.
- [ ] Preserve all refs and dirty/untracked work before integration; preserve
  required ignored assets separately and verify recovery. Keep app worktrees.
- [ ] Classify every branch as already reachable, selected, owner-held or deferred
  with a reason. Review main-only/branch-only commits and file differences;
  do not merge all branches merely because they exist.
- [ ] Use one manual integration branch/worktree for the selected changes and
  an isolated writable build/test environment. Reuse only documented sibling
  layouts. Commit released dirty work on its owning branch after preservation.
- [ ] Integrate selected commits in dependency order, resolve conflicts against
  existing contracts and retain failed checks and provenance pins.
- [ ] Run the documented gates below against the exact candidate revision;
  record command, revision, result and unmet platform/external prerequisites.
- [ ] Recheck main HEAD, canonical dirty state and writer ownership immediately
  before local integration. Advance main only if those guards still hold.
- [ ] Record resulting commit and remaining work. Preserve recovery refs and
  worktrees until their owners release them and ignored-asset custody is verified.

## Required checks and completion evidence

Run from this repository's root, respecting its wrapper/toolchain prerequisites:

```sh
./tools/check.sh
git diff --check
```

For this documentation task, use the repository's scoped documentation/instruction
checks and record actual results separately from the full integration gate.
No product gate is claimed by writing this plan. Readiness requires ownership
release, preserved work and passing applicable checks on the candidate; failures,
skips or unavailable prerequisites remain explicit blockers. Platform, hardware,
model and production qualification retain their existing independent criteria.
Publishing, remote merges, releases and deployment are outside this task.
