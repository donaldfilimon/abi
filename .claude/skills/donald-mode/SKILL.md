---
name: donald-mode
description: >-
  Donald Filimon's agent style for ABI and multi-CLI work: terse path-anchored
  status, claim-honest verification, cursor/ branches with draft PRs, and
  mid-task skill routing. Use for Donald, /donald-mode, or requests to work in
  this style.
disable-model-invocation: true
---

# Donald mode

Working conventions for agents helping this user. Prefer `abi/AGENTS.md` for
toolchain and frozen surfaces. This skill is style and process only.

## Response style

- Lead with the verdict or status. Keep body short.
- Absolute paths when naming files or projects. A lone path drop means orient
  there and report status, do not ask what the path is.
- Tables for gates, tracks, and option menus. Bullets only when items are
  parallel. Skip essay wrap-ups.
- Sparse bold. No emoji decoration. No "I hope this helps."

## Autonomy

- "continue", "continue with all", or a mid-task slash skill means broaden and
  keep going. Do not stop at a green gate unless the human named a stop.
- Reversible work proceeds without asking. Pause for force-push, data deletion,
  or anything that cannot be undone without the human.
- When `/abi` is invoked for implementation, route through the `abi` subagent.
- Prefer scoped tracks over clean-slate rewrites. Confirm scope before a large
  plan.

## Review and verify

- ABI done bar: `./build.sh check` on macOS. Use `./build.sh full-check` when
  the change touches integration, benchmarks, or TUI.
- After a green gate (or on "test all features"), also smoke the live binary:
  `./zig-out/bin/abi backends`, then representative commands (`complete`,
  `wdbx`, `dashboard --once`, `plugin list`, `scheduler status`). Build gates
  alone are not the full verify bar when the human asks to test features.
- Confirm `zig version` matches `abi/.zigversion` before trusting a build.
- Interactive dashboard/TUI: `.agents/skills/run-tui/tui.sh`. Do not put
  Homebrew ahead of the pinned Zig on `PATH`.
- Reviews: `/review-bugbot` and `/review-security` when the human asks for
  review. Empty diff on clean `main` is a valid result, say so and stop.
- Honest digests and labeled demos only. No fake live bridges when IPC or
  production capability is absent. Claims gate:
  `docs/contracts/external-claims-audit.mdx`.

## Process

- Branch from `origin/main` with a `cursor/` prefix. Never commit or push
  straight to `main`. Never force-push `main`.
- Land finished work via PR/merge. Prefer draft PRs when the create-PR flow
  offers draft. Do not leave stranded feature branches.
- Conventional Commits. Commit only when asked.

## Skills and CLI hygiene

- Project skill home for ABI is `.agents/skills/<name>/SKILL.md` (tracked).
  `.cursor/` is gitignored here. Mirror personal copies to
  `~/.cursor/skills/<name>/` (and `~/.codex/skills/<name>/` when installing
  for Codex) after content changes.
- Fix skills at the central source (`~/.grok/skills`, `~/plugins/abi-mega`),
  not at sync targets. Sync with `.agents/skills/sync-clis/launch.sh` or the
  home `/sync-clis` skill.
- "Improve all" / codebase self-improve: run `/abi-skills` together with the
  `self-improving-codebase-loop` skill (ABI profile
  `references/abi.md`). Bounded cycles, project gate, no force-push.
- Mid-task skill attach wins over inventing a parallel workflow. Broken skill
  mid-task: fix it in its own PR, do not silently work around it.
- Keep `CLAUDE.md` / `GEMINI.md` as thin redirects to `AGENTS.md`. Do not
  re-inflate them.

## References

- `AGENTS.md` (canonical ABI agent instructions)
- `tasks/todo.md` (active board)
- `.agents/skills/abi-doc-claims-sync/SKILL.md`
- `.agents/skills/sync-clis/SKILL.md` (or home sync-clis skill)
- `/abi-skills` and `self-improving-codebase-loop` (improve-all loop)
- Cursor built-ins: `review-bugbot`, `review-security`, `check-work`
