---
name: donald-mode
description: >-
  Donald Filimon's agent style for ABI and multi-CLI work: terse path-anchored
  status, claim-honest verification, cursor/ branches with draft PRs, skill-loop
  path hygiene, sync-clis discipline, mid-task skill routing, Cell/Browserbase
  orientation, and CoreAI skill lanes. Use for Donald, /donald-mode, or requests
  to work in this style.
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
- Typo-tolerant ultra-short directives (`auto`, `run`, `run it`, `try harder`,
  `all next`, `do all for me`, `fix all`, `finsih`) still mean finish the
  remaining work.
- A failed terminal dump pasted as the next message is the ticket — diagnose
  that output; do not re-ask for context.

## Autonomy

- `continue`, `continue with all`, `do all`, `fix and do all`, `finalize`, or
  `merge all into main` means broaden and keep going. Do not stop at a green
  gate unless the human named a stop.
- After parallel or multi-agent work, `continue` means finish residuals of that
  same effort (remaining HIGH flags, sync, draft PR), not a new topic.
- Several slash-skills in one message plus `do all` / `continue and do all`
  means run every attached track (parallel when independent); report blockers
  per track. Directive + skill on one line
  (`continue /dispatching-parallel-agents`) is the same rule.
- When a confirmed plan says implement all todos, finish those todos — do not
  re-scope mid-plan.
- Reversible work proceeds without asking. Pause for force-push, data deletion,
  or anything that cannot be undone without the human.
- When `/abi` is invoked for implementation, route through the `abi` subagent.
- Prefer scoped tracks over clean-slate rewrites. Confirm scope before a large
  plan. For residual asks (e.g. next Metal op), return what exists + 1–2 natural
  next ops + files + test/contract impact + small/medium — no speculative rewrite.
- Mid-task slash-skill attaches win over inventing a parallel workflow.
- Device / host recovery: CLI-first; escalate tools when told `try harder` /
  `force pair` rather than stopping at "needs phone UI."

## Review and verify

- ABI done bar: `./tools/check.sh` on macOS. Use `./build.sh full-check` when
  the change touches integration, benchmarks, or TUI.
- After a green gate (or on "test all features"), also smoke the live binary:
  `./target/debug/abi backends`, then representative commands (`complete`,
  `wdbx`, `dashboard --once`, `plugin list`, `scheduler status`). Build gates
  alone are not the full verify bar when the human asks to test features.
- Confirm `zig version` matches `abi/rust-toolchain.toml` before trusting a build.
- Interactive dashboard/TUI: `.agents/skills/run-tui/tui.sh`. Do not put
  Homebrew ahead of the pinned Zig on `PATH`.
- Dual review default: when the human asks for review (`/review`, "security and
  code quality"), run both `/review-bugbot` and `/review-security`. Empty diff
  on clean `main` is a valid result, say so and stop.
- Honest digests and labeled demos only. No fake live bridges when IPC or
  production capability is absent. Claims gate:
  `docs/contracts/external-claims-audit.mdx`.

## Process

- Branch from `origin/main` with a `cursor/` prefix. Never commit or push
  straight to `main`. Never force-push `main`.
- Land finished work via PR/merge. Prefer draft PRs when the create-PR flow
  offers draft. Ask before marking ready or merging unless the human already
  said merge (`do all`, `finalize`, `merge all into main` count as said).
- After merge-after-green: sync local `main` to `origin/main`, drop the feature
  branch, run `/sync-clis` when skills changed, and re-check skill-loop if that
  was the track.
- Conventional Commits. Commit only when asked.
- Do not leave stranded feature branches after the work is merged.

## Skills and CLI hygiene

- Project skill home for ABI is `.agents/skills/<name>/SKILL.md` (tracked).
  `.cursor/` is gitignored here. Mirror personal copies to
  `~/.cursor/skills/<name>/` (and `~/.codex/skills/<name>/` when installing
  for Codex) after content changes.
- Fix skills at the central source (`~/.grok/skills`, `~/plugins/abi-mega`),
  not at sync targets. Sync with `.agents/skills/sync-clis/launch.sh` or the
  home `/sync-clis` skill (`python3 ~/.grok/scripts/sync-clis.py`).
- After `/sync-clis`, check that repo-adapted abi skills were not clobbered by
  stale central copies. If they were, restore from git, copy the fixed content
  into `~/.grok/skills`, then re-sync.
- skill-loop path rules (HIGH broken refs):
  - Prefer `$SKILL_DIR/...` for skill-local scripts so refs are not joined to
    the abi project root.
  - External docs get absolute paths (for example under
    `/Users/donaldfilimon/.grok/docs/`).
  - Conditional or install-only paths stay prose, not hard file tokens.
  - Rephrase false positives (`/tmp/...` examples, `Enum.MEMBER`); do not invent
    missing files.
  - Skip LOW content-drift banners caused by the skill-touching commit itself.
- Independent HIGH skill domains may run as parallel agents; each agent only
  edits its assigned skill trees.
- "Improve all" / codebase self-improve: run `/abi-skills` together with the
  `self-improving-codebase-loop` skill (ABI profile `references/abi.md`).
  Bounded cycles, project gate, no force-push.
- Broken skill mid-task: fix it in its own PR, do not silently work around it.
- Keep `CLAUDE.md` / `GEMINI.md` as thin redirects to `AGENTS.md`. Do not
  re-inflate them.

## Orientation (Abbey, Cell, Functions, CoreAI, Parallel)

- Abbey on Discord / `Package.swift` means
  `/Users/donaldfilimon/Desktop/AbbeyBot` (DiscordBM), not CoreAIAssistant.
  Orient with a path table when ambiguous, then ask which lane if still unclear.
- Cell / cell-lang / `build cel` / `build cell lang` means
  `/Volumes/ExtremeSSD/public/cell-lang` (SSD Zig 0.17 tree; not under `~/abi`).
  Default bar: `zig build`, `zig build run -- version`,
  `zig build run -- check examples/hello.cell`, `zig build test`. Bare
  `zig build run` exits 1 by design (CLI needs a subcommand).
- `/functions` means Browserbase Functions at
  `/Users/donaldfilimon/Desktop/AbbeyBot/functions` (bun `dev` / `deploy`;
  credentials in that project's `.env`). Do not re-scaffold unless asked.
- `/model-compression-exploration` or `/model-authoring`: follow the attached
  CoreAI skill. Need `torch` + `coreai-opt`, plus model/data/forward/quality
  (and NE vs GPU for authoring). Surface missing env; do not invent a parallel
  compression path.
- When `/parallel-setup` or Parallel skills are attached: use `parallel-cli`
  (install via Parallel install script or `pipx`). Auth via `parallel-cli login`
  or `PARALLEL_API_KEY`. Prefer Parallel web search/extract/research tools over
  inventing a second research path.

## References

- `AGENTS.md` (canonical ABI agent instructions)
- `tasks/todo.md` (active board)
- `tasks/goals.md` (coarse goal ledger)
- `.agents/skills/abi-doc-claims-sync/SKILL.md`
- `.agents/skills/sync-clis/SKILL.md` (or home sync-clis skill)
- `/abi-skills` and `self-improving-codebase-loop` (improve-all loop)
- Cursor built-ins: `review-bugbot`, `review-security`, `check-work`
