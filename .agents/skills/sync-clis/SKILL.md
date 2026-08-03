---
name: sync-clis
description: Sync canonical skills/personas/commands from central (~/.grok/skills + abi-mega) to all CLIs (grok, claude, codex, opencode, abi, agents, cursor, hermes, openclaw, factory, coreai, gemini). Idempotent. Launch with /sync-clis or launch.sh.
---
# /sync-clis

There are **two distinct sync mechanisms**. They are not substitutes — they move
different files between different directories. Know which one you are running.

## A. Central driver — what `/sync-clis` actually runs

`~/.grok/skills/sync-clis/launch.sh`
  -> `~/.grok/scripts/run-sync-clis.sh`  (tee wrapper; writes `sync.log` + `verify-evidence-main.txt`)
  -> `~/.grok/scripts/sync-clis.py --verbose`

- **Source of truth:** `~/.grok/skills` (`central.skills` in `~/.grok/sync-targets.json`).
- **Targets:** the 12 entries in `~/.grok/sync-targets.json` — grok, claude,
  codex, opencode, abi, agents, cursor, hermes, openclaw, factory, coreai, gemini.
- **Scope is narrow:** only the ~13 names in the script's `CORE_SKILLS` list
  (`sl`, `check-work`, `code-review`, `abi-doc-claims-sync`, `help`,
  `create-skill`, `imagine`, `docx`, `pptx`, `xlsx`, `sync-clis`, `swift`,
  `goal-ledger`) plus `CORE_PERSONAS`. It does **not** sync the full skill set.
- `opencode` gets a flat `command/<name>.md` wrapper instead of a `SKILL.md` subdir.
- `references/`, `scripts/`, `examples/`, `assets/` are **rmtree'd then copytree'd** —
  a destructive replace, not a merge.

## B. Repo launcher — in-repo mirrors

`~/abi/.agents/skills/sync-clis/launch.sh [--dry-run]`

- **Source:** `~/abi/.agents/skills/` (76 skills at last count).
- **Targets:** in-repo `~/abi/.claude/skills/` and `~/abi/.grok/` (64 each —
  the launcher skips 12 universal skills listed in its `case` statement).
- Copies `SKILL.md` plus `references/`/`examples/`; never copies `.sh` launchers.
- Its `--dry-run` is honest: it echoes every intended write.

The central driver never touches the in-repo mirrors, so running A alone leaves
`~/abi/.claude/skills/` and `~/abi/.grok/` stale. Run A, then B.

## Gotchas — verified, do not skip

1. **The python `--dry-run` lies by omission.** It reports a skill as changed
   only when the destination `SKILL.md` is *missing*; it never detects a
   *divergent* file. A clean dry-run says nothing about overwrites.
2. **The driver writes git-tracked files in `~/abi`.** The `abi` target's
   `skillsDir` is `~/abi/.agents/skills`, and the `CORE_SKILLS` loop copies into
   it unguarded. The seed-only "skip if it exists" guard protects only the
   later *abi-mega* branch, so `abi-doc-claims-sync` (in both lists) is written
   by the unguarded path first. **Diff before running on `main`:**
   ```bash
   for s in sl check-work code-review abi-doc-claims-sync help create-skill \
            imagine docx pptx xlsx sync-clis swift goal-ledger; do
     diff -q ~/.grok/skills/$s/SKILL.md ~/abi/.agents/skills/$s/SKILL.md \
       >/dev/null 2>&1 || echo "DIVERGENT/absent: $s"
   done
   ```
   Check the *direction* of any divergence before deciding — the repo copy is
   sometimes the stale one, in which case the write is a fix, not a clobber.
   Either way it dirties a tracked file: surface the diff, do not auto-commit.
3. **Markers rewrite on every run** — `.plugins-synced-from-central` in each
   target `skillsDir`, and `~/abi/src/plugins/.central-synced`. Both are
   gitignored, so they do not dirty the abi checkout. That marker is the only
   thing left under `~/abi/src/`; it is not source.
4. **`~/.claude/skills/sync-clis/` is a sync target, not a source.** Editing it
   is pointless — the next run overwrites it from `~/.grok/skills/sync-clis/`.
   Edit the canonical copy.
5. The repo launcher's `sed` rewrite of `^Base directory for this skill:` is a
   no-op against these files — that line is harness-injected at invocation, not
   stored in `SKILL.md`.
6. The interactive shell here has `noclobber` set; plain `cat > file` is refused.
   Use `cat >| file` when scripting edits by hand.

## ABI context

`~/abi` is a **nightly Rust** workspace (`crates/*`, `./tools/cargo.sh`,
`./tools/check.sh`). The Zig tree was removed in the Rust rewrite — synced skill
text must not reference `build.zig`, `zig build`, `.zigversion`, or `src/` as
ABI source. (Unrelated Zig projects such as `cell-lang` are fine to mention as
such.)
