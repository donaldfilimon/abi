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
- **Scope is narrow:** only the ~14 names in the script's `CORE_SKILLS` list
  (`sl`, `check-work`, `code-review`, `abi-doc-claims-sync`, `help`,
  `create-skill`, `imagine`, `docx`, `pptx`, `xlsx`, `sync-clis`, `swift`,
  `goal-ledger`, `aggressive-macos-cleanup`) plus all seven task-role files in
  `CORE_PERSONAS` (`design-doc-reviewer`, `design-doc-writer`, `implementer`,
  `researcher`, `reviewer`, `security-auditor`, and `test-writer`). Codex is the
  only non-source persona destination. The driver does **not** sync the full
  skill set or the global Abbey charter.
- `opencode` gets a flat `command/<name>.md` wrapper instead of a `SKILL.md` subdir.
- `references/`, `scripts/`, `examples/`, and `assets/` are compared by entry
  type and bytes. A divergent destination is **rmtree'd then copytree'd**; an
  identical tree is left untouched. Replacement is still destructive, not a
  merge, when a difference exists.

## B. Repo launcher — in-repo mirrors

`~/dev/active/abi/.agents/skills/sync-clis/launch.sh [--dry-run]`

- **Source:** `~/dev/active/abi/.agents/skills/`.
- **Targets:** in-repo `~/dev/active/abi/.claude/skills/` and
  `~/dev/active/abi/.grok/` (when present —
  the launcher skips 12 universal skills listed in its `case` statement).
- Copies `SKILL.md` plus `references/`/`examples/`; never copies `.sh` launchers.
- Its `--dry-run` is honest: it echoes every intended write.

The central driver never touches the in-repo mirrors, so running A alone leaves
`~/dev/active/abi/.claude/skills/` and `~/dev/active/abi/.grok/` stale. Run A,
then B.

## Gotchas — verified, do not skip

1. **The python `--dry-run` lies by omission.** It reports a skill as changed
   only when the destination `SKILL.md` is *missing*; it never detects a
   *divergent* file. A clean dry-run says nothing about overwrites.
2. **The driver writes git-tracked files in `~/dev/active/abi`.** The `abi`
   target's `skillsDir` is `~/dev/active/abi/.agents/skills`, and the
   `CORE_SKILLS` loop copies into it unguarded. The seed-only "skip if it
   exists" guard protects only the later *abi-mega* branch, so
   `abi-doc-claims-sync` (in both lists) is written by the unguarded path first.
   **Diff before running on `main`:**
   ```bash
   for s in sl check-work code-review abi-doc-claims-sync help create-skill \
            imagine docx pptx xlsx sync-clis swift goal-ledger; do
     diff -q ~/.grok/skills/$s/SKILL.md \
       ~/dev/active/abi/.agents/skills/$s/SKILL.md \
       >/dev/null 2>&1 || echo "DIVERGENT/absent: $s"
   done
   ```
   Check the *direction* of any divergence before deciding — the repo copy is
   sometimes the stale one, in which case the write is a fix, not a clobber.
   Either way it dirties a tracked file: surface the diff, do not auto-commit.
3. **The marker is deterministic.** `.plugins-synced-from-central` in each
   target `skillsDir` records only the target identity and is rewritten only
   when missing or divergent. The retired ABI root `src/` tree is not a sync
   target and must not be recreated. ABI Mega is discovered from
   `central.abiMega` in `~/.grok/sync-targets.json`. A complete unchanged second
   run must report `0 actions/changes`.
4. **`~/.claude/skills/sync-clis/` is a sync target, not a source.** Editing it
   is pointless — the next run overwrites it from `~/.grok/skills/sync-clis/`.
   Edit the canonical copy.
5. The repo launcher's `sed` rewrite of `^Base directory for this skill:` is a
   no-op against these files — that line is harness-injected at invocation, not
   stored in `SKILL.md`.
6. The interactive shell here has `noclobber` set; plain `cat > file` is refused.
   Use `cat >| file` when scripting edits by hand.

7. **`abi-mega` is a live dependency — never archive or delete it.** It lives at
   `~/dev/active/plugins/abi-mega` (moved twice: `~/plugins/` → `~/Projects/plugins/`
   2026-08-04 → `~/dev/active/plugins/` 2026-08-09; the manifest
   `~/.grok/sync-targets.json` was repointed both times and all 14 sync paths
   verified). Central sources: `~/.grok/skills` (14 core synced skills) + abi-mega's `skills/`,
   plus `~/.grok/bundled/personas` and `~/.grok/bundled/roles`. The sync recreates
   `~/tmp/grok-goal-scratch/implementer/` for logs — that dir reappearing is
   expected, not clutter.
8. **Two sync mechanisms; the central one does not cover the other's targets.**
   (A) the central driver above; (B) `~/dev/active/abi/.agents/skills/sync-clis/launch.sh`,
   which mirrors abi's in-repo skills into `~/dev/active/abi/.claude/skills` and
   `~/dev/active/abi/.grok`. **Run A, then B** — A alone leaves the in-repo mirrors
   stale. B is idempotent (run 2026-08-19: 0 git changes).
9. **opencode gets flat `command/<name>.md` wrappers, and `sync-clis.py` writes
   them only `if not wrapper.exists()` — it can never repair or update one.**
   New wrappers copy the complete central `SKILL.md` plus a source marker;
   existing wrappers remain untouched. Found 2026-08-19: all 11 wrappers had accumulated
   duplicate YAML frontmatter, so the visible description was the stub `synced from
   central`. Rebuilt all 13 by hand (backup:
   `~/Archive/2026-08-18-home-cleanup/opencode-command-before-2026-08-19/`). The
   first-six cap was removed 2026-08-23 so new core skills can reach OpenCode;
   the missing-only behavior still means divergent wrappers need manual repair.
10. Parsing a `SKILL.md` frontmatter description with a regex is wrong — many use
    folded scalars (`description: >`), and a naive pattern captures the literal
    `>`. Parse the block scalar properly.

## ABI context

`~/dev/active/abi` is a **nightly Rust** workspace (`crates/*`,
`./tools/cargo.sh`, `./tools/check.sh`). The Zig tree was removed in the Rust
rewrite — synced skill text must not reference `build.zig`, `zig build`,
`.zigversion`, or `src/` as ABI source. (Unrelated Zig projects such as
`cell-lang` are fine to mention as such.)
