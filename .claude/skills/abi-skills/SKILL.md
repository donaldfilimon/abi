---
name: abi-skills
description: Coordinate ABI codebase health, skill telemetry, bundled-plugin runtime checks, ABI Mega inventories, nightly Rust gates, and cross-CLI skill synchronization. Use for full ABI health reviews and claim-honest skill/plugin improvement cycles.
---

# abi-skills — ABI codebase health and skill synchronization

Use this skill for bounded, reviewable improvement cycles across the ABI repository,
its canonical skills, the 16 bundled ABI plugin fixtures, and the local ABI Mega
Codex plugin.

## Sources of truth

- Repository instructions: `AGENTS.md`, then `tasks/lessons.md` and `tasks/todo.md`.
- Nightly pin: `rust-toolchain.toml` and `.github/workflows/ci.yml` must agree.
- Canonical repository skills: `.agents/skills/`.
- Repository mirrors: `.claude/skills/`, synchronized by
  `.agents/skills/sync-clis/launch.sh`. The launcher also targets `.grok/` when
  that directory is present; it is absent from this repository (verified
  2026-08-22), so `.claude/skills/` is the only live in-repo mirror.
- OpenCode: `.opencode/skills` is a symlink to `.agents/skills`.
- Codex home skills: `~/.codex/skills/<name>/SKILL.md` are installed explicitly.
- ABI bundled plugins: 16 build-time fixtures under
  `crates/abi-plugins/plugins/` (including `tui-plugin`); verify manifest,
  registry, and runtime-dispatch agreement with
  `.agents/skills/plugin-runtime-tester/plugins.sh`.
- ABI Mega source: `~/dev/active/plugins/abi-mega/`; marketplace registration alone does not
  prove that the current version is installed.
- Live Rust pin: read repo-root `rust-toolchain.toml`. Always use `./tools/cargo.sh`
  (Homebrew stable `cargo` may shadow rustup nightly).

## Two traps that have already cost a session

**The home copy of this skill is not synchronized, and it drifts.** `/sync-clis`
copies only the names in `CORE_SKILLS` in `~/.grok/scripts/sync-clis.py`, and
`abi-skills` is not among them (verified 2026-08-22). Nothing keeps
`~/.claude/skills/abi-skills/SKILL.md` in step with the canonical file here, and
on 2026-08-22 that home copy was still the pre-rewrite Zig text — prescribing
`.zigversion`, `~/.zvm` PATH prefixes, `./build.sh check-parity`, and fixtures
under `src/plugins/`, none of which exist. A session invoked through the home
skill therefore starts from instructions the repository contradicts. **Diff the
copy you were handed against `.agents/skills/abi-skills/SKILL.md` before
following it**, and when you correct this file, push the same text to
`~/.claude/skills/abi-skills/SKILL.md` and `~/.codex/skills/abi-skills/SKILL.md`
by hand, because the launcher will not.

**Do not run `sync-clis/launch.sh` from a worktree.** The launcher stamps an
absolute `Base directory for this skill: <checkout>/.claude/skills/<name>` line
into each mirrored `SKILL.md`, so the mirror content depends on which checkout
produced it. Run from `~/dev/active/abi-wt-.../` it rewrites those lines to the
worktree path (measured 2026-08-22: 5 files — `codebase-analysis`,
`modern-patterns`, and the three `refactor-*` skills), and committing that
points every mirror at a directory that disappears when the worktree is removed.
Sync from the main checkout, or revert any file whose only diff is that line.

**An `abi` worktree must be created as a `~/dev/active/` sibling.** `Cargo.toml`
reaches the substrate through relative paths (`../wdbx/crates/…`), so a worktree
placed anywhere else resolves them to nothing and cargo fails with `no matching
package named abi-compute` — an error that names the consumer and reads as a
broken repository. `~/dev/active/abi-wt-<topic>-<date>/` satisfies it. For the
same reason, `git fetch && git status` in `abi`, `abbey`, and `wdbx` before
believing any cargo error in any of them: a stale sibling breaks the other two,
and the default build hides it because both deps are optional.

The retired root `src/` implementation tree is fully absent. Do not recreate
the Zig-era tree as a health check. The live Rust fixtures are
`crates/abi-plugins/plugins/` and must remain in manifest/compile-time/runtime
parity.

## Skill Loop

Skill Loop is the optional npm package under the `stylusnexus` scope named
`skill-loop-cli`, pinned to version `0.3.3` (declared in repo `.mcp.json` /
`opencode.json`). A missing bare `skill-loop` command is expected on hosts
without a global install. Prefer the one-shot form (package + pin via env so
scanners do not treat the npm locator as a repo path):

```bash
SKILL_LOOP_PKG="@stylusnexus/skill-loop-cli"
SKILL_LOOP_VER="0.3.3"
npx -y -p "${SKILL_LOOP_PKG}@${SKILL_LOOP_VER}" skill-loop <command>
```

The relevant commands are:

| Need | Command |
| --- | --- |
| Build or refresh the registry | `skill-loop init` |
| Show health | `skill-loop status` |
| Inspect stale content and references | `skill-loop inspect` |
| Log this workflow | `skill-loop log abi-skills <outcome>` |

There is no `skill-loop scan` command. Counts and broken-reference totals are
live telemetry, not durable capability claims.

## Workflow

1. **Freeze the target** — inspect `git status --short --branch`,
   `git worktree list --porcelain`, branches, stashes, and remotes. If another
   process moves the checkout, stop and use an isolated worktree.
2. **Select the toolchain locally** — use `./tools/cargo.sh` for all ABI gates
   (Homebrew stable `cargo` may shadow rustup nightly).
3. **Establish a baseline** — run `./tools/check.sh` non-interactively. Build
   `abi-cli` explicitly before user-facing smoke so the executable being tested
   is current and the evidence names the binary-producing command.
4. **Refresh ABI Mega evidence**:
   - `~/dev/active/plugins/abi-mega/skills/abi-goal-orchestrator/scripts/refresh-inventory.sh`
   - `~/dev/active/plugins/abi-mega/skills/abi-markdown-auditor/scripts/scan-markdown.sh`
5. **Inspect skills** — run Skill Loop `init`, `status`, and `inspect` when the
   npm tool is available; otherwise perform a targeted manual reference audit.
6. **Fix a bounded slice** — repair actionable stale paths or false claims.
   Preserve template placeholders and intentionally external references.
7. **Verify plugins** — run
   `.agents/skills/plugin-runtime-tester/plugins.sh`; do not infer runtime
   dispatch from registry listing alone.
8. **Synchronize mirrors** — preview with
   `.agents/skills/sync-clis/launch.sh --dry-run`, then run the launcher.
9. **Install Codex skill text** — copy the corrected `SKILL.md` to the matching
   `~/.codex/skills/<name>/SKILL.md`. Companion-resource parity is a separate
   policy decision.
10. **Validate** — run `./tools/check.sh` once (it already contains fmt,
    clippy, build, tests, and docs; the older form listed it twice and named
    clippy separately) plus `.agents/skills/docs-validate/validate.sh`. Rebuild
    with `./tools/cargo.sh build -p abi-cli` only if a smoke step replaced the
    binary.
11. **Log** — `skill-loop log abi-skills success` (or `partial`/`failure`) when
    telemetry is initialized.
12. **Integrate** — use a `cursor/` feature branch and PR; never force-push
    `main`. Re-inventory immediately before merge and cleanup.

## ABI Mega refresh

```bash
~/dev/active/plugins/abi-mega/skills/abi-goal-orchestrator/scripts/refresh-inventory.sh \
  "$PWD" ~/dev/active/plugins/abi-mega/assets/abi-current-inventory.md
~/dev/active/plugins/abi-mega/skills/abi-markdown-auditor/scripts/scan-markdown.sh \
  "$PWD" ~/dev/active/plugins/abi-mega/assets/abi-markdown-audit.md
```

These commands refresh local plugin assets. They do not install or upgrade the
Codex plugin. Confirm the source manifest version and installed-plugin state
separately through Plugin Management.

## Validation commands

```bash
./tools/check.sh                                # full gate: fmt, clippy, build, tests, docs
.agents/skills/plugin-runtime-tester/plugins.sh # registry + run dispatch, all 16 fixtures
.agents/skills/docs-validate/validate.sh
./tools/cargo.sh build -p abi-cli               # only if a smoke step replaced the binary
```

**Commands that no longer exist.** `./build.sh check-parity` and `./build.sh
lint` both exit 2 with `unknown target` (measured 2026-08-22). `build.sh` is a
compatibility shim that forwards only `check`, `cli`, `mcp`, `test`, and `fmt`;
`./build.sh -l` lists what it actually accepts. Any `.zigversion` or
`~/.zvm/$(cat .zigversion)` PATH prefix is dead — the file does not exist after
the Rust rewrite.

Run `./tools/check.sh` non-interactively as a matter of habit, but do not
justify it with the old dashboard-smoke reason. Verified 2026-08-22: `check.sh`
invokes no TUI step at all — it covers toolchain versions, repository policy
tests, the Abbey contract corpus, Rust source-size limits, fmt, clippy, build,
tests, Metal/CUDA device features, benchmark regression, and docs. A PTY does
not invalidate it. The interactive-mode hazard is real, but it belongs to the
TUI surfaces `check.sh` never calls — `.agents/skills/dashboard-smoke/dashboard.sh`
and `tools/run_tui_smoke.sh`. Run *those* non-interactively.

Capture gate results by redirecting to a file and echoing `$?` from the command
itself. Piping to `tail` reports `tail`'s exit status and has manufactured a
false green on this machine before.

## Claim boundaries

- The 16 bundled plugins are build-time Rust modules, not sandboxed marketplace
  extensions or hot-reloadable code.
- Registry presence does not prove `plugin run` dispatch; use the runtime tester.
- Marketplace registration does not prove ABI Mega is installed or current.
- Skill Loop reports scanner findings that require review; a count alone does not
  prove every reference is actionable or broken.
