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
- Repository mirrors: `.claude/skills/` and `.grok/`, synchronized by
  `.agents/skills/sync-clis/launch.sh`.
- OpenCode: `.opencode/skills` is a symlink to `.agents/skills`.
- Codex home skills: `~/.codex/skills/<name>/SKILL.md` are installed explicitly.
- ABI bundled plugins: 16 build-time fixtures under `crates/abi-plugins/plugins/` (includes
  `tui-plugin`); verify them with `.agents/skills/plugin-runtime-tester/plugins.sh`.
- ABI Mega source: `~/plugins/abi-mega/`; marketplace registration alone does not
  prove that the current version is installed.
- Live Rust pin: read repo-root `rust-toolchain.toml`. Always use `./tools/cargo.sh`
  (Homebrew stable `cargo` may shadow rustup nightly).

`crates/abi-plugins/plugins/` is intentionally absent and is rejected by
`modern-refactor/scripts/verify-phase1-scope.sh`. Do not recreate it as a health
check. Use the runtime tester plus the repository gates.

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
3. **Establish a baseline** — run `./tools/check.sh` without a PTY. Afterward,
   run `./tools/cargo.sh build -p abi-cli` because feature-stub smoke overwrites `target/debug/abi`.
4. **Refresh ABI Mega evidence**:
   - `~/plugins/abi-mega/skills/abi-goal-orchestrator/scripts/refresh-inventory.sh`
   - `~/plugins/abi-mega/skills/abi-markdown-auditor/scripts/scan-markdown.sh`
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
10. **Validate** — run `./tools/check.sh`, `./tools/cargo.sh clippy --workspace --all-targets -- -D warnings`,
    `.agents/skills/docs-validate/validate.sh`, and `./tools/check.sh` with
    nightly Rust. Restore the full CLI with `./tools/cargo.sh build -p abi-cli`.
11. **Log** — `skill-loop log abi-skills success` (or `partial`/`failure`) when
    telemetry is initialized.
12. **Integrate** — use a `cursor/` feature branch and PR; never force-push
    `main`. Re-inventory immediately before merge and cleanup.

## ABI Mega refresh

```bash
~/plugins/abi-mega/skills/abi-goal-orchestrator/scripts/refresh-inventory.sh \
  "$PWD" ~/plugins/abi-mega/assets/abi-current-inventory.md
~/plugins/abi-mega/skills/abi-markdown-auditor/scripts/scan-markdown.sh \
  "$PWD" ~/plugins/abi-mega/assets/abi-markdown-audit.md
```

These commands refresh local plugin assets. They do not install or upgrade the
Codex plugin. Confirm the source manifest version and installed-plugin state
separately through Plugin Management.

## Validation commands

```bash
./tools/check.sh
./tools/cargo.sh clippy --workspace --all-targets -- -D warnings
.agents/skills/plugin-runtime-tester/plugins.sh
.agents/skills/docs-validate/validate.sh
./tools/check.sh
./tools/cargo.sh build -p abi-cli
```

Run `./tools/check.sh` through pipes/non-interactive execution. Allocating a PTY
causes CLI dashboard smoke to enter interactive mode and invalidates the gate.

## Claim boundaries

- The 16 bundled plugins are build-time Rust modules, not sandboxed marketplace
  extensions or hot-reloadable code.
- Registry presence does not prove `plugin run` dispatch; use the runtime tester.
- Marketplace registration does not prove ABI Mega is installed or current.
- Skill Loop reports scanner findings that require review; a count alone does not
  prove every reference is actionable or broken.
