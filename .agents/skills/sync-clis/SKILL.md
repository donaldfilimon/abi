---
name: sync-clis
description: Synchronize the explicitly cataloged portable skills and runtime-native task-agent and command adapters across Grok, Claude, Codex, OpenCode, ABI, shared agents, Cursor, Hermes, OpenClaw, Factory, CoreAI, and Gemini. Use for /sync-clis, cross-CLI skill drift, agent adapter drift, or OpenCode command repair.
---

# /sync-clis

Two complementary mechanisms exist. Run them in this order.

## A. Central cross-CLI synchronization

Entry points:

- `~/.grok/skills/sync-clis/launch.sh`
- `~/.grok/scripts/run-sync-clis.sh`
- `python3 ~/.grok/scripts/sync-clis.py --verbose`

Authoritative configuration:

- Manifest and artifact catalog: `~/.grok/sync-targets.json`
- Portable skill source: `~/.grok/skills`
- Task persona source: `~/.grok/bundled/personas`
- Role source: `~/.grok/bundled/roles`
- ABI Mega seed source: `~/dev/active/plugins/abi-mega/skills`

The catalog distinguishes four classes:

1. `portableSkills`: user-maintained canonical skills synchronized to declared targets.
2. `taskAgents`: central personas and roles rendered through runtime-native adapters.
3. `externalLinkedSkills`: vendor/plugin-owned symlinks recorded for inventory but not mirrored as canonical content.
4. `repositorySkillRoots`: repository-specific skills that remain under repository authority.

Unmatched target files are preserved. The driver updates only names explicitly
listed in the catalog. Supporting `references/`, `scripts/`, `examples/`,
and `assets/` trees are synchronized as a source projection; destination-only
entries are not deleted.

### Runtime adapters

- Codex task agents: `~/.codex/agents/*.toml`
- Claude Code subagents: `~/.claude/agents/*.md`
- OpenCode subagents: `~/.config/opencode/agents/*.md`
- OpenCode skills: `~/.config/opencode/skills/<name>/SKILL.md`
- OpenCode commands: both the documented `commands/*.md` surface and the
  installed version's `command/*.md` compatibility surface

The OpenCode command renderer writes one frontmatter block: the canonical
`SKILL.md` content followed by one source marker. Existing managed wrappers
are compared and repaired, not merely seeded when absent.

### Dry-run contract

`--dry-run` performs the same byte and projected-tree comparisons as an apply
run without writing managed targets. It reports missing and divergent skills,
agents, commands, supporting files, and markers.

A successful apply must be followed by a second full run. Acceptance is:

```text
Done. 0 actions/changes.
```

A zero-change run is meaningful only after the driver and manifest validate and
all requested targets were included.

## B. ABI in-repository mirror synchronization

Run after the central pass:

```bash
~/dev/active/abi/.agents/skills/sync-clis/launch.sh [--dry-run]
```

This mechanism mirrors ABI's repository-authoritative skills into its
`.claude/skills/` and `.grok/` surfaces. The central driver does not replace
this in-repository mirror step.

Before applying it:

1. Inspect `git -C ~/dev/active/abi status --short --branch`.
2. Preserve unrelated dirty work.
3. Run the launcher's dry-run and review every named write.
4. Apply.
5. Confirm the resulting ABI diff is limited to intended mirror changes.
6. Run it again and require no further writes.

## Safety invariants

- Never edit a synchronized target copy when the canonical central source is the
  intended authority.
- Never treat `~/.codex/memories` as a task-agent destination.
- Never copy external/vendor symlinks into the portable catalog without first
  choosing and preserving their real upstream authority.
- Never delete unmatched runtime-native skills, agents, commands, or plugins.
- Never delete or archive `~/dev/active/plugins/abi-mega`; it is a live source.
- ABI's repo-adapted ABI Mega skills are seed-only and are not overwritten.
- The ABI repository is a nightly Rust workspace; do not reintroduce removed
  Zig-era paths or commands.
- Central synchronization may intentionally dirty tracked ABI skill files when
  canonical portable content changed. Inspect the diff and run repository gates
  before committing.
- Preserve recovery evidence before catalog repair or any destructive retirement.

## Verification checklist

```bash
python3 -c 'from pathlib import Path; compile(Path.home().joinpath(".grok/scripts/sync-clis.py").read_text(), "sync-clis.py", "exec")'
python3 -m json.tool ~/.grok/sync-targets.json >/dev/null
python3 ~/.grok/scripts/sync-clis.py --dry-run --verbose
~/.grok/skills/sync-clis/launch.sh
~/dev/active/abi/.agents/skills/sync-clis/launch.sh
~/.grok/skills/sync-clis/launch.sh
```

Then verify:

- the final central run reports `0 actions/changes`;
- every managed direct skill copy matches its source `SKILL.md`;
- OpenCode's resolved agent list includes the managed task agents;
- Codex agent TOML parses;
- Claude and OpenCode Markdown agents have exactly one frontmatter block;
- ABI's final diff is reviewed and its matching gate is run before closeout.
