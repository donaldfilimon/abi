# ABI Mega Plugin

This repository documents a **local Codex personal plugin** (outside the git tree)
that consolidates ABI goals, roadmaps, specs, skills, and validation workflows.
It is **not** stranded branch work and does **not** ship inside `abi` releases.

## Location

- Plugin: `/Users/donaldfilimon/dev/active/plugins/abi-mega` (live dependency for `/sync-clis`; do not archive)
- Sync registry: `/Users/donaldfilimon/.grok/sync-targets.json` (`central.abiMega`)
- Generated inventory: `/Users/donaldfilimon/dev/active/plugins/abi-mega/assets/abi-current-inventory.md`

The plugin is an operator-local Codex artifact. It does not replace repository
source, tests, or docs. When the plugin and repo disagree, trust `Cargo.toml`,
`crates/`, `tools/check.sh`, golden/contract tests, and the current
Markdown sources in this repo. ABI now uses nightly Rust; always invoke Cargo
through `./tools/cargo.sh`.

## Included Skills

- `abi-goal-orchestrator`: turns `tasks/todo.md`, roadmap/spec docs, and
  nightly-Rust repo rules into executable implementation slices.
- `abi-doc-claims-sync`: keeps README/docs/instruction claims aligned with
  executable behavior and `docs/contracts/external-claims-audit.mdx`.
- `abi-surface-validator`: runs focused ABI CLI, MCP, WDBX, TUI, and docs gates.
- `abi-markdown-auditor`: scans repo Markdown, old roadmap plans, mirrored
  skills, and instruction files for stale claims and contract drift.

## Source Map

Primary inputs compiled into the plugin:

- `tasks/todo.md`
- `tasks/lessons.md`
- `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`
- `README.md`, `CHANGELOG.md`, `walkthrough.md`, `abi-threat-model.md`
- `docs/contracts/external-claims-audit.mdx`
- `docs/contracts/public-api.mdx`
- `docs/spec/*.mdx`
- `docs/superpowers/archive/plans/*.md`
- `docs/superpowers/archive/specs/ABI-MASTER-SPEC.md`
- `.agents/skills/*/SKILL.md`
- `.claude/skills/*/SKILL.md`
- `.claude/agents/*.md`

## Refresh

Refresh the inventory after changing TODOs, specs, skills, CLI/MCP contracts, or
validation guidance:

```bash
/Users/donaldfilimon/dev/active/plugins/abi-mega/skills/abi-goal-orchestrator/scripts/refresh-inventory.sh \
  /Users/donaldfilimon/dev/active/abi \
  /Users/donaldfilimon/dev/active/plugins/abi-mega/assets/abi-current-inventory.md
```

Run the plugin's focused ABI gates:

```bash
/Users/donaldfilimon/dev/active/plugins/abi-mega/skills/abi-surface-validator/scripts/run-fast-gates.sh \
  /Users/donaldfilimon/dev/active/abi
```

Audit Markdown and old plan files:

```bash
/Users/donaldfilimon/dev/active/plugins/abi-mega/skills/abi-markdown-auditor/scripts/scan-markdown.sh \
  /Users/donaldfilimon/dev/active/abi \
  /Users/donaldfilimon/dev/active/plugins/abi-mega/assets/abi-markdown-audit.md
```

Validate the four plugin skills (Codex `skill-creator` cache path; install
PyYAML into `/tmp/codex-pyyaml` if needed):

```bash
python3 -m pip install --target /tmp/codex-pyyaml pyyaml
VALIDATE="$HOME/.codex/plugins/cache/claude-plugins-official/skill-creator/local/skills/skill-creator/scripts/quick_validate.py"
for s in abi-goal-orchestrator abi-doc-claims-sync abi-surface-validator abi-markdown-auditor; do
  PYTHONPATH=/tmp/codex-pyyaml python3 "$VALIDATE" \
    "$HOME/dev/active/plugins/abi-mega/skills/$s"
done
```

There is no separate `validate_plugin.py` on this host; skill-level
`quick_validate.py` is the supported check. The live fan-out registration is
`~/.grok/sync-targets.json` → `central.abiMega`; legacy marketplace metadata is
not the location authority.

## Maintenance Rules

- Keep the plugin concise. Detailed repo facts belong in references or generated
  inventory, not in every `SKILL.md`.
- Do not turn Proposed/Partial roadmap items into current claims unless source,
  tests, or benchmark artifacts prove them.
- Preserve the frozen CLI command set and MCP 12-tool surface unless contract
  tests are deliberately updated.
- Use the plugin inventory as a planning aid, not as proof of completion.
