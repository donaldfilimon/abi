# Claude Guide

Claude is a lightweight quick reference for ABI work. It is not the canonical
policy source.

## Canonical Sources

- Repo workflow: [AGENTS.md](AGENTS.md)
- Active execution tracker: [tasks/todo.md](tasks/todo.md)
- Correction log: [tasks/lessons.md](tasks/lessons.md)
- Zig validation contract:
  `[$zig-master](/Users/donaldfilimon/.codex/skills/zig-master/SKILL.md)`

## Quick Local Checks

```bash
which zig
zig version
cat .zigversion
zig build toolchain-doctor
zig build check-docs
zig build typecheck
zig build cli-tests
zig build tui-tests
zig build full-check
zig build check-cli-registry
zig build verify-all
zig build check-workflow-orchestration-strict --summary all
```

## Notes

- `build.zig` is the authoritative source for build step wiring.
- Generated docs live under [`docs/_docs/`](docs/_docs) and [`docs/api/`](docs/api).
- Use this file as a convenience wrapper; update `AGENTS.md` and `zig-master`
  when the underlying contract changes.
