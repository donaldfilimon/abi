# Contributing

ABI accepts focused, reviewable changes.

## Workflow

1. Use the Zig toolchain pinned in [.zigversion](.zigversion).
2. Review [AGENTS.md](AGENTS.md), [tasks/todo.md](tasks/todo.md), and [tasks/lessons.md](tasks/lessons.md) before non-trivial work.
3. Run the mandatory tri-CLI consensus flow for non-trivial work before implementation.
4. Prefer small commits with a clear validation trail.
5. Update documentation when public behavior, commands, or file layout changes.

## Validation

Use the `zig-master` workflow as the default contract for Zig changes:

```bash
which zig
zig version
cat .zigversion
zig build toolchain-doctor
zig build gendocs -- --check --no-wasm --untracked-md
zig build check-docs
zig build typecheck
zig build cli-tests
zig build tui-tests
zig build full-check
zig build check-cli-registry
zig build verify-all
zig build check-workflow-orchestration-strict --summary all
```

If a command is blocked by the local environment, record the exact failure and distinguish it from repo-local regressions.

## Related Guides

- [AGENTS.md](AGENTS.md) for the canonical repo workflow contract
- [CLAUDE.md](CLAUDE.md) for the local quick reference wrapper
- [SECURITY.md](SECURITY.md) for responsible disclosure guidance
- [docs/README.md](docs/README.md) for docs structure and generated outputs
