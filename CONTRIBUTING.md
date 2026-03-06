# Contributing

ABI accepts focused, reviewable changes.

## Workflow

1. Use the Zig toolchain pinned in [.zigversion](.zigversion).
2. Review [AGENTS.md](AGENTS.md) and project task tracking in [tasks/todo.md](tasks/todo.md) before non-trivial work.
3. Prefer small commits with a clear validation trail.
4. Update documentation when public behavior, commands, or file layout changes.

## Validation

Use the `zig-master` workflow as the default contract for Zig changes:

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
```

If a command is blocked by the local environment, record the exact failure and distinguish it from repo-local regressions.

## Related Guides

- [CLAUDE.md](CLAUDE.md) for the current development quick reference
- [SECURITY.md](SECURITY.md) for responsible disclosure guidance
- [docs/README.md](docs/README.md) for docs structure and generated outputs
