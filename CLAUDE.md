# Development Guide

This is the current quick reference for local ABI development on Zig `0.16.0-dev.2694+74f361a5c`.

## Toolchain Baseline

```bash
which zig
zig version
cat .zigversion
zig build toolchain-doctor
```

## Common Checks

```bash
zig build check-docs
zig build typecheck
zig build cli-tests
zig build tui-tests
zig build full-check
zig build check-cli-registry
zig build verify-all
zig build check-workflow-orchestration-strict --summary all
```

## Useful Targeted Commands

```bash
zig test src/services/runtime/mod.zig
zig test src/services/tests/mod.zig --test-filter "pattern"
zig build benchmarks
zig build v3-lib
```

## Notes

- The authoritative build step definitions live in [build.zig](build.zig).
- Generated docs live under [`docs/_docs/`](docs/_docs) and [`docs/api/`](docs/api).
- Workflow and persona coordination rules live in [AGENTS.md](AGENTS.md).
