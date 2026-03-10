# ABI Agent FAQ

This file is the quick-reference companion to `AGENTS.md`. Use it for the
details that are easy to forget during implementation or review.

## Startup checklist

Before non-trivial work:

1. Read `AGENTS.md`, `CONTRIBUTING.md`, `CLAUDE.md`, `tasks/todo.md`, and `tasks/lessons.md`.
2. Confirm the Zig pin in `.zigversion`.
3. Add or update the active task entry in `tasks/todo.md`.
4. Prefer the smallest change set that fixes the real drift or bug.

## Canonical commands

### Build and test

```bash
zig build
zig build test --summary all
zig build feature-tests --summary all
zig build full-check
zig build verify-all
```

### Formatting

```bash
zig build fix
zig build lint
./tools/scripts/fmt_repo.sh --check
zig fmt --check build.zig build src tools examples
```

Never run `zig fmt .` from the repo root.

### Docs and CLI metadata

```bash
zig build gendocs
zig build gendocs -- --check --no-wasm --untracked-md
zig build check-docs
zig build refresh-cli-registry
zig build check-cli-registry
```

## Public surface reminders

- External consumers import `@import("abi")`, which resolves to `src/root.zig`.
- `src/abi.zig` is the internal composition layer, not the external package root.
- The canonical database public surface is `abi.features.database`.
- The canonical behavior-profile surface is `abi.features.ai.profiles`.
- `abi.features.ai.personas` still exists as a compatibility alias during phase 4.

## Feature module rules

- Every `src/features/<name>/` module must keep `mod.zig` and `stub.zig` public signatures aligned.
- Feature modules should use relative imports inside the feature tree.
- Files listed in `build/test_discovery.zig` must compile standalone.
- Cross-directory relative imports above the module root are a recurring source of broken targeted tests.

## Docs rules

- `docs/api/` and `docs/plans/` are generator-owned; fix `tools/gendocs/` instead of hand-editing those outputs.
- Update docs when public API names, commands, file layout, or workflow rules change.
- If you change CLI command metadata or command layout, refresh the CLI registry snapshot.

## Darwin / toolchain rules

- On macOS 26+, stock Zig may fail before `build.zig` runs because the build runner itself cannot link.
- Use `./tools/scripts/run_build.sh <step>` when you need build-system behavior with the stock toolchain.
- Use `zig fmt --check build.zig build src tools examples` for no-link formatting validation.
- Use `zig test <path> -fno-emit-bin` for compile-only targeted validation.
- Use `abi bootstrap-zig ...` for the repo-local bootstrap Zig bridge.
- Never recommend `use_lld = true` for macOS targets.

## Completion checklist

Before closing a task:

1. Verify the touched files match the intended scope.
2. Run the strongest validation the current environment allows.
3. Record blockers or environment limits precisely.
4. Update `tasks/todo.md` with outcomes and residual risk.
5. If you fixed a recurring pitfall, add the prevention rule to `tasks/lessons.md`.
