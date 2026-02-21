---
title: Contributing
description: Development workflow and contribution guidelines
section: Reference
order: 5
---

# Contributing

Thanks for helping improve ABI. This page covers the development workflow,
coding style, and PR expectations.

---

## Quick Reference

| Task | Command |
|------|---------|
| Build | `zig build` |
| Test | `zig build test --summary all` |
| Feature tests | `zig build feature-tests --summary all` |
| Format | `zig fmt .` |
| Validate flags | `zig build validate-flags` |
| CLI smoke test | `zig build cli-tests` |
| Full check | `zig build full-check` |
| Single file test | `zig test src/path/to/file.zig` |
| Zig version check | `zig run tools/scripts/check_zig_version_consistency.zig` |

---

## Workflow

| Step | Action | Details |
|------|--------|---------|
| 1 | Fork and clone | `git clone https://github.com/donaldfilimon/abi.git && cd abi` |
| 2 | Create branch | `git checkout -b feature/my-feature` |
| 3 | Make changes | Keep scope focused; do not mix refactors with behavior changes |
| 4 | Format | `zig fmt .` |
| 5 | Run full check | `zig build full-check` |
| 6 | Update docs | For any public API changes |
| 7 | Submit PR | Clear description, linked issues, passing tests |

Ensure your Zig toolchain matches the version pinned in `.zigversion`:

```bash
cat .zigversion
zig version
zig run tools/scripts/toolchain_doctor.zig
```

---

## Style Guide

| Rule | Convention |
|------|-----------|
| Indentation | 4 spaces, no tabs |
| Line length | Under 100 characters |
| Types | `PascalCase` |
| Functions / variables | `camelCase` |
| Config structs | `*Config` suffix |
| Imports | Explicit only (no `usingnamespace`) |
| Allocators | Prefer `std.ArrayListUnmanaged` over `std.ArrayList` |
| Logging | `std.log.*` in library code; `std.debug.print` only in CLI tools |
| Cleanup | Prefer `defer` / `errdefer` for resource management |
| Error handling | Specific error sets, not `anyerror` |

### Import Conventions

```zig
// Public API consumers
const abi = @import("abi");

// Feature modules (cannot import "abi" due to circular dep)
const shared = @import("../../services/shared/mod.zig");

// Internal sub-modules
const parent = @import("mod.zig");
```

### Format Specifiers (Zig 0.16)

Use modern format specifiers instead of manual conversions:

```zig
// Enums and errors: {t}
std.debug.print("Status: {t}\n", .{status});

// Byte sizes: {B} or {Bi}
std.debug.print("Size: {B}\n", .{file_size});

// Durations: {D}
std.debug.print("Elapsed: {D}\n", .{duration});
```

---

## Commit Convention

```
<type>: <short summary>
```

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change (no feature/fix) |
| `test` | Adding or updating tests |
| `chore` | Maintenance, CI, tooling |

Keep commits focused. Each commit should be a single logical change.

---

## Testing

ABI maintains two test baselines that must not regress:

| Suite | Expected | Command |
|-------|----------|---------|
| Main tests | 1270 pass, 5 skip | `zig build test --summary all` |
| Feature tests | 1534 pass | `zig build feature-tests --summary all` |

### Test types

- **Unit tests**: `*_test.zig` files alongside the code they test
- **Integration tests**: `src/services/tests/`
- **Parity tests**: Verify `mod.zig` and `stub.zig` export the same interface
- **CLI smoke tests**: `zig build cli-tests`
- **Flag validation**: `zig build validate-flags` (tests 34 feature flag combos)

### Hardware-gated tests

Tests that require specific hardware should skip gracefully:

```zig
test "gpu kernel dispatch" {
    if (!abi.gpu.backends.detect.moduleEnabled()) return error.SkipZigTest;
    // ... test body ...
}
```

### Test discovery

Use `test { _ = @import(...); }` to include submodule tests. Note that
`comptime {}` does **not** discover tests.

---

## Feature Module Changes

When modifying a feature module, follow the stub-sync rule:

1. Update `mod.zig` with the new public function
2. Update `stub.zig` with a matching signature that returns `error.FeatureDisabled`
3. Verify: `zig build -Denable-<feature>=false`

### Adding a new feature module

There are 8 integration points (see [Architecture](architecture.html)):

1. `build/options.zig` -- build flag
2. `build/flags.zig` -- validation matrix
3. `src/features/<name>/mod.zig` + `stub.zig`
4. `src/abi.zig` -- comptime import
5. `src/core/config/mod.zig` -- config field
6. `src/core/registry/types.zig` -- registry case
7. `src/core/framework.zig` -- framework integration
8. `src/services/tests/stub_parity.zig` -- parity test

Verify with `zig build validate-flags`.

---

## PR Checklist

Before submitting a pull request:

- [ ] Clear description of changes
- [ ] Linked issues (if applicable)
- [ ] `zig fmt .` run (no formatting diffs)
- [ ] `zig build test --summary all` passes (1270+ pass, 5 skip)
- [ ] `zig build feature-tests --summary all` passes (1534+ pass)
- [ ] Stub updated if `mod.zig` changed
- [ ] Documentation updated for public API changes
- [ ] Commit messages follow the `<type>: <summary>` convention

For a comprehensive check:

```bash
zig build full-check
```

---

## Security

Do not open public PRs for security vulnerabilities. Report them through
the process described in `SECURITY.md`.

---

## Related Pages

- [API Overview](api.html) -- Public API surface
- [Architecture](architecture.html) -- Module hierarchy and design patterns
- [Troubleshooting](troubleshooting.html) -- Common errors and debugging
- [Examples](examples.html) -- 36 runnable examples for reference

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
