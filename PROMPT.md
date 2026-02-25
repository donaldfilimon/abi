---
title: "PROMPT"
tags: [requirements, standards, kpi, architecture]
---
# ABI Framework: Zig 0.16 Perfection Sprint

<!-- Replace this with your task description. -->

## Goal

Achieve production-grade code quality across the entire ABI framework. Every .zig file should compile cleanly, follow Zig 0.16 idioms, and pass all tests.

## Acceptance Criteria

- [ ] `zig build test --summary all` passes (1290+ pass, 6 skip)
- [ ] `zig build feature-tests --summary all` passes (2360+ pass)
- [ ] `zig build full-check` passes (format + tests + feature-tests + flag validation + CLI smoke)
- [ ] `zig build -Denable-ai=false` compiles cleanly (stub parity)
- [ ] `zig build -Denable-ai=true` compiles cleanly
- [ ] `zig build examples` compiles all examples
- [ ] All mod.zig/stub.zig pairs have matching public API signatures
- [ ] No deprecated Zig 0.13/0.14/0.15 patterns remain
- [ ] No empty or orphan files in tracked tree
- [ ] All CLI commands compile and pass smoke tests
- [ ] .gitignore covers all build artifacts (*.spv, *.wdbx, etc.)
- [ ] API_REFERENCE.md version matches v0.4.0

## Iteration Strategy

Each iteration should:
1. Run `zig build full-check` to find failures
2. Fix the highest-priority compilation error
3. Verify the fix with targeted build
4. Run full test suite to check for regressions
5. Move to next error

## Priority Order

1. Compilation errors (anything that prevents `zig build`)
2. Stub parity gaps (mod.zig vs stub.zig mismatches)
3. Test failures
4. Code quality (dead code, unused imports, const correctness)
5. Documentation drift

## Key Rules

- Zig 0.16: `std.Io.Dir.cwd()`, `.empty` init, `catch {`, `{t}` format
- No `@import("abi")` inside `src/features/` (circular imports)
- Every file ends with `test { std.testing.refAllDecls(@This()); }`
- Prefer `std.ArrayListUnmanaged` with explicit allocator passing
