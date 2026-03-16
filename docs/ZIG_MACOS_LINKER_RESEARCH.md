---
title: Zig on macOS 26+: ABI Linker Notes
purpose: Research and workarounds for Darwin linker failures
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2905+5d71e3051
---

# Zig on macOS 26+: ABI Linker Notes

This document captures the Darwin linker failure mode that affects ABI when a
stock prebuilt Zig cannot link the build runner on newer macOS releases.

## The important part

When this failure happens, the first broken binary is usually the build runner
that executes `build.zig`. That means:

- the failure happens before your `build.zig` logic runs
- toggling `use_llvm` or other build settings inside `build.zig` cannot fix that first failure
- the practical fixes are wrapper-based validation, compile-only checks, or a Zig toolchain built on the current machine

## Typical symptoms

Common undefined symbols include:

- `__availability_version_check`
- `_arc4random_buf`
- `_abort`
- `_malloc_size`
- `_nanosleep`

The error typically appears during `zig build`, `zig build test`, or other
binary-emitting commands. If the output references `build_zcu.o`,
`libcompiler_rt.a`, or other early build-runner artifacts, assume the build
runner link is the first thing to verify.

## Why this happens

ABI tracks a Zig 0.16-dev nightly. On Darwin 26+, prebuilt Zig distributions
can drift from the host SDK or linker expectations. The recurring failure mode
is not a normal source compile error; it is a mismatch between the toolchain's
link environment and the current macOS system libraries.

Two details matter:

1. Darwin availability checks pull in system symbols such as `__availability_version_check`.
2. If Zig does not resolve the correct SDK / libSystem inputs for the host, the link fails before ABI code runs.

## What does not work

These are common wrong turns:

- Expecting `build.zig` changes to fix the initial build-runner link
- Running `zig fmt .` from the repo root instead of the repo-safe format surface
- Treating `use_lld = true` as a macOS fix
- Assuming older helper wrappers are still the canonical front door

ABI guidance is explicit here: never recommend `use_lld = true` for macOS
targets.

## Supported ABI workarounds

### 1. Use the build-runner wrapper

For build-system steps that need `zig build` semantics:

```bash
./tools/scripts/run_build.sh test --summary all
./tools/scripts/run_build.sh feature-tests --summary all
./tools/scripts/run_build.sh full-check
```

This is the fastest way to keep using the stock toolchain when the failure is
limited to the initial build-runner link.

### 2. Use direct no-link validation

For formatting:

```bash
zig fmt --check build.zig build src tools examples
./tools/scripts/fmt_repo.sh --check
```

For compile-only targeted checks:

```bash
zig test src/services/tests/mod.zig -fno-emit-bin
zig test src/features/database/mod.zig -fno-emit-bin
```

These do not replace full runtime validation, but they are useful when the
environment is linker-blocked.

### 3. Use wrapper and compile-only validation

ABI no longer carries a repo-local workaround toolchain. On blocked Darwin hosts,
use the wrapper for build-system behavior and use compile-only checks when the
host linker still cannot emit binaries.

```bash
./tools/scripts/run_build.sh test --summary all
zig fmt --check build.zig build src tools examples
zig test src/services/tests/mod.zig -fno-emit-bin
```

## Decision guide

Use this sequence when working locally on macOS 26+:

1. Need formatting only: run `zig fmt --check build.zig build src tools examples`
2. Need `zig build` behavior: run `./tools/scripts/run_build.sh <step>`
3. Need targeted syntax/type validation: use `zig test <path> -fno-emit-bin`
4. Need binary-emitting validation: use Linux CI or another host with a working Zig linker

## How to classify a failure

Treat it as the known Darwin linker issue when all of these are true:

- the failure happens during a binary-emitting Zig command
- the output reports undefined system or compiler runtime symbols
- the symbols are Darwin / libSystem related rather than project symbols
- the failure shows up before any ABI tests actually run

Treat it as a normal code issue when:

- you get syntax, type, import, or test failures
- the missing symbol is from ABI itself rather than libSystem or compiler runtime
- a compile-only command reproduces the issue without linking

## Repo implications

- `zig build lint` may be environment-blocked on affected Darwin hosts
- `zig build full-check` remains the canonical gate, but blocked hosts must record alternate evidence
- docs and task notes should record exactly which command failed and which fallback was used

## Related files

- `AGENTS.md`
- `CLAUDE.md`
- `tasks/lessons.md`
- `tools/scripts/run_build.sh`
