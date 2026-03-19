---
title: database API
purpose: Generated API reference for database
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2905+5d71e3051
---

# database

> Database Feature Module

Canonical public entrypoint for vector database operations.
Delegates to the core unified database engine (`core/database/mod.zig`).

This module re-exports the full public API surface so that callers get
identical types and functions regardless of whether `feat_database` selects
`mod.zig` (this file) or `stub.zig` (no-op).

**Source:** [`src/features/database/mod.zig`](../../src/features/database/mod.zig)

**Build flag:** `-Dfeat_database=true`

---

## API

### <a id="pub-fn-isenabled-bool"></a>`pub fn isEnabled() bool`

<sup>**fn**</sup> | [source](../../src/features/database/mod.zig#L86)

Check if the database module is enabled at compile time.



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ABI expects a host-built or otherwise known-good Zig matching `.zigversion`. If stock prebuilt Zig is linker-blocked, record `zig fmt --check ...` plus `./tools/scripts/run_build.sh typecheck --summary all` as fallback evidence while replacing the toolchain.
