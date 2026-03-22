---
title: ai API
purpose: Generated API reference for ai
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2962+08416b44f
---

# ai

> AI feature facade.

This top-level module presents the canonical `abi.ai` surface for framework
code, tests, and external callers. Compatibility aliases delegate here while
the stub-facing contract stays aligned with `stub.zig`.

**Source:** [`src/features/ai/mod.zig`](../../src/features/ai/mod.zig)

**Build flag:** `-Dfeat_ai=true`

---

## API

No documented public symbols were discovered.



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ABI expects a host-built or otherwise known-good Zig matching `.zigversion`. If stock prebuilt Zig is linker-blocked, record `zig fmt --check ...` plus `zig test <file> -fno-emit-bin` as fallback evidence while replacing the toolchain.
