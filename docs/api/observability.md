---
title: observability API
purpose: Generated API reference for observability
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2905+5d71e3051
---

# observability

> Observability Module

Unified observability with metrics, tracing, and profiling.

## Features
- Metrics collection and export (Prometheus, OpenTelemetry, StatsD)
- Distributed tracing
- Performance profiling
- Circuit breakers and error aggregation
- Alerting rules and notifications

**Source:** [`src/features/observability/mod.zig`](../../src/features/observability/mod.zig)

**Build flag:** `-Dfeat_profiling=true`

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
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ABI expects a host-built or otherwise known-good Zig matching `.zigversion`. If stock prebuilt Zig is linker-blocked, record `zig fmt --check ...` plus `./tools/scripts/run_build.sh typecheck --summary all` as fallback evidence while replacing the toolchain.
