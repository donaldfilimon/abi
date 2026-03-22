---
title: API Documentation Coverage
purpose: Per-module documentation coverage of public symbols
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2934+47d2e5de9
---

# API Documentation Coverage

> Per-module documentation coverage of public symbols.

---
| Module | Documented | Total | Coverage |
| --- | --- | --- | --- |
| [app](app.md) | 24 | 24 | 100% |
| [config](config.md) | 10 | 10 | 100% |
| [errors](errors.md) | 16 | 16 | 100% |
| [registry](registry.md) | 18 | 18 | 100% |
| [benchmarks](benchmarks.md) | 4 | 4 | 100% |
| [gpu](gpu.md) | 9 | 9 | 100% |
| [inference](inference.md) | 0 | 0 | 100% |
| [runtime](runtime.md) | 89 | 89 | 100% |
| [ai](ai.md) | 0 | 0 | 100% |
| [cache](cache.md) | 12 | 12 | 100% |
| [database](database.md) | 1 | 1 | 100% |
| [search](search.md) | 10 | 10 | 100% |
| [storage](storage.md) | 9 | 9 | 100% |
| [acp](acp.md) | 10 | 10 | 100% |
| [cloud](cloud.md) | 19 | 19 | 100% |
| [gateway](gateway.md) | 10 | 10 | 100% |
| [ha](ha.md) | 12 | 12 | 100% |
| [mcp](mcp.md) | 0 | 0 | 100% |
| [messaging](messaging.md) | 9 | 9 | 100% |
| [mobile](mobile.md) | 10 | 10 | 100% |
| [network](network.md) | 6 | 6 | 100% |
| [observability](observability.md) | 0 | 0 | 100% |
| [pages](pages.md) | 9 | 9 | 100% |
| [web](web.md) | 7 | 7 | 100% |
| [analytics](analytics.md) | 13 | 13 | 100% |
| [auth](auth.md) | 5 | 5 | 100% |
| [compute](compute.md) | 0 | 0 | 100% |
| [connectors](connectors.md) | 5 | 5 | 100% |
| [desktop](desktop.md) | 0 | 0 | 100% |
| [documents](documents.md) | 0 | 0 | 100% |
| [foundation](foundation.md) | 26 | 26 | 100% |
| [lsp](lsp.md) | 0 | 0 | 100% |
| [platform](platform.md) | 12 | 12 | 100% |
| [tasks](tasks.md) | 30 | 30 | 100% |

**Overall: 385/385 symbols documented (100%)**


---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ABI expects a host-built or otherwise known-good Zig matching `.zigversion`. If stock prebuilt Zig is linker-blocked, record `zig fmt --check ...` plus `zig test <file> -fno-emit-bin` as fallback evidence while replacing the toolchain.
