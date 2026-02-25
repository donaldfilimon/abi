---
title: "PLAN"
tags: [planning, sprint, development]
---
# Current Development Focus
> **Codebase Status:** Synced with repository as of 2026-02-25.

<p align="center">
  <img src="https://img.shields.io/badge/Sprint-Active-blue?style=for-the-badge" alt="Sprint Active"/>
  <img src="https://img.shields.io/badge/Main_Tests-1290%2F1296-success?style=for-the-badge" alt="Main Tests"/>
  <img src="https://img.shields.io/badge/Feature_Tests-2360%2F2365-success?style=for-the-badge" alt="Feature Tests"/>
</p>

## This Sprint

**Focus: CLI/TUI Developer Experience & Refactoring**

### In Progress
- [ ] TUI refactoring: extract base `InteractiveDashboard` pattern from duplicated dashboard code
- [ ] Centralized keybinding dispatch for TUI panels
- [ ] Reduce panel code duplication (brain.zig, gpu.zig, model.zig share identical boilerplate)

### Completed This Sprint
- [x] **Toolchain fix** — macOS 26 Xcode-beta SDK dropped `arm64-macos` from TBD stubs; fixed via `DEVELOPER_DIR=/Library/Developer/CommandLineTools` (2026-02-25)
- [x] **Version bump** — `.zigversion` updated from `0.16.0-dev.2637+6a9510c0e` to `0.16.0-dev.2653+784e89fd4` across 50 files (2026-02-25)
- [x] **New CLI commands** — `abi doctor` (health check), `abi clean` (artifact removal), `abi env` (environment variables), `abi init` (project scaffolding) (2026-02-25)
- [x] **CLI output consistency** — Converted `std.debug.print` to `utils.output` across 22 commands for NO_COLOR compliance and structured formatting (2026-02-25)
- [x] **AGENTS.md** — Added workflow orchestration rules, task management guidelines, and core principles (2026-02-25)

---

## Blocked

Waiting on external dependencies:

| Item | Blocker | Workaround |
|------|---------|------------|
| Native HTTP downloads | Zig 0.16 `std.Io.File.Writer` API unstable | Falls back to curl/wget instructions |
| Toolchain CLI | Zig 0.16 API incompatibilities | Command disabled; manual zig installation |
| macOS 26 Xcode-beta | `arm64e`-only TBD stubs break Zig linker | `DEVELOPER_DIR=/Library/Developer/CommandLineTools` in `~/.zshenv` |

**Note:** macOS 26 fix is a workaround. Permanent fix requires either `sudo xcode-select -s /Applications/Xcode.app` or Apple restoring `arm64-macos` in Xcode-beta SDK.

---

## Next Sprint Preview

Potential focus areas for upcoming work:

- [ ] TUI dirty-rect rendering (eliminate full screen clear every frame)
- [ ] TUI layout system improvements (nesting, flex-box)
- [ ] Language bindings reimplementation (Python, Rust, Go, JS/WASM, C headers)
- [ ] Additional competitive benchmarks
- [ ] Community contribution tooling

---

## Recently Completed

- **CLI output consistency** — Converted all 22 CLI command files from `std.debug.print` to `utils.output.*` for colored, structured, NO_COLOR-compliant output; Phase 3 of CLI modernization (2026-02-25)
- **New developer commands** — `abi doctor` (environment health check: Zig version, framework init, GPU detection, API keys, feature modules), `abi clean` (build cache / state / model cleanup with safety guards), `abi env` (list/validate/export ABI environment variables), `abi init` (project scaffolding with 4 templates: default, llm-app, agent, training) (2026-02-25)
- **Stream error recovery** - Per-backend circuit breakers, exponential backoff retry, session caching, recovery events
- **Streaming integration tests** - E2E tests with fault injection for circuit breaker, session cache, metrics
- **Security hardening** - JWT none algorithm warning, master key requirement option, secure API key wiping
- **Streaming documentation** - SSE/WebSocket streaming guide (docs site: `docs/content/api.html`)
- **Model management guide** - Downloading, caching, hot-reload (docs site: `docs/content/ai.html`)
- **Metal backend enhancements** - Accelerate framework (vBLAS/vDSP/vForce), unified memory manager, zero-copy tensors
- **GPU backend test coverage complete** - Added inline tests to ALL GPU backends: WebGPU, OpenGL, OpenGL ES, Vulkan (17 error cases), Metal (10 error cases), WebGL2, stdgpu (2026-01-31)
- **Documentation cleanup** - Removed 23 redundant files; Added standardized error module; Added inline tests to config/loader.zig and platform/detection.zig (2026-01-31)
- **Zig 0.16 pattern modernization** - Replaced @tagName() with {t} format specifier, converted std.ArrayList to ArrayListUnmanaged, updated std.json.stringify (2026-01-31)
- **Configuration loader with env vars** - New ConfigLoader for runtime configuration via environment variables (2026-01-31)
- **Build system improvements** - Fixed pathExists() for Zig 0.16, synced package version to 0.4.0, cross-platform cli-tests (2026-01-31)
- **AI stub parity complete** - Full stub/real API parity for `-Denable-ai=false` builds; all feature flag combinations compile (2026-01-31)
- **Multi-Model Training Infrastructure** - Complete forward/backward training loops for LLM, Vision (ViT), and Multimodal (CLIP) models (2026-01-25)
- **Lock-free concurrency primitives** - Chase-Lev deque, epoch reclamation, MPMC queue, NUMA-aware work stealing (2026-01-25)

---

## Quick Links

- [ROADMAP.md](ROADMAP.md) - Full project roadmap
- [CLAUDE.md](CLAUDE.md) - Development guidelines
- [AGENTS.md](AGENTS.md) - Workflow orchestration rules
