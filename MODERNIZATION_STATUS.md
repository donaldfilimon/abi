# Modernization Status Dashboard

## Overview
The ABI modernization initiative targets full compatibility with Zig 0.16-dev, production-ready tooling, and reproducible deployments. This dashboard consolidates doc-side TODO tracking so contributors can see, at a glance, which phases are complete and which still need attention.

## Phase Summary

| Phase | Scope Highlights | Status | Notes |
|-------|------------------|--------|-------|
| Build System | `build.zig`, modular artifacts, deterministic outputs | 🟡 In Progress | Core build pipeline compiles successfully, but artifact normalization and cache policy reviews are still pending a final audit. |
| I/O Boundaries | Writer injection, structured logging, recovery paths | 🟡 In Progress | CLI layers adopt injected writers, yet deeper subsystems (GPU services, HTTP adapters) retain direct `stdout` calls earmarked for refactor. |
| CLI Experience | Modern CLI + comprehensive command surface | ✅ Complete | `modern_cli.zig`, `comprehensive_modern_cli.zig`, and router integration are TODO-free and validated via smoke tests. |
| Parser & Diagnostics | Slice-based parsing, diagnostics streams | 🔴 Pending | Parser refactor backlog not yet started; existing modules still rely on pointer arithmetic and lack diagnostics plumbing. |
| CI/CD Platform | Multi-OS builds, docs publishing, analyzers | 🔶 Planned | Workflow definitions exist in draft form, but cross-platform runners and doc artifacts need implementation.

_Status icons:_ ✅ Complete · 🟡 In Progress · 🔴 Pending · 🔶 Planned

## Outstanding TODO Hotspots

The following table lists the top sources of remaining TODO comments gathered on 2024-09-30 using PowerShell (`Get-ChildItem -Path src -Filter *.zig -Recurse | Select-String -Pattern 'TODO' | Group-Object Path | Sort-Object Count -Descending`).

| File | TODO Count | Subsystem | Follow-up |
|------|------------|-----------|-----------|
| `src/features/gpu/libraries/vulkan_bindings.zig` | 21 | GPU backends | Implement Vulkan initialization, resource management, and advanced pipeline wiring. |
| `src/features/gpu/libraries/cuda_integration.zig` | 13 | GPU backends | Replace CUDA stubs with cudaz-backed detection, memory, and kernel launch APIs. |
| `src/features/gpu/libraries/mach_gpu_integration.zig` | 12 | GPU backends | Flesh out Mach GPU lifecycle (init, cleanup, pipelines, shader handling). |
| `src/features/gpu/testing/cross_platform_tests.zig` | 17 | GPU validation | Supply concrete GPU smoke/benchmark tests across APIs and stress scenarios. |
| `src/features/gpu/optimizations/backend_detection.zig` | 8 | GPU orchestration | Swap placeholder capability detection with platform-specific probes. |
| `src/features/gpu/wasm_support.zig` | 6 | WASM/GPU tooling | Add real WASM module execution and WebGPU/WebGL initialization routines. |
| `src/tools/interactive_cli.zig` | 6 | Tooling UX | Remaining occurrences reference historical TODO messaging; convert messaging once GPU todo backlog is cleared. |
| `src/tools/advanced_code_analyzer.zig` | 5 | Tooling | Connect analyzer TODO detection with issue creation and suppression controls. |
| `src/features/gpu/compute/kernels.zig` | 4 | GPU compute | Implement GPU-accelerated kernels for matrix math, softmax, and normalization. |
| `src/features/gpu/mobile/mobile_platform_support.zig` | 4 | Mobile GPU | Fill in Metal, Vulkan, WebGPU mobile initialisation paths. |

> **Tip:** Re-run the command above after each refactor to keep this dashboard accurate. If the command reports new files, append them to the list with context so the entire team maintains visibility.

## Verification Commands

- **Recount TODO markers:**
  ```powershell
  Get-ChildItem -Path src -Filter *.zig -Recurse | Select-String -Pattern 'TODO' | Group-Object Path | Sort-Object Count -Descending | Select-Object -First 15 Name,Count
  ```
- **Check modernization-critical modules for lingering TODOs:**
  ```powershell
  Get-ChildItem -Path src/tools -Filter *.zig -Recurse | Select-String -Pattern 'TODO'
  ```
- **Check for reintroduced legacy CLI summary files:**
  ```powershell
  Get-ChildItem -Path . -Filter 'CLI_*SUMMARY.md' -Recurse
  ```
  > The historical CLI/GPU summary exports have been removed from the repository; this command should return no matches.

## Next Actions

1. **GPU backend implementations:** Prioritise completing Vulkan, CUDA, and Mach code paths so benchmark claims align with runtime capabilities.
2. **Testing coverage:** Replace placeholder GPU cross-platform tests with failing tests that describe the desired behaviour, then implement the supporting code.
3. **Parser phase kick-off:** Draft a migration plan for the parser subsystem, covering diagnostics, slice-first APIs, and golden tests.
4. **CI/CD uplift:** Wire reproducible builds and documentation publishing into CI so modernization progress is automatically validated.
5. **Documentation hygiene:** Keep this dashboard and the canonical docs up to date, and ensure obsolete one-off summary exports stay out of the tree so contributors always land in the authoritative references.

Maintaining this dashboard ensures the modernization effort stays evidence-based and prevents TODO drift as the codebase evolves.