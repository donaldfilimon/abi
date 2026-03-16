# ABI Development Queue

Active task tracker. Use `git add -f tasks/todo.md` to stage.

## Completed (Archived)

Historical phases of the master-branch structure redesign and normalization (completed March 2026):

- **Phases 1–3**: Logical graph normalization, physical relayout, and feature module consolidation (AI contexts, shared primitives).
- **Phases 4–11**: Post-restructure cleanup, governance drift sweeps, deep contract hardening, and AI import hygiene.
- **Phases 12 (Core), 13 & 15**: Gemini Wave logic consolidation, Darwin/macOS stability refinements, and documentation gate recovery.
- **Key Outcomes**: Single `abi` module, canonical `abi.<domain>` API, 54 validated flag combinations, and full 165-step validation success.

## Phase 14 — Final Cleanup and Universal Support (Completed)

Successfully achieved the repository's final architectural vision and expanded platform support.

- [x] **Inference Promotion**: Promoted `inference` to canonical top-level `abi.inference` and reverted temporary relocation.
- [x] **Compat Bridge Removal**: Removed legacy `abi.features.*` and `abi.services.*` bridges from `src/root.zig`.
- [x] **Terminology Alignment**: Executed repository-wide rename of "persona" to "profile" across 150+ files (config, catalog, CLI, examples).
- [x] **AI Consolidation**: Consolidated AI nested mirror directories (`abi`, `aviva`, `routing`) and standardized on flat canonical files.
- [x] **Docs Refresh**: Refreshed `gendocs` inputs/outputs for the canonical API and `abi.inference`, achieving 100% coverage.
- [x] **Darwin Guidance**: Updated `README.md` with refined Darwin 26.4 guidance favoring host-built Zig validation.
- [x] **CI Integration**: Integrated `zig build cross-check` into `.github/workflows/ci.yml`.
- [x] **Freestanding Audit**: Completed freestanding/bare-metal audit and gated `std.fs`, `std.Thread`, `std.process`, and `std.net` usage.
- [x] **Mobile Refinement**: Refined iOS/Android framework linking in `build/link.zig` (UIKit vs AppKit/Cocoa).
- [x] **Runtime Probing**: Implemented runtime SIMD capability detection in `src/core/database/simd.zig` for AVX-512/NEON graceful degradation.

## Phase 15 — Future Roadmap

Upcoming goals for framework maturity and ecosystem expansion.

### Stage Status
- **Stage 1** (push + PR triage): COMPLETE — 2 commits pushed, 8 PRs (488-495) closed as superseded
- **Stage 2** (integration gates): NOT STARTED (blocked by Darwin linker; CI is authoritative)
- **Stage 3** (close plans): IN PROGRESS
- **Stage 4** (Phase 15 roadmap): NOT STARTED
- **Stage 5** (housekeeping): IN PROGRESS

### Roadmap Items
1. [x] **Plugin Ecosystem**: Formalize the plugin API for external C/WASM modules to register with `abi.registry`.
2. [x] **Distributed Raft V2**: Harden the `abi.network` Raft implementation for production-grade partition tolerance.
3. [ ] **Semantic Store Persistence**: Optimize the WDBX block-chain layout for multi-terabyte vector indices.
4. [ ] **Mobile Native Bridges**: Implement native Swift/Kotlin bridges for iOS and Android high-level UI integration.
5. [ ] **Bare Metal Examples**: Add `examples/embedded/` showing deployment to RISC-V 32 and Thumb bare-metal boards.
6. [x] **Import-Rule Guardrail Hardening**: Restored bare `build_options` named imports, normalized targeted AI shorthand imports to explicit relative `.zig` paths, and strengthened `check-imports` to distinguish named-module imports from file imports. Evidence: `zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/`, `./tools/scripts/run_build.sh check-imports --summary all`, `./tools/scripts/run_build.sh typecheck --summary all`.

47. [ ] **Zig 0.16 Syntax Perfection**: Refactor codebase to achieve syntax perfection per Zig 0.16-dev requirements, including .zig extensions in imports, correct error handling, API updates, and feature module contract completion.
    - Completed: Added .zig extensions to imports in:
        src/core/database/distributed/cluster.zig
        src/core/database/distributed/mod.zig
        src/features/gpu/backends/tests/performance_refactor_test.zig
        src/features/gpu/tests/performance_benchmark_test.zig
    - Pending: Fix src/services/tests/ai_quantization_test.zig module path issues
