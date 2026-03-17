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

47. [x] **Import Boundary and Skill Repair Wave**: Repaired the broken Zig 0.16 import cleanup, aligned skill guidance with ABI's real module boundaries, and deferred baseline sync until executable suite counts are available.
    - [x] Restored the touched `src/services/tests/` and property tests to the separate-root `@import("abi")` pattern.
    - [x] Replaced the distributed database's nonexistent `network` file imports with `build_options`-gated `features/network` imports.
    - [x] Replaced the touched GPU test placeholder imports with real GPU-relative module paths without broadening test discovery.
    - [x] Updated repo/home ABI skills, baseline-sync, and the ABI code-review helper to document separate-root callers, missing consensus-wrapper fallback, and dirty-worktree review.
    - [x] Validation evidence: `zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/`; `./tools/scripts/run_build.sh typecheck --summary all`; `./tools/scripts/run_build.sh test --summary all`; `./tools/scripts/run_build.sh feature-tests --summary all`; `python3 /Users/donaldfilimon/.codex/skills/abi-code-review/scripts/review_prep.py --repo /Users/donaldfilimon/abi --base HEAD`.
    - Residual: `test` and `feature-tests` were compile-only cached steps on the blocked Darwin host, so they did not emit runnable suite counts; `tools/scripts/baseline.zig` was intentionally left unchanged this wave.

48. [ ] **Pinned macOS Host-Zig Bootstrap Wave**: Add a deterministic host-built Zig bootstrap/discovery path for Darwin without repinning ABI away from `0.16.0-dev.2905+5d71e3051`.
    - [x] Kept the existing dirty worktree intact and left unrelated source-fix edits untouched while landing the toolchain changes.
    - [x] Added `tools/scripts/bootstrap_host_zig.sh` plus shared shell helpers for pinned-version discovery, canonical cache paths, and consistent Zig resolution order (`ABI_HOST_ZIG`, `ZIG_REAL`, `ZIG`, canonical cache, PATH).
    - [x] Taught `build.zig` and the Darwin-specific build surfaces to leave degraded compile-only mode when the active Zig is the canonical cached host-built compiler.
    - [x] Updated `toolchain_doctor`, `check_zig_version_consistency`, and `abi doctor` to report the active Zig path/version, canonical cache state, and the next bootstrap/path-export command.
    - [x] Updated contributor docs and generated-guide sources with the bootstrap/path-export flow and the Darwin fallback boundary.
    - [ ] Validate the helper and the pinned host-built path with `zig fmt --check ...`, `zig build toolchain-doctor`, `zig build check-zig-version`, `zig build full-check`, `zig build check-docs`, `zig build gendocs -- --check --no-wasm --untracked-md`, plus `./tools/scripts/run_build.sh typecheck --summary all` as fallback regression evidence.
    - Validation evidence so far:
      `zig fmt --check build.zig build/ tools/cli/commands/dev/doctor.zig tools/gendocs/render_guides_md.zig tools/scripts/check_zig_version_consistency.zig tools/scripts/toolchain_doctor.zig tools/scripts/toolchain_support.zig`
      `bash -n tools/scripts/bootstrap_host_zig.sh tools/scripts/inspect_toolchain.sh tools/scripts/zig_toolchain.sh tools/scripts/run_build.sh tools/scripts/fmt_repo.sh tools/scripts/zig_darwin26_wrapper.sh`
      `./tools/scripts/inspect_toolchain.sh`
      `./tools/scripts/run_build.sh typecheck --summary all`
      `git diff --check`
    - Residual blocker:
      `./tools/scripts/bootstrap_host_zig.sh` still does not produce `/Users/donaldfilimon/.cache/abi-host-zig/0.16.0-dev.2905+5d71e3051/bin/zig` on this Darwin/Xcode-beta host. The helper now gets past the build-runner link wall, but the final `compile exe zig` self-link remains unresolved. Additional detached-worktree experiments with `/Users/donaldfilimon/zig/build-release/stage4-release/bin/zig` (`0.16.0-dev.2922+abd099e97`), explicit SDK/Homebrew library paths, and bootstrap-only source compatibility edits advanced the failure but did not produce a pinned compiler.
    - Note: multi-CLI consensus helper unavailable in this environment (`/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` missing); proceeding best-effort.
