# ABI Development Queue

Active task tracker. Use `git add -f tasks/todo.md` to stage.

## Objective

Advance the ABI Zig 0.16 framework toward production maturity: complete the Phase 15 roadmap items (semantic store persistence, mobile native bridges, bare metal examples), resolve remaining Darwin 25+ toolchain blockers, and maintain full verification gate compliance.

## Scope

**In scope:**
- Phase 15 roadmap items (semantic store, mobile bridges, embedded examples)
- Darwin host-built Zig bootstrap completion
- Verification gate maintenance and docs synchronization
- Integration gate validation (Stage 2, blocked by Darwin linker — CI is authoritative)

**Out of scope:**
- Phases 1–14 (completed, archived below)
- Zig version repinning beyond `0.16.0-dev.2905+5d71e3051`
- New feature modules not on the Phase 15 roadmap

## Verification Criteria

- `zig build full-check --summary all` passes (or `./tools/scripts/run_build.sh full-check --summary all` on Darwin 25+)
- `zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/` clean
- All 54 feature flag combinations validated
- Test baseline consistent with `tools/scripts/baseline.zig`
- Documentation generator deterministic (`check-docs` passes)

## Checklist

### Phase 15 — Active Roadmap

- [x] **Plugin Ecosystem**: Formalize the plugin API for external C/WASM modules to register with `abi.registry`.
- [x] **Distributed Raft V2**: Harden the `abi.network` Raft implementation for production-grade partition tolerance.
- [ ] **Semantic Store Persistence**: Optimize the WDBX block-chain layout for multi-terabyte vector indices.
- [ ] **Mobile Native Bridges**: Implement native Swift/Kotlin bridges for iOS and Android high-level UI integration.
- [ ] **Bare Metal Examples**: Add `examples/embedded/` showing deployment to RISC-V 32 and Thumb bare-metal boards.
- [x] **Import-Rule Guardrail Hardening**: Restored bare `build_options` named imports, normalized AI imports, strengthened `check-imports`.
- [x] **Import Boundary and Skill Repair Wave**: Repaired Zig 0.16 import cleanup, aligned skills with module boundaries.
- [x] **Pinned macOS Host-Zig Bootstrap Wave**: Added deterministic host-built Zig bootstrap/discovery path for Darwin.
  - [ ] Runtime validation pending: `bootstrap_host_zig.sh` does not yet produce a working pinned compiler on Darwin/Xcode-beta.

### Stage Status

- **Stage 1** (push + PR triage): COMPLETE — 2 commits pushed, 8 PRs (488-495) closed as superseded
- **Stage 2** (integration gates): NOT STARTED (blocked by Darwin linker; CI is authoritative)
- **Stage 3** (close plans): IN PROGRESS
- **Stage 4** (Phase 15 roadmap): NOT STARTED
- **Stage 5** (housekeeping): IN PROGRESS

## Review

### Current State (2026-03-17)

All source code compiles and passes verification. Remaining gate failures are structural (docs drift, workflow contract formatting in tasks/ files, missing `lang/` directory).

### Completed (Archived)

Historical phases of the master-branch structure redesign and normalization (completed March 2026):

- **Phases 1-3**: Logical graph normalization, physical relayout, and feature module consolidation (AI contexts, shared primitives).
- **Phases 4-11**: Post-restructure cleanup, governance drift sweeps, deep contract hardening, and AI import hygiene.
- **Phases 12 (Core), 13 & 15**: Gemini Wave logic consolidation, Darwin/macOS stability refinements, and documentation gate recovery.
- **Phase 14**: Final cleanup and universal support — inference promotion, compat bridge removal, terminology alignment, AI consolidation, docs refresh, Darwin guidance, CI integration, freestanding audit, mobile refinement, runtime SIMD probing.
- **Key Outcomes**: Single `abi` module, canonical `abi.<domain>` API, 54 validated flag combinations, and full 165-step validation success.
