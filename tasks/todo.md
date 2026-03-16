# ABI Development Queue

Active task tracker. Use `git add -f tasks/todo.md` to stage.

## Completed — Structure Redesign Foundations

The following structural changes have landed on `main`:

- [x] Version pin bumped to `0.16.0-dev.2905+5d71e3051`
- [x] `foundation` namespace (`abi.foundation` via `src/services/shared/mod.zig`) replaces old `shared_services` — part of the single `abi` module, not a separate named module
- [x] `core` named module removed — files in `src/core/` are part of the `abi` module
- [x] `wireAbiImports` signature: `(module, build_opts)` — wires `build_options` import only
- [x] 3,294 imports updated with explicit `.zig` extensions (Zig 0.16 dev.2905+ requirement)
- [x] Test entrypoints rooted in `src/` for module root compliance (former `tests/zig/` shims removed)
- [x] Bindings relocated to top-level `bindings/` (c/, wasm/)
- [x] `build.zig.zon` paths updated to include `bindings`, `lang`, `tests`
- [x] Plugin migration from Codex to Claude Code completed (cel-language, abi-code-review skills added)

## Active — Master-Branch Structure Redesign v2

### Phase 1 — Logical Graph Normalization

- [x] Rewrite `src/root.zig` to expose the direct-domain surface (`abi.runtime`, `abi.database`, `abi.ai`, `abi.foundation`, etc.)
- [x] Add `build/module_catalog.zig` as the build/docs/test source of truth for public modules and feature-test entries
- [x] Replace tracked generated test roots with build outputs and stop writing to `src/generated_feature_tests.zig`
- [x] Make `tests/zig/` authoritative for aggregate test entrypoints (migrated to `src/` to fix module path constraints)
- [x] Fix current master-branch import failures:
  - [x] package-root import assumptions in the public surface
  - [x] pseudo-submodule imports in `src/core/database/*`
  - [x] ambient file imports in `src/services/tests/mod.zig`
- [x] Rewire `tools/gendocs` to discover modules from the new catalog and root surface
- [x] Keep the existing build command surface stable (`test`, `feature-tests`, `full-check`, `verify-all`, `check-docs`, `validate-flags`)

### Phase 2 — Physical Relayout

- [x] Establish `src/internal/` family wrappers for app, foundation, runtime, ai, data, network, platform, integrations, observe, and tooling
- [x] Move bindings from `src/bindings/` to top-level `bindings/` and update build/install paths
- [x] Reserve the `lang/cel/` lane and wire package metadata/build paths for future CEL relocation without changing stage0 behavior
- [x] Update docs/templates/CLI surfaces toward the direct-domain API
- [x] Delete obsolete tracked generated files and stale structure assumptions where Phase 1 now has authoritative replacements (`src/generated_feature_tests.zig`, old bindings paths)

### Phase 3 — Wave 5: Feature Modules Restructure

- [x] Wave 5A: Consolidate AI sub-contexts (Core, Inference, Training, Reasoning) into a single `ai_mod.Context` and remove obsolete AI facades
- [x] Wave 5B: Consolidate shared primitives (`Confidence`, `EmotionalState`, `InstanceId`, `SessionId`) into canonical shared/types module
- [x] Wave 5C: Update integration roots and tests to the final module topology (direct-domain API usage verified in tests and CLI)

### Phase 4 — Validation and Cleanup

- [x] Run full validation suite (`./tools/scripts/run_build.sh full-check --summary all`)
- [x] Address edge-case API regressions caused by stub mismatch (DeepResearcher, DynamicApiLearner, RuntimeBridge, OSControlManager)
- [x] Verify TUI and Launcher test targets pass successfully under the new strict module boundaries.

### Validation

- [x] `zig fmt --check build.zig build/ src/ tools/ tests/ bindings/ lang/`
- [x] `./tools/scripts/run_build.sh typecheck --summary all` (all errors cleared, including SPIR-V/HNSW/ArrayListUnmanaged)
- [x] `./tools/scripts/run_build.sh feature-tests --summary all` (resolved module ownership conflicts by pruning redundant catalog entries and normalizing cross-module imports)
- [x] `./tools/scripts/run_build.sh validate-flags`
- [x] `./tools/scripts/run_build.sh database-fast-tests` (all errors cleared)
- [x] `./tools/scripts/run_build.sh cli-tests`
- [x] `./tools/scripts/run_build.sh tui-tests`
- [x] `./tools/scripts/run_build.sh check-cli-registry`
- [x] `./tools/scripts/run_build.sh check-docs`
- [x] `./tools/scripts/run_build.sh full-check --summary all`

Validation evidence:
- `2026-03-16`: `zig fmt --check build.zig build/ src/ tools/ tests/ bindings/ lang/` passed.
- `2026-03-16`: `./tools/scripts/run_build.sh typecheck --summary all` passed for the entire package graph, including database/GPU roots (SPIR-V/HNSW/ArrayListUnmanaged errors fixed).
- `2026-03-16`: `./tools/scripts/run_build.sh database-fast-tests` passed after fixing Zig 0.16 compatibility.
- `2026-03-16`: Established `src/internal/` family wrappers for all core domains. Updated `src/root.zig` to use these wrappers.
- `2026-03-16`: Consolidated AI sub-contexts into a single `ai_mod.Context` and removed obsolete AI facades (Wave 5A).
- `2026-03-16`: Consolidated shared primitives (`Confidence`, `EmotionalState`, `InstanceId`, `SessionId`) into `src/services/shared/types.zig` and `src/features/ai/types.zig` (Wave 5B).
- `2026-03-16`: Migrated all CLI commands in `tools/cli/` to use the direct-domain API (`abi.<domain>`).
- `2026-03-16`: `./tools/scripts/run_build.sh full-check --summary all` passes completely (165/165 steps). The repository's entire target graph, including CLI tests, documentation generation tests, feature flags, and UI components are fundamentally stable under the new topology.

### Phase 4 — Post-Restructure Cleanup

- [x] Pruned stale AI facade references from `build/module_catalog.zig` (4 entries pointing at deleted `src/features/ai/facades/` files)
- [x] Fixed `@import("abi")` used from within the `abi` module in 6 files (`ai/core/config.zig`, `ai/core/mod.zig`, `ai/abbey/mod.zig`, `ai/database/wdbx.zig`, `ai/database/export.zig`, `ai/database/brain_export.zig`) — replaced with relative imports
- [x] Fixed cross-feature imports bypassing feature gates (`web/handlers/chat.zig` → AI, `ai/profiles/mod.zig` → observability, `ai/streaming/server.zig` → observability) — added `build_options` conditional imports
- [x] Updated CLI `build_options_stub.zig` with missing `feat_compute`, `feat_documents`, `feat_desktop` flags
- [x] Added AI subfeature validation combos to `build/flags.zig` (`llm-only`, `training-only`, `reasoning-only`, `no-llm`, `no-training`, `no-reasoning`)
- [x] Updated docs generation: removed Codex references, renamed `shared.md` → `foundation.md`, updated index.md
- [x] Clarified `src/abi.zig` as legacy (dead) file in README.md
- [x] Updated mod/stub/types contract docs to clarify `types.zig` is required only when shared public types exist
- [x] Unified format-check surface across `AGENTS.md`, `CLAUDE.md`, and `tools/scripts/fmt_repo.sh` to include `tests/ bindings/ lang/`
- [x] Retired transitional `v3-*` build steps in `build.zig` → renamed to `static-lib` and `server`, removed redundant `v3-test`
- [x] Reconciled `foundation` module documentation in `tasks/lessons.md` — corrected to match reality (namespace within `abi` module, not a separate named module)
- [x] `zig fmt --check build.zig build/ src/ tools/ tests/ bindings/ lang/` passes

### Phase 5 — Post-Cleanup Follow-On

- [x] Purged stale Codex footers from 21 `docs/api/*.md` files (two footer variants)
- [x] Fixed `ai/metrics.zig` cross-feature gate bypass — observability import now gated by `build_options.feat_profiling`
- [x] Fixed broken `shared.md` links in `docs/api/index.md` and `docs/api/coverage.md` — updated to `foundation.md`
- [x] Cleaned remaining `$zig-master` Codex references in `tools/gendocs/render_guides_md.zig` and `tools/gendocs/templates/docs/contributing.md.tpl`
- [x] Expanded compute/documents/desktop to standard module contract (`isEnabled`, `isInitialized`, `Context`, `Error`) in both mod and stub
- [x] Updated `build/validate/stub_surface_check.zig` to exercise new compute/documents/desktop contract symbols
- [x] Documented `check_import_rules.zig` scope (features only; services/tests excluded because they're wired as separate test modules)
- [x] Removed dead `tests/zig/` shims (`mod.zig`, `database_fast_tests_root.zig`, `database_wdbx_tests_root.zig`) — build.zig uses `src/`-rooted entries
- [x] Added `feat_explore` and `feat_vision` to `FlagCombo` in `build/flags.zig` with proper inheritance logic
- [x] `zig fmt --check build.zig build/ src/ tools/ tests/ bindings/ lang/` passes

### Phase 6 — Governance Drift Sweep & Validation Hardening

- [x] Fixed 5 remaining cross-feature gate bypasses: `ai/training/mod.zig` (database), `ai/memory/long_term.zig` (database), `ai/coordination/mod.zig` (database), `ai/context_engine/triad.zig` (database gate on wrong flag), `ai/streaming/metrics.zig` (observability)
- [x] Corrected stale foundation-module wiring descriptions in `tasks/todo.md` (was: named module with 5-arg wireAbiImports; now: namespace with 2-arg wireAbiImports)
- [x] Updated `docs/STRUCTURE.md` — removed references to deleted `tests/zig/` wrappers
- [x] Fixed `.claude/agents/darwin-build-doctor.md` — updated deleted test root path to `src/services/tests/mod.zig`
- [x] Fixed `docs/PATTERNS.md` — corrected types.zig contract to match AGENTS.md (required only when shared types exist)
- [x] Updated combo count: CLAUDE.md, zig-abi-plugin/abi-architecture (42 → 54)
- [x] Added explore/vision solo+no-X validation combos to `build/flags.zig` (explore-only, vision-only, no-explore, no-vision) and ensured all no-X combos include explore+vision
- [x] Added canonical top-level `abi.<domain>` API checks to `build/validate/stub_surface_check.zig` (was: only compat bridges `abi.features.*`/`abi.services.*`)
- [x] Synced CI format coverage in `.github/workflows/ci.yml` to match repo contract (`examples/ tests/ bindings/ lang/` added)
- [x] Unified fmt surface with `examples/` across `AGENTS.md`, `CLAUDE.md`, `tools/scripts/fmt_repo.sh` to match `build.zig` lint step
- [x] Fixed zig-abi-plugin docs: removed stale foundation named-module claims, updated `src/abi.zig` → `src/root.zig` in new-feature command and code-review identity check, corrected zig-016-patterns named-module guidance
- [x] `zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/` passes

### Phase 7 — Final Sweep

- [x] Fixed last unguarded cross-feature import: `ai/streaming/backends/local.zig` — LLM import now gated by `build_options.feat_llm`
- [x] Fixed `GEMINI.md` stale `$zig-master` Codex skill references → CLAUDE.md + `zig build full-check`
- [x] Fixed `abi-architecture/SKILL.md` — removed `foundation` from named modules table (3 modules, not 4), removed `createFoundationModule()`, fixed combo count 48→54
- [x] Fixed `docs/STRUCTURE.md` combo count 42→54
- [x] Fixed `review_prep.py` ABI marker `src/abi.zig` → `src/root.zig`
- [x] Fixed `review_prep.py` + `abi-code-review/SKILL.md` stale `src/wdbx/` paths → `src/core/database/`
- [x] Fixed stale `@import("shared")` in `docs/api/foundation.md`, `docs/api/index.md`, and `src/services/shared/mod.zig` doc comments
- [x] Made `all-enabled` combo explicitly set all AI subfeature flags (explore, llm, vision, training, reasoning)
- [x] Final verification sweep: zero stale refs for Codex, $zig-master, @import("foundation"), createFoundationModule, v3-*, tests/zig/, 42 combos
- [x] `zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/` passes

### Phase 8 — Deep Contract Hardening

- [x] **CRITICAL**: Fixed `cloud/mod.zig:261` `isEnabled()` — was checking `feat_web` instead of `feat_cloud`
- [x] Fixed stale `shared.` variable references in `docs/api/{foundation,index}.md` code examples — now uses `foundation.` consistently
- [x] Fixed broken 4-level import path in `ai/embeddings_logic/persona_index.zig:19` — was `../../../../` (resolves to repo root), now `../../../`
- [x] Fixed `observability/otel.zig:5` confusing self-relative import `../observability/mod.zig` → `mod.zig`
- [x] Added `pub const Error` alias to 9 feature modules (auth, analytics, search, storage, messaging, cache, gateway, mobile, benchmarks) — both mod and stub, for consistent `feature.Error` API
- [x] Added `isInitialized()` to `benchmarks/{mod,stub}.zig` for lifecycle contract parity with all other features
- [x] Fixed `gateway/stub.zig` type source mismatch — config types now imported from shared `types.zig` (same source as mod.zig) instead of `core/config/gateway.zig`
- [x] Removed dead `addValidationScriptStep` function from `build.zig:823-831`
- [x] Fixed `.claude/settings.json` hook — changed `src/abi.zig` trigger to `src/root.zig`
- [x] Updated `stub_surface_check.zig` with all new `Error` aliases and `benchmarks.isInitialized`
- [x] `zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/` passes

### Phase 9 — Master-Branch Organization Cleanup

- [x] Fixed `docs/PATTERNS.md` — updated combo count 42→54, fmt scope includes all source dirs
- [x] Fixed `check_test_baseline_consistency.zig` — corrected WDBX fast test root path
- [x] Replaced `src/abi.zig` with minimal tombstone (was full shadow root duplicating `root.zig`)
- [x] Fixed `zig-abi-plugin/commands/check.md` — updated fmt scope, removed stale `foundation` named module claim
- [x] Fixed `zig-abi-plugin/agents/stub-sync-validator.md` — removed stale `wdbx` named module refs, updated import-check guidance
- [x] Fixed legacy `tests/` standalone files — added explicit `.zig` extensions to imports (`integration_test.zig`, `simd_test.zig`, `hnsw_test.zig`)
- [x] Updated `docs/STRUCTURE.md` — corrected `abi.zig` description to tombstone, `tests/` to legacy, removed stale counts
- [x] Migrated example `database.zig` from `abi.features.database` to `abi.database` canonical API
- [x] Migrated example `concurrent_pipeline.zig` from `abi.services.runtime`/`abi.services.shared` to `abi.runtime`/`abi.foundation`
- [x] Updated `README.md` — canonical `abi.<domain>` API in feature list (was compat bridges)
- [x] Verified Darwin/macOS linking: ZVM=0.14.0, Zig=0.16.0-dev.2905+5d71e3051, macOS=26.4, `run_build.sh` workaround documented
- [x] `zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/` passes

### Phase 10 — ABI Cleanup Wave

- [x] Verified remaining `examples/` call sites already use canonical `abi.<domain>` exports; no further compat-bridge edits needed in this wave
- [x] Migrated the remaining service tests (`web`, `network`, `cloud`, `analytics`) from compat bridges to canonical top-level exports
- [x] Fixed the stale shim-wrapper wording in `docs/STRUCTURE.md`
- [x] Fixed `ai/agents/gpu_agent.zig` to gate GPU imports through `build_options` with the standard `mod.zig`/`stub.zig` pattern
- [x] Fixed `build/flags.zig` AI subfeature inheritance so `ai-only` still enables all AI subfeatures, while explicit `*-only` and `no-*` combos remain effective
- [x] Removed 6 clean registered automation worktrees under `.claude/worktrees/` via `git worktree remove`; kept 2 docs branches because they were not clearly disposable snapshots

### Phase 11 — AI Import Hygiene

- [x] Replaced 46 bare AI imports (`types`, `agents`, `prompts`, `training`, `self_learning`) with explicit relative `.zig` paths under `src/features/ai/`
- [x] Clarified docs generation guidance for Darwin 25+ / 26+ so `gendocs` compile-check behavior is documented without implying local regeneration

Validation evidence:
- `2026-03-16`: `zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/` passed after the AI import cleanup.
- `2026-03-16`: `./tools/scripts/run_build.sh typecheck --summary all` passed after replacing the bare AI imports with explicit relative paths.

Validation evidence:
- `2026-03-16`: Confirmed `examples/` no longer contains `abi.features.*` or `abi.services.*` call sites.
- `2026-03-16`: `zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/` passed after the cleanup wave.
- `2026-03-16`: `./tools/scripts/run_build.sh typecheck --summary all` passed after the build-flag and GPU import fixes.

### Notes

- [x] Tri-CLI consensus helper unavailable locally: `/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` not present in this environment.

### Phase 12 — Gemini Wave: Docs, Linker, Logic Consolidation, and Compat Planning

- [x] Deleted old `docs/api` and `docs/data` and regenerated them dynamically with `docgen`, ensuring 95 artifacts populate successfully.
- [x] Fixed `build.zig` Darwin linking natively for `gendocs` via dynamic `libcompiler_rt.a` cache discovery and `std.Io.Dir` operations.
- [x] Documented that `run_build.sh` remains necessary on Darwin 25/26 for Zig's own `build_runner` bootstrap phase, as that occurs before `build.zig` execution.
- [x] Validated that all 22 `examples/` already use canonical `abi.<domain>` APIs and require no further changes.
- [x] Resolved `src/inference/` orphan by safely folding it into `src/features/ai/llm/inference/`.
- [x] Cleaned `.claude/worktrees/` by removing obsolete `agent-a01c0e27` and `agent-a25debec` snapshots.
- [x] Consolidated AI subsystem `*_logic` folders (`abi_logic`, `aviva_logic`, `routing_logic`, `abbey_logic`, `embeddings_logic`, `templates_logic`) by renaming overlapping `mod.zig` files (`persona.zig`, `orchestrator.zig`, `industry.zig`) and merging directories.
- [x] Updated internal imports to point to the consolidated AI paths.
- [x] `zig fmt --check build.zig build/ src/ tools/` passes after all structural modifications.

#### Compat Bridge Retirement Plan (abi.features.*, abi.services.*, personas)

1. **Audit & Enforce**: Ensure `build/validate/stub_surface_check.zig` enforces the canonical `abi.<domain>` architecture (completed). Add a compiler warning or deprecation notice inside the old `abi.features` and `abi.services` namespace declarations in `src/root.zig`.
2. **Internal Sweep**: Grep the `src/` tree for any lingering internal usages of `abi.features.*` and `abi.services.*` and migrate them to relative imports or canonical `abi.<domain>` aliases.
3. **Tests Migration**: Migrate all integration tests in `src/services/tests/` (e.g., `simd_validation_test.zig`, `test_matrix.zig`, `ha_test.zig`, `stress/*`) that still use `abi.services.*` and `abi.features.*` to the canonical top-level API.
4. **Deprecation Phase**: Once all internal usages and tests are migrated, change the namespaces in `src/root.zig` to use `@compileError("Deprecated")` to catch any external plugins or un-migrated scripts.
5. **Final Removal**: After a grace period, completely remove `pub const features` and `pub const services` from `src/root.zig`.

### Phase 13 — ABI Refinement and Darwin Stability

- [x] Fixed `src/features/web/handlers/chat.zig` regression where `f32` was accessed as struct (post-Confidence consolidation).
- [x] Verified `test` target passes on Darwin 26.4 via `run_build.sh`.
- [x] Improved `build.zig` to use native macOS 15.0 clamping for all targets on Darwin 26+.

### Phase 14 — Final Cleanup Wave

- [ ] Promote inference to canonical top-level `abi.inference` and remove the temporary `src/features/ai/llm/inference/` relocation.
- [ ] Remove `abi.features.*` / `abi.services.*` compat bridges from `src/root.zig` and migrate remaining in-tree callers/tests/docs to canonical exports.
- [ ] Rename public `personas` compatibility surface to `profiles` across config, feature catalog, docs data, examples, and AI module exports.
- [ ] Remove duplicate AI nested mirror directories (`abi/abi_logic`, `aviva/aviva_logic`, `routing/routing_logic`) and keep the flat canonical files.
- [ ] Reset gendocs inputs/outputs for canonical API + `abi.inference`, then refresh generated docs.
- [ ] Reframe Darwin 26.4 guidance around host-built / known-good Zig as the supported full-validation path; keep wrappers as fallback only.
- [ ] Run verification/search gates and record evidence for the final cleanup wave.

### Phase 14 — Universal Platform Support Plan (Desktop/Mobile/Embedded)

To ensure flawless support across the newly expanded `cross-check` matrix, the following steps must be executed:

1. **CI Integration**:
   - Update `.github/workflows/ci.yml` to execute `zig build cross-check`. This will enforce compilation across all 20+ targets (Linux, Windows, macOS, BSDs, iOS, Android, WASM, RISC-V, Thumb).
2. **Freestanding / Bare Metal Audit**:
   - Audit the codebase for `std.fs`, `std.Thread`, `std.process`, and networking imports.
   - Ensure these are strictly gated behind `builtin.os.tag != .freestanding` or specific feature flags (`feat_network`, `feat_storage`). Embedded platforms (RISC-V 32, Thumb) cannot use OS-level APIs.
3. **Mobile (iOS/Android) Refinement**:
   - **Android**: Verify JNI bridging and ensure the `applyAndroidLinks` logic correctly integrates with the Android NDK for `.so` generation.
   - **iOS**: Adjust `applyFrameworkLinks` to differentiate between macOS-specific frameworks (`AppKit`, `Cocoa`) and iOS equivalents (`UIKit`), ensuring Metal/CoreML link correctly on iOS.
4. **WASM Validation**:
   - Ensure `wasm32-freestanding` and `wasm32-wasi` targets bypass all GPU/Metal/Accelerate linking.
   - Verify that SIMD math falls back to scalar implementations or `wasm32` specific intrinsics when `stdgpu` is unavailable.
5. **Runtime Capability Probing**:
   - Implement dynamic capability checking for things like AVX512/NEON/SVE so the same binary can degrade gracefully on older desktop/mobile hardware.

### Phase 15 — Docs Gate Recovery

- [x] Fixed `build.zig` CLI smoke-test wiring on Darwin by unwrapping the selected compile artifact before passing it to `build/cli_tests.zig`
- [x] Fixed `tools/gendocs/check.zig` so maintained docs and `.DS_Store` are not reported as drift extras
- [x] Tightened `--untracked-md` handling so missing generated markdown is allowed, while changed or stale generated outputs still fail `check-docs`
- [x] Updated canonical API wording at the source-comment / generator layer (`abi.ai`, `abi.auth.*`, `abi.cloud`, `abi.gpu`) and aligned Darwin docs-generation guidance with the live `run_build.sh gendocs` behavior
- [x] Regenerated the docs pipeline outputs after the checker and source-comment fixes

Validation evidence:
- `2026-03-16`: `./tools/scripts/run_build.sh check-docs --summary all` passed after the docs drift checker fix and docs regeneration.
- `2026-03-16`: `./tools/scripts/run_build.sh gendocs` completed successfully on Darwin via the relinked `gendocs` path (`OK: generated 95 docs artifacts`).
- `2026-03-16`: `./tools/scripts/run_build.sh typecheck --summary all` passed after the `build.zig` CLI smoke-test fix.
- `2026-03-16`: `zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/` passed after the docs-gate recovery edits.
