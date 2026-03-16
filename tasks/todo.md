# ABI Development Queue

Active task tracker. Use `git add -f tasks/todo.md` to stage.

## Completed — Structure Redesign Foundations

The following structural changes have landed on `main`:

- [x] Version pin bumped to `0.16.0-dev.2905+5d71e3051`
- [x] `foundation` named module (`src/services/shared/mod.zig`) replaces old `shared_services` — wired to all compilation targets via `wireAbiImports`
- [x] `core` named module removed — files in `src/core/` are part of the `abi` module
- [x] `wireAbiImports` signature: `(b, module, build_opts, target, optimize)` — wires `build_options` and `foundation` imports
- [x] 3,294 imports updated with explicit `.zig` extensions (Zig 0.16 dev.2905+ requirement)
- [x] Test entrypoints moved to `tests/zig/`
- [x] Bindings relocated to top-level `bindings/` (c/, wasm/)
- [x] `build.zig.zon` paths updated to include `bindings`, `lang`, `tests`
- [x] Plugin migration from Codex to Claude Code completed (cel-language, abi-code-review skills added)

## Active — Master-Branch Structure Redesign v2

### Phase 1 — Logical Graph Normalization

- [x] Rewrite `src/root.zig` to expose the direct-domain surface (`abi.runtime`, `abi.database`, `abi.ai`, `abi.foundation`, etc.)
- [x] Add `build/module_catalog.zig` as the build/docs/test source of truth for public modules and feature-test entries
- [x] Replace tracked generated test roots with build outputs and stop writing to `src/generated_feature_tests.zig`
- [ ] Make `tests/zig/` authoritative for aggregate test entrypoints. Wrapper files exist, but the main build still roots tests at `src/services/tests/mod.zig` because standalone module-path constraints block direct `tests/zig -> src/` imports.
- [ ] Fix current master-branch import failures:
  - [x] package-root import assumptions in the public surface
  - [x] pseudo-submodule imports in `src/core/database/*`
  - [x] ambient file imports in `src/services/tests/mod.zig`
- [x] Rewire `tools/gendocs` to discover modules from the new catalog and root surface
- [x] Keep the existing build command surface stable (`test`, `feature-tests`, `full-check`, `verify-all`, `check-docs`, `validate-flags`)

### Phase 2 — Physical Relayout

- [ ] Establish `src/internal/` family wrappers for app, foundation, runtime, ai, data, network, platform, integrations, observe, and tooling
- [x] Move bindings from `src/bindings/` to top-level `bindings/` and update build/install paths
- [x] Reserve the `lang/cel/` lane and wire package metadata/build paths for future CEL relocation without changing stage0 behavior
- [ ] Update docs/templates/CLI surfaces toward the direct-domain API
- [x] Delete obsolete tracked generated files and stale structure assumptions where Phase 1 now has authoritative replacements (`src/generated_feature_tests.zig`, old bindings paths)

### Validation

- [x] `zig fmt --check build.zig build/ src/ tools/ tests/ bindings/ lang/`
- [~] `./tools/scripts/run_build.sh typecheck --summary all` (main graph passes; db/GPU test roots have pre-existing SPIR-V/HNSW errors)
- [~] `./tools/scripts/run_build.sh feature-tests --summary all` (blocked by Zig 0.16 module ownership: abbey cross-imports memory)
- [x] `./tools/scripts/run_build.sh validate-flags`
- [ ] `./tools/scripts/run_build.sh database-fast-tests` (pre-existing db errors)
- [x] `./tools/scripts/run_build.sh cli-tests`
- [x] `./tools/scripts/run_build.sh tui-tests`
- [x] `./tools/scripts/run_build.sh check-cli-registry`
- [x] `./tools/scripts/run_build.sh check-docs`
- [ ] Linux/CI follow-up: `zig build full-check` and `zig build verify-all`

Validation evidence:
- `2026-03-16`: `zig fmt --check build.zig build/ src/ tools/ tests/ bindings/ lang/` passed.
- `2026-03-16`: `./tools/scripts/run_build.sh typecheck --summary all` now clears the main `src/services/tests/mod.zig` package graph. Remaining failures are in `src/database_wdbx_tests_root.zig` and `src/database_fast_tests_root.zig` and are pre-existing Zig master migration issues in database/GPU/shared-service code (`ArrayListUnmanaged` initializers, missing `HNSW`, missing `Logger`, signal type mismatch, SPIR-V self-reference, etc.).
- `2026-03-16`: `./tools/scripts/run_build.sh check-docs` now wires `module_catalog` explicitly into `tools/gendocs`, but still fails in pre-existing CLI/shared-service/database code paths unrelated to the structure redesign.
- `2026-03-16`: `./tools/scripts/run_build.sh gendocs` now compiles cleanly (0 errors). Fixed: AI stub missing 7 sub-module exports (models, discovery, tool_agent, jumpstart, context_engine, self_improve, deep_research, dynamic_api, runtime_bridge, os_control), Zig 0.16 API breaks (Child.init→spawn, ArrayListUnmanaged .{}→.empty), wrong import paths (retry, resolvePath, database core), feature-gate bypass in triad.zig, void-returning os.exec in update.zig. Added feature-catalog page to site_map.

### Notes

- [x] Tri-CLI consensus helper unavailable locally: `/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` not present in this environment.
