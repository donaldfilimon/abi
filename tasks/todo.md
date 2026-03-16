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

### Validation

- [x] `zig fmt --check build.zig build/ src/ tools/ tests/ bindings/ lang/`
- [x] `./tools/scripts/run_build.sh typecheck --summary all` (passes; fixed AI stubs importing `abi` within `abi` module; Abbey no longer imports `abi`)
- [x] `./tools/scripts/run_build.sh feature-tests --summary all` (Resolved by using shared abi module in test_discovery.zig (commit d1e8c129))
- [x] `./tools/scripts/run_build.sh validate-flags`
- [x] `./tools/scripts/run_build.sh database-fast-tests` (all errors cleared)
- [x] `./tools/scripts/run_build.sh cli-tests`
- [x] `./tools/scripts/run_build.sh tui-tests`
- [x] `./tools/scripts/run_build.sh check-cli-registry`
- [x] `./tools/scripts/run_build.sh check-docs`
- [ ] Linux/CI follow-up: `zig build full-check` and `zig build verify-all`

Validation evidence:
- `2026-03-16`: `zig fmt --check build.zig build/ src/ tools/ tests/ bindings/ lang/` passed.
- `2026-03-16`: `./tools/scripts/run_build.sh typecheck --summary all` passed (after removing `@import("abi")` from AI stub facades and Abbey).
- `2026-03-16`: `./tools/scripts/run_build.sh feature-tests --summary all` still fails on Zig 0.16 module ownership conflicts in the generated feature-tests compilation graph (not an Abbey import-cycle anymore).
- `2026-03-16`: `./tools/scripts/run_build.sh database-fast-tests` passed after fixing Zig 0.16 compatibility.
- `2026-03-16`: `./tools/scripts/run_build.sh check-docs` passed after fixing Zig 0.16 compatibility.
- `2026-03-16`: Established `src/internal/` family wrappers for all core domains. Updated `src/root.zig` to use these wrappers.
- `2026-03-16`: `./tools/scripts/run_build.sh typecheck --summary all` still passes with the new `src/internal/` layout.
- `2026-03-16`: Redesigned all Markdown files with YAML frontmatter and unified headers. Updated `tools/gendocs` templates.
- 2026-03-16: feature-tests passing after rewriting test_discovery.zig to use abi module directly (eliminates per-entry module ownership conflicts).
- 2026-03-16: API surface redesign landed (fdfc9db3): src/internal/ wrappers, 57 CLI files migrated, Zig 0.16 platform fixes.

### Notes

- [x] Tri-CLI consensus helper unavailable locally: `/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` not present in this environment.
