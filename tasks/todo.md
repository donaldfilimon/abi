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

- [ ] Rewrite `src/root.zig` to expose the direct-domain surface (`abi.runtime`, `abi.database`, `abi.ai`, `abi.foundation`, etc.)
- [ ] Add `build/module_catalog.zig` as the build/docs/test source of truth for public modules and feature-test entries
- [ ] Replace tracked generated test roots with build outputs and stop writing to `src/generated_feature_tests.zig`
- [x] Move aggregate test entrypoints to `tests/zig/` wrappers while preserving current test coverage
- [ ] Fix current master-branch import failures:
  - [ ] package-root import assumptions in the public surface
  - [ ] pseudo-submodule imports in `src/core/database/*`
  - [ ] ambient file imports in `src/services/tests/mod.zig`
- [ ] Rewire `tools/gendocs` to discover modules from the new catalog and root surface
- [ ] Keep the existing build command surface stable (`test`, `feature-tests`, `full-check`, `verify-all`, `check-docs`, `validate-flags`)

### Phase 2 — Physical Relayout

- [ ] Establish `src/internal/` family wrappers for app, foundation, runtime, ai, data, network, platform, integrations, observe, and tooling
- [x] Move bindings from `src/bindings/` to top-level `bindings/` and update build/install paths
- [ ] Reserve the `lang/cel/` lane and wire package metadata/build paths for future CEL relocation without changing stage0 behavior
- [ ] Update docs/templates/CLI surfaces toward the direct-domain API
- [ ] Delete obsolete tracked generated files and stale structure assumptions after the new paths are authoritative

### Validation

- [ ] `zig fmt --check build.zig build/ src/ tools/ tests/ bindings/`
- [ ] `./tools/scripts/run_build.sh typecheck --summary all`
- [ ] `./tools/scripts/run_build.sh feature-tests --summary all`
- [ ] `./tools/scripts/run_build.sh validate-flags`
- [ ] `./tools/scripts/run_build.sh database-fast-tests`
- [ ] `./tools/scripts/run_build.sh cli-tests`
- [ ] `./tools/scripts/run_build.sh tui-tests`
- [ ] `./tools/scripts/run_build.sh check-cli-registry`
- [ ] `./tools/scripts/run_build.sh check-docs`
- [ ] Linux/CI follow-up: `zig build full-check` and `zig build verify-all`

### Notes

- [ ] Record the unavailable tri-cli consensus helper as an environment limitation if implementation completes without it
