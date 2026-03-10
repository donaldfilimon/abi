# Task Tracker

This file is the canonical execution tracker for active ABI work.
Active and in-progress plans stay at the top. Completed plans move into the
archive section with their evidence preserved.

**How to use:** Work from the Active Queue top-down. Mark items `[x]` when done;
move completed plans to Archive with evidence. Backlog items can be promoted to
a new plan when starting a wave.

---

## Active Queue

### In Progress - Tooling and File Cleanup Sweep (2026-03-10)

#### Objective
Reduce obvious repo drift by removing or consolidating high-confidence stale,
duplicative, or awkwardly split files without changing ABI or CEL behavior.

#### Plan
- [ ] Inventory current cleanup candidates in the dirty tree and separate safe deletions/consolidations from unrelated user work.
- [ ] Normalize the formatter/tooling surface so contributors have one canonical repo-safe formatting path.
- [ ] Audit CEL/bootstrap docs and helper scripts for stale duplication between `.cel`, `.zig-bootstrap`, and `tools/scripts/`.
- [ ] Consolidate small redundant helpers or files only where the replacement path is already proven and documented.
- [ ] Leave risky or architectural cleanups in backlog instead of deleting speculatively.
- [ ] Re-run targeted validation for any touched scripts/docs/build flow and record evidence below.

#### Notes
- This sweep is intentionally limited to safe cleanup and consolidation. It is not a license to delete broad areas of bootstrap or CEL infrastructure without proof that they are unused.
- The AGENTS-required tri-CLI consensus wrapper is still absent locally (`/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh`), so planning continues best-effort with the blocker recorded explicitly.

### Ready - Post-Validation Cleanup (2026-03-10)

#### Objective
Keep `main` healthy after the Zig 0.16 validation recovery wave and only push or
open the next change wave once the current fixes are either committed or
explicitly handed off.

#### Plan
- [ ] Commit the current validation-recovery fixes when requested.
- [x] Add a dedicated repo-local format wrapper and docs guidance so contributors stop running `zig fmt .` across vendored bootstrap fixtures.
- [ ] Continue the broader cross-platform CLI/tooling audit from a clean post-recovery base.

#### Notes
- HEAD is currently `2087d055`.
- The AGENTS-required tri-CLI consensus wrapper is still absent locally (`/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh`), so this wave remained best-effort with the blocker recorded explicitly.
- `zig-bootstrap-emergency/zig/test/cases/compile_errors/` contains intentional upstream invalid Zig fixtures; repo-safe formatting continues to be `build.zig`, `build/`, `src/`, `tools/`, and `examples/`, or the canonical `zig build lint` / `zig build fix` steps.

#### Review Notes
- Validation now passes on this Darwin host through the Apple-`ld` build-runner wrapper:
  - `NO_COLOR=1 ./tools/scripts/run_build.sh validate-flags` ✅
  - `NO_COLOR=1 ./tools/scripts/run_build.sh feature-tests` ✅
  - `NO_COLOR=1 ./tools/scripts/run_build.sh examples` ✅
  - `NO_COLOR=1 ./tools/scripts/run_build.sh verify-all` ✅
  - `NO_COLOR=1 ./tools/scripts/run_build.sh full-check` ✅
  - `NO_COLOR=1 ./tools/scripts/run_build.sh benchmarks` ✅
- The current recovery fixes include:
  - `tools/scripts/fmt_repo.sh` now exposes the repo-safe `zig fmt` surface (`build.zig`, `build/`, `src/`, `tools/`, `examples/`) so contributors do not recurse into `zig-bootstrap-emergency/zig/test/cases/compile_errors/`.
  - `build/test_discovery.zig` now generates one ignored in-tree feature-test root instead of instantiating every manifest entry as a separate Zig module, which avoids duplicate file ownership and the unstable per-entry path generation that produced malformed `sfeatures/...` cache paths.
  - `src/features/ai/orchestration/*` and `src/features/ai/streaming/*` no longer use `@import("abi")` from inside feature modules, restoring compatibility with `validate-flags` and the import-rule contract.
  - `src/features/ai/orchestration/fallback.zig` keeps both shared `time` and `utils` imports so timeout bookkeeping and circuit-breaker timestamps each use the correct helper surface.
  - Contributor-facing docs (`AGENTS.md`, `README.md`, `CLAUDE.md`, and the zig-abi-plugin build references) now point to the safe formatter helper and the 42-combo flag matrix.
  - Hosted CI failures on `main` from runs `#795` and `#798` are still limited to the early `Shell Script Lint` and `Format Check` jobs; no heavier jobs ran before those failures.

---

## Next steps (actionable)

1. [ ] Commit the current validation recovery wave if the user wants the work preserved as a coherent checkpoint.
2. [ ] Push or open the next review step only after that checkpoint is in place.
3. [x] Add a small repo-local formatter wrapper and stronger docs guidance so `zig fmt .` no longer trips contributors over the vendored bootstrap tree.
4. [ ] Resume the broader cross-platform CLI/tooling audit from this now-green baseline.

---

## Backlog (expanded tasks)

### Build / Toolchain
- [ ] **Tri-CLI wrapper restoration**: Restore or replace `/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` so ABI workflow compliance is no longer best-effort.
- [ ] **Format wrapper**: Consider adding a repo-local format helper that only targets `build.zig`, `build/`, `src/`, `tools/`, and `examples/`.
- [ ] **Hosted CI follow-up**: Reconfirm hosted CI/billing health and mirror the now-green local validation wave there when needed.

### WDBX / Distributed
- [ ] **MCP server hardening**: Validate combined WDBX+ZLS MCP server end-to-end; add integration tests for `services/mcp/mod.zig`.

### CLI / Docs
- [ ] **CLI registry discipline**: Run `zig build refresh-cli-registry` and `zig build check-cli-registry` after command changes; keep docs in sync.

### Process
- [ ] **lessons.md upkeep**: After future production bugs or user corrections, append the prevention rule in the same wave.

---

## Archive

### Completed - Hosted Validation Recovery (2026-03-10)

#### Objective
Recover the blocked local validation wave on Darwin by fixing the remaining
repo-side Zig 0.16 issues rather than treating every failure as an upstream
toolchain problem.

#### Plan
- [x] Reproduce the failing high-level gates with the Darwin Apple-`ld` wrapper path.
- [x] Fix the manifest-driven feature-test builder so it no longer creates duplicate module ownership or malformed per-entry source paths.
- [x] Fix the AI feature-module import regressions surfaced by `validate-flags`.
- [x] Re-run `validate-flags`, `feature-tests`, `examples`, `verify-all`, `full-check`, and `benchmarks`.
- [x] Record the validation evidence and the formatter caveat in the tracker.

#### Notes
- The vendored emergency bootstrap tree remains intentionally excluded from normal formatting because it includes upstream compile-error fixtures.
- The Apple-`ld` runner path was sufficient on this host once the repo-side semantic and module-graph regressions were fixed.

#### Review Notes
- `build/test_discovery.zig` now generates `src/generated_feature_tests.zig` as an ignored file so all manifest entries are imported through one module graph instead of 178 separate synthetic modules.
- `src/features/ai/orchestration/mod.zig`, `src/features/ai/orchestration/fallback.zig`, `src/features/ai/orchestration/router.zig`, `src/features/ai/streaming/mod.zig`, `src/features/ai/streaming/server.zig`, `src/features/ai/streaming/session_cache.zig`, `src/features/ai/streaming/recovery.zig`, `src/features/ai/streaming/backpressure.zig`, and `src/features/ai/streaming/circuit_breaker.zig` were brought back to direct local/service imports instead of self-importing `abi`.
- Validation evidence on this host:
  - `NO_COLOR=1 ./tools/scripts/run_build.sh validate-flags` ✅
  - `NO_COLOR=1 ./tools/scripts/run_build.sh feature-tests` ✅
  - `NO_COLOR=1 ./tools/scripts/run_build.sh examples` ✅
  - `NO_COLOR=1 ./tools/scripts/run_build.sh verify-all` ✅
  - `NO_COLOR=1 ./tools/scripts/run_build.sh full-check` ✅
  - `NO_COLOR=1 ./tools/scripts/run_build.sh benchmarks` ✅
