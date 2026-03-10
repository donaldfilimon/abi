# ABI Development Queue

Active task tracker. Use `git add -f tasks/todo.md` to stage.

## Completed — Codebase Quality Sweep (PR #485)

All 5 waves committed on branch `fix/codebase-quality-sweep`:

- [x] Wave 1: Fix 33 corrupted files, create database/stub.zig, repair migration artifacts
- [x] Wave 2: Repair build system — broken test manifest, stale wdbx test root, dead persona tests
- [x] Wave 3: Repair coordination module — broken personas/ and database/ imports
- [x] Wave 4: Deep corruption sweep — 118+ additional truncated string literals across 66 files
- [x] Wave 5: AI integration bridges, mod/stub parity, doc updates, validation matrix fixes
- [x] Commit `68dcf34c` — 1081 files changed, +5157/-10927 lines
- [x] PR #485 updated with full change list

### New AI Integration (Wave 5)
- [x] `feedback/learning_bridge.zig` — FeedbackSystem → SelfLearningSystem closed loop
- [x] `agents/agent.zig` — AdvancedCognition + BackendMetrics
- [x] `multi_agent/runner.zig` — blackboard → experience buffer
- [x] `ralph/skills_store.zig` — skill quality tracking (execution_count, success_count, avg_quality)
- [x] `database/mod.zig` — expanded to 91-line API (parity with stub.zig)
- [x] `gpu/stub.zig` — added 3 missing sub-module stubs

---

## Active — Post-Sweep Cleanup

- [x] **Database boundary refactor (single change set)**: removed the public `wdbx` package surface, standardized on `abi.features.database`, rewired build/test roots away from the named `wdbx` module, and migrated in-tree callers/docs/parity checks together.
- [ ] Consensus status: best-effort tri-CLI wrapper unavailable at `/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh`; record this refactor as proceeding without wrapper output.
- [x] 2026-03-10 evidence: `zig fmt` applied to touched Zig files; compile-only checks passed via `zig build-obj -fno-emit-bin` for `src/root.zig`, `src/features/database/mod.zig`, `src/features/database/stub.zig`, `src/generated_feature_tests.zig`, `src/services/mcp/mod.zig`, and `src/core/database_fast_tests_root.zig`.
- [x] 2026-03-10 residual risk: `zig build validate-flags` and `zig build feature-tests --summary all` both failed in this environment with the known Darwin linker undefined-symbol issue before broader gate execution; `full-check` / `verify-all` remain pending on a host with a working Zig link path.
- [x] Fix stale README.md references (`abi.personas` → `abi.features.ai.profiles`)
- [x] Fix stale docs/api/v1.md references
- [x] Fix coordination mod/stub — inline CoordinationContext and InteractionCoordinator (removed dependency on deleted MultiPersonaSystem)
- [ ] Merge PR #485 to main

---

## Next Phase — Release & Scale

- [ ] **CI Restoration**: Push to main and verify GitHub Actions pass on Linux
- [ ] **WASM Optimization**: Refine freestanding distance functions for browser-side inference
- [ ] **API Expansion**: Implement missing OpenAI-compatible streaming endpoints
- [ ] **CEL Toolchain**: Build Zig from source on Darwin 25+ for native linking
- [ ] **Plugin Registry**: Push `zig-abi-plugin` to the official Claude Code registry

## Backlog

1. [ ] Finalize automated doc generation for cross-language bindings
2. [ ] Audit `tools/cli/commands/` for Windows compatibility
3. [ ] Implement distributed WAL for WDBX clusters
4. [ ] MCP server hardening (WDBX + ZLS integration)
5. [ ] Comprehensive test suite run on Linux CI to verify all waves
