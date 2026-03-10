# ABI Development Queue

This document tracks the active roadmap and sub-tasks. Use `git add -f tasks/todo.md` to stage changes. Update the queue at the end of every wave.

## Active Queue

### Completed - Architectural Consolidation & Close-out (2026-03-10)

#### Objective
Finalize the migration to Profiles, flatten the AI feature structure, and unify the vector database into a canonical V3 engine.

#### Accomplished
- [x] **Documentation**: Consolidated `AGENTS.md` and centralized guidance in `docs/FAQ-agents.md`.
- [x] **Technical Stability**: Implemented macOS 26+ bypass in `build.zig`.
- [x] **AI Migration**: Fully transitioned from legacy `personas` to the new modular `profiles` architecture.
- [x] **Framework Flattening**: Removed implementation suffixes and unified the `src/features/ai/` directory.
- [x] **Database Unification**: Merged `src/wdbx` into `src/core/database/` and updated all internal references.
- [x] **Platform Sanitization**: Audited `/tmp/` usage and introduced platform-agnostic OS utilities.
- [x] **Example Consolidation**: Merged disparate AI examples into a unified `examples/ai_suite.zig`.
- [x] **Orphan Cleanup (Wave 2)**: Removed deleted `src/personas/` directory, pruned legacy `src/features/ai/personas/` tree, verified no dangling imports.
- [x] **Import Hygiene (Wave 3)**: Fixed `wdbx` module to re-export `semantic_store` (required by `src/features/ai/memory/long_term.zig`), updated stale `personas.zig` comment in `ralph.zig`.
- [x] **Corruption Sweep (Wave 4)**: Fixed 118+ truncated string literals across 66 files caused by bulk "zig" stripping — restored `@import` paths, `.zig` extensions, `"zig"` comparisons, and displaced `")` characters.
- [x] **Doc Number Audit**: Updated stale counts across CLAUDE.md, README.md, SKILL.md — 19 feature modules, 90+ CLI commands, 42 flag combos.
- [x] **Mod/Stub Parity Audit**: Audited all 19 features; found 5 mismatches (database, gpu critical; web, cloud, observability minor).
- [x] **Validation Matrix Fix**: Added `.feat_mobile = true` to all 19 no-X entries that were missing it.
- [x] **Prompts/mod.zig Fix**: Replaced `undefined` return in `getPersona()` with working implementation using `personas` module.
- [x] **Database stub.zig**: Created comprehensive stub matching core/database public API (58 items, 20 sub-module stubs, semantic_store namespace).
- [x] **Orphaned Logic Dirs**: Fixed routing_logic/mod.zig (missing), aviva_logic/mod.zig (circular self-import), removed empty safety_logic/.

---

### In Progress - Codebase Quality Sweep Wave 5 (PR #485)

#### Agents Running (5 parallel)
- [ ] Create `src/features/ai/feedback/learning_bridge.zig` — feedback→learning closed loop
- [ ] Wire AdvancedCognition into `src/features/ai/agents/agent.zig`
- [ ] Add skill quality tracking to `tools/cli/commands/ai/ralph/skills_store.zig`
- [ ] Fix 5 broken imports (coordination, routing_logic, wdbx_fast_tests_root)
- [ ] Fix mod/stub parity (database mod.zig re-exports, gpu stub.zig missing modules)

#### Post-Agent Work
- [ ] Format check all modified files
- [ ] Commit and push all waves
- [ ] Update PR #485 description

---

### Planned - Release & Scale (Next Phase)

#### Objective
Transition from stability to broad feature expansion and ecosystem growth.

#### Plan
- [ ] **GitHub Actions Restoration**: Re-enable hosted CI now that the baseline is green.
- [ ] **WASM Optimization**: Refine freestanding distance functions for browser-side inference.
- [ ] **API Expansion**: Implement missing OpenAI-compatible streaming endpoints.
- [ ] **Ecosystem Growth**: Push the `zig-abi-plugin` to the official registry.

---

## Backlog

1. [ ] Finalize automated doc generation for cross-language bindings.
2. [ ] Audit `tools/cli/commands/` for full Windows compatibility.
3. [ ] Implement distributed WAL (Write Ahead Log) for WDBX clusters.
