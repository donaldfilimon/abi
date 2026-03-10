# ABI Development Queue

This document tracks the active roadmap and sub-tasks. Use `git add -f tasks/todo.md` to stage changes. Update the queue at the end of every wave.

## Active Queue

### Completed - Documentation Consolidation & Cleanup Wave (2026-03-10)

#### Objective
Finalize the consolidation of repository guidelines into a canonical AGENTS.md, relocate detailed guidance to docs/FAQ-agents.md, and prune overlaps in CONTRIBUTING.md and CLAUDE.md.

#### Accomplished
- [x] Finalize `AGENTS.md` structure and content (~100 lines target).
- [x] Relocate detailed style and command FAQ to `docs/FAQ-agents.md`.
- [x] Create `docs/guides/cursor_rules.md` for policy scaffolding.
- [x] Prune redundant validation lists from `CONTRIBUTING.md`.
- [x] Shorten architecture and build sections in `CLAUDE.md`.
- [x] Implement `addHostScriptStep` in `build.zig` for macOS 26+ script validation.
- [x] Verify full baseline with `zig build full-check` and `refresh-cli-registry`.

---

### In Progress - Governance & Final Review (2026-03-10)

#### Objective
Establish a long-term maintenance cadence for the consolidated guidelines.

#### Plan
- [ ] **Quarterly Review (2026-06-10)**: Audit `AGENTS.md` and `docs/FAQ-agents.md` for drift.
- [ ] Review Cursor/Copilot policy placeholders as policy decisions are finalized.
- [x] Draft and finalize the PR summary for the consolidation wave.

---

## Backlog (expanded tasks)

1. [ ] Finalize `profiles` migration to enable deletion of legacy `personas/`.
2. [ ] Audit `tools/cli/commands/` for cross-platform robustness (Windows/WASM).
3. [ ] Consolidate `examples/ai_*.zig` into a unified `ai_suite.zig`.
4. [ ] Implement automated doc generation for the C bindings (if reinstated).
5. [ ] Prepare release notes and changelog entry for the consolidation wave (link to PR 483).
