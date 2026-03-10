# Cleanup Plan: Consolidate Guidelines & Duplicates

Overview
- The repo hosts multiple guideline documents (AGENTS.md, CONTRIBUTING.md, CLAUDE.md, etc.). This plan outlines a safe, auditable path to consolidate and simplify these docs, reduce duplication, and ensure consistent governance.

Scope
- Merge overlapping guidance from AGENTS.md, CONTRIBUTING.md, CLAUDE.md where appropriate.
- Remove obviously redundant or outdated material, or move to a centralized FAQ or docs folder.
- Preserve the canonical references used by automated agents (e.g., AGENTS.md acts as the task contract).

Phases & Deliverables
- **Phase 1: Inventory** [DONE]
  * Inventory all guideline docs: AGENTS.md, CONTRIBUTING.md, CLAUDE.md, docs/ index pages that reference policies.
  * Collect snippets that define build/test commands, coding style, and governance rules.
- **Phase 2: Draft consolidation plan** [DONE]
  * Propose a single, cohesive AGENTS.md structure and content map.
  * Propose where to relocate or duplicate content (e.g., common docs folder).
- **Phase 3: Patch & validation** [IN PROGRESS]
  * Apply patch(s) to AGENTS.md and any relocated files.
  * Run full build/test checks to ensure no regressions; update references.
- **Phase 4: Cleanup and governance** [PLANNED]
  * Remove or archive dead docs; set up quarterly review cadence.

Targets for Consolidation
- Build/Test/Lint commands: unify phrasing and examples. [DONE]
- Code style guidelines: align with Zig conventions and repo conventions. [DONE]
- Cursor/Copilot policy sections: add if present; otherwise keep as a note. [DONE]
- Onboarding and governance references: centralize cross-doc links. [DONE]

Patch & Validation Plan
- Patch AGENTS.md to reflect the new consolidated structure. [DONE]
- Create a minimal FAQ under docs/ if needed to capture edge cases. [DONE]
- Validate with: `zig build full-check` and CLI registry refresh.

Acceptance Criteria
- AGENTS.md contains a clear, concise, task-oriented plan for build, lint, test, code style, and governance.
- No duplicative content remains; references are centralized or clearly linked.
- All automated checks pass locally (builds/tests/docs).

Notes
- This is a planning document; actual edits will be limited to patching the source files identified above.
