# CLI/TUI DSL + Global Addability Reorg Plan

## Objective
Complete the phased CLI/TUI organization migration so command/panel addability is metadata-driven and repetitive wrapper/catalog plumbing is removed from migrated families.

## Scope
- In scope:
- Finalize migrated command families to use `command` helper handlers and descriptor-derived unknown-subcommand suggestions.
- Keep UI command entrypoint canonical (`abi ui ...`) without legacy top-level aliases.
- Keep launcher/completion/registry organization on generated+override model and strengthen DSL consistency checks.
- Update source-of-truth docs and task tracking to match the new structure.
- Out of scope:
- Full declarative rewrite of command business logic bodies.
- Non-requested behavioral rewrites outside CLI/TUI metadata+routing.

## Verification Criteria
- `rg` guard patterns for migrated files show no legacy `wrapX` or `*_subcommands` artifacts in the targeted groups.
- CLI DSL consistency checker covers launcher/registry + migrated command anti-patterns.
- Documentation reflects canonical CLI/TUI addability flow and canonical API naming.

## Checklist
- [x] Plan logged before implementation
- [x] Implementation completed
- [ ] Verification commands executed
- [x] Review section completed

## Review
- Trigger: User-requested full phased implementation for CLI/TUI comptime DSL + global addability.
- Impact: Migrated command groups now rely on shared command helper DSL; unknown-subcommand suggestion and child wiring are descriptor-driven; legacy UI aliases removed; guardrails/docs updated.
- Plan change: Added context-aware parser helper to avoid per-command wrapper shims while preserving business logic functions.
- Verification change: Deferred full Zig matrix execution until explicitly requested.
