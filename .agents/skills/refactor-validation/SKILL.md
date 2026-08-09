---
name: refactor-validation
description: This skill should be used when the user asks to verify a refactor is done correctly — e.g. 'did I break anything', 'is this refactor complete', 'validate this change meets modern standards' — as the final gate.
---

# Refactor Validation

Validation layers for modernization: behavioral parity, modern quality, structural.

## Layers

- Behavioral: contracts, tests, ./tools/check.sh, check-parity pass.
- Modern: apply patterns from modern-patterns, no legacy smells.
- Structural: boundaries clean, no god files, explicit over implicit.

## Additional Resources

- `.agents/skills/refactor-validation/references/validation-checklist.md`

Run the validation skill plus a Rust-aware `abi` or `refactor-planner` agent
review as the final step.

Base directory for this skill: /Users/donaldfilimon/abi/.agents/skills/refactor-validation
Relative paths in this skill (e.g., references/) are relative to this base directory.
