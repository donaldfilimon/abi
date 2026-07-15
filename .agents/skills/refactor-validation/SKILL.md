---
name: refactor-validation
description: Parity, modern quality criteria, and validation checklists for ABI refactors. Use as final step to ensure clean-slate changes meet standards.
---

# Refactor Validation

Validation layers for modernization: behavioral parity, modern quality, structural.

## Layers

- Behavioral: contracts, tests, ./build.sh check, check-parity pass.
- Modern: apply patterns from modern-patterns, no legacy smells.
- Structural: boundaries clean, no god files, explicit over implicit.

## Additional Resources

- `references/validation-checklist.md`

Run the validation skill + modern-refactorer agent review as final step.

Base directory for this skill: /Users/donaldfilimon/abi/.agents/skills/refactor-validation
Relative paths in this skill (e.g., references/) are relative to this base directory.
