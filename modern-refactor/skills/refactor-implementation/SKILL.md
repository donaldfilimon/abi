---
name: refactor-implementation
description: Safe transformation techniques and implementation playbooks for clean-slate refactors in ABI. Use when executing modernization plans.
---

# Refactor Implementation

Safe transformation techniques for applying modern designs while preserving behavior.

## Principles

- Write modern impl beside old (parallel) when risk high.
- Use strangler fig for gradual cutover.
- Validate at each step with ./build.sh check, parity, contracts.
- Prefer direct boring code.

## Additional Resources

- `references/implementation-playbook.md`
- `examples/parallel-extract-outline.md`

Pair with `modern-refactorer` agent for larger modules.

Base directory for this skill: /Users/donaldfilimon/abi/modern-refactor/skills/refactor-implementation
Relative paths in this skill (e.g., references/) are relative to this base directory.
