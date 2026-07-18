---
name: refactor-implementation
description: This skill should be used when the user is ready to execute a refactor plan and asks how to apply it safely — e.g. 'implement this modernization plan', 'extract this safely', 'how do I cut over without breaking things'.
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

Base directory for this skill: /Users/donaldfilimon/abi/.claude/skills/refactor-implementation
Relative paths in this skill (e.g., references/) are relative to this base directory.
