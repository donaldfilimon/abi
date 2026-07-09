---
name: Refactor Implementation
description: This skill should be used when the user asks to "implement the refactor", "rewrite this using modern patterns", "apply the clean slate design", "modernize this function/module", "perform the rewrite", or needs step-by-step guidance for safely transforming code while following from-scratch designs.
version: 0.1.0
---

# Refactor Implementation

Practical guidance for executing the modernization while maintaining correctness.

## Core Rules

- Never delete the old implementation until parity is proven.
- Write or extend tests first (or in parallel) for the target behavior.
- Make one semantic change at a time when possible.
- After each significant change, run relevant validation (see validation skill).
- Use the modern patterns skill constantly.

## Safe Transformation Techniques

- Parallel implementation (new file/module next to old).
- Strangler fig (introduce new behind flag or router).
- Expand / contract (add modern API, migrate callers, remove old).
- Extract pure functions before changing behavior.

## Working With Agents

Pair with `modern-refactorer` agent for larger modules.

## Additional Resources

- `references/implementation-playbook.md` — preconditions, gate table, cutover rules
- `examples/parallel-extract-outline.md` — thin-extract / strangler outline

Always leave the codebase in a better state than you found it, even mid-refactor.
