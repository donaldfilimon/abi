---
name: Refactor Validation
description: This skill should be used when the user asks to "validate the refactor", "check parity", "from scratch quality review", "ensure modernization succeeded", "run modernization tests", "verify clean slate implementation", or needs criteria and methods to confirm that a refactored module truly achieves modern quality.
version: 0.1.0
---

# Refactor Validation

Defines what "as good as written from scratch" means and how to prove it.

## Validation Layers

1. **Behavioral Parity**
   - Existing tests still pass.
   - Property-based or contract tests cover key invariants.
   - Manual/edge scenarios from original.

2. **Modern Quality**
   - Passes modern patterns review (no legacy smells).
   - Improved or equal performance characteristics.
   - Better or equal observability and error messages.
   - Code is easier to understand and change (subjective + metrics).

3. **Structural**
   - Clear module boundaries.
   - No unnecessary coupling.
   - Documentation and types reflect the modern design.

## Recommended Checks

- Run full test suite + targeted new tests.
- Static analysis / linters with modern rules.
- Manual review using the modern-patterns skill.
- Compare size and complexity metrics (when meaningful).

## Additional Resources

- `references/validation-checklist.md`
- Use agents for deep review.

Only declare a refactor complete when all layers pass.
