# Refactor Validation Checklist

Use before declaring any modernization complete.

## Behavioral Parity
- [ ] All original tests still pass
- [ ] New characterizing tests added for key paths
- [ ] Manual scenarios from original behavior verified
- [ ] Contracts / public API surface unchanged (or intentionally evolved with docs)

## Modern Quality
- [ ] Modern patterns applied (see modern-patterns skill)
- [ ] Error handling is explicit and typed where appropriate
- [ ] No silent failures or broad catches
- [ ] Code is easier to read and reason about
- [ ] Observability / logging improved or maintained

## Structural
- [ ] Clear module boundaries
- [ ] No god objects or excessive coupling
- [ ] Documentation and types reflect the new design
- [ ] Dead code from old approach removed (after cutover)

Run the validation skill + modern-refactorer agent review as final step.
