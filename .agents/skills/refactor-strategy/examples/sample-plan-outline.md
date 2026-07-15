# Sample Modernization Plan Outline (from a real module)

## Target: Legacy Config Loader

**Clean Slate Vision**
- Strong typed config model (sum types for variants)
- Result-based loading with rich errors
- Single responsibility: load + validate + provide
- Easy to test with in-memory sources

**Strategy**: Direct rewrite (small surface, good existing tests)

**Phases**
1. Extract current contracts + tests (1 day)
2. Implement typed model + loader (parallel file)
3. Parity test suite (property tests for valid/invalid)
4. Cutover + delete old
5. Update callers

**Validation Gates**
- 100% of original test cases pass on new impl
- Error messages are actionable
- New loader used in 2+ call sites without breakage
