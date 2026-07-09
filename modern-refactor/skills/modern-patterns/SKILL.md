---
name: Modern Patterns
description: This skill should be used when the user asks to "apply modern patterns", "modernize this code", "use current idioms", "improve code with modern best practices", "refactor using modern error handling", "update to contemporary style", or needs concrete modern code patterns for types, errors, modularity, and structure.
version: 0.1.0
---

# Modern Patterns

Catalog of contemporary, language-agnostic and language-specific patterns that represent "how you would write it today."

## When to use

During implementation and review phases of any modernization.

## General Modern Principles (apply everywhere)

- Explicit over implicit.
- Fail fast with clear errors.
- Prefer composition and small focused units.
- Make illegal states unrepresentable (strong types).
- Separate pure logic from effects.
- Observability and testability built in.

## Common Modernization Targets

### Error Handling
- Use result types / typed errors instead of exceptions for expected failures.
- Distinguish domain errors, infrastructure errors, and bugs.
- Never swallow errors silently.

### Types & Modeling
- Use sum types / discriminated unions for variants.
- Prefer immutable data where possible.
- Model the domain accurately (newtypes, branded types).

### Modularity & Boundaries
- Clear public API surface.
- Dependency inversion for core logic.
- Avoid god objects and deep inheritance.

### Concurrency & Effects
- Structured concurrency.
- Explicit async boundaries.
- Prefer channels / streams over shared mutable state.

## Language Idioms

Consult language-specific references when targeting a particular stack (Zig, Rust, TypeScript, etc.).

## Additional Resources

- `references/patterns-catalog.md` — concrete before/after examples.
- `examples/` — small modernized modules.

Always compare proposed code against the clean-slate version of these patterns.
