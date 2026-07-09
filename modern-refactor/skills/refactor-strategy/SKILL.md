---
name: Refactor Strategy
description: This skill should be used when the user asks to "plan a refactor", "from scratch modernization", "refactor strategy", "clean slate refactor", "phased modernization plan", "modernization roadmap", "how to approach this refactor from scratch", or needs guidance on choosing between incremental, phased, and full clean-slate rewrites.
version: 0.1.0
---

# Refactor Strategy

Provides structured approaches for modernizing codebases using clean-slate thinking while preserving semantics.

## When to use

Apply this skill at the beginning of any significant modernization effort.

## Core Principles

- Always start by defining the ideal modern implementation ("what would this look like written today?").
- Identify invariants and behavioral contracts that must be preserved.
- Choose strategy based on risk, size, and coupling: incremental, strangler-fig phased, or full rewrite of a module.
- Produce a concrete plan with milestones, validation gates, and rollback options.

## Recommended Process

1. Capture current behavior and success criteria (tests, contracts, SLAs).
2. Sketch the ideal modern design (modules, types, error handling, concurrency model, APIs).
3. Perform gap analysis between current and ideal.
4. Decide transformation strategy:
   - Small/low-risk → direct rewrite of the module.
   - Large/high-risk → phased (strangler fig, parallel implementation, feature flags).
5. Define validation criteria for each phase (parity tests, property tests, performance budgets).
6. Document the plan with clear "Definition of Done" for each step.

## Clean-Slate Mindset

When designing the target:
- Use current language idioms and stdlib features.
- Prefer explicit over implicit.
- Design for testability and observability from day one.
- Eliminate accidental complexity introduced by historical constraints.
- Choose composition and clear boundaries.

## Additional Resources

- `references/strategy-guide.md` — detailed decision trees and examples of each strategy.
- `examples/` — real plan outlines from previous modernizations.

Use this skill before touching code.
