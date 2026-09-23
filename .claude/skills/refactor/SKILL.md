---
name: refactor
description: Plan, execute, and validate a codebase modernization or refactor with clean-slate thinking while preserving semantics. Use when the user asks how to approach a rewrite vs incremental change or what the risk is (strategy), is ready to apply a refactor plan safely (implementation), or asks whether a refactor broke anything or is complete (validation).
---

# Refactor

One skill, three phases (merged from the former `refactor-strategy`, `refactor-implementation` and `refactor-validation` skills on 2026-09-21). Run the host gate before and after.

**Host gate** means the repository's own full gate, named in its `AGENTS.md`/`CLAUDE.md`. In abi (`~/dev/active/abi`) that is `./tools/check.sh` (it drives `./tools/cargo.sh`; never invoke bare `cargo` there); hand-run `cargo test` there needs `< /dev/null`. A bare `cargo test` is not a gate: in a `default-members` workspace it can run zero tests and pass.

## Phase 1: strategy

*Use when:* This skill should be used when the user asks how to approach a modernization — e.g. 'should we rewrite or do this incrementally', 'plan a refactor of X', 'what's the risk here' — at the start of significant work.


Provides structured approaches for modernizing codebases using clean-slate thinking while preserving semantics.

### When to use

Apply this skill at the beginning of any significant modernization effort.

### Core Principles

- Always start by defining the ideal modern implementation ("what would this look like written today?").
- Identify invariants and behavioral contracts that must be preserved.
- Choose strategy based on risk, size, and coupling: incremental, strangler-fig phased, or full rewrite of a module.
- Produce a concrete plan with milestones, validation gates, and rollback options.

### Recommended Process

1. Capture current behavior and success criteria (tests, contracts, SLAs).
2. Sketch the ideal modern design (modules, types, error handling, concurrency model, APIs).
3. Perform gap analysis between current and ideal.
4. Decide transformation strategy:
   - Small/low-risk → direct rewrite of the module.
   - Large/high-risk → phased (strangler fig, parallel implementation, feature flags).
5. Define validation criteria for each phase (parity tests, property tests, performance budgets).
6. Document the plan with clear "Definition of Done" for each step.

### Clean-Slate Mindset

When designing the target:
- Use current language idioms and stdlib features.
- Prefer explicit over implicit.
- Design for testability and observability from day one.
- Eliminate accidental complexity introduced by historical constraints.
- Choose composition and clear boundaries.

### Additional Resources

- `references/strategy-guide.md` — detailed decision trees and examples of each strategy.
- `examples/sample-plan-outline.md` — real plan outlines from previous modernizations.

Use this skill before touching code. Always run the host gate (see above) before and after.

### Optional host settings

For repo-local strictness (focus areas, gate list, claims discipline), copy the example template from the modern-refactor package to the host project as an optional Claude-side local override file (host-side only; this skill does not ship or auto-load host overrides).

## Phase 2: implementation

*Use when:* This skill should be used when the user is ready to execute a refactor plan and asks how to apply it safely — e.g. 'implement this modernization plan', 'extract this safely', 'how do I cut over without breaking things'.


Safe transformation techniques for applying modern designs while preserving behavior.

### Principles

- Write modern impl beside old (parallel) when risk high.
- Use strangler fig for gradual cutover.
- Validate at each step with the host gate, parity, contracts.
- Prefer direct boring code.

### Additional Resources

- `references/implementation-playbook.md`
- `examples/parallel-extract-outline.md`

Pair with the Rust-aware `abi` or `refactor-planner` agent for larger modules.

## Phase 3: validation

*Use when:* This skill should be used when the user asks to verify a refactor is done correctly — e.g. 'did I break anything', 'is this refactor complete', 'validate this change meets modern standards' — as the final gate.


Validation layers for modernization: behavioral parity, modern quality, structural.

### Layers

- Behavioral: contracts, tests, host gate, check-parity pass.
- Modern: apply patterns from modern-patterns, no legacy smells.
- Structural: boundaries clean, no god files, explicit over implicit.

### Additional Resources

- `references/validation-checklist.md`

Run the validation skill plus a Rust-aware `abi` or `refactor-planner` agent
review as the final step.

<!-- synced from central: ~/.grok/skills/refactor/SKILL.md -->
