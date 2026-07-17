---
name: abi-superpower-constitution
description: Constitution governance superpower. 6-principle response audit with E-score, veto, and surfaced telemetry.
superpower:
  command: "execute"
  parameters:
    - name: "action"
      type: "string"
      enum: ["audit", "evaluate", "principles"]
      description: "Constitution action"
    - name: "response"
      type: "string"
      description: "Response text to audit"
    - name: "profile"
      type: "string"
      enum: ["abbey", "aviva", "abi"]
      description: "Profile context for audit"
---

# ABI Superpower: Constitution

Exposes the 6-principle constitutional audit as a superpower. **Observability-only, not a gate** — sets `audit_passed`/`audit_vetoed`/`escore` in metadata and logs warnings, but `complete`/`run` still return the response.

## Actions

### audit
Run full constitutional audit on a response:
```
/abi-superpower-constitution audit --response "Your response text here" --profile abbey
```
Returns:
- `audit_passed` (bool)
- `audit_vetoed` (bool) — hard veto if safety OR privacy < 0.5
- `escore` (f32) — weighted constitutional score
- Per-principle scores (0.0-1.0)

### evaluate
Evaluate a response against a specific principle:
```
/abi-superpower-constitution evaluate --response "response" --principle safety
```

### principles
List the 6 constitutional principles:
```
/abi-superpower-constitution principles
```

## The 6 Principles

| Principle | Description | Veto Class |
|-----------|-------------|------------|
| **truthfulness** | Factual accuracy, no hallucination | — |
| **safety** | No harm, violence, illegal acts | **Safety** (hard veto if < 0.5) |
| **helpfulness** | Useful, actionable, complete | — |
| **fairness** | No bias, discrimination | — |
| **privacy** | No PII, secrets, confidential data | **Safety** (hard veto if < 0.5) |
| **transparency** | Clear about limitations, sources | — |

## Scoring Mechanics

- **Substring matching (infix, case-insensitive)** — "harm" fires on "harmless"
- **7 hardcoded negative substrings** per principle — cannot detect novel patterns
- **Weighted E-score** — aggregates principle scores with configurable weights
- **Hard veto** — if `safety < 0.5` OR `privacy < 0.5`, `audit_vetoed = true`

## Surfaced Telemetry

When `store_result=true` in completion:
- `audit_passed` (bool)
- `audit_vetoed` (bool)
- `escore` (f32)

MCP tools `ai_complete`/`ai_run`/`ai_learn` include audit fields in response.

## Implementation

| Component | Source |
|-----------|--------|
| Constitution Core | `src/features/ai/constitution.zig` — `validate()`, `evaluateResponse()`, `AuditResult` |
| Completion Integration | `src/features/ai/completion.zig` — audit called post-generation, metadata stored |
| MCP Tools | `src/mcp/ai_tools.zig` — audit fields in tool responses |

## Feature Gates

Requires `feat-ai=true` (default). When disabled, audit returns default-passed result.

## Claim Boundary

Per `docs/contracts/external-claims-audit.mdx` and `docs/spec/abi-refactor-design.mdx` §5.3:
- ✅ 6-principle governance validation with per-principle scores
- ✅ Weighted E-score and hard safety/privacy veto
- ✅ Surfaced in completion metadata and MCP responses
- ❌ NOT a gate — responses still returned even on veto
- ❌ NOT novel harm detection — only 7 hardcoded substrings per principle
- ❌ NOT case-sensitive — infix substring match ("harm" matches "harmless")
- ❌ NOT regulatory certification — repo has no compliance evidence