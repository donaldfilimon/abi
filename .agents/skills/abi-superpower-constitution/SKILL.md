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

## Real access path

ABI does not expose `/abi-superpower-constitution` or a standalone constitution
CLI command. The audit runs inside completion flows; exercise it with
`abi complete <input>` or the MCP `ai_complete` / `ai_run` / `ai_learn` tools
and inspect the returned/stored metadata:

- `audit_passed` (bool)
- `audit_vetoed` (bool) — hard veto if safety OR privacy < 0.5
- `escore` (f32) — weighted constitutional score
- Per-principle scores (0.0-1.0)

Direct per-principle evaluation is a Rust library/test surface, not a public
slash command.

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
| Constitution Core | `crates/abi-ai/src/constitution.rs` — `validate()`, `evaluateResponse()`, `AuditResult` |
| Completion Integration | `crates/abi-ai/src/completion.rs` — audit called post-generation, metadata stored |
| MCP Tools | `crates/abi-mcp/src/ai_tools.rs` — audit fields in tool responses |

## Claim Boundary

Per `docs/contracts/external-claims-audit.mdx` and `docs/spec/multi-persona-technical.mdx`:
- ✅ 6-principle governance validation with per-principle scores
- ✅ Weighted E-score and safety/privacy lexical veto telemetry
- ✅ Surfaced in completion metadata and MCP responses
- ❌ NOT a gate — responses still returned even on veto
- ❌ NOT novel-harm detection or a general safety classifier; the current
  implementation is a small lexical rule set
- ❌ NOT case-sensitive — infix substring match ("harm" matches "harmless")
- ❌ NOT regulatory certification — repo has no compliance evidence
