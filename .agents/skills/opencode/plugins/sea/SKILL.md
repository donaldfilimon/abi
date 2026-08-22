---
name: sea
description: SEA (Sparse Evidence Attention) learning superpower. Evidence-augmented completion with task-aware scoring and adaptive modulation.
---

# SEA Superpower Plugin

Claim-honest routing to SEA through the real ABI CLI. This skill does not add a
`/abi-superpower-sea` binary or standalone adaptive/evidence commands.

## Capabilities

- SEA subsystem integration
- Plugin framework registration
- Runtime lifecycle management
- Configuration and settings management
- Status monitoring and reporting

## Integration Points

- ABI's SEA subsystem integration
- OpenCode plugin framework integration
- Runtime lifecycle management
- Configuration and settings management

## Real CLI path

```bash
SEA_STORE=$(mktemp -d)
ABI_WDBX_PATH="$SEA_STORE" ./target/debug/abi complete --learn "explain bias"
```

Task class is inferred. The public CLI does not expose `--task` or
`--evidence-limit`; MCP `ai_learn` accepts a capped `evidence_limit` up to 100.

## SEA Architecture

- 8 signals: semantic, keyword, metadata, recency, authority, graph,
  contradiction, and task fit
- Code repair, project recall, and benchmark review adjust baseline weights;
  all inferred task classes affect task-fit scoring
- Deterministic record/token/cluster/prompt budgets and a 100-candidate cap

## Implementation

Maps to:
- `crates/abi-sea/src/learn_loop.rs`
- `crates/abi-sea/src/evidence.rs`
- `crates/abi-ai/src/constitution.rs`
- `../wdbx/crates/abi-wdbx/src/store.rs`

## Runtime Boundary

SEA is linked into the Rust workspace and selected by `--learn`. Tests and
smokes must use a scratch store; `ABI_WDBX_PATH=:memory:` selects the no-store
fallback and does not prove evidence recall.
