---
name: abi-superpower-sea
description: SEA (Sparse Evidence Attention) learning superpower. Evidence-augmented completion with task-aware scoring and adaptive modulation.
---

# ABI Superpower: SEA

Documents Sparse Evidence Attention learning as a superpower. There is no
`/abi-superpower-sea` binary or slash command; use the real CLI path below.

## Use

Run local evidence-augmented completion against a disposable durable store:

```bash
abi_sea_scratch="$(mktemp -d)"
ABI_WDBX_PATH="$abi_sea_scratch/wdbx" \
  ./target/debug/abi complete --learn "analyze document"
rm -rf "$abi_sea_scratch"
```

The task class and evidence limit are internal to this CLI path. There are no
public `--task` or `--evidence-limit` flags, and the CLI has no separate
`learn`, `adaptive`, or `evidence` subcommands. The MCP `ai_learn` tool accepts
an optional `evidence_limit` capped at 100.

## SEA Architecture

- **8 Signal Types**: semantic, keyword, metadata, recency, authority, graph,
  contradiction, and task fit
- **Task-Aware Weighting**: code repair, project recall, and benchmark review
  adjust the default weights; the other inferred task classes use the defaults
- **AdaptiveModulator**: EMA weights (alpha=0.3) stored in WDBX key `modulator:weights`
- **Evidence Budget**: deterministic selection bounded by record, token, cluster,
  and prompt-byte limits

## Implementation

Maps to:
- `crates/abi-sea/src/learn_loop.rs` - Core SEA algorithm
- `crates/abi-sea/src/evidence.rs` - Evidence scoring and recall
- `crates/abi-sea/src/scorer.rs` - Task-aware signal weights and ranking
- `crates/abi-sea/src/query_plan.rs` - Query classification and retrieval plan
- `crates/abi-ai/src/constitution.rs` - 6-principle constitutional audit
- `crates/abi-wdbx/src/store.rs` - Persistent modulator weights

## Runtime Boundary

SEA is linked into the Rust workspace; `abi complete --learn` selects the
evidence path. Without `--learn`, completion uses the base route. The
`:memory:` sentinel currently selects the disclosed no-store fallback, so SEA
evidence smokes must set `ABI_WDBX_PATH` to a scratch durable directory and
must never open the user's live `~/.abi` store.
