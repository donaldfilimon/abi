---
name: abi-superpower-ai
description: AI completion and SEA learning guide. Maps completion, streaming, training, and learning requests to real ABI CLI paths.
---

# ABI Superpower: AI

Maps AI completion, SEA learning, and training to the real CLI. There is no
`/abi-superpower-ai` binary or slash command.

## Actions

### complete
Run completion with optional streaming:
```bash
./target/debug/abi complete --model claude-fable-5 "explain the Rust workspace"
./target/debug/abi complete --learn --stream "code review"
```

### train
Train agent profiles against WDBX:
```bash
./target/debug/abi agent train abbey
```

### learn
Run SEA self-learning loop:
```bash
SEA_STORE=$(mktemp -d)
ABI_WDBX_PATH="$SEA_STORE" ./target/debug/abi complete --learn "task"
```

### stream
Stream completion tokens:
```bash
./target/debug/abi complete --stream --model claude-fable-5 "write a function"
```

### status
Show current AI configuration:
```bash
./target/debug/abi backends
```

## Profiles

- **abbey** - Primary empathetic polymath: warm, creative, explanatory, and technically precise
- **aviva** - Direct expert: concise, candid, analytical, and action-oriented
- **abi** - Adaptive orchestration/governance: intent, risk, context, policy, and mode selection

These are deterministic local profile routes in the ABI Rust runtime. The
canonical product identity and Current/Partial/Proposed capability mapping live
in `docs/spec/abbey-core-identity.mdx`; the labels are not model-quality claims.

## Implementation

Maps to:
- `crates/abi-ai/src/completion.rs` - deterministic completion and persistence helpers
- `crates/abi-sea/src/learn_loop.rs` - `run_learn_loop`, evidence recall
- `crates/abi-ai/src/constitution.rs` - 6-principle audit
- `crates/abi-ai/src/router.rs` - sentiment analysis, profile selection

## Runtime Boundary

The Rust workspace always links the local AI/SEA paths. Live HTTP completion is
Anthropic-only and requires explicit credentials; local mode is deterministic
persona-template generation, not a production model-quality claim.
