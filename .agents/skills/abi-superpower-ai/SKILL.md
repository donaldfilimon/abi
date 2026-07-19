---
name: abi-superpower-ai
description: AI completion and SEA learning superpower. Run completions, training, streaming, and adaptive learning.
superpower:
  command: "execute"
  parameters:
    - name: "action"
      type: "string"
      enum: ["complete", "train", "learn", "stream", "status"]
      description: "AI action"
    - name: "input"
      type: "string"
      description: "Input prompt"
    - name: "model"
      type: "string"
      description: "Model ID (e.g., claude-fable-5)"
    - name: "profile"
      type: "string"
      enum: ["abbey", "aviva", "abi", "all"]
      description: "Agent profile for training"
---

# ABI Superpower: AI

Exposes AI completion, SEA learning, and training as a superpower.

## Actions

### complete
Run completion with optional streaming:
```
/abi-superpower-ai complete --input "explain Zig 0.17" --model claude-fable-5
/abi-superpower-ai complete --input "code review" --learn --stream
```

### train
Train agent profiles against WDBX:
```
/abi-superpower-ai train --profile abbey --dataset /path/to/dataset.jsonl
```

### learn
Run SEA self-learning loop:
```
/abi-superpower-ai learn --input "task" --evidence-limit 10
```

### stream
Stream completion tokens:
```
/abi-superpower-ai stream --input "write a function" --model local
```

### status
Show current AI configuration:
```
/abi-superpower-ai status
```

## Profiles

- **abbey** - Primary empathetic polymath: warm, creative, explanatory, and technically precise
- **aviva** - Direct expert: concise, candid, analytical, and action-oriented
- **abi** - Adaptive orchestration/governance: intent, risk, context, policy, and mode selection

These are deterministic local profile routes in the ABI Zig runtime. The
canonical product identity and Current/Partial/Proposed capability mapping live
in `docs/spec/abbey-core-identity.mdx`; the labels are not model-quality claims.

## Implementation

Maps to:
- `src/features/ai/completion.zig` - `completeWithStore()`, `completeWithStoreAdaptive()`
- `src/features/sea/learn_loop.zig` - `runLearnLoop()`, evidence recall
- `src/features/ai/constitution.zig` - 6-principle audit
- `src/features/ai/router.zig` - sentiment analysis, profile selection

## Feature Gate

Requires `feat-ai=true` and `feat-sea=true` (both default).
