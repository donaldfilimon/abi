---
name: abi-superpower-sea
description: SEA (Sparse Evidence Attention) learning superpower. Evidence-augmented completion with task-aware scoring and adaptive modulation.
superpower:
  command: "execute"
  parameters:
    - name: "action"
      type: "string"
      enum: ["complete", "learn", "adaptive", "evidence"]
      description: "SEA action"
    - name: "input"
      type: "string"
      description: "Input prompt"
    - name: "task"
      type: "string"
      enum: ["writing", "coding", "math", "analysis", "creative", "debug"]
      description: "Task type for adaptive scoring"
    - name: "evidence_limit"
      type: "integer"
      description: "Max evidence references (0-100)"
---

# ABI Superpower: SEA

Exposes Sparse Evidence Attention learning as a superpower.

## Actions

### complete
Evidence-augmented completion with 8-signal scorer:
```
/abi-superpower-sea complete --input "explain bias" --task writing --evidence-limit 20
```

### learn
Train SEA modulator with evidence recall:
```
/abi-superpower-sea learn --input "analyze document" --task analysis
```

### adaptive
Adaptive completion with EMA weights:
```
/abi-superpower-sea adaptive --input "respond" --profile abbey
```

### evidence
Search and rank evidence for input:
```
/abi-superpower-sea evidence --input "what is bias?" --limit 10
```

## SEA Architecture

- **8 Signal Types**: semantic, temporal, causal, persona, consistency, novelty, relevance, coherence
- **Task-Aware Weighting**: 7 task types shift signal weights
- **AdaptiveModulator**: EMA weights (alpha=0.3) stored in WDBX key `modulator:weights`
- **Evidence Budget**: Limited greedy selection within budget

## Implementation

Maps to:
- `src/features/sea/learn_loop.zig` - Core SEA algorithm
- `src/features/sea/evidence.zig` - Evidence scoring and recall
- `src/features/ai/constitution.zig` - 6-principle constitutional audit
- `src/features/wdbx/store.zig` - Persistent modulator weights

## Feature Gate

Requires `feat-sea=true` (default). When disabled, falls back to base completion.