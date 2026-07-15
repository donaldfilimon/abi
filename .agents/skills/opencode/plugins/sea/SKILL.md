---
name: sea
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

# SEA Superpower Plugin

Core SEA capabilities for OpenCode within the ABI framework.

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

## Actions

### complete
Evidence-augmented completion with 8-signal scorer:
```
/abi-superpower-sea complete --input "explain bias" --task writing --evidence-limit 20
```

### learn
Train SEA modulator with evidence recall.

### adaptive
Adaptive completion with EMA weights.

### evidence
Search and rank evidence for input.

## SEA Architecture

- 8 Signal Types: semantic, temporal, causal, persona, consistency, novelty, relevance, coherence
- Task-Aware Weighting, AdaptiveModulator (EMA alpha=0.3) in WDBX
- Evidence Budget

## Implementation

Maps to:
- `src/features/sea/learn_loop.zig`
- `src/features/sea/evidence.zig`
- `src/features/ai/constitution.zig`
- `src/features/wdbx/store.zig`

## Feature Gate

Requires `feat-sea=true` (default).
