---
name: sea
description: Plan abi SEA (Sparse Evidence Attention) self-learning work — evidence-augmented completion, 8-signal scorer, EMA modulator persistence, and constitution audit. Use when working on ai_learn / complete --learn / evidence recall. Routes to sea-learn-loop, abi-superpower-sea, abi-superpower-constitution, and sea-learning-controller.
---

# sea

Entry point for abi's SEA self-learning loop (`src/features/ai/`). Routes:

| You want to… | Use |
| --- | --- |
| Drive `abi complete --learn` end-to-end | `sea-learn-loop` |
| Deep-dive SEA scoring / adaptive modulation | `abi-superpower-sea` |
| Constitution audit (observability-only) | `abi-superpower-constitution` |
| Toggle SEA in the REPL (`/learn`) | `sea-learning-controller` |

## Facts that constrain any SEA plan
- SEA = evidence-augmented self-learning completion with an 8-signal scorer +
  budgeted greedy selection; task-aware (7 task types shift signal weights).
- `AdaptiveModulator` weights (EMA, `alpha=0.3`, key `modulator:weights`)
  persist in WDBX **only on the `--learn`/SEA path**. Plain `complete` re-runs
  sentiment each turn with no EMA persistence.
- Constitution audit (6 principles) is **observability-only, not a gate** —
  sets `audit_passed` / `audit_vetoed` / `escore`, `std.log.warn`s on violation,
  but `complete` / `run` still return the response. Safety+privacy hard-veto
  only when either < 0.5.
- `routeInputAdaptive` in `router.zig` is unreferenced; the live EMA path is
  `completeAdaptive` / `completeWithStoreAdaptive` via `runLearnLoop` only.

## Honest boundary
No accuracy / energy / learning-gain claims without a repo benchmark.
