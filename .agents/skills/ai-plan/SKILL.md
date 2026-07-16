---
name: ai-plan
description: Plan abi AI work — model routing, completion, SEA self-learning, and training. Use when the user asks to plan AI orchestration, choose a model, run a completion, drive the SEA learn loop, or schedule training. Routes to complete-base, sea-learn-loop, agent-plan-train, and the abi-superpower-ai/sea/constitution superpowers.
---

# ai-plan

Entry point for abi's AI subsystem (`src/features/ai/`). Routes to the
specialist skills instead of duplicating them:

| You want to… | Use |
| --- | --- |
| Smoke-test base local completion (`abi complete`) | `complete-base` |
| Drive the SEA self-learning loop (`abi complete --learn`) | `sea-learn-loop` |
| Plan + train an agent (`abi agent plan` / `train`) | `agent-plan-train` |
| Deep-dive AI routing/constitution/SEA | `abi-superpower-ai`, `abi-superpower-sea`, `abi-superpower-constitution` |
| Toggle SEA mode in the REPL | `sea-learning-controller` (`/learn`) |

## Facts that constrain any AI plan (from source, not marketing)
- Default model `claude-fable-5`; `src/features/ai/models.zig` is the single
  source of truth for ids/aliases/provider routing (mod/stub parity).
- Router `selectBestProfile` ties resolve `abbey > aviva > abi`; neutral input
  routes to `abi`. `analyzeSentiment` is **prefix-only** keyword matching.
- Constitution audit is **observability-only, not a gate** — sets
  `audit_passed` / `audit_vetoed` / `escore`; safety+privacy hard-veto only
  when either scores < 0.5. Matching is case-insensitive **substring** (infix),
  so "harm" fires on "harmless" — it cannot detect novel harm patterns.
- EMA weights persist **only** on the `--learn`/SEA path; plain `complete`
  re-runs sentiment each turn with no EMA persistence.

## Honest boundary
No energy-efficiency, accuracy, or QPS claims without a repo test/benchmark
proving them (see `docs/contracts/external-claims-audit.mdx`).
