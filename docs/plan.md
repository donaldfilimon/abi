---
title: Quality Plan
description: Test infrastructure and baseline management
---

# Quality Plan

For AI assistants: baseline numbers must match `scripts/project_baseline.env`. When test counts change, run `/baseline-sync` (see [CLAUDE.md — Skills, Plans, and Agents](../CLAUDE.md#skills-plans-and-agents-full-index)) or update all 10 files manually. Execution phases and multi-agent roles: [plans/plan.md](../plans/plan.md).

## Test Baselines

Canonical baseline: `scripts/project_baseline.env`

| Suite | Pass | Skip | Total |
|-------|------|------|-------|
| Main | 1270 | 5 | 1275 |
| Feature | 1535 | 0 | 1535 |

Canonical baseline 1270/1275 (5 skip), 1535/1535 feature — synchronized across all docs by `scripts/check_test_baseline_consistency.sh`.

## Quality Gates

| Gate | Command | Enforced |
|------|---------|----------|
| Format | `zig build lint` | CI |
| Unit tests | `zig build test --summary all` | CI |
| Feature tests | `zig build feature-tests --summary all` | CI |
| Flag validation | `zig build validate-flags` | CI |
| Baseline consistency | `scripts/check_test_baseline_consistency.sh` | CI |
| Zig 0.16 patterns | `scripts/check_zig_016_patterns.sh` | CI |
| Feature catalog | `scripts/check_feature_catalog_consistency.sh` | CI |
| CLI smoke tests | `zig build cli-tests` | CI |
| Full local gate | `zig build full-check` | Pre-PR |
