---
name: abi-claims-validator
description: Validate ABI docs and external collateral against repo evidence and docs/contracts/external-claims-audit.mdx. Use when auditing claims, scanning docs for unsupported QPS/sharding/FHE/AES language, checking one claim before publish, or re-confirming the external-claims audit. Agent procedure only — not a CLI or slash command. Pair with abi-doc-claims-sync when editing docs.
---

# ABI Claims Validator

Agent-side procedure to validate documentation, collateral, and public claims
against the repository source of truth (`build.zig`, `src/`, contract tests,
`docs/contracts/external-claims-audit.mdx`). Use before publishing any external
artifact. There is **no** `/abi-claims-validator` binary or slash command — follow
the steps below (or dispatch the `external-claims-auditor` subagent).

For *editing* docs to stay claim-honest, prefer `abi-doc-claims-sync`.

## Procedure (agent actions)

### scan
1. Open the target tree (default `docs/`, plus README/AGENTS/CHANGELOG/walkthrough as needed).
2. Search for claim patterns (table below) with ripgrep or equivalent.
3. For each hit, cross-reference source/tests/`external-claims-audit.mdx`.
4. In `--strict` mode, treat any unproven number or deployment claim as a fail.

Detects:
- Performance numbers (QPS, latency, throughput, accuracy)
- Deployment claims (Kubernetes, H100, multi-node)
- Implementation language claims (Swift, Python, TensorFlow, PyTorch)
- Security claims (AES-256, RBAC, certifications)
- Distributed claims (sharding, multi-host)
- Benchmark comparisons (SQuAD, CodeSearchNet, GPT)
- Energy/GPU speedup claims

### check-claim
Given one claim string, return `SUPPORTED` | `UNSUPPORTED` | `PARTIAL` plus the
evidence path (source, test, or audit section). Prefer executable evidence over prose.

### audit
Produce a short markdown report of flagged claims vs evidence. The durable
ledger lives in `docs/contracts/external-claims-audit.mdx` — update that file
when the audit changes reality, do not invent a parallel truth source.

## Source of Truth Hierarchy

1. **Executable config**: `build.zig`, `build.zig.zon`, `.zigversion`
2. **Source implementation**: `src/`, `tests/contracts/`
3. **Contract tests**: `tests/contracts/*.zig` (surface, mcp_tools, feature_modules, plugin_registry, public_docs)
4. **Explicit claim boundary**: `docs/contracts/external-claims-audit.mdx`
5. **North-star mapping**: `docs/spec/wdbx-north-star.mdx` §2 (Current/Partial/Proposed)
6. **Prose docs**: `README.md`, `docs/*.mdx` — **lowest priority**, must reconcile upward

## Common Claim Patterns to Flag

| Pattern | Likely Status | Check Against |
|---------|---------------|---------------|
| `\d+\s*(QPS|req/s|throughput)` | ❌ Unsupported | `src/benchmarks.zig` artifact |
| `\d+\s*ms\s*latency` | ❌ Unsupported | `src/benchmarks.zig` artifact |
| `distributed|sharding|multi-host|cluster` | ⚠️ Partial | `cluster_rpc.zig` tests |
| `AES|RBAC|encryption|certified` | ❌ Unsupported | `external-claims-audit.mdx` §2 |
| `Swift|Python|TensorFlow|PyTorch|Kubernetes` | ❌ Unsupported | `build.zig` deps |
| `H100|A100|InfiniBand|NVLink` | ❌ Unsupported | CI config |
| `SQuAD|CodeSearchNet|GPT|benchmark` | ❌ Unsupported | Contract tests |
| `energy|kWh|efficiency|green` | ❌ Unsupported | No measurement artifact |

## Replacement Wording

From `docs/contracts/external-claims-audit.mdx` §Reusable Delta:

> Current ABI repo evidence supports a Zig 0.17 local AI orchestration framework with deterministic Abbey/Aviva/Abi profile routing, an in-process WDBX vector/key-value/block store, segment checkpoint plus WAL persistence with runtime recovery, a compatibility JSONL snapshot mirror, epoch reclamation helpers, snapshot-persisted temporal/causal graph records, MCP hybrid WDBX query ranking, HNSW-style cosine search, SHA-256-linked conversation blocks, 3D spatial search, feature-off stubs, CLI/MCP contract coverage, explicit connector live-mode boundaries, GPU capability reporting with CPU fallback, and a tested WDBX consensus RPC transport with shared-secret and optional peer-allowlist controls. The repo does not currently prove distributed sharding, production multi-host deployment, AES/RBAC WDBX storage, Swift/Python/TensorFlow implementation claims, Kubernetes/H100 deployment claims, regulatory certifications, QPS/latency/accuracy targets, GPU speedup figures, energy-efficiency metrics, or SQuAD/CodeSearchNet/GPT comparative scores.

## Integration

Pre-publish gate (agent runs the scan procedure after code gates):
```bash
./build.sh check-parity
./build.sh check
# then: scan docs/ + collateral per Procedure above (--strict)
```

No feature gates — procedural validation against source.