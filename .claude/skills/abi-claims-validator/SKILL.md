---
name: abi-claims-validator
description: ABI external claims validator. Validates docs/collateral against repo source, tests, and external-claims-audit.mdx. Prevents unsupported claims in public artifacts.
---

# ABI Claims Validator

Validates documentation, collateral, and public claims against the repository source of truth (`build.zig`, `src/`, contract tests, `docs/contracts/external-claims-audit.mdx`). Use before publishing any external artifact.

## Usage

```
/abi-claims-validator scan [--path docs/] [--strict]
/abi-claims-validator check-claim "<claim text>"
/abi-claims-validator audit [--output claims-audit.md]
```

## Actions

### scan
Scan Markdown/MDX files for claim patterns and cross-reference with repo evidence:
```
/abi-claims-validator scan --path docs/
```

Detects:
- Performance numbers (QPS, latency, throughput, accuracy)
- Deployment claims (Kubernetes, H100, multi-node)
- Implementation language claims (Swift, Python, TensorFlow, PyTorch)
- Security claims (AES-256, RBAC, certifications)
- Distributed claims (sharding, multi-host)
- Benchmark comparisons (SQuAD, CodeSearchNet, GPT)
- Energy/GPU speedup claims

### check-claim
Validate a specific claim against repo evidence:
```
/abi-claims-validator check-claim "WDBX achieves 12,000 QPS"
```

Returns: `SUPPORTED` | `UNSUPPORTED` | `PARTIAL` + evidence path

### audit
Generate full claims audit report:
```
/abi-claims-validator audit --output claims-audit.md
```

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

Run as pre-publish gate:
```bash
# In CI or local
zig build check-parity --summary all
./build.sh check
/abi-claims-validator scan --path docs/ --strict
```

## Feature Gates

No feature gates — runs as standalone validation tool against source.