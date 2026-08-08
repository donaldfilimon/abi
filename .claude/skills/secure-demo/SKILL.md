---
name: secure-demo
description: Build the abi CLI and run the WDBX security demo — int8 embedding compression, additive homomorphic-encryption sum, and a DGHV somewhat-homomorphic eval. Use when asked about WDBX compression ratios, homomorphic encryption, or the secure-demo output. Demo-grade, not security-audited.
---

# secure-demo — drive WDBX compression + homomorphic-encryption demo

Driver: **`.agents/skills/secure-demo/secure.sh`** (paths relative to repo root).
Read-only CLI capture — evidence is the `RESULT:` line + the per-section output.

## Run (agent path)
```bash
.agents/skills/secure-demo/secure.sh
```
Builds the CLI, runs `abi wdbx secure demo`, and asserts `compression:`,
`additive HE:`, `homomorphic eval:`, and `match=true`. Prints `RESULT: PASS`
(exit 0) or a FAIL count.

Current Rust driver: requires each compression/additive-HE/DGHV section and
`match=true`; it does not freeze a quality ratio or imply a cryptographic audit.

## Gotchas
- ⚠️ **Demo-grade, NOT production crypto.** The CLI says so: "DGHV somewhat-
  homomorphic scheme … reference parameters / bounded depth — not security-
  audited." Do not represent this as AES/RBAC/production encryption (see
  `docs/contracts/external-claims-audit.mdx`). Use the
  `compression-security-reviewer` subagent for an audit.
- Combines what the discovery split into "compression-demo" + "fhe-demo" — both
  ride the single `abi wdbx secure demo` surface.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL | `./tools/check.sh`. |
| missing `match=true` | a HE/compression invariant broke — check `crates/abi-wdbx/src/{compression,crypto_he,fhe}.rs`. |
