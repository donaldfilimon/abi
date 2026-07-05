---
name: secure-demo
description: Build the abi CLI and run the WDBX security demo — int8 embedding compression, additive homomorphic-encryption sum, and a DGHV somewhat-homomorphic eval. Use when asked about WDBX compression ratios, homomorphic encryption, or the secure-demo output. Demo-grade, not security-audited.
---

# secure-demo — drive WDBX compression + homomorphic-encryption demo

Driver: **`.claude/skills/secure-demo/secure.sh`** (paths relative to repo root).
Read-only CLI capture — evidence is the `RESULT:` line + the per-section output.

## Run (agent path)
```bash
.claude/skills/secure-demo/secure.sh
```
Builds the CLI, runs `abi wdbx secure demo`, and asserts `compression:`,
`additive HE:`, `homomorphic eval:`, and `match=true`. Prints `RESULT: PASS`
(exit 0) or a FAIL count.

Verified this session: **PASS** on Zig master `0.17.0-dev.1099` — int8 ratio≈3.76x
(max_error≈0.0039), additive HE sum decrypts correctly, DGHV `(1 AND 1) XOR 0`→1.

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
| `build` FAIL | `/zig-build-doctor` or `./build.sh check`. |
| missing `match=true` | a HE/compression invariant broke — check `src/features/wdbx/{compression,crypto_he,fhe}.zig`. |
