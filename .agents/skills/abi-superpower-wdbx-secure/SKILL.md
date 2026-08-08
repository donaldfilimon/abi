---
name: abi-superpower-wdbx-secure
description: WDBX secure demos superpower. Int8 quantization, exact Huffman entropy coding, reference autoencoder, additive HE, DGHV somewhat-homomorphic add/multiply.
superpower:
  command: "execute"
  parameters:
    - name: "action"
      type: "string"
      enum: ["compression", "entropy", "neural", "he", "fhe", "demo"]
      description: "Secure demo action"
    - name: "data"
      type: "string"
      description: "Input data for demo"
    - name: "vectors"
      type: "string"
      description: "Vectors for compression (JSON array)"
---

# ABI Superpower: WDBX Secure Demos

Exposes WDBX security/compression demos as a superpower. All are **reference-grade, not production** — explicitly disclosed in `docs/spec/wdbx-north-star.mdx` §2 and `docs/contracts/external-claims-audit.mdx`.

## Real CLI action

The public CLI deliberately exposes one combined reference demonstration:

```bash
abi wdbx secure demo
```

Int8 quantization, order-0 Huffman, the reference autoencoder, additive HE,
and DGHV somewhat-homomorphic operations are library components exercised by
that demo and their crate tests. They are not separate production CLI tools.

## Implementation

| Module | Purpose | Status |
|--------|---------|--------|
| `crates/abi-wdbx/src/compression.rs` | Int8 quantization | Current |
| `crates/abi-wdbx/src/entropy.rs` | Order-0 Huffman | Current |
| `crates/abi-wdbx/src/neural_compress.rs` | Autoencoder (hand backprop) | Current |
| `crates/abi-wdbx/src/crypto_he.rs` | Additive HE | Current |
| `crates/abi-wdbx/src/fhe.rs` | DGHV SHE (add+multiply, depth-3) | Current |

## Claim Boundary

Per `docs/spec/wdbx-north-star.mdx` §2 and `docs/contracts/external-claims-audit.mdx`:

| Demo | What it IS | What it is NOT |
|------|------------|----------------|
| Int8 quantization | ~4× embedding compression | Production learned codec |
| Order-0 Huffman | Exact lossless entropy coding | ANS/arithmetic/context-model |
| Neural compress | Hand-written backprop autoencoder | SOTA/production-scale learned codec |
| Additive HE | Single-key encrypted aggregation | Multi-key/FHE |
| DGHV SHE | Encrypted add+multiply, depth-3 | Bootstrapped full FHE, security-audited |

**Do not present as**: production encryption, bootstrapped FHE, SOTA compression, multi-key HE, or security-audited schemes.

## CLI Access

```
abi wdbx secure demo
```

## Build and runtime boundary

`abi-wdbx` is a normal Rust workspace crate; there is no `feat-wdbx` switch or
`FeatureDisabled` stub. The public CLI intentionally exposes the combined
`abi wdbx secure demo`, not separate production cryptography commands.

## Testing

- `compression.rs` — quantization round-trip + determinism
- `entropy.rs` — encode/decode round-trip, compression ratio
- `neural_compress.rs` — training determinism, reconstruction error
- `crypto_he.rs` — additive homomorphism verification
- `fhe.rs` — DGHV add+multiply chain depth-3

All tests pass `./tools/check.sh`.
