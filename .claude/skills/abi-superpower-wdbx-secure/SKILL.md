---
name: abi-superpower-wdbx-secure
description: WDBX secure/compression demos. Deterministic persisted PQ/autoencoder artifacts, additive/DGHV references, educational refresh, and optional pinned TFHE-rs execution with strict claim boundaries.
superpower:
  command: "execute"
  parameters:
    - name: "action"
      type: "string"
      enum: ["compression", "entropy", "neural", "he", "fhe", "dghv-bootstrap", "tfhe", "demo"]
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

The public CLI exposes a combined reference demo and two feature-gated,
explicitly non-production demonstrations:

```bash
abi wdbx secure demo
abi wdbx secure dghv-bootstrap
abi wdbx secure tfhe
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
| `crates/abi-wdbx/src/codecs/{pq,autoencoder}.rs` | Deterministic versioned persisted artifacts + validation/metrics | Current local/reference scope |
| `crates/abi-wdbx/src/v2/segment/codec.rs` | Segment codec integration and persisted metrics | Current local/reference scope |
| `crates/abi-wdbx/src/crypto_he.rs` | Additive HE | Current |
| `crates/abi-wdbx/src/fhe.rs` | DGHV SHE + feature-gated secret-key-assisted educational refresh | Current reference/demo |
| `crates/abi-wdbx/src/tfhe_demo.rs` | Pinned TFHE-rs boolean/integer/programmable-bootstrap execution | Optional `full-fhe` demo |

## Claim Boundary

Per `docs/spec/wdbx-north-star.mdx` §2 and `docs/contracts/external-claims-audit.mdx`:

| Demo | What it IS | What it is NOT |
|------|------------|----------------|
| Int8 quantization | ~4× embedding compression | Production learned codec |
| Order-0 Huffman | Exact lossless entropy coding | ANS/arithmetic/context-model |
| Neural compress | Hand-written backprop autoencoder | SOTA/production-scale learned codec |
| PQ/autoencoder artifacts | Deterministic persisted versioned codecs with local quality metrics | SOTA, production-quality, or universal compression |
| Additive HE | Single-key encrypted aggregation | Multi-key/FHE |
| DGHV SHE / refresh | Add+multiply plus secret-key-assisted educational refresh | Cryptographic bootstrapping, production FHE, security audit |
| TFHE-rs demo | Feature-gated pinned dependency APIs execute with ephemeral keys | ABI-authored FHE, independent audit, protective-use approval |

**Do not present as**: production encryption, ABI-authored bootstrapped FHE,
SOTA compression, multi-key HE, independently audited cryptography, or approved
protective use.

## CLI Access

```
abi wdbx secure demo
abi wdbx secure dghv-bootstrap   # feature experimental-dghv-bootstrap
abi wdbx secure tfhe             # release build, feature full-fhe
```

## Build and runtime boundary

`abi-wdbx` is a normal Rust workspace crate; there is no `feat-wdbx` switch or
`FeatureDisabled` stub. The public CLI intentionally exposes the combined
These commands are demonstrations, not production cryptography tools.

## Testing

- `compression.rs` — quantization round-trip + determinism
- `entropy.rs` — encode/decode round-trip, compression ratio
- `neural_compress.rs` — training determinism, reconstruction error
- `crypto_he.rs` — additive homomorphism verification
- `fhe.rs` — DGHV add+multiply chain depth-3
- `fhe.rs` — complete educational refresh/NAND truth tables behind its feature
- `tfhe_demo.rs` — pinned API execution via `./tools/check_full_fhe.sh`

Default/reference tests run under `./tools/check.sh`; the optional pinned
TFHE-rs release-mode path is gated separately by `./tools/check_full_fhe.sh`.
