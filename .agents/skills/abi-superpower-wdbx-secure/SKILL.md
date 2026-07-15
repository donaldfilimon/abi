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

## Actions

### compression
Int8 embedding quantization (~4× compression):
```
/abi-superpower-wdbx-secure compression --vectors '[[0.1,0.2],[0.3,0.4]]'
```

### entropy
Exact order-0 Huffman entropy coding (NOT ANS/arithmetic):
```
/abi-superpower-wdbx-secure entropy --data "compressible text data"
```

### neural
In-process trained autoencoder codec (reference, NOT SOTA):
```
/abi-superpower-wdbx-secure neural --vectors '[[0.1,0.2,0.3],[0.4,0.5,0.6]]'
```

### he
Additive single-key homomorphic encryption (NOT multi-key/FHE):
```
/abi-superpower-wdbx-secure he --data "aggregate values"
```

### fhe
DGHV somewhat-homomorphic encryption — encrypted add (XOR) + multiply (AND), depth-3 tested (reference parameters, NOT security-audited):
```
/abi-superpower-wdbx-secure fhe --data "encrypted computation"
```

### demo
Run full secure demo (all of the above):
```
/abi-superpower-wdbx-secure demo
```

## Implementation

| Module | Purpose | Status |
|--------|---------|--------|
| `src/features/wdbx/compression.zig` | Int8 quantization | Current |
| `src/features/wdbx/entropy.zig` | Order-0 Huffman | Current |
| `src/features/wdbx/neural_compress.zig` | Autoencoder (hand backprop) | Current |
| `src/features/wdbx/crypto_he.zig` | Additive HE | Current |
| `src/features/wdbx/fhe.zig` | DGHV SHE (add+multiply, depth-3) | Current |

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

## Feature Gates

Requires `feat-wdbx=true` (default). When disabled, returns `FeatureDisabled`.

## Testing

- `compression.zig` — quantization round-trip + determinism
- `entropy.zig` — encode/decode round-trip, compression ratio
- `neural_compress.zig` — training determinism, reconstruction error
- `crypto_he.zig` — additive homomorphism verification
- `fhe.zig` — DGHV add+multiply chain depth-3

All tests pass `./build.sh check`.