---
title: "fpga-inference-acceleration"
tags: ["research", "fpga", "inference", "llm", "hardware"]
---
# FPGA Acceleration for LLM Inference

> **Document Version:** 1.0
> **Date:** January 2026
> **Status:** Research Phase
> **Related:** [hardware-acceleration-fpga-asic.md](./hardware-acceleration-fpga-asic.md)

## Executive Summary

This document explores the application of FPGAs (Field-Programmable Gate Arrays) specifically for accelerating Large Language Model (LLM) inference workloads within the ABI framework. FPGAs offer unique advantages for LLM inference including deterministic latency, custom datapath optimization for quantized formats, and superior power efficiency compared to GPUs for fixed workloads.

### Key Opportunities

| Operation | FPGA Advantage | Expected Improvement |
|-----------|----------------|---------------------|
| Quantized MatMul (Q4/Q8) | Native bit-width support | 15-25x perf/watt vs GPU |
| Attention Softmax | Streaming pipeline | 5-10x latency reduction |
| KV-Cache Management | On-chip SRAM | Eliminates memory bottleneck |
| Token Generation | Deterministic latency | <1ms variance |

---

## Table of Contents

1. [LLM Inference Bottlenecks](#1-llm-inference-bottlenecks)
2. [FPGA Architecture for LLM](#2-fpga-architecture-for-llm)
3. [Quantization Acceleration](#3-quantization-acceleration)
4. [Attention Mechanism Optimization](#4-attention-mechanism-optimization)
5. [Memory Architecture](#5-memory-architecture)
6. [Implementation Strategy](#6-implementation-strategy)
7. [Performance Projections](#7-performance-projections)
8. [Vendor Platforms](#8-vendor-platforms)
9. [References](#9-references)

---

## 1. LLM Inference Bottlenecks

### 1.1 Compute-Bound Operations

LLM inference is dominated by matrix-vector products during token generation:

```
Token Generation (LLaMA 7B):
├── Q @ K^T attention: ~134M FLOPs/token
├── Attention @ V:     ~67M FLOPs/token
├── FFN (up proj):     ~90M FLOPs/token
├── FFN (down proj):   ~90M FLOPs/token
└── Total:             ~380M FLOPs/token
```

**Current ABI Implementation:**
- `src/ai/llm/ops/matmul.zig` - 64x64 blocked matrix multiplication
- `src/ai/llm/ops/matmul_quant.zig` - Quantized formats (Q4_0, Q4_1, Q5, Q8)
- `src/ai/llm/ops/attention.zig` - Multi-head attention with optional flash attention

### 1.2 Memory-Bound Operations

The prefill phase and KV-cache access are memory-bandwidth limited:

```
Memory Bandwidth Requirements (7B model, batch=1):
├── Weight loading:     ~14 GB per full forward pass
├── KV-cache read:      ~2 MB per layer (context 2048)
├── KV-cache write:     ~128 KB per layer per token
└── Activation memory:  ~50 MB peak
```

### 1.3 Latency Sensitivity

Real-time applications require predictable latency:

| Metric | GPU Typical | FPGA Target |
|--------|-------------|-------------|
| First token latency | 50-200ms | 20-50ms |
| Inter-token latency | 15-30ms | 5-10ms |
| Latency variance | 10-50% | <5% |
| P99 latency | 2-3x median | 1.1x median |

---

## 2. FPGA Architecture for LLM

### 2.1 Proposed Block Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                     ABI LLM FPGA Accelerator                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Quantized Matrix Engine                   │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │   │
│  │  │ Q4 Dequant    │  │   Systolic    │  │ Accumulate    │   │   │
│  │  │ Pipeline      │→ │   Array       │→ │ + Bias        │   │   │
│  │  │ (32 values/   │  │ (256×256      │  │ + Activation  │   │   │
│  │  │  cycle)       │  │  INT8 MACs)   │  │               │   │   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Attention Engine                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │   │
│  │  │  Q×K^T   │→ │  Scale   │→ │  Softmax │→ │  Attn×V  │   │   │
│  │  │  MatMul  │  │  +Mask   │  │  (stream)│  │  MatMul  │   │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌───────────────────┐  ┌───────────────────────────────────────┐ │
│  │   KV-Cache SRAM   │  │          Memory Controller            │ │
│  │  ┌─────────────┐  │  │  ┌─────────┐  ┌─────────┐            │ │
│  │  │ Bank 0-15   │  │  │  │  DDR4   │  │  HBM2   │            │ │
│  │  │ (32 MB      │  │  │  │  64 GB  │  │  16 GB  │            │ │
│  │  │  total)     │  │  │  │  @3200  │  │  @460GB/s│           │ │
│  │  └─────────────┘  │  │  └─────────┘  └─────────┘            │ │
│  └───────────────────┘  └───────────────────────────────────────┘ │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    Control + DMA Engine                       │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │ │
│  │  │ Sequence │  │  Weight  │  │ Prefetch │  │  PCIe    │     │ │
│  │  │ Control  │  │  Loader  │  │  Engine  │  │  Gen5    │     │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │ │
│  └───────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

### 2.2 Resource Allocation

**Target Platform: AMD Alveo U250**

| Component | LUTs | DSPs | BRAM | URAM |
|-----------|------|------|------|------|
| Quantized Matrix Engine | 200K | 8,192 | 1,000 | 256 |
| Attention Engine | 150K | 2,048 | 500 | 64 |
| KV-Cache SRAM | 100K | 0 | 1,500 | 512 |
| Memory Controller | 50K | 0 | 100 | 0 |
| Control + DMA | 80K | 0 | 200 | 0 |
| **Total** | **580K** | **10,240** | **3,300** | **832** |
| **Available** | **1,727K** | **12,288** | **5,376** | **1,280** |
| **Utilization** | **34%** | **83%** | **61%** | **65%** |

---

## 3. Quantization Acceleration

### 3.1 Native Quantized Datapath

FPGAs excel at quantized inference because they can implement exact bit-width arithmetic:

```
Standard GPU Approach:
  Q4 → FP16 → FP16 MAC → FP16 → Q4
  (dequant)            (requant)

FPGA Native Approach:
  Q4 → INT4 × INT8 → INT32 accumulator → scaling → output
  (no intermediate FP conversion)
```

**Benefits:**
- 4x throughput (4-bit vs 16-bit)
- 8x energy efficiency (INT vs FP)
- Reduced memory bandwidth (quantized weights stay quantized)

### 3.2 Dequantization Pipeline

**ABI Q4_0 Format Mapping:**

```zig
// From src/ai/llm/tensor/quantized.zig
pub const Q4_0_Block = extern struct {
    d: f16,              // Scale factor
    qs: [16]u8,          // 32 4-bit values packed
};
```

**FPGA Pipeline Design:**

```
Clock 0: Load block (18 bytes)
Clock 1: Extract scale (d), broadcast to 32 lanes
Clock 2: Unpack 4-bit values (2 per byte)
Clock 3: Sign-extend INT4 → INT8
Clock 4: Scale by d (FP16 × INT8 → FP16)
Clock 5: Output 32 FP16 values

Throughput: 32 values per 5 cycles @ 300 MHz = 1.92 GOPS
```

### 3.3 Mixed-Precision Support

```
┌─────────────────────────────────────────────────────┐
│           Mixed-Precision Compute Unit               │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Input Format Selection:                             │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐           │
│  │ Q4_0 │  │ Q4_1 │  │ Q8_0 │  │ FP16 │           │
│  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘           │
│     │        │        │        │                   │
│     └────────┴────────┴────────┘                   │
│                 │                                   │
│                 ▼                                   │
│  ┌──────────────────────────────────────────┐     │
│  │        Unified MAC Array (INT32 acc)      │     │
│  │        256 parallel multiply-add ops      │     │
│  └──────────────────────────────────────────┘     │
│                 │                                   │
│                 ▼                                   │
│  ┌──────────────────────────────────────────┐     │
│  │     Output Scaling + Activation           │     │
│  │     (ReLU, SiLU, GELU)                    │     │
│  └──────────────────────────────────────────┘     │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## 4. Attention Mechanism Optimization

### 4.1 Streaming Softmax

Traditional softmax requires two passes (find max, compute exp/sum). FPGA enables single-pass streaming:

**Online Softmax Algorithm:**

```
def streaming_softmax(x_stream):
    m = -inf  # running max
    d = 0     # running denominator

    for x_i in x_stream:
        m_new = max(m, x_i)
        d = d * exp(m - m_new) + exp(x_i - m_new)
        m = m_new

    # Output phase
    for i, x_i in enumerate(x_stream):
        yield exp(x_i - m) / d
```

**FPGA Implementation:**

```
┌─────────────────────────────────────────────────┐
│              Streaming Softmax Unit              │
├─────────────────────────────────────────────────┤
│                                                  │
│  Input ──┬──▶ max(m, x) ──▶ m_new               │
│          │                    │                  │
│          │   ┌────────────────┘                  │
│          │   │                                   │
│          │   ▼                                   │
│          └──▶ exp(x - m_new) ──┬──▶ d_update    │
│                                 │                │
│                                 │                │
│  Buffer ◀──────────────────────┘                │
│  (store scaled values)                          │
│                                                  │
│  Output ◀── buffer[i] / d ◀── final normalize  │
│                                                  │
└─────────────────────────────────────────────────┘

Latency: seq_len + 10 cycles (vs 2 × seq_len for two-pass)
```

### 4.2 Flash Attention on FPGA

**Tiled Computation with On-Chip SRAM:**

```
For each query block Qi (size Br × d):
  Load Qi to SRAM
  For each key block Kj (size Bc × d):
    Load Kj to SRAM
    Compute Sij = Qi @ Kj^T in systolic array
    Apply streaming softmax (partial)
    Accumulate output Oi
  Normalize Oi and write back
```

**Memory Hierarchy:**

| Level | Size | Latency | Bandwidth |
|-------|------|---------|-----------|
| Registers | 64 KB | 1 cycle | 10 TB/s |
| BRAM | 4 MB | 2 cycles | 2 TB/s |
| URAM | 32 MB | 4 cycles | 500 GB/s |
| HBM | 16 GB | 100 cycles | 460 GB/s |
| DDR | 64 GB | 200 cycles | 77 GB/s |

### 4.3 Multi-Head Parallelism

```
┌─────────────────────────────────────────────────────────────┐
│                  Multi-Head Attention Array                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Head 0  │  │  Head 1  │  │  Head 2  │  │  Head 3  │   │
│  │  ├─Q─┐   │  │  ├─Q─┐   │  │  ├─Q─┐   │  │  ├─Q─┐   │   │
│  │  ├─K─┼─▶ │  │  ├─K─┼─▶ │  │  ├─K─┼─▶ │  │  ├─K─┼─▶ │   │
│  │  └─V─┘   │  │  └─V─┘   │  │  └─V─┘   │  │  └─V─┘   │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │             │             │          │
│       └─────────────┴─────────────┴─────────────┘          │
│                           │                                 │
│                           ▼                                 │
│                    ┌──────────────┐                        │
│                    │   Concat +   │                        │
│                    │   Linear     │                        │
│                    └──────────────┘                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘

With 32 heads and 4 parallel attention units:
  - 8 sequential steps per layer
  - Each step: 4 heads computed in parallel
```

---

## 5. Memory Architecture

### 5.1 KV-Cache Design

**On-Chip KV-Cache for Low Latency:**

```
Context length 2048, Model 7B (32 layers, 32 heads, head_dim 128):

KV-Cache per layer: 2 × 2048 × 32 × 128 × 2 bytes = 32 MB
Total KV-Cache: 32 × 32 MB = 1 GB

Strategy: Keep current layer KV in on-chip URAM (32 MB)
          Stream other layers from HBM as needed
```

**Hierarchical Cache Management:**

```
┌─────────────────────────────────────────────────────────┐
│                KV-Cache Hierarchy                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Level 1: On-chip URAM (32 MB)                          │
│  ├── Current layer K: 16 MB                             │
│  └── Current layer V: 16 MB                             │
│  Latency: 4 cycles                                      │
│                                                          │
│  Level 2: HBM Cache (256 MB)                            │
│  ├── Next 8 layers pre-fetched                          │
│  └── MRU eviction policy                                │
│  Latency: 100 cycles                                    │
│                                                          │
│  Level 3: DDR Storage (8+ GB)                           │
│  ├── Full context for all layers                        │
│  └── Async prefetch pipeline                            │
│  Latency: 200 cycles                                    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Weight Streaming

**Double-Buffering for Continuous Execution:**

```
Timeline:
┌──────────────────────────────────────────────────────────┐
│ Compute │ Layer N  │ Layer N+1 │ Layer N+2 │ Layer N+3  │
├─────────┼──────────┼───────────┼───────────┼────────────┤
│ DMA     │   N+1    │    N+2    │    N+3    │    N+4     │
│         │ prefetch │  prefetch │  prefetch │  prefetch  │
└──────────────────────────────────────────────────────────┘

Weight buffer: 2 × (layer_size / num_banks)
With 4 DDR banks @ 77 GB/s each:
  - 308 GB/s aggregate
  - Layer weights (7B/32 layers ≈ 220 MB) loaded in <1ms
```

---

## 6. Implementation Strategy

### 6.1 Phase 1: Single-Kernel Validation (Months 1-2)

**Objective:** Prove FPGA advantage for quantized matmul

**Deliverables:**
- HLS implementation of Q4_0 matrix-vector multiply
- Integration with `src/gpu/backends/fpga/kernels.zig`
- Benchmark vs CPU SIMD baseline

**Success Criteria:**
- 10x throughput improvement over CPU
- <5% accuracy loss vs FP32 reference

### 6.2 Phase 2: Attention Accelerator (Months 3-5)

**Objective:** Complete attention mechanism on FPGA

**Deliverables:**
- Streaming softmax kernel
- Multi-head attention with KV-cache
- Flash attention tiled implementation

**Success Criteria:**
- End-to-end attention latency <1ms (seq_len 2048)
- Memory bandwidth utilization >80%

### 6.3 Phase 3: Full Model Integration (Months 6-9)

**Objective:** Run complete 7B model on FPGA

**Deliverables:**
- Full inference pipeline
- Batched inference support
- Speculative decoding support

**Success Criteria:**
- 50+ tokens/second for 7B model
- <15W power consumption

---

## 7. Performance Projections

### 7.1 Single-User Inference

| Model Size | CPU (AVX-512) | GPU (A100) | FPGA (U250) |
|------------|---------------|------------|-------------|
| 7B Q4 | 8 tok/s | 120 tok/s | 60 tok/s |
| 13B Q4 | 4 tok/s | 80 tok/s | 35 tok/s |
| 70B Q4 | 0.5 tok/s | 25 tok/s | 8 tok/s |

### 7.2 Power Efficiency

| Platform | 7B Q4 Performance | Power | tok/s/W |
|----------|-------------------|-------|---------|
| CPU (Xeon 8380) | 8 tok/s | 270W | 0.03 |
| GPU (A100 PCIe) | 120 tok/s | 250W | 0.48 |
| FPGA (U250) | 60 tok/s | 75W | 0.80 |

**FPGA achieves 1.67x better tok/s/W than A100**

### 7.3 Latency Analysis

| Metric | GPU | FPGA |
|--------|-----|------|
| First token latency | 45ms | 25ms |
| Inter-token latency | 8ms | 15ms |
| P99 latency | 65ms | 28ms |
| Jitter (std dev) | 12ms | 2ms |

---

## 8. Vendor Platforms

### 8.1 AMD/Xilinx

**Recommended: Alveo U250/U280**

| Feature | U250 | U280 |
|---------|------|------|
| LUTs | 1.7M | 1.3M |
| DSPs | 12,288 | 9,024 |
| BRAM | 54 MB | 36 MB |
| Memory | 64 GB DDR4 | 8 GB HBM2 |
| Price | ~$6,500 | ~$12,000 |

**Development Tools:**
- Vitis HLS 2024.2+
- Vitis AI for model quantization
- XRT runtime for Zig integration

### 8.2 Intel

**Recommended: Agilex 7 F-Series**

| Feature | AGF027 | AGF014 |
|---------|--------|--------|
| ALMs | 2.5M | 1.4M |
| DSPs | 11,520 | 6,912 |
| M20K | 14,000 | 7,500 |
| Memory | HBM2e | DDR5 |
| Price | ~$15,000 | ~$8,000 |

**Development Tools:**
- Intel oneAPI HLS
- OpenVINO integration
- OpenCL runtime

### 8.3 Comparison

| Aspect | AMD/Xilinx | Intel |
|--------|------------|-------|
| Ecosystem maturity | Excellent | Good |
| HLS quality | Better | Good |
| AI tools (Vitis AI) | Excellent | Limited |
| Price/performance | Better | Moderate |
| Zig integration | Easier (XRT) | Harder (oneAPI) |

**Recommendation: AMD Alveo U250 for initial development**

---

## 9. References

### Academic Research

- Shao, J., et al. "FlexAttention: Efficient Attention Mechanism on FPGAs." FPGA 2024.
- Wang, Z., et al. "Efficient Transformer Inference on FPGAs." MICRO 2023.
- Huang, Y., et al. "FPGA-based LLM Inference with Mixed-Precision Quantization." DAC 2024.

### Industry Resources

- AMD Vitis AI Documentation: https://docs.xilinx.com/r/en-US/ug1414-vitis-ai
- AMD Vitis HLS User Guide: https://docs.amd.com/r/en-US/ug1399-vitis-hls
- Intel oneAPI FPGA Development: https://www.intel.com/content/www/us/en/developer/tools/oneapi/fpga.html

### ABI Framework References

- `src/ai/llm/ops/matmul_quant.zig` - Quantization formats
- `src/ai/llm/ops/attention.zig` - Attention implementation
- `src/gpu/backends/fpga/` - FPGA backend stubs
- `docs/research/hardware-acceleration-fpga-asic.md` - Overview document

---

*Document prepared for ABI Framework - FPGA LLM Inference Research*
