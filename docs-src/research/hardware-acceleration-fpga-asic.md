---
title: "hardware-acceleration-fpga-asic"
tags: []
---
# Hardware Acceleration Research: FPGA & ASIC for ABI
> **Codebase Status:** Synced with repository as of 2026-01-23.

**Document Version:** 1.0
**Date:** January 2026
**Status:** Research & Planning Phase

## Executive Summary

This document presents a comprehensive analysis of hardware acceleration opportunities for the ABI framework using FPGAs (Field-Programmable Gate Arrays) and ASICs (Application-Specific Integrated Circuits). Based on detailed analysis of the codebase architecture and current industry trends, we identify high-impact acceleration targets and propose an implementation roadmap.

### Key Findings

| Area | Current State | FPGA Potential | ASIC Potential |
|------|--------------|----------------|----------------|
| LLM Inference | CPU + optional GPU | 5-15× speedup | 20-50× speedup |
| Vector Search (HNSW) | SIMD + GPU batch | 10-50× speedup | 30-100× speedup |
| Quantized MatMul | CPU with unrolled loops | 10-20× speedup | 50-100× speedup |
| K-Means Clustering | CPU sequential | 20-100× speedup | 100×+ speedup |

### Recommendations

1. **Short-term (0-6 months):** Implement FPGA-accelerated vector similarity search using AMD Vitis HLS
2. **Medium-term (6-18 months):** Develop quantized matrix multiplication FPGA cores for LLM inference
3. **Long-term (18+ months):** Evaluate custom ASIC development for high-volume deployment scenarios

---

## Related Documents

This document is part of the FPGA/ASIC research series:

- **[FPGA Inference Acceleration](./fpga-inference-acceleration.md)** - Detailed FPGA design for LLM inference
- **[Custom ASIC Considerations](./custom-asic-considerations.md)** - When and how to pursue ASIC development
- **[Hybrid GPU-FPGA Architecture](./hybrid-gpu-fpga-architecture.md)** - Combined GPU and FPGA approaches
- **[FPGA Backend README](../../src/gpu/backends/fpga/README.md)** - Implementation status and usage

---

## Table of Contents

1. [FPGA vs ASIC Trade-offs](#1-fpga-vs-asic-trade-offs)
2. [Current Architecture Analysis](#2-current-architecture-analysis)
3. [Compute-Intensive Workloads](#3-compute-intensive-workloads)
4. [FPGA Acceleration Opportunities](#4-fpga-acceleration-opportunities)
5. [ASIC Acceleration Opportunities](#5-asic-acceleration-opportunities)
6. [Vendor Options](#6-vendor-options)
7. [Development Tools & Frameworks](#7-development-tools--frameworks)
8. [Implementation Roadmap](#8-implementation-roadmap)
9. [Cost-Benefit Analysis](#9-cost-benefit-analysis)
10. [Performance Projections](#10-performance-projections)
11. [Timeline Considerations](#11-timeline-considerations)
12. [Risk Analysis](#12-risk-analysis)
13. [References & Resources](#13-references--resources)

---

## 1. FPGA vs ASIC Trade-offs

### 1.1 Fundamental Differences

| Aspect | FPGA | ASIC |
|--------|------|------|
| **Definition** | Reconfigurable logic fabric | Fixed-function silicon |
| **NRE Cost** | $100K-$500K | $10M-$50M |
| **Unit Cost (10K vol)** | $5,000-$15,000 | $50-$200 |
| **Time to Market** | 6-12 months | 18-36 months |
| **Performance** | 1-5x vs GPU | 10-50x vs GPU |
| **Power Efficiency** | 5-10x vs GPU | 20-100x vs GPU |
| **Flexibility** | Full reconfiguration | None post-silicon |
| **Risk** | Low-Medium | High |

### 1.2 Decision Matrix

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hardware Selection Guide                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Volume < 10K units/year?                                        │
│  ├── Yes → FPGA or GPU                                          │
│  └── No ↓                                                        │
│                                                                  │
│  Workload stable for 3+ years?                                   │
│  ├── No → FPGA (reconfigurable)                                 │
│  └── Yes ↓                                                       │
│                                                                  │
│  Power budget < 25W?                                             │
│  ├── No → GPU may suffice                                       │
│  └── Yes ↓                                                       │
│                                                                  │
│  Latency < 1ms required?                                         │
│  ├── No → FPGA                                                  │
│  └── Yes → FPGA or ASIC depending on volume                     │
│                                                                  │
│  Budget > $15M NRE?                                              │
│  ├── No → FPGA                                                  │
│  └── Yes → Evaluate ASIC                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Use Case Alignment

| Use Case | Best Choice | Rationale |
|----------|-------------|-----------|
| Prototype/R&D | FPGA | Low risk, rapid iteration |
| Low volume (<1K/year) | FPGA | Unit economics favor FPGA |
| Medium volume (1K-100K/year) | FPGA or Structured ASIC | Balance of cost and flexibility |
| High volume (>100K/year) | Full Custom ASIC | NRE amortizes, unit cost dominates |
| Edge deployment | FPGA initially, ASIC later | Validate design before committing |
| Evolving algorithms | FPGA | Can update in field |
| Fixed algorithms | ASIC | Maximum efficiency |

### 1.4 ABI Framework Recommendation

Given current stage and requirements:

1. **Immediate (2026)**: FPGA development for validation
2. **Near-term (2027)**: Production FPGA deployment
3. **Long-term (2028+)**: ASIC evaluation if volume justifies

---

## 2. Current Architecture Analysis

### 2.1 GPU Backend Architecture

The ABI framework implements a sophisticated, layered GPU acceleration system:

```
┌─────────────────────────────────────────────────────────┐
│  User API Layer (src/gpu/unified.zig)                   │
├─────────────────────────────────────────────────────────┤
│  Dispatcher Layer (src/gpu/dispatcher.zig)              │
│  Routes operations to backends, manages kernel cache    │
├─────────────────────────────────────────────────────────┤
│  Backend Factory (src/gpu/backend_factory.zig)          │
│  Instantiates backends with priority selection          │
├─────────────────────────────────────────────────────────┤
│  Backend Interface (src/gpu/interface.zig)              │
│  VTable-based polymorphism for runtime dispatch         │
├─────────────────────────────────────────────────────────┤
│  Concrete Backends                                      │
│  ┌─────────┬─────────┬───────┬────────┬────────┐       │
│  │  CUDA   │ Vulkan  │ Metal │ WebGPU │ OpenGL │       │
│  └─────────┴─────────┴───────┴────────┴────────┘       │
└─────────────────────────────────────────────────────────┘
```

**Key Architectural Patterns:**

- **VTable Interface:** Type-erased polymorphism via `*anyopaque` pointers
- **Portable Kernel DSL:** Backend-agnostic kernel definition with multi-target code generation
- **Unified Buffer System:** Smart buffers with automatic CPU/GPU synchronization
- **Execution Coordinator:** Adaptive fallback chain (GPU → SIMD → Scalar)

### 2.2 Runtime Engine

The work-stealing task execution engine (`src/runtime/engine/`) provides:

```zig
pub const WorkloadHints = struct {
    cpu_affinity: ?u32 = null,
    estimated_duration_us: ?u64 = null,
    prefers_gpu: bool = false,      // Soft preference
    requires_gpu: bool = false,     // Hard requirement
};
```

**Integration Points for Hardware Accelerators:**

1. **Dual VTable Architecture:** `WorkloadVTable` (CPU) + `GPUWorkloadVTable` (GPU/accelerator)
2. **Priority Queue Scheduling:** Multi-level scheduler with aging prevention
3. **NUMA-Aware Execution:** CPU affinity and topology detection
4. **Sharded Results Storage:** 16-shard map for reduced lock contention

### 2.3 Current SIMD Optimizations

The database module (`src/shared/simd.zig`) implements vectorized operations:

| Operation | Implementation | Vector Width |
|-----------|----------------|--------------|
| `vectorDot` | SIMD accumulation | Auto-detected (AVX-512/NEON/WASM) |
| `vectorL2Norm` | SIMD squared-sum | Auto-detected |
| `cosineSimilarity` | Fused dot + norms | Auto-detected |
| `batchCosineSimilarity` | Pre-computed query norm | Auto-detected |

---

## 3. Compute-Intensive Workloads

### 3.1 LLM Inference Operations

**Per-Token Compute Requirements (LLaMA 7B):**

| Operation | FLOPs/Token | Memory Access | Parallelism |
|-----------|-------------|---------------|-------------|
| Attention (Q@K^T) | ~134M | O(N²) | High (per-head) |
| Softmax | ~4M | O(N) per row | High (per-row) |
| FFN (SwiGLU) | ~180M | O(dim × ffn_dim) | Very High |
| RMSNorm | ~16K | O(dim) | High (reduction) |
| RoPE | ~8K | O(head_dim) | High (per-pair) |

**Quantization Formats Supported:**

```
Q4_0: 32 values in 18 bytes (4-bit signed, f16 scale)
Q4_1: 32 values in 20 bytes (4-bit unsigned, f16 scale + min)
Q5_0/Q5_1: 5-bit quantization
Q8_0: 8-bit signed quantization
```

**Key Files:**
- `src/ai/implementation/llm/ops/attention.zig` - Attention mechanisms
- `src/ai/implementation/llm/ops/matmul.zig` - Matrix multiplication (64×64 blocks)
- `src/ai/implementation/llm/ops/matmul_quant.zig` - Quantized matmul
- `src/ai/implementation/llm/tensor/quantized.zig` - Quantization formats

### 3.2 Vector Database Operations

**HNSW Search Complexity:**

```
Per search: O(ef_construction × dimension) distance computations
Default: ef_construction = 100, dimension = 768
→ ~76,800 float operations per search (dot product + norm)
```

**K-Means Clustering:**

```
Per iteration: O(n_vectors × n_clusters × dimension)
Typical: 300 iterations × 10k vectors × 16 clusters × 768 dims
→ 36.8 billion FLOPs for index construction
```

**Key Files:**
- `src/database/hnsw.zig` - Graph-based ANN search
- `src/database/clustering.zig` - K-means implementation
- `src/database/gpu_accel.zig` - GPU acceleration interface

### 3.3 Training Operations

**Backward Pass Requirements:**

| Operation | Compute | Memory |
|-----------|---------|--------|
| Attention backward | 3× forward | 2× activations |
| MatMul backward | 2× forward | Weight gradients |
| RMSNorm backward | 1× forward | Input cache |
| Loss + Softmax | O(vocab_size × batch) | Per-token |

---

## 4. FPGA Acceleration Opportunities

### 4.1 Why FPGAs for ABI?

**Advantages:**

1. **Reconfigurability:** Adapt to evolving model architectures without new silicon
2. **Low Latency:** Deterministic execution, no OS/driver overhead
3. **Power Efficiency:** 5-10× better perf/watt vs GPUs for fixed workloads
4. **Custom Data Paths:** Native support for quantized formats (Q4, Q5, Q8)
5. **Memory Architecture:** On-chip SRAM eliminates memory bandwidth bottlenecks

**Industry Validation:**

- SmartANNS (USENIX ATC 2024): FPGA-based HNSW on computational storage devices
- Falcon: FPGA graph vector search on AMD Alveo U250, achieves near-linear scaling
- hls4ml: Open-source framework deploying neural networks on FPGAs

### 4.2 Priority Acceleration Targets

#### Tier 1: Highest Impact

**1. Quantized Matrix Multiplication**

```
Current: CPU with inline dequant, unrolled loops
FPGA Design:
  ├─ Custom Q4/Q8 → FP32 dequantization pipeline
  ├─ Systolic array for matrix multiply (256×256 PE)
  ├─ On-chip weight buffer (fits 4096×4096 Q4 matrix)
  └─ Streaming output to next operation

Expected Speedup: 10-20×
Power Reduction: 5-8×
```

**2. HNSW Distance Computation**

```
Current: Sequential vectorDot() + vectorL2Norm()
FPGA Design:
  ├─ Parallel dot product units (256+ MACs)
  ├─ Streaming vector input from DDR/HBM
  ├─ On-chip distance cache (16KB LRU)
  ├─ Pipelined output to result heap
  └─ Prefetch next neighbors while computing

Expected Speedup: 10-50×
Latency: <1μs per distance computation
```

**3. Attention Softmax**

```
Current: Numerically stable max-based normalization
FPGA Design:
  ├─ Parallel reduction tree for max/sum
  ├─ Pipelined exp() using LUT + polynomial
  ├─ Fused scale + mask application
  └─ Streaming output (no intermediate storage)

Expected Speedup: 5-10×
```

#### Tier 2: Medium Impact

**4. K-Means Centroid Assignment**

```
Current: n_vectors × n_clusters distance computations
FPGA Design:
  ├─ All centroids in on-chip BRAM (<256KB for 1k×768)
  ├─ Stream vectors through
  ├─ Parallel distance to all centroids
  ├─ Argmin logic in hardware
  └─ Output cluster ID stream

Expected Speedup: 20-100×
```

**5. RoPE (Rotary Position Embeddings)**

```
Current: Precomputed sin/cos + rotation
FPGA Design:
  ├─ On-chip sin/cos table (max_seq_len entries)
  ├─ 2D rotation units (complex multiply)
  ├─ Streaming Q/K application
  └─ Zero additional memory bandwidth

Expected Speedup: 3-5×
```

**6. Product Quantization (IVF-PQ)**

```
Current: LUT lookup + linear interpolation
FPGA Design:
  ├─ 64-entry LUT per subvector hardcoded
  ├─ 8 subvectors × parallel decode
  └─ 1 billion codes/sec @ 1 GHz

Expected Speedup: 5-10×
```

### 4.3 FPGA Backend Integration

**Proposed Architecture:**

```
src/gpu/backends/fpga/
├── mod.zig           # Module entry point
├── loader.zig        # Bitstream loading (Vivado/Vitis)
├── memory.zig        # DDR/HBM memory management
├── vtable.zig        # VTable implementation
├── kernels/
│   ├── distance.zig  # Vector distance computation
│   ├── matmul.zig    # Quantized matrix multiply
│   ├── softmax.zig   # Attention softmax
│   └── kmeans.zig    # K-means centroid matching
└── hls/              # HLS source files (C++)
    ├── distance.cpp
    ├── matmul_q4.cpp
    └── softmax.cpp
```

**VTable Implementation:**

```zig
pub const FpgaVTable = gpu.interface.Backend.VTable{
    .deinit = fpgaDeinit,
    .getDeviceCount = fpgaGetDeviceCount,
    .getDeviceCaps = fpgaGetDeviceCaps,
    .allocate = fpgaAllocate,      // DDR/HBM allocation
    .free = fpgaFree,
    .copyToDevice = fpgaCopyToDevice,
    .copyFromDevice = fpgaCopyFromDevice,
    .compileKernel = fpgaLoadBitstream,  // Load pre-compiled bitstream
    .launchKernel = fpgaLaunchKernel,
    .destroyKernel = fpgaDestroyKernel,
    .synchronize = fpgaSynchronize,
};
```

### 4.4 Target FPGA Platforms

| Platform | LUTs | DSPs | BRAM | HBM | Use Case |
|----------|------|------|------|-----|----------|
| AMD Alveo U250 | 1.7M | 12,288 | 54 MB | 64 GB DDR4 | Data center inference |
| AMD Alveo U55C | 1.3M | 9,024 | 36 MB | 16 GB HBM2 | High-bandwidth workloads |
| Intel Agilex 7 | 2.5M | 11,520 | 100+ MB | HBM2e | Enterprise AI |
| AMD Versal AI Core | 400K | 1,968 | 35 MB | - | Edge AI with AI Engines |

---

## 5. ASIC Acceleration Opportunities

### 5.1 When ASICs Make Sense

**Criteria for ASIC Investment:**

1. **Volume:** >100,000 units/year amortizes NRE costs
2. **Stability:** Workload patterns stable for 3-5 years
3. **Power Critical:** Edge/mobile deployment constraints
4. **Latency Critical:** Sub-microsecond response requirements

### 5.2 ASIC Design Options

#### Option A: Custom ASIC (Full Custom)

**Pros:**
- Maximum performance and efficiency
- Complete control over architecture
- Optimal for specific workloads

**Cons:**
- $10-50M NRE costs
- 18-24 month development cycle
- No post-silicon flexibility

**Partners:** Broadcom, Marvell (designed Google TPU, Meta MTIA)

#### Option B: Structured ASIC / eFPGA

**Pros:**
- Reduced NRE ($1-5M)
- Faster time to market (6-12 months)
- Some reconfigurability retained

**Cons:**
- Lower density than full custom
- Limited by base architecture

**Vendors:** Achronix, Flex Logix (eFPGA IP)

#### Option C: AI Accelerator IP Integration

**Pros:**
- Proven, validated designs
- Lowest risk path
- Can integrate into SoC

**IP Options:**
- Arm Ethos NPU series
- Cadence Tensilica DNA
- Synopsys ARC NPU
- CEVA NeuPro

### 5.3 Proposed ASIC Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ABI Vector Accelerator                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐    │
│  │            Quantized Matrix Engine (QME)            │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │    │
│  │  │ Q4 MACs │ │ Q4 MACs │ │ Q4 MACs │ │ Q4 MACs │   │    │
│  │  │  256×   │ │  256×   │ │  256×   │ │  256×   │   │    │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │    │
│  │              1024 INT4 MACs = 2 TOPS @ 1GHz         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           Vector Distance Unit (VDU)                │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │    │
│  │  │ 64× FP32    │  │ Reduction   │  │  Compare/  │  │    │
│  │  │ Dot Product │→ │   Tree      │→ │  TopK      │  │    │
│  │  └─────────────┘  └─────────────┘  └────────────┘  │    │
│  │              768-dim vector in single cycle          │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │  SRAM Buffer  │  │   DMA Engine  │  │  Control Unit │   │
│  │    4 MB       │  │   PCIe Gen5   │  │   RISC-V      │   │
│  └───────────────┘  └───────────────┘  └───────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Performance Targets:**

| Metric | Target | Comparison |
|--------|--------|------------|
| Quantized MatMul | 50 TOPS (INT4) | 10× vs A100 per watt |
| Vector Distance | 1M vectors/sec | 100× vs CPU |
| Power | <15W | Edge deployable |
| Latency | <100μs | Real-time inference |

---

## 6. Vendor Options

### 6.1 FPGA Vendors

#### AMD/Xilinx (Recommended for ABI)

| Product Line | Target Market | Key Devices | ABI Fit |
|--------------|---------------|-------------|---------|
| **Alveo** | Data center | U250, U280, U55C | Excellent - primary target |
| **Versal** | AI/Edge | AI Core, AI Edge | Good - future edge deployment |
| **Kintex** | Mid-range | UltraScale+ | Good - cost-sensitive deployments |
| **Artix** | Cost-optimized | UltraScale+ | Limited - basic applications |

**Strengths:**
- Industry-leading HLS tools (Vitis)
- Extensive AI ecosystem (Vitis AI)
- Strong community and documentation
- Best-in-class XRT runtime for Zig integration

**Development Tools:**
- Vitis HLS 2024.2+ (C/C++ to RTL)
- Vitis AI (model quantization, deployment)
- Vivado (RTL design, implementation)
- XRT (Xilinx Runtime) - primary integration point

#### Intel

| Product Line | Target Market | Key Devices | ABI Fit |
|--------------|---------------|-------------|---------|
| **Agilex 7** | High-performance | F-Series, I-Series | Good - HBM2e option |
| **Stratix 10** | Enterprise | GX, MX | Moderate - DDR4 only |
| **Agilex 5** | Mid-range | E-Series | Good - cost-effective |

**Strengths:**
- oneAPI unified programming model
- Strong enterprise relationships
- HBM2e support in Agilex 7

**Considerations:**
- FPGA division transition (recently acquired by partners)
- Smaller AI ecosystem vs AMD
- oneAPI integration more complex than XRT

#### Lattice (Edge Focus)

| Product Line | Target Market | Key Devices | ABI Fit |
|--------------|---------------|-------------|---------|
| **Nexus** | Low-power AI | CrossLink-NX | Limited - very small |
| **CertusPro** | General | NX | Limited |

**Best for:** Ultra-low-power edge deployments (<1W)

### 6.2 ASIC Partners

| Partner | Specialization | NRE Range | Notable Projects |
|---------|----------------|-----------|------------------|
| **Broadcom** | Custom silicon | $15-50M | Google TPU |
| **Marvell** | AI accelerators | $10-30M | Amazon Graviton |
| **Synopsys** | DesignWare IP | $2-10M (IP) | AI subsystems |
| **Cadence** | Tensilica DSP | $5-15M | Edge AI |
| **Flex Logix** | eFPGA IP | $1-5M | Embedded reconfigurability |

### 6.3 Vendor Selection Criteria

| Criteria | Weight | AMD/Xilinx | Intel | Lattice |
|----------|--------|------------|-------|---------|
| Tool maturity | 25% | 9/10 | 7/10 | 6/10 |
| AI ecosystem | 20% | 9/10 | 6/10 | 4/10 |
| Price/performance | 20% | 8/10 | 7/10 | 7/10 |
| Zig integration ease | 15% | 8/10 | 5/10 | 6/10 |
| Long-term roadmap | 10% | 9/10 | 7/10 | 7/10 |
| Community support | 10% | 8/10 | 6/10 | 5/10 |
| **Weighted Score** | 100% | **8.5** | **6.4** | **5.6** |

**Recommendation:** AMD/Xilinx Alveo U250 for initial development, with U55C (HBM) for high-bandwidth workloads.

---

## 7. Development Tools & Frameworks

### 7.1 FPGA Development

#### AMD Vitis HLS

```bash
# Install Vitis 2024.2+
# Write C/C++ with HLS pragmas
vitis_hls -f run_hls.tcl

# Key pragmas for optimization
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS ARRAY_PARTITION variable=weights cyclic factor=16
#pragma HLS INTERFACE m_axi port=input offset=slave
```

**Integration with Zig:**

```zig
// Load pre-compiled bitstream
const xclbin = @embedFile("kernels/matmul_q4.xclbin");
const fpga = try FpgaBackend.init(allocator, xclbin);
defer fpga.deinit();

// Launch kernel
try fpga.launchKernel("matmul_q4", .{
    .grid = .{ M / 64, N / 64, 1 },
    .block = .{ 64, 64, 1 },
}, &[_]*anyopaque{ a_buf, b_buf, c_buf });
```

#### hls4ml (Open Source)

```python
# Convert trained model to HLS
import hls4ml

config = hls4ml.utils.config_from_keras_model(model, granularity='name')
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir='hls_output',
    backend='VitisHLS'
)
hls_model.compile()
hls_model.build(csim=True, synth=True)
```

#### Intel oneAPI (Note: FPGA support transitioning)

```cpp
// SYCL kernel for Intel FPGAs
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

queue q(selector_v<ext::intel::fpga_emulator>);
q.submit([&](handler& h) {
    h.single_task<class VectorDot>([=]() {
        [[intel::fpga_register]] float acc = 0;
        #pragma unroll 16
        for (int i = 0; i < 768; i++) {
            acc += a[i] * b[i];
        }
        result[0] = acc;
    });
});
```

### 7.2 ASIC Development

| Stage | Tool | Vendor |
|-------|------|--------|
| RTL Design | SystemVerilog | - |
| Synthesis | Design Compiler | Synopsys |
| Place & Route | Innovus | Cadence |
| Verification | VCS / Xcelium | Synopsys / Cadence |
| DFT | TetraMAX | Synopsys |
| Signoff | PrimeTime | Synopsys |

**Open Source Alternative (for prototyping):**

```bash
# OpenLane flow for ASIC
git clone https://github.com/The-OpenROAD-Project/OpenLane
cd OpenLane
make
./flow.tcl -design abi_vector_unit -tag run1
```

---

## 8. Implementation Roadmap

### Phase 1: Foundation (Months 1-3)

**Goals:**
- [ ] Define FPGA backend interface in `src/gpu/backends/fpga/`
- [ ] Implement basic bitstream loading and memory management
- [ ] Create HLS template for vector distance computation
- [ ] Validate on AMD Alveo U250 development board

**Deliverables:**
- FPGA backend skeleton with VTable implementation
- Single-kernel proof of concept (cosine similarity)
- Benchmark comparison vs CPU SIMD baseline

### Phase 2: Core Kernels (Months 4-8)

**Goals:**
- [ ] Implement quantized matrix multiplication (Q4, Q8)
- [ ] Implement HNSW distance computation with prefetching
- [ ] Implement attention softmax kernel
- [ ] Integrate with existing `GpuAccelerator` dispatch

**Deliverables:**
- Production-ready FPGA kernels for inference workloads
- Automated benchmark suite
- Documentation and usage examples

### Phase 3: Optimization (Months 9-12)

**Goals:**
- [ ] Profile and optimize memory bandwidth utilization
- [ ] Implement kernel fusion (dequant + matmul + activation)
- [ ] Add multi-FPGA support for larger models
- [ ] Evaluate Intel Agilex / AMD Versal alternatives

**Deliverables:**
- Optimized production deployment package
- Multi-device scaling implementation
- Performance tuning guide

### Phase 4: ASIC Evaluation (Months 12-18)

**Goals:**
- [ ] Develop RTL prototype of Vector Distance Unit
- [ ] Synthesize and validate on FPGA (ASIC emulation)
- [ ] Cost-benefit analysis for ASIC tape-out
- [ ] Partner evaluation (Broadcom, Marvell, Flex Logix)

**Deliverables:**
- ASIC architecture specification
- Validated RTL design
- Business case and ROI analysis

---

## 9. Cost-Benefit Analysis

### 9.1 FPGA Cost Model

**Capital Costs (Initial Investment):**

| Component | Cost | Notes |
|-----------|------|-------|
| Development board (U250) | $6,500 | One-time |
| Vitis/Vivado license | $3,000/year | Required for development |
| Engineering (6 months) | $150,000 | 2 FTEs at $150K/year |
| Test infrastructure | $10,000 | Servers, equipment |
| **Total Year 1** | **~$170,000** | |

**Operating Costs (Per Node/Year):**

| Component | GPU (A100) | FPGA (U250) | Savings |
|-----------|------------|-------------|---------|
| Hardware depreciation | $3,750 | $1,625 | 57% |
| Power (8760 hrs @ $0.10/kWh) | $219 | $66 | 70% |
| Cooling | $44 | $13 | 70% |
| Maintenance | $500 | $300 | 40% |
| **Total/Year** | **$4,513** | **$2,004** | **56%** |

### 9.2 Break-Even Analysis

**Scenario: 10 inference nodes**

```
GPU Path:
  10 nodes × $15,000/A100 = $150,000 hardware
  10 nodes × $4,513/year = $45,130/year operating

FPGA Path:
  10 nodes × $6,500/U250 = $65,000 hardware
  Development: $170,000 (one-time)
  10 nodes × $2,004/year = $20,040/year operating

Year 1: GPU = $195,130, FPGA = $255,040 (FPGA worse)
Year 2: GPU = $240,260, FPGA = $275,080 (FPGA worse)
Year 3: GPU = $285,390, FPGA = $295,120 (approaching parity)
Year 4: GPU = $330,520, FPGA = $315,160 (FPGA wins)
Year 5: GPU = $375,650, FPGA = $335,200 (FPGA saves $40K)
```

**Break-even point: ~3.5 years at 10 nodes**

### 9.3 ROI by Deployment Scale

| Scale | Break-Even | 5-Year ROI | Recommendation |
|-------|------------|------------|----------------|
| 1-5 nodes | Never | Negative | Use GPU |
| 6-10 nodes | 3-4 years | 10-20% | Consider FPGA |
| 11-50 nodes | 2-3 years | 30-50% | FPGA recommended |
| 50+ nodes | 1-2 years | 50-100% | FPGA strongly recommended |

### 9.4 ASIC Cost Model

**Development Investment:**

| Item | Cost | Timeline |
|------|------|----------|
| Architecture & specification | $1M | 4 months |
| RTL development | $4M | 8 months |
| Verification | $3M | 6 months |
| Physical design | $2M | 4 months |
| Mask set (7nm) | $4M | - |
| Packaging development | $500K | 2 months |
| Silicon validation | $1M | 3 months |
| **Total NRE** | **~$15.5M** | **18-24 months** |

**Unit Economics:**

| Volume/Year | Unit Cost | Amortized NRE | Total Unit |
|-------------|-----------|---------------|------------|
| 1,000 | $150 | $5,167 | $5,317 |
| 10,000 | $80 | $517 | $597 |
| 100,000 | $50 | $52 | $102 |
| 1,000,000 | $30 | $5 | $35 |

**ASIC only makes sense at >10K units/year over 3+ years.**

---

## 10. Performance Projections

### 10.1 FPGA Performance Model

**Assumptions:**
- Platform: AMD Alveo U250 (12,288 DSPs, 64 GB DDR4)
- Clock: 300 MHz (typical for compute-bound kernels)
- Efficiency: 70% DSP utilization

**Projected Throughput:**

| Workload | CPU Baseline | FPGA Projected | Speedup |
|----------|-------------|----------------|---------|
| HNSW Search (1M vectors) | 15 ms | 0.8 ms | 18.75× |
| Q4 MatMul (4096×4096) | 12 ms | 0.6 ms | 20× |
| Softmax (2048×2048) | 2.1 ms | 0.3 ms | 7× |
| K-Means Iteration (10k×768) | 85 ms | 2.1 ms | 40× |
| LLM Token (7B params) | 180 ms | 15 ms | 12× |

### 10.2 Power Efficiency

| Platform | LLM Inference (tokens/sec/W) | Vector Search (queries/sec/W) |
|----------|------------------------------|-------------------------------|
| CPU (Xeon 8380) | 0.5 | 200 |
| GPU (A100 80GB) | 8 | 5,000 |
| FPGA (U250) | 12 | 15,000 |
| ASIC (projected) | 50 | 50,000 |

### 10.3 Cost Analysis

**FPGA Deployment (per node):**

| Item | Cost |
|------|------|
| AMD Alveo U250 | $6,500 |
| Host server | $8,000 |
| Development tools (Vitis) | $3,000/year |
| Engineering (6 months) | $150,000 |
| **Total Year 1** | **$167,500** |

**Break-even vs GPU:**
- At 10 queries/sec sustained, FPGA matches GPU cost in ~8 months
- Power savings: $2,000/year per node at $0.10/kWh

---

## 11. Timeline Considerations

### 11.1 FPGA Development Timeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FPGA Development Timeline (12 months)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Month 1-2: Foundation                                                       │
│  ├── Environment setup (Vitis, XRT)                                         │
│  ├── Backend interface design                                                │
│  ├── Memory management implementation                                        │
│  └── First "hello world" kernel                                             │
│                                                                              │
│  Month 3-4: Core Kernel Development                                          │
│  ├── Vector distance HLS kernel                                             │
│  ├── Initial optimization (pipelining, unrolling)                           │
│  └── CPU vs FPGA benchmark validation                                       │
│                                                                              │
│  Month 5-6: Quantized Operations                                             │
│  ├── Q4_0 dequantization pipeline                                           │
│  ├── Quantized matrix multiplication                                         │
│  └── Performance tuning                                                      │
│                                                                              │
│  Month 7-8: Integration                                                      │
│  ├── Integration with ABI GPU interface                                     │
│  ├── Automatic dispatch (GPU vs FPGA)                                       │
│  └── Error handling and fallback                                            │
│                                                                              │
│  Month 9-10: LLM Operations                                                  │
│  ├── Attention softmax kernel                                               │
│  ├── KV-cache management                                                    │
│  └── End-to-end inference test                                              │
│                                                                              │
│  Month 11-12: Production Hardening                                           │
│  ├── Multi-device support                                                   │
│  ├── Documentation and examples                                             │
│  └── Performance benchmarks and tuning guide                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 11.2 ASIC Development Timeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ASIC Development Timeline (24 months)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Month 1-4: Specification                                                    │
│  ├── Architecture definition                                                │
│  ├── IP selection and licensing                                             │
│  ├── Power/area budgeting                                                   │
│  └── Design partner selection                                               │
│                                                                              │
│  Month 5-12: RTL Development                                                 │
│  ├── RTL implementation                                                     │
│  ├── Unit-level verification                                                │
│  ├── FPGA emulation                                                         │
│  └── System-level verification                                              │
│                                                                              │
│  Month 13-16: Physical Design                                                │
│  ├── Synthesis                                                              │
│  ├── Floor planning                                                         │
│  ├── Place and route                                                        │
│  └── Timing closure                                                         │
│                                                                              │
│  Month 17-18: Signoff                                                        │
│  ├── DRC/LVS                                                                │
│  ├── Timing signoff                                                         │
│  ├── Power analysis                                                         │
│  └── Tape-out                                                               │
│                                                                              │
│  Month 19-24: Silicon                                                        │
│  ├── Fabrication (2-3 months)                                               │
│  ├── Packaging (1 month)                                                    │
│  └── Silicon validation (2-3 months)                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 11.3 Critical Path Items

| Phase | Critical Dependencies | Risk Mitigation |
|-------|----------------------|-----------------|
| FPGA Foundation | XRT/Vitis compatibility with Zig | Early integration testing |
| HLS Kernels | Algorithm stability | Freeze algorithm before HLS |
| Integration | GPU interface compatibility | Parallel development |
| ASIC Spec | Volume projections | Conservative estimates |
| RTL Development | IP availability | Early IP engagement |
| Physical Design | Timing closure | 10% margin in spec |
| Silicon | Yield | Multi-die strategy |

### 11.4 Decision Gates

| Gate | Timing | Criteria | Go/No-Go Decision |
|------|--------|----------|-------------------|
| G1 | Month 3 | First kernel validated | Continue FPGA development |
| G2 | Month 6 | 5x speedup achieved | Expand kernel coverage |
| G3 | Month 12 | Production-ready backend | Deploy to customers |
| G4 | Month 18 | 10K+ unit demand validated | Initiate ASIC evaluation |
| G5 | Month 24 | ASIC business case approved | Begin ASIC development |

### 11.5 Resource Requirements

**FPGA Development Team:**

| Role | Headcount | Duration | Total Person-Months |
|------|-----------|----------|---------------------|
| FPGA/HLS Engineer | 2 | 12 months | 24 |
| Backend Integration | 1 | 6 months | 6 |
| Test/Validation | 1 | 6 months | 6 |
| **Total** | **4** | | **36 person-months** |

**ASIC Development Team (if pursued):**

| Role | Headcount | Duration | Total Person-Months |
|------|-----------|----------|---------------------|
| Architect | 2 | 24 months | 48 |
| RTL Designer | 6 | 18 months | 108 |
| Verification | 4 | 18 months | 72 |
| Physical Design | 3 | 12 months | 36 |
| Firmware/SW | 2 | 12 months | 24 |
| **Total** | **17** | | **288 person-months** |

---

## 12. Risk Analysis

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| HLS optimization ceiling | Medium | High | Profile early, consider manual RTL for critical paths |
| Memory bandwidth bottleneck | Medium | Medium | Use HBM platforms (U55C), optimize access patterns |
| Kernel fusion complexity | Low | Medium | Start with simple kernels, add fusion incrementally |
| Tool compatibility issues | Medium | Low | Maintain multiple backend support (Vitis, oneAPI) |

### Business Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| FPGA supply constraints | Low | High | Qualify multiple vendors/platforms |
| Rapid GPU improvements | High | Medium | Focus on power efficiency and latency (GPU weak points) |
| ASIC NRE cost overrun | Medium | High | Use FPGA validation extensively before tape-out |
| Talent availability | Medium | Medium | Partner with FPGA consultancies, use hls4ml |

---

## 13. References & Resources

### Industry Research

- [FPGA in AI: Accelerating Deep Learning Inference](https://fidus.com/blog/the-role-of-fpgas-in-ai-acceleration/) - Fidus Systems
- [FPGA-based Deep Learning Inference Accelerators](https://dl.acm.org/doi/full/10.1145/3613963) - ACM TRETS Survey
- [Global AI Hardware Landscape 2025](https://www.geniatech.com/ai-hardware-2025/) - Geniatech
- [AI and Deep Learning Accelerators Beyond GPUs](https://www.bestgpusforai.com/blog/ai-accelerators) - 2025 Overview
- [Beyond the GPU: Strategic Role of FPGAs in AI](https://arxiv.org/html/2511.11614v1) - arXiv 2024

### Vector Search Acceleration

- [SmartANNS: FPGA-based HNSW on CSDs](https://www.usenix.org/system/files/atc24-tian.pdf) - USENIX ATC 2024
- [Falcon: Fast Graph Vector Search](https://arxiv.org/html/2406.12385) - Hardware Acceleration
- [Efficient Vector Search on Disaggregated Memory](https://arxiv.org/html/2505.11783v1) - d-HNSW

### Development Tools

- [AMD Vitis HLS User Guide](https://docs.amd.com/r/en-US/ug1399-vitis-hls) - AMD Documentation
- [hls4ml: ML on FPGAs using HLS](https://github.com/fastmachinelearning/hls4ml) - GitHub Repository
- [hls4ml Paper](https://arxiv.org/html/2512.01463) - Flexible Deep Learning on FPGAs

### ASIC Landscape

- [TPUs vs GPUs vs ASICs: AI Hardware Guide 2025](https://howaiworks.ai/blog/tpu-gpu-asic-ai-hardware-market-2025) - HowAIWorks
- [CPU vs GPU vs TPU vs NPU Architecture Guide](https://www.thepurplestruct.com/blog/cpu-vs-gpu-vs-tpu-vs-npu-ai-hardware-architecture-guide-2025) - 2025 Comparison
- [Custom AI Chip Development](https://www.cnbc.com/2025/11/21/nvidia-gpus-google-tpus-aws-trainium-comparing-the-top-ai-chips.html) - CNBC 2025

---

## Appendix A: ABI Codebase Integration Points

### GPU Backend Interface

**File:** `src/gpu/interface.zig`

```zig
pub const Backend = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        deinit: *const fn (*anyopaque) void,
        getDeviceCount: *const fn (*anyopaque) u32,
        getDeviceCaps: *const fn (*anyopaque, u32) BackendError!DeviceCaps,
        allocate: *const fn (*anyopaque, usize, MemoryFlags) MemoryError!*anyopaque,
        free: *const fn (*anyopaque, *anyopaque) void,
        copyToDevice: *const fn (*anyopaque, *anyopaque, []const u8) MemoryError!void,
        copyFromDevice: *const fn (*anyopaque, []u8, *anyopaque) MemoryError!void,
        compileKernel: *const fn (*anyopaque, Allocator, []const u8, []const u8) KernelError!*anyopaque,
        launchKernel: *const fn (*anyopaque, *anyopaque, LaunchConfig, []const *anyopaque) KernelError!void,
        destroyKernel: *const fn (*anyopaque, *anyopaque) void,
        synchronize: *const fn (*anyopaque) BackendError!void,
    };
};
```

### Database GPU Acceleration

**File:** `src/database/gpu_accel.zig`

```zig
pub const GpuAccelerator = struct {
    gpu_ctx: if (build_options.enable_gpu) ?*gpu.Gpu else void,
    dispatcher: if (build_options.enable_gpu) ?*gpu.KernelDispatcher else void,
    batch_threshold: usize = 1024,  // GPU only for batch >= 1024
};
```

### Runtime Workload Hints

**File:** `src/runtime/workload.zig`

```zig
pub const WorkloadHints = struct {
    cpu_affinity: ?u32 = null,
    estimated_duration_us: ?u64 = null,
    prefers_gpu: bool = false,
    requires_gpu: bool = false,
};
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **ANNS** | Approximate Nearest Neighbor Search |
| **DSP** | Digital Signal Processor (FPGA multiply-accumulate unit) |
| **HBM** | High Bandwidth Memory |
| **HLS** | High-Level Synthesis (C/C++ to hardware) |
| **HNSW** | Hierarchical Navigable Small World (graph index) |
| **IVF-PQ** | Inverted File with Product Quantization |
| **LUT** | Look-Up Table (FPGA basic logic element) |
| **NRE** | Non-Recurring Engineering (one-time development cost) |
| **PE** | Processing Element |
| **QME** | Quantized Matrix Engine |
| **RTL** | Register Transfer Level (hardware description) |
| **VDU** | Vector Distance Unit |
| **VTable** | Virtual function table (polymorphism pattern) |

---

*Document prepared for ABI Framework - Hardware Acceleration Research Initiative*

