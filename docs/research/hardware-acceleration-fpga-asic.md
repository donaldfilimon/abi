---
title: "hardware-acceleration-fpga-asic"
tags: []
---
# Hardware Acceleration Research: FPGA & ASIC for ABI
> **Codebase Status:** Synced with repository as of 2026-01-22.

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

## Table of Contents

1. [Current Architecture Analysis](#1-current-architecture-analysis)
2. [Compute-Intensive Workloads](#2-compute-intensive-workloads)
3. [FPGA Acceleration Opportunities](#3-fpga-acceleration-opportunities)
4. [ASIC Acceleration Opportunities](#4-asic-acceleration-opportunities)
5. [Development Tools & Frameworks](#5-development-tools--frameworks)
6. [Implementation Roadmap](#6-implementation-roadmap)
7. [Performance Projections](#7-performance-projections)
8. [Risk Analysis](#8-risk-analysis)
9. [References & Resources](#9-references--resources)

---

## 1. Current Architecture Analysis

### 1.1 GPU Backend Architecture

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

### 1.2 Runtime Engine

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

### 1.3 Current SIMD Optimizations

The database module (`src/shared/simd.zig`) implements vectorized operations:

| Operation | Implementation | Vector Width |
|-----------|----------------|--------------|
| `vectorDot` | SIMD accumulation | Auto-detected (AVX-512/NEON/WASM) |
| `vectorL2Norm` | SIMD squared-sum | Auto-detected |
| `cosineSimilarity` | Fused dot + norms | Auto-detected |
| `batchCosineSimilarity` | Pre-computed query norm | Auto-detected |

---

## 2. Compute-Intensive Workloads

### 2.1 LLM Inference Operations

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

### 2.2 Vector Database Operations

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

### 2.3 Training Operations

**Backward Pass Requirements:**

| Operation | Compute | Memory |
|-----------|---------|--------|
| Attention backward | 3× forward | 2× activations |
| MatMul backward | 2× forward | Weight gradients |
| RMSNorm backward | 1× forward | Input cache |
| Loss + Softmax | O(vocab_size × batch) | Per-token |

---

## 3. FPGA Acceleration Opportunities

### 3.1 Why FPGAs for ABI?

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

### 3.2 Priority Acceleration Targets

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

### 3.3 FPGA Backend Integration

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

### 3.4 Target FPGA Platforms

| Platform | LUTs | DSPs | BRAM | HBM | Use Case |
|----------|------|------|------|-----|----------|
| AMD Alveo U250 | 1.7M | 12,288 | 54 MB | 64 GB DDR4 | Data center inference |
| AMD Alveo U55C | 1.3M | 9,024 | 36 MB | 16 GB HBM2 | High-bandwidth workloads |
| Intel Agilex 7 | 2.5M | 11,520 | 100+ MB | HBM2e | Enterprise AI |
| AMD Versal AI Core | 400K | 1,968 | 35 MB | - | Edge AI with AI Engines |

---

## 4. ASIC Acceleration Opportunities

### 4.1 When ASICs Make Sense

**Criteria for ASIC Investment:**

1. **Volume:** >100,000 units/year amortizes NRE costs
2. **Stability:** Workload patterns stable for 3-5 years
3. **Power Critical:** Edge/mobile deployment constraints
4. **Latency Critical:** Sub-microsecond response requirements

### 4.2 ASIC Design Options

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

### 4.3 Proposed ASIC Architecture

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

## 5. Development Tools & Frameworks

### 5.1 FPGA Development

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

### 5.2 ASIC Development

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

## 6. Implementation Roadmap

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

## 7. Performance Projections

### 7.1 FPGA Performance Model

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

### 7.2 Power Efficiency

| Platform | LLM Inference (tokens/sec/W) | Vector Search (queries/sec/W) |
|----------|------------------------------|-------------------------------|
| CPU (Xeon 8380) | 0.5 | 200 |
| GPU (A100 80GB) | 8 | 5,000 |
| FPGA (U250) | 12 | 15,000 |
| ASIC (projected) | 50 | 50,000 |

### 7.3 Cost Analysis

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

## 8. Risk Analysis

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

## 9. References & Resources

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

