---
title: "hybrid-gpu-fpga-architecture"
tags: ["research", "hybrid", "gpu", "fpga", "architecture"]
---
# Hybrid GPU-FPGA Architecture for ABI

> **Document Version:** 1.0
> **Date:** January 2026
> **Status:** Research Phase
> **Related:** [hardware-acceleration-fpga-asic.md](./hardware-acceleration-fpga-asic.md)

## Executive Summary

This document explores hybrid architectures that combine GPU and FPGA accelerators to leverage the strengths of each platform. GPUs excel at high-throughput parallel computation with flexible programming models, while FPGAs offer deterministic latency, custom datapaths, and superior power efficiency for fixed workloads.

### Hybrid Value Proposition

| Workload | Best Platform | Reason |
|----------|---------------|--------|
| LLM prefill (batch) | GPU | High parallelism, memory bandwidth |
| LLM decode (single) | FPGA | Low latency, power efficiency |
| Vector bulk indexing | GPU | Throughput-oriented |
| Vector real-time search | FPGA | Latency-critical |
| Training | GPU | Flexibility, ecosystem |
| Inference edge | FPGA | Power constraints |

---

## Table of Contents

1. [Architecture Patterns](#1-architecture-patterns)
2. [Workload Partitioning](#2-workload-partitioning)
3. [Communication and Synchronization](#3-communication-and-synchronization)
4. [Implementation in ABI](#4-implementation-in-abi)
5. [Use Case Analysis](#5-use-case-analysis)
6. [Performance Modeling](#6-performance-modeling)
7. [Deployment Topologies](#7-deployment-topologies)
8. [Implementation Roadmap](#8-implementation-roadmap)
9. [References](#9-references)

---

## 1. Architecture Patterns

### 1.1 Offload Pattern

GPU handles most work; FPGA accelerates specific bottlenecks.

```
┌─────────────────────────────────────────────────────────────────┐
│                      Offload Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Host Application                                                │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────┐                    │
│  │              GPU (Primary)              │                    │
│  │                                          │                    │
│  │  ┌────────┐  ┌────────┐  ┌────────┐   │                    │
│  │  │ Prefill│  │ Batch  │  │  KV    │   │                    │
│  │  │ Phase  │  │ Decode │  │ Cache  │   │                    │
│  │  └────────┘  └────────┘  └────────┘   │                    │
│  │                                          │                    │
│  │         │ Offload bottleneck            │                    │
│  │         ▼                                │                    │
│  │  ┌─────────────────────────────────┐   │                    │
│  │  │     FPGA (Accelerator)          │   │                    │
│  │  │                                  │   │                    │
│  │  │  ┌────────────┐  ┌───────────┐ │   │                    │
│  │  │  │ Quantized  │  │ Attention │ │   │                    │
│  │  │  │ MatMul     │  │ Softmax   │ │   │                    │
│  │  │  └────────────┘  └───────────┘ │   │                    │
│  │  └─────────────────────────────────┘   │                    │
│  └─────────────────────────────────────────┘                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Use when:
- Single operation dominates latency
- FPGA IP already developed
- Minimal data transfer overhead
```

### 1.2 Pipeline Pattern

GPU and FPGA process different stages concurrently.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Pipeline Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Request Stream                                                  │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────┐                                               │
│  │   Stage 1    │  GPU: Embedding lookup, initial layers        │
│  │   (GPU)      │                                               │
│  └──────┬───────┘                                               │
│         │ intermediate tensor                                   │
│         ▼                                                          │
│  ┌──────────────┐                                               │
│  │   Stage 2    │  FPGA: Attention mechanism                    │
│  │   (FPGA)     │                                               │
│  └──────┬───────┘                                               │
│         │ attention output                                      │
│         ▼                                                          │
│  ┌──────────────┐                                               │
│  │   Stage 3    │  GPU: FFN layers, final layers                │
│  │   (GPU)      │                                               │
│  └──────┬───────┘                                               │
│         │                                                        │
│         ▼                                                          │
│  ┌──────────────┐                                               │
│  │   Stage 4    │  FPGA: Sampling, post-processing              │
│  │   (FPGA)     │                                               │
│  └──────┬───────┘                                               │
│         │                                                        │
│         ▼                                                          │
│  Response Stream                                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Use when:
- Multiple requests need processing
- Stages have different characteristics
- Overlap hides transfer latency
```

### 1.3 Parallel Pattern

GPU and FPGA handle different workload types simultaneously.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Parallel Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Unified Scheduler                     │   │
│  │                                                          │   │
│  │  Classify workload → Route to optimal accelerator        │   │
│  └───────────────────────┬─────────────────────────────────┘   │
│                          │                                      │
│          ┌───────────────┴───────────────┐                     │
│          │                               │                      │
│          ▼                               ▼                      │
│  ┌───────────────────┐     ┌───────────────────┐              │
│  │       GPU         │     │       FPGA        │              │
│  │                   │     │                   │              │
│  │  Workloads:       │     │  Workloads:       │              │
│  │  - Batch prefill  │     │  - Single decode  │              │
│  │  - Training       │     │  - Vector search  │              │
│  │  - Large batch    │     │  - Low-latency    │              │
│  │    inference      │     │    requests       │              │
│  │                   │     │                   │              │
│  └───────────────────┘     └───────────────────┘              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Use when:
- Mixed workload types
- Different latency requirements
- Resource utilization optimization
```

### 1.4 Redundancy Pattern

Both platforms available for failover and load balancing.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Redundancy Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Load Balancer                          │   │
│  │                                                          │   │
│  │  Health check → Route based on:                          │   │
│  │  - Device health                                         │   │
│  │  - Current load                                          │   │
│  │  - Power budget                                          │   │
│  │  - Latency SLA                                           │   │
│  └───────────────────────┬─────────────────────────────────┘   │
│                          │                                      │
│          ┌───────────────┴───────────────┐                     │
│          │                               │                      │
│          ▼                               ▼                      │
│  ┌───────────────────┐     ┌───────────────────┐              │
│  │   GPU Instance    │     │  FPGA Instance    │              │
│  │                   │     │                   │              │
│  │  Same workload    │     │  Same workload    │              │
│  │  capability       │     │  capability       │              │
│  │                   │     │                   │              │
│  │  Fallback: FPGA   │     │  Fallback: GPU    │              │
│  └───────────────────┘     └───────────────────┘              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Use when:
- High availability required
- Gradual FPGA adoption
- A/B testing hardware
```

---

## 2. Workload Partitioning

### 2.1 LLM Inference Partitioning

**Analysis of LLM operations:**

| Operation | GPU Advantage | FPGA Advantage | Recommendation |
|-----------|--------------|----------------|----------------|
| Embedding lookup | High bandwidth | N/A | GPU |
| QKV projection | Batching | Quantized | Hybrid |
| Attention (prefill) | Parallelism | N/A | GPU |
| Attention (decode) | N/A | Low latency | FPGA |
| FFN (prefill) | Batching | N/A | GPU |
| FFN (decode) | N/A | Streaming | FPGA |
| Softmax | N/A | Streaming | FPGA |
| RMSNorm | Trivial | Trivial | Either |
| Sampling | N/A | Deterministic | FPGA |

**Optimal Partitioning Strategy:**

```
Prefill Phase (batch processing):
┌─────────────────────────────────────────┐
│                  GPU                     │
│  Embedding → Attention → FFN → Output   │
└─────────────────────────────────────────┘

Decode Phase (autoregressive):
┌────────────────────────────────────────────────────────────────┐
│  GPU (QKV projection)                                          │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  FPGA (Attention + FFN + Sampling)                        │ │
│  │                                                           │ │
│  │  Streaming execution with on-chip KV-cache                │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 Vector Database Partitioning

**Analysis of vector operations:**

| Operation | GPU Advantage | FPGA Advantage | Recommendation |
|-----------|--------------|----------------|----------------|
| Bulk insert | Batching | N/A | GPU |
| Index building | Parallelism | N/A | GPU |
| Batch search | Throughput | N/A | GPU |
| Single search | N/A | Latency | FPGA |
| Reranking | N/A | Streaming | FPGA |
| K-means | Batching | N/A | GPU |
| HNSW graph update | N/A | Random access | FPGA |

**Hybrid Search Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                  Hybrid Vector Search                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Query arrives:                                                  │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────────────────────────────┐                       │
│  │         Query Classification         │                       │
│  │                                       │                       │
│  │  Batch > 10 queries → GPU path       │                       │
│  │  Single query, SLA < 1ms → FPGA path │                       │
│  └───────────────────┬──────────────────┘                       │
│                      │                                           │
│      ┌───────────────┴───────────────┐                          │
│      │                               │                           │
│      ▼                               ▼                           │
│  ┌────────────────┐      ┌────────────────┐                     │
│  │  GPU Path      │      │  FPGA Path     │                     │
│  │                │      │                │                     │
│  │  1. Batch      │      │  1. HNSW       │                     │
│  │     distance   │      │     traverse   │                     │
│  │  2. Parallel   │      │  2. Stream     │                     │
│  │     top-k      │      │     distances  │                     │
│  │  3. Merge      │      │  3. HW top-k   │                     │
│  │                │      │                │                     │
│  │  Latency: 5ms  │      │  Latency: 0.5ms│                     │
│  │  Throughput:   │      │  Throughput:   │                     │
│  │  100K qps      │      │  10K qps       │                     │
│  └────────────────┘      └────────────────┘                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Decision Criteria

```zig
// From src/runtime/workload.zig (extended concept)
pub const HybridWorkloadHints = struct {
    // Existing fields
    cpu_affinity: ?u32 = null,
    estimated_duration_us: ?u64 = null,

    // Hybrid-specific fields
    latency_sla_us: ?u64 = null,        // Strict latency requirement
    batch_size: u32 = 1,                 // Number of items to process
    power_budget_watts: ?f32 = null,     // Power constraint
    prefers_deterministic: bool = false, // Needs predictable timing

    // Computed routing
    pub fn recommendedPlatform(self: HybridWorkloadHints) Platform {
        // Strict latency requirement → FPGA
        if (self.latency_sla_us) |sla| {
            if (sla < 1000) return .fpga;
        }

        // High batch → GPU
        if (self.batch_size > 16) return .gpu;

        // Power constrained → FPGA
        if (self.power_budget_watts) |budget| {
            if (budget < 50) return .fpga;
        }

        // Deterministic requirement → FPGA
        if (self.prefers_deterministic) return .fpga;

        // Default to GPU for flexibility
        return .gpu;
    }

    pub const Platform = enum { cpu, gpu, fpga, hybrid };
};
```

---

## 3. Communication and Synchronization

### 3.1 Data Transfer Paths

```
┌─────────────────────────────────────────────────────────────────┐
│                    System Interconnect                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                      ┌───────────────────┐                      │
│                      │    Host CPU       │                      │
│                      │    (DDR5 Memory)  │                      │
│                      └─────────┬─────────┘                      │
│                                │                                 │
│              PCIe Gen5 x16     │     PCIe Gen5 x16              │
│              (64 GB/s)         │     (64 GB/s)                  │
│          ┌─────────────────────┼─────────────────────┐          │
│          │                     │                     │          │
│          ▼                     │                     ▼          │
│  ┌───────────────────┐         │         ┌───────────────────┐  │
│  │       GPU         │         │         │       FPGA        │  │
│  │  (HBM: 2 TB/s)    │         │         │  (HBM: 460 GB/s)  │  │
│  └─────────┬─────────┘         │         └─────────┬─────────┘  │
│            │                   │                   │            │
│            │     NVLink/CXL    │    PCIe P2P       │            │
│            │     (900 GB/s)    │    (32 GB/s)      │            │
│            └───────────────────┴───────────────────┘            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Transfer Optimization Strategies

**1. Zero-Copy Shared Memory (CXL)**

```
CXL Type 3 Memory:
┌─────────────────────────────────────────────────────────────┐
│                    CXL Shared Pool                           │
│                    (cache-coherent)                          │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   CPU View   │  │   GPU View   │  │  FPGA View   │      │
│  │              │  │              │  │              │      │
│  │  Virtual     │  │  BAR mapped  │  │  AXI mapped  │      │
│  │  address     │  │  to HBM      │  │  to DDR      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  Latency: ~200ns (vs ~2μs for PCIe DMA)                     │
│  Bandwidth: 64 GB/s bidirectional                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**2. Pinned Buffer Pool**

```zig
// Hybrid memory pool for GPU-FPGA communication
pub const HybridBufferPool = struct {
    gpu_allocator: GpuAllocator,
    fpga_allocator: FpgaAllocator,

    // Pre-allocated pinned buffers for fast transfer
    pinned_pool: []PinnedBuffer,

    pub fn allocateShared(self: *HybridBufferPool, size: usize) !SharedBuffer {
        // Allocate in host pinned memory
        const host_ptr = try self.allocatePinned(size);

        // Map to GPU address space
        const gpu_ptr = try self.gpu_allocator.mapHostMemory(host_ptr);

        // Map to FPGA address space
        const fpga_ptr = try self.fpga_allocator.mapHostMemory(host_ptr);

        return SharedBuffer{
            .host_ptr = host_ptr,
            .gpu_ptr = gpu_ptr,
            .fpga_ptr = fpga_ptr,
            .size = size,
        };
    }

    pub const SharedBuffer = struct {
        host_ptr: [*]u8,
        gpu_ptr: *anyopaque,
        fpga_ptr: *anyopaque,
        size: usize,

        pub fn syncToGpu(self: *SharedBuffer) !void {
            // Flush CPU caches, GPU can read
        }

        pub fn syncToFpga(self: *SharedBuffer) !void {
            // Flush CPU caches, FPGA can read
        }

        pub fn syncFromGpu(self: *SharedBuffer) !void {
            // Invalidate CPU caches after GPU write
        }

        pub fn syncFromFpga(self: *SharedBuffer) !void {
            // Invalidate CPU caches after FPGA write
        }
    };
};
```

### 3.3 Synchronization Primitives

**Cross-Device Events:**

```zig
pub const HybridEvent = struct {
    gpu_event: ?GpuEvent = null,
    fpga_event: ?FpgaEvent = null,
    host_flag: std.atomic.Atomic(bool),

    pub fn record(self: *HybridEvent, device: Device) !void {
        switch (device) {
            .gpu => {
                self.gpu_event = try GpuEvent.record();
            },
            .fpga => {
                self.fpga_event = try FpgaEvent.record();
            },
        }
    }

    pub fn waitOn(self: *HybridEvent, device: Device) !void {
        // GPU waiting on FPGA: poll host flag
        // FPGA waiting on GPU: poll host flag
        // Both poll until completion then proceed

        while (!self.host_flag.load(.Acquire)) {
            std.Thread.yield();
        }
    }

    pub fn signal(self: *HybridEvent) void {
        self.host_flag.store(true, .Release);
    }
};
```

---

## 4. Implementation in ABI

### 4.1 Unified Accelerator Interface

```zig
// Proposed extension to src/gpu/interface.zig
pub const HybridBackend = struct {
    gpu: ?*gpu.Backend = null,
    fpga: ?*fpga.FpgaBackend = null,
    scheduler: HybridScheduler,
    transfer_manager: TransferManager,

    pub fn init(allocator: std.mem.Allocator, config: HybridConfig) !*HybridBackend {
        var self = try allocator.create(HybridBackend);

        // Initialize GPU if available
        if (gpu.isAvailable()) {
            self.gpu = try gpu.Backend.init(allocator, config.gpu_config);
        }

        // Initialize FPGA if available
        if (fpga.isAvailable()) {
            self.fpga = try fpga.FpgaBackend.init(allocator, config.fpga_config);
        }

        self.scheduler = HybridScheduler.init(self.gpu, self.fpga);
        self.transfer_manager = TransferManager.init(allocator);

        return self;
    }

    pub fn executeWorkload(
        self: *HybridBackend,
        workload: *Workload,
        hints: HybridWorkloadHints,
    ) !void {
        const platform = hints.recommendedPlatform();

        switch (platform) {
            .gpu => try self.executeOnGpu(workload),
            .fpga => try self.executeOnFpga(workload),
            .hybrid => try self.executeHybrid(workload),
            .cpu => try self.executeOnCpu(workload),
        }
    }

    fn executeHybrid(self: *HybridBackend, workload: *Workload) !void {
        // Split workload into GPU and FPGA portions
        const split = self.scheduler.partitionWorkload(workload);

        // Set up inter-device buffers
        var shared_buf = try self.transfer_manager.allocateShared(split.intermediate_size);
        defer self.transfer_manager.free(&shared_buf);

        // Execute GPU portion
        try self.executeOnGpu(split.gpu_portion);
        try shared_buf.syncToFpga();

        // Execute FPGA portion
        try self.executeOnFpga(split.fpga_portion);
        try shared_buf.syncFromFpga();
    }
};
```

### 4.2 Runtime Integration

```zig
// Extension to src/runtime/engine/engine.zig
pub const HybridEngine = struct {
    base_engine: *Engine,
    hybrid_backend: *HybridBackend,

    pub fn submitHybridTask(
        self: *HybridEngine,
        task: *Task,
        hints: HybridWorkloadHints,
    ) !TaskHandle {
        // Determine optimal execution path
        const platform = hints.recommendedPlatform();

        // Create appropriate workload wrapper
        const workload = switch (platform) {
            .gpu => try createGpuWorkload(task),
            .fpga => try createFpgaWorkload(task),
            .hybrid => try createHybridWorkload(task),
            .cpu => return self.base_engine.submit(task),
        };

        // Submit to hybrid backend
        return try self.submitWorkload(workload, hints);
    }
};
```

### 4.3 Configuration

```zig
// Extension to src/config.zig
pub const HybridConfig = struct {
    // Enable hybrid mode
    enable_hybrid: bool = false,

    // Individual backend configs
    gpu_config: ?GpuConfig = null,
    fpga_config: ?FpgaConfig = null,

    // Scheduling policy
    scheduling_policy: SchedulingPolicy = .latency_optimized,

    // Transfer optimization
    use_pinned_memory: bool = true,
    transfer_threshold_bytes: usize = 4096,

    // Workload routing thresholds
    fpga_latency_threshold_us: u64 = 1000,
    gpu_batch_threshold: u32 = 16,

    pub const SchedulingPolicy = enum {
        latency_optimized,  // Minimize latency
        throughput_optimized,  // Maximize throughput
        power_optimized,  // Minimize power
        balanced,  // Balance all factors
    };
};
```

---

## 5. Use Case Analysis

### 5.1 Real-Time RAG System

**Requirements:**
- Query latency: <50ms P99
- Index size: 10M vectors
- Query rate: 1000 qps
- LLM model: 7B parameters

**Hybrid Solution:**

```
Query Flow:
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  User Query                                                      │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────┐                                               │
│  │   FPGA       │  Vector search (5ms)                          │
│  │   Search     │  - HNSW traversal                             │
│  │              │  - Distance computation                       │
│  │              │  - Top-10 retrieval                           │
│  └──────┬───────┘                                               │
│         │ context vectors                                       │
│         ▼                                                          │
│  ┌──────────────┐                                               │
│  │   GPU        │  LLM prefill (15ms)                           │
│  │   Prefill    │  - Embed context                              │
│  │              │  - Process prompt                             │
│  └──────┬───────┘                                               │
│         │ KV-cache                                              │
│         ▼                                                          │
│  ┌──────────────┐                                               │
│  │   FPGA       │  LLM decode (25ms for 50 tokens)              │
│  │   Decode     │  - Streaming generation                       │
│  │              │  - Low-latency tokens                         │
│  └──────┬───────┘                                               │
│         │                                                        │
│         ▼                                                          │
│  Response Stream                                                 │
│                                                                  │
│  Total latency: 45ms (vs 80ms GPU-only)                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Batch Processing Pipeline

**Requirements:**
- Throughput: 1M documents/hour
- Index update: Real-time
- Power budget: 5kW rack

**Hybrid Solution:**

```
Batch Pipeline:
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Document Stream (1M docs)                                       │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────────────────────────────────────┐              │
│  │                  GPU Farm (8x A100)          │              │
│  │                                               │              │
│  │  Parallel processing:                         │              │
│  │  - Chunking and tokenization                 │              │
│  │  - Embedding generation (batch 512)          │              │
│  │  - Initial clustering                        │              │
│  │                                               │              │
│  │  Throughput: 50K embeddings/sec              │              │
│  │  Power: 2kW                                  │              │
│  └──────────────────────────────────────────────┘              │
│                                                                  │
│       │ embeddings stream                                       │
│       ▼                                                          │
│  ┌──────────────────────────────────────────────┐              │
│  │                FPGA Array (4x U250)          │              │
│  │                                               │              │
│  │  Streaming insertion:                         │              │
│  │  - HNSW graph update                         │              │
│  │  - IVF-PQ encoding                          │              │
│  │  - Index compaction                         │              │
│  │                                               │              │
│  │  Throughput: 50K inserts/sec                │              │
│  │  Power: 300W                                │              │
│  └──────────────────────────────────────────────┘              │
│                                                                  │
│  Total power: 2.3kW (vs 4kW GPU-only)                          │
│  40% power reduction with same throughput                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Edge Deployment

**Requirements:**
- Power: <100W total system
- Latency: <20ms response
- Model: 7B quantized
- Offline capability

**Hybrid Solution:**

```
Edge Device:
┌─────────────────────────────────────────────────────────────────┐
│                  Compact Edge System                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────┐  ┌───────────────────┐                  │
│  │   Mobile GPU      │  │   FPGA Module     │                  │
│  │   (RTX 4060)      │  │   (Versal AI)     │                  │
│  │                   │  │                   │                  │
│  │   35W TDP         │  │   25W TDP         │                  │
│  │                   │  │                   │                  │
│  │   Roles:          │  │   Roles:          │                  │
│  │   - Prefill       │  │   - Decode        │                  │
│  │   - Batch queries │  │   - Real-time     │                  │
│  │   - Training      │  │     queries       │                  │
│  │     (fine-tune)   │  │   - Low-power     │                  │
│  │                   │  │     standby       │                  │
│  └───────────────────┘  └───────────────────┘                  │
│                                                                  │
│  Power modes:                                                    │
│  - Standby (FPGA only): 10W                                     │
│  - Normal (FPGA + GPU idle): 35W                                │
│  - Peak (both active): 60W                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Performance Modeling

### 6.1 Latency Model

```
T_total = T_cpu_prep + T_transfer_to_accel + T_compute + T_transfer_back + T_cpu_post

For GPU-only:
  T_total_gpu = 0.1ms + 0.5ms + T_compute_gpu + 0.5ms + 0.1ms
              = 1.2ms + T_compute_gpu

For FPGA-only:
  T_total_fpga = 0.1ms + 0.3ms + T_compute_fpga + 0.3ms + 0.1ms
               = 0.8ms + T_compute_fpga

For Hybrid (pipeline):
  T_total_hybrid = T_transfer_to_gpu + max(T_gpu, T_fpga) + T_transfer_back
                 = 0.5ms + max(T_gpu, T_fpga) + 0.3ms

Hybrid wins when: T_gpu and T_fpga can overlap
```

### 6.2 Throughput Model

```
Throughput = min(GPU_throughput, FPGA_throughput, Transfer_throughput)

GPU throughput: 100K ops/sec
FPGA throughput: 50K ops/sec
Transfer throughput: 64 GB/s / op_size

For 1KB operations:
  Transfer limited: 64M ops/sec (not bottleneck)
  System throughput: ~50K ops/sec (FPGA limited)

For 1MB operations:
  Transfer limited: 64K ops/sec (becomes bottleneck)
  System throughput: ~64K ops/sec (transfer limited)
```

### 6.3 Power Model

```
P_total = P_gpu * utilization_gpu + P_fpga * utilization_fpga + P_transfer + P_host

Example (balanced workload):
  GPU: 250W * 0.5 utilization = 125W
  FPGA: 75W * 0.8 utilization = 60W
  Transfer: 10W
  Host: 50W
  Total: 245W

vs GPU-only (same throughput):
  GPU: 250W * 1.0 = 250W
  Host: 50W
  Total: 300W

Power savings: 18%
```

---

## 7. Deployment Topologies

### 7.1 Single Node

```
┌─────────────────────────────────────────────────────────────────┐
│                     Single Server Node                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CPU: Xeon 8480+ (56 cores)                                     │
│  RAM: 512 GB DDR5                                                │
│  GPU: 2x A100 80GB (PCIe Gen5)                                  │
│  FPGA: 2x Alveo U250 (PCIe Gen4)                                │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     PCIe Topology                        │   │
│  │                                                          │   │
│  │     CPU Socket 0          CPU Socket 1                  │   │
│  │         │                      │                         │   │
│  │    ┌────┴────┐            ┌────┴────┐                   │   │
│  │    │         │            │         │                    │   │
│  │  A100-0   U250-0       A100-1   U250-1                  │   │
│  │                                                          │   │
│  │  GPU-FPGA pairs on same CPU for optimal transfer         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Workload distribution:                                          │
│  - Pair 0: Production inference                                 │
│  - Pair 1: Batch processing / failover                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Multi-Node Cluster

```
┌─────────────────────────────────────────────────────────────────┐
│                     Hybrid Cluster (4 Nodes)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐                              │
│  │   Node 1    │  │   Node 2    │    GPU-Heavy Tier             │
│  │  4x A100    │  │  4x A100    │    - Prefill                  │
│  │  2x U250    │  │  2x U250    │    - Training                 │
│  └──────┬──────┘  └──────┬──────┘    - Batch inference          │
│         │                 │                                      │
│         └────────┬────────┘                                     │
│                  │ 100GbE / InfiniBand                           │
│         ┌────────┴────────┐                                     │
│         │                 │                                      │
│  ┌──────┴──────┐  ┌──────┴──────┐                              │
│  │   Node 3    │  │   Node 4    │    FPGA-Heavy Tier            │
│  │  1x A100    │  │  1x A100    │    - Decode                   │
│  │  4x U250    │  │  4x U250    │    - Vector search            │
│  └─────────────┘  └─────────────┘    - Real-time inference      │
│                                                                  │
│  Load balancer routes based on workload type                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Disaggregated Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Disaggregated Hybrid System                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    CXL Fabric Switch                     │   │
│  │                    (cache-coherent)                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │           │           │           │                     │
│       ▼           ▼           ▼           ▼                     │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐              │
│  │  CPU    │ │  GPU    │ │  FPGA   │ │ Memory  │              │
│  │  Pool   │ │  Pool   │ │  Pool   │ │  Pool   │              │
│  │         │ │         │ │         │ │         │              │
│  │ 4x Xeon │ │ 8x A100 │ │ 8x U250 │ │ 2 TB    │              │
│  │         │ │         │ │         │ │ CXL RAM │              │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘              │
│                                                                  │
│  Benefits:                                                       │
│  - Independent scaling of each resource type                    │
│  - Shared memory pool reduces transfers                         │
│  - Dynamic resource allocation                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Implementation Roadmap

### 8.1 Phase 1: Foundation (Q1-Q2 2026)

**Objectives:**
- Define hybrid interface abstractions
- Implement basic workload routing
- Create shared memory infrastructure

**Deliverables:**
- `src/gpu/hybrid/mod.zig` - Hybrid backend module
- `src/gpu/hybrid/scheduler.zig` - Workload scheduler
- `src/gpu/hybrid/transfer.zig` - Transfer manager
- Documentation and examples

### 8.2 Phase 2: Integration (Q3-Q4 2026)

**Objectives:**
- Integrate with existing GPU and FPGA backends
- Implement pipeline execution
- Add monitoring and profiling

**Deliverables:**
- Runtime engine integration
- Metrics collection for hybrid operations
- Performance benchmarks

### 8.3 Phase 3: Optimization (2027)

**Objectives:**
- Optimize transfer paths
- Implement advanced scheduling
- Production hardening

**Deliverables:**
- CXL support (when available)
- Auto-tuning workload partitioning
- Production deployment guide

---

## 9. References

### Academic Papers

- "FPGA-GPU Heterogeneous Computing: A Survey" - ACM Computing Surveys 2024
- "Efficient Data Transfer in Heterogeneous Systems" - ISCA 2023
- "CXL-based Disaggregated Computing" - ASPLOS 2024

### Industry Resources

- AMD/Xilinx Heterogeneous Computing Guide
- NVIDIA GPU Direct with FPGAs
- Intel Heterogeneous Computing Documentation

### ABI Framework References

- `src/gpu/interface.zig` - GPU backend interface
- `src/gpu/backends/fpga/` - FPGA backend
- `src/runtime/engine/` - Runtime engine
- `docs/research/hardware-acceleration-fpga-asic.md` - Main research doc

---

*Document prepared for ABI Framework - Hybrid GPU-FPGA Architecture Research*
