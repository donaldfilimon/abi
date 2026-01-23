# FPGA Backend for ABI

> **Status:** Development Phase
> **Version:** 0.1.0
> **Last Updated:** January 2026

## Overview

The FPGA backend provides hardware acceleration for compute-intensive operations in the ABI framework using Field-Programmable Gate Arrays. This backend integrates with the unified GPU interface, enabling seamless switching between GPU and FPGA acceleration.

### Key Features

- **Multi-Platform Support**: AMD/Xilinx (Vitis/XRT) and Intel (oneAPI) FPGAs
- **Unified Interface**: Implements the standard `gpu.interface.Backend` VTable
- **Pre-Compiled Kernels**: Optimized bitstreams for common operations
- **Memory Management**: DDR, HBM, and on-chip memory support
- **Quantized Operations**: Native support for Q4, Q5, Q8 quantization formats

## Architecture

```
src/gpu/backends/fpga/
├── mod.zig          # Module entry point, public API
├── types.zig        # Core type definitions
├── vtable.zig       # GPU interface implementation
├── loader.zig       # Bitstream loading and device management
├── memory.zig       # Memory allocation and transfer
└── kernels.zig      # Kernel implementations (CPU simulation + HLS templates)
```

### Module Dependencies

```
mod.zig
    ├── types.zig      (core types)
    ├── vtable.zig     (backend implementation)
    │       ├── loader.zig
    │       ├── memory.zig
    │       └── kernels.zig
    └── ../../interface.zig  (GPU interface)
```

## Supported Operations

### Vector Operations

| Operation | Status | Description |
|-----------|--------|-------------|
| `vector_distance` | Implemented | Cosine, L2, dot product similarity |
| `vector_add` | Implemented | Element-wise vector addition |
| `vector_scale` | Planned | Scalar multiplication |

### Matrix Operations

| Operation | Status | Description |
|-----------|--------|-------------|
| `matmul` | Implemented | FP32 matrix multiplication |
| `quantized_matmul` | Implemented | Q4/Q8 quantized matmul |
| `matmul_blocked` | Planned | Tiled matrix multiplication |

### Neural Network Operations

| Operation | Status | Description |
|-----------|--------|-------------|
| `softmax` | Implemented | Streaming softmax |
| `rmsnorm` | Planned | RMS normalization |
| `silu_activation` | Planned | SiLU/Swish activation |
| `rope_embedding` | Planned | Rotary position embeddings |
| `attention` | Planned | Multi-head attention |

### Vector Database Operations

| Operation | Status | Description |
|-----------|--------|-------------|
| `kmeans_assign` | Implemented | K-means centroid assignment |
| `hnsw_search` | Planned | HNSW graph traversal |
| `pq_encode` | Planned | Product quantization encoding |

## Usage

### Basic Initialization

```zig
const std = @import("std");
const fpga = @import("abi").gpu.backends.fpga;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Check if FPGA is available
    if (!fpga.isAvailable()) {
        std.debug.print("FPGA not available\n", .{});
        return;
    }

    // Create FPGA backend
    var backend = try fpga.FpgaBackend.init(allocator, .{
        .platform = .xilinx,
        .device_index = 0,
    });
    defer backend.deinit();

    // Load bitstream with kernels
    try backend.loadBitstream("kernels/vector_ops.xclbin");

    // Use the backend...
}
```

### Memory Management

```zig
// Allocate device memory
const buffer = try backend.allocate(1024 * 1024, .{
    .device = true,
    .host_visible = false,
});
defer backend.free(buffer);

// Copy data to device
try backend.copyToDevice(buffer, host_data);

// Execute kernel...

// Copy results back
try backend.copyFromDevice(result_host, buffer);
```

### Kernel Execution

```zig
// Compile/load kernel (uses pre-compiled bitstream)
const kernel = try backend.compileKernel(allocator, "", "vector_distance");
defer backend.destroyKernel(kernel);

// Launch kernel
try backend.launchKernel(kernel, .{
    .grid = .{ 1024, 1, 1 },
    .block = .{ 256, 1, 1 },
}, &[_]*anyopaque{ query_buf, db_buf, result_buf, params_ptr });

// Wait for completion
try backend.synchronize();
```

### Integration with Unified GPU API

```zig
const abi = @import("abi");

// Create unified GPU with FPGA backend preference
var gpu = try abi.Gpu.init(allocator, .{
    .preferred_backend = .fpga,
    .allow_fallback = true,  // Fall back to GPU/CPU if FPGA unavailable
});
defer gpu.deinit();

// Use unified API - automatically routes to FPGA
const result = try gpu.vectorAdd(a, b, output);
```

## Configuration

### Build Options

Enable FPGA support at build time:

```bash
# Enable FPGA backend
zig build -Dgpu-backend=fpga

# Enable multiple backends
zig build -Dgpu-backend=cuda,vulkan,fpga

# FPGA with specific platform
zig build -Dgpu-backend=fpga -Dfpga-platform=xilinx
```

### Runtime Configuration

```zig
const config = fpga.FpgaConfig{
    // Platform selection
    .platform = .auto,          // Auto-detect, or .xilinx, .intel
    .device_index = 0,          // Device to use

    // Bitstream
    .bitstream_path = "path/to/kernels.xclbin",

    // Memory
    .default_memory_bank = .auto,  // Auto-select bank

    // Profiling
    .enable_profiling = true,
};
```

## Development

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ABI_FPGA_XILINX_DEVICE` | Enable simulated Xilinx device | `1` |
| `ABI_FPGA_INTEL_DEVICE` | Enable simulated Intel device | `1` |
| `XCL_EMULATION_MODE` | Xilinx emulation mode | `sw_emu`, `hw_emu` |

### Testing

```bash
# Run FPGA backend tests
zig build test --test-filter "fpga"

# With simulated device
ABI_FPGA_XILINX_DEVICE=1 zig build test --test-filter "fpga"
```

### HLS Kernel Development

Kernels are developed using High-Level Synthesis (HLS) in C/C++:

```cpp
// Example: vector_distance.cpp
void vector_distance(
    const float* query,
    const float* vectors,
    float* results,
    int dimension,
    int num_vectors
) {
    #pragma HLS INTERFACE m_axi port=query offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=vectors offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=results offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=dimension bundle=control
    #pragma HLS INTERFACE s_axilite port=num_vectors bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Implementation with pipelining and unrolling
    for (int v = 0; v < num_vectors; v++) {
        float dot = 0, norm_q = 0, norm_v = 0;

        for (int i = 0; i < dimension; i++) {
            #pragma HLS PIPELINE II=1
            float q = query[i];
            float db = vectors[v * dimension + i];
            dot += q * db;
            norm_q += q * q;
            norm_v += db * db;
        }

        results[v] = dot / (sqrt(norm_q) * sqrt(norm_v));
    }
}
```

Compile with Vitis HLS:

```bash
vitis_hls -f run_hls.tcl
v++ -c -t hw --platform xilinx_u250_gen3x16_xdma_4_1_202210_1 \
    -k vector_distance vector_distance.cpp -o vector_distance.xo
v++ -l -t hw --platform xilinx_u250_gen3x16_xdma_4_1_202210_1 \
    vector_distance.xo -o vector_ops.xclbin
```

## Performance

### Expected Speedups (vs CPU baseline)

| Operation | CPU (SIMD) | FPGA | Speedup |
|-----------|------------|------|---------|
| Vector distance (1M x 768) | 15ms | 0.8ms | 18.75x |
| Q4 MatMul (4096x4096) | 12ms | 0.6ms | 20x |
| Softmax (2048x2048) | 2.1ms | 0.3ms | 7x |
| K-means (10K x 768) | 85ms | 2.1ms | 40x |

### Power Efficiency

| Platform | Performance | Power | Perf/Watt |
|----------|-------------|-------|-----------|
| CPU (Xeon 8380) | 1x | 270W | 0.004x |
| GPU (A100) | 15x | 250W | 0.06x |
| FPGA (U250) | 12x | 75W | 0.16x |

## Supported Hardware

### AMD/Xilinx

| Device | LUTs | DSPs | Memory | Status |
|--------|------|------|--------|--------|
| Alveo U250 | 1.7M | 12,288 | 64 GB DDR4 | Supported |
| Alveo U280 | 1.3M | 9,024 | 8 GB HBM2 | Supported |
| Alveo U55C | 1.3M | 9,024 | 16 GB HBM2 | Planned |
| Versal AI Core | 400K | 1,968 | Varies | Planned |

### Intel

| Device | ALMs | DSPs | Memory | Status |
|--------|------|------|--------|--------|
| Agilex 7 F-Series | 2.5M | 11,520 | HBM2e | Planned |
| Stratix 10 GX | 1.0M | 5,760 | DDR4 | Planned |

## Roadmap

### Phase 1: Foundation (Current)
- [x] Basic backend structure
- [x] Memory management
- [x] VTable implementation
- [x] CPU simulation for testing
- [ ] Xilinx XRT integration

### Phase 2: Core Kernels
- [ ] Optimized vector distance (HLS)
- [ ] Quantized matrix multiplication
- [ ] Streaming softmax
- [ ] K-means acceleration

### Phase 3: LLM Support
- [ ] Full attention mechanism
- [ ] KV-cache management
- [ ] RoPE embeddings
- [ ] Multi-head parallelism

### Phase 4: Production
- [ ] Intel oneAPI support
- [ ] Multi-FPGA scaling
- [ ] Dynamic reconfiguration
- [ ] Production benchmarks

## References

- [ABI GPU Documentation](../../../../docs/gpu.md)
- [FPGA/ASIC Research](../../../../docs/research/hardware-acceleration-fpga-asic.md)
- [FPGA Inference Acceleration](../../../../docs/research/fpga-inference-acceleration.md)
- [Hybrid GPU-FPGA Architecture](../../../../docs/research/hybrid-gpu-fpga-architecture.md)
- [AMD Vitis Documentation](https://docs.xilinx.com/r/en-US/ug1393-vitis-application-acceleration)
- [Intel oneAPI FPGA](https://www.intel.com/content/www/us/en/developer/tools/oneapi/fpga.html)
