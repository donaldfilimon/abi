# GPU Kernel Library

Pre-built kernel definitions using the GPU DSL for common operations.
These kernels can be compiled to any supported backend (CUDA, Vulkan, Metal, WebGPU).

## Available Kernels

### Attention (`flash_attention.zig`)

Memory-efficient attention implementation based on the FlashAttention algorithm.

**Features:**
- O(N) memory complexity instead of O(N²)
- Tiled computation with online softmax normalization
- Causal masking support
- Half-precision (f16) support for tensor cores
- Tuned configurations for different GPU architectures

**Usage:**
```zig
const flash = @import("kernels").flash_attention;

// Create kernel IR with default config
const ir = try flash.createFlashAttentionKernel(allocator, flash.default_config);

// Or use architecture-specific tuning
const ir_ampere = try flash.createFlashAttentionKernel(allocator, flash.TunedConfigs.ampere);
```

**Tuned Configurations:**
| Architecture | Block Size Q | Block Size KV | Head Dim | Half Precision |
|--------------|--------------|---------------|----------|----------------|
| Ampere (A100, RTX 30xx) | 128 | 128 | 64 | Yes |
| Hopper (H100, RTX 40xx) | 128 | 256 | 128 | Yes |
| RDNA3 (AMD) | 64 | 64 | 64 | Yes |
| Apple Silicon | 64 | 64 | 64 | Yes |
| Fallback | 32 | 32 | 64 | No |

### Fused Operations (`fused_ops.zig`)

Fused kernels that combine multiple operations to reduce memory bandwidth.

**Available Fusions:**
- **LayerNorm + Linear**: Combines layer normalization with linear projection
- **RMSNorm + RoPE**: Combines RMS normalization with rotary positional embedding
- **SwiGLU**: Fused gated linear unit with SiLU activation
- **Residual + LayerNorm**: Combines residual connection with layer norm
- **GeGLU**: Fused gated linear unit with GELU activation
- **Softmax + Dropout**: Combines softmax with dropout for training

**Usage:**
```zig
const fused = @import("kernels").fused_ops;

// Create SwiGLU kernel
const swiglu_ir = try fused.createSwiGLUKernel(allocator, .{
    .hidden_dim = 4096,
    .use_half = true,
});

// Create RMSNorm + RoPE kernel
const rope_ir = try fused.createRMSNormRoPEKernel(allocator, .{
    .hidden_dim = 4096,
    .head_dim = 128,
    .max_seq_len = 8192,
});
```

## Compiling Kernels

Kernels produce `KernelIR` which can be compiled to any backend:

```zig
const spirv = @import("codegen").spirv;
const cuda = @import("codegen").cuda;

// Compile to SPIR-V for Vulkan
const spirv_code = try spirv.generate(allocator, &ir);

// Compile to CUDA PTX
const cuda_code = try cuda.generate(allocator, &ir);
```

## Memory Requirements

Use `requiredSharedMemory()` to check shared memory requirements:

```zig
const flash = @import("kernels").flash_attention;
const config = flash.TunedConfigs.ampere;
const shared_mem = flash.requiredSharedMemory(config);

// Validate against GPU limits
try flash.validateConfig(config, gpu_max_shared_memory);
```

## Architecture

```
kernels/
├── mod.zig              # Module exports and vendor detection
├── flash_attention.zig  # FlashAttention kernel
├── fused_ops.zig        # Fused operation kernels
└── README.md            # This file
```

## References

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135) - Dao et al., 2022
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) - Improved algorithm
- [SPIR-V Specification](https://registry.khronos.org/SPIR-V/)
