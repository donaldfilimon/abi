# Suggested Improvements to CLAUDE.md

Based on recent development work (GPU training, paged attention, quantization), here are recommended additions to CLAUDE.md:

## Section: LLM CLI Examples (line 752)

**Current:**
```
The LLM feature (`src/features/ai/llm/`) provides local GGUF model inference with BPE tokenization, quantized tensors (Q4_0, Q4_1, Q8_0), transformer ops (matmul, attention, RoPE, RMSNorm), KV cache, and sampling strategies (greedy, top-k, top-p, temperature).
```

**Suggested improvement:**
```
The LLM feature (`src/features/ai/llm/`) provides local GGUF model inference with:
- **Tokenization**: BPE and SentencePiece (Viterbi)
- **Quantization**: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 with roundtrip encoding
- **Transformer Ops**: MatMul, attention, RoPE, RMSNorm, SiLU with SIMD
- **KV Cache**: Standard, sliding window, and paged attention (vLLM-style)
- **GPU Acceleration**: CUDA kernels for softmax, RMSNorm, SiLU with CPU fallback
- **Sampling**: Greedy, top-k, top-p, temperature, tail-free, mirostat (v1/v2)
- **Export**: GGUF writer for trained model export
```

## New Section: GPU LLM Operations (after line 341)

Add a new subsection under "GPU Backend Development Patterns":

```markdown
### GPU-Accelerated LLM Operations

The `src/features/ai/llm/ops/gpu.zig` provides GPU-accelerated LLM inference with automatic CPU fallback:

```zig
var gpu_ctx = abi.llm.ops.gpu.GpuOpsContext.init(allocator);
defer gpu_ctx.deinit();

// Operations automatically use GPU when available, fall back to CPU
gpu_ctx.rmsNorm(x, weight, eps);
gpu_ctx.softmax(x);
gpu_ctx.silu(x);
gpu_ctx.matrixMultiply(a, b, c, m, k, n);
```

**Features**:
- **CUDA Kernels**: Softmax, RMSNorm, SiLU, elementwise ops compiled via NVRTC
- **cuBLAS Integration**: Matrix operations use cuBLAS SGEMM when available
- **Automatic Fallback**: All operations fall back to CPU if GPU unavailable
- **Statistics**: Track GPU utilization via `gpu_ctx.stats.gpuUtilization()`

**Key files**:
- `src/compute/gpu/backends/cuda/llm_kernels.zig` - CUDA kernel implementations
- `src/features/ai/llm/ops/gpu.zig` - Unified GPU ops context

**Performance notes**:
- GPU operations allocate device memory on each call (consider buffer pooling for production)
- CUDA kernels use block_size=256 (may need tuning for specific GPUs)
- cuBLAS uses column-major, code handles row-major conversion
```

## New Section: LLM Training with GPU Acceleration (after Train CLI Examples)

Add after line 787:

```markdown
### LLM Training Architecture

The training infrastructure (`src/features/ai/training/`) provides:

**Optimizers**: SGD, Adam, AdamW with gradient accumulation and clipping
**LR Schedules**: constant, cosine, warmup_cosine, step, polynomial
**Advanced Features**: LoRA, mixed precision, gradient checkpointing

**GPU-Accelerated Training**:
```zig
var gpu_backward = abi.llm.ops.backward.GpuBackwardContext.init(allocator);
defer gpu_backward.deinit();

// Backward pass uses cuBLAS for matmul gradients when available
gpu_backward.matmulBackward(dC, A, B, dA, dB, m, k, n);
gpu_backward.batchedMatmulBackward(dC, A, B, dA, dB, batch, m, k, n);

// Check GPU utilization
const stats = gpu_backward.stats();
std.log.info("GPU training utilization: {d:.1}%", .{stats.gpu_utilization * 100});
```

**Key files**:
- `src/features/ai/llm/ops/backward/` - CPU backward operations
- `src/features/ai/llm/ops/backward/gpu_backward.zig` - GPU-accelerated backward ops
- `src/features/ai/training/trainable_model.zig` - Trainable model wrapper
- `src/features/ai/training/llm_trainer.zig` - Full training loop

**Checkpoint Format**: Portable binary serialization with metadata
**GGUF Export**: Export trained models to llama.cpp format via `gguf_writer.zig`
```

## New Section: KV Cache Strategies (after line 856)

Add a new subsection:

```markdown
### KV Cache Strategies

The codebase provides three KV cache implementations:

**1. Standard KV Cache** (`src/features/ai/llm/cache/kv_cache.zig`):
- Pre-allocated contiguous memory
- Best for fixed-length sequences
- Memory: `seq_len * kv_dim * layers * 2 * sizeof(f32)`

**2. Sliding Window Cache**:
- Ring buffer for long sequences
- Memory savings: Uses only `window_size` instead of full `seq_len`
- Enable with `KvCacheConfig{ .sliding_window = 512 }`

**3. Paged Attention** (`src/features/ai/llm/cache/paged_kv_cache.zig`):
- Block-based allocation (inspired by vLLM)
- On-demand page allocation (default 16 tokens/page)
- Sequence forking with copy-on-write for prefix sharing
- Best for variable-length batched inference

```zig
var paged_cache = try abi.llm.cache.PagedKvCache.init(allocator, .{
    .num_layers = 32,
    .num_kv_heads = 8,
    .head_dim = 128,
    .max_pages = 1024,
    .page_size = 16,
});
defer paged_cache.deinit();

// Create sequences
const seq1 = try paged_cache.createSequence();
const seq2 = try paged_cache.forkSequence(seq1); // Share prefix

// Append tokens
try paged_cache.appendToken(seq1, layer, &k, &v);

// Stats
const stats = paged_cache.getStats();
std.log.info("Page utilization: {d:.1}%", .{stats.utilization * 100});
```

**When to use paged attention**:
- Variable-length sequences in a batch
- Beam search or parallel decoding
- Shared prefixes across sequences
- Memory-constrained environments
```

## Section Update: GPU Backend Development Patterns (line 283)

Add to the "Key patterns" list:

```markdown
- **Memory Pooling**: For production, implement device buffer pooling to avoid allocation overhead
- **Kernel Tuning**: Default block_size=256 may not be optimal for all GPUs
- **Error Diagnostics**: Log CUDA error codes and provide context for debugging
```

## Section Update: Testing Utilities (line 825)

Add after Hardware-Gated Tests:

```markdown
### GPU Testing Best Practices

When testing GPU code paths:

```zig
test "gpu operation with fallback verification" {
    if (!build_options.enable_gpu) return error.SkipZigTest;

    var gpu_ctx = GpuOpsContext.init(allocator);
    defer gpu_ctx.deinit();

    // Test GPU path if available
    if (gpu_ctx.isGpuAvailable()) {
        // Verify GPU results match CPU reference
        gpu_ctx.softmax(&gpu_data);
        activations.softmaxInPlace(&cpu_data);

        for (gpu_data, cpu_data) |g, c| {
            try std.testing.expectApproxEqAbs(g, c, 1e-4);
        }
    }
}
```

**Note**: Most tests use CPU fallback automatically. Create separate GPU-specific tests when you need to verify GPU correctness or performance.
```

## New Reference Link (line 885)

Add to the References section:

```markdown
- **LLM Reference Vectors**: `src/tests/llm_reference_vectors.zig` - llama.cpp parity test vectors
```

---

## Implementation Plan

To apply these improvements:

1. Insert GPU LLM Operations section after line 341
2. Update LLM feature description at line 762
3. Add LLM Training Architecture section after line 787
4. Add KV Cache Strategies section after line 856
5. Update GPU Backend Development Patterns at line 283
6. Add GPU Testing Best Practices after line 825
7. Add reference link at line 885

These additions document the major features implemented in this session while maintaining the existing structure and style of CLAUDE.md.
[Main Workspace](MAIN_WORKSPACE.md)
