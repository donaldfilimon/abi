# Feature Flags Reference
> **Codebase Status:** Synced with repository as of 2026-01-18.

> **Single source of truth for all ABI framework build options**

This document describes all feature flags available when building the ABI framework. Use these flags with `zig build` to customize your build.

## Quick Reference

```bash
# Enable specific features
zig build -Denable-ai=true -Denable-gpu=false

# Select GPU backend
zig build -Dgpu-cuda=true -Dgpu-vulkan=false

# Production build with all features
zig build -Doptimize=ReleaseFast
```

---

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-Denable-ai` | `true` | AI agents, LLM connectors, training, federated learning |
| `-Denable-gpu` | `true` | GPU acceleration framework with unified API |
| `-Denable-database` | `true` | Vector database (WDBX) with HNSW indexing |
| `-Denable-network` | `true` | Distributed compute, node discovery, Raft consensus |
| `-Denable-web` | `true` | HTTP client/server, web utilities |
| `-Denable-profiling` | `true` | Performance profiling, metrics, monitoring |
| `-Denable-explore` | `true` | Codebase exploration and search (requires `-Denable-ai`) |
| `-Denable-llm` | `true` | Local LLM inference with GGUF models (requires `-Denable-ai`) |

### Flag Dependencies

```
enable-ai ──┬── enable-explore
            └── enable-llm
```

When `-Denable-ai=false`, both `-Denable-explore` and `-Denable-llm` are automatically disabled.

---

## GPU Backend Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-Dgpu-vulkan` | `true` | Vulkan compute backend (cross-platform, recommended) |
| `-Dgpu-cuda` | `false` | NVIDIA CUDA backend (requires CUDA toolkit) |
| `-Dgpu-metal` | `false` | Apple Metal backend (macOS/iOS only) |
| `-Dgpu-webgpu` | `true`* | WebGPU backend (when `-Denable-web=true`) |
| `-Dgpu-opengl` | `false` | OpenGL compute backend (legacy) |
| `-Dgpu-opengles` | `false` | OpenGL ES backend (mobile/embedded) |
| `-Dgpu-stdgpu` | `false` | Zig std.gpu SPIR-V backend (CPU fallback) |
| `-Dgpu-webgl2` | `true`* | WebGL2 backend (when `-Denable-web=true`) |

*WebGPU and WebGL2 default to `true` when `-Denable-web=true`.

### Backend Selection Guidelines

- **Cross-platform desktop**: Use `-Dgpu-vulkan=true` (default)
- **NVIDIA GPU optimization**: Use `-Dgpu-cuda=true -Dgpu-vulkan=false`
- **Apple platforms**: Use `-Dgpu-metal=true`
- **Web deployment**: Use `-Dgpu-webgpu=true`
- **CPU-only fallback**: Use `-Dgpu-stdgpu=true -Dgpu-vulkan=false`

### Backend Conflict Warnings

The build system will warn if potentially conflicting backends are enabled:
- CUDA + Vulkan (may cause resource conflicts)
- OpenGL + WebGL2 (prefer one or the other)

---

## WASM Target Limitations

When building for WebAssembly targets, certain features are automatically disabled:

| Feature | Status | Reason |
|---------|--------|--------|
| `-Denable-database` | Disabled | File system access |
| `-Denable-network` | Disabled | Socket operations |
| `-Denable-gpu` | Limited | Only WebGPU/WebGL2 available |

---

## Build Configurations

### Development (Default)

```bash
zig build
```

All features enabled with debug symbols.

### Production

```bash
zig build -Doptimize=ReleaseFast
```

Optimized build with all features.

### Minimal AI

```bash
zig build -Denable-ai=true -Denable-gpu=false -Denable-database=false -Denable-network=false
```

AI features only, no GPU/database/network.

### GPU Compute Only

```bash
zig build -Denable-gpu=true -Denable-ai=false -Denable-database=false
```

GPU acceleration without AI or database.

### CUDA-Optimized

```bash
zig build -Dgpu-cuda=true -Dgpu-vulkan=false -Doptimize=ReleaseFast
```

NVIDIA CUDA backend for maximum GPU performance.

### Database-Only

```bash
zig build -Denable-database=true -Denable-ai=false -Denable-gpu=false
```

Vector database without AI or GPU.

---

## Cache Options

| Flag | Default | Description |
|------|---------|-------------|
| `-Dcache-dir` | `.zig-cache` | Directory for build cache |
| `-Dglobal-cache-dir` | (none) | Directory for global build cache |

---

## Environment Variables

The framework also respects environment variables for runtime configuration:

### AI Connectors

| Variable | Default | Description |
|----------|---------|-------------|
| `ABI_OPENAI_API_KEY` or `OPENAI_API_KEY` | - | OpenAI API authentication |
| `ABI_OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI base URL |
| `ABI_OLLAMA_HOST` or `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL |
| `ABI_OLLAMA_MODEL` | `gpt-oss` | Default Ollama model |
| `ABI_HF_API_TOKEN` or `HF_API_TOKEN` | - | HuggingFace API access |
| `DISCORD_BOT_TOKEN` | - | Discord integration token |

### Network Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ABI_LOCAL_SCHEDULER_URL` | `http://127.0.0.1:8081` | Local scheduler URL |

---

## Checking Enabled Features

At runtime, check if a feature is enabled:

```zig
const abi = @import("abi");

if (abi.ai.isEnabled()) {
    // AI features available
}

if (abi.gpu.isEnabled()) {
    // GPU acceleration available
}

if (abi.database.isEnabled()) {
    // Vector database available
}
```

---

## See Also

- [README.md](../README.md) - Project overview
- [CLAUDE.md](../CLAUDE.md) - Development guidelines
- [docs/gpu.md](gpu.md) - GPU programming guide
- [docs/ai.md](ai.md) - AI features guide
