---
title: "models"
tags: [ai, models, gguf, huggingface, download, cache]
---
# Model Management
> **Codebase Status:** Synced with repository as of 2026-01-30.

<p align="center">
  <img src="https://img.shields.io/badge/Module-Models-orange?style=for-the-badge&logo=huggingface&logoColor=white" alt="Models Module"/>
  <img src="https://img.shields.io/badge/Status-Production_Ready-success?style=for-the-badge" alt="Production Ready"/>
  <img src="https://img.shields.io/badge/HuggingFace-Integrated-yellow?style=for-the-badge" alt="HuggingFace"/>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#cli-commands">CLI Commands</a> •
  <a href="#huggingface-integration">HuggingFace</a> •
  <a href="#cache-management">Cache</a> •
  <a href="#hot-reload">Hot-Reload</a>
</p>

---

> **Developer Guide**: See [AI Guide](ai.md) for the full AI module documentation.
> **Streaming**: See [Streaming Guide](streaming.md) for real-time inference.

The **Models** module (`abi.ai.models`) provides model download, caching, and management functionality similar to `ollama pull`.

## Features Overview

| Feature | Description | Status |
|---------|-------------|--------|
| **Download** | Download GGUF models from HuggingFace or URLs | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Cache Management** | Automatic caching with metadata tracking | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **HuggingFace Search** | Browse and search the HuggingFace model hub | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Resume Support** | Resume interrupted downloads | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Hot-Reload** | Swap models without server restart | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Checksum Verification** | Validate downloaded files | ![Ready](https://img.shields.io/badge/-Ready-success) |

## Architecture

```
src/ai/models/
├── mod.zig           # Module entry point
├── manager.zig       # Model cache manager
├── downloader.zig    # HTTP download with progress
├── huggingface.zig   # HuggingFace API client
└── stub.zig          # Stub when AI disabled
```

## Quick Start

### Download a Model

```bash
# Download from HuggingFace using shorthand
abi model download TheBloke/Llama-2-7B-GGUF:Q4_K_M

# Download from direct URL
abi model download https://example.com/model.gguf

# Download with custom name
abi model download TheBloke/Mistral-7B-GGUF:Q4_K_M --name mistral-7b
```

### List Cached Models

```bash
abi model list
```

**Output:**
```
Cached Models (3 total, 14.2 GB)
────────────────────────────────────────────────────────
  llama-2-7b-q4_k_m.gguf       3.8 GB    2026-01-26 10:30
  mistral-7b-q4_k_m.gguf       3.8 GB    2026-01-25 14:22
  phi-2-q8_0.gguf              2.8 GB    2026-01-24 09:15

Cache directory: ~/.abi/models/
```

### Use with Streaming Server

```bash
# Start server with cached model
abi llm serve -m $(abi model path llama-2-7b-q4_k_m)

# Or use the full path
abi llm serve -m ~/.abi/models/llama-2-7b-q4_k_m.gguf
```

## CLI Commands

### `abi model list`

List all cached models with sizes and metadata.

```bash
# Basic list
abi model list

# JSON output
abi model list --json

# Without sizes (faster)
abi model list --no-size
```

**Options:**
| Option | Description |
|--------|-------------|
| `--json` | Output as JSON array |
| `--no-size` | Skip size calculation |

### `abi model info <name>`

Show detailed information about a cached model.

```bash
abi model info llama-2-7b-q4_k_m
```

**Output:**
```
Model: llama-2-7b-q4_k_m.gguf
────────────────────────────────────────────────────────
Path:         ~/.abi/models/llama-2-7b-q4_k_m.gguf
Size:         3.83 GB (4,113,248,256 bytes)
Downloaded:   2026-01-26 10:30:45 UTC
Source:       TheBloke/Llama-2-7B-GGUF
Quantization: Q4_K_M
SHA256:       a1b2c3d4e5f6...

GGUF Metadata:
  Architecture:    llama
  Context Length:  4096
  Embedding Size:  4096
  Parameters:      6.74B
  Vocab Size:      32000
```

### `abi model download <id>`

Download a model from HuggingFace or a direct URL.

```bash
# HuggingFace shorthand (repo:quantization)
abi model download TheBloke/Llama-2-7B-GGUF:Q4_K_M

# Full HuggingFace URL
abi model download https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf

# Direct URL
abi model download https://example.com/model.gguf

# Custom output name
abi model download TheBloke/Mistral-7B-GGUF:Q4_K_M --name my-mistral

# Skip checksum verification
abi model download TheBloke/Phi-2-GGUF:Q8_0 --no-verify

# Force re-download (overwrite existing)
abi model download TheBloke/Llama-2-7B-GGUF:Q4_K_M --force
```

**Options:**
| Option | Description |
|--------|-------------|
| `--name <name>` | Custom filename for the downloaded model |
| `--no-verify` | Skip SHA256 checksum verification |
| `--force` | Overwrite existing file |
| `--quiet` | Suppress progress output |

**Progress Display:**
```
Downloading: llama-2-7b.Q4_K_M.gguf
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 67% 2.5/3.8 GB
Speed: 45.2 MB/s | ETA: 00:28
```

### `abi model remove <name>`

Remove a cached model.

```bash
# Remove single model
abi model remove llama-2-7b-q4_k_m

# Remove with confirmation skip
abi model remove llama-2-7b-q4_k_m --yes

# Remove multiple models
abi model remove llama-2-7b-q4_k_m mistral-7b-q4_k_m
```

**Options:**
| Option | Description |
|--------|-------------|
| `--yes`, `-y` | Skip confirmation prompt |

### `abi model search <query>`

Search the HuggingFace model hub for GGUF models.

```bash
# Search for models
abi model search llama

# Filter by quantization
abi model search mistral --quant Q4_K_M

# Limit results
abi model search phi --limit 5

# JSON output
abi model search llama --json
```

**Output:**
```
Search Results for "llama" (showing 10 of 234)
────────────────────────────────────────────────────────
  TheBloke/Llama-2-7B-GGUF
    ⬇ 1.2M downloads | ⭐ 892 likes | Updated: 2026-01-15
    Quantizations: Q2_K, Q3_K_S, Q3_K_M, Q4_0, Q4_K_S, Q4_K_M, Q5_0, Q5_K_S, Q5_K_M, Q6_K, Q8_0

  TheBloke/Llama-2-13B-GGUF
    ⬇ 856K downloads | ⭐ 654 likes | Updated: 2026-01-14
    Quantizations: Q2_K, Q3_K_S, Q3_K_M, Q4_0, Q4_K_S, Q4_K_M, Q5_0, Q5_K_S, Q5_K_M, Q6_K, Q8_0

  ...
```

**Options:**
| Option | Description |
|--------|-------------|
| `--quant <type>` | Filter by quantization type |
| `--limit <n>` | Maximum results (default: 10) |
| `--json` | Output as JSON |
| `--sort <field>` | Sort by: downloads, likes, updated |

### `abi model path [name]`

Show or set the cache directory path.

```bash
# Show cache directory
abi model path

# Get path to specific model
abi model path llama-2-7b-q4_k_m

# Set custom cache directory
abi model path --set /mnt/models
```

**Output:**
```
~/.abi/models/llama-2-7b-q4_k_m.gguf
```

## HuggingFace Integration

### Shorthand Syntax

The recommended way to specify HuggingFace models:

```
<owner>/<repo>:<quantization>
```

**Examples:**
```bash
TheBloke/Llama-2-7B-GGUF:Q4_K_M
TheBloke/Mistral-7B-Instruct-v0.2-GGUF:Q5_K_S
microsoft/phi-2-GGUF:Q8_0
```

### Quantization Types

| Type | Description | Size | Quality |
|------|-------------|------|---------|
| `Q2_K` | 2-bit, K-quants | Smallest | Lowest |
| `Q3_K_S` | 3-bit, K-quants small | Very small | Low |
| `Q3_K_M` | 3-bit, K-quants medium | Small | Low-Medium |
| `Q4_0` | 4-bit, legacy | Small | Medium |
| `Q4_K_S` | 4-bit, K-quants small | Small | Medium |
| `Q4_K_M` | 4-bit, K-quants medium | Medium | **Good balance** |
| `Q5_0` | 5-bit, legacy | Medium | Good |
| `Q5_K_S` | 5-bit, K-quants small | Medium | Good |
| `Q5_K_M` | 5-bit, K-quants medium | Medium-Large | Very good |
| `Q6_K` | 6-bit, K-quants | Large | Excellent |
| `Q8_0` | 8-bit | Largest | Near-lossless |

**Recommendation:** Use `Q4_K_M` for the best balance of size and quality.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `ABI_HF_API_TOKEN` | HuggingFace API token for private repos |
| `ABI_MODELS_DIR` | Custom cache directory |

## Cache Management

### Default Cache Locations

| Platform | Default Path |
|----------|--------------|
| Linux | `~/.abi/models/` |
| macOS | `~/.abi/models/` |
| Windows | `%LOCALAPPDATA%\abi\models\` |

### Cache Structure

```
~/.abi/models/
├── llama-2-7b-q4_k_m.gguf
├── mistral-7b-q4_k_m.gguf
├── phi-2-q8_0.gguf
└── .metadata/
    ├── llama-2-7b-q4_k_m.json
    ├── mistral-7b-q4_k_m.json
    └── phi-2-q8_0.json
```

### Metadata Format

Each model has an associated metadata file:

```json
{
  "name": "llama-2-7b-q4_k_m.gguf",
  "source": "TheBloke/Llama-2-7B-GGUF",
  "quantization": "Q4_K_M",
  "size_bytes": 4113248256,
  "sha256": "a1b2c3d4e5f6...",
  "downloaded_at": "2026-01-26T10:30:45Z",
  "huggingface": {
    "repo_id": "TheBloke/Llama-2-7B-GGUF",
    "filename": "llama-2-7b.Q4_K_M.gguf",
    "commit": "abc123"
  }
}
```

### Clearing the Cache

```bash
# Remove all cached models
rm -rf ~/.abi/models/*

# Or remove specific models
abi model remove llama-2-7b-q4_k_m
```

## Hot-Reload

The streaming server supports swapping models without restart.

### Via Admin Endpoint

```bash
# Reload with a new model
curl -X POST http://localhost:8080/admin/reload \
  -H "Content-Type: application/json" \
  -d '{"model_path": "/path/to/new-model.gguf"}'
```

### Reload Behavior

1. **Drain Phase**: Server stops accepting new requests
2. **Wait Phase**: Waits for active streams to complete (30s timeout)
3. **Unload Phase**: Current model is unloaded from memory
4. **Load Phase**: New model is loaded
5. **Resume Phase**: Server resumes accepting requests

### Limitations

- No rollback on failure (server left without model)
- No authentication on admin endpoint (secure at network level)
- Model must be a valid GGUF file

## Programmatic Usage

### Zig API

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize model manager
    var manager = try abi.ai.models.Manager.init(allocator, .{
        .cache_dir = null,  // Use default
        .auto_scan = true,  // Scan cache on init
    });
    defer manager.deinit();

    // List cached models
    const models = manager.listModels();
    for (models) |model| {
        std.debug.print("{s}: {d} MB\n", .{
            model.name,
            model.size_bytes / 1024 / 1024,
        });
    }

    // Download a model
    try manager.download("TheBloke/Llama-2-7B-GGUF:Q4_K_M", .{
        .progress_callback = struct {
            fn callback(progress: abi.ai.models.DownloadProgress) void {
                std.debug.print("\r{d}%", .{progress.percent});
            }
        }.callback,
    });

    // Get model path
    const path = try manager.getModelPath("llama-2-7b-q4_k_m");
    std.debug.print("Model at: {s}\n", .{path});
}
```

### Manager Configuration

```zig
pub const ManagerConfig = struct {
    /// Custom cache directory (null = platform default)
    cache_dir: ?[]const u8 = null,
    /// Scan cache directory on init
    auto_scan: bool = true,
    /// Verify checksums on load
    verify_checksums: bool = true,
};
```

### Download Configuration

```zig
pub const DownloadConfig = struct {
    /// Progress callback
    progress_callback: ?*const fn(DownloadProgress) void = null,
    /// Resume partial downloads
    resume: bool = true,
    /// Verify SHA256 after download
    verify_checksum: bool = true,
    /// Custom output filename
    output_name: ?[]const u8 = null,
    /// Overwrite existing file
    force: bool = false,
};
```

## Best Practices

### 1. Use Quantization Wisely

Choose quantization based on your hardware:

| RAM Available | Recommended |
|---------------|-------------|
| 4 GB | Q2_K or Q3_K_S (7B models only) |
| 8 GB | Q4_K_M (7B models) |
| 16 GB | Q5_K_M (7B) or Q4_K_M (13B) |
| 32 GB | Q6_K (13B) or Q4_K_M (30B) |
| 64 GB+ | Q8_0 for maximum quality |

### 2. Pre-download for Production

Download models before deployment:

```bash
# Download required models
abi model download TheBloke/Llama-2-7B-GGUF:Q4_K_M
abi model download TheBloke/Mistral-7B-GGUF:Q4_K_M

# Verify downloads
abi model list
```

### 3. Use Checksums

Always verify downloads in production:

```bash
# Checksum is verified by default
abi model download TheBloke/Llama-2-7B-GGUF:Q4_K_M

# Only skip for testing/development
abi model download TheBloke/Llama-2-7B-GGUF:Q4_K_M --no-verify
```

### 4. Set Custom Cache for Shared Storage

For multi-user or distributed setups:

```bash
# Set shared cache directory
export ABI_MODELS_DIR=/mnt/shared/models

# Or per-command
abi model path --set /mnt/shared/models
```

## Troubleshooting

### Download Fails

**Symptom:** Download fails with network error

**Solutions:**
1. Check internet connectivity
2. Verify HuggingFace is accessible
3. For private repos, set `ABI_HF_API_TOKEN`
4. Try direct URL if shorthand fails

### Checksum Mismatch

**Symptom:** `Checksum verification failed`

**Solutions:**
1. Re-download the model: `abi model download ... --force`
2. Check disk space
3. Verify network stability during download

### Model Not Found

**Symptom:** `Model not found in cache`

**Solutions:**
1. List cached models: `abi model list`
2. Check exact filename (case-sensitive)
3. Verify cache directory: `abi model path`

### Out of Disk Space

**Symptom:** Download fails partway through

**Solutions:**
1. Check available space: `df -h`
2. Clear unused models: `abi model remove <name>`
3. Use smaller quantization (Q4_K_M instead of Q8_0)

## See Also

- [AI Guide](ai.md) - Full AI module documentation
- [Streaming Guide](streaming.md) - Real-time inference with SSE/WebSocket
- [LLM Guide](api_ai-llm.md) - LLM inference API reference
- [CLI Reference](../CLAUDE.md#cli-commands) - Full CLI documentation
