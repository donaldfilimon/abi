# ABI Framework Documentation Digest

## Project Overview

**ABI** is a modern Zig framework for modular AI services, vector search, and systems tooling. Built with Zig 0.16.x, it provides a comprehensive set of features including AI agent runtime, high-performance compute engine, GPU backends, vector database helpers (WDBX), web utilities, and monitoring capabilities.

## Quick Start

### Prerequisites

- Zig 0.16.x (tested with 0.16.0-dev)

### Basic Build & Test

```bash
zig build              # Build library + CLI
zig build test         # Run all tests
zig fmt .              # Format code
zig build run -- --help # CLI help (if enabled)
```

### Feature Configuration

```bash
zig build -Denable-ai=true -Denable-gpu=false -Denable-web=true -Denable-database=true
```

## Architecture Overview

### Core Structure

- `src/abi.zig`: Public API surface and curated re-exports
- `src/root.zig`: Root module entrypoint
- `src/framework/`: Runtime config, feature orchestration, lifecycle
- `src/features/`: Vertical feature stacks (AI, GPU, database, web, monitoring)
- `src/shared/`: Shared utilities (logging, observability, platform, utils)

### Key Features

#### Compute Engine

- Work-stealing scheduler with worker thread pools
- Lock-free data structures (Chase-Lev deque, ShardedMap)
- GPU workload support with CPU fallback
- Memory management with stable allocator + worker arenas
- Result caching with metadata tracking

#### GPU Support

- Multiple backends: CUDA, Vulkan, Metal, WebGPU
- GPU memory management (buffers, pools, async transfers)
- Feature gating for conditional compilation

#### AI & Agents

- Agent runtime, training pipelines, data structures
- Tool interfaces and model registry
- Federated learning capabilities

#### Database & Vector Search

- WDBX vector database helpers
- Unified API for vector operations (insert, search, update, delete)
- Sharding and optimization support

#### Network & Distributed Compute

- Node registry and discovery
- Task/result serialization
- Distributed workload execution

#### Web & Connectors

- HTTP client/server helpers
- Weather utilities and web services
- AI service connectors (OpenAI, Hugging Face, Ollama, Local Scheduler)
- Environment variable configuration for API keys

#### Monitoring & Profiling

- Thread-safe metrics collection
- Performance histograms and timing statistics
- Logging, tracing, and health monitoring
- Prometheus/Grafana integration templates

## API Reference Summary

### Core Entry Points

```zig
const abi = @import("abi");
var framework = try abi.init(gpa.allocator(), abi.FrameworkOptions{});
defer abi.shutdown(&framework);
```

### Framework Types

- `abi.Framework`, `abi.FrameworkOptions`, `abi.RuntimeConfig`
- Feature namespaces: `abi.ai`, `abi.database`, `abi.gpu`, `abi.web`, `abi.monitoring`

### WDBX API

- Database lifecycle: `createDatabase`, `connectDatabase`, `closeDatabase`
- Vector operations: `insertVector`, `searchVectors`, `updateVector`, `getVector`
- Management: `getStats`, `optimize`, `backup`, `restore`

## Development Guidelines

### Code Style

- **Formatting**: 4 spaces, 100 char lines, `zig fmt` required
- **Naming**: PascalCase for types, snake_case for functions/variables, UPPER_SNAKE_CASE for constants
- **Imports**: Group std imports first, then internal. No `usingnamespace`. Prefer qualified access.
- **Error Handling**: Use `!` return types, specific enums over generic, `errdefer` for cleanup
- **Memory**: Stable allocator for long-lived data, worker arenas for scratch, `defer`/`errdefer` for cleanup

### Testing

- `test` blocks at file end, use `testing.allocator`
- Co-locate `*_test.zig`, use `mod.zig` for re-exports
- Run `zig build test` to validate changes

### Zig 0.16 Specifics

- Use `cmpxchgStrong`/`cmpxchgWeak` (returns `?T`)
- Use `std.atomic.spinLoopHint()` instead of `spinLoop`
- Use `std.Thread.spawn(.{}, ...)` (options struct first)

## Environment Variables

### AI Connectors

- `ABI_OPENAI_API_KEY`, `OPENAI_API_KEY`
- `ABI_OPENAI_BASE_URL` (default: `https://api.openai.com/v1`)
- `ABI_HF_API_TOKEN`, `HF_API_TOKEN`, `HUGGING_FACE_HUB_TOKEN`
- `ABI_HF_BASE_URL` (default: `https://api-inference.huggingface.co`)
- `ABI_LOCAL_SCHEDULER_URL`, `LOCAL_SCHEDULER_URL`
- `ABI_OLLAMA_HOST`, `OLLAMA_HOST` (default: `http://127.0.0.1:11434`)

## Build Flags & Feature Gating

| Flag | Default | Description |
|------|---------|-------------|
| `-Denable-ai` | `true` | AI features and modules |
| `-Denable-gpu` | `true` | GPU acceleration features |
| `-Denable-web` | `true` | Web utilities and HTTP features |
| `-Denable-database` | `true` | Database and vector search features |
| `-Denable-network` | `false` | Distributed network compute |
| `-Denable-profiling` | `false` | Profiling and metrics collection |
| `-Dgpu-cuda` | - | Enable CUDA GPU backend |
| `-Dgpu-vulkan` | - | Enable Vulkan GPU backend |
| `-Dgpu-metal` | - | Enable Metal GPU backend |
| `-Dgpu-webgpu` | - | Enable WebGPU backend |

## Deployment

### Build Release Binary

```bash
zig build -Doptimize=ReleaseFast
```

### Deployment Assets

- `deploy/docker`: Compose definitions
- `deploy/kubernetes`: Base manifests
- `deploy/monitoring`: Prometheus/Grafana templates
- `deploy/nginx`: Reverse proxy templates

## Testing Coverage

### Test Categories

- Compute engine: Worker threads, work-stealing, result caching
- GPU: Buffer allocation, memory pool, serialization
- Network: Task/result serialization, node registry
- Profiling: Metrics collection, histograms
- Integration: 10+ end-to-end tests with feature gating

### Running Tests

```bash
zig build test                                    # All tests
zig build test -Denable-gpu=true -Denable-network=true # With features
zig test src/compute/runtime/engine.zig            # Specific file
zig test --test-filter="engine init"               # Filtered tests
```

## Recent Changes (v0.2.0)

### Major Additions

- High-performance compute runtime with work-stealing scheduler
- GPU support with multiple backends (CUDA, Vulkan, Metal, WebGPU)
- Network distributed compute with serialization
- Profiling & metrics with thread-safe collectors
- Benchmarking framework with pre-built workloads
- Updated build system with feature flags for Zig 0.16

### Build System Updates

- Feature flags for conditional compilation
- Updated APIs for Zig 0.16 (cmpxchgStrong, std.time.Timer, spinLoopHint)
- 10+ integration tests with feature gating

## Security & Contributing

### Security Policy

- Supported versions: 0.1.x and later
- Report vulnerabilities via GitHub Security Advisories
- Best practices: Keep Zig updated, validate inputs, use latest version

### Contributing

- Development workflow: Clone → Build → Test → Format → PR
- Code style: 4 spaces, <100 chars, explicit imports, `!` return types
- Testing: Unit tests required for new features, run `zig build test`
- Documentation: Update docs for public API changes

### License

Apache License 2.0 (Copyright © 2025 Donald Filimon)

## Gotchas & Best Practices

1. **Memory Management**: Use stable allocator for persistent data, worker arenas for temporary allocations. Never destroy arenas mid-session.

2. **Feature Gating**: Test with different feature combinations since conditional compilation affects available APIs.

3. **Error Handling**: Use specific error enums rather than generic ones. Leverage `errdefer` for cleanup.

4. **GPU Resources**: Validate hardware availability before enabling GPU features in production.

5. **Environment Variables**: Required for AI connectors - ensure they're set in deployment environments.

6. **Thread Safety**: Many components are thread-safe but verify assumptions for custom integrations.

7. **Zig Version**: Requires Zig 0.16.x - older versions won't compile due to API changes.
