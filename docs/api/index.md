# ABI API Reference
> **Codebase Status:** Synced with repository as of 2026-01-22.

Auto-generated API documentation index. For detailed API references, see module-specific documentation.

## Quick Navigation

- [Core Framework](#core-framework)
- [GPU Acceleration](#gpu-acceleration)
- [AI & Agents](#ai--agents)
- [Database](#database)
- [Network](#network)
- [Compute Runtime](#compute-runtime)

---

## Core Framework

**Module:** `abi`  
**Source:** `src/abi.zig`  
**Guide:** [Framework Guide](../framework.md#api-reference)

High-level entrypoints and re-exports for the modernized runtime.

### Core Components

- `Framework` - Framework orchestration layer
- `Config` - Unified configuration system
- `Feature` - Feature enum and runtime checking
- `init()`, `shutdown()` - Lifecycle management

**Related Documentation:**

- [Framework Guide](../framework.md)
- [Configuration System](../framework.md#configuration)

---

## GPU Acceleration

**Module:** `gpu`  
**Source:** `src/gpu/unified.zig`  
**Guide:** [GPU Guide](../gpu.md#api-reference)

Unified GPU API supporting multiple backends (CUDA, Vulkan, Metal, WebGPU, OpenGL).

### Quick Start

```zig
const abi = @import("abi");

var fw = try abi.init(allocator, .{ .gpu = .{ .backend = .vulkan } });
defer fw.deinit();

const gpu = try fw.getGpu();

// Create buffers and perform operations
var a = try gpu.createBufferFromSlice(f32, &[_]f32{ 1, 2, 3, 4 }, .{});
var b = try gpu.createBufferFromSlice(f32, &[_]f32{ 5, 6, 7, 8 }, .{});
var result = try gpu.createBuffer(4 * @sizeOf(f32), .{});

try gpu.vectorAdd(f32, a, b, result);
```

### GPU Components

- `Gpu` - Main GPU context
- `GpuBuffer` - GPU memory management
- `GpuDevice` - Device selection and capabilities
- `KernelBuilder` - Custom kernel compilation
- `vectorAdd`, `matrixMultiply` - High-level operations

**Related Documentation:**

- [GPU Guide](../gpu.md)
- [GPU Architecture](../diagrams/gpu-architecture.md)

---

## AI & Agents

**Module:** `ai`
**Source:** `src/ai/mod.zig`
**Guide:** [AI Guide](../ai.md#api-reference)

AI feature module with agents, transformers, training, and federated learning.

### AI Components

- `Agent` - Conversational AI agents
- `TransformerConfig`, `TransformerModel` - LLM inference
- `LlmTrainer`, `TrainableModel` - Training pipelines
- `ExploreAgent` - Codebase exploration
- `Tool`, `ToolRegistry` - Agent tooling system

**Related Documentation:**

- [AI Guide](../ai.md)
- [AI Dataflow](../diagrams/ai-dataflow.md)

---

## Database

**Module:** `database`
**Source:** `src/database/mod.zig`
**Guide:** [Database Guide](../database.md#api-reference)

WDBX vector database with HNSW indexing and hybrid search.

### Database Components

- `createDatabase`, `connectDatabase` - Database management
- `insertVector`, `searchVectors` - Vector operations
- `updateVector`, `deleteVector` - Vector updates
- `getVector`, `listVectors` - Vector retrieval

**Related Documentation:**

- [Database Guide](../database.md)
- [Vector Database Tutorial](../tutorials/vector-database.md)

---

## Network

**Module:** `network`  
**Source:** `src/network/mod.zig`  
**Guide:** [Network Guide](../network.md)

Distributed compute and Raft consensus for cluster coordination.

### Network Components

- `NetworkConfig` - Network configuration
- `NetworkState` - Cluster state management
- Distributed task execution

**Related Documentation:**

- [Network Guide](../network.md)

---

## Compute Runtime

**Module:** `runtime`
**Source:** `src/runtime/mod.zig`
**Guide:** [Compute Guide](../compute.md#api-reference)

Work-stealing scheduler, futures, cancellation, and task groups.

### Runtime Components

- `DistributedComputeEngine` - Main compute engine
- `Future`, `Promise` - Async result handling
- `CancellationToken` - Cooperative cancellation
- `TaskGroup` - Hierarchical task organization
- `runTask`, `submitTask`, `waitForResult` - Task execution

**Related Documentation:**

- [Compute Guide](../compute.md)

---

## Module-Specific References

For detailed API documentation with complete symbol listings, see the API Reference section in each guide:

| Module | Documentation |
|--------|---------------|
| Core Framework | [Framework Guide - API Reference](../framework.md#api-reference) |
| GPU | [GPU Guide - API Reference](../gpu.md#api-reference) |
| AI | [AI Guide - API Reference](../ai.md#api-reference) |
| Database | [Database Guide - API Reference](../database.md#api-reference) |
| Network | [Network Guide](../network.md) |
| Compute | [Compute Guide - API Reference](../compute.md#api-reference) |

---

*Generated API documentation. See guide files for complete API references.*
