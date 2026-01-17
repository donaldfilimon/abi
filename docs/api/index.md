# ABI API Reference

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
**Details:** [api_abi.md](../api_abi.md)

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
**Details:** [api_gpu.md](../api_gpu.md)

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
**Source:** `src/features/ai/mod.zig`  
**Details:** [api_ai.md](../api_ai.md)

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
**Source:** `src/features/database/mod.zig`  
**Details:** [api_database.md](../api_database.md)

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
**Details:** [api_network.md](../api_network.md)

Distributed compute and Raft consensus for cluster coordination.

### Network Components

- `NetworkConfig` - Network configuration
- `NetworkState` - Cluster state management
- Distributed task execution

**Related Documentation:**

- [Network Guide](../network.md)

---

## Compute Runtime

**Module:** `runtime` / `compute`  
**Source:** `src/compute/runtime/mod.zig`  
**Details:** [api_compute.md](../api_compute.md)

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

For detailed API documentation with complete symbol listings, see:

| Module | Documentation |
|--------|---------------|
| Core Framework | [api_abi.md](../api_abi.md) |
| GPU | [api_gpu.md](../api_gpu.md) |
| AI | [api_ai.md](../api_ai.md) |
| Database | [api_database.md](../api_database.md) |
| Network | [api_network.md](../api_network.md) |
| Compute | [api_compute.md](../api_compute.md) |

---

*Generated API documentation. See module-specific files for complete symbol references.*
