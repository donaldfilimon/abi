# ABI Framework Refactor Design (Zig 0.17.0)

**Date:** 2026-05-14
**Status:** Approved
**Scope:** Full codebase refactor based on WDBX, ABI, and Abbey specifications.

---

## 1. Architectural Vision: Data-Oriented Composition

We are moving from a nested, object-oriented-like structure to a flat, registry-based composition model optimized for Zig 0.17.0.

### 1.1 Core Principles
- **Explicit Memory Management**: Use dedicated allocators (Arena, Pool) for different component lifecycles.
- **Data-Oriented Design**: Organize data (embeddings, blocks) for cache locality and SIMD friendliness.
- **Comptime-Gated Features**: Use the Mod/Stub pattern enforced by `build_options` to allow aggressive tree-shaking.
- **Registry-Based Lifecycle**: A central `Registry` in `src/core/registry.zig` manages initialization and deinitialization of all top-level modules.

---

## 2. Directory Structure

```
src/
├── root.zig           # Public API and feature exports
├── main.zig           # CLI/MCP entry point
├── core/              # Foundational systems
│   ├── registry.zig   # Central component lifecycle management
│   ├── memory.zig     # Custom allocators and pools
│   └── scheduler.zig  # Task-based concurrency
├── foundation/        # OS and primitive abstractions
│   ├── io.zig         # Optimized async IO
│   ├── time.zig       # Unified time
│   └── sync.zig       # RwLock and atomics
├── features/          # Domain-specific modules
│   ├── ai/            # Abbey-Aviva-Abi Pipeline
│   │   ├── mod.zig
│   │   ├── stub.zig
│   │   ├── pipeline.zig
│   │   ├── profiles.zig
│   │   └── constitution/
│   └── wdbx/           # Vector Storage & Block Chain
│       ├── mod.zig
│       ├── stub.zig
│       ├── index/      # HNSW, DiskANN
│       └── storage/    # Block chain, MVCC
└── tests/             # Integration and stress tests
```

---

## 3. WDBX Substrate: HNSW Index (Priority 1)

The Hierarchical Navigable Small World (HNSW) index is the foundation for semantic retrieval.

### 3.1 Data Structures
- **VectorStorage**: A contiguous array of `f32` vectors, optimized for SIMD `@Vector`.
- **Node**: 
  - `id: u32`
  - `neighbors: [MAX_LAYERS]ArrayListUnmanaged(u32)`
- **LayerManager**: Manages the multi-layered graph structure.

### 3.2 Algorithms
- **SIMD Distance**: Cosine similarity using `std.simd` primitives for Zig 0.17.
- **Concurrent Insert**: Lock-free edge updates using `std.atomic.Value`.
- **Heuristic Search**: Optimized neighborhood traversal with priority queues.

---

## 4. WDBX Substrate: Block Chain Memory (Priority 2)

Cryptographically chained conversation blocks for immutable state management.

### 4.1 ConversationBlock
```zig
const ConversationBlock = struct {
    id: [32]u8,          // SHA-256 Hash
    prev_id: [32]u8,     // Reference to parent
    timestamp: i64,      // unixMs
    profile_id: Profile, // Abbey, Aviva, or Abi
    query_v: []f32,      // Query embedding
    response_v: []f32,   // Response embedding
    metadata: Metadata,  // Intent, Policy flags, etc.
};
```

### 4.2 MVCC (Multiversion Concurrency Control)
- Use a persistent, lock-free linked list for the chain.
- Readers acquire a "head" reference to ensure snapshot isolation during retrieval.

---

## 5. AI Pipeline: Abbey-Aviva-Abi

The three-way weighted routing and blending pipeline.

### 5.1 Routing (Abi)
- **Sentiment Analysis**: Fast-path intent detection.
- **Weighting**: Calculate `w_abbey`, `w_aviva`, `w_abi` based on complexity and safety.

### 5.2 Constitution (Governance)
- Six principles: Safety, Honesty, Privacy, Fairness, Autonomy, Transparency.
- Post-generation validation via `evaluateResponse()`.

---

## 6. Implementation Strategy

1. **Foundation**: Implement `core/memory.zig` and `foundation/sync.zig`.
2. **HNSW Substrate**: Implement the core vector index in `features/wdbx/index/hnsw.zig`.
3. **Block Chain**: Implement `features/wdbx/storage/chain.zig`.
4. **AI Pipeline**: Implement the router and profiles.
5. **Integration**: Wire everything into `main.zig` and the MCP server.
