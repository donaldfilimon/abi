//! # Compute Module
//!
//! High-performance compute engine with work-stealing scheduler and concurrency primitives.
//! This is the implementation layer for runtime and GPU functionality.
//!
//! ## Architecture
//!
//! The compute module provides the underlying implementation that is re-exported by:
//!
//! - `src/runtime/` - Re-exports from `compute/runtime/` with Framework Context
//! - `src/gpu/` - Re-exports from `compute/gpu/` with Framework Context
//!
//! For Framework-integrated usage, prefer the top-level modules. Use compute/
//! directly when you need implementation-level access.
//!
//! ## Features
//!
//! - **Work-Stealing Scheduler**: Efficient task distribution across worker threads
//! - **Concurrency Primitives**: Lock-free queues, stacks, and sharded maps
//! - **Memory Management**: Arena allocators and memory pools
//! - **GPU Integration**: Unified GPU/CPU workload execution
//! - **NUMA Awareness**: Topology detection and thread affinity
//! - **Profiling**: Built-in metrics collection
//!
//! ## Sub-modules
//!
//! | Module | Top-Level | Description |
//! |--------|-----------|-------------|
//! | `runtime/` | `src/runtime/` | Engine, scheduler, NUMA, cancellation, futures |
//! | `concurrency/` | (via runtime) | Lock-free queues, work-stealing, priority queues |
//! | `memory/` | (via runtime) | Arena allocators, memory pools |
//! | `gpu/` | `src/gpu/` | GPU integration layer and backends |
//! | `profiling/` | `src/observability/` | Metrics collection (feature-gated) |
//!
//! ## Usage
//!
//! **Preferred: Using Framework**
//!
//! ```zig
//! const abi = @import("abi");
//!
//! var fw = try abi.init(allocator);
//! defer fw.deinit();
//!
//! // Get runtime context (always available)
//! const runtime = fw.getRuntime();
//!
//! // Create task groups for parallel work
//! var group = try runtime.createTaskGroup(.{});
//! defer group.deinit();
//! ```
//!
//! **Direct access (advanced)**
//!
//! ```zig
//! const abi = @import("abi");
//!
//! var engine = try abi.compute.createDefaultEngine(allocator);
//! defer engine.deinit();
//!
//! fn myTask(_: std.mem.Allocator) !u32 {
//!     return 42;
//! }
//!
//! const result = try abi.compute.runTask(&engine, u32, myTask, 1000);
//! ```
//!
//! ## Configuration
//!
//! ```zig
//! const config = abi.compute.EngineConfig{
//!     .worker_count = 8,
//!     .numa_enabled = true,
//!     .cpu_affinity_enabled = true,
//! };
//! var engine = try abi.compute.createEngine(allocator, config);
//! ```
//!
//! ## See Also
//!
//! - [src/runtime/](../runtime/) - Framework-integrated runtime module
//! - [src/gpu/](../gpu/) - Framework-integrated GPU module
//! - [Compute Documentation](../../docs/compute.md)
//! - [Concurrency Primitives](concurrency/README.md)
//! - [GPU Layer](gpu/README.md)
