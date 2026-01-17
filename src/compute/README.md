//! # Compute Module
//!
//! High-performance compute engine with work-stealing scheduler and concurrency primitives.
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
//! | Module | Description |
//! |--------|-------------|
//! | `runtime/` | Engine, scheduler, NUMA, cancellation, futures |
//! | `concurrency/` | Lock-free queues, work-stealing, priority queues |
//! | `memory/` | Arena allocators, memory pools |
//! | `gpu/` | GPU integration layer and backends |
//! | `network/` | Distributed compute (feature-gated) |
//! | `profiling/` | Metrics collection (feature-gated) |
//!
//! ## Usage
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
//! - [Compute Documentation](../../docs/compute.md)
//! - [Concurrency Primitives](concurrency/README.md)
//! - [GPU Layer](gpu/README.md)

## Contacts

src/shared/contacts.zig provides a centralized list of maintainer contacts extracted from the repository markdown files. Import this module wherever contact information is needed.
