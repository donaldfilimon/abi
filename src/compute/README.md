//! # Compute Module
//!
//! High-performance compute engine with work-stealing scheduler and concurrency primitives.
//!
//! ## Architecture
//!
//! Re-exported by:
//! - `src/runtime/` - Framework-integrated runtime (preferred)
//! - `src/gpu/` - GPU acceleration (separate module)
//!
//! **Use top-level modules for Framework integration. Use `compute/` directly for implementation access.**
//!
//! ## Features
//!
//! - Work-stealing scheduler for efficient task distribution
//! - Lock-free concurrency primitives (queues, stacks, maps)
//! - Memory management (arenas, pools)
//! - NUMA-aware scheduling and CPU affinity
//! - Futures, cancellation, task groups
//!
//! ## Usage
//!
//! **Framework (preferred):**
//! ```zig
//! var fw = try abi.init(allocator);
//! const runtime = fw.getRuntime();
//! var group = try runtime.createTaskGroup(.{});
//! ```
//!
//! **Direct access:**
//! ```zig
//! var engine = try abi.compute.createEngine(allocator, .{});
//! const result = try abi.compute.runTask(&engine, u32, myTask, 1000);
//! ```
//!
//! See [docs/compute.md](../../docs/compute.md) for details.
