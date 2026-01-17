//! # Compute Module (Legacy)
//!
//! **Status:** Legacy re-exports for backward compatibility.
//!
//! **Recommended:** Use `src/runtime/` instead for new code.
//!
//! ## Migration Notice
//!
//! This module now re-exports from `src/runtime/` which contains the consolidated
//! implementation of the compute engine. The runtime module was consolidated as
//! part of the 2026-01-17 refactoring (Phases 1-6).
//!
//! ## New Structure
//!
//! The compute functionality has been reorganized into `src/runtime/`:
//!
//! | Old Location | New Location |
//! |--------------|--------------|
//! | `compute/runtime/` | `runtime/engine/`, `runtime/scheduling/` |
//! | `compute/concurrency/` | `runtime/concurrency/` |
//! | `compute/memory/` | `runtime/memory/` |
//!
//! ## Features (via runtime/)
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
//! **Direct access (new path):**
//! ```zig
//! const runtime = @import("runtime/mod.zig");
//! var engine = try runtime.Engine.init(allocator, .{});
//! ```
//!
//! **Legacy access (still works):**
//! ```zig
//! var engine = try abi.compute.createEngine(allocator, .{});
//! const result = try abi.compute.runTask(&engine, u32, myTask, 1000);
//! ```
//!
//! See [docs/compute.md](../../docs/compute.md) for details.
