//! Compute Runtime Module
//!
//! High-performance compute runtime with SIMD, memory management,
//! concurrency primitives, and workload execution.

const std = @import("std");

pub const simd = @import("simd/mod.zig");
pub const memory = @import("memory/mod.zig");
pub const concurrency = @import("concurrency/mod.zig");
pub const runtime = @import("runtime/mod.zig");
pub const workloads = @import("workloads/mod.zig");

pub const VectorOps = simd.VectorOps;
pub const ComputeVector = simd.ComputeVector;

pub const CacheAlignedBuffer = memory.CacheAlignedBuffer;
pub const PoolAllocator = memory.PoolAllocator;

pub const ChaseLevDeque = concurrency.ChaseLevDeque;
pub const InjectionQueue = concurrency.InjectionQueue;
pub const ShardedMap = concurrency.ShardedMap;

pub const Engine = runtime.Engine;
pub const WorkloadVTable = runtime.WorkloadVTable;
pub const WorkItem = runtime.WorkItem;
pub const ResultHandle = runtime.ResultHandle;
pub const ResultVTable = runtime.ResultVTable;
pub const ExecutionContext = runtime.ExecutionContext;
pub const WorkloadHints = runtime.WorkloadHints;
pub const DEFAULT_HINTS = runtime.DEFAULT_HINTS;
pub const EngineConfig = runtime.config.EngineConfig;
pub const DEFAULT_CONFIG = runtime.config.DEFAULT_CONFIG;
pub const MetricsCollector = runtime.MetricsCollector;
pub const config = runtime.config;

pub const Matrix = workloads.Matrix;
pub const MatrixMultiplication = workloads.MatrixMultiplication;
pub const NeuralInference = workloads.NeuralInference;

pub fn init(allocator: std.mem.Allocator) !void {
    _ = allocator;
}

pub fn deinit() void {}
