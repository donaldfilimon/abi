//! Runtime module
//!
//! Core compute runtime including work items, engine,
//! execution contexts, and result management.

pub const config = @import("config.zig");
pub const workload = @import("workload.zig");
pub const engine = @import("engine.zig");
pub const topology = @import("topology.zig");
pub const benchmark = @import("benchmark.zig");

pub const WorkloadVTable = workload.WorkloadVTable;
pub const WorkItem = workload.WorkItem;
pub const ResultVTable = workload.ResultVTable;
pub const ResultHandle = workload.ResultHandle;
pub const ResultMetadata = engine.ResultMetadata;
pub const Engine = engine.Engine;

pub const ComputeBenchmark = benchmark.ComputeBenchmark;
pub const BenchmarkResult = benchmark.BenchmarkResult;
pub const runBenchmark = benchmark.runBenchmark;
pub const printBenchmarkResults = benchmark.printBenchmarkResults;
pub const MatrixMultBenchmark = benchmark.MatrixMultBenchmark;
pub const MemoryAllocationBenchmark = benchmark.MemoryAllocationBenchmark;
pub const FibonacciBenchmark = benchmark.FibonacciBenchmark;
pub const HashingBenchmark = benchmark.HashingBenchmark;
