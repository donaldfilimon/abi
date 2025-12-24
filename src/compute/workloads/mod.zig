//! Workload definitions
//!
//! Matrix multiplication, neural inference, and other compute kernels.

const std = @import("std");
const runtime_workload = @import("../runtime/workload.zig");

pub const matmul = @import("matmul.zig");
pub const neural_inference = @import("neural_inference.zig");

pub const Matrix = matmul.Matrix;
pub const MatrixMultiplication = matmul.MatrixMultiplication;
pub const NeuralInference = neural_inference.NeuralInference;

pub const WorkloadVTable = runtime_workload.WorkloadVTable;
pub const WorkItem = runtime_workload.WorkItem;
pub const WorkloadHints = runtime_workload.WorkloadHints;
pub const DEFAULT_HINTS = runtime_workload.DEFAULT_HINTS;
pub const ExecutionContext = runtime_workload.ExecutionContext;
pub const ResultVTable = runtime_workload.ResultVTable;
pub const ResultHandle = runtime_workload.ResultHandle;
