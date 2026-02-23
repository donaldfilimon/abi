//! GPU Kernel Dispatch
//!
//! Unified module for GPU kernel dispatch, execution types, and batching.
//!
//! - `types`: Error types, configuration structs, execution results
//! - `coordinator`: Kernel compilation and dispatch to backends
//! - `batch`: Batched small-operation dispatcher
const std = @import("std");

pub const types = @import("types.zig");
pub const coordinator = @import("coordinator.zig");
pub const batch = @import("batch.zig");

// Re-export core dispatch types
pub const DispatchError = types.DispatchError;
pub const CompiledKernelHandle = types.CompiledKernelHandle;
pub const LaunchConfig = types.LaunchConfig;
pub const KernelArgs = types.KernelArgs;
pub const ExecutionResult = types.ExecutionResult;
pub const QueuedLaunch = types.QueuedLaunch;
pub const KernelHandle = types.CompiledKernelHandle;

// Re-export dispatcher
pub const KernelDispatcher = coordinator.KernelDispatcher;
pub const Backend = coordinator.Backend;
pub const Device = coordinator.Device;
pub const Buffer = coordinator.Buffer;
pub const KernelIR = coordinator.KernelIR;
pub const KernelRing = coordinator.KernelRing;

// Re-export batch types
pub const BatchedOp = batch.BatchedOp;
pub const BatchedDispatcher = batch.BatchedDispatcher;

test {
    _ = types;
    _ = coordinator;
    _ = batch;
}

test {
    std.testing.refAllDecls(@This());
}
