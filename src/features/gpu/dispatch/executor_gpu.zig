const coordinator = @import("coordinator.zig");
const std = @import("std");

pub const KernelDispatcher = coordinator.KernelDispatcher;
pub const DispatchError = coordinator.DispatchError;
pub const CompiledKernelHandle = coordinator.CompiledKernelHandle;
pub const KernelArgs = coordinator.KernelArgs;
pub const LaunchConfig = coordinator.LaunchConfig;
pub const ExecutionResult = coordinator.ExecutionResult;

test {
    std.testing.refAllDecls(@This());
}
