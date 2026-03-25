//! Unified Execution Coordinator Facade
//!
//! Provides seamless fallback: GPU → SIMD → scalar
//! Automatically selects the best execution method based on hardware availability,
//! data size, operation type, and user preferences.

const std = @import("std");

pub const coordinator_impl = @import("execution_coordinator/coordinator.zig");
pub const types = @import("execution_coordinator/types.zig");

pub const ExecutionCoordinator = coordinator_impl.ExecutionCoordinator;
pub const ExecutionMethod = types.ExecutionMethod;
pub const CoordinatorConfig = types.CoordinatorConfig;
pub const OperationType = types.OperationType;
pub const PerformanceSample = types.PerformanceSample;
pub const AdaptiveThresholds = types.AdaptiveThresholds;

test {
    std.testing.refAllDecls(@This());
}
