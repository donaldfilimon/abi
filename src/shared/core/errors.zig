const std = @import("std");

/// Unified error types for the entire framework.
pub const AbiError = error{
    // Core system errors
    SystemNotInitialized,
    SystemAlreadyInitialized,
    InvalidConfiguration,
    ResourceExhausted,

    // Memory errors
    OutOfMemory,
    InvalidAllocation,
    MemoryLeak,
    BufferOverflow,
    BufferUnderflow,

    // I/O errors
    FileNotFound,
    PermissionDenied,
    DiskFull,
    NetworkError,
    Timeout,

    // Validation errors
    InvalidInput,
    InvalidState,
    InvalidOperation,
    DimensionMismatch,

    // Performance errors
    PerformanceThresholdExceeded,
    ResourceLimitExceeded,
    ConcurrencyError,

    // AI/ML specific errors
    ModelNotLoaded,
    InvalidModel,
    TrainingFailed,
    InferenceFailed,

    // Database errors
    DatabaseError,
    IndexError,
    QueryError,
    TransactionError,

    // Miscellaneous
    OperationFailed, // Generic operation failure used by legacy modules
};

/// Alias for legacy compatibility.
pub const FrameworkError = AbiError;

/// Result type for operations that can fail.
pub fn Result(comptime T: type) type {
    return std.meta.Result(T, AbiError);
}

/// Success result helper.
pub fn ok(comptime T: type, value: T) Result(T) {
    return .{ .ok = value };
}

/// Error result helper.
pub fn err(comptime T: type, error_type: AbiError) Result(T) {
    return .{ .err = error_type };
}
