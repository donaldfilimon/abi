const std = @import("std");

pub const Error = error{
    RuntimeDisabled,
    EngineCreationFailed,
    TaskCreationFailed,
    TaskGroupFailed,
    SchedulingFailed,
    ConcurrencyError,
    MemoryPoolError,
    AlreadyInitialized,
    NotInitialized,
    ModuleDisabled,
    FeatureNotAvailable,
    InvalidOperation,
};

pub const EngineError = Error;
pub const SchedulingError = Error;
pub const ConcurrencyError = Error;
pub const MemoryError = Error;
