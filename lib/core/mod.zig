//! Core Module
//!
//! Fundamental types, utilities, and patterns used throughout the framework

pub const collections = @import("collections.zig");
pub const types = @import("types.zig");
pub const allocators = @import("allocators.zig");
pub const errors = @import("errors.zig");

// Re-export commonly used types for convenience
pub const ArrayList = collections.ArrayList;
pub const StringHashMap = collections.StringHashMap;
pub const AutoHashMap = collections.AutoHashMap;
pub const ArenaAllocator = collections.ArenaAllocator;

pub const ErrorCode = types.ErrorCode;
pub const Result = types.Result;
pub const Version = types.Version;

pub const Error = errors.Error;
pub const ErrorContext = errors.ErrorContext;

pub const AllocationStrategy = allocators.AllocationStrategy;
pub const AllocatorConfig = allocators.AllocatorConfig;
pub const TrackedAllocator = allocators.TrackedAllocator;
pub const AllocatorFactory = allocators.AllocatorFactory;

test {
    std.testing.refAllDecls(@This());
}

const std = @import("std");
