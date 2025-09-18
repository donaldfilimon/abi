//! Minimal core module for CLI compatibility
//! This provides basic core functionality

const std = @import("std");

// Basic error types
pub const AbiError = error{
    SystemNotInitialized,
    InvalidConfiguration,
    OutOfMemory,
    InvalidInput,
};

// Basic result type
pub fn Result(comptime T: type) type {
    return std.meta.Result(T, AbiError);
}

// Basic initialization
var initialized: bool = false;

pub fn init(allocator: std.mem.Allocator) AbiError!void {
    _ = allocator;
    initialized = true;
}

pub fn deinit() void {
    initialized = false;
}

pub fn isInitialized() bool {
    return initialized;
}

pub fn getAllocator() std.mem.Allocator {
    return std.heap.page_allocator;
}
