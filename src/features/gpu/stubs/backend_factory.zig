const std = @import("std");
const backend_mod = @import("backend.zig");

pub const BackendFactory = struct {
    pub fn init(_: std.mem.Allocator) @This() {
        return .{};
    }
    pub fn deinit(_: *@This()) void {}
};

pub const BackendInstance = struct {};

pub const BackendFeature = enum { compute, graphics, transfer, sparse };

/// Errors that can occur during backend creation.
pub const FactoryError = error{
    BackendNotAvailable,
    BackendInitializationFailed,
    OutOfMemory,
    UnsupportedBackend,
    NoBackendsAvailable,
};

/// Errors specific to strict backend selection.
pub const BackendSelectionError = error{
    RequestedBackendUnavailable,
    NoBackendsAvailable,
    OutOfMemory,
};

/// Backend selection options.
pub const SelectionOptions = struct {
    preferred: ?backend_mod.Backend = null,
    fallback_chain: []const backend_mod.Backend = &.{},
    required_features: []const BackendFeature = &.{},
    fallback_to_cpu: bool = true,
    /// When true, return `BackendSelectionError.RequestedBackendUnavailable`
    /// instead of falling back to alternatives when the preferred backend
    /// is unavailable. Default: false (keep legacy fallback behaviour).
    strict: bool = false,
};

pub fn createBackend(_: std.mem.Allocator, _: backend_mod.Backend) !BackendInstance {
    return error.GpuDisabled;
}

pub fn createBestBackend(_: std.mem.Allocator) !BackendInstance {
    return error.GpuDisabled;
}

pub fn createBestBackendWithOptions(_: std.mem.Allocator, _: SelectionOptions) !BackendInstance {
    return error.GpuDisabled;
}

pub fn destroyBackend(_: *BackendInstance) void {}

pub fn listAvailableBackends(_: std.mem.Allocator) ![]backend_mod.Backend {
    return error.GpuDisabled;
}

pub fn detectAvailableBackends(_: std.mem.Allocator) ![]backend_mod.Backend {
    return error.GpuDisabled;
}

pub fn isBackendAvailable(_: backend_mod.Backend) bool {
    return false;
}

pub fn selectBestBackendWithFallback(_: std.mem.Allocator, _: SelectionOptions) !?backend_mod.Backend {
    return error.GpuDisabled;
}

pub fn selectBackendWithFeatures(_: std.mem.Allocator, _: SelectionOptions) !?backend_mod.Backend {
    return error.GpuDisabled;
}

test {
    std.testing.refAllDecls(@This());
}
