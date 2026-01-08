//! GPU error handling with detailed error reporting.
//!
//! Provides comprehensive error types, formatting, and recovery
//! strategies for GPU operations.

const std = @import("std");

pub const GpuErrorCode = enum(u32) {
    success = 0,
    invalid_value = 1,
    out_of_memory = 2,
    not_initialized = 3,
    invalid_device = 101,
    invalid_context = 201,
    invalid_handle = 400,
    illegal_address = 700,
    launch_failure = 701,
    launch_out_of_resources = 702,
    launch_timeout = 704,
    launch_incompatible_texturing = 705,
    peer_access_not_enabled = 708,
    invalid_pctx = 716,
    invalid_resource_handle = 717,
    invalid_configuration = 723,
    invalid_operation = 724,
    unknown = 9999999,
};

pub const GpuErrorType = enum {
    initialization,
    device,
    memory,
    kernel,
    stream,
    runtime,
    driver,
    compilation,
    launch,
    synchronization,
};

pub const GpuError = struct {
    code: GpuErrorCode,
    error_type: GpuErrorType,
    message: []const u8,
    backend: ?[]const u8 = null,
    device_id: ?i32 = null,
    timestamp: i64 = 0,

    pub fn format(
        self: GpuError,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        try writer.print("GPU Error: {s} ({t})", .{
            self.error_type,
            self.code,
        });

        if (self.backend) |b| {
            try writer.print(" [Backend: {s}]", .{b});
        }

        if (self.device_id) |id| {
            try writer.print(" [Device: {d}]", .{id});
        }

        if (self.message.len > 0) {
            try writer.print(": {s}", .{self.message});
        }

        if (self.timestamp != 0) {
            try writer.print(" @ {d}", .{self.timestamp});
        }
    }
};

pub const ErrorContext = struct {
    allocator: std.mem.Allocator,
    errors: std.ArrayListUnmanaged(GpuError),
    max_errors: usize = 100,

    pub fn init(allocator: std.mem.Allocator) ErrorContext {
        return .{
            .allocator = allocator,
            .errors = std.ArrayListUnmanaged(GpuError).empty,
        };
    }

    pub fn deinit(self: *ErrorContext) void {
        for (self.errors.items) |*err| {
            self.allocator.free(err.message);
        }
        self.errors.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn reportError(
        self: *ErrorContext,
        code: GpuErrorCode,
        error_type: GpuErrorType,
        message: []const u8,
    ) !void {
        if (self.errors.items.len >= self.max_errors) {
            try self.errors.removeOrError(self.errors.items.len - 1);
        }

        const msg_copy = try self.allocator.dupe(u8, message);
        errdefer self.allocator.free(msg_copy);

        const timestamp = std.time.timestamp();

        const gpu_error = GpuError{
            .code = code,
            .error_type = error_type,
            .message = msg_copy,
            .timestamp = timestamp,
        };

        try self.errors.append(self.allocator, gpu_error);
    }

    pub fn getLastError(self: *const ErrorContext) ?GpuError {
        if (self.errors.items.len == 0) return null;
        return self.errors.items[self.errors.items.len - 1];
    }

    pub fn getErrorsByType(
        self: *const ErrorContext,
        error_type: GpuErrorType,
    ) []const GpuError {
        var filtered = std.ArrayListUnmanaged(GpuError).empty;

        for (self.errors.items) |err| {
            if (err.error_type == error_type) {
                filtered.append(self.allocator, err) catch continue;
            }
        }

        return filtered.toSlice(self.allocator);
    }

    pub fn clear(self: *ErrorContext) void {
        for (self.errors.items) |*err| {
            self.allocator.free(err.message);
        }
        self.errors.clearRetainingCapacity();
    }

    pub fn getErrorCount(self: *const ErrorContext) usize {
        return self.errors.items.len;
    }

    pub fn getErrorStatistics(self: *const ErrorContext) ErrorStatistics {
        var stats = ErrorStatistics{
            .total = self.errors.items.len,
            .initialization = 0,
            .device = 0,
            .memory = 0,
            .kernel = 0,
            .stream = 0,
            .runtime = 0,
            .driver = 0,
            .compilation = 0,
            .launch = 0,
            .synchronization = 0,
        };

        for (self.errors.items) |err| {
            switch (err.error_type) {
                .initialization => stats.initialization += 1,
                .device => stats.device += 1,
                .memory => stats.memory += 1,
                .kernel => stats.kernel += 1,
                .stream => stats.stream += 1,
                .runtime => stats.runtime += 1,
                .driver => stats.driver += 1,
                .compilation => stats.compilation += 1,
                .launch => stats.launch += 1,
                .synchronization => stats.synchronization += 1,
            }
        }

        return stats;
    }
};

pub const ErrorStatistics = struct {
    total: usize,
    initialization: usize,
    device: usize,
    memory: usize,
    kernel: usize,
    stream: usize,
    runtime: usize,
    driver: usize,
    compilation: usize,
    launch: usize,
    synchronization: usize,

    pub fn format(
        self: ErrorStatistics,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        try writer.print("GPU Error Statistics:\n", .{});
        try writer.print("  Total: {d}\n", .{self.total});
        try writer.print("  Initialization: {d}\n", .{self.initialization});
        try writer.print("  Device: {d}\n", .{self.device});
        try writer.print("  Memory: {d}\n", .{self.memory});
        try writer.print("  Kernel: {d}\n", .{self.kernel});
        try writer.print("  Stream: {d}\n", .{self.stream});
        try writer.print("  Runtime: {d}\n", .{self.runtime});
        try writer.print("  Driver: {d}\n", .{self.driver});
        try writer.print("  Compilation: {d}\n", .{self.compilation});
        try writer.print("  Launch: {d}\n", .{self.launch});
        try writer.print("  Synchronization: {d}\n", .{self.synchronization});
    }
};

pub fn mapCudaResult(result: i32) GpuErrorCode {
    return switch (result) {
        0 => .success,
        1 => .invalid_value,
        2 => .out_of_memory,
        3 => .not_initialized,
        101 => .invalid_device,
        201 => .invalid_context,
        400 => .invalid_handle,
        700 => .illegal_address,
        701 => .launch_failure,
        702 => .launch_out_of_resources,
        704 => .launch_timeout,
        705 => .launch_incompatible_texturing,
        708 => .peer_access_not_enabled,
        716 => .invalid_pctx,
        717 => .invalid_resource_handle,
        723 => .invalid_configuration,
        724 => .invalid_operation,
        else => .unknown,
    };
}

pub fn isSuccess(code: GpuErrorCode) bool {
    return code == .success;
}

pub fn getRecoverySuggestion(code: GpuErrorCode) []const u8 {
    return switch (code) {
        .out_of_memory => "Try reducing memory allocation size or clearing unused memory",
        .invalid_device => "Check that the GPU is properly installed and accessible",
        .not_initialized => "Call initialization functions before using GPU resources",
        .invalid_context => "Verify that the GPU context was created successfully",
        .launch_out_of_resources => "Reduce resource usage (registers, shared memory, threads)",
        .launch_timeout => "Check for deadlocks or kernel hangs",
        .illegal_address => "Verify all memory addresses are valid and allocated",
        .invalid_configuration => "Review kernel launch parameters and grid/block dimensions",
        else => "See documentation for detailed error information",
    };
}

test "error context tracks errors" {
    const allocator = std.testing.allocator;
    var ctx = ErrorContext.init(allocator);
    defer ctx.deinit();

    try ctx.reportError(.out_of_memory, .memory, "Failed to allocate GPU memory");
    try ctx.reportError(.launch_failure, .launch, "Kernel launch failed");

    try std.testing.expectEqual(@as(usize, 2), ctx.getErrorCount());

    const last_err = ctx.getLastError().?;
    try std.testing.expectEqual(GpuErrorType.launch, last_err.error_type);
}

test "error statistics categorize correctly" {
    const allocator = std.testing.allocator;
    var ctx = ErrorContext.init(allocator);
    defer ctx.deinit();

    try ctx.reportError(.out_of_memory, .memory, "Memory error 1");
    try ctx.reportError(.out_of_memory, .memory, "Memory error 2");
    try ctx.reportError(.launch_failure, .launch, "Launch error");

    const stats = ctx.getErrorStatistics();
    try std.testing.expectEqual(@as(usize, 3), stats.total);
    try std.testing.expectEqual(@as(usize, 2), stats.memory);
    try std.testing.expectEqual(@as(usize, 1), stats.launch);
}
