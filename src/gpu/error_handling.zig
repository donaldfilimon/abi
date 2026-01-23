//! GPU error handling with detailed error reporting.
//!
//! Provides comprehensive error types, formatting, and recovery
//! strategies for GPU operations.

const std = @import("std");
const interface = @import("interface.zig");
const platform_time = @import("../shared/time.zig");

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
    // Additional specific GPU errors
    buffer_overflow = 800,
    no_host_memory = 801,
    no_device_memory = 802,
    synchronization_timeout = 803,
    kernel_compilation_failed = 804,
    kernel_link_failed = 805,
    backend_not_supported = 806,
    device_lost = 807,
    memory_corruption = 808,
    invalid_kernel_arguments = 809,
    stream_synchronization_failed = 810,
    peer_access_already_enabled = 811,
    memory_pool_exhausted = 812,
    cache_miss = 813,
    profiling_not_enabled = 814,
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
    buffer,
    backend,
    cache,
    profiling,
};

/// Operation context for tracking what was being performed when error occurred.
pub const OperationContext = enum {
    none,
    initialization,
    device_query,
    memory_allocation,
    memory_free,
    memory_transfer_h2d,
    memory_transfer_d2h,
    memory_transfer_d2d,
    kernel_compile,
    kernel_launch,
    kernel_destroy,
    synchronization,
    backend_switch,
    failover,
};

pub const GpuError = struct {
    code: GpuErrorCode,
    error_type: GpuErrorType,
    message: []const u8,
    backend: ?[]const u8 = null,
    backend_type: ?interface.BackendType = null,
    device_id: ?i32 = null,
    timestamp: i64 = 0,
    operation: OperationContext = .none,
    /// Native error code from backend (e.g., CUDA error code, Vulkan VkResult)
    native_code: ?i64 = null,
    /// Additional context like kernel name, buffer size, etc.
    extra_context: ?[]const u8 = null,

    pub fn format(
        self: GpuError,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        try writer.print("GPU Error: {t} ({t})", .{
            self.error_type,
            self.code,
        });

        if (self.backend_type) |bt| {
            try writer.print(" [Backend: {s}]", .{bt.name()});
        } else if (self.backend) |b| {
            try writer.print(" [Backend: {s}]", .{b});
        }

        if (self.device_id) |id| {
            try writer.print(" [Device: {d}]", .{id});
        }

        if (self.operation != .none) {
            try writer.print(" [Op: {t}]", .{self.operation});
        }

        if (self.native_code) |nc| {
            try writer.print(" [Native: {d}]", .{nc});
        }

        if (self.message.len > 0) {
            try writer.print(": {s}", .{self.message});
        }

        if (self.extra_context) |ctx| {
            try writer.print(" ({s})", .{ctx});
        }

        if (self.timestamp != 0) {
            try writer.print(" @ {d}", .{self.timestamp});
        }
    }

    /// Create a simple error with minimal context.
    pub fn simple(code: GpuErrorCode, error_type: GpuErrorType, message: []const u8) GpuError {
        return .{
            .code = code,
            .error_type = error_type,
            .message = message,
        };
    }

    /// Create an error with backend context.
    pub fn withBackend(
        code: GpuErrorCode,
        error_type: GpuErrorType,
        message: []const u8,
        backend_type: interface.BackendType,
        device_id: ?i32,
    ) GpuError {
        return .{
            .code = code,
            .error_type = error_type,
            .message = message,
            .backend_type = backend_type,
            .device_id = device_id,
        };
    }

    /// Create a fully detailed error.
    pub fn detailed(
        code: GpuErrorCode,
        error_type: GpuErrorType,
        message: []const u8,
        backend_type: interface.BackendType,
        device_id: ?i32,
        operation: OperationContext,
        native_code: ?i64,
    ) GpuError {
        return .{
            .code = code,
            .error_type = error_type,
            .message = message,
            .backend_type = backend_type,
            .device_id = device_id,
            .operation = operation,
            .native_code = native_code,
        };
    }
};

pub const ErrorContext = struct {
    allocator: std.mem.Allocator,
    errors: std.ArrayListUnmanaged(GpuError),
    max_errors: usize = 100,
    /// Current backend type for automatic context
    current_backend: ?interface.BackendType = null,
    /// Current device ID for automatic context
    current_device: ?i32 = null,

    pub fn init(allocator: std.mem.Allocator) ErrorContext {
        return .{
            .allocator = allocator,
            .errors = std.ArrayListUnmanaged(GpuError).empty,
        };
    }

    /// Initialize with backend context.
    pub fn initWithBackend(
        allocator: std.mem.Allocator,
        backend_type: interface.BackendType,
        device_id: ?i32,
    ) ErrorContext {
        return .{
            .allocator = allocator,
            .errors = std.ArrayListUnmanaged(GpuError).empty,
            .current_backend = backend_type,
            .current_device = device_id,
        };
    }

    pub fn deinit(self: *ErrorContext) void {
        for (self.errors.items) |*err| {
            self.allocator.free(err.message);
            if (err.extra_context) |ctx| {
                self.allocator.free(ctx);
            }
        }
        self.errors.deinit(self.allocator);
        self.* = undefined;
    }

    /// Set the current backend context for automatic error attribution.
    pub fn setBackendContext(self: *ErrorContext, backend_type: interface.BackendType, device_id: ?i32) void {
        self.current_backend = backend_type;
        self.current_device = device_id;
    }

    pub const ReportError = error{
        OutOfMemory,
    };

    pub fn reportError(
        self: *ErrorContext,
        code: GpuErrorCode,
        error_type: GpuErrorType,
        message: []const u8,
    ) ReportError!void {
        try self.reportErrorFull(code, error_type, message, .none, null, null);
    }

    /// Report an error with full context.
    pub fn reportErrorFull(
        self: *ErrorContext,
        code: GpuErrorCode,
        error_type: GpuErrorType,
        message: []const u8,
        operation: OperationContext,
        native_code: ?i64,
        extra_context: ?[]const u8,
    ) ReportError!void {
        if (self.errors.items.len >= self.max_errors) {
            const last_index = self.errors.items.len - 1;
            self.allocator.free(self.errors.items[last_index].message);
            if (self.errors.items[last_index].extra_context) |ctx| {
                self.allocator.free(ctx);
            }
            _ = self.errors.swapRemove(last_index);
        }

        const msg_copy = try self.allocator.dupe(u8, message);
        errdefer self.allocator.free(msg_copy);

        const extra_copy: ?[]const u8 = if (extra_context) |ctx|
            try self.allocator.dupe(u8, ctx)
        else
            null;
        errdefer if (extra_copy) |ctx| self.allocator.free(ctx);

        const timestamp_ms: i64 = @intCast(platform_time.timestampMs());

        const gpu_error = GpuError{
            .code = code,
            .error_type = error_type,
            .message = msg_copy,
            .backend_type = self.current_backend,
            .device_id = self.current_device,
            .operation = operation,
            .native_code = native_code,
            .extra_context = extra_copy,
            .timestamp = timestamp_ms,
        };

        try self.errors.append(self.allocator, gpu_error);
    }

    /// Report a memory error with size context.
    pub fn reportMemoryError(
        self: *ErrorContext,
        code: GpuErrorCode,
        message: []const u8,
        operation: OperationContext,
        size: ?usize,
    ) ReportError!void {
        var buf: [64]u8 = undefined;
        const extra: ?[]const u8 = if (size) |s|
            std.fmt.bufPrint(&buf, "size={d}", .{s}) catch null
        else
            null;
        try self.reportErrorFull(code, .memory, message, operation, null, extra);
    }

    /// Report a kernel error with kernel name context.
    pub fn reportKernelError(
        self: *ErrorContext,
        code: GpuErrorCode,
        message: []const u8,
        operation: OperationContext,
        kernel_name: ?[]const u8,
    ) ReportError!void {
        try self.reportErrorFull(code, .kernel, message, operation, null, kernel_name);
    }

    pub fn getLastError(self: *const ErrorContext) ?GpuError {
        if (self.errors.items.len == 0) return null;
        return self.errors.items[self.errors.items.len - 1];
    }

    pub const GetErrorsByTypeError = error{
        OutOfMemory,
    };

    pub fn getErrorsByType(
        self: *const ErrorContext,
        allocator: std.mem.Allocator,
        error_type: GpuErrorType,
    ) GetErrorsByTypeError![]GpuError {
        var filtered = std.ArrayListUnmanaged(GpuError).empty;
        errdefer filtered.deinit(allocator);

        for (self.errors.items) |err| {
            if (err.error_type == error_type) {
                try filtered.append(allocator, err);
            }
        }

        return filtered.toOwnedSlice(allocator);
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
        .buffer_overflow => "Ensure data size fits within buffer capacity",
        .no_host_memory => "Check host memory availability or enable unified memory",
        .no_device_memory => "Free unused GPU memory or reduce allocation size",
        .synchronization_timeout => "Check for deadlocks or increase timeout values",
        .kernel_compilation_failed => "Review kernel code for syntax errors or unsupported features",
        .kernel_link_failed => "Check kernel dependencies and linking requirements",
        .backend_not_supported => "Try a different GPU backend or check driver compatibility",
        .device_lost => "Reset GPU context or restart the application",
        .memory_corruption => "Check for buffer overflows or race conditions",
        .invalid_kernel_arguments => "Verify kernel argument types and buffer bindings",
        .stream_synchronization_failed => "Check stream dependencies and ordering",
        .peer_access_already_enabled => "Peer access is already enabled for these devices",
        .memory_pool_exhausted => "Increase memory pool size or free unused allocations",
        .cache_miss => "Consider pre-warming cache or adjusting cache policies",
        .profiling_not_enabled => "Enable profiling in GPU configuration",
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
