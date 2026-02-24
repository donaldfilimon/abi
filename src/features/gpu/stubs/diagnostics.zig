const std = @import("std");

pub const DiagnosticsInfo = struct {
    available: bool = false,
    backend: []const u8 = "none",
    device_count: usize = 0,
};

pub const ErrorContext = struct {
    code: GpuErrorCode = .unknown,
    error_type: GpuErrorType = .runtime,
    message: []const u8 = "GPU disabled",
};

pub const GpuErrorCode = enum {
    unknown,
    out_of_memory,
    device_lost,
    invalid_operation,
    compilation_failed,
};

pub const GpuErrorType = enum {
    runtime,
    compilation,
    resource,
    synchronization,
};

test {
    std.testing.refAllDecls(@This());
}
