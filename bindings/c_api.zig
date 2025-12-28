//! C API for ABI framework
//!
//! Provides C-compatible functions for Python, JavaScript, and other language bindings.

const std = @import("std");

export const AbiErrorCode = enum(c_int) {
    success = 0,
    out_of_memory = 1,
    invalid_argument = 2,
    not_found = 3,
    timeout = 4,
    queue_full = 5,
    io_error = 6,
    unknown_error = 999,
};

export const AbiLogLevel = enum(c_int) {
    debug = 0,
    info = 1,
    warn = 2,
    err = 3,
    fatal = 4,
};

const AbiConfigInternal = packed struct {
    max_tasks: usize,
    numa_enabled: c_int,
    cpu_affinity_enabled: c_int,
};

export const AbiTaskId = u64;
export const AbiHandle = *anyopaque;

var global_allocator: ?std.mem.Allocator = null;

export fn abi_init(config: AbiConfigInternal) AbiErrorCode {
    _ = config;
    if (global_allocator == null) {
        global_allocator = std.heap.page_allocator;
    }
    return AbiErrorCode.success;
}

export fn abi_deinit() void {
    global_allocator = null;
}

export fn abi_get_cpu_count() usize {
    return std.Thread.getCpuCount();
}

export fn abi_get_numa_node_count() usize {
    return 1;
}

export fn abi_set_thread_affinity(cpu_id: usize) AbiErrorCode {
    _ = cpu_id;
    return AbiErrorCode.io_error;
}

export fn abi_get_current_cpu() AbiErrorCode {
    return AbiErrorCode.io_error;
}

export fn abi_submit_task(task_fn: *const fn (*anyopaque) callconv(.c) *anyopaque, user_data: *anyopaque) AbiTaskId {
    _ = task_fn;
    _ = user_data;
    return 0;
}

export fn abi_wait_result(task_id: AbiTaskId, timeout_ms: u64) AbiErrorCode {
    _ = task_id;
    _ = timeout_ms;
    return AbiErrorCode.success;
}

export fn abi_database_create(name: [*c]const u8) AbiHandle {
    _ = name;
    return null;
}

export fn abi_database_insert(handle: AbiHandle, vector: [*]const f32, dim: usize, metadata: [*c]const u8) AbiErrorCode {
    _ = handle;
    _ = vector;
    _ = dim;
    _ = metadata;
    return AbiErrorCode.success;
}

export fn abi_database_search(handle: AbiHandle, query: [*]const f32, dim: usize, limit: usize) AbiErrorCode {
    _ = handle;
    _ = query;
    _ = dim;
    _ = limit;
    return AbiErrorCode.success;
}

export fn abi_database_destroy(handle: AbiHandle) AbiErrorCode {
    _ = handle;
    return AbiErrorCode.success;
}

export fn abi_string_free(str: [*c]const u8) void {
    if (global_allocator) |alloc| {
        alloc.free(std.mem.sliceTo(str, std.mem.len(str)));
    }
}

export fn abi_get_last_error() [*c]const u8 {
    return "No error";
}

export fn abi_set_log_callback(callback: *const fn (AbiLogLevel, [*c]const u8) callconv(.c) void) void {
    _ = callback;
}
