const std = @import("std");
const abi = @import("abi");

// ============================================================================
// Types
// ============================================================================

pub const AbiStatus = enum(c_int) {
    ABI_SUCCESS = 0,
    ABI_ERROR_UNKNOWN = 1,
    ABI_ERROR_INVALID_ARGUMENT = 2,
    ABI_ERROR_OUT_OF_MEMORY = 3,
    ABI_ERROR_INITIALIZATION_FAILED = 4,
    ABI_ERROR_NOT_INITIALIZED = 5,
};

// Global allocator for C-allocated resources
// In a real shared library, we might want to use the host's allocator or a stable one.
// GPA is fine for now.
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// We need to keep the framework instance alive if we want to use it globally,
// but the C API passes a handle.
// For `abi_init`, we can return a pointer to a Context struct that holds the framework.

const FrameworkContext = struct {
    // We might wrap the framework builder result here.
    // But `abi.Framework` is the high level entry.
    // Let's assume for now we just use the database module directly as requested by the C API structure (db_create takes a framework handle but doesn't seem to use it heavily yet).
    // Actually, `abi.Framework` might initialize global state.
    
    // For now, let's keep it simple.
    dummy: u8,
};

// ============================================================================
// Core
// ============================================================================

export fn abi_init() ?*anyopaque {
    // Initialize the ABI framework subsystems if needed.
    if (abi.database.init(allocator)) {
        const ctx = allocator.create(FrameworkContext) catch return null;
        ctx.dummy = 0;
        return ctx;
    } else |_| {
        return null;
    }
}

export fn abi_shutdown(handle: ?*anyopaque) void {
    if (handle) |h| {
        abi.database.deinit();
        allocator.destroy(@as(*FrameworkContext, @ptrCast(@alignCast(h))));
    }
}

export fn abi_version() [*:0]const u8 {
    return "0.4.0";
}

// ============================================================================
// Database
// ============================================================================

// We wrap the Zig DatabaseHandle to ensure pointer stability and C-compatibility if needed.
// `abi.database.DatabaseHandle` is a struct (likely), so we need to allocate it on the heap.
const C_DatabaseHandle = struct {
    handle: abi.database.DatabaseHandle,
};

export fn abi_db_create(handle: ?*anyopaque, dimension: u32, db_out: *?*anyopaque) AbiStatus {
    _ = handle; // Framework handle unused for now, but good for future context
    _ = dimension; // WDBX typically infers dimension or sets it via config. 
                   // For now we ignore it in creation and assume the DB handles it dynamically or defaults.
                   // TODO: Pass dimension to wdbx if API supports it.

    // Generate a unique temporary name for the in-memory/embedded DB
    // In a real C API we'd probably want to pass a path.
    // For this contract, we'll use a memory-backed DB if possible or a temp file.
    // `abi.database.open` takes a path.
    const name = std.fmt.allocPrint(allocator, "mem_db_{d}", .{@intFromPtr(&allocator)}) catch return .ABI_ERROR_OUT_OF_MEMORY;
    defer allocator.free(name);

    const db_handle = abi.database.open(allocator, name) catch return .ABI_ERROR_UNKNOWN;
    
    const wrapper = allocator.create(C_DatabaseHandle) catch return .ABI_ERROR_OUT_OF_MEMORY;
    wrapper.handle = db_handle;
    
    db_out.* = wrapper;
    return .ABI_SUCCESS;
}

export fn abi_db_insert(db_handle: ?*anyopaque, id: u64, vector: [*]const f32, vector_len: usize) AbiStatus {
    if (db_handle == null) return .ABI_ERROR_INVALID_ARGUMENT;
    const wrapper = @as(*C_DatabaseHandle, @ptrCast(@alignCast(db_handle)));

    const vec_slice = vector[0..vector_len];
    
    // We pass null metadata for now
    abi.database.insert(&wrapper.handle, id, vec_slice, null) catch return .ABI_ERROR_UNKNOWN;

    return .ABI_SUCCESS;
}

export fn abi_db_search(
    db_handle: ?*anyopaque,
    vector: [*]const f32,
    vector_len: usize,
    k: u32,
    ids_out: [*]u64,
    scores_out: [*]f32,
) AbiStatus {
    if (db_handle == null) return .ABI_ERROR_INVALID_ARGUMENT;
    const wrapper = @as(*C_DatabaseHandle, @ptrCast(@alignCast(db_handle)));

    const vec_slice = vector[0..vector_len];

    const results = abi.database.search(&wrapper.handle, allocator, vec_slice, k) catch return .ABI_ERROR_UNKNOWN;
    defer allocator.free(results);

    // Copy results to output buffers
    // Caller MUST ensure buffers are large enough for k elements (as per C contract typicality)
    var i: usize = 0;
    for (results) |res| {
        if (i >= k) break;
        ids_out[i] = res.id;
        scores_out[i] = res.score;
        i += 1;
    }

    return .ABI_SUCCESS;
}

export fn abi_db_destroy(db_handle: ?*anyopaque) void {
    if (db_handle) |h| {
        const wrapper = @as(*C_DatabaseHandle, @ptrCast(@alignCast(h)));
        abi.database.close(&wrapper.handle);
        allocator.destroy(wrapper);
    }
}
