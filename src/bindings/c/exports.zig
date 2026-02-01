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
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// ============================================================================
// Core
// ============================================================================

export fn abi_init() ?*anyopaque {
    // In a real implementation, this would initialize the framework using the builder pattern
    // and return a handle to it. For now, we return a dummy handle.
    const handle = allocator.create(u8) catch return null;
    return handle;
}

export fn abi_shutdown(handle: ?*anyopaque) void {
    if (handle) |h| {
        allocator.destroy(@as(*u8, @ptrCast(@alignCast(h))));
    }
}

export fn abi_version() [*:0]const u8 {
    return "0.4.0";
}

// ============================================================================
// Database
// ============================================================================

const DatabaseHandle = struct {
    // This is a placeholder. In a real impl, this would hold the WDBX instance.
    dimension: u32,
    vectors: std.AutoHashMapUnmanaged(u64, []f32),

    pub fn init(dim: u32) !*DatabaseHandle {
        const self = try allocator.create(DatabaseHandle);
        self.* = .{
            .dimension = dim,
            .vectors = .{},
        };
        return self;
    }

    pub fn deinit(self: *DatabaseHandle) void {
        var iter = self.vectors.valueIterator();
        while (iter.next()) |vec| {
            allocator.free(vec.*);
        }
        self.vectors.deinit(allocator);
        allocator.destroy(self);
    }
};

export fn abi_db_create(handle: ?*anyopaque, dimension: u32, db_out: *?*anyopaque) AbiStatus {
    _ = handle;
    const db = DatabaseHandle.init(dimension) catch return .ABI_ERROR_OUT_OF_MEMORY;
    db_out.* = db;
    return .ABI_SUCCESS;
}

export fn abi_db_insert(db_handle: ?*anyopaque, id: u64, vector: [*]const f32, vector_len: usize) AbiStatus {
    if (db_handle == null) return .ABI_ERROR_INVALID_ARGUMENT;
    const db = @as(*DatabaseHandle, @ptrCast(@alignCast(db_handle)));

    if (vector_len != db.dimension) return .ABI_ERROR_INVALID_ARGUMENT;

    const vec_copy = allocator.alloc(f32, vector_len) catch return .ABI_ERROR_OUT_OF_MEMORY;
    @memcpy(vec_copy, vector[0..vector_len]);

    db.vectors.put(allocator, id, vec_copy) catch {
        allocator.free(vec_copy);
        return .ABI_ERROR_OUT_OF_MEMORY;
    };

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
    const db = @as(*DatabaseHandle, @ptrCast(@alignCast(db_handle)));

    if (vector_len != db.dimension) return .ABI_ERROR_INVALID_ARGUMENT;

    // Dummy implementation: return empty or random results
    // In a real implementation, we would call abi.db.search
    _ = vector;
    _ = k;
    _ = ids_out;
    _ = scores_out;

    return .ABI_SUCCESS;
}

export fn abi_db_destroy(db_handle: ?*anyopaque) void {
    if (db_handle) |h| {
        const db = @as(*DatabaseHandle, @ptrCast(@alignCast(h)));
        db.deinit();
    }
}
