//! JavaScript/WebAssembly bindings
//!
//! Provides a JavaScript-compatible interface via Emscripten/WASI
//! for browser and Node.js environments.

const std = @import("std");

export fn abi_init(config_ptr: [*]const u8, config_len: usize) i32 {
    _ = config_ptr;
    _ = config_len;
    return 0;
}

export fn abi_deinit() void {}

export fn abi_get_cpu_count() usize {
    return std.Thread.getCpuCount();
}

export fn abi_get_numa_node_count() usize {
    return 1;
}

export fn abi_set_thread_affinity(cpu_id: usize) i32 {
    _ = cpu_id;
    return 0;
}

export fn abi_get_current_cpu() i32 {
    return 6;
}

export fn abi_submit_task(
    task_fn_ptr: ?*anyopaque,
    user_data_ptr: ?*anyopaque,
) u64 {
    _ = task_fn_ptr;
    _ = user_data_ptr;
    return std.time.nanoTimestamp();
}

export fn abi_wait_result(task_id: u64, timeout_ms: u64) i32 {
    _ = task_id;
    _ = timeout_ms;
    return 0;
}

export fn abi_database_create(name_ptr: [*]const u8, name_len: usize) ?*anyopaque {
    _ = name_ptr;
    _ = name_len;
    return null;
}

export fn abi_database_insert(
    handle: ?*anyopaque,
    vector_ptr: [*]const f32,
    dim: usize,
    metadata_ptr: [*]const u8,
    metadata_len: usize,
) i32 {
    _ = handle;
    _ = vector_ptr;
    _ = dim;
    _ = metadata_ptr;
    _ = metadata_len;
    return 0;
}

export fn abi_database_search(
    handle: ?*anyopaque,
    query_ptr: [*]const f32,
    dim: usize,
    limit: usize,
) i32 {
    _ = handle;
    _ = query_ptr;
    _ = dim;
    _ = limit;
    return 0;
}

export fn abi_database_destroy(handle: ?*anyopaque) i32 {
    _ = handle;
    return 0;
}

export fn abi_string_free(ptr: [*]const u8) void {
    _ = ptr;
}

export fn abi_get_last_error() [*]const u8 {
    return "No error";
}

export fn abi_get_version() [*]const u8 {
    return "0.2.2";
}

export fn abi_get_info_json() [*]const u8 {
    return "{\"cpu_count\":8,\"numa_nodes\":1,\"version\":\"0.2.2\"}";
}

export fn abi_malloc(size: usize) ?*anyopaque {
    return std.heap.page_allocator.rawAlloc(size, @alignOf(usize), @returnAddress()) catch null;
}

export fn abi_free(ptr: ?*anyopaque) void {
    // Note: This is a simplified implementation for WASM.
    // In a real WASM environment, you'd use the WASM allocator.
    _ = ptr;
}

export fn abi_realloc(ptr: ?*anyopaque, new_size: usize) ?*anyopaque {
    const allocator = std.heap.page_allocator;

    if (ptr) |p| {
        const ptr_info = std.heap.rawAllocatorInfo(p);
        const aligned_ptr: [*]u8 = @ptrCast(@alignCast(ptr_info.ptr));
        const aligned_len = ptr_info.len;

        const new_ptr = allocator.realloc(aligned_ptr[0..aligned_len], new_size, @alignOf(usize), @returnAddress()) catch return null;
        return new_ptr;
    }

    return allocator.rawAlloc(new_size, @alignOf(usize), @returnAddress()) catch null;
}

export fn abi_memset(ptr: ?*anyopaque, value: u8, size: usize) void {
    if (ptr) |p| {
        const ptr_info = std.heap.rawAllocatorInfo(p);
        const aligned_ptr: [*]u8 = @ptrCast(@alignCast(ptr_info.ptr));
        const safe_size = @min(size, ptr_info.len);
        @memset(aligned_ptr[0..safe_size], value);
    }
}

export fn abi_memcpy(dst: ?*anyopaque, src: ?*const anyopaque, size: usize) void {
    if (dst) |d| {
        if (src) |s| {
            std.mem.copy(u8, @as([*]u8, @ptrCast(d))[0..size], @as([*const]u8, @ptrCast(s))[0..size]);
        }
    }
}

export fn abi_strlen(ptr: [*]const u8) usize {
    if (ptr) |p| {
        var len: usize = 0;
        while (p[len] != 0) : (len += 1) {}
        return len;
    }
    return 0;
}

export fn abi_strcmp(str1: [*]const u8, str2: [*]const u8) i32 {
    if (str1) |s1| {
        if (str2) |s2| {
            return if (std.mem.orderZ(u8, std.mem.spanZ(s1, abi_strlen(s1)), std.mem.spanZ(s2, abi_strlen(s2))) == .lt) -1 else 1;
        }
    }
    return 0;
}

export fn abi_strcpy(dst: [*]u8, src: [*]const u8, max_len: usize) [*]u8 {
    var i: usize = 0;
    while (i < max_len and src[i] != 0) : (i += 1) {
        dst[i] = src[i];
    }
    dst[i] = 0;
    return dst;
}

export fn abi_vec_add(
    a_ptr: [*]const f32,
    b_ptr: [*]const f32,
    c_ptr: [*]f32,
    n: usize,
) void {
    var i: usize = 0;
    while (i < n) : (i += 1) {
        c_ptr[i] = a_ptr[i] + b_ptr[i];
    }
}

export fn abi_vec_dot(
    a_ptr: [*]const f32,
    b_ptr: [*]const f32,
    n: usize,
) f32 {
    var result: f32 = 0.0;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        result += a_ptr[i] * b_ptr[i];
    }
    return result;
}

export fn abi_vec_l2_norm(
    ptr: [*]const f32,
    n: usize,
) f32 {
    var sum: f32 = 0.0;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        sum += ptr[i] * ptr[i];
    }
    return @sqrt(sum);
}

export fn abi_vec_cosine_similarity(
    a_ptr: [*]const f32,
    b_ptr: [*]const f32,
    n: usize,
) f32 {
    var dot_product: f32 = 0.0;
    var norm_a: f32 = 0.0;
    var norm_b: f32 = 0.0;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        dot_product += a_ptr[i] * b_ptr[i];
        norm_a += a_ptr[i] * a_ptr[i];
        norm_b += b_ptr[i] * b_ptr[i];
    }

    if (norm_a == 0.0 or norm_b == 0.0) {
        return 0.0;
    }

    const similarity = dot_product / (@sqrt(norm_a) * @sqrt(norm_b));
    return similarity;
}
