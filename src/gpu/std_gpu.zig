//! Zig std.gpu Integration Module
//!
//! Provides direct access to Zig 0.16's std.gpu facilities for GPU compute,
//! including address spaces, shader built-ins, and SPIR-V code generation.
//!
//! ## Overview
//!
//! This module bridges the gap between our DSL-based GPU abstraction and
//! Zig's native GPU support, enabling:
//!
//! - Direct use of GPU address spaces (global, shared, constant, etc.)
//! - Access to SPIR-V built-in shader variables
//! - Compute kernel definition using `callconv(.spirv_kernel)`
//! - Fragment shader support via `callconv(.spirv_fragment)`
//!
//! ## Address Space Mapping
//!
//! | DSL AddressSpace | Zig AddressSpace | SPIR-V Storage Class |
//! |------------------|------------------|----------------------|
//! | private          | generic          | Function             |
//! | workgroup        | shared           | Workgroup            |
//! | storage          | storage_buffer   | StorageBuffer        |
//! | uniform          | uniform          | Uniform              |
//!
//! ## Example: Compute Kernel (GPU target only)
//!
//! ```zig
//! const std_gpu = @import("std_gpu.zig");
//!
//! // Define a compute kernel with explicit workgroup size
//! fn vectorAdd(
//!     a: std_gpu.GlobalPtr(f32),
//!     b: std_gpu.GlobalPtr(f32),
//!     result: std_gpu.GlobalPtr(f32),
//! ) callconv(.spirv_kernel) void {
//!     const gid = std_gpu.globalInvocationId();
//!     result[gid[0]] = a[gid[0]] + b[gid[0]];
//! }
//!
//! comptime {
//!     std_gpu.setLocalSize(vectorAdd, 256, 1, 1);
//! }
//! ```

const std = @import("std");
const builtin = @import("builtin");
const dsl_types = @import("dsl/types.zig");

// ============================================================================
// Target Detection
// ============================================================================

/// Check if we're compiling for a GPU target (SPIR-V)
pub const is_gpu_target = builtin.cpu.arch.isSpirV();

/// Check if std.gpu is available in the current Zig version
pub const std_gpu_available = @hasDecl(std, "gpu");

// ============================================================================
// Address Space Types
// ============================================================================

/// Zig's builtin address spaces for GPU memory
pub const AddressSpace = std.builtin.AddressSpace;

/// Pointer to global (device) memory
/// On non-GPU targets, returns a regular pointer for compatibility
pub fn GlobalPtr(comptime T: type) type {
    if (is_gpu_target) {
        return *addrspace(.global) T;
    }
    return *T;
}

/// Const pointer to global (device) memory
pub fn GlobalConstPtr(comptime T: type) type {
    if (is_gpu_target) {
        return *addrspace(.global) const T;
    }
    return *const T;
}

/// Pointer to shared (workgroup) memory
pub fn SharedPtr(comptime T: type) type {
    if (is_gpu_target) {
        return *addrspace(.shared) T;
    }
    return *T;
}

/// Const pointer to shared (workgroup) memory
pub fn SharedConstPtr(comptime T: type) type {
    if (is_gpu_target) {
        return *addrspace(.shared) const T;
    }
    return *const T;
}

/// Pointer to constant (uniform) memory
pub fn ConstantPtr(comptime T: type) type {
    if (is_gpu_target) {
        return *addrspace(.constant) const T;
    }
    return *const T;
}

/// Pointer to uniform buffer memory
pub fn UniformPtr(comptime T: type) type {
    if (is_gpu_target) {
        return *addrspace(.uniform) const T;
    }
    return *const T;
}

/// Pointer to storage buffer memory
pub fn StoragePtr(comptime T: type) type {
    if (is_gpu_target) {
        return *addrspace(.storage_buffer) T;
    }
    return *T;
}

/// Const pointer to storage buffer memory
pub fn StorageConstPtr(comptime T: type) type {
    if (is_gpu_target) {
        return *addrspace(.storage_buffer) const T;
    }
    return *const T;
}

/// Pointer to push constant memory
pub fn PushConstantPtr(comptime T: type) type {
    if (is_gpu_target) {
        return *addrspace(.push_constant) const T;
    }
    return *const T;
}

// ============================================================================
// Address Space Conversion
// ============================================================================

/// Convert DSL AddressSpace to Zig's builtin AddressSpace
pub fn dslToZigAddressSpace(dsl_space: dsl_types.AddressSpace) AddressSpace {
    return switch (dsl_space) {
        .private => .generic,
        .workgroup => .shared,
        .storage => .storage_buffer,
        .uniform => .uniform,
    };
}

/// Convert Zig's AddressSpace to DSL AddressSpace (best effort)
pub fn zigToDslAddressSpace(zig_space: AddressSpace) ?dsl_types.AddressSpace {
    return switch (zig_space) {
        .generic => .private,
        .shared => .workgroup,
        .storage_buffer => .storage,
        .uniform => .uniform,
        .constant => .uniform, // Map constant to uniform (read-only)
        .global => .storage, // Map global to storage
        else => null, // No DSL equivalent
    };
}

// ============================================================================
// Shader Built-in Variables (SPIR-V)
// ============================================================================

/// Get the global invocation ID (3D thread index across all workgroups)
/// On non-GPU targets, returns zeros (for testing)
pub inline fn globalInvocationId() @Vector(3, u32) {
    if (is_gpu_target and std_gpu_available) {
        return std.gpu.global_invocation_id;
    }
    // Fallback for non-GPU context (testing)
    return .{ 0, 0, 0 };
}

/// Get the workgroup ID (which workgroup this thread belongs to)
pub inline fn workgroupId() @Vector(3, u32) {
    if (is_gpu_target and std_gpu_available) {
        return std.gpu.workgroup_id;
    }
    return .{ 0, 0, 0 };
}

/// Get the local invocation ID (thread index within workgroup)
pub inline fn localInvocationId() @Vector(3, u32) {
    if (is_gpu_target and std_gpu_available) {
        return std.gpu.local_invocation_id;
    }
    return .{ 0, 0, 0 };
}

/// Get the workgroup size
pub inline fn workgroupSize() @Vector(3, u32) {
    if (is_gpu_target and std_gpu_available) {
        return std.gpu.workgroup_size;
    }
    return .{ 1, 1, 1 };
}

/// Get the number of workgroups
pub inline fn numWorkgroups() @Vector(3, u32) {
    if (is_gpu_target and std_gpu_available) {
        return std.gpu.num_workgroups;
    }
    return .{ 1, 1, 1 };
}

/// Get vertex index (vertex shader)
pub inline fn vertexIndex() u32 {
    if (is_gpu_target and std_gpu_available) {
        return std.gpu.vertex_index;
    }
    return 0;
}

/// Get instance index (vertex shader)
pub inline fn instanceIndex() u32 {
    if (is_gpu_target and std_gpu_available) {
        return std.gpu.instance_index;
    }
    return 0;
}

/// Get fragment coordinates (fragment shader)
pub inline fn fragCoord() @Vector(4, f32) {
    if (is_gpu_target and std_gpu_available) {
        return std.gpu.frag_coord;
    }
    return .{ 0, 0, 0, 1 };
}

/// Get point coordinates (fragment shader)
pub inline fn pointCoord() @Vector(2, f32) {
    if (is_gpu_target and std_gpu_available) {
        return std.gpu.point_coord;
    }
    return .{ 0, 0 };
}

// ============================================================================
// Execution Mode Configuration
// ============================================================================

/// Local workgroup size configuration
pub const LocalSize = struct { x: u32, y: u32, z: u32 };

/// Execution mode for compute and fragment shaders
pub const ExecutionMode = union(Tag) {
    /// Sets origin of the framebuffer to the upper-left corner
    origin_upper_left,
    /// Sets origin of the framebuffer to the lower-left corner
    origin_lower_left,
    /// Indicates that the fragment shader writes to `frag_depth`
    depth_replacing,
    /// Fragment depth >= interpolated depth
    depth_greater,
    /// Fragment depth <= interpolated depth
    depth_less,
    /// Fragment depth unchanged
    depth_unchanged,
    /// Workgroup size in x, y, z dimensions
    local_size: LocalSize,

    pub const Tag = enum(u32) {
        origin_upper_left = 7,
        origin_lower_left = 8,
        depth_replacing = 12,
        depth_greater = 14,
        depth_less = 15,
        depth_unchanged = 16,
        local_size = 17,
    };
};

/// Set the local (workgroup) size for a compute kernel
/// Call this in a comptime block after defining your kernel
pub fn setLocalSize(comptime kernel_fn: anytype, comptime x: u32, comptime y: u32, comptime z: u32) void {
    if (is_gpu_target and std_gpu_available) {
        std.gpu.executionMode(kernel_fn, .{ .local_size = .{ .x = x, .y = y, .z = z } });
    }
}

/// Set execution mode for a shader entry point
pub fn setExecutionMode(comptime entry_point: anytype, comptime mode: ExecutionMode) void {
    if (is_gpu_target and std_gpu_available) {
        // Convert our ExecutionMode to std.gpu.ExecutionMode
        const gpu_mode: std.gpu.ExecutionMode = switch (mode) {
            .origin_upper_left => .origin_upper_left,
            .origin_lower_left => .origin_lower_left,
            .depth_replacing => .depth_replacing,
            .depth_greater => .depth_greater,
            .depth_less => .depth_less,
            .depth_unchanged => .depth_unchanged,
            .local_size => |size| .{ .local_size = .{ .x = size.x, .y = size.y, .z = size.z } },
        };
        std.gpu.executionMode(entry_point, gpu_mode);
    }
}

// ============================================================================
// Workgroup Synchronization
// ============================================================================

/// Barrier synchronizing all threads in a workgroup
/// Ensures all workgroup memory accesses before the barrier are visible
/// to all threads in the workgroup after the barrier
pub inline fn workgroupBarrier() void {
    if (is_gpu_target) {
        // SPIR-V: OpControlBarrier Workgroup Workgroup AcquireRelease|WorkgroupMemory
        asm volatile ("OpControlBarrier %[workgroup] %[workgroup] %[semantics]"
            :
            : [workgroup] "c" (@as(u32, 2)), // Workgroup scope
              [semantics] "c" (@as(u32, 0x108)), // AcquireRelease | WorkgroupMemory
        );
    }
}

/// Memory barrier for workgroup (shared) memory
pub inline fn workgroupMemoryBarrier() void {
    if (is_gpu_target) {
        asm volatile ("OpMemoryBarrier %[workgroup] %[semantics]"
            :
            : [workgroup] "c" (@as(u32, 2)), // Workgroup scope
              [semantics] "c" (@as(u32, 0x100)), // WorkgroupMemory
        );
    }
}

/// Memory barrier for device (global) memory
pub inline fn deviceMemoryBarrier() void {
    if (is_gpu_target) {
        asm volatile ("OpMemoryBarrier %[device] %[semantics]"
            :
            : [device] "c" (@as(u32, 1)), // Device scope
              [semantics] "c" (@as(u32, 0x40)), // UniformMemory
        );
    }
}

// ============================================================================
// Compute Helpers
// ============================================================================

/// Calculate linear global thread index from 3D global invocation ID
pub inline fn getGlobalIndex() u32 {
    const gid = globalInvocationId();
    const num_wg = numWorkgroups();
    const wg_size = workgroupSize();
    const grid_size_x = num_wg[0] * wg_size[0];
    const grid_size_y = num_wg[1] * wg_size[1];
    return gid[0] + gid[1] * grid_size_x + gid[2] * grid_size_x * grid_size_y;
}

/// Calculate linear local thread index within workgroup
pub inline fn getLocalIndex() u32 {
    const lid = localInvocationId();
    const wg_size = workgroupSize();
    return lid[0] + lid[1] * wg_size[0] + lid[2] * wg_size[0] * wg_size[1];
}

/// Grid stride loop helper for processing large arrays
/// Returns the starting index and stride for the current thread
pub fn gridStrideLoop(total_elements: u32) struct { start: u32, stride: u32 } {
    _ = total_elements; // Used for bounds checking at call site
    const gid = globalInvocationId();
    const num_wg = numWorkgroups();
    const wg_size = workgroupSize();
    const total_threads = num_wg[0] * wg_size[0];
    return .{
        .start = gid[0],
        .stride = total_threads,
    };
}

// ============================================================================
// Atomic Operations (Zig 0.16 compatible)
// ============================================================================

/// Atomic add for u32 in device scope
pub inline fn atomicAddU32(ptr: *u32, value: u32) u32 {
    return @atomicRmw(u32, ptr, .Add, value, .acq_rel);
}

/// Atomic add for i32 in device scope
pub inline fn atomicAddI32(ptr: *i32, value: i32) i32 {
    return @atomicRmw(i32, ptr, .Add, value, .acq_rel);
}

/// Atomic add for f32 (uses CAS loop if not natively supported)
pub inline fn atomicAddF32(ptr: *f32, value: f32) f32 {
    var current = @atomicLoad(f32, ptr, .acquire);
    while (true) {
        const new_value = current + value;
        const result = @cmpxchgWeak(
            f32,
            ptr,
            current,
            new_value,
            .acq_rel,
            .acquire,
        );
        if (result) |actual| {
            current = actual;
        } else {
            return current;
        }
    }
}

/// Atomic max for u32
pub inline fn atomicMaxU32(ptr: *u32, value: u32) u32 {
    return @atomicRmw(u32, ptr, .Max, value, .acq_rel);
}

/// Atomic min for u32
pub inline fn atomicMinU32(ptr: *u32, value: u32) u32 {
    return @atomicRmw(u32, ptr, .Min, value, .acq_rel);
}

/// Atomic exchange
pub inline fn atomicExchange(comptime T: type, ptr: *T, value: T) T {
    return @atomicRmw(T, ptr, .Xchg, value, .acq_rel);
}

/// Atomic compare and exchange
pub inline fn atomicCompareExchange(comptime T: type, ptr: *T, expected: T, desired: T) ?T {
    return @cmpxchgStrong(T, ptr, expected, desired, .acq_rel, .acquire);
}

// ============================================================================
// Type Utilities
// ============================================================================

/// Check if a type is a GPU pointer type (has address space)
pub fn isGpuPointer(comptime T: type) bool {
    const info = @typeInfo(T);
    if (info != .pointer) return false;
    const addr_space = info.pointer.address_space;
    return addr_space == .global or
        addr_space == .shared or
        addr_space == .constant or
        addr_space == .uniform or
        addr_space == .storage_buffer or
        addr_space == .push_constant;
}

/// Get the address space of a pointer type
pub fn getAddressSpace(comptime T: type) ?AddressSpace {
    const info = @typeInfo(T);
    if (info != .pointer) return null;
    return info.pointer.address_space;
}

/// Get the element type of a pointer
pub fn PointerChild(comptime T: type) type {
    const info = @typeInfo(T);
    if (info != .pointer) @compileError("Expected pointer type");
    return info.pointer.child;
}

// ============================================================================
// Tests
// ============================================================================

test "address space conversion" {
    // DSL to Zig
    try std.testing.expectEqual(AddressSpace.generic, dslToZigAddressSpace(.private));
    try std.testing.expectEqual(AddressSpace.shared, dslToZigAddressSpace(.workgroup));
    try std.testing.expectEqual(AddressSpace.storage_buffer, dslToZigAddressSpace(.storage));
    try std.testing.expectEqual(AddressSpace.uniform, dslToZigAddressSpace(.uniform));

    // Zig to DSL
    try std.testing.expectEqual(dsl_types.AddressSpace.private, zigToDslAddressSpace(.generic).?);
    try std.testing.expectEqual(dsl_types.AddressSpace.workgroup, zigToDslAddressSpace(.shared).?);
    try std.testing.expectEqual(dsl_types.AddressSpace.storage, zigToDslAddressSpace(.storage_buffer).?);
    try std.testing.expectEqual(dsl_types.AddressSpace.uniform, zigToDslAddressSpace(.uniform).?);
}

test "pointer type utilities on non-GPU target" {
    // On non-GPU targets, pointer helpers return regular pointers
    const GlobalF32 = GlobalPtr(f32);
    const info = @typeInfo(GlobalF32);
    try std.testing.expect(info == .pointer);
    try std.testing.expectEqual(f32, PointerChild(GlobalF32));

    // SharedPtr also returns regular pointer on non-GPU
    const SharedU32 = SharedPtr(u32);
    try std.testing.expectEqual(u32, PointerChild(SharedU32));

    // StoragePtr also returns regular pointer on non-GPU
    const StorageI32 = StoragePtr(i32);
    try std.testing.expectEqual(i32, PointerChild(StorageI32));

    // Regular pointer check
    const RegularPtr = *f32;
    const reg_info = @typeInfo(RegularPtr);
    try std.testing.expect(reg_info == .pointer);

    // Non-pointer check
    try std.testing.expectEqual(@as(?AddressSpace, null), getAddressSpace(f32));
}

test "shader built-in fallbacks" {
    // These should return zeros in non-GPU context
    const gid = globalInvocationId();
    try std.testing.expectEqual(@as(u32, 0), gid[0]);
    try std.testing.expectEqual(@as(u32, 0), gid[1]);
    try std.testing.expectEqual(@as(u32, 0), gid[2]);

    const wid = workgroupId();
    try std.testing.expectEqual(@as(u32, 0), wid[0]);

    const lid = localInvocationId();
    try std.testing.expectEqual(@as(u32, 0), lid[0]);

    try std.testing.expectEqual(@as(u32, 0), getGlobalIndex());
    try std.testing.expectEqual(@as(u32, 0), getLocalIndex());
}

test "grid stride loop" {
    const loop_info = gridStrideLoop(1000);
    try std.testing.expectEqual(@as(u32, 0), loop_info.start);
    // In non-GPU context, stride is 1 (1x1x1 workgroup)
    try std.testing.expectEqual(@as(u32, 1), loop_info.stride);
}

test "atomic operations" {
    var value: u32 = 10;
    const old = atomicAddU32(&value, 5);
    try std.testing.expectEqual(@as(u32, 10), old);
    try std.testing.expectEqual(@as(u32, 15), value);

    var max_val: u32 = 5;
    _ = atomicMaxU32(&max_val, 10);
    try std.testing.expectEqual(@as(u32, 10), max_val);

    _ = atomicMaxU32(&max_val, 3);
    try std.testing.expectEqual(@as(u32, 10), max_val);
}

test "execution mode types" {
    // Test that ExecutionMode enum works
    const mode = ExecutionMode{ .local_size = .{ .x = 256, .y = 1, .z = 1 } };
    try std.testing.expectEqual(@as(u32, 256), mode.local_size.x);
    try std.testing.expectEqual(@as(u32, 1), mode.local_size.y);
    try std.testing.expectEqual(@as(u32, 1), mode.local_size.z);

    // Test tag values match SPIR-V spec
    try std.testing.expectEqual(@as(u32, 17), @intFromEnum(ExecutionMode.Tag.local_size));
    try std.testing.expectEqual(@as(u32, 7), @intFromEnum(ExecutionMode.Tag.origin_upper_left));
}

test "is_gpu_target detection" {
    // On x86/x64, this should be false
    try std.testing.expect(!is_gpu_target);
}

// ============================================================================
// Subgroup Operations (Zig 0.16 std.gpu)
// ============================================================================

/// Subgroup size (warp/wavefront size)
pub inline fn subgroupSize() u32 {
    if (is_gpu_target and std_gpu_available) {
        return std.gpu.subgroup_size;
    }
    // Return typical value for testing
    return 32;
}

/// Subgroup invocation ID (lane ID within subgroup)
pub inline fn subgroupInvocationId() u32 {
    if (is_gpu_target and std_gpu_available) {
        return std.gpu.subgroup_local_invocation_id;
    }
    return 0;
}

/// Subgroup ballot - returns mask of threads where predicate is true
pub inline fn subgroupBallot(predicate: bool) u64 {
    if (is_gpu_target) {
        // SPIR-V ballot instruction
        var result: u64 = 0;
        asm volatile ("OpGroupNonUniformBallot %[result] %[scope] %[pred]"
            : [result] "=r" (result),
            : [scope] "c" (@as(u32, 3)), // Subgroup scope
              [pred] "r" (predicate),
        );
        return result;
    }
    return if (predicate) 1 else 0;
}

/// Subgroup broadcast - broadcast value from leader lane to all lanes
/// On non-GPU targets, lane_id is ignored and value is returned directly.
pub const subgroupBroadcast = if (is_gpu_target) subgroupBroadcastGpu else subgroupBroadcastCpu;

fn subgroupBroadcastGpu(comptime T: type, value: T, lane_id: u32) T {
    var result: T = undefined;
    asm volatile ("OpGroupNonUniformBroadcast %[result] %[scope] %[value] %[lane]"
        : [result] "=r" (result),
        : [scope] "c" (@as(u32, 3)), // Subgroup scope
          [value] "r" (value),
          [lane] "r" (lane_id),
    );
    return result;
}

fn subgroupBroadcastCpu(comptime T: type, value: T, lane_id: u32) T {
    _ = lane_id;
    return value;
}

/// Subgroup shuffle - exchange values between lanes
/// On non-GPU targets, lane_id is ignored and value is returned directly.
pub const subgroupShuffle = if (is_gpu_target) subgroupShuffleGpu else subgroupShuffleCpu;

fn subgroupShuffleGpu(comptime T: type, value: T, lane_id: u32) T {
    var result: T = undefined;
    asm volatile ("OpGroupNonUniformShuffle %[result] %[scope] %[value] %[lane]"
        : [result] "=r" (result),
        : [scope] "c" (@as(u32, 3)), // Subgroup scope
          [value] "r" (value),
          [lane] "r" (lane_id),
    );
    return result;
}

fn subgroupShuffleCpu(comptime T: type, value: T, lane_id: u32) T {
    _ = lane_id;
    return value;
}

/// Subgroup reduction - add across all lanes
pub inline fn subgroupAdd(comptime T: type, value: T) T {
    if (is_gpu_target) {
        var result: T = undefined;
        asm volatile ("OpGroupNonUniformFAdd %[result] %[scope] Reduce %[value]"
            : [result] "=r" (result),
            : [scope] "c" (@as(u32, 3)), // Subgroup scope
              [value] "r" (value),
        );
        return result;
    }
    return value;
}

/// Subgroup reduction - max across all lanes
pub inline fn subgroupMax(comptime T: type, value: T) T {
    if (is_gpu_target) {
        var result: T = undefined;
        asm volatile ("OpGroupNonUniformFMax %[result] %[scope] Reduce %[value]"
            : [result] "=r" (result),
            : [scope] "c" (@as(u32, 3)), // Subgroup scope
              [value] "r" (value),
        );
        return result;
    }
    return value;
}

/// Subgroup reduction - min across all lanes
pub inline fn subgroupMin(comptime T: type, value: T) T {
    if (is_gpu_target) {
        var result: T = undefined;
        asm volatile ("OpGroupNonUniformFMin %[result] %[scope] Reduce %[value]"
            : [result] "=r" (result),
            : [scope] "c" (@as(u32, 3)), // Subgroup scope
              [value] "r" (value),
        );
        return result;
    }
    return value;
}

// ============================================================================
// Vector Type Utilities (std.simd integration)
// ============================================================================

/// Suggested f32 vector size for current GPU target
pub const VecF32Size = if (is_gpu_target)
    std.simd.suggestVectorLength(f32) orelse 4
else
    std.simd.suggestVectorLength(f32) orelse 8;

/// Suggested i32 vector size for current GPU target
pub const VecI32Size = if (is_gpu_target)
    std.simd.suggestVectorLength(i32) orelse 4
else
    std.simd.suggestVectorLength(i32) orelse 8;

/// Standard f32 vector type for GPU operations
pub const VecF32 = @Vector(VecF32Size, f32);

/// Standard i32 vector type for GPU operations
pub const VecI32 = @Vector(VecI32Size, i32);

/// Create a VecF32 filled with a single value
pub inline fn splatF32(value: f32) VecF32 {
    return @splat(value);
}

/// Create a VecI32 filled with a single value
pub inline fn splatI32(value: i32) VecI32 {
    return @splat(value);
}

/// Reduce f32 vector to scalar sum
pub inline fn reduceAddF32(vec: VecF32) f32 {
    return @reduce(.Add, vec);
}

/// Reduce f32 vector to scalar max
pub inline fn reduceMaxF32(vec: VecF32) f32 {
    return @reduce(.Max, vec);
}

/// Reduce f32 vector to scalar min
pub inline fn reduceMinF32(vec: VecF32) f32 {
    return @reduce(.Min, vec);
}

/// Reduce i32 vector to scalar sum
pub inline fn reduceAddI32(vec: VecI32) i32 {
    return @reduce(.Add, vec);
}

// ============================================================================
// Fused Multiply-Add (FMA) Operations
// ============================================================================

/// Fused multiply-add for f32: a * b + c
/// Uses hardware FMA instruction when available
pub inline fn fmaF32(a: f32, b: f32, c: f32) f32 {
    return @mulAdd(f32, a, b, c);
}

/// Fused multiply-add for VecF32: a * b + c
pub inline fn fmaVecF32(a: VecF32, b: VecF32, c: VecF32) VecF32 {
    return @mulAdd(VecF32, a, b, c);
}

// ============================================================================
// Texture/Image Operations (SPIR-V)
// ============================================================================

/// Image dimension types
pub const ImageDim = enum(u32) {
    @"1d" = 0,
    @"2d" = 1,
    @"3d" = 2,
    cube = 3,
    rect = 4,
    buffer = 5,
    subpass_data = 6,
};

/// Image format
pub const ImageFormat = enum(u32) {
    unknown = 0,
    rgba32f = 1,
    rgba16f = 2,
    r32f = 3,
    rgba8 = 4,
    rgba8_snorm = 5,
    // ... more formats as needed
};

/// Image operation result for typed image reads
pub const ImageResult = struct {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
};

// ============================================================================
// Compute Dispatch Helpers
// ============================================================================

/// Calculate number of workgroups needed to cover total_elements
pub fn calculateWorkgroups(total_elements: u32, workgroup_size: u32) u32 {
    return (total_elements + workgroup_size - 1) / workgroup_size;
}

/// Calculate 2D workgroup counts for a 2D grid
pub fn calculateWorkgroups2D(
    width: u32,
    height: u32,
    wg_size_x: u32,
    wg_size_y: u32,
) struct { x: u32, y: u32 } {
    return .{
        .x = (width + wg_size_x - 1) / wg_size_x,
        .y = (height + wg_size_y - 1) / wg_size_y,
    };
}

/// Calculate 3D workgroup counts
pub fn calculateWorkgroups3D(
    width: u32,
    height: u32,
    depth: u32,
    wg_size_x: u32,
    wg_size_y: u32,
    wg_size_z: u32,
) struct { x: u32, y: u32, z: u32 } {
    return .{
        .x = (width + wg_size_x - 1) / wg_size_x,
        .y = (height + wg_size_y - 1) / wg_size_y,
        .z = (depth + wg_size_z - 1) / wg_size_z,
    };
}

// ============================================================================
// Additional Tests
// ============================================================================

test "subgroup operations fallback" {
    // Test fallback values on non-GPU target
    try std.testing.expectEqual(@as(u32, 32), subgroupSize());
    try std.testing.expectEqual(@as(u32, 0), subgroupInvocationId());

    const ballot = subgroupBallot(true);
    try std.testing.expectEqual(@as(u64, 1), ballot);
}

test "vector type utilities" {
    const vec = splatF32(3.14);
    const sum_val = reduceAddF32(vec);
    try std.testing.expectApproxEqAbs(3.14 * @as(f32, @floatFromInt(VecF32Size)), sum_val, 0.01);

    const max_val = reduceMaxF32(vec);
    try std.testing.expectApproxEqAbs(3.14, max_val, 0.001);
}

test "fma operations" {
    const result = fmaF32(2.0, 3.0, 4.0);
    try std.testing.expectApproxEqAbs(10.0, result, 0.001); // 2*3+4 = 10

    const a = splatF32(2.0);
    const b = splatF32(3.0);
    const c = splatF32(4.0);
    const vec_result = fmaVecF32(a, b, c);
    const sum_fma = reduceAddF32(vec_result);
    try std.testing.expectApproxEqAbs(10.0 * @as(f32, @floatFromInt(VecF32Size)), sum_fma, 0.01);
}

test "workgroup calculations" {
    try std.testing.expectEqual(@as(u32, 4), calculateWorkgroups(1000, 256));
    try std.testing.expectEqual(@as(u32, 1), calculateWorkgroups(100, 256));
    try std.testing.expectEqual(@as(u32, 2), calculateWorkgroups(257, 256));

    const wg2d = calculateWorkgroups2D(1920, 1080, 16, 16);
    try std.testing.expectEqual(@as(u32, 120), wg2d.x);
    try std.testing.expectEqual(@as(u32, 68), wg2d.y);
}
