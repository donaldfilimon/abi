//! Metal Ray Tracing (Metal 3+)
//!
//! Hardware-accelerated ray tracing using Metal's acceleration structure API.
//! Requires Apple7+ GPU family (A15/M2 or later).
//!
//! ## Architecture
//! Metal ray tracing uses a two-level acceleration structure:
//! 1. **Primitive** (bottom-level): BVH over triangles/bounding boxes
//! 2. **Instance** (top-level): References primitive structures with transforms
//!
//! ## Build Flow
//! ```
//! Triangle Geometry → Primitive AS Descriptor → Query Sizes
//!   → Allocate Scratch + Result → Build via Compute Encoder
//! ```
//!
//! ## Intersection
//! ```
//! Ray origin + direction → Intersection query → Hit result (distance, coords)
//! ```
//!
//! ## Usage
//! ```zig
//! var accel = try AccelerationStructure.buildFromTriangles(device, .{
//!     .vertex_buffer = vertex_buf,
//!     .vertex_count = 1024,
//!     .vertex_stride = 12,
//!     .index_buffer = index_buf,
//!     .index_count = 3072,
//! });
//! defer accel.destroy();
//! ```

const std = @import("std");
const builtin = @import("builtin");
const metal_types = @import("../metal_types.zig");
const gpu_family = @import("gpu_family.zig");

const ID = metal_types.ID;
const SEL = metal_types.SEL;
const Class = metal_types.Class;

pub const RayTracingError = error{
    UnsupportedDevice,
    BuildFailed,
    InvalidGeometry,
    AllocationFailed,
    EncodeFailed,
    FrameworkNotAvailable,
    IntersectionFailed,
    DescriptorCreationFailed,
};

// ============================================================================
// Geometry Types
// ============================================================================

/// Index type for triangle geometry (matches MTLIndexType).
pub const IndexType = enum(u32) {
    uint16 = 0,
    uint32 = 1,

    pub fn byteSize(self: IndexType) u32 {
        return switch (self) {
            .uint16 => 2,
            .uint32 => 4,
        };
    }
};

/// Triangle geometry descriptor for building primitive acceleration structures.
pub const TriangleGeometry = struct {
    /// Metal buffer containing vertex position data.
    vertex_buffer: ID = null,
    /// Number of vertices in the buffer.
    vertex_count: u32 = 0,
    /// Byte stride between consecutive vertices (typically 12 for float3).
    vertex_stride: u32 = 12,
    /// Byte offset into the vertex buffer.
    vertex_buffer_offset: u32 = 0,

    /// Metal buffer containing index data (optional — null for non-indexed).
    index_buffer: ID = null,
    /// Number of indices (must be multiple of 3).
    index_count: u32 = 0,
    /// Index element type.
    index_type: IndexType = .uint32,
    /// Byte offset into the index buffer.
    index_buffer_offset: u32 = 0,

    /// 4x4 affine transform applied to this geometry (column-major, null = identity).
    transform_buffer: ID = null,
    transform_buffer_offset: u32 = 0,

    /// Whether this geometry is opaque (skips any-hit shader).
    is_opaque: bool = true,

    pub fn triangleCount(self: *const TriangleGeometry) u32 {
        if (self.index_count > 0) {
            return self.index_count / 3;
        }
        return self.vertex_count / 3;
    }
};

/// Bounding box for custom geometry intersection.
pub const BoundingBox = extern struct {
    min_x: f32,
    min_y: f32,
    min_z: f32,
    max_x: f32,
    max_y: f32,
    max_z: f32,
};

/// Bounding box geometry descriptor for custom intersection functions.
pub const BoundingBoxGeometry = struct {
    /// Metal buffer containing BoundingBox data.
    bounding_box_buffer: ID = null,
    /// Number of bounding boxes.
    bounding_box_count: u32 = 0,
    /// Byte stride between consecutive bounding boxes (default = @sizeOf(BoundingBox)).
    bounding_box_stride: u32 = @sizeOf(BoundingBox),
    /// Byte offset into the bounding box buffer.
    bounding_box_buffer_offset: u32 = 0,
    /// Whether this geometry is opaque.
    is_opaque: bool = true,
};

// ============================================================================
// Instance Descriptor
// ============================================================================

/// Describes an instance in a top-level (instance) acceleration structure.
/// Matches MTLAccelerationStructureInstanceDescriptor layout.
pub const InstanceDescriptor = extern struct {
    /// 4x3 affine transform matrix (column-major, row 4 implicit [0,0,0,1]).
    transform: [4][3]f32 = identityTransform(),
    /// Bitfield options (opaque, non-opaque, etc.).
    options: u32 = 0,
    /// Visibility mask for intersection testing.
    mask: u32 = 0xFF,
    /// Offset into the intersection function table.
    intersection_function_table_offset: u32 = 0,
    /// Index of the primitive acceleration structure in the instance array.
    acceleration_structure_index: u32 = 0,

    fn identityTransform() [4][3]f32 {
        return .{
            .{ 1.0, 0.0, 0.0 },
            .{ 0.0, 1.0, 0.0 },
            .{ 0.0, 0.0, 1.0 },
            .{ 0.0, 0.0, 0.0 },
        };
    }
};

/// Instance options matching MTLAccelerationStructureInstanceOptions.
pub const InstanceOptions = struct {
    pub const none: u32 = 0;
    pub const disable_triangle_cull: u32 = 1;
    pub const triangle_front_facing_winding_ccw: u32 = 2;
    pub const @"opaque": u32 = 4;
    pub const non_opaque: u32 = 8;
};

// ============================================================================
// Acceleration Structure Sizes
// ============================================================================

/// Size requirements returned by Metal for building an acceleration structure.
pub const AccelerationStructureSizes = struct {
    /// Size in bytes for the acceleration structure itself.
    acceleration_structure_size: u64 = 0,
    /// Size in bytes for the scratch buffer during build.
    build_scratch_buffer_size: u64 = 0,
    /// Size in bytes for the scratch buffer during refit.
    refit_scratch_buffer_size: u64 = 0,
};

// ============================================================================
// Acceleration Structure
// ============================================================================

/// A built acceleration structure ready for intersection queries.
pub const AccelerationStructure = struct {
    /// The Metal acceleration structure object.
    structure: ID = null,
    /// Whether this is a primitive (bottom-level) or instance (top-level) structure.
    is_instance_structure: bool = false,
    /// Obj-C runtime function pointers (cached from creation).
    msg_send_fn: ?*const fn (ID, SEL) callconv(.c) ID = null,
    sel_register_fn: ?*const fn ([*:0]const u8) callconv(.c) SEL = null,

    /// Build a primitive (bottom-level) acceleration structure from triangle geometry.
    pub fn buildFromTriangles(
        device: ID,
        geometry: TriangleGeometry,
        msg_send: *const fn (ID, SEL) callconv(.c) ID,
        sel_register: *const fn ([*:0]const u8) callconv(.c) SEL,
        get_class: *const fn ([*:0]const u8) callconv(.c) ?Class,
    ) RayTracingError!AccelerationStructure {
        if (device == null) return RayTracingError.UnsupportedDevice;
        if (geometry.vertex_buffer == null) return RayTracingError.InvalidGeometry;
        if (geometry.vertex_count == 0) return RayTracingError.InvalidGeometry;

        const sel_alloc = sel_register("alloc");
        const sel_init = sel_register("init");
        const sel_release = sel_register("release");
        const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);

        // Create MTLAccelerationStructureTriangleGeometryDescriptor
        const tri_desc_class = get_class(
            "MTLAccelerationStructureTriangleGeometryDescriptor",
        ) orelse return RayTracingError.FrameworkNotAvailable;

        const tri_desc_alloc = msg_send(@ptrCast(tri_desc_class), sel_alloc);
        if (tri_desc_alloc == null) return RayTracingError.DescriptorCreationFailed;
        const tri_desc = msg_send(tri_desc_alloc, sel_init);
        if (tri_desc == null) return RayTracingError.DescriptorCreationFailed;

        // Set vertex buffer and properties
        const set_obj_fn: *const fn (ID, SEL, ID) callconv(.c) void = @ptrCast(msg_send);
        const set_u32_fn: *const fn (ID, SEL, u32) callconv(.c) void = @ptrCast(msg_send);
        const set_bool_fn: *const fn (ID, SEL, bool) callconv(.c) void = @ptrCast(msg_send);

        set_obj_fn(tri_desc, sel_register("setVertexBuffer:"), geometry.vertex_buffer);
        set_u32_fn(tri_desc, sel_register("setVertexBufferOffset:"), geometry.vertex_buffer_offset);
        set_u32_fn(tri_desc, sel_register("setVertexStride:"), geometry.vertex_stride);
        set_u32_fn(tri_desc, sel_register("setTriangleCount:"), geometry.triangleCount());
        set_bool_fn(tri_desc, sel_register("setOpaque:"), geometry.is_opaque);

        // Set index buffer if provided
        if (geometry.index_buffer != null) {
            set_obj_fn(tri_desc, sel_register("setIndexBuffer:"), geometry.index_buffer);
            set_u32_fn(
                tri_desc,
                sel_register("setIndexBufferOffset:"),
                geometry.index_buffer_offset,
            );
            set_u32_fn(
                tri_desc,
                sel_register("setIndexType:"),
                @intFromEnum(geometry.index_type),
            );
        }

        // Set transform buffer if provided
        if (geometry.transform_buffer != null) {
            set_obj_fn(
                tri_desc,
                sel_register("setTransformationMatrixBuffer:"),
                geometry.transform_buffer,
            );
            set_u32_fn(
                tri_desc,
                sel_register("setTransformationMatrixBufferOffset:"),
                geometry.transform_buffer_offset,
            );
        }

        // Create primitive acceleration structure descriptor
        const prim_desc_class = get_class(
            "MTLPrimitiveAccelerationStructureDescriptor",
        ) orelse {
            release_fn(tri_desc, sel_release);
            return RayTracingError.FrameworkNotAvailable;
        };

        const prim_desc_alloc = msg_send(@ptrCast(prim_desc_class), sel_alloc);
        if (prim_desc_alloc == null) {
            release_fn(tri_desc, sel_release);
            return RayTracingError.DescriptorCreationFailed;
        }
        const prim_desc = msg_send(prim_desc_alloc, sel_init);
        if (prim_desc == null) {
            release_fn(tri_desc, sel_release);
            return RayTracingError.DescriptorCreationFailed;
        }

        // Wrap geometry in NSArray and set on descriptor
        const nsarray_class = get_class("NSArray") orelse {
            release_fn(tri_desc, sel_release);
            release_fn(prim_desc, sel_release);
            return RayTracingError.FrameworkNotAvailable;
        };
        const sel_array_with = sel_register("arrayWithObject:");
        const array_fn: *const fn (?Class, SEL, ID) callconv(.c) ID = @ptrCast(msg_send);
        const geo_array = array_fn(nsarray_class, sel_array_with, tri_desc);

        if (geo_array != null) {
            set_obj_fn(
                prim_desc,
                sel_register("setGeometryDescriptors:"),
                geo_array,
            );
        }

        // Query sizes from device
        const sel_sizes = sel_register(
            "accelerationStructureSizesWithDescriptor:",
        );

        // AccelerationStructureSizes is returned as a struct.
        // On ARM64 (Apple Silicon), small structs are returned in registers.
        const sizes_fn: *const fn (
            ID,
            SEL,
            ID,
        ) callconv(.c) AccelerationStructureSizes = @ptrCast(msg_send);
        const sizes = sizes_fn(device, sel_sizes, prim_desc);

        if (sizes.acceleration_structure_size == 0) {
            release_fn(tri_desc, sel_release);
            release_fn(prim_desc, sel_release);
            return RayTracingError.AllocationFailed;
        }

        // Create acceleration structure with the required size
        const sel_new_accel = sel_register(
            "newAccelerationStructureWithSize:",
        );
        const new_accel_fn: *const fn (ID, SEL, u64) callconv(.c) ID = @ptrCast(msg_send);
        const accel_struct = new_accel_fn(
            device,
            sel_new_accel,
            sizes.acceleration_structure_size,
        );

        if (accel_struct == null) {
            release_fn(tri_desc, sel_release);
            release_fn(prim_desc, sel_release);
            return RayTracingError.AllocationFailed;
        }

        // Create scratch buffer
        const sel_new_buf = sel_register("newBufferWithLength:options:");
        const new_buf_fn: *const fn (ID, SEL, u64, u32) callconv(.c) ID = @ptrCast(msg_send);
        const scratch_buf = new_buf_fn(
            device,
            sel_new_buf,
            sizes.build_scratch_buffer_size,
            metal_types.MTLResourceStorageModePrivate,
        );

        if (scratch_buf == null) {
            release_fn(accel_struct, sel_release);
            release_fn(tri_desc, sel_release);
            release_fn(prim_desc, sel_release);
            return RayTracingError.AllocationFailed;
        }

        // Create command queue and encode build
        const sel_new_queue = sel_register("newCommandQueue");
        const cmd_queue = msg_send(device, sel_new_queue);
        if (cmd_queue == null) {
            release_fn(scratch_buf, sel_release);
            release_fn(accel_struct, sel_release);
            release_fn(tri_desc, sel_release);
            release_fn(prim_desc, sel_release);
            return RayTracingError.BuildFailed;
        }

        const sel_cmd_buf = sel_register("commandBuffer");
        const cmd_buf = msg_send(cmd_queue, sel_cmd_buf);
        if (cmd_buf == null) {
            release_fn(cmd_queue, sel_release);
            release_fn(scratch_buf, sel_release);
            release_fn(accel_struct, sel_release);
            release_fn(tri_desc, sel_release);
            release_fn(prim_desc, sel_release);
            return RayTracingError.BuildFailed;
        }

        // Create acceleration structure command encoder
        const sel_accel_encoder = sel_register(
            "accelerationStructureCommandEncoder",
        );
        const encoder = msg_send(cmd_buf, sel_accel_encoder);
        if (encoder == null) {
            release_fn(cmd_queue, sel_release);
            release_fn(scratch_buf, sel_release);
            release_fn(accel_struct, sel_release);
            release_fn(tri_desc, sel_release);
            release_fn(prim_desc, sel_release);
            return RayTracingError.BuildFailed;
        }

        // Encode build command
        const sel_build = sel_register(
            "buildAccelerationStructure:descriptor:scratchBuffer:scratchBufferOffset:",
        );
        const build_fn: *const fn (
            ID,
            SEL,
            ID,
            ID,
            ID,
            u64,
        ) callconv(.c) void = @ptrCast(msg_send);
        build_fn(encoder, sel_build, accel_struct, prim_desc, scratch_buf, 0);

        // End encoding and commit
        const sel_end = sel_register("endEncoding");
        const end_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
        end_fn(encoder, sel_end);

        const sel_commit = sel_register("commit");
        const commit_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
        commit_fn(cmd_buf, sel_commit);

        const sel_wait = sel_register("waitUntilCompleted");
        const wait_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
        wait_fn(cmd_buf, sel_wait);

        // Release temporaries
        release_fn(cmd_queue, sel_release);
        release_fn(scratch_buf, sel_release);
        release_fn(tri_desc, sel_release);
        release_fn(prim_desc, sel_release);

        return .{
            .structure = accel_struct,
            .is_instance_structure = false,
            .msg_send_fn = msg_send,
            .sel_register_fn = sel_register,
        };
    }

    /// Build an instance (top-level) acceleration structure from primitive structures.
    pub fn buildInstanceStructure(
        device: ID,
        instance_buffer: ID,
        instance_count: u32,
        primitive_structures: []const ID,
        msg_send: *const fn (ID, SEL) callconv(.c) ID,
        sel_register: *const fn ([*:0]const u8) callconv(.c) SEL,
        get_class: *const fn ([*:0]const u8) callconv(.c) ?Class,
    ) RayTracingError!AccelerationStructure {
        if (device == null) return RayTracingError.UnsupportedDevice;
        if (instance_buffer == null) return RayTracingError.InvalidGeometry;
        if (instance_count == 0) return RayTracingError.InvalidGeometry;

        const sel_alloc = sel_register("alloc");
        const sel_init = sel_register("init");
        const sel_release = sel_register("release");
        const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);

        // Create instance acceleration structure descriptor
        const inst_desc_class = get_class(
            "MTLInstanceAccelerationStructureDescriptor",
        ) orelse return RayTracingError.FrameworkNotAvailable;

        const inst_desc_alloc = msg_send(@ptrCast(inst_desc_class), sel_alloc);
        if (inst_desc_alloc == null) return RayTracingError.DescriptorCreationFailed;
        const inst_desc = msg_send(inst_desc_alloc, sel_init);
        if (inst_desc == null) return RayTracingError.DescriptorCreationFailed;

        // Set instance descriptor buffer
        const set_obj_fn: *const fn (ID, SEL, ID) callconv(.c) void = @ptrCast(msg_send);
        const set_u32_fn: *const fn (ID, SEL, u32) callconv(.c) void = @ptrCast(msg_send);

        set_obj_fn(
            inst_desc,
            sel_register("setInstanceDescriptorBuffer:"),
            instance_buffer,
        );
        set_u32_fn(
            inst_desc,
            sel_register("setInstanceCount:"),
            instance_count,
        );

        // Create NSArray of primitive acceleration structures
        const nsarray_class = get_class("NSMutableArray") orelse {
            release_fn(inst_desc, sel_release);
            return RayTracingError.FrameworkNotAvailable;
        };

        const array_alloc = msg_send(@ptrCast(nsarray_class), sel_alloc);
        if (array_alloc == null) {
            release_fn(inst_desc, sel_release);
            return RayTracingError.AllocationFailed;
        }
        const array = msg_send(array_alloc, sel_init);
        if (array == null) {
            release_fn(inst_desc, sel_release);
            return RayTracingError.AllocationFailed;
        }

        // Add each primitive structure to the array
        const sel_add = sel_register("addObject:");
        for (primitive_structures) |prim| {
            if (prim != null) {
                set_obj_fn(array, sel_add, prim);
            }
        }

        set_obj_fn(
            inst_desc,
            sel_register("setInstancedAccelerationStructures:"),
            array,
        );

        // Query sizes
        const sel_sizes = sel_register(
            "accelerationStructureSizesWithDescriptor:",
        );
        const sizes_fn: *const fn (
            ID,
            SEL,
            ID,
        ) callconv(.c) AccelerationStructureSizes = @ptrCast(msg_send);
        const sizes = sizes_fn(device, sel_sizes, inst_desc);

        if (sizes.acceleration_structure_size == 0) {
            release_fn(array, sel_release);
            release_fn(inst_desc, sel_release);
            return RayTracingError.AllocationFailed;
        }

        // Create acceleration structure
        const sel_new_accel = sel_register(
            "newAccelerationStructureWithSize:",
        );
        const new_accel_fn: *const fn (ID, SEL, u64) callconv(.c) ID = @ptrCast(msg_send);
        const accel_struct = new_accel_fn(
            device,
            sel_new_accel,
            sizes.acceleration_structure_size,
        );

        if (accel_struct == null) {
            release_fn(array, sel_release);
            release_fn(inst_desc, sel_release);
            return RayTracingError.AllocationFailed;
        }

        // Create scratch buffer
        const sel_new_buf = sel_register("newBufferWithLength:options:");
        const new_buf_fn: *const fn (ID, SEL, u64, u32) callconv(.c) ID = @ptrCast(msg_send);
        const scratch_buf = new_buf_fn(
            device,
            sel_new_buf,
            sizes.build_scratch_buffer_size,
            metal_types.MTLResourceStorageModePrivate,
        );

        if (scratch_buf == null) {
            release_fn(accel_struct, sel_release);
            release_fn(array, sel_release);
            release_fn(inst_desc, sel_release);
            return RayTracingError.AllocationFailed;
        }

        // Create command queue and encode build
        const sel_new_queue = sel_register("newCommandQueue");
        const cmd_queue = msg_send(device, sel_new_queue);
        if (cmd_queue == null) {
            release_fn(scratch_buf, sel_release);
            release_fn(accel_struct, sel_release);
            release_fn(array, sel_release);
            release_fn(inst_desc, sel_release);
            return RayTracingError.BuildFailed;
        }

        const sel_cmd_buf = sel_register("commandBuffer");
        const cmd_buf = msg_send(cmd_queue, sel_cmd_buf);
        if (cmd_buf == null) {
            release_fn(cmd_queue, sel_release);
            release_fn(scratch_buf, sel_release);
            release_fn(accel_struct, sel_release);
            release_fn(array, sel_release);
            release_fn(inst_desc, sel_release);
            return RayTracingError.BuildFailed;
        }

        const sel_accel_encoder = sel_register(
            "accelerationStructureCommandEncoder",
        );
        const encoder = msg_send(cmd_buf, sel_accel_encoder);
        if (encoder == null) {
            release_fn(cmd_queue, sel_release);
            release_fn(scratch_buf, sel_release);
            release_fn(accel_struct, sel_release);
            release_fn(array, sel_release);
            release_fn(inst_desc, sel_release);
            return RayTracingError.BuildFailed;
        }

        // Encode build
        const sel_build = sel_register(
            "buildAccelerationStructure:descriptor:scratchBuffer:scratchBufferOffset:",
        );
        const build_fn: *const fn (
            ID,
            SEL,
            ID,
            ID,
            ID,
            u64,
        ) callconv(.c) void = @ptrCast(msg_send);
        build_fn(encoder, sel_build, accel_struct, inst_desc, scratch_buf, 0);

        const end_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
        end_fn(encoder, sel_register("endEncoding"));

        const commit_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
        commit_fn(cmd_buf, sel_register("commit"));

        const wait_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
        wait_fn(cmd_buf, sel_register("waitUntilCompleted"));

        // Release temporaries
        release_fn(cmd_queue, sel_release);
        release_fn(scratch_buf, sel_release);
        release_fn(array, sel_release);
        release_fn(inst_desc, sel_release);

        return .{
            .structure = accel_struct,
            .is_instance_structure = true,
            .msg_send_fn = msg_send,
            .sel_register_fn = sel_register,
        };
    }

    /// Refit an existing acceleration structure (for animated geometry).
    /// Refitting is faster than a full rebuild but produces a less optimal BVH.
    pub fn refit(
        self: *const AccelerationStructure,
        device: ID,
        descriptor: ID,
    ) RayTracingError!void {
        if (self.structure == null) return RayTracingError.BuildFailed;
        if (device == null) return RayTracingError.UnsupportedDevice;
        const msg_send = self.msg_send_fn orelse return RayTracingError.FrameworkNotAvailable;
        const sel_fn = self.sel_register_fn orelse return RayTracingError.FrameworkNotAvailable;
        const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);

        // Query refit scratch size
        const sel_sizes = sel_fn(
            "accelerationStructureSizesWithDescriptor:",
        );
        const sizes_fn: *const fn (
            ID,
            SEL,
            ID,
        ) callconv(.c) AccelerationStructureSizes = @ptrCast(msg_send);
        const sizes = sizes_fn(device, sel_sizes, descriptor);

        // Create scratch buffer for refit
        const sel_new_buf = sel_fn("newBufferWithLength:options:");
        const new_buf_fn: *const fn (ID, SEL, u64, u32) callconv(.c) ID = @ptrCast(msg_send);
        const scratch_buf = new_buf_fn(
            device,
            sel_new_buf,
            sizes.refit_scratch_buffer_size,
            metal_types.MTLResourceStorageModePrivate,
        );
        if (scratch_buf == null) return RayTracingError.AllocationFailed;

        // Encode refit
        const sel_new_queue = sel_fn("newCommandQueue");
        const cmd_queue = msg_send(device, sel_new_queue);
        if (cmd_queue == null) {
            release_fn(scratch_buf, sel_fn("release"));
            return RayTracingError.BuildFailed;
        }

        const cmd_buf = msg_send(cmd_queue, sel_fn("commandBuffer"));
        if (cmd_buf == null) {
            release_fn(cmd_queue, sel_fn("release"));
            release_fn(scratch_buf, sel_fn("release"));
            return RayTracingError.BuildFailed;
        }

        const encoder = msg_send(cmd_buf, sel_fn("accelerationStructureCommandEncoder"));
        if (encoder == null) {
            release_fn(cmd_queue, sel_fn("release"));
            release_fn(scratch_buf, sel_fn("release"));
            return RayTracingError.BuildFailed;
        }

        const sel_refit = sel_fn(
            "refitAccelerationStructure:descriptor:destination:scratchBuffer:scratchBufferOffset:",
        );
        const refit_fn: *const fn (
            ID,
            SEL,
            ID,
            ID,
            ID,
            ID,
            u64,
        ) callconv(.c) void = @ptrCast(msg_send);
        // Refit in-place (destination = same structure)
        refit_fn(
            encoder,
            sel_refit,
            self.structure,
            descriptor,
            self.structure,
            scratch_buf,
            0,
        );

        const end_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
        end_fn(encoder, sel_fn("endEncoding"));
        const commit_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
        commit_fn(cmd_buf, sel_fn("commit"));
        const wait_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
        wait_fn(cmd_buf, sel_fn("waitUntilCompleted"));

        release_fn(cmd_queue, sel_fn("release"));
        release_fn(scratch_buf, sel_fn("release"));
    }

    pub fn destroy(self: *AccelerationStructure) void {
        if (self.structure != null) {
            if (self.msg_send_fn) |msg_send| {
                if (self.sel_register_fn) |sel_fn| {
                    const sel_rel = sel_fn("release");
                    const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
                    release_fn(self.structure, sel_rel);
                }
            }
            self.structure = null;
        }
    }
};

// ============================================================================
// Intersection Function Table
// ============================================================================

/// Wraps MTLIntersectionFunctionTable for custom intersection testing.
/// Used with bounding box geometry to implement custom ray-shape intersections.
pub const IntersectionFunctionTable = struct {
    table: ID = null,
    msg_send_fn: ?*const fn (ID, SEL) callconv(.c) ID = null,
    sel_register_fn: ?*const fn ([*:0]const u8) callconv(.c) SEL = null,

    /// Create an intersection function table from a compute pipeline.
    pub fn create(
        pipeline: ID,
        function_count: u32,
        msg_send: *const fn (ID, SEL) callconv(.c) ID,
        sel_register: *const fn ([*:0]const u8) callconv(.c) SEL,
        get_class: *const fn ([*:0]const u8) callconv(.c) ?Class,
    ) RayTracingError!IntersectionFunctionTable {
        if (pipeline == null) return RayTracingError.FrameworkNotAvailable;
        _ = get_class;

        const sel_alloc = sel_register("alloc");
        const sel_init = sel_register("init");

        // Create descriptor
        const sel_fn_table_desc = sel_register(
            "intersectionFunctionTableDescriptor",
        );
        // MTLIntersectionFunctionTableDescriptor is created via class method
        const desc = msg_send(pipeline, sel_fn_table_desc);
        if (desc == null) {
            // Try creating via alloc/init pattern
            const cls_desc_sel = sel_register(
                "MTLIntersectionFunctionTableDescriptor",
            );
            _ = cls_desc_sel;
            return RayTracingError.DescriptorCreationFailed;
        }

        // Set function count
        const set_u32_fn: *const fn (ID, SEL, u32) callconv(.c) void = @ptrCast(msg_send);
        set_u32_fn(desc, sel_register("setFunctionCount:"), function_count);

        // Create table from pipeline
        const sel_make = sel_register(
            "newIntersectionFunctionTableWithDescriptor:",
        );
        const make_fn: *const fn (ID, SEL, ID) callconv(.c) ID = @ptrCast(msg_send);
        const table = make_fn(pipeline, sel_make, desc);

        _ = sel_alloc;
        _ = sel_init;

        if (table == null) return RayTracingError.IntersectionFailed;

        return .{
            .table = table,
            .msg_send_fn = msg_send,
            .sel_register_fn = sel_register,
        };
    }

    /// Set a function handle at the given index in the table.
    pub fn setFunction(
        self: *const IntersectionFunctionTable,
        function_handle: ID,
        index: u32,
    ) RayTracingError!void {
        if (self.table == null) return RayTracingError.IntersectionFailed;
        const msg_send = self.msg_send_fn orelse return RayTracingError.FrameworkNotAvailable;
        const sel_fn = self.sel_register_fn orelse return RayTracingError.FrameworkNotAvailable;

        const sel_set = sel_fn("setFunction:atIndex:");
        const set_fn: *const fn (ID, SEL, ID, u32) callconv(.c) void = @ptrCast(msg_send);
        set_fn(self.table, sel_set, function_handle, index);
    }

    pub fn destroy(self: *IntersectionFunctionTable) void {
        if (self.table != null) {
            if (self.msg_send_fn) |msg_send| {
                if (self.sel_register_fn) |sel_fn| {
                    const sel_rel = sel_fn("release");
                    const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
                    release_fn(self.table, sel_rel);
                }
            }
            self.table = null;
        }
    }
};

// ============================================================================
// MSL Ray Tracing Attributes
// ============================================================================

/// MSL attribute strings for ray tracing parameters in shaders.
pub const ray_tracing_attributes = struct {
    pub const ray_data = "[[ray_data]]";
    pub const intersection_result = "[[intersection_result]]";
    pub const primitive_id = "[[primitive_id]]";
    pub const instance_id = "[[instance_id]]";
    pub const geometry_id = "[[geometry_id]]";
    pub const world_space_origin = "[[world_space_origin]]";
    pub const world_space_direction = "[[world_space_direction]]";
    pub const ray_flags = "[[ray_flags]]";
};

// ============================================================================
// Tests
// ============================================================================

test "TriangleGeometry triangle count" {
    // Indexed geometry
    const indexed = TriangleGeometry{
        .vertex_count = 100,
        .index_count = 300,
    };
    try std.testing.expectEqual(@as(u32, 100), indexed.triangleCount());

    // Non-indexed geometry
    const non_indexed = TriangleGeometry{
        .vertex_count = 99,
    };
    try std.testing.expectEqual(@as(u32, 33), non_indexed.triangleCount());
}

test "IndexType byte size" {
    try std.testing.expectEqual(@as(u32, 2), IndexType.uint16.byteSize());
    try std.testing.expectEqual(@as(u32, 4), IndexType.uint32.byteSize());
}

test "InstanceDescriptor identity transform" {
    const desc = InstanceDescriptor{};
    // Column 0 should be [1, 0, 0]
    try std.testing.expectEqual(@as(f32, 1.0), desc.transform[0][0]);
    try std.testing.expectEqual(@as(f32, 0.0), desc.transform[0][1]);
    // Column 1 should be [0, 1, 0]
    try std.testing.expectEqual(@as(f32, 1.0), desc.transform[1][1]);
    // Default mask is 0xFF
    try std.testing.expectEqual(@as(u32, 0xFF), desc.mask);
}

test "AccelerationStructureSizes defaults" {
    const sizes = AccelerationStructureSizes{};
    try std.testing.expectEqual(@as(u64, 0), sizes.acceleration_structure_size);
    try std.testing.expectEqual(@as(u64, 0), sizes.build_scratch_buffer_size);
    try std.testing.expectEqual(@as(u64, 0), sizes.refit_scratch_buffer_size);
}

test "BoundingBox size" {
    // BoundingBox should be 24 bytes (6 x f32)
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(BoundingBox));
}

test {
    std.testing.refAllDecls(@This());
}
