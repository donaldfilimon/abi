//! Metal Mesh Shaders (Metal 3+)
//!
//! Mesh shaders replace the traditional vertex/tessellation pipeline with a
//! two-stage programmable geometry pipeline:
//! 1. Object shader — per-meshlet culling and LOD decisions
//! 2. Mesh shader — generates vertex/primitive data for rasterization
//!
//! Requires Apple7+ GPU family (A15/M2 or later).
//!
//! ## Pipeline Structure
//! ```
//! Object Shader → Mesh Shader → Fragment Shader
//!   (optional)     (required)     (required)
//! ```
//!
//! ## Usage
//! ```zig
//! var pipeline = try MeshPipeline.create(device, .{
//!     .mesh_function = "meshMain",
//!     .fragment_function = "fragmentMain",
//!     .library = library,
//! });
//! defer pipeline.destroy();
//!
//! pipeline.encode(encoder, .{
//!     .threadgroups = .{ 32, 1, 1 },
//!     .threads_per_object = .{ 32, 1, 1 },
//!     .threads_per_mesh = .{ 128, 1, 1 },
//! });
//! ```

const std = @import("std");
const builtin = @import("builtin");
const metal_types = @import("../metal_types.zig");
const gpu_family = @import("gpu_family.zig");

const ID = metal_types.ID;
const SEL = metal_types.SEL;
const Class = metal_types.Class;
const MTLSize = metal_types.MTLSize;

pub const MeshShaderError = error{
    UnsupportedDevice,
    PipelineCreationFailed,
    FunctionNotFound,
    EncodeFailed,
    LibraryRequired,
    FrameworkNotAvailable,
};

/// Which stages are active in the mesh pipeline.
pub const MeshStage = enum {
    object,
    mesh,
};

/// Configuration for creating a mesh render pipeline.
pub const MeshPipelineConfig = struct {
    /// Name of the object function in the Metal library (optional).
    object_function: ?[]const u8 = null,
    /// Name of the mesh function in the Metal library (required).
    mesh_function: []const u8,
    /// Name of the fragment function in the Metal library (required).
    fragment_function: []const u8,
    /// Metal library containing the functions.
    library: ID = null,
    /// Maximum total threads per object threadgroup.
    max_total_threads_per_object_threadgroup: u32 = 32,
    /// Maximum total threads per mesh threadgroup.
    max_total_threads_per_mesh_threadgroup: u32 = 128,
    /// Payload memory length in bytes (for object→mesh communication).
    payload_memory_length: u32 = 0,
    /// Maximum meshes per object threadgroup.
    max_meshes_per_object_threadgroup: u32 = 1,
};

/// Dispatch configuration for mesh shader encoding.
pub const MeshDispatchConfig = struct {
    /// Number of threadgroups to dispatch.
    threadgroups: MTLSize = MTLSize.init(1, 1, 1),
    /// Threads per object threadgroup (if object function is used).
    threads_per_object: MTLSize = MTLSize.init(32, 1, 1),
    /// Threads per mesh threadgroup.
    threads_per_mesh: MTLSize = MTLSize.init(128, 1, 1),
};

/// A compiled mesh render pipeline state.
pub const MeshPipeline = struct {
    pipeline_state: ID = null,
    has_object_stage: bool = false,
    config: MeshPipelineConfig,

    // Obj-C runtime pointers
    msg_send_fn: ?*const fn (ID, SEL) callconv(.c) ID = null,
    sel_register_fn: ?*const fn ([*:0]const u8) callconv(.c) SEL = null,

    /// Create a mesh render pipeline from the given configuration.
    /// Requires Metal 3+ (Apple7+ GPU family).
    pub fn create(
        device: ID,
        config: MeshPipelineConfig,
        msg_send: *const fn (ID, SEL) callconv(.c) ID,
        sel_register: *const fn ([*:0]const u8) callconv(.c) SEL,
        get_class: *const fn ([*:0]const u8) callconv(.c) ?Class,
    ) MeshShaderError!MeshPipeline {
        if (device == null) return MeshShaderError.UnsupportedDevice;
        if (config.library == null) return MeshShaderError.LibraryRequired;

        // Create MTLMeshRenderPipelineDescriptor
        const desc_class = get_class("MTLMeshRenderPipelineDescriptor") orelse
            return MeshShaderError.FrameworkNotAvailable;

        const sel_alloc = sel_register("alloc");
        const sel_init = sel_register("init");

        const desc_alloc = msg_send(@ptrCast(desc_class), sel_alloc);
        if (desc_alloc == null) return MeshShaderError.PipelineCreationFailed;
        const desc = msg_send(desc_alloc, sel_init);
        if (desc == null) return MeshShaderError.PipelineCreationFailed;

        const sel_release = sel_register("release");

        // Helper to create NSString and get function from library
        const get_function = struct {
            fn call(
                lib: ID,
                func_name: []const u8,
                ms: *const fn (ID, SEL) callconv(.c) ID,
                sr: *const fn ([*:0]const u8) callconv(.c) SEL,
                gc: *const fn ([*:0]const u8) callconv(.c) ?Class,
            ) ?ID {
                const nsstring_cls = gc("NSString") orelse return null;
                var buf: [256]u8 = undefined;
                const len = @min(func_name.len, buf.len - 1);
                @memcpy(buf[0..len], func_name[0..len]);
                buf[len] = 0;
                const sel_str = sr("stringWithUTF8String:");
                const str_fn: *const fn (?Class, SEL, [*:0]const u8) callconv(.c) ID = @ptrCast(ms);
                const ns_name = str_fn(nsstring_cls, sel_str, buf[0..len :0]);
                if (ns_name == null) return null;
                const sel_func = sr("newFunctionWithName:");
                const func_fn: *const fn (ID, SEL, ID) callconv(.c) ID = @ptrCast(ms);
                return func_fn(lib, sel_func, ns_name);
            }
        }.call;

        // Set mesh function (required)
        const mesh_func = get_function(
            config.library,
            config.mesh_function,
            msg_send,
            sel_register,
            get_class,
        ) orelse {
            const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
            release_fn(desc, sel_release);
            return MeshShaderError.FunctionNotFound;
        };
        const sel_set_mesh = sel_register("setMeshFunction:");
        const set_fn: *const fn (ID, SEL, ID) callconv(.c) void = @ptrCast(msg_send);
        set_fn(desc, sel_set_mesh, mesh_func);

        // Set fragment function (required)
        const frag_func = get_function(
            config.library,
            config.fragment_function,
            msg_send,
            sel_register,
            get_class,
        ) orelse {
            const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
            release_fn(mesh_func, sel_release);
            release_fn(desc, sel_release);
            return MeshShaderError.FunctionNotFound;
        };
        const sel_set_frag = sel_register("setFragmentFunction:");
        set_fn(desc, sel_set_frag, frag_func);

        // Set object function (optional)
        var has_object = false;
        if (config.object_function) |obj_name| {
            if (get_function(
                config.library,
                obj_name,
                msg_send,
                sel_register,
                get_class,
            )) |obj_func| {
                const sel_set_obj = sel_register("setObjectFunction:");
                set_fn(desc, sel_set_obj, obj_func);
                has_object = true;
                const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
                release_fn(obj_func, sel_release);
            }
        }

        // Create pipeline state
        var pipeline_error: ID = null;
        const sel_create = sel_register(
            "newRenderPipelineStateWithMeshDescriptor:error:",
        );
        const create_fn: *const fn (ID, SEL, ID, *ID) callconv(.c) ID = @ptrCast(msg_send);
        const pipeline_state = create_fn(device, sel_create, desc, &pipeline_error);

        // Release temporary objects
        const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
        release_fn(mesh_func, sel_release);
        release_fn(frag_func, sel_release);
        release_fn(desc, sel_release);

        if (pipeline_state == null) {
            return MeshShaderError.PipelineCreationFailed;
        }

        return .{
            .pipeline_state = pipeline_state,
            .has_object_stage = has_object,
            .config = config,
            .msg_send_fn = msg_send,
            .sel_register_fn = sel_register,
        };
    }

    /// Encode a mesh shader dispatch into a render command encoder.
    pub fn encode(
        self: *const MeshPipeline,
        encoder: ID,
        dispatch: MeshDispatchConfig,
    ) MeshShaderError!void {
        if (self.pipeline_state == null) return MeshShaderError.PipelineCreationFailed;
        if (encoder == null) return MeshShaderError.EncodeFailed;
        const msg_send = self.msg_send_fn orelse return MeshShaderError.FrameworkNotAvailable;
        const sel_fn = self.sel_register_fn orelse return MeshShaderError.FrameworkNotAvailable;

        // Set pipeline state on encoder
        const sel_set_pipeline = sel_fn("setRenderPipelineState:");
        const set_fn: *const fn (ID, SEL, ID) callconv(.c) void = @ptrCast(msg_send);
        set_fn(encoder, sel_set_pipeline, self.pipeline_state);

        // Dispatch mesh threadgroups
        // drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:
        const sel_dispatch = sel_fn(
            "drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:",
        );
        const dispatch_fn: *const fn (
            ID,
            SEL,
            MTLSize,
            MTLSize,
            MTLSize,
        ) callconv(.c) void = @ptrCast(msg_send);
        dispatch_fn(
            encoder,
            sel_dispatch,
            dispatch.threadgroups,
            dispatch.threads_per_object,
            dispatch.threads_per_mesh,
        );
    }

    pub fn destroy(self: *MeshPipeline) void {
        if (self.pipeline_state != null) {
            if (self.msg_send_fn) |msg_send| {
                if (self.sel_register_fn) |sel_fn| {
                    const sel_rel = sel_fn("release");
                    const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
                    release_fn(self.pipeline_state, sel_rel);
                }
            }
            self.pipeline_state = null;
        }
    }
};

// ============================================================================
// MSL Mesh Shader Attributes
// ============================================================================

/// MSL attribute strings for mesh shader parameters.
pub const mesh_attributes = struct {
    pub const object_payload = "[[payload]]";
    pub const mesh_grid_properties = "[[mesh_grid_properties]]";
    pub const object_thread_position_in_grid = "[[thread_position_in_grid]]";
    pub const object_threadgroup_position_in_grid = "[[threadgroup_position_in_grid]]";
    pub const object_threads_per_grid = "[[threads_per_grid]]";
};

// ============================================================================
// Tests
// ============================================================================

test "MeshPipelineConfig defaults" {
    const config = MeshPipelineConfig{
        .mesh_function = "meshMain",
        .fragment_function = "fragMain",
    };
    try std.testing.expect(config.object_function == null);
    try std.testing.expectEqual(@as(u32, 128), config.max_total_threads_per_mesh_threadgroup);
}

test "MeshDispatchConfig defaults" {
    const dispatch = MeshDispatchConfig{};
    try std.testing.expectEqual(@as(usize, 1), dispatch.threadgroups.width);
    try std.testing.expectEqual(@as(usize, 128), dispatch.threads_per_mesh.width);
}
