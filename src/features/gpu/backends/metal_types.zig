//! Metal backend type definitions and error types.
//!
//! Pure type definitions extracted from metal.zig for modularity.
//! This file contains no mutable state or implementation logic.

const std = @import("std");

// ============================================================================
// Error Types
// ============================================================================

pub const MetalError = error{
    InitializationFailed,
    DeviceNotFound,
    ShaderCompilationFailed,
    PipelineCreationFailed,
    CommandQueueCreationFailed,
    BufferCreationFailed,
    CommandBufferCreationFailed,
    KernelExecutionFailed,
    MemoryCopyFailed,
    ObjcRuntimeUnavailable,
    SelectorNotFound,
    InvalidGridSize,
    InvalidBlockSize,
    NSStringCreationFailed,
    DeviceQueryFailed,
    MPSError,
    CoreMLError,
    MeshShaderError,
    RayTracingError,
    UnsupportedFeature,
};

// ============================================================================
// Objective-C Runtime Types
// ============================================================================

pub const SEL = *anyopaque;
pub const Class = *anyopaque;
pub const ID = ?*anyopaque;

// ============================================================================
// Metal Struct Types
// ============================================================================

/// MTLSize struct - matches Metal's definition exactly.
/// Used for specifying grid and threadgroup dimensions.
pub const MTLSize = extern struct {
    width: usize,
    height: usize,
    depth: usize,

    pub fn init(w: usize, h: usize, d: usize) MTLSize {
        return .{ .width = w, .height = h, .depth = d };
    }

    pub fn from3D(dims: [3]u32) MTLSize {
        return .{
            .width = dims[0],
            .height = dims[1],
            .depth = dims[2],
        };
    }
};

/// MTLOrigin struct - matches Metal's definition for region origins.
pub const MTLOrigin = extern struct {
    x: usize,
    y: usize,
    z: usize,
};

/// MTLRegion struct - for buffer/texture regions.
pub const MTLRegion = extern struct {
    origin: MTLOrigin,
    size: MTLSize,
};

// ============================================================================
// Objective-C Function Pointer Types
// ============================================================================

pub const ObjcMsgSendFn = *const fn (ID, SEL) callconv(.c) ID;
pub const ObjcMsgSendIntFn = *const fn (ID, SEL, usize, u32) callconv(.c) ID;
pub const ObjcMsgSendPtrFn = *const fn (ID, SEL, ID) callconv(.c) ID;
pub const ObjcMsgSendPtr2Fn = *const fn (ID, SEL, ID, ID) callconv(.c) ID;
pub const ObjcMsgSendPtr3Fn = *const fn (ID, SEL, ID, ID, ID) callconv(.c) ID;
pub const ObjcMsgSendVoidFn = *const fn (ID, SEL) callconv(.c) void;
pub const ObjcMsgSendVoidPtrFn = *const fn (ID, SEL, ID) callconv(.c) void;
pub const ObjcMsgSendVoidPtrIntIntFn = *const fn (ID, SEL, ID, usize, u32) callconv(.c) void;
pub const ObjcMsgSendU64Fn = *const fn (ID, SEL) callconv(.c) u64;
pub const ObjcMsgSendU32Fn = *const fn (ID, SEL) callconv(.c) u32;
pub const ObjcMsgSendBoolFn = *const fn (ID, SEL) callconv(.c) bool;
pub const SelRegisterNameFn = *const fn ([*:0]const u8) callconv(.c) SEL;
pub const ObjcGetClassFn = *const fn ([*:0]const u8) callconv(.c) Class;

// Function pointer for dispatching with MTLSize structs
// On ARM64 (Apple Silicon), we can pass structs directly
// On x86_64, small structs are passed in registers
pub const ObjcMsgSendMTLSize2Fn = *const fn (ID, SEL, MTLSize, MTLSize) callconv(.c) void;

// NSString creation function pointer
pub const NSStringWithUTF8Fn = *const fn (Class, SEL, [*:0]const u8) callconv(.c) ID;

// ============================================================================
// Metal C-callable Function Types
// ============================================================================

pub const MtlCreateSystemDefaultDeviceFn = *const fn () callconv(.c) ID;
pub const MtlCopyAllDevicesFn = *const fn () callconv(.c) ID; // Returns NSArray of MTLDevice

// ============================================================================
// MTLResourceOptions - matches Metal headers
// ============================================================================

pub const MTLResourceStorageModeShared: u32 = 0;
pub const MTLResourceStorageModeManaged: u32 = 1 << 4;
pub const MTLResourceStorageModePrivate: u32 = 2 << 4;
pub const MTLResourceCPUCacheModeDefaultCache: u32 = 0;
pub const MTLResourceCPUCacheModeWriteCombined: u32 = 1;

// ============================================================================
// Internal Metal Structs
// ============================================================================

pub const MetalKernel = struct {
    pipeline_state: ID,
    library: ID,
    function: ID,
};

pub const MetalBuffer = struct {
    buffer: ID,
    size: usize,
    allocator: std.mem.Allocator,
};

// ============================================================================
// Safe Pointer Casting Types
// ============================================================================

/// Magic value used to validate MetalKernel pointers before casting.
/// This helps detect use-after-free and invalid pointer issues.
pub const kernel_magic: u64 = 0x4D45544B_45524E53; // "METKERNS" in hex

/// Magic value used to validate MetalBuffer pointers before casting.
pub const buffer_magic: u64 = 0x4D455442_55465300; // "METBUFS\0" in hex

/// Extended MetalKernel with validation magic for safe pointer casting.
pub const SafeMetalKernel = struct {
    magic: u64 = kernel_magic,
    inner: MetalKernel,
};

/// Extended MetalBuffer with validation magic for safe pointer casting.
pub const SafeMetalBuffer = struct {
    magic: u64 = buffer_magic,
    inner: MetalBuffer,
};

// ============================================================================
// Device Info
// ============================================================================

/// Detailed device information struct.
pub const DeviceInfo = struct {
    name: []const u8,
    total_memory: u64,
    max_buffer_length: u64,
    max_threads_per_threadgroup: u32,
    has_unified_memory: bool,
    // Metal 3+ feature detection
    gpu_family: u32 = 0,
    supports_mesh_shaders: bool = false,
    supports_ray_tracing: bool = false,
    supports_mps: bool = false,
    supports_neural_engine: bool = false,
};
