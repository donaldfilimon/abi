//! Shared mutable state for the Metal backend.
//!
//! This module centralizes all module-level variables that are shared across
//! metal_device.zig, metal_buffers.zig, metal_compute.zig, and the orchestrator metal.zig.

const std = @import("std");
const metal_types = @import("metal_types");

// Re-export types used by consumers
pub const SEL = metal_types.SEL;
pub const Class = metal_types.Class;
pub const ID = metal_types.ID;
pub const MTLSize = metal_types.MTLSize;
pub const MetalError = metal_types.MetalError;
pub const MetalKernel = metal_types.MetalKernel;
pub const MetalBuffer = metal_types.MetalBuffer;
pub const SafeMetalKernel = metal_types.SafeMetalKernel;
pub const SafeMetalBuffer = metal_types.SafeMetalBuffer;
pub const kernel_magic = metal_types.kernel_magic;
pub const buffer_magic = metal_types.buffer_magic;
pub const MTLResourceStorageModeShared = metal_types.MTLResourceStorageModeShared;

// Objective-C runtime function pointer types
pub const ObjcMsgSendFn = metal_types.ObjcMsgSendFn;
pub const ObjcMsgSendIntFn = metal_types.ObjcMsgSendIntFn;
pub const ObjcMsgSendPtrFn = metal_types.ObjcMsgSendPtrFn;
pub const ObjcMsgSendPtr2Fn = metal_types.ObjcMsgSendPtr2Fn;
pub const ObjcMsgSendPtr3Fn = metal_types.ObjcMsgSendPtr3Fn;
pub const ObjcMsgSendVoidFn = metal_types.ObjcMsgSendVoidFn;
pub const ObjcMsgSendVoidPtrFn = metal_types.ObjcMsgSendVoidPtrFn;
pub const ObjcMsgSendVoidPtrIntIntFn = metal_types.ObjcMsgSendVoidPtrIntIntFn;
pub const ObjcMsgSendU64Fn = metal_types.ObjcMsgSendU64Fn;
pub const ObjcMsgSendU32Fn = metal_types.ObjcMsgSendU32Fn;
pub const ObjcMsgSendBoolFn = metal_types.ObjcMsgSendBoolFn;
pub const ObjcMsgSendMTLSize2Fn = metal_types.ObjcMsgSendMTLSize2Fn;
pub const NSStringWithUTF8Fn = metal_types.NSStringWithUTF8Fn;
pub const SelRegisterNameFn = metal_types.SelRegisterNameFn;
pub const ObjcGetClassFn = metal_types.ObjcGetClassFn;
pub const MtlCreateSystemDefaultDeviceFn = metal_types.MtlCreateSystemDefaultDeviceFn;
pub const MtlCopyAllDevicesFn = metal_types.MtlCopyAllDevicesFn;

// ============================================================================
// Dynamic library handles
// ============================================================================

pub var metal_lib: ?std.DynLib = null;
pub var objc_lib: ?std.DynLib = null;
pub var foundation_lib: ?std.DynLib = null;
pub var metal_initialized: bool = false;
pub var metal_device: ID = null;
pub var metal_command_queue: ID = null;

// ============================================================================
// Device properties cache
// ============================================================================

pub var device_name_buf: [256]u8 = undefined;
pub var device_name_len: usize = 0;
pub var device_total_memory: u64 = 0;
pub var device_max_threads_per_group: u32 = 0;
pub var device_max_buffer_length: u64 = 0;

// ============================================================================
// Cached GPU feature set
// ============================================================================

const gpu_family = @import("metal/gpu_family");
const capabilities = @import("metal/capabilities");
pub const MetalGpuFamily = gpu_family.MetalGpuFamily;
pub const MetalFeatureSet = gpu_family.MetalFeatureSet;
pub const MetalLevel = capabilities.MetalLevel;

pub var cached_feature_set: ?MetalFeatureSet = null;
pub var cached_metal_level: MetalLevel = .none;

// ============================================================================
// Pending command buffers for synchronization
// ============================================================================

pub var pending_command_buffers: std.ArrayListUnmanaged(ID) = .empty;
pub var pending_buffers_allocator: ?std.mem.Allocator = null;

// ============================================================================
// Cached allocator for buffer metadata
// ============================================================================

pub var buffer_allocator: ?std.mem.Allocator = null;

// ============================================================================
// Pipeline cache
// ============================================================================

pub var pipeline_cache: std.StringHashMapUnmanaged(ID) = .empty;
pub var pipeline_cache_allocator: ?std.mem.Allocator = null;

// ============================================================================
// Objective-C runtime function pointers
// ============================================================================

pub var objc_msgSend: ?ObjcMsgSendFn = null;
pub var objc_msgSend_int: ?ObjcMsgSendIntFn = null;
pub var objc_msgSend_ptr: ?ObjcMsgSendPtrFn = null;
pub var objc_msgSend_ptr2: ?ObjcMsgSendPtr2Fn = null;
pub var objc_msgSend_ptr3: ?ObjcMsgSendPtr3Fn = null;
pub var objc_msgSend_void: ?ObjcMsgSendVoidFn = null;
pub var objc_msgSend_void_ptr: ?ObjcMsgSendVoidPtrFn = null;
pub var objc_msgSend_void_ptr_int_int: ?ObjcMsgSendVoidPtrIntIntFn = null;
pub var objc_msgSend_u64: ?ObjcMsgSendU64Fn = null;
pub var objc_msgSend_u32: ?ObjcMsgSendU32Fn = null;
pub var objc_msgSend_bool: ?ObjcMsgSendBoolFn = null;
pub var objc_msgSend_mtlsize2: ?ObjcMsgSendMTLSize2Fn = null;
pub var objc_msgSend_nsstring: ?NSStringWithUTF8Fn = null;
pub var sel_registerName: ?SelRegisterNameFn = null;
pub var objc_getClass: ?ObjcGetClassFn = null;

// NSString class reference
pub var nsstring_class: ?Class = null;

// Metal C-callable function pointers
pub var mtlCreateSystemDefaultDevice: ?MtlCreateSystemDefaultDeviceFn = null;
pub var mtlCopyAllDevices: ?MtlCopyAllDevicesFn = null;

// ============================================================================
// Cached selectors
// ============================================================================

pub var sel_newCommandQueue: SEL = undefined;
pub var sel_newLibraryWithSource: SEL = undefined;
pub var sel_newFunctionWithName: SEL = undefined;
pub var sel_newComputePipelineStateWithFunction: SEL = undefined;
pub var sel_newBufferWithLength: SEL = undefined;
pub var sel_commandBuffer: SEL = undefined;
pub var sel_computeCommandEncoder: SEL = undefined;
pub var sel_setComputePipelineState: SEL = undefined;
pub var sel_setBuffer: SEL = undefined;
pub var sel_dispatchThreads: SEL = undefined;
pub var sel_dispatchThreadgroups: SEL = undefined;
pub var sel_endEncoding: SEL = undefined;
pub var sel_commit: SEL = undefined;
pub var sel_waitUntilCompleted: SEL = undefined;
pub var sel_contents: SEL = undefined;
pub var sel_length: SEL = undefined;
pub var sel_release: SEL = undefined;
pub var sel_retain: SEL = undefined;

// Device property selectors
pub var sel_name: SEL = undefined;
pub var sel_recommendedMaxWorkingSetSize: SEL = undefined;
pub var sel_maxThreadsPerThreadgroup: SEL = undefined;
pub var sel_maxBufferLength: SEL = undefined;
pub var sel_supportsFamily: SEL = undefined;
pub var sel_registryID: SEL = undefined;
pub var sel_isLowPower: SEL = undefined;
pub var sel_isHeadless: SEL = undefined;
pub var sel_hasUnifiedMemory: SEL = undefined;

// NSString selectors
pub var sel_stringWithUTF8String: SEL = undefined;
pub var sel_UTF8String: SEL = undefined;

// NSArray selectors
pub var sel_count: SEL = undefined;
pub var sel_objectAtIndex: SEL = undefined;

// Pipeline state selectors
pub var sel_maxTotalThreadsPerThreadgroup: SEL = undefined;
pub var sel_threadExecutionWidth: SEL = undefined;

pub var selectors_initialized: bool = false;
