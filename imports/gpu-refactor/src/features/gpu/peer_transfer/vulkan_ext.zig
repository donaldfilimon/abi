//! Vulkan Extension Function Loader
//!
//! Provides dynamic loading of Vulkan extension functions required for
//! peer-to-peer transfers, including:
//!
//! - VK_KHR_external_memory_fd (Linux)
//! - VK_KHR_external_memory_win32 (Windows)
//! - VK_KHR_timeline_semaphore (cross-platform)
//!
//! These extensions enable zero-copy GPU-to-GPU transfers and cross-device
//! synchronization via timeline semaphores.

const std = @import("std");
const builtin = @import("builtin");

const vulkan = @import("../backends/vulkan.zig");
const vulkan_types = vulkan;

// Re-export Vulkan types for convenience
pub const VkDevice = vulkan_types.VkDevice;
pub const VkDeviceMemory = vulkan_types.VkDeviceMemory;
pub const VkResult = vulkan_types.VkResult;
pub const VkDeviceSize = vulkan_types.VkDeviceSize;

// ============================================================================
// Platform-specific Handle Types
// ============================================================================

/// External memory handle type (platform-specific)
pub const ExternalMemoryHandle = switch (builtin.os.tag) {
    .windows => std.os.windows.HANDLE,
    .linux => std.posix.fd_t,
    else => *anyopaque,
};

// ============================================================================
// Vulkan Structure Type Constants
// ============================================================================

/// VkStructureType values for external memory and timeline semaphore extensions
pub const VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO = 1000072002;
pub const VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR = 1000074000;
pub const VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR = 1000074001;
pub const VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR = 1000074002;
pub const VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR = 1000073000;
pub const VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR = 1000073001;
pub const VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR = 1000073002;
pub const VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR = 1000073003;
pub const VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO = 1000207002;
pub const VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO = 1000207003;
pub const VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO = 1000207004;
pub const VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO = 1000207005;
pub const VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO = 9;

// ============================================================================
// External Memory Handle Type Bits
// ============================================================================

pub const VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT = 0x00000001;
pub const VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT = 0x00000002;
pub const VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT = 0x00000004;
pub const VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT = 0x00000008;
pub const VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT = 0x00000010;
pub const VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT = 0x00000020;
pub const VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT = 0x00000040;
pub const VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT = 0x00000200;

// ============================================================================
// Semaphore Type Constants
// ============================================================================

pub const VK_SEMAPHORE_TYPE_BINARY = 0;
pub const VK_SEMAPHORE_TYPE_TIMELINE = 1;

// ============================================================================
// Extension Structures
// ============================================================================

/// VkExportMemoryAllocateInfo - Chain to VkMemoryAllocateInfo for exportable memory
pub const VkExportMemoryAllocateInfo = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
    pNext: ?*const anyopaque = null,
    handleTypes: u32 = 0,
};

/// VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO
pub const VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO = 1000072000;

/// VkExternalMemoryBufferCreateInfo - Chain to VkBufferCreateInfo for exportable buffers
pub const VkExternalMemoryBufferCreateInfo = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    handleTypes: u32 = 0,
};

/// VkMemoryGetFdInfoKHR - Info for getting a POSIX file descriptor
pub const VkMemoryGetFdInfoKHR = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
    pNext: ?*const anyopaque = null,
    memory: VkDeviceMemory = undefined,
    handleType: u32 = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
};

/// VkImportMemoryFdInfoKHR - Chain to VkMemoryAllocateInfo to import from fd
pub const VkImportMemoryFdInfoKHR = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
    pNext: ?*const anyopaque = null,
    handleType: u32 = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    fd: std.posix.fd_t = -1,
};

/// VkMemoryFdPropertiesKHR - Properties of external memory from fd
pub const VkMemoryFdPropertiesKHR = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR,
    pNext: ?*anyopaque = null,
    memoryTypeBits: u32 = 0,
};

/// VkMemoryGetWin32HandleInfoKHR - Info for getting a Windows handle
pub const VkMemoryGetWin32HandleInfoKHR = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
    pNext: ?*const anyopaque = null,
    memory: VkDeviceMemory = undefined,
    handleType: u32 = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
};

/// VkImportMemoryWin32HandleInfoKHR - Chain to VkMemoryAllocateInfo to import from Win32 handle
pub const VkImportMemoryWin32HandleInfoKHR = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR,
    pNext: ?*const anyopaque = null,
    handleType: u32 = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
    handle: ?std.os.windows.HANDLE = null,
    name: ?[*:0]const u16 = null,
};

/// VkMemoryWin32HandlePropertiesKHR - Properties of external memory from Win32 handle
pub const VkMemoryWin32HandlePropertiesKHR = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR,
    pNext: ?*anyopaque = null,
    memoryTypeBits: u32 = 0,
};

/// VkSemaphoreTypeCreateInfo - Timeline semaphore creation info
pub const VkSemaphoreTypeCreateInfo = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    semaphoreType: u32 = VK_SEMAPHORE_TYPE_TIMELINE,
    initialValue: u64 = 0,
};

/// VkSemaphoreCreateInfo - Base semaphore creation info
pub const VkSemaphoreCreateInfo = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
};

/// VkSemaphoreWaitInfo - Info for waiting on timeline semaphores
pub const VkSemaphoreWaitInfo = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    semaphoreCount: u32 = 0,
    pSemaphores: ?[*]const *anyopaque = null,
    pValues: ?[*]const u64 = null,
};

/// VkSemaphoreSignalInfo - Info for signaling timeline semaphores
pub const VkSemaphoreSignalInfo = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
    pNext: ?*const anyopaque = null,
    semaphore: *anyopaque = undefined,
    value: u64 = 0,
};

// ============================================================================
// Vulkan Semaphore Handle Type
// ============================================================================

pub const VkSemaphore = *anyopaque;

// ============================================================================
// Function Pointer Types
// ============================================================================

/// vkGetMemoryFdKHR - Export device memory to POSIX file descriptor
pub const VkGetMemoryFdKHRFn = *const fn (
    VkDevice,
    *const VkMemoryGetFdInfoKHR,
    *std.posix.fd_t,
) callconv(.c) VkResult;

/// vkGetMemoryFdPropertiesKHR - Query memory properties from fd
pub const VkGetMemoryFdPropertiesKHRFn = *const fn (
    VkDevice,
    u32, // handleType
    std.posix.fd_t,
    *VkMemoryFdPropertiesKHR,
) callconv(.c) VkResult;

/// vkGetMemoryWin32HandleKHR - Export device memory to Windows handle
pub const VkGetMemoryWin32HandleKHRFn = *const fn (
    VkDevice,
    *const VkMemoryGetWin32HandleInfoKHR,
    *std.os.windows.HANDLE,
) callconv(.c) VkResult;

/// vkGetMemoryWin32HandlePropertiesKHR - Query memory properties from Windows handle
pub const VkGetMemoryWin32HandlePropertiesKHRFn = *const fn (
    VkDevice,
    u32, // handleType
    std.os.windows.HANDLE,
    *VkMemoryWin32HandlePropertiesKHR,
) callconv(.c) VkResult;

/// vkCreateSemaphore - Create semaphore (works for both binary and timeline)
pub const VkCreateSemaphoreFn = *const fn (
    VkDevice,
    *const VkSemaphoreCreateInfo,
    ?*anyopaque, // allocator
    *VkSemaphore,
) callconv(.c) VkResult;

/// vkDestroySemaphore - Destroy semaphore
pub const VkDestroySemaphoreFn = *const fn (
    VkDevice,
    VkSemaphore,
    ?*anyopaque, // allocator
) callconv(.c) void;

/// vkWaitSemaphores - Wait on timeline semaphore values
pub const VkWaitSemaphoresFn = *const fn (
    VkDevice,
    *const VkSemaphoreWaitInfo,
    u64, // timeout
) callconv(.c) VkResult;

/// vkSignalSemaphore - Signal timeline semaphore from host
pub const VkSignalSemaphoreFn = *const fn (
    VkDevice,
    *const VkSemaphoreSignalInfo,
) callconv(.c) VkResult;

/// vkGetSemaphoreCounterValue - Query current timeline semaphore value
pub const VkGetSemaphoreCounterValueFn = *const fn (
    VkDevice,
    VkSemaphore,
    *u64,
) callconv(.c) VkResult;

/// vkGetDeviceProcAddr - Load device extension functions
pub const VkGetDeviceProcAddrFn = *const fn (
    VkDevice,
    [*:0]const u8,
) callconv(.c) ?*const anyopaque;

// ============================================================================
// Extension Function Container
// ============================================================================

/// Container for loaded Vulkan extension functions
pub const VulkanExtFunctions = struct {
    // External memory (Linux)
    vkGetMemoryFdKHR: ?VkGetMemoryFdKHRFn = null,
    vkGetMemoryFdPropertiesKHR: ?VkGetMemoryFdPropertiesKHRFn = null,

    // External memory (Windows)
    vkGetMemoryWin32HandleKHR: ?VkGetMemoryWin32HandleKHRFn = null,
    vkGetMemoryWin32HandlePropertiesKHR: ?VkGetMemoryWin32HandlePropertiesKHRFn = null,

    // Timeline semaphores
    vkCreateSemaphore: ?VkCreateSemaphoreFn = null,
    vkDestroySemaphore: ?VkDestroySemaphoreFn = null,
    vkWaitSemaphores: ?VkWaitSemaphoresFn = null,
    vkSignalSemaphore: ?VkSignalSemaphoreFn = null,
    vkGetSemaphoreCounterValue: ?VkGetSemaphoreCounterValueFn = null,

    // Capabilities
    external_memory_fd_supported: bool = false,
    external_memory_win32_supported: bool = false,
    timeline_semaphores_supported: bool = false,

    const Self = @This();

    /// Load extension functions from a Vulkan device
    pub fn load(device: VkDevice, getProcAddr: VkGetDeviceProcAddrFn) Self {
        var funcs = Self{};

        // Load external memory functions based on platform
        if (builtin.os.tag == .linux) {
            funcs.vkGetMemoryFdKHR = @ptrCast(getProcAddr(device, "vkGetMemoryFdKHR"));
            funcs.vkGetMemoryFdPropertiesKHR = @ptrCast(getProcAddr(device, "vkGetMemoryFdPropertiesKHR"));
            funcs.external_memory_fd_supported = funcs.vkGetMemoryFdKHR != null;
        } else if (builtin.os.tag == .windows) {
            funcs.vkGetMemoryWin32HandleKHR = @ptrCast(getProcAddr(device, "vkGetMemoryWin32HandleKHR"));
            funcs.vkGetMemoryWin32HandlePropertiesKHR = @ptrCast(getProcAddr(device, "vkGetMemoryWin32HandlePropertiesKHR"));
            funcs.external_memory_win32_supported = funcs.vkGetMemoryWin32HandleKHR != null;
        }

        // Load timeline semaphore functions (Vulkan 1.2 core or KHR extension)
        funcs.vkCreateSemaphore = @ptrCast(getProcAddr(device, "vkCreateSemaphore"));
        funcs.vkDestroySemaphore = @ptrCast(getProcAddr(device, "vkDestroySemaphore"));
        funcs.vkWaitSemaphores = @ptrCast(getProcAddr(device, "vkWaitSemaphores"));
        funcs.vkSignalSemaphore = @ptrCast(getProcAddr(device, "vkSignalSemaphore"));
        funcs.vkGetSemaphoreCounterValue = @ptrCast(getProcAddr(device, "vkGetSemaphoreCounterValue"));

        // Try KHR variants if core functions not available
        if (funcs.vkWaitSemaphores == null) {
            funcs.vkWaitSemaphores = @ptrCast(getProcAddr(device, "vkWaitSemaphoresKHR"));
        }
        if (funcs.vkSignalSemaphore == null) {
            funcs.vkSignalSemaphore = @ptrCast(getProcAddr(device, "vkSignalSemaphoreKHR"));
        }
        if (funcs.vkGetSemaphoreCounterValue == null) {
            funcs.vkGetSemaphoreCounterValue = @ptrCast(getProcAddr(device, "vkGetSemaphoreCounterValueKHR"));
        }

        funcs.timeline_semaphores_supported = funcs.vkWaitSemaphores != null and
            funcs.vkSignalSemaphore != null and
            funcs.vkCreateSemaphore != null;

        return funcs;
    }

    /// Load extension functions from Vulkan library directly
    pub fn loadFromLib(lib: std.DynLib, device: VkDevice) Self {
        // First try to get vkGetDeviceProcAddr from the library
        const getProcAddr = lib.lookup(VkGetDeviceProcAddrFn, "vkGetDeviceProcAddr") orelse {
            // Fall back to direct library lookups
            var funcs = Self{};
            funcs.loadDirectFromLib(lib);
            return funcs;
        };

        return Self.load(device, getProcAddr);
    }

    /// Load functions directly from library (fallback)
    fn loadDirectFromLib(self: *Self, lib: std.DynLib) void {
        if (builtin.os.tag == .linux) {
            self.vkGetMemoryFdKHR = lib.lookup(VkGetMemoryFdKHRFn, "vkGetMemoryFdKHR");
            self.vkGetMemoryFdPropertiesKHR = lib.lookup(VkGetMemoryFdPropertiesKHRFn, "vkGetMemoryFdPropertiesKHR");
            self.external_memory_fd_supported = self.vkGetMemoryFdKHR != null;
        } else if (builtin.os.tag == .windows) {
            self.vkGetMemoryWin32HandleKHR = lib.lookup(VkGetMemoryWin32HandleKHRFn, "vkGetMemoryWin32HandleKHR");
            self.vkGetMemoryWin32HandlePropertiesKHR = lib.lookup(VkGetMemoryWin32HandlePropertiesKHRFn, "vkGetMemoryWin32HandlePropertiesKHR");
            self.external_memory_win32_supported = self.vkGetMemoryWin32HandleKHR != null;
        }

        self.vkCreateSemaphore = lib.lookup(VkCreateSemaphoreFn, "vkCreateSemaphore");
        self.vkDestroySemaphore = lib.lookup(VkDestroySemaphoreFn, "vkDestroySemaphore");
        self.vkWaitSemaphores = lib.lookup(VkWaitSemaphoresFn, "vkWaitSemaphores") orelse
            lib.lookup(VkWaitSemaphoresFn, "vkWaitSemaphoresKHR");
        self.vkSignalSemaphore = lib.lookup(VkSignalSemaphoreFn, "vkSignalSemaphore") orelse
            lib.lookup(VkSignalSemaphoreFn, "vkSignalSemaphoreKHR");
        self.vkGetSemaphoreCounterValue = lib.lookup(VkGetSemaphoreCounterValueFn, "vkGetSemaphoreCounterValue") orelse
            lib.lookup(VkGetSemaphoreCounterValueFn, "vkGetSemaphoreCounterValueKHR");

        self.timeline_semaphores_supported = self.vkWaitSemaphores != null and
            self.vkSignalSemaphore != null and
            self.vkCreateSemaphore != null;
    }

    /// Check if external memory export is supported on this platform
    pub fn hasExternalMemorySupport(self: *const Self) bool {
        return switch (builtin.os.tag) {
            .linux => self.external_memory_fd_supported,
            .windows => self.external_memory_win32_supported,
            else => false,
        };
    }

    /// Get the handle type for this platform
    pub fn getPlatformHandleType() u32 {
        return switch (builtin.os.tag) {
            .linux => VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
            .windows => VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
            else => 0,
        };
    }
};

// ============================================================================
// Global Extension Functions State
// ============================================================================

var ext_functions: ?VulkanExtFunctions = null;
var ext_functions_device: ?VkDevice = null;

/// Initialize extension functions for a device
pub fn initExtFunctions(device: VkDevice) !void {
    // If already initialized for this device, skip
    if (ext_functions_device) |existing| {
        if (existing == device) return;
    }

    const lib = vulkan.vulkan_lib orelse return error.VulkanNotLoaded;

    ext_functions = VulkanExtFunctions.loadFromLib(lib, device);
    ext_functions_device = device;
}

/// Get loaded extension functions
pub fn getExtFunctions() ?*const VulkanExtFunctions {
    if (ext_functions) |*funcs| {
        return funcs;
    }
    return null;
}

/// Reset extension functions state
pub fn deinitExtFunctions() void {
    ext_functions = null;
    ext_functions_device = null;
}

// ============================================================================
// Tests
// ============================================================================

test "VulkanExtFunctions default values" {
    const funcs = VulkanExtFunctions{};

    try std.testing.expect(funcs.vkGetMemoryFdKHR == null);
    try std.testing.expect(funcs.vkWaitSemaphores == null);
    try std.testing.expect(!funcs.timeline_semaphores_supported);
    try std.testing.expect(!funcs.hasExternalMemorySupport());
}

test "platform handle type" {
    const handle_type = VulkanExtFunctions.getPlatformHandleType();

    if (builtin.os.tag == .linux) {
        try std.testing.expectEqual(VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT, handle_type);
    } else if (builtin.os.tag == .windows) {
        try std.testing.expectEqual(VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT, handle_type);
    } else {
        try std.testing.expectEqual(@as(u32, 0), handle_type);
    }
}

test "structure type constants" {
    // Verify structure type constants match Vulkan spec
    try std.testing.expectEqual(@as(i32, 1000207002), VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO);
    try std.testing.expectEqual(@as(i32, 1000207004), VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO);
    try std.testing.expectEqual(@as(i32, 1000207005), VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO);
}

test "VkSemaphoreTypeCreateInfo initialization" {
    const type_info = VkSemaphoreTypeCreateInfo{
        .initialValue = 42,
    };

    try std.testing.expectEqual(VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, type_info.sType);
    try std.testing.expectEqual(VK_SEMAPHORE_TYPE_TIMELINE, type_info.semaphoreType);
    try std.testing.expectEqual(@as(u64, 42), type_info.initialValue);
}
