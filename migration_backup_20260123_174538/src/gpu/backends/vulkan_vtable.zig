//! Minimal Vulkan VTable backend stub.
//!
//! The full VTable implementation lives in the original `vulkan_vtable.zig`
//! file which has been removed for consolidation. To keep the rest of the
//! codebase compiling, we provide a lightweight stub that satisfies the
//! expected API surface. All functions simply return `BackendError.NotAvailable`
//! indicating that the VTable backend is not currently usable.

const std = @import("std");
const interface = @import("../interface.zig");

pub fn createVulkanVTable(allocator: std.mem.Allocator) interface.BackendError!interface.Backend {
    _ = allocator;
    return error.NotAvailable;
}
