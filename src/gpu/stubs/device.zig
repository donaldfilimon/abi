const std = @import("std");
const backend_mod = @import("backend.zig");

pub const Device = struct {
    id: u32 = 0,
    backend: backend_mod.Backend = .cpu,
    name: []const u8 = "disabled",
};
pub const DeviceType = enum { cpu, gpu, accelerator };
pub const DeviceCapability = struct {
    unified_memory: bool = false,
    supports_fp16: bool = false,
    supports_int8: bool = false,
    supports_async_transfers: bool = false,
    max_threads_per_block: ?u32 = null,
    max_shared_memory_bytes: ?u32 = null,
};
pub const DeviceInfo = struct {
    id: u32 = 0,
    backend: backend_mod.Backend = .cpu,
    name: []const u8 = "disabled",
    total_memory_bytes: ?u64 = null,
    is_emulated: bool = true,
    capability: DeviceCapability = .{},
    device_type: DeviceType = .cpu,
};
pub const DeviceFeature = enum { compute, graphics };
pub const DeviceSelector = struct {};
pub const DeviceManager = struct {};
