pub const unified = @import("unified.zig");
pub const unified_buffer = @import("unified_buffer.zig");
pub const device = @import("device.zig");
pub const devices = @import("device.zig");
pub const stream = @import("stream.zig");
pub const dsl = @import("../dsl/mod.zig");
pub const runtime = @import("../runtime/mod.zig");
pub const factory = @import("../factory/mod.zig");
pub const interface = @import("interface.zig");
pub const platform = @import("platform.zig");
pub const backends = @import("../backends/mod.zig");
pub const dispatch = @import("../dispatch/mod.zig");
pub const multi = @import("multi.zig");
pub const multi_device = @import("multi_device.zig");
pub const policy = @import("../policy/mod.zig");
pub const backend = @import("backend.zig");
pub const backend_shared = @import("../backends/shared.zig");
pub const cuda_loader = if (backend_shared.dynlibSupported)
    @import("../backends/cuda/loader.zig")
else
    struct {
        pub const CuResult = enum(i32) { success = 0, _ };
        pub const CoreFunctions = struct {
            cuInit: ?*const fn (u32) callconv(.c) CuResult = null,
            cuDeviceGetCount: ?*const fn (*i32) callconv(.c) CuResult = null,
        };
        pub const CudaFunctions = struct {
            core: CoreFunctions = .{},
        };
        pub fn load(_: @import("std").mem.Allocator) error{PlatformNotSupported}!*const CudaFunctions {
            return error.PlatformNotSupported;
        }
        pub fn unload() void {}
        pub fn getFunctions() ?*const CudaFunctions {
            return null;
        }
        pub fn isAvailableWithAlloc(_: @import("std").mem.Allocator) bool {
            return false;
        }
    };
