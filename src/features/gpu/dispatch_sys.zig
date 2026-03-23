const std = @import("std");
const backend_shared = @import("backends/shared.zig");

pub const cuda_loader = if (backend_shared.dynlibSupported)
    @import("backends/cuda/loader.zig")
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
        pub fn load(_: std.mem.Allocator) error{PlatformNotSupported}!*const CudaFunctions {
            return error.PlatformNotSupported;
        }
        pub fn unload() void {}
        pub fn getFunctions() ?*const CudaFunctions {
            return null;
        }
        pub fn isAvailableWithAlloc(_: std.mem.Allocator) bool {
            return false;
        }
    };
