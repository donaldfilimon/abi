//! Common VTable Backend Wrapper logic
//!
//! Provides a generic structure for backend implementations to reduce
//! boilerplate across different GPU backends.

const std = @import("std");
const interface = @import("../interface.zig");

pub fn BackendWrapper(comptime BackendImpl: type) type {
    return struct {
        allocator: std.mem.Allocator,
        initialized: bool,
        impl: *BackendImpl,

        allocations: std.ArrayListUnmanaged(Allocation) = .empty,
        kernels: std.ArrayListUnmanaged(CompiledKernel) = .empty,

        const Allocation = struct {
            ptr: *anyopaque,
            size: usize,
        };

        const CompiledKernel = struct {
            handle: *anyopaque,
            name: []const u8,
        };

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, impl: *BackendImpl) !*Self {
            const self = try allocator.create(Self);
            self.* = .{
                .allocator = allocator,
                .initialized = true,
                .impl = impl,
            };
            return self;
        }

        pub fn deinit(self: *Self, deinitImplFn: *const fn (*BackendImpl) void) void {
            for (self.allocations.items) |alloc| {
                self.impl.free(alloc.ptr);
            }
            self.allocations.deinit(self.allocator);

            for (self.kernels.items) |kernel| {
                self.impl.destroyKernel(kernel.handle);
                self.allocator.free(kernel.name);
            }
            self.kernels.deinit(self.allocator);

            deinitImplFn(self.impl);
            self.allocator.destroy(self);
        }

        pub fn trackAllocation(self: *Self, ptr: *anyopaque, size: usize) !void {
            try self.allocations.append(self.allocator, .{ .ptr = ptr, .size = size });
        }

        pub fn untrackAllocation(self: *Self, ptr: *anyopaque) void {
            for (self.allocations.items, 0..) |alloc, i| {
                if (alloc.ptr == ptr) {
                    _ = self.allocations.swapRemove(i);
                    return;
                }
            }
        }

        pub fn trackKernel(self: *Self, handle: *anyopaque, name: []const u8) !void {
            const name_copy = try self.allocator.dupe(u8, name);
            try self.kernels.append(self.allocator, .{ .handle = handle, .name = name_copy });
        }

        pub fn untrackKernel(self: *Self, handle: *anyopaque) void {
            for (self.kernels.items, 0..) |k, i| {
                if (k.handle == handle) {
                    self.allocator.free(k.name);
                    _ = self.kernels.swapRemove(i);
                    return;
                }
            }
        }
    };
}
