//! Shared resource tracker for GPU backends to manage allocations and kernels.
const std = @import("std");

/// Generic resource tracker for GPU backends.
/// Tries to unify the boilerplate of tracking device memory and compiled kernels.
pub fn ResourceTracker(comptime Allocation: type, comptime Kernel: type) type {
    return struct {
        allocations: std.ArrayListUnmanaged(Allocation) = .empty,
        kernels: std.ArrayListUnmanaged(Kernel) = .empty,
        allocator: std.mem.Allocator,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{ .allocator = allocator };
        }

        pub fn deinit(self: *Self, comptime freeAlloc: fn (std.mem.Allocator, Allocation) void, comptime destroyKernel: fn (std.mem.Allocator, Kernel) void) void {
            for (self.allocations.items) |alloc| {
                freeAlloc(self.allocator, alloc);
            }
            self.allocations.deinit(self.allocator);

            for (self.kernels.items) |kernel| {
                destroyKernel(self.allocator, kernel);
            }
            self.kernels.deinit(self.allocator);
        }

        pub fn trackAllocation(self: *Self, alloc: Allocation) !void {
            try self.allocations.append(self.allocator, alloc);
        }

        pub fn untrackAllocation(self: *Self, ptr: *anyopaque, comptime getPtr: fn (Allocation) *anyopaque) ?Allocation {
            for (self.allocations.items, 0..) |alloc, i| {
                if (getPtr(alloc) == ptr) {
                    return self.allocations.swapRemove(i);
                }
            }
            return null;
        }

        pub fn trackKernel(self: *Self, kernel: Kernel) !void {
            try self.kernels.append(self.allocator, kernel);
        }

        pub fn untrackKernel(self: *Self, handle: *anyopaque, comptime getHandle: fn (Kernel) *anyopaque) ?Kernel {
            for (self.kernels.items, 0..) |k, i| {
                if (getHandle(k) == handle) {
                    return self.kernels.swapRemove(i);
                }
            }
            return null;
        }
    };
}

test "ResourceTracker basic usage" {
    const allocator = std.testing.allocator;
    const DummyAlloc = struct { ptr: *anyopaque };
    const DummyKernel = struct { handle: *anyopaque, name: []const u8 };

    var tracker = ResourceTracker(DummyAlloc, DummyKernel).init(allocator);

    var ptr1: u8 = 1;
    try tracker.trackAllocation(.{ .ptr = &ptr1 });
    try std.testing.expectEqual(@as(usize, 1), tracker.allocations.items.len);

    const untracked = tracker.untrackAllocation(&ptr1, struct {
        fn get(a: DummyAlloc) *anyopaque {
            return a.ptr;
        }
    }.get);
    try std.testing.expect(untracked != null);
    try std.testing.expectEqual(@as(usize, 0), tracker.allocations.items.len);

    tracker.deinit(struct {
        fn f(_: std.mem.Allocator, _: DummyAlloc) void {}
    }.f, struct {
        fn f(_: std.mem.Allocator, _: DummyKernel) void {}
    }.f);
}
