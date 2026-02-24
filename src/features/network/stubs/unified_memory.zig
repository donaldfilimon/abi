const std = @import("std");

pub const UnifiedMemoryManager = struct {
    pub fn init(_: std.mem.Allocator, _: UnifiedMemoryConfig) !@This() {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const UnifiedMemoryConfig = struct {
    max_regions: usize = 256,
    default_region_size: usize = 4096,
    coherence: CoherenceProtocol = .msi,
};

pub const UnifiedMemoryError = error{
    NetworkDisabled,
    RegionNotFound,
    OutOfMemory,
    CoherenceViolation,
};

pub const MemoryRegion = struct {
    id: RegionId = .{ .value = 0 },
    size: usize = 0,
    state: RegionState = .invalid,
};

pub const RegionId = struct {
    value: u64 = 0,
};

pub const RegionFlags = packed struct {
    readable: bool = true,
    writable: bool = true,
    executable: bool = false,
    shared: bool = false,
    _padding: u4 = 0,
};

pub const RegionState = enum { invalid, shared, exclusive, modified };

pub const CoherenceProtocol = enum { msi, mesi, moesi };

pub const CoherenceState = enum { modified, owned, exclusive, shared, invalid };

pub const RemotePtr = struct {
    node_id: []const u8 = "",
    region_id: RegionId = .{ .value = 0 },
    offset: usize = 0,
};

pub const RemoteSlice = struct {
    ptr: RemotePtr = .{},
    len: usize = 0,
};

pub const MemoryNode = struct {
    id: []const u8 = "",
    total_memory: usize = 0,
    available_memory: usize = 0,
};

test {
    std.testing.refAllDecls(@This());
}
