//! Memory Region Management
//!
//! Defines memory regions that can be shared across nodes in a unified
//! memory system. Regions track ownership, permissions, and coherence state.

const std = @import("std");
const shared_utils = @import("../../../services/shared/utils.zig");

/// Unique identifier for a memory region.
pub const RegionId = u64;

/// Memory region permissions and flags.
pub const RegionFlags = struct {
    /// Allow read access.
    read: bool = true,
    /// Allow write access.
    write: bool = false,
    /// Allow execute (for code regions).
    execute: bool = false,
    /// Region can be migrated between nodes.
    migratable: bool = true,
    /// Region is pinned in physical memory.
    pinned: bool = false,
    /// Region uses huge pages.
    huge_pages: bool = false,
    /// Region is copy-on-write.
    copy_on_write: bool = false,
    /// Region is persistent (survives restarts).
    persistent: bool = false,
    /// Region is encrypted at rest.
    encrypted: bool = false,

    /// Read-only access.
    pub const read_only: RegionFlags = .{ .read = true, .write = false };

    /// Read-write access.
    pub const read_write: RegionFlags = .{ .read = true, .write = true };

    /// Full access including execute.
    pub const full_access: RegionFlags = .{ .read = true, .write = true, .execute = true };

    /// Convert to permission bits for comparison.
    pub fn toBits(self: RegionFlags) u16 {
        var bits: u16 = 0;
        if (self.read) bits |= 0x001;
        if (self.write) bits |= 0x002;
        if (self.execute) bits |= 0x004;
        if (self.migratable) bits |= 0x008;
        if (self.pinned) bits |= 0x010;
        if (self.huge_pages) bits |= 0x020;
        if (self.copy_on_write) bits |= 0x040;
        if (self.persistent) bits |= 0x080;
        if (self.encrypted) bits |= 0x100;
        return bits;
    }

    /// Create flags from permission bits.
    pub fn fromBits(bits: u16) RegionFlags {
        return .{
            .read = (bits & 0x001) != 0,
            .write = (bits & 0x002) != 0,
            .execute = (bits & 0x004) != 0,
            .migratable = (bits & 0x008) != 0,
            .pinned = (bits & 0x010) != 0,
            .huge_pages = (bits & 0x020) != 0,
            .copy_on_write = (bits & 0x040) != 0,
            .persistent = (bits & 0x080) != 0,
            .encrypted = (bits & 0x100) != 0,
        };
    }
};

/// Memory region coherence state.
pub const RegionState = enum(u8) {
    /// Region is not valid (needs to be fetched).
    invalid = 0,
    /// Region is shared (read-only copy).
    shared = 1,
    /// Region is exclusive (only this node has it).
    exclusive = 2,
    /// Region is modified (dirty, needs writeback).
    modified = 3,
    /// Region is owned (can be read by others, but we're owner).
    owned = 4,
    /// Region is being transferred.
    transferring = 5,
    /// Region is locked for exclusive access.
    locked = 6,

    pub fn canRead(self: RegionState) bool {
        return switch (self) {
            .shared, .exclusive, .modified, .owned => true,
            .invalid, .transferring, .locked => false,
        };
    }

    pub fn canWrite(self: RegionState) bool {
        return switch (self) {
            .exclusive, .modified => true,
            .invalid, .shared, .owned, .transferring, .locked => false,
        };
    }

    pub fn needsInvalidation(self: RegionState) bool {
        return switch (self) {
            .modified, .owned => true,
            .invalid, .shared, .exclusive, .transferring, .locked => false,
        };
    }
};

/// Statistics for a memory region.
pub const RegionStats = struct {
    /// Number of read operations.
    reads: u64 = 0,
    /// Number of write operations.
    writes: u64 = 0,
    /// Number of remote reads served.
    remote_reads_served: u64 = 0,
    /// Number of remote writes received.
    remote_writes_received: u64 = 0,
    /// Number of cache invalidations.
    invalidations: u64 = 0,
    /// Number of ownership transfers.
    ownership_transfers: u64 = 0,
    /// Bytes read locally.
    bytes_read: u64 = 0,
    /// Bytes written locally.
    bytes_written: u64 = 0,
    /// Bytes transferred to remote.
    bytes_transferred_out: u64 = 0,
    /// Bytes received from remote.
    bytes_transferred_in: u64 = 0,
    /// Creation timestamp (ms since epoch).
    created_at_ms: i64 = 0,
    /// Last access timestamp.
    last_access_ms: i64 = 0,
    /// Last modification timestamp.
    last_modified_ms: i64 = 0,

    pub fn recordRead(self: *RegionStats, bytes: usize) void {
        self.reads += 1;
        self.bytes_read += bytes;
        self.last_access_ms = shared_utils.unixMs();
    }

    pub fn recordWrite(self: *RegionStats, bytes: usize) void {
        self.writes += 1;
        self.bytes_written += bytes;
        self.last_access_ms = shared_utils.unixMs();
        self.last_modified_ms = self.last_access_ms;
    }
};

/// A memory region that can be shared across nodes.
pub const MemoryRegion = struct {
    /// Unique region identifier.
    id: RegionId,

    /// Base pointer to the memory.
    base_ptr: [*]u8,

    /// Size of the region in bytes.
    size: usize,

    /// Access flags.
    flags: RegionFlags,

    /// Current coherence state.
    state: RegionState,

    /// Node ID of the owner (0 = local).
    owner_node: u64,

    /// Page size used for this region.
    page_size: usize,

    /// Per-page states (for fine-grained coherence).
    page_states: ?[]RegionState = null,

    /// Per-page dirty bits.
    dirty_pages: ?std.DynamicBitSetUnmanaged = null,

    /// Region statistics.
    stats: RegionStats,

    /// User-defined metadata.
    metadata: ?[]const u8 = null,

    /// Lock for thread-safe access.
    lock: std.Thread.Mutex = .{},

    /// Initialize a new memory region.
    pub fn init(
        allocator: std.mem.Allocator,
        id: RegionId,
        ptr: [*]u8,
        size: usize,
        flags: RegionFlags,
        page_size: usize,
    ) !MemoryRegion {
        const num_pages = (size + page_size - 1) / page_size;

        var region = MemoryRegion{
            .id = id,
            .base_ptr = ptr,
            .size = size,
            .flags = flags,
            .state = .exclusive,
            .owner_node = 0,
            .page_size = page_size,
            .stats = .{ .created_at_ms = shared_utils.unixMs() },
        };

        // Allocate per-page tracking if needed
        if (num_pages > 1) {
            region.page_states = try allocator.alloc(RegionState, num_pages);
            @memset(region.page_states.?, .exclusive);

            region.dirty_pages = try std.DynamicBitSetUnmanaged.initEmpty(allocator, num_pages);
        }

        return region;
    }

    /// Deinitialize and free tracking structures.
    pub fn deinit(self: *MemoryRegion, allocator: std.mem.Allocator) void {
        if (self.page_states) |ps| {
            allocator.free(ps);
        }
        if (self.dirty_pages) |*dp| {
            dp.deinit(allocator);
        }
        if (self.metadata) |m| {
            allocator.free(m);
        }
        self.* = undefined;
    }

    /// Get the number of pages in this region.
    pub fn pageCount(self: *const MemoryRegion) usize {
        return (self.size + self.page_size - 1) / self.page_size;
    }

    /// Get page index for an offset.
    pub fn pageIndex(self: *const MemoryRegion, offset: usize) usize {
        return offset / self.page_size;
    }

    /// Get the state of a specific page.
    pub fn getPageState(self: *const MemoryRegion, page_idx: usize) RegionState {
        if (self.page_states) |ps| {
            if (page_idx < ps.len) {
                return ps[page_idx];
            }
        }
        return self.state;
    }

    /// Set the state of a specific page.
    pub fn setPageState(self: *MemoryRegion, page_idx: usize, new_state: RegionState) void {
        if (self.page_states) |ps| {
            if (page_idx < ps.len) {
                ps[page_idx] = new_state;
            }
        } else {
            self.state = new_state;
        }
    }

    /// Mark a page as dirty.
    pub fn markDirty(self: *MemoryRegion, page_idx: usize) void {
        if (self.dirty_pages) |*dp| {
            dp.set(page_idx);
        }
        self.setPageState(page_idx, .modified);
    }

    /// Check if a page is dirty.
    pub fn isDirty(self: *const MemoryRegion, page_idx: usize) bool {
        if (self.dirty_pages) |dp| {
            return dp.isSet(page_idx);
        }
        return self.state == .modified;
    }

    /// Clear dirty bit for a page (after writeback).
    pub fn clearDirty(self: *MemoryRegion, page_idx: usize) void {
        if (self.dirty_pages) |*dp| {
            dp.unset(page_idx);
        }
    }

    /// Get slice of the region's memory.
    pub fn slice(self: *MemoryRegion) []u8 {
        return self.base_ptr[0..self.size];
    }

    /// Get slice at offset.
    pub fn sliceAt(self: *MemoryRegion, offset: usize, len: usize) ?[]u8 {
        if (offset + len > self.size) return null;
        return self.base_ptr[offset .. offset + len];
    }

    /// Read from the region.
    pub fn read(self: *MemoryRegion, offset: usize, buffer: []u8) !void {
        self.lock.lock();
        defer self.lock.unlock();

        if (!self.flags.read) return error.PermissionDenied;
        if (offset + buffer.len > self.size) return error.OutOfBounds;

        const page_idx = self.pageIndex(offset);
        const page_state = self.getPageState(page_idx);
        if (!page_state.canRead()) return error.InvalidState;

        @memcpy(buffer, self.base_ptr[offset .. offset + buffer.len]);
        self.stats.recordRead(buffer.len);
    }

    /// Write to the region.
    pub fn write(self: *MemoryRegion, offset: usize, data: []const u8) !void {
        self.lock.lock();
        defer self.lock.unlock();

        if (!self.flags.write) return error.PermissionDenied;
        if (offset + data.len > self.size) return error.OutOfBounds;

        const page_idx = self.pageIndex(offset);
        const page_state = self.getPageState(page_idx);
        if (!page_state.canWrite()) return error.InvalidState;

        @memcpy(self.base_ptr[offset .. offset + data.len], data);
        self.markDirty(page_idx);
        self.stats.recordWrite(data.len);
    }

    /// Check if the region can be read.
    pub fn canRead(self: *const MemoryRegion) bool {
        return self.flags.read and self.state.canRead();
    }

    /// Check if the region can be written.
    pub fn canWrite(self: *const MemoryRegion) bool {
        return self.flags.write and self.state.canWrite();
    }

    /// Get region information as a struct for serialization.
    pub fn getInfo(self: *const MemoryRegion) RegionInfo {
        return .{
            .id = self.id,
            .size = self.size,
            .flags = self.flags.toBits(),
            .state = self.state,
            .owner_node = self.owner_node,
            .page_count = self.pageCount(),
            .stats = self.stats,
        };
    }

    /// Region information for network transfer.
    pub const RegionInfo = struct {
        id: RegionId,
        size: usize,
        flags: u16,
        state: RegionState,
        owner_node: u64,
        page_count: usize,
        stats: RegionStats,
    };

    pub const Error = error{
        PermissionDenied,
        OutOfBounds,
        InvalidState,
    };
};

/// Page descriptor for fine-grained memory management.
pub const PageDescriptor = struct {
    /// Page index within region.
    index: usize,
    /// Physical address (if pinned).
    physical_addr: ?u64,
    /// Current state.
    state: RegionState,
    /// Last accessing node.
    last_accessor: u64,
    /// Access count.
    access_count: u32,
    /// Flags.
    flags: PageFlags,

    pub const PageFlags = struct {
        dirty: bool = false,
        pinned: bool = false,
        accessed: bool = false,
        present: bool = true,
    };
};

// ============================================================================
// Tests
// ============================================================================

test "RegionFlags bit conversion" {
    const flags = RegionFlags{ .read = true, .write = true, .execute = false, .migratable = true };
    const bits = flags.toBits();
    const restored = RegionFlags.fromBits(bits);

    try std.testing.expect(restored.read);
    try std.testing.expect(restored.write);
    try std.testing.expect(!restored.execute);
    try std.testing.expect(restored.migratable);
}

test "RegionState access checks" {
    try std.testing.expect(RegionState.shared.canRead());
    try std.testing.expect(!RegionState.shared.canWrite());
    try std.testing.expect(RegionState.exclusive.canRead());
    try std.testing.expect(RegionState.exclusive.canWrite());
    try std.testing.expect(!RegionState.invalid.canRead());
    try std.testing.expect(RegionState.modified.needsInvalidation());
}

test "MemoryRegion basic operations" {
    const allocator = std.testing.allocator;

    var data: [8192]u8 = undefined;
    @memset(&data, 0xAB);

    var region = try MemoryRegion.init(
        allocator,
        1,
        &data,
        data.len,
        RegionFlags.read_write,
        4096,
    );
    defer region.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), region.pageCount());
    try std.testing.expect(region.canRead());
    try std.testing.expect(region.canWrite());

    // Test read
    var buffer: [16]u8 = undefined;
    try region.read(0, &buffer);
    try std.testing.expectEqual(@as(u8, 0xAB), buffer[0]);

    // Test write
    try region.write(0, "Hello, World!");
    try region.read(0, &buffer);
    try std.testing.expectEqualSlices(u8, "Hello, World!\xab\xab\xab", &buffer);
}

test "MemoryRegion page state tracking" {
    const allocator = std.testing.allocator;

    var data: [16384]u8 = undefined;
    var region = try MemoryRegion.init(
        allocator,
        1,
        &data,
        data.len,
        RegionFlags.read_write,
        4096,
    );
    defer region.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 4), region.pageCount());
    try std.testing.expectEqual(RegionState.exclusive, region.getPageState(0));

    region.setPageState(1, .shared);
    try std.testing.expectEqual(RegionState.shared, region.getPageState(1));

    region.markDirty(2);
    try std.testing.expect(region.isDirty(2));
    try std.testing.expectEqual(RegionState.modified, region.getPageState(2));

    region.clearDirty(2);
    try std.testing.expect(!region.isDirty(2));
}
