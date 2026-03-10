//! Append-only segments.
//!
//! Uses opaque file handle compatible with Zig 0.16 std.Io.

const std = @import("std");
const core = @import("../core");
const block = @import("block");

pub const SegmentLog = struct {
    allocator: std.mem.Allocator,
    /// Opaque file handle — callers provide and own the file lifecycle.
    /// In Zig 0.16, file I/O is done through `std.Io.Dir` and `std.Io.File`.
    /// This struct stores the path for deferred I/O operations rather than
    /// holding a file descriptor, since `std.fs.File` no longer exists.
    path: []const u8,
    current_offset: u64,

    pub fn init(allocator: std.mem.Allocator, path: []const u8, initial_size: u64) !SegmentLog {
        return SegmentLog{
            .allocator = allocator,
            .path = path,
            .current_offset = initial_size,
        };
    }

    pub fn deinit(self: *SegmentLog) void {
        _ = self;
        // Path is caller-owned; nothing to free.
    }

    pub fn append(self: *SegmentLog, b: block.StoredBlock) !u64 {
        const offset = self.current_offset;
        // In a real implementation, we would encode the block and write it
        // through the Io vtable. For now, we stub it out.
        _ = b;
        self.current_offset += 1;
        return offset;
    }
};
