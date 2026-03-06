//! Append-only segments.

const std = @import("std");
const core = @import("../core/mod.zig");
const block = @import("block.zig");

pub const SegmentLog = struct {
    allocator: std.mem.Allocator,
    file: std.fs.File,
    current_offset: u64,

    pub fn init(allocator: std.mem.Allocator, file: std.fs.File) !SegmentLog {
        const stat = try file.stat();
        return SegmentLog{
            .allocator = allocator,
            .file = file,
            .current_offset = stat.size,
        };
    }

    pub fn deinit(self: *SegmentLog) void {
        self.file.close();
    }

    pub fn append(self: *SegmentLog, b: block.StoredBlock) !u64 {
        const offset = self.current_offset;
        // In a real implementation, we would encode the block and write it.
        // For now, we stub it out.
        _ = b;
        self.current_offset += 1;
        return offset;
    }
};
