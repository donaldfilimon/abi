//! Merge, dedupe, reclaim, rewrite.

const std = @import("std");
const block = @import("block.zig");
const codec = @import("codec.zig");
const segment_log = @import("segment_log.zig");

pub const CompactionJob = struct {
    pub const Status = enum { pending, running, completed, failed };

    status: Status = .pending,
    blocks_processed: usize = 0,
    bytes_reclaimed: u64 = 0,
    /// Optional segment log for on-disk operations.
    log: ?*segment_log.SegmentLog = null,

    /// Merge blocks into a single consolidated block.
    ///
    /// When a `log` is attached, each content hash in `block_offsets` is
    /// treated as a u64 LE file offset packed into the first 8 bytes of the
    /// 32-byte hash.  The block is read from the log, and the combined
    /// payloads are appended as a new block.
    ///
    /// When no log is set, the function falls back to counting blocks
    /// (legacy in-memory behaviour).
    pub fn merge(self: *CompactionJob, allocator: std.mem.Allocator, blocks: []const [32]u8) !void {
        self.status = .running;

        if (self.log) |seg| {
            // Collect payloads from the segment file.
            var combined = std.ArrayListUnmanaged(u8).empty;
            defer combined.deinit(allocator);

            for (blocks) |hash| {
                // Interpret the first 8 bytes as a u64 LE offset.
                const offset = std.mem.readInt(u64, hash[0..8], .little);
                const stored = seg.readAt(offset) catch {
                    self.blocks_processed += 1;
                    continue;
                };
                defer allocator.free(stored.payload);

                try combined.appendSlice(allocator, stored.payload);
                self.blocks_processed += 1;
            }

            // Append a consolidated block back to the log.
            if (combined.items.len > 0) {
                const consolidated = block.StoredBlock{
                    .header = .{
                        .id = .{ .id = [_]u8{0} ** 32 },
                        .kind = @enumFromInt(0),
                        .version = 1,
                        .content_hash = [_]u8{0} ** 32,
                        .timestamp = .{ .counter = 0 },
                        .size = @intCast(combined.items.len),
                        .flags = 0,
                        .compression_marker = 0,
                    },
                    .payload = combined.items,
                };
                _ = try seg.append(consolidated);
            }
        } else {
            // Legacy in-memory simulation.
            for (blocks) |_| {
                self.blocks_processed += 1;
            }
            _ = allocator;
        }

        self.status = .completed;
    }

    /// Deduplicate blocks by content hash. When a log is attached, only the
    /// first occurrence of each content hash is kept; duplicates are counted
    /// as reclaimed.
    pub fn dedupe(self: *CompactionJob, allocator: std.mem.Allocator, blocks: []const [32]u8) !void {
        self.status = .running;

        var seen: std.AutoHashMap([32]u8, void) = .empty;
        defer seen.deinit(allocator);

        if (self.log) |seg| {
            for (blocks) |hash| {
                self.blocks_processed += 1;
                const offset = std.mem.readInt(u64, hash[0..8], .little);

                const stored = seg.readAt(offset) catch continue;
                defer allocator.free(stored.payload);

                const res = try seen.getOrPut(allocator, stored.header.content_hash);
                if (res.found_existing) {
                    self.bytes_reclaimed += stored.payload.len;
                }
            }
        } else {
            for (blocks) |block_id| {
                self.blocks_processed += 1;
                const res = try seen.getOrPut(allocator, block_id);
                if (res.found_existing) {
                    self.bytes_reclaimed += 1024; // Simulated reclaimed bytes
                }
            }
        }

        self.status = .completed;
    }

    /// Rewrite the segment log by reading all frames sequentially and
    /// writing them to a new temporary file, then renaming.
    pub fn rewrite(self: *CompactionJob, allocator: std.mem.Allocator) !void {
        self.status = .running;

        if (self.log) |seg| {
            const tmp_path = try std.fmt.allocPrint(allocator, "{s}.tmp", .{seg.path});
            defer allocator.free(tmp_path);

            var io_backend = std.Io.Threaded.init(allocator, .{
                .environ = std.process.Environ.empty,
            });
            defer io_backend.deinit();
            const io = io_backend.io();

            // Clean up any stale temp file.
            defer std.Io.Dir.cwd().deleteFile(io, tmp_path) catch {};

            var new_log = try segment_log.SegmentLog.init(allocator, tmp_path, 0);
            defer new_log.deinit();

            // Walk the original log frame-by-frame.
            var cursor: u64 = 0;
            while (cursor < seg.current_offset) {
                const stored = seg.readAt(cursor) catch break;
                defer allocator.free(stored.payload);

                _ = try new_log.append(stored);

                // Advance cursor: 4-byte length prefix + encoded block size.
                const encoded = try codec.encodeBlock(allocator, stored);
                defer allocator.free(encoded);
                cursor += @as(u64, 4) + @as(u64, encoded.len);
            }

            // Atomic rename tmp -> original.
            std.Io.Dir.rename(std.Io.Dir.cwd(), tmp_path, std.Io.Dir.cwd(), seg.path, io) catch {
                self.status = .failed;
                return;
            };
            seg.current_offset = new_log.current_offset;
        } else {
            _ = allocator;
        }

        self.status = .completed;
    }
};
