const std = @import("std");

pub fn Persistence(
    comptime Manager: type,
    comptime OperationType: type,
) type {
    return struct {
        pub fn saveOperationLog(self: *Manager, path: []const u8) !void {
            self.mutex.lock();
            defer self.mutex.unlock();

            var buf = std.ArrayListUnmanaged(u8).empty;
            defer buf.deinit(self.allocator);

            const count: u64 = self.operation_log.items.len;
            try buf.appendSlice(self.allocator, &std.mem.toBytes(@as(u64, count)));

            for (self.operation_log.items) |op| {
                try buf.appendSlice(self.allocator, &std.mem.toBytes(op.timestamp));
                try buf.appendSlice(self.allocator, &std.mem.toBytes(op.sequence_number));
                try buf.append(self.allocator, @intFromEnum(op.type));

                const key_len: u32 = @intCast(op.key.len);
                try buf.appendSlice(self.allocator, &std.mem.toBytes(key_len));
                try buf.appendSlice(self.allocator, op.key);

                if (op.value) |v| {
                    const val_len: u32 = @intCast(v.len);
                    try buf.appendSlice(self.allocator, &std.mem.toBytes(val_len));
                    try buf.appendSlice(self.allocator, v);
                } else {
                    try buf.appendSlice(self.allocator, &std.mem.toBytes(@as(u32, std.math.maxInt(u32))));
                }

                if (op.previous_value) |v| {
                    const prev_len: u32 = @intCast(v.len);
                    try buf.appendSlice(self.allocator, &std.mem.toBytes(prev_len));
                    try buf.appendSlice(self.allocator, v);
                } else {
                    try buf.appendSlice(self.allocator, &std.mem.toBytes(@as(u32, std.math.maxInt(u32))));
                }
            }

            try atomicWriteFile(self.allocator, path, buf.items);
        }

        pub fn saveCheckpoints(self: *Manager, path: []const u8) !void {
            self.mutex.lock();
            defer self.mutex.unlock();

            var buf = std.ArrayListUnmanaged(u8).empty;
            defer buf.deinit(self.allocator);

            const count: u64 = self.recovery_points.items.len;
            try buf.appendSlice(self.allocator, &std.mem.toBytes(count));

            for (self.recovery_points.items) |rp| {
                try buf.appendSlice(self.allocator, &std.mem.toBytes(rp.sequence));
                try buf.appendSlice(self.allocator, &std.mem.toBytes(rp.timestamp));
                try buf.appendSlice(self.allocator, &std.mem.toBytes(rp.size_bytes));
                try buf.appendSlice(self.allocator, &std.mem.toBytes(rp.operation_count));
                try buf.appendSlice(self.allocator, &rp.checksum);
            }

            try atomicWriteFile(self.allocator, path, buf.items);
        }

        pub fn loadCheckpoints(self: *Manager, path: []const u8) !void {
            self.mutex.lock();
            defer self.mutex.unlock();

            const max_size: usize = 16 * 1024 * 1024;
            var io_backend = initIoBackend(self.allocator);
            defer io_backend.deinit();
            const io = io_backend.io();

            const data = try std.Io.Dir.cwd().readFileAlloc(io, path, self.allocator, .limited(max_size));
            defer self.allocator.free(data);

            self.recovery_points.clearRetainingCapacity();

            var pos: usize = 0;
            if (data.len < 8) return error.UnexpectedEndOfFile;
            const count = std.mem.readInt(u64, data[pos..][0..8], .little);
            pos += 8;

            const entry_size: usize = 8 + 8 + 8 + 8 + 32;
            var i: u64 = 0;
            while (i < count) : (i += 1) {
                if (pos + entry_size > data.len) return error.UnexpectedEndOfFile;

                const sequence = std.mem.readInt(u64, data[pos..][0..8], .little);
                pos += 8;
                const timestamp = std.mem.readInt(u64, data[pos..][0..8], .little);
                pos += 8;
                const size_bytes = std.mem.readInt(u64, data[pos..][0..8], .little);
                pos += 8;
                const op_count = std.mem.readInt(u64, data[pos..][0..8], .little);
                pos += 8;
                var checksum: [32]u8 = undefined;
                @memcpy(&checksum, data[pos..][0..32]);
                pos += 32;

                try self.recovery_points.append(self.allocator, .{
                    .sequence = sequence,
                    .timestamp = timestamp,
                    .size_bytes = size_bytes,
                    .operation_count = op_count,
                    .checksum = checksum,
                });
            }
        }

        pub fn loadOperationLog(self: *Manager, path: []const u8) !void {
            self.mutex.lock();
            defer self.mutex.unlock();

            var io_backend = initIoBackend(self.allocator);
            defer io_backend.deinit();
            const io = io_backend.io();

            const max_size = 256 * 1024 * 1024;
            const data = try std.Io.Dir.cwd().readFileAlloc(io, path, self.allocator, .limited(max_size));
            defer self.allocator.free(data);

            for (self.operation_log.items) |op| {
                self.allocator.free(op.key);
                if (op.value) |v| self.allocator.free(v);
                if (op.previous_value) |v| self.allocator.free(v);
            }
            self.operation_log.clearRetainingCapacity();

            var pos: usize = 0;

            if (data.len < 8) return error.UnexpectedEndOfFile;
            const count = std.mem.readInt(u64, data[pos..][0..8], .little);
            pos += 8;

            var max_seq: u64 = 0;
            var i: u64 = 0;
            while (i < count) : (i += 1) {
                if (pos + 17 > data.len) return error.UnexpectedEndOfFile;
                const timestamp = std.mem.readInt(i64, data[pos..][0..8], .little);
                pos += 8;
                const seq = std.mem.readInt(u64, data[pos..][0..8], .little);
                pos += 8;
                const op_type: OperationType = @enumFromInt(data[pos]);
                pos += 1;

                if (seq > max_seq) max_seq = seq;

                if (pos + 4 > data.len) return error.UnexpectedEndOfFile;
                const key_len = std.mem.readInt(u32, data[pos..][0..4], .little);
                pos += 4;
                if (pos + key_len > data.len) return error.UnexpectedEndOfFile;
                const key = try self.allocator.dupe(u8, data[pos .. pos + key_len]);
                errdefer self.allocator.free(key);
                pos += key_len;

                if (pos + 4 > data.len) return error.UnexpectedEndOfFile;
                const val_sentinel = std.mem.readInt(u32, data[pos..][0..4], .little);
                pos += 4;
                var val: ?[]const u8 = null;
                if (val_sentinel != std.math.maxInt(u32)) {
                    if (pos + val_sentinel > data.len) return error.UnexpectedEndOfFile;
                    val = try self.allocator.dupe(u8, data[pos .. pos + val_sentinel]);
                    pos += val_sentinel;
                }
                errdefer if (val) |v| self.allocator.free(v);

                if (pos + 4 > data.len) return error.UnexpectedEndOfFile;
                const prev_sentinel = std.mem.readInt(u32, data[pos..][0..4], .little);
                pos += 4;
                var prev: ?[]const u8 = null;
                if (prev_sentinel != std.math.maxInt(u32)) {
                    if (pos + prev_sentinel > data.len) return error.UnexpectedEndOfFile;
                    prev = try self.allocator.dupe(u8, data[pos .. pos + prev_sentinel]);
                    pos += prev_sentinel;
                }
                errdefer if (prev) |v| self.allocator.free(v);

                try self.operation_log.append(self.allocator, .{
                    .type = op_type,
                    .timestamp = timestamp,
                    .sequence_number = seq,
                    .key = key,
                    .value = val,
                    .previous_value = prev,
                });
            }

            if (max_seq >= self.next_op_sequence) {
                self.next_op_sequence = max_seq + 1;
            }
        }

        fn initIoBackend(allocator: std.mem.Allocator) std.Io.Threaded {
            return std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
        }

        fn atomicWriteFile(allocator: std.mem.Allocator, path: []const u8, data: []const u8) !void {
            const tmp_path = try std.fmt.allocPrint(allocator, "{s}.tmp", .{path});
            defer allocator.free(tmp_path);

            var io_backend = initIoBackend(allocator);
            defer io_backend.deinit();
            const io = io_backend.io();

            {
                var file = try std.Io.Dir.cwd().createFile(io, tmp_path, .{ .truncate = true });
                defer file.close(io);
                try file.writeStreamingAll(io, data);
                file.sync(io) catch {};
            }

            const cwd = std.Io.Dir.cwd();
            cwd.rename(tmp_path, cwd, path, io) catch {
                cwd.deleteFile(io, tmp_path) catch {};
                return error.PersistFailed;
            };
        }
    };
}
