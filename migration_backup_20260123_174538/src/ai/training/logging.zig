//! Training metrics logging (TensorBoard + W&B offline).
const std = @import("std");
const time = @import("../../shared/utils.zig");

pub const LogError =
    std.mem.Allocator.Error ||
    std.Io.Dir.CreateDirPathError ||
    std.Io.Dir.OpenError ||
    std.Io.File.OpenError ||
    std.Io.File.Writer.Error ||
    std.Io.File.WritePositionalError ||
    std.Io.File.StatError ||
    error{OutOfBounds};

pub const LoggerConfig = struct {
    log_dir: []const u8,
    enable_tensorboard: bool = false,
    enable_wandb: bool = false,
    wandb_project: ?[]const u8 = null,
    wandb_run_name: ?[]const u8 = null,
    wandb_entity: ?[]const u8 = null,
};

pub const Metric = struct {
    key: []const u8,
    value: f64,
};

pub const TrainingLogger = struct {
    allocator: std.mem.Allocator,
    tensorboard: ?TensorboardLogger,
    wandb: ?WandbLogger,

    pub fn init(allocator: std.mem.Allocator, config: LoggerConfig) LogError!TrainingLogger {
        var logger = TrainingLogger{
            .allocator = allocator,
            .tensorboard = null,
            .wandb = null,
        };

        if (config.enable_tensorboard) {
            logger.tensorboard = try TensorboardLogger.init(allocator, config.log_dir);
        }

        if (config.enable_wandb) {
            logger.wandb = try WandbLogger.init(allocator, config);
        }

        return logger;
    }

    pub fn deinit(self: *TrainingLogger) void {
        if (self.tensorboard) |*tb| tb.deinit();
        if (self.wandb) |*wb| wb.deinit();
        self.* = undefined;
    }

    pub fn logScalar(self: *TrainingLogger, tag: []const u8, value: f32, step: u64) LogError!void {
        if (self.tensorboard) |*tb| {
            try tb.logScalar(tag, value, step);
        }
        if (self.wandb) |*wb| {
            try wb.logScalar(tag, value, step);
        }
    }

    pub fn writeSummary(self: *TrainingLogger, metrics: []const Metric) LogError!void {
        if (self.wandb) |*wb| {
            try wb.writeSummary(metrics);
        }
    }
};

const TensorboardLogger = struct {
    allocator: std.mem.Allocator,
    io_backend: std.Io.Threaded,
    io: std.Io,
    file: std.Io.File,

    pub fn init(allocator: std.mem.Allocator, log_dir: []const u8) LogError!TensorboardLogger {
        var io_backend = std.Io.Threaded.init(allocator, .{ .environ = .empty });
        errdefer io_backend.deinit();
        const io = io_backend.io();

        try std.Io.Dir.cwd().createDirPath(io, log_dir);

        var run_id_allocated = true;
        const run_id = makeRunId(allocator) catch blk: {
            run_id_allocated = false;
            break :blk "run";
        };
        defer if (run_id_allocated) allocator.free(run_id);

        const filename = try std.fmt.allocPrint(allocator, "{s}/events.out.tfevents.{s}", .{ log_dir, run_id });
        defer allocator.free(filename);

        const file = try std.Io.Dir.cwd().createFile(io, filename, .{ .truncate = true });
        errdefer file.close(io);

        return .{
            .allocator = allocator,
            .io_backend = io_backend,
            .io = io,
            .file = file,
        };
    }

    pub fn deinit(self: *TensorboardLogger) void {
        self.file.close(self.io);
        self.io_backend.deinit();
        self.* = undefined;
    }

    pub fn logScalar(self: *TensorboardLogger, tag: []const u8, value: f32, step: u64) LogError!void {
        const wall_time = @as(f64, @floatFromInt(time.unixSeconds()));
        const payload = try encodeEvent(self.allocator, wall_time, step, tag, value);
        defer self.allocator.free(payload);
        try writeTfRecord(self.io, self.file, payload);
    }
};

const WandbLogger = struct {
    allocator: std.mem.Allocator,
    io_backend: std.Io.Threaded,
    io: std.Io,
    history_file: std.Io.File,
    history_offset: u64,
    summary_path: []const u8,
    run_dir: []const u8,
    project: []const u8,
    run_name: []const u8,
    entity: []const u8,

    pub fn init(allocator: std.mem.Allocator, config: LoggerConfig) LogError!WandbLogger {
        var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
        errdefer io_backend.deinit();
        const io = io_backend.io();

        const project = config.wandb_project orelse "abi";
        const run_name = config.wandb_run_name orelse "run";
        const entity = config.wandb_entity orelse "default";

        var run_id_allocated = true;
        const run_id = makeRunId(allocator) catch blk: {
            run_id_allocated = false;
            break :blk "run";
        };
        defer if (run_id_allocated) allocator.free(run_id);

        const wandb_dir = try std.fmt.allocPrint(allocator, "{s}/wandb", .{config.log_dir});
        errdefer allocator.free(wandb_dir);
        try std.Io.Dir.cwd().createDirPath(io, wandb_dir);

        const run_dir = try std.fmt.allocPrint(allocator, "{s}/run-{s}", .{ wandb_dir, run_id });
        errdefer allocator.free(run_dir);
        try std.Io.Dir.cwd().createDirPath(io, run_dir);

        const files_dir = try std.fmt.allocPrint(allocator, "{s}/files", .{run_dir});
        errdefer allocator.free(files_dir);
        try std.Io.Dir.cwd().createDirPath(io, files_dir);

        const history_path = try std.fmt.allocPrint(allocator, "{s}/wandb-history.jsonl", .{files_dir});
        defer allocator.free(history_path);
        const history_file = try std.Io.Dir.cwd().createFile(io, history_path, .{ .truncate = false });
        errdefer history_file.close(io);

        const history_stat = history_file.stat(io) catch |err| switch (err) {
            error.Streaming => std.Io.File.Stat{
                .inode = 0,
                .nlink = 0,
                .size = 0,
                .permissions = @enumFromInt(0),
                .kind = .file,
                .atime = null,
                .mtime = .{ .nanoseconds = 0 },
                .ctime = .{ .nanoseconds = 0 },
                .block_size = 0,
            },
            else => return err,
        };

        const summary_path = try std.fmt.allocPrint(allocator, "{s}/wandb-summary.json", .{files_dir});

        allocator.free(wandb_dir);
        allocator.free(files_dir);

        return .{
            .allocator = allocator,
            .io_backend = io_backend,
            .io = io,
            .history_file = history_file,
            .history_offset = history_stat.size,
            .summary_path = summary_path,
            .run_dir = run_dir,
            .project = project,
            .run_name = run_name,
            .entity = entity,
        };
    }

    pub fn deinit(self: *WandbLogger) void {
        self.history_file.close(self.io);
        self.io_backend.deinit();
        self.allocator.free(self.summary_path);
        self.allocator.free(self.run_dir);
        self.* = undefined;
    }

    pub fn logScalar(self: *WandbLogger, tag: []const u8, value: f32, step: u64) LogError!void {
        const timestamp = @as(u64, @intCast(time.unixSeconds()));
        const line = try std.fmt.allocPrint(
            self.allocator,
            "{{\"_step\":{d},\"_timestamp\":{d},\"{s}\":{d}}}\n",
            .{ step, timestamp, tag, value },
        );
        defer self.allocator.free(line);

        try self.history_file.writePositionalAll(self.io, line, self.history_offset);
        self.history_offset += line.len;
    }

    pub fn writeSummary(self: *WandbLogger, metrics: []const Metric) LogError!void {
        var buffer = std.ArrayListUnmanaged(u8).empty;
        defer buffer.deinit(self.allocator);

        try buffer.appendSlice(self.allocator, "{");
        var first = true;
        for (metrics) |metric| {
            if (!first) {
                try buffer.appendSlice(self.allocator, ",");
            }
            first = false;
            try buffer.appendSlice(self.allocator, "\"");
            try buffer.appendSlice(self.allocator, metric.key);
            try buffer.appendSlice(self.allocator, "\":");
            var temp: [64]u8 = undefined;
            const rendered = std.fmt.bufPrint(&temp, "{d}", .{metric.value}) catch "0";
            try buffer.appendSlice(self.allocator, rendered);
        }
        try buffer.appendSlice(self.allocator, "}\n");

        var file = try std.Io.Dir.cwd().createFile(self.io, self.summary_path, .{ .truncate = true });
        defer file.close(self.io);
        try file.writeStreamingAll(self.io, buffer.items);
    }
};

fn makeRunId(allocator: std.mem.Allocator) ![]const u8 {
    var rng = std.Random.DefaultPrng.init(@as(u64, @intCast(time.unixSeconds())));
    const rand = rng.random().int(u32);
    return std.fmt.allocPrint(allocator, "{d}-{x}", .{ time.unixSeconds(), rand });
}

const WireType = enum(u3) {
    varint = 0,
    fixed64 = 1,
    len_delim = 2,
    fixed32 = 5,
};

const ProtoWriter = struct {
    allocator: std.mem.Allocator,
    buffer: std.ArrayListUnmanaged(u8),

    fn init(allocator: std.mem.Allocator) ProtoWriter {
        return .{
            .allocator = allocator,
            .buffer = std.ArrayListUnmanaged(u8).empty,
        };
    }

    fn deinit(self: *ProtoWriter) void {
        self.buffer.deinit(self.allocator);
        self.* = undefined;
    }

    fn appendByte(self: *ProtoWriter, byte: u8) !void {
        try self.buffer.append(self.allocator, byte);
    }

    fn appendBytes(self: *ProtoWriter, bytes: []const u8) !void {
        try self.buffer.appendSlice(self.allocator, bytes);
    }

    fn writeKey(self: *ProtoWriter, field_number: u32, wire_type: WireType) !void {
        const key = (field_number << 3) | @intFromEnum(wire_type);
        try self.writeVarint(@as(u64, key));
    }

    fn writeVarint(self: *ProtoWriter, value: u64) !void {
        var v = value;
        while (true) {
            const byte: u8 = @intCast(v & 0x7f);
            v >>= 7;
            if (v == 0) {
                try self.appendByte(byte);
                break;
            } else {
                try self.appendByte(byte | 0x80);
            }
        }
    }

    fn writeFixed32(self: *ProtoWriter, value: u32) !void {
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &bytes, value, .little);
        try self.appendBytes(&bytes);
    }

    fn writeFixed64(self: *ProtoWriter, value: u64) !void {
        var bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &bytes, value, .little);
        try self.appendBytes(&bytes);
    }

    fn writeFloat(self: *ProtoWriter, value: f32) !void {
        try self.writeFixed32(@bitCast(value));
    }

    fn writeDouble(self: *ProtoWriter, value: f64) !void {
        try self.writeFixed64(@bitCast(value));
    }

    fn writeBytes(self: *ProtoWriter, bytes: []const u8) !void {
        try self.writeVarint(@intCast(bytes.len));
        try self.appendBytes(bytes);
    }

    fn toOwnedSlice(self: *ProtoWriter) ![]u8 {
        return self.buffer.toOwnedSlice(self.allocator);
    }
};

fn encodeEvent(
    allocator: std.mem.Allocator,
    wall_time: f64,
    step: u64,
    tag: []const u8,
    value: f32,
) ![]u8 {
    var value_writer = ProtoWriter.init(allocator);
    defer value_writer.deinit();
    try value_writer.writeKey(1, .len_delim);
    try value_writer.writeBytes(tag);
    try value_writer.writeKey(2, .fixed32);
    try value_writer.writeFloat(value);
    const value_payload = try value_writer.toOwnedSlice();
    defer allocator.free(value_payload);

    var summary_writer = ProtoWriter.init(allocator);
    defer summary_writer.deinit();
    try summary_writer.writeKey(1, .len_delim);
    try summary_writer.writeBytes(value_payload);
    const summary_payload = try summary_writer.toOwnedSlice();
    defer allocator.free(summary_payload);

    var event_writer = ProtoWriter.init(allocator);
    defer event_writer.deinit();
    try event_writer.writeKey(1, .fixed64);
    try event_writer.writeDouble(wall_time);
    try event_writer.writeKey(2, .varint);
    try event_writer.writeVarint(step);
    try event_writer.writeKey(3, .len_delim);
    try event_writer.writeBytes(summary_payload);

    return event_writer.toOwnedSlice();
}

fn writeTfRecord(io: std.Io, file: std.Io.File, payload: []const u8) !void {
    // Build the TFRecord in a buffer for Zig 0.16 compatibility
    var record_buf: std.ArrayListUnmanaged(u8) = .empty;
    defer record_buf.deinit(std.heap.page_allocator);

    // Length (8 bytes little-endian)
    var len_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_bytes, @intCast(payload.len), .little);
    const len_crc = maskedCrc32c(&len_bytes);
    var len_crc_bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &len_crc_bytes, len_crc, .little);

    const data_crc = maskedCrc32c(payload);
    var data_crc_bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &data_crc_bytes, data_crc, .little);

    try record_buf.appendSlice(std.heap.page_allocator, &len_bytes);
    try record_buf.appendSlice(std.heap.page_allocator, &len_crc_bytes);
    try record_buf.appendSlice(std.heap.page_allocator, payload);
    try record_buf.appendSlice(std.heap.page_allocator, &data_crc_bytes);

    // Write using writeStreamingAll for Zig 0.16 compatibility
    try file.writeStreamingAll(io, record_buf.items);
}

fn maskedCrc32c(data: []const u8) u32 {
    const crc = std.hash.crc.Crc32Iscsi.hash(data);
    const rotated = (crc >> 15) | (crc << 17);
    return rotated +% 0xa282ead8;
}
