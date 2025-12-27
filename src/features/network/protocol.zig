//! Serialization protocol for network task and result transmission.
//!
//! Defines the wire format for distributed compute tasks and results, including
//! magic bytes, versioning, and encoding/decoding functions.

const std = @import("std");

pub const ProtocolError = error{
    InvalidFormat,
    UnsupportedVersion,
    TruncatedData,
    PayloadTooLarge,
};

const task_magic = "ABIT";
const result_magic = "ABIR";
const protocol_version: u16 = 1;

pub const TaskEnvelope = struct {
    id: u64,
    kind: []const u8,
    payload: []const u8,

    pub fn deinit(self: *TaskEnvelope, allocator: std.mem.Allocator) void {
        allocator.free(self.kind);
        allocator.free(self.payload);
        self.* = undefined;
    }
};

pub const ResultStatus = enum(u8) {
    ok = 0,
    failed = 1,
};

pub const ResultEnvelope = struct {
    id: u64,
    status: ResultStatus,
    payload: []const u8,

    pub fn deinit(self: *ResultEnvelope, allocator: std.mem.Allocator) void {
        allocator.free(self.payload);
        self.* = undefined;
    }
};

pub fn encodeTask(allocator: std.mem.Allocator, task: TaskEnvelope) ![]u8 {
    if (task.kind.len > std.math.maxInt(u16)) return ProtocolError.PayloadTooLarge;
    if (task.payload.len > std.math.maxInt(u32)) return ProtocolError.PayloadTooLarge;

    var buffer = std.ArrayList(u8).empty;
    errdefer buffer.deinit(allocator);

    try buffer.appendSlice(allocator, task_magic);
    try appendInt(&buffer, allocator, u16, protocol_version);
    try appendInt(&buffer, allocator, u64, task.id);
    try appendInt(&buffer, allocator, u16, @intCast(task.kind.len));
    try appendInt(&buffer, allocator, u32, @intCast(task.payload.len));
    try buffer.appendSlice(allocator, task.kind);
    try buffer.appendSlice(allocator, task.payload);

    return buffer.toOwnedSlice(allocator);
}

pub fn decodeTask(allocator: std.mem.Allocator, data: []const u8) ProtocolError!TaskEnvelope {
    var cursor = Cursor{ .data = data };
    const magic = try cursor.readBytes(task_magic.len);
    if (!std.mem.eql(u8, magic, task_magic)) return ProtocolError.InvalidFormat;

    const version = try cursor.readInt(u16);
    if (version != protocol_version) return ProtocolError.UnsupportedVersion;

    const id = try cursor.readInt(u64);
    const kind_len: usize = @intCast(try cursor.readInt(u16));
    const payload_len: usize = @intCast(try cursor.readInt(u32));

    const kind_bytes = try cursor.readBytes(kind_len);
    const payload_bytes = try cursor.readBytes(payload_len);

    const kind = try allocator.dupe(u8, kind_bytes);
    errdefer allocator.free(kind);
    const payload = try allocator.dupe(u8, payload_bytes);

    return .{
        .id = id,
        .kind = kind,
        .payload = payload,
    };
}

pub fn encodeResult(allocator: std.mem.Allocator, result: ResultEnvelope) ![]u8 {
    if (result.payload.len > std.math.maxInt(u32)) return ProtocolError.PayloadTooLarge;

    var buffer = std.ArrayList(u8).empty;
    errdefer buffer.deinit(allocator);

    try buffer.appendSlice(allocator, result_magic);
    try appendInt(&buffer, allocator, u16, protocol_version);
    try appendInt(&buffer, allocator, u64, result.id);
    try buffer.append(allocator, @intFromEnum(result.status));
    try appendInt(&buffer, allocator, u32, @intCast(result.payload.len));
    try buffer.appendSlice(allocator, result.payload);

    return buffer.toOwnedSlice(allocator);
}

pub fn decodeResult(allocator: std.mem.Allocator, data: []const u8) ProtocolError!ResultEnvelope {
    var cursor = Cursor{ .data = data };
    const magic = try cursor.readBytes(result_magic.len);
    if (!std.mem.eql(u8, magic, result_magic)) return ProtocolError.InvalidFormat;

    const version = try cursor.readInt(u16);
    if (version != protocol_version) return ProtocolError.UnsupportedVersion;

    const id = try cursor.readInt(u64);
    const status_byte = try cursor.readInt(u8);
    const status = std.meta.intToEnum(ResultStatus, status_byte) catch
        return ProtocolError.InvalidFormat;
    const payload_len: usize = @intCast(try cursor.readInt(u32));
    const payload_bytes = try cursor.readBytes(payload_len);
    const payload = try allocator.dupe(u8, payload_bytes);

    return .{
        .id = id,
        .status = status,
        .payload = payload,
    };
}

const Cursor = struct {
    data: []const u8,
    index: usize = 0,

    fn readBytes(self: *Cursor, len: usize) ProtocolError![]const u8 {
        if (self.index + len > self.data.len) return ProtocolError.TruncatedData;
        const slice = self.data[self.index .. self.index + len];
        self.index += len;
        return slice;
    }

    fn readInt(self: *Cursor, comptime T: type) ProtocolError!T {
        const bytes = try self.readBytes(@sizeOf(T));
        return std.mem.readInt(
            T,
            @as(*const [@sizeOf(T)]u8, @ptrCast(bytes.ptr)),
            .little,
        );
    }
};

fn appendInt(
    buffer: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
    comptime T: type,
    value: T,
) !void {
    var bytes: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &bytes, value, .little);
    try buffer.appendSlice(allocator, &bytes);
}

test "encode/decode task envelope" {
    const allocator = std.testing.allocator;
    const task = TaskEnvelope{
        .id = 42,
        .kind = "compute.dot",
        .payload = "payload",
    };
    const encoded = try encodeTask(allocator, task);
    defer allocator.free(encoded);

    var decoded = try decodeTask(allocator, encoded);
    defer decoded.deinit(allocator);

    try std.testing.expectEqual(@as(u64, 42), decoded.id);
    try std.testing.expectEqualStrings("compute.dot", decoded.kind);
    try std.testing.expectEqualStrings("payload", decoded.payload);
}

test "encode/decode result envelope" {
    const allocator = std.testing.allocator;
    const result = ResultEnvelope{
        .id = 9,
        .status = .ok,
        .payload = "result",
    };
    const encoded = try encodeResult(allocator, result);
    defer allocator.free(encoded);

    var decoded = try decodeResult(allocator, encoded);
    defer decoded.deinit(allocator);

    try std.testing.expectEqual(@as(u64, 9), decoded.id);
    try std.testing.expectEqual(ResultStatus.ok, decoded.status);
    try std.testing.expectEqualStrings("result", decoded.payload);
}

test "decode rejects invalid magic" {
    const allocator = std.testing.allocator;
    const invalid_data = "INVALID" ++ [1]u8{0} ** 10;

    try std.testing.expectError(ProtocolError.InvalidFormat, decodeTask(allocator, invalid_data));
    try std.testing.expectError(ProtocolError.InvalidFormat, decodeResult(allocator, invalid_data));
}

test "decode rejects unsupported version" {
    const allocator = std.testing.allocator;
    const task = TaskEnvelope{
        .id = 42,
        .kind = "compute.dot",
        .payload = "payload",
    };
    const encoded = try encodeTask(allocator, task);
    defer allocator.free(encoded);

    var tampered = try allocator.dupe(u8, encoded);
    defer allocator.free(tampered);
    tampered[4] = 0xFF;

    try std.testing.expectError(ProtocolError.UnsupportedVersion, decodeTask(allocator, tampered));
}

test "decode rejects truncated data" {
    const allocator = std.testing.allocator;
    const task = TaskEnvelope{
        .id = 42,
        .kind = "compute.dot",
        .payload = "payload",
    };
    const encoded = try encodeTask(allocator, task);
    defer allocator.free(encoded);

    const truncated = encoded[0 .. encoded.len - 1];
    try std.testing.expectError(ProtocolError.TruncatedData, decodeTask(allocator, truncated));
}

test "encode rejects oversized payloads" {
    const allocator = std.testing.allocator;
    const oversized_kind: []u8 = &([1]u8{'x'} ** (std.math.maxInt(u16) + 1));
    const oversized_payload: []u8 = &([1]u8{'y'} ** (std.math.maxInt(u32) + 1));

    const task = TaskEnvelope{
        .id = 42,
        .kind = oversized_kind,
        .payload = "payload",
    };
    try std.testing.expectError(ProtocolError.PayloadTooLarge, encodeTask(allocator, task));

    const result = ResultEnvelope{
        .id = 42,
        .status = .ok,
        .payload = oversized_payload,
    };
    try std.testing.expectError(ProtocolError.PayloadTooLarge, encodeResult(allocator, result));
}
