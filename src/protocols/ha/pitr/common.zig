const std = @import("std");

pub fn Helpers(comptime Operation: type) type {
    return struct {
        pub fn freeOperation(allocator: std.mem.Allocator, op: Operation) void {
            allocator.free(op.key);
            if (op.value) |v| allocator.free(v);
            if (op.previous_value) |v| allocator.free(v);
        }

        pub fn cloneOperation(allocator: std.mem.Allocator, op: Operation) !Operation {
            const key_copy = try allocator.dupe(u8, op.key);
            errdefer allocator.free(key_copy);

            var value_copy: ?[]const u8 = null;
            if (op.value) |v| {
                value_copy = try allocator.dupe(u8, v);
            }
            errdefer if (value_copy) |v| allocator.free(v);

            var prev_copy: ?[]const u8 = null;
            if (op.previous_value) |v| {
                prev_copy = try allocator.dupe(u8, v);
            }
            errdefer if (prev_copy) |v| allocator.free(v);

            return .{
                .type = op.type,
                .timestamp = op.timestamp,
                .sequence_number = op.sequence_number,
                .key = key_copy,
                .value = value_copy,
                .previous_value = prev_copy,
            };
        }
    };
}

test {
    std.testing.refAllDecls(@This());
}
