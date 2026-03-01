const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();

    const args = try init.minimal.args.toSlice(arena);

    if (args.len < 2) {
        std.debug.print("Usage: json2zon <file.zon>\n", .{});
        return;
    }

    const input_path = args[1];

    var io_backend = std.Io.Threaded.init(arena, .{ .environ = init.minimal.environ });
    defer io_backend.deinit();
    const io = io_backend.io();

    const file = try io.openFile(input_path, .{});
    defer file.close();

    const json_content = try file.readToEndAlloc(arena, 10 * 1024 * 1024);

    var parsed = try std.zon.parseFromSlice(std.zon.Value, arena, json_content, .{});
    defer parsed.deinit();

    const stdout = io.getStdOut().writer();
    try writeValue(stdout, parsed.value, 0);
    try stdout.writeAll("\n");
}

fn writeValue(writer: anytype, value: std.zon.Value, indent: usize) !void {
    switch (value) {
        .null => try writer.writeAll("null"),
        .bool => |b| try writer.print("{}", .{b}),
        .integer => |i| try writer.print("{}", .{i}),
        .float => |f| try writer.print("{d}", .{f}),
        .string => |s| {
            try writer.writeAll("\"");
            try writer.writeAll(s);
            try writer.writeAll("\"");
        },
        .array => |arr| {
            try writer.writeAll(".{\n");
            for (arr.items) |v| {
                try writeIndent(writer, indent + 1);
                try writeValue(writer, v, indent + 1);
                try writer.writeAll(",\n");
            }
            try writeIndent(writer, indent);
            try writer.writeAll("}");
        },
        .object => |obj| {
            try writer.writeAll(".{\n");
            var it = obj.iterator();
            while (it.next()) |entry| {
                try writeIndent(writer, indent + 1);
                try writer.print(".@\"{s}\" = ", .{entry.key_ptr.*});
                try writeValue(writer, entry.value_ptr.*, indent + 1);
                try writer.writeAll(",\n");
            }
            try writeIndent(writer, indent);
            try writer.writeAll("}");
        },
        .number_string => |s| try writer.print("{s}", .{s}),
    }
}

fn writeIndent(writer: anytype, indent: usize) !void {
    try writer.writeByteNTimes(' ', indent * 2);
}
