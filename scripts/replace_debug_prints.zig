//! Script to replace std.debug.print with proper logging
//! This script helps automate the replacement of debug prints with the new I/O abstraction

const std = @import("std");
const io = @import("../lib/core/io.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <file_path>\n", .{args[0]});
        return;
    }

    const file_path = args[1];
    const file = std.fs.cwd().openFile(file_path, .{}) catch |err| {
        std.debug.print("Error opening file {s}: {}\n", .{ file_path, err });
        return;
    };
    defer file.close();

    const content = file.readToEndAlloc(allocator, 1024 * 1024) catch |err| {
        std.debug.print("Error reading file {s}: {}\n", .{ file_path, err });
        return;
    };
    defer allocator.free(content);

    // Replace std.debug.print with writer.print
    var output = std.ArrayList(u8).init(allocator);
    defer output.deinit();

    var lines = std.mem.split(u8, content, "\n");
    var line_number: u32 = 0;

    while (lines.next()) |line| {
        line_number += 1;

        if (std.mem.indexOf(u8, line, "std.debug.print") != null) {
            // Replace with writer.print
            const new_line = std.mem.replaceOwned(u8, allocator, line, "std.debug.print", "writer.print") catch |err| {
                std.debug.print("Error replacing in line {d}: {}\n", .{ line_number, err });
                continue;
            };
            defer allocator.free(new_line);

            try output.appendSlice(new_line);
            try output.append('\n');
        } else {
            try output.appendSlice(line);
            try output.append('\n');
        }
    }

    // Write back to file
    const output_file = std.fs.cwd().createFile(file_path, .{}) catch |err| {
        std.debug.print("Error creating file {s}: {}\n", .{ file_path, err });
        return;
    };
    defer output_file.close();

    try output_file.writeAll(output.items);
    std.debug.print("Updated {s}\n", .{file_path});
}
