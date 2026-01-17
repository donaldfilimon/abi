//! API Documentation Generator
//!
//! Generates markdown documentation from Zig doc comments.

const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
    defer io_backend.deinit();
    const io = io_backend.io();
    const cwd = std.Io.Dir.cwd();

    const modules = [_]struct { path: []const u8, name: []const u8 }{
        .{ .path = "src/abi.zig", .name = "abi" },
        .{ .path = "src/compute/gpu/unified.zig", .name = "gpu" },
        .{ .path = "src/features/ai/mod.zig", .name = "ai" },
        .{ .path = "src/features/database/mod.zig", .name = "database" },
        .{ .path = "src/features/network/mod.zig", .name = "network" },
        .{ .path = "src/compute/runtime/mod.zig", .name = "compute" },
    };

    for (modules) |mod| {
        generateDoc(allocator, io, cwd, mod.path, mod.name) catch |err| {
             std.debug.print("Failed to generate docs for {s}: {}\n", .{mod.name, err});
        };
    }
}

fn generateDoc(allocator: std.mem.Allocator, io: std.Io, cwd: std.Io.Dir, path: []const u8, name: []const u8) !void {
    const source = cwd.readFileAlloc(io, path, allocator, .limited(1024 * 1024)) catch |err| {
        std.debug.print("Could not read {s}: {}\n", .{path, err});
        return err;
    };
    defer allocator.free(source);

    // Output to docs/api_<name>.md to avoid directory creation issues
    const out_name = try std.fmt.allocPrint(allocator, "docs/api_{s}.md", .{name});
    defer allocator.free(out_name);

    var file = cwd.createFile(io, out_name, .{}) catch |err| {
        std.debug.print("Could not create {s}: {}\n", .{out_name, err});
        return err;
    };
    defer file.close(io);
    var writer = file.writer(io);

    try writer.print("# {s} API Reference\n\n", .{name});
    try writer.print("**Source:** `{s}`\n\n", .{path});

    var lines = std.mem.splitScalar(u8, source, '\n');
    var doc_buffer = std.ArrayList(u8).init(allocator);
    defer doc_buffer.deinit();

    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \r");

        if (std.mem.startsWith(u8, trimmed, "//!")) {
            if (trimmed.len > 3) {
                try writer.print("{s}\n", .{trimmed[3..]});
            } else {
                try writer.writeAll("\n");
            }
            continue;
        }

        if (std.mem.startsWith(u8, trimmed, "///")) {
            if (trimmed.len > 3) {
                try doc_buf_append(&doc_buffer, trimmed[3..]);
            }
            try doc_buf_append(&doc_buffer, "\n");
            continue;
        }

        if (std.mem.startsWith(u8, trimmed, "pub ")) {
            // Found public declaration
            if (doc_buffer.items.len > 0) {
                try writer.print("### `{s}`\n\n", .{extractDeclSignature(trimmed)});
                try writer.print("{s}\n", .{doc_buffer.items});
                doc_buffer.clearRetainingCapacity();
            }
        } else if (trimmed.len > 0) {
            // Non-doc, non-pub line, clear buffer
            doc_buffer.clearRetainingCapacity();
        }
    }

    std.debug.print("Generated {s}\n", .{out_name});
}

fn doc_buf_append(buf: *std.ArrayList(u8), slice: []const u8) !void {
    try buf.appendSlice(slice);
}

fn extractDeclSignature(line: []const u8) []const u8 {
    // Basic extraction: take up to opening brace or semicolon
    var end = line.len;
    for (line, 0..) |c, i| {
        if (c == '{' or c == ';' or c == '=') {
            end = i;
            break;
        }
    }
    return std.mem.trimRight(u8, line[0..end], " ");
}
