const std = @import("std");
const unified = @import("unified.zig");
const db_helpers = @import("db_helpers.zig");
const http = @import("http.zig");

pub fn run(allocator: std.mem.Allocator, args: []const [:0]u8) !void {
    if (args.len == 0) {
        printHelp();
        return;
    }

    const command = std.mem.span(args[0]);
    if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help")) {
        printHelp();
        return;
    }

    if (std.mem.eql(u8, command, "add")) {
        try handleAdd(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "query")) {
        try handleQuery(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "stats")) {
        try handleStats(allocator);
        return;
    }

    if (std.mem.eql(u8, command, "serve")) {
        try handleServe(allocator, args[1..]);
        return;
    }

    std.debug.print("Unknown db command: {s}\n", .{command});
    printHelp();
}

fn printHelp() void {
    const text =
        "Usage: abi db <command> [options]\n\n" ++
        "Commands:\n" ++
        "  add --id <id> --vector <csv> [--meta <text>]\n" ++
        "  query --vector <csv> [--top-k <n>]\n" ++
        "  stats\n" ++
        "  serve [--addr <host:port>]\n";
    std.debug.print("{s}", .{text});
}

fn handleAdd(allocator: std.mem.Allocator, args: []const [:0]u8) !void {
    var id: ?u64 = null;
    var vector_text: ?[]const u8 = null;
    var meta: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) {
        const arg = std.mem.span(args[i]);
        i += 1;

        if (std.mem.eql(u8, arg, "--id")) {
            if (i < args.len) {
                id = std.fmt.parseInt(u64, std.mem.span(args[i]), 10) catch null;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--vector")) {
            if (i < args.len) {
                vector_text = std.mem.span(args[i]);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--meta")) {
            if (i < args.len) {
                meta = std.mem.span(args[i]);
                i += 1;
            }
            continue;
        }
    }

    const id_value = id orelse {
        std.debug.print("Missing --id\n", .{});
        return;
    };
    const vector_input = vector_text orelse {
        std.debug.print("Missing --vector\n", .{});
        return;
    };

    const vector = try db_helpers.parseVector(allocator, vector_input);
    defer allocator.free(vector);

    var handle = try unified.createDatabase(allocator, "cli");
    defer unified.closeDatabase(&handle);

    try unified.insertVector(&handle, id_value, vector, meta);
    std.debug.print("Inserted vector {d} (dim {d}).\n", .{ id_value, vector.len });
}

fn handleQuery(allocator: std.mem.Allocator, args: []const [:0]u8) !void {
    var vector_text: ?[]const u8 = null;
    var top_k: usize = 3;

    var i: usize = 0;
    while (i < args.len) {
        const arg = std.mem.span(args[i]);
        i += 1;

        if (std.mem.eql(u8, arg, "--vector")) {
            if (i < args.len) {
                vector_text = std.mem.span(args[i]);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--top-k")) {
            if (i < args.len) {
                top_k = std.fmt.parseInt(usize, std.mem.span(args[i]), 10) catch top_k;
                i += 1;
            }
            continue;
        }
    }

    const vector_input = vector_text orelse {
        std.debug.print("Missing --vector\n", .{});
        return;
    };

    const query = try db_helpers.parseVector(allocator, vector_input);
    defer allocator.free(query);

    var handle = try unified.createDatabase(allocator, "cli");
    defer unified.closeDatabase(&handle);
    try seedDatabase(&handle);

    const results = try unified.searchVectors(&handle, allocator, query, top_k);
    defer allocator.free(results);

    if (results.len == 0) {
        std.debug.print("No results.\n", .{});
        return;
    }

    std.debug.print("Results:\n", .{});
    for (results) |result| {
        std.debug.print("  id {d} score {d:.4}\n", .{ result.id, result.score });
    }
}

fn handleStats(allocator: std.mem.Allocator) !void {
    var handle = try unified.createDatabase(allocator, "cli");
    defer unified.closeDatabase(&handle);
    try seedDatabase(&handle);
    const stats = unified.getStats(&handle);
    std.debug.print("Database stats: {d} vectors, dimension {d}\n", .{ stats.count, stats.dimension });
}

fn handleServe(allocator: std.mem.Allocator, args: []const [:0]u8) !void {
    var address: []const u8 = "127.0.0.1:9191";
    var i: usize = 0;
    while (i < args.len) {
        const arg = std.mem.span(args[i]);
        i += 1;
        if (std.mem.eql(u8, arg, "--addr") and i < args.len) {
            address = std.mem.span(args[i]);
            i += 1;
        }
    }
    try http.serve(allocator, address);
}

fn seedDatabase(handle: *unified.DatabaseHandle) !void {
    const samples = [_][]const f32{
        &.{ 0.1, 0.2, 0.3 },
        &.{ 0.0, 0.4, 0.8 },
        &.{ 0.9, 0.1, 0.0 },
    };
    var id: u64 = 1;
    for (samples) |vector| {
        _ = unified.insertVector(handle, id, vector, null) catch {};
        id += 1;
    }
}
