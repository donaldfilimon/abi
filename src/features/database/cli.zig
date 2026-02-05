const std = @import("std");
const build_options = @import("build_options");
const wdbx = @import("wdbx.zig");
const database = @import("database.zig");
const storage = @import("storage.zig");
const db_helpers = @import("db_helpers.zig");
const http = @import("http.zig");
const transformer = @import("../ai/transformer/mod.zig");

pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        printHelp();
        return;
    }

    const command = std.mem.sliceTo(args[0], 0);
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
        try handleStats(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "optimize")) {
        try handleOptimize(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "backup")) {
        try handleBackup(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "restore")) {
        try handleRestore(allocator, args[1..]);
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
        "  add --id <id> (--vector <csv> | --embed <text>) [--meta <text>]\n" ++
        "  query (--vector <csv> | --embed <text>) [--top-k <n>]\n" ++
        "  stats\n" ++
        "  optimize\n" ++
        "  backup --path <file>\n" ++
        "  restore --path <file>\n" ++
        "  serve [--addr <host:port>]\n\n" ++
        "Options:\n" ++
        "  --path <file>   Load/save database state from file\n";
    std.debug.print("{s}", .{text});
}

fn handleAdd(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var id: ?u64 = null;
    var vector_text: ?[]const u8 = null;
    var meta: ?[]const u8 = null;
    var embed_text: ?[]const u8 = null;
    var path: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) {
        const arg = std.mem.sliceTo(args[i], 0);
        i += 1;

        if (std.mem.eql(u8, arg, "--id")) {
            if (i < args.len) {
                id = std.fmt.parseInt(u64, std.mem.sliceTo(args[i], 0), 10) catch null;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--vector")) {
            if (i < args.len) {
                vector_text = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--meta")) {
            if (i < args.len) {
                meta = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--embed")) {
            if (i < args.len) {
                embed_text = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--path")) {
            if (i < args.len) {
                path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }
    }

    const id_value = id orelse {
        std.debug.print("Missing --id\n", .{});
        return;
    };
    const has_vector = vector_text != null;
    const has_embed = embed_text != null;
    if (has_vector == has_embed) {
        std.debug.print("Specify exactly one of --vector or --embed\n", .{});
        return;
    }

    var vector: []f32 = &.{};
    defer if (vector.len > 0) allocator.free(vector);

    if (vector_text) |vector_input| {
        vector = try db_helpers.parseVector(allocator, vector_input);
    } else if (embed_text) |text| {
        if (!build_options.enable_ai) {
            std.debug.print("Embedding requires -Denable-ai=true\n", .{});
            return;
        }
        var model = try transformer.TransformerModel.init(allocator, .{});
        defer model.deinit();
        vector = try model.embed(allocator, text);
    }

    var ctx = try DbContext.init(allocator, path);
    defer ctx.deinit();

    try wdbx.insertVector(&ctx.handle, id_value, vector, meta);
    try ctx.persist();
    std.debug.print("Inserted vector {d} (dim {d}).\n", .{ id_value, vector.len });
}

fn handleQuery(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var vector_text: ?[]const u8 = null;
    var embed_text: ?[]const u8 = null;
    var top_k: usize = 3;
    var path: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) {
        const arg = std.mem.sliceTo(args[i], 0);
        i += 1;

        if (std.mem.eql(u8, arg, "--vector")) {
            if (i < args.len) {
                vector_text = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--top-k")) {
            if (i < args.len) {
                top_k = std.fmt.parseInt(usize, std.mem.sliceTo(args[i], 0), 10) catch top_k;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--embed")) {
            if (i < args.len) {
                embed_text = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--path")) {
            if (i < args.len) {
                path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }
    }

    const has_vector = vector_text != null;
    const has_embed = embed_text != null;
    if (has_vector == has_embed) {
        std.debug.print("Specify exactly one of --vector or --embed\n", .{});
        return;
    }

    var query: []f32 = &.{};
    defer if (query.len > 0) allocator.free(query);

    if (vector_text) |vector_input| {
        query = try db_helpers.parseVector(allocator, vector_input);
    } else if (embed_text) |text| {
        if (!build_options.enable_ai) {
            std.debug.print("Embedding requires -Denable-ai=true\n", .{});
            return;
        }
        var model = try transformer.TransformerModel.init(allocator, .{});
        defer model.deinit();
        query = try model.embed(allocator, text);
    }

    var ctx = try DbContext.init(allocator, path);
    defer ctx.deinit();
    if (ctx.path == null) {
        try seedDatabase(&ctx.handle);
    }

    const results = try wdbx.searchVectors(&ctx.handle, allocator, query, top_k);
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

fn handleStats(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var path: ?[]const u8 = null;
    if (args.len >= 2 and std.mem.eql(u8, std.mem.sliceTo(args[0], 0), "--path")) {
        path = std.mem.sliceTo(args[1], 0);
    }

    var ctx = try DbContext.init(allocator, path);
    defer ctx.deinit();
    if (ctx.path == null) {
        try seedDatabase(&ctx.handle);
    }
    const stats = wdbx.getStats(&ctx.handle);
    std.debug.print(
        "Database stats: {d} vectors, dimension {d}\n",
        .{ stats.count, stats.dimension },
    );
}

fn handleServe(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var address: []const u8 = "127.0.0.1:9191";
    var i: usize = 0;
    while (i < args.len) {
        const arg = std.mem.sliceTo(args[i], 0);
        i += 1;
        if (std.mem.eql(u8, arg, "--addr") and i < args.len) {
            address = std.mem.sliceTo(args[i], 0);
            i += 1;
        }
    }
    try http.serve(allocator, address);
}

fn handleOptimize(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var path: ?[]const u8 = null;
    if (args.len >= 2 and std.mem.eql(u8, std.mem.sliceTo(args[0], 0), "--path")) {
        path = std.mem.sliceTo(args[1], 0);
    }

    var ctx = try DbContext.init(allocator, path);
    defer ctx.deinit();
    try wdbx.optimize(&ctx.handle);
    try ctx.persist();
    std.debug.print("Database optimized.\n", .{});
}

fn handleBackup(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var path: ?[]const u8 = null;
    if (args.len >= 2 and std.mem.eql(u8, std.mem.sliceTo(args[0], 0), "--path")) {
        path = std.mem.sliceTo(args[1], 0);
    }
    const output = path orelse {
        std.debug.print("Missing --path\n", .{});
        return;
    };
    var ctx = try DbContext.init(allocator, path);
    defer ctx.deinit();
    try wdbx.backup(&ctx.handle, output);
    std.debug.print("Backup written to {s}\n", .{output});
}

fn handleRestore(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var path: ?[]const u8 = null;
    if (args.len >= 2 and std.mem.eql(u8, std.mem.sliceTo(args[0], 0), "--path")) {
        path = std.mem.sliceTo(args[1], 0);
    }
    const input = path orelse {
        std.debug.print("Missing --path\n", .{});
        return;
    };
    var ctx = try DbContext.init(allocator, path);
    defer ctx.deinit();
    try wdbx.restore(&ctx.handle, input);
    const stats = wdbx.getStats(&ctx.handle);
    std.debug.print("Restored database: {d} vectors.\n", .{stats.count});
}

fn seedDatabase(handle: *wdbx.DatabaseHandle) !void {
    const samples = [_][]const f32{
        &.{ 0.1, 0.2, 0.3 },
        &.{ 0.0, 0.4, 0.8 },
        &.{ 0.9, 0.1, 0.0 },
    };
    var id: u64 = 1;
    for (samples) |vector| {
        try wdbx.insertVector(handle, id, vector, null);
        id += 1;
    }
}

const fs = @import("../../services/shared/utils.zig").fs;

/// Parse CLI arguments for the add command. Returns parsed options or null on validation error.
/// Exposed for testing purposes.
pub const AddOptions = struct {
    id: ?u64 = null,
    vector_text: ?[]const u8 = null,
    meta: ?[]const u8 = null,
    embed_text: ?[]const u8 = null,
    path: ?[]const u8 = null,
};

pub fn parseAddArgs(args: []const [:0]const u8) AddOptions {
    var opts = AddOptions{};
    var i: usize = 0;
    while (i < args.len) {
        const arg = std.mem.sliceTo(args[i], 0);
        i += 1;

        if (std.mem.eql(u8, arg, "--id")) {
            if (i < args.len) {
                opts.id = std.fmt.parseInt(u64, std.mem.sliceTo(args[i], 0), 10) catch null;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--vector")) {
            if (i < args.len) {
                opts.vector_text = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--meta")) {
            if (i < args.len) {
                opts.meta = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--embed")) {
            if (i < args.len) {
                opts.embed_text = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--path")) {
            if (i < args.len) {
                opts.path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }
    }
    return opts;
}

/// Validate that exactly one of vector or embed is specified.
pub fn validateVectorOrEmbed(has_vector: bool, has_embed: bool) bool {
    return has_vector != has_embed;
}

/// Normalize a database path for consistent handling.
/// Returns the path unchanged if valid, or null if empty.
pub fn normalizePath(path: ?[]const u8) ?[]const u8 {
    const p = path orelse return null;
    if (p.len == 0) return null;
    return p;
}

const DbContext = struct {
    handle: wdbx.DatabaseHandle,
    path: ?[]const u8,

    fn init(allocator: std.mem.Allocator, path: ?[]const u8) !DbContext {
        if (path) |file_path| {
            // Normalize path to backups/ directory (same as backup/restore)
            const safe_path = fs.normalizeBackupPath(allocator, file_path) catch |norm_err| {
                std.log.debug("Path normalization failed for '{s}': {t}, trying direct load", .{ file_path, norm_err });
                // If path validation fails, try loading directly (for legacy compatibility)
                const loaded = storage.loadDatabase(allocator, file_path);
                if (loaded) |db| {
                    return .{ .handle = .{ .db = db }, .path = file_path };
                } else |load_err| switch (load_err) {
                    std.Io.Dir.ReadFileAllocError.FileNotFound => {
                        const handle = try wdbx.createDatabase(allocator, file_path);
                        return .{ .handle = handle, .path = file_path };
                    },
                    else => return load_err,
                }
            };
            defer allocator.free(safe_path);

            const loaded = storage.loadDatabase(allocator, safe_path);
            if (loaded) |db| {
                return .{ .handle = .{ .db = db }, .path = file_path };
            } else |err| switch (err) {
                std.Io.Dir.ReadFileAllocError.FileNotFound => {
                    const handle = try wdbx.createDatabase(allocator, file_path);
                    return .{ .handle = handle, .path = file_path };
                },
                else => return err,
            }
        }
        const handle = try wdbx.createDatabase(allocator, "cli");
        return .{ .handle = handle, .path = null };
    }

    fn deinit(self: *DbContext) void {
        wdbx.closeDatabase(&self.handle);
    }

    fn persist(self: *DbContext) !void {
        if (self.path) |file_path| {
            try wdbx.backup(&self.handle, file_path);
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "parseAddArgs parses all options" {
    const args = [_][:0]const u8{
        "--id",
        "42",
        "--vector",
        "1.0,2.0,3.0",
        "--meta",
        "test metadata",
        "--path",
        "/tmp/test.db",
    };
    const opts = parseAddArgs(&args);

    try std.testing.expectEqual(@as(?u64, 42), opts.id);
    try std.testing.expectEqualStrings("1.0,2.0,3.0", opts.vector_text.?);
    try std.testing.expectEqualStrings("test metadata", opts.meta.?);
    try std.testing.expectEqualStrings("/tmp/test.db", opts.path.?);
    try std.testing.expect(opts.embed_text == null);
}

test "parseAddArgs handles missing values gracefully" {
    // Missing value after --id flag
    const args_missing_value = [_][:0]const u8{"--id"};
    const opts = parseAddArgs(&args_missing_value);

    try std.testing.expect(opts.id == null);

    // Empty args
    const empty_args: []const [:0]const u8 = &.{};
    const empty_opts = parseAddArgs(empty_args);
    try std.testing.expect(empty_opts.id == null);
    try std.testing.expect(empty_opts.vector_text == null);
}

test "parseAddArgs handles invalid id" {
    const args = [_][:0]const u8{ "--id", "not_a_number" };
    const opts = parseAddArgs(&args);

    // Invalid number should result in null
    try std.testing.expect(opts.id == null);
}

test "validateVectorOrEmbed requires exactly one" {
    // Both false - invalid
    try std.testing.expect(!validateVectorOrEmbed(false, false));

    // Both true - invalid
    try std.testing.expect(!validateVectorOrEmbed(true, true));

    // Vector only - valid
    try std.testing.expect(validateVectorOrEmbed(true, false));

    // Embed only - valid
    try std.testing.expect(validateVectorOrEmbed(false, true));
}

test "normalizePath handles empty and null" {
    // Null input
    try std.testing.expect(normalizePath(null) == null);

    // Empty string
    try std.testing.expect(normalizePath("") == null);

    // Valid path
    const valid_path = normalizePath("/tmp/test.db");
    try std.testing.expect(valid_path != null);
    try std.testing.expectEqualStrings("/tmp/test.db", valid_path.?);
}
