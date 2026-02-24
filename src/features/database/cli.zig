const std = @import("std");
const build_options = @import("build_options");
const wdbx = @import("wdbx.zig");
const database = @import("database.zig");
const storage = @import("storage.zig");
const db_helpers = @import("db_helpers.zig");
const http = @import("http.zig");
const transformer = if (build_options.enable_ai) @import("../ai/transformer/mod.zig") else struct {
    pub const TransformerModel = struct {
        pub fn init(_: std.mem.Allocator, _: anytype) !TransformerModel {
            return error.AiDisabled;
        }
        pub fn deinit(_: *TransformerModel) void {}
        pub fn embed(_: *TransformerModel, _: std.mem.Allocator, _: []const u8) ![]f32 {
            return error.AiDisabled;
        }
    };
};

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
        "  add --id <id> (--vector <csv> | --embed <text>) [--meta <text>] [--db <path>]\n" ++
        "  query (--vector <csv> | --embed <text>) [--top-k <n>] [--db <path>]\n" ++
        "  stats [--db <path>]\n" ++
        "  optimize [--db <path>]\n" ++
        "  backup --db <path> --out <path> (legacy: --path <path>)\n" ++
        "  restore --db <path> --in <path> (legacy: --path <path>)\n" ++
        "  serve [--addr <host:port>]\n\n" ++
        "Options:\n" ++
        "  --path <path>   Legacy shorthand for both source and destination\n" ++
        "  --db <path>     Database file path to load/write\n" ++
        "  --out <path>    Backup output path\n" ++
        "  --in <path>     Restore input path\n" ++
        "  --top-k <n>     Query result count (default: 10)\n";
    std.debug.print("{s}", .{text});
}

fn wantsHelp(args: []const [:0]const u8) bool {
    for (args) |a| {
        const s = std.mem.sliceTo(a, 0);
        if (std.mem.eql(u8, s, "help") or std.mem.eql(u8, s, "--help") or std.mem.eql(u8, s, "-h"))
            return true;
    }
    return args.len == 0;
}

fn handleAdd(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (wantsHelp(args)) {
        printHelp();
        return;
    }
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

        if (std.mem.eql(u8, arg, "--db")) {
            if (i < args.len) {
                path = std.mem.sliceTo(args[i], 0);
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
    if (wantsHelp(args)) {
        printHelp();
        return;
    }
    var vector_text: ?[]const u8 = null;
    var embed_text: ?[]const u8 = null;
    var top_k: usize = 10;
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

        if (std.mem.eql(u8, arg, "--db")) {
            if (i < args.len) {
                path = std.mem.sliceTo(args[i], 0);
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
    if (wantsHelp(args)) {
        printHelp();
        return;
    }
    const path = parseDbPath(args);

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
    if (wantsHelp(args)) {
        printHelp();
        return;
    }
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
    if (wantsHelp(args)) {
        printHelp();
        return;
    }
    const path = parseDbPath(args);

    var ctx = try DbContext.init(allocator, path);
    defer ctx.deinit();
    try wdbx.optimize(&ctx.handle);
    try ctx.persist();
    std.debug.print("Database optimized.\n", .{});
}

const BackupRestoreArgs = struct {
    db_path: ?[]const u8 = null,
    out_path: ?[]const u8 = null,
    in_path: ?[]const u8 = null,
    legacy_path: ?[]const u8 = null,
};

const BackupPaths = struct {
    db_path: []const u8,
    out_path: []const u8,
};

const RestorePaths = struct {
    db_path: []const u8,
    in_path: []const u8,
};

fn parseBackupRestoreArgs(args: []const [:0]const u8) BackupRestoreArgs {
    var parsed = BackupRestoreArgs{};
    var i: usize = 0;
    while (i < args.len) {
        const arg = std.mem.sliceTo(args[i], 0);
        i += 1;

        if (std.mem.eql(u8, arg, "--db")) {
            if (i < args.len) {
                parsed.db_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--out")) {
            if (i < args.len) {
                parsed.out_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--in")) {
            if (i < args.len) {
                parsed.in_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--path")) {
            if (i < args.len) {
                parsed.legacy_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }
    }
    return parsed;
}

fn resolveBackupPaths(parsed: BackupRestoreArgs) ?BackupPaths {
    const db_path = parsed.db_path orelse parsed.legacy_path orelse return null;
    const out_path = parsed.out_path orelse parsed.legacy_path orelse return null;
    return .{
        .db_path = db_path,
        .out_path = out_path,
    };
}

fn resolveRestorePaths(parsed: BackupRestoreArgs) ?RestorePaths {
    const db_path = parsed.db_path orelse parsed.legacy_path orelse return null;
    const in_path = parsed.in_path orelse parsed.legacy_path orelse return null;
    return .{
        .db_path = db_path,
        .in_path = in_path,
    };
}

fn handleBackup(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (wantsHelp(args)) {
        printHelp();
        return;
    }
    const parsed = parseBackupRestoreArgs(args);
    const resolved = resolveBackupPaths(parsed) orelse {
        std.debug.print("Missing backup paths. Use --db <path> --out <path> (or legacy --path <path>).\n", .{});
        return;
    };

    var ctx = try DbContext.init(allocator, resolved.db_path);
    defer ctx.deinit();
    try wdbx.backupToPath(&ctx.handle, resolved.out_path);
    std.debug.print("Backup written to {s} (db: {s})\n", .{ resolved.out_path, resolved.db_path });
}

fn handleRestore(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (wantsHelp(args)) {
        printHelp();
        return;
    }
    const parsed = parseBackupRestoreArgs(args);
    const resolved = resolveRestorePaths(parsed) orelse {
        std.debug.print("Missing restore paths. Use --db <path> --in <path> (or legacy --path <path>).\n", .{});
        return;
    };

    var ctx = try DbContext.init(allocator, resolved.db_path);
    defer ctx.deinit();
    try restoreWithLegacyFallback(&ctx.handle, allocator, resolved.in_path);
    try ctx.persist();
    const stats = wdbx.getStats(&ctx.handle);
    std.debug.print("Restored database: {d} vectors (db: {s}).\n", .{ stats.count, resolved.db_path });
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

        if (std.mem.eql(u8, arg, "--db")) {
            if (i < args.len) {
                opts.path = std.mem.sliceTo(args[i], 0);
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

fn isBareLegacyFilename(path: []const u8) bool {
    if (path.len == 0) return false;
    if (std.fs.path.basename(path).len != path.len) return false;
    if (std.mem.indexOfScalar(u8, path, '/') != null) return false;
    if (std.mem.indexOfScalar(u8, path, '\\') != null) return false;
    if (std.mem.indexOfScalar(u8, path, ':') != null) return false;
    return true;
}

fn parseDbPath(args: []const [:0]const u8) ?[]const u8 {
    var path: ?[]const u8 = null;
    var i: usize = 0;
    while (i < args.len) {
        const arg = std.mem.sliceTo(args[i], 0);
        i += 1;

        if (std.mem.eql(u8, arg, "--db")) {
            if (i < args.len) {
                path = std.mem.sliceTo(args[i], 0);
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
    return path;
}

fn loadDatabaseWithLegacyFallback(allocator: std.mem.Allocator, file_path: []const u8) !database.Database {
    if (storage.loadDatabase(allocator, file_path)) |db| {
        return db;
    } else |err| switch (err) {
        error.FileNotFound => {
            if (!isBareLegacyFilename(file_path)) return err;
            const legacy_path = fs.normalizeBackupPath(allocator, file_path) catch return err;
            defer allocator.free(legacy_path);
            return storage.loadDatabase(allocator, legacy_path);
        },
        else => return err,
    }
}

fn restoreWithLegacyFallback(
    handle: *wdbx.DatabaseHandle,
    allocator: std.mem.Allocator,
    input_path: []const u8,
) !void {
    if (wdbx.restoreFromPath(handle, input_path)) {
        return;
    } else |err| switch (err) {
        error.FileNotFound => {
            if (!isBareLegacyFilename(input_path)) return err;
            const legacy_path = fs.normalizeBackupPath(allocator, input_path) catch return err;
            defer allocator.free(legacy_path);
            try wdbx.restoreFromPath(handle, legacy_path);
        },
        else => return err,
    }
}

const DbContext = struct {
    handle: wdbx.DatabaseHandle,
    path: ?[]const u8,

    fn init(allocator: std.mem.Allocator, path: ?[]const u8) !DbContext {
        if (path) |file_path| {
            const loaded = loadDatabaseWithLegacyFallback(allocator, file_path);
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
            try wdbx.backupToPath(&self.handle, file_path);
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

test "parseAddArgs accepts --db as path alias" {
    const args = [_][:0]const u8{
        "--id",
        "1",
        "--vector",
        "0.1,0.2",
        "--db",
        "vector.db",
    };
    const opts = parseAddArgs(&args);
    try std.testing.expectEqualStrings("vector.db", opts.path.?);
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

test "parseBackupRestoreArgs parses explicit and legacy flags" {
    const args = [_][:0]const u8{
        "--db",
        "state.db",
        "--out",
        "backup.db",
        "--in",
        "source.db",
        "--path",
        "legacy.db",
    };
    const parsed = parseBackupRestoreArgs(&args);

    try std.testing.expectEqualStrings("state.db", parsed.db_path.?);
    try std.testing.expectEqualStrings("backup.db", parsed.out_path.?);
    try std.testing.expectEqualStrings("source.db", parsed.in_path.?);
    try std.testing.expectEqualStrings("legacy.db", parsed.legacy_path.?);
}

test "resolveBackupPaths uses explicit then legacy precedence" {
    const explicit = resolveBackupPaths(.{
        .db_path = "db.explicit",
        .out_path = "out.explicit",
        .legacy_path = "legacy",
    }) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("db.explicit", explicit.db_path);
    try std.testing.expectEqualStrings("out.explicit", explicit.out_path);

    const legacy_only = resolveBackupPaths(.{
        .legacy_path = "legacy.db",
    }) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("legacy.db", legacy_only.db_path);
    try std.testing.expectEqualStrings("legacy.db", legacy_only.out_path);

    try std.testing.expect(resolveBackupPaths(.{}) == null);
}

test "resolveRestorePaths uses explicit then legacy precedence" {
    const explicit = resolveRestorePaths(.{
        .db_path = "db.explicit",
        .in_path = "in.explicit",
        .legacy_path = "legacy",
    }) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("db.explicit", explicit.db_path);
    try std.testing.expectEqualStrings("in.explicit", explicit.in_path);

    const legacy_only = resolveRestorePaths(.{
        .legacy_path = "legacy.db",
    }) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("legacy.db", legacy_only.db_path);
    try std.testing.expectEqualStrings("legacy.db", legacy_only.in_path);

    try std.testing.expect(resolveRestorePaths(.{}) == null);
}

test "isBareLegacyFilename identifies bare names only" {
    try std.testing.expect(isBareLegacyFilename("legacy.db"));
    try std.testing.expect(!isBareLegacyFilename("nested/legacy.db"));
    try std.testing.expect(!isBareLegacyFilename("../legacy.db"));
    try std.testing.expect(!isBareLegacyFilename(""));
}

test {
    std.testing.refAllDecls(@This());
}
