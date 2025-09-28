const std = @import("std");
const wdbx = @import("wdbx");
const common = @import("common.zig");

pub const command = common.Command{
    .name = "db",
    .aliases = &.{"wdbx"},
    .summary = "Manage vector databases",
    .usage = "abi db <init|add|query|stats|optimize> [options]",
    .details = "  init      Create a new database\n" ++
        "  add       Insert a vector entry\n" ++
        "  query     Search for nearest vectors\n" ++
        "  stats     Display database statistics\n" ++
        "  optimize  Optimize the database store\n",
    .run = run,
};

pub fn run(ctx: *common.Context, args: [][:0]u8) !void {
    const allocator = ctx.allocator;
    if (args.len < 3 or common.isHelpToken(args[2])) {
        printDbHelp();
        return;
    }

    const sub = args[2];
    if (std.mem.eql(u8, sub, "init") or std.mem.eql(u8, sub, "create")) {
        try handleDbInit(args[3..]);
        return;
    }
    if (std.mem.eql(u8, sub, "add")) {
        try handleDbAdd(args[3..], allocator);
        return;
    }
    if (std.mem.eql(u8, sub, "query") or std.mem.eql(u8, sub, "search")) {
        try handleDbQuery(args[3..], allocator);
        return;
    }
    if (std.mem.eql(u8, sub, "stats") or std.mem.eql(u8, sub, "status")) {
        try handleDbStats(args[3..], allocator);
        return;
    }
    if (std.mem.eql(u8, sub, "optimize")) {
        try handleDbOptimize(args[3..], allocator);
        return;
    }

    std.debug.print("Unknown db subcommand: {s}\n", .{sub});
    printDbHelp();
}

fn handleDbInit(raw_args: [][:0]u8) !void {
    var db_path: ?[]const u8 = null;
    var dimension: ?usize = null;
    var force = false;

    var i: usize = 0;
    while (i < raw_args.len) : (i += 1) {
        const arg = raw_args[i];
        if (common.isHelpToken(arg)) {
            printDbHelp();
            return;
        } else if (std.mem.eql(u8, arg, "--db") and i + 1 < raw_args.len) {
            i += 1;
            db_path = raw_args[i];
        } else if ((std.mem.eql(u8, arg, "--dimension") or std.mem.eql(u8, arg, "--dim")) and i + 1 < raw_args.len) {
            i += 1;
            dimension = std.fmt.parseInt(usize, raw_args[i], 10) catch {
                std.debug.print("Invalid dimension value: {s}\n", .{raw_args[i]});
                return;
            };
        } else if (std.mem.eql(u8, arg, "--force")) {
            force = true;
        }
    }

    if (db_path == null or dimension == null) {
        std.debug.print("db init requires --db <path> and --dimension <N>\n", .{});
        printDbHelp();
        return;
    }

    const dim_value = dimension.?;
    if (dim_value == 0) {
        std.debug.print("Dimension must be greater than zero.\n", .{});
        return;
    }

    const dim_u16 = std.math.cast(u16, dim_value) orelse {
        std.debug.print("Dimension {d} exceeds supported range (u16).\n", .{dim_value});
        return;
    };

    if (!force) {
        if (std.fs.cwd().openFile(db_path.?, .{})) |file| {
            defer file.close();
            std.debug.print("Database '{s}' already exists. Use --force to overwrite.\n", .{db_path.?});
            return;
        } else |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        }
    }

    const db = try wdbx.Db.open(db_path.?, true);
    defer db.close();

    db.init(dim_u16) catch |err| {
        std.debug.print("Failed to initialise database: {any}\n", .{err});
        return err;
    };

    std.debug.print("Initialised database '{s}' with dimension {d}.\n", .{ db_path.?, dim_u16 });
}

fn handleDbAdd(raw_args: [][:0]u8, allocator: std.mem.Allocator) !void {
    var db_path: ?[]const u8 = null;
    var vector_literal: ?[]const u8 = null;
    var quiet = false;

    var i: usize = 0;
    while (i < raw_args.len) : (i += 1) {
        const arg = raw_args[i];
        if (common.isHelpToken(arg)) {
            printDbHelp();
            return;
        } else if (std.mem.eql(u8, arg, "--db") and i + 1 < raw_args.len) {
            i += 1;
            db_path = raw_args[i];
        } else if (std.mem.eql(u8, arg, "--vector") and i + 1 < raw_args.len) {
            i += 1;
            vector_literal = raw_args[i];
        } else if (std.mem.eql(u8, arg, "--quiet")) {
            quiet = true;
        }
    }

    if (db_path == null or vector_literal == null) {
        std.debug.print("db add requires --db <path> and --vector \"v1,v2,...\"\n", .{});
        printDbHelp();
        return;
    }

    const values = try common.parseCsvFloats(allocator, vector_literal.?);
    defer allocator.free(values);

    if (values.len == 0) {
        std.debug.print("Vector is empty; nothing to add.\n", .{});
        return;
    }

    const db = wdbx.Db.open(db_path.?, false) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("Database '{s}' not found.\n", .{db_path.?});
            return;
        },
        else => return err,
    };
    defer db.close();

    const dim = db.getDimension();
    if (dim != values.len) {
        std.debug.print("Dimension mismatch: database={d}, vector={d}.\n", .{ dim, values.len });
        return;
    }

    const id = db.add(values) catch |err| {
        std.debug.print("Failed to add vector: {any}\n", .{err});
        return err;
    };

    if (!quiet) {
        std.debug.print("Added vector with id {d} to '{s}'.\n", .{ id, db_path.? });
    }
}

fn handleDbQuery(raw_args: [][:0]u8, allocator: std.mem.Allocator) !void {
    var db_path: ?[]const u8 = null;
    var vector_literal: ?[]const u8 = null;
    var top_k: usize = 5;

    var i: usize = 0;
    while (i < raw_args.len) : (i += 1) {
        const arg = raw_args[i];
        if (common.isHelpToken(arg)) {
            printDbHelp();
            return;
        } else if (std.mem.eql(u8, arg, "--db") and i + 1 < raw_args.len) {
            i += 1;
            db_path = raw_args[i];
        } else if (std.mem.eql(u8, arg, "--vector") and i + 1 < raw_args.len) {
            i += 1;
            vector_literal = raw_args[i];
        } else if ((std.mem.eql(u8, arg, "--k") or std.mem.eql(u8, arg, "--top")) and i + 1 < raw_args.len) {
            i += 1;
            top_k = std.fmt.parseInt(usize, raw_args[i], 10) catch {
                std.debug.print("Invalid --k value: {s}\n", .{raw_args[i]});
                return;
            };
        }
    }

    if (db_path == null or vector_literal == null) {
        std.debug.print("db query requires --db <path> and --vector \"v1,v2,...\"\n", .{});
        printDbHelp();
        return;
    }

    if (top_k == 0) top_k = 1;

    const values = try common.parseCsvFloats(allocator, vector_literal.?);
    defer allocator.free(values);

    if (values.len == 0) {
        std.debug.print("Vector is empty; nothing to search.\n", .{});
        return;
    }

    const db = wdbx.Db.open(db_path.?, false) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("Database '{s}' not found.\n", .{db_path.?});
            return;
        },
        else => return err,
    };
    defer db.close();

    const dim = db.getDimension();
    if (dim != values.len) {
        std.debug.print("Dimension mismatch: database={d}, query={d}.\n", .{ dim, values.len });
        return;
    }

    const results = db.search(values, top_k, allocator) catch |err| {
        std.debug.print("Search failed: {any}\n", .{err});
        return err;
    };
    defer allocator.free(results);

    if (results.len == 0) {
        std.debug.print("No results found.\n", .{});
        return;
    }

    std.debug.print("Top {d} matches for dimension {d}:\n", .{ results.len, dim });
    for (results, 0..) |res, idx| {
        std.debug.print("  {d}: id={d} distance={d:.6}\n", .{ idx + 1, res.index, res.score });
    }
}

fn handleDbStats(raw_args: [][:0]u8, allocator: std.mem.Allocator) !void {
    _ = allocator;
    var db_path: ?[]const u8 = null;

    var i: usize = 0;
    while (i < raw_args.len) : (i += 1) {
        const arg = raw_args[i];
        if (common.isHelpToken(arg)) {
            printDbHelp();
            return;
        } else if (std.mem.eql(u8, arg, "--db") and i + 1 < raw_args.len) {
            i += 1;
            db_path = raw_args[i];
        }
    }

    if (db_path == null) {
        std.debug.print("db stats requires --db <path>\n", .{});
        printDbHelp();
        return;
    }

    const db = wdbx.Db.open(db_path.?, false) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("Database '{s}' not found.\n", .{db_path.?});
            return;
        },
        else => return err,
    };
    defer db.close();

    const stats = db.getStats();
    std.debug.print("Database: {s}\n", .{db_path.?});
    std.debug.print("  Dimension : {d}\n", .{db.getDimension()});
    std.debug.print("  Rows      : {d}\n", .{db.getRowCount()});
    std.debug.print("  Writes    : {d}\n", .{stats.write_count});
    std.debug.print("  Searches  : {d}\n", .{stats.search_count});
    std.debug.print("  Avg search: {d} us\n", .{stats.getAverageSearchTime()});
}

fn handleDbOptimize(raw_args: [][:0]u8, allocator: std.mem.Allocator) !void {
    _ = allocator;
    var db_path: ?[]const u8 = null;

    var i: usize = 0;
    while (i < raw_args.len) : (i += 1) {
        const arg = raw_args[i];
        if (common.isHelpToken(arg)) {
            printDbHelp();
            return;
        } else if (std.mem.eql(u8, arg, "--db") and i + 1 < raw_args.len) {
            i += 1;
            db_path = raw_args[i];
        }
    }

    if (db_path == null) {
        std.debug.print("db optimize requires --db <path>\n", .{});
        printDbHelp();
        return;
    }

    const db = wdbx.Db.open(db_path.?, false) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("Database '{s}' not found.\n", .{db_path.?});
            return;
        },
        else => return err,
    };
    defer db.close();

    std.debug.print("Database optimisation is not implemented yet.\n", .{});
}

fn printDbHelp() void {
    std.debug.print("\nDatabase commands:\n{s}\n", .{command.details orelse ""});
}
